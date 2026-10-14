# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Study bookkeeping shared by the tuning phases.

A trial moves through phases recorded in its user attributes. With a bench in
the loop the order is ``asked`` (sampled), ``screened`` (its deployed latency
is known and within the bound, so it may train), ``trained`` (a checkpoint
exists) and ``measured`` (its accuracy was reported to the sampler). A trial
whose latency exceeds the bound is ``rejected`` before any training and told
to the sampler as an infeasible pruned trial, so the sampler learns where the
bound runs without a training slot being spent. Trials carried over from an
earlier study start as ``imported``: their accuracy is known, only their
bench latency is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.trial import FrozenTrial, TrialState

from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.utils.mlflow_helpers import resolve_tracking_uri

ATTR_PHASE = "phase"
PHASE_ASKED = "asked"
PHASE_IMPORTED = "imported"
PHASE_SCREENED = "screened"
PHASE_TRAINED = "trained"
PHASE_MEASURED = "measured"
PHASE_REJECTED = "rejected"
# A proposal with the parameters of an earlier trial, failed without spending anything.
PHASE_DUPLICATE = "duplicate"

ATTR_OVERRIDES = "overrides"
ATTR_STATIC = "static"
ATTR_REFERENCE = "reference"
ATTR_RUN_ID = "mlflow_run_id"
ATTR_HYDRA_DIR = "hydra_dir"
ATTR_CHECKPOINT = "checkpoint"
ATTR_TRAIN_LOG = "train_log"
ATTR_METRICS = "metrics"
# The eager PyTorch latency proxy measured on the training GPU.
ATTR_LATENCY = "latency"
# The deployed TensorRT latency measured on the bench, with random weights.
ATTR_BENCH = "bench"
ATTR_BENCH_ATTEMPTS = "bench_attempts"
ATTR_LATENCY_LIMIT = "latency_limit_ms"
ATTR_PACKAGE = "package_dir"
ATTR_CONSTRAINTS = "constraints"
ATTR_ERROR = "error"
ATTR_MEASURE_ATTEMPTS = "measure_attempts"
ATTR_IMPORT = "imported_from"
ATTR_TRAINED_HERE = "trained_here"
ATTR_DUPLICATE_OF = "duplicate_of"

# Study-level attributes written by the termination check.
STUDY_ATTR_REFERENCE_LATENCY = "reference_latency"
STUDY_ATTR_TERMINATION = "termination"
STUDY_ATTR_TERMINATED = "terminated"

LATENCY_SOURCE = "latency"
METRIC_SOURCE = "metric"
SQLITE_PREFIX = "sqlite:///"


@dataclass(frozen=True)
class Objective:
    """One optimization objective.

    Attributes:
        name: Metric key for ``metric`` objectives, or a label for latency.
        direction: ``minimize`` or ``maximize``.
        source: ``metric`` reads the validation metrics, ``latency`` reads
            the trial's latency in milliseconds.
    """

    name: str
    direction: str
    source: str

    def __post_init__(self) -> None:
        """Validate the direction and the source."""
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(f"Objective {self.name!r}: direction must be minimize or maximize.")
        if self.source not in (METRIC_SOURCE, LATENCY_SOURCE):
            raise ValueError(f"Objective {self.name!r}: source must be metric or latency.")


@dataclass(frozen=True)
class Constraint:
    """A bound the sampler is told about.

    A ``metric`` constraint bounds a validation metric. A ``latency``
    constraint bounds the trial's latency in milliseconds, either by a fixed
    ``max`` or relative to a reference trial: the bound is that trial's
    measured latency plus ``margin``, so the study asks for models no slower
    than a known one instead of guessing an absolute number.

    Attributes:
        metric: Metric key, or a label for latency constraints.
        min: Lower bound, if any.
        max: Upper bound, if any.
        source: ``metric`` or ``latency``.
        reference: Name of the reference trial anchoring a relative bound.
        margin: Milliseconds added to the reference latency.
    """

    metric: str
    min: float | None = None
    max: float | None = None
    source: str = METRIC_SOURCE
    reference: str | None = None
    margin: float = 0.0

    def __post_init__(self) -> None:
        """Require exactly one way of bounding the value."""
        if self.source not in (METRIC_SOURCE, LATENCY_SOURCE):
            raise ValueError(f"Constraint on {self.metric!r}: source must be metric or latency.")
        if self.reference is not None:
            if self.source != LATENCY_SOURCE:
                raise ValueError(f"Constraint on {self.metric!r}: only latency can be relative.")
            if self.min is not None or self.max is not None:
                raise ValueError(
                    f"Constraint on {self.metric!r}: a reference bound excludes min and max."
                )
        elif self.min is None and self.max is None:
            raise ValueError(f"Constraint on {self.metric!r} needs a min, a max or a reference.")

    @property
    def relative(self) -> bool:
        """Whether the bound is derived from a reference trial's latency."""
        return self.reference is not None

    def upper_bound(self, reference_value: float | None = None) -> float | None:
        """Return the effective upper bound, resolving a relative one.

        Args:
            reference_value: The reference trial's latency for relative bounds.

        Returns:
            The upper bound, or ``None`` when only a lower bound exists.

        Raises:
            ValueError: If the bound is relative and no reference value is given.
        """
        if not self.relative:
            return self.max
        if reference_value is None:
            raise ValueError(
                f"Constraint on {self.metric!r} is relative to {self.reference!r}, whose "
                "latency has not been measured yet."
            )
        return float(reference_value) + self.margin

    def violation(self, value: float, reference_value: float | None = None) -> float:
        """Return how far ``value`` lies outside the bounds.

        Positive means infeasible by that much; zero or negative means
        feasible with that much slack, which is what the sampler expects
        and what the status output shows.
        """
        candidates = []
        if self.min is not None:
            candidates.append(self.min - value)
        upper = self.upper_bound(reference_value)
        if upper is not None:
            candidates.append(value - upper)
        return max(candidates)


def load_objectives(entries: Sequence[Mapping[str, Any]]) -> list[Objective]:
    """Parse the ``tuning.objectives`` config list."""
    objectives = [
        Objective(str(entry["name"]), str(entry["direction"]), str(entry.get("source", "metric")))
        for entry in entries
    ]
    if not objectives:
        raise ValueError("At least one objective is required.")
    names = [objective.name for objective in objectives]
    if len(set(names)) != len(names):
        raise ValueError(f"Objective names must be unique, got {names}.")
    return objectives


def load_constraints(entries: Sequence[Mapping[str, Any]] | None) -> list[Constraint]:
    """Parse the ``tuning.constraints`` config list.

    A metric entry reads ``{metric, min, max}``; a latency entry reads
    ``{source: latency, name, max}`` or ``{source: latency, name, reference,
    margin_ms}``.
    """
    constraints = []
    for entry in entries or ():
        source = str(entry.get("source", METRIC_SOURCE))
        name = entry.get("metric") if source == METRIC_SOURCE else entry.get("name", "latency_ms")
        if name is None:
            raise ValueError(f"Constraint entry {dict(entry)} names no metric.")
        constraints.append(
            Constraint(
                str(name),
                None if entry.get("min") is None else float(entry["min"]),
                None if entry.get("max") is None else float(entry["max"]),
                source=source,
                reference=None if entry.get("reference") is None else str(entry["reference"]),
                margin=float(entry.get("margin_ms", 0.0)),
            )
        )
    return constraints


def latency_constraints(constraints: Sequence[Constraint]) -> list[Constraint]:
    """Return the latency constraints of a study."""
    return [constraint for constraint in constraints if constraint.source == LATENCY_SOURCE]


def objective_values(
    objectives: Sequence[Objective], metrics: Mapping[str, float], latency_ms: float | None
) -> list[float]:
    """Assemble the objective vector in study order.

    Args:
        objectives: Study objectives.
        metrics: Validation metrics keyed by their logged name.
        latency_ms: The trial's latency, if any objective reads it.

    Returns:
        One float per objective.

    Raises:
        KeyError: If a metric objective was not logged.
        ValueError: If a latency objective exists but no latency was measured.
    """
    values = []
    for objective in objectives:
        if objective.source == LATENCY_SOURCE:
            if latency_ms is None:
                raise ValueError(f"Objective {objective.name!r} needs a measured latency.")
            values.append(float(latency_ms))
        elif objective.name not in metrics:
            raise KeyError(
                f"Objective metric {objective.name!r} was not logged. Available: {sorted(metrics)}"
            )
        else:
            values.append(float(metrics[objective.name]))
    return values


def constraint_values(
    constraints: Sequence[Constraint],
    metrics: Mapping[str, float],
    latency_ms: float | None = None,
    reference_latency_ms: float | None = None,
    unknown_metric_violation: float | None = None,
) -> list[float]:
    """Evaluate every constraint; values at most zero are feasible.

    Args:
        constraints: Study constraints in order.
        metrics: Validation metrics, possibly empty for an untrained trial.
        latency_ms: The trial's latency for latency constraints.
        reference_latency_ms: The reference trial's latency for relative bounds.
        unknown_metric_violation: Value recorded for a metric constraint whose
            metric is missing, such as the debris recall of a trial rejected
            on latency before it trained. ``None`` raises instead.

    Returns:
        One violation per constraint.
    """
    values = []
    for constraint in constraints:
        if constraint.source == LATENCY_SOURCE:
            if latency_ms is None:
                raise ValueError(f"Constraint {constraint.metric!r} needs a measured latency.")
            values.append(constraint.violation(float(latency_ms), reference_latency_ms))
        elif constraint.metric in metrics:
            values.append(constraint.violation(float(metrics[constraint.metric])))
        elif unknown_metric_violation is not None:
            values.append(float(unknown_metric_violation))
        else:
            raise KeyError(
                f"Constraint metric {constraint.metric!r} was not logged. "
                f"Available: {sorted(metrics)}"
            )
    return values


def constraints_from_user_attrs(
    constraint_count: int,
) -> Callable[[FrozenTrial], Sequence[float]]:
    """Build the sampler's ``constraints_func`` reading the stored violations.

    A trial that never reached measurement carries no violations; it is
    reported feasible so the sampler treats it by its state alone.
    """

    def read(trial: FrozenTrial) -> Sequence[float]:
        stored = trial.user_attrs.get(ATTR_CONSTRAINTS)
        if stored is None:
            return [0.0] * constraint_count
        return [float(value) for value in stored]

    return read


def trial_is_feasible(trial: FrozenTrial) -> bool:
    """Whether every stored constraint violation of ``trial`` is at most zero."""
    stored = trial.user_attrs.get(ATTR_CONSTRAINTS) or []
    return all(float(value) <= 0.0 for value in stored)


def feasible_complete_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Return the completed trials that satisfy every constraint, by number."""
    return sorted(
        (
            trial
            for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if trial_is_feasible(trial)
        ),
        key=lambda trial: trial.number,
    )


def best_feasible_trial(study: optuna.Study, objective: Objective) -> FrozenTrial | None:
    """Return the feasible completed trial with the best first objective value."""
    trials = [trial for trial in feasible_complete_trials(study) if trial.values]
    if not trials:
        return None

    def first_value(trial: FrozenTrial) -> float:
        return float(trial.values[0])

    if objective.direction == "maximize":
        return max(trials, key=first_value)
    return min(trials, key=first_value)


def trial_bench_latency_ms(trial: FrozenTrial) -> float | None:
    """Return the trial's bench latency in milliseconds, if measured."""
    bench = trial.user_attrs.get(ATTR_BENCH)
    if not bench:
        return None
    return float(bench["latency_ms"])


def trial_proxy_latency_ms(trial: FrozenTrial) -> float | None:
    """Return the trial's eager proxy latency in milliseconds, if measured."""
    proxy = trial.user_attrs.get(ATTR_LATENCY)
    if not proxy:
        return None
    return float(proxy["mean_ms"])


def find_reference_trial(study: optuna.Study, name: str) -> FrozenTrial | None:
    """Return the trial queued or imported under reference name ``name``."""
    for trial in sorted(study.get_trials(deepcopy=False), key=lambda item: item.number):
        if trial.user_attrs.get(ATTR_REFERENCE) == name:
            return trial
    return None


def reference_model_latencies(study: optuna.Study) -> dict[str, dict[str, Any]]:
    """Return the bench measurements of the reference models, by name.

    A reference model is a configuration outside the search space (another
    architecture, such as the deployed PTv3) that anchors the latency bound;
    its measurement lives on the study rather than on a trial.
    """
    stored = study.user_attrs.get(STUDY_ATTR_REFERENCE_LATENCY) or {}
    return {str(name): dict(value) for name, value in stored.items()}


def set_reference_model_latency(
    study: optuna.Study, name: str, measurement: Mapping[str, Any]
) -> None:
    """Record a reference model's bench measurement on the study."""
    stored = reference_model_latencies(study)
    stored[name] = dict(measurement)
    study.set_user_attr(STUDY_ATTR_REFERENCE_LATENCY, stored)


def reference_latency_ms(study: optuna.Study, constraints: Sequence[Constraint]) -> float | None:
    """Return the bench latency of the reference the latency bound needs.

    The reference is a trial of the study when a trial carries its name,
    and a reference model recorded on the study otherwise.

    Returns:
        The latency, or ``None`` when no constraint is relative or the
        reference has not been measured yet.
    """
    for constraint in latency_constraints(constraints):
        if constraint.reference is None:
            continue
        trial = find_reference_trial(study, constraint.reference)
        if trial is not None:
            return trial_bench_latency_ms(trial)
        stored = reference_model_latencies(study).get(constraint.reference)
        if stored is None:
            return None
        return float(stored["latency_ms"])
    return None


def count_trained_here(study: optuna.Study) -> int:
    """Count the trials whose training this study paid for.

    A training stopped because its trial turned out to repeat an earlier one
    does not count: the budget is spent on distinct configurations.
    """
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get(ATTR_TRAINED_HERE) and ATTR_DUPLICATE_OF not in trial.user_attrs
    )


def resolve_storage_url(storage: str) -> str:
    """Make a relative SQLite storage path absolute and create its directory."""
    if not storage.startswith(SQLITE_PREFIX):
        return storage
    resolved = resolve_tracking_uri(storage)
    Path(resolved.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_sampler(
    sampler_cfg: DictConfig | Mapping[str, Any] | None, constraint_count: int
) -> optuna.samplers.BaseSampler:
    """Instantiate the configured sampler, wiring the constraint reader in.

    Args:
        sampler_cfg: ``_target_`` config of the sampler, or ``None`` for TPE.
        constraint_count: Number of study constraints; zero disables them.

    Returns:
        The sampler.
    """
    kwargs: dict[str, Any] = {}
    if constraint_count > 0:
        kwargs["constraints_func"] = constraints_from_user_attrs(constraint_count)
    if sampler_cfg is None:
        return optuna.samplers.TPESampler(**kwargs)
    return hydra.utils.instantiate(sampler_cfg, **kwargs)


def open_study(
    study_name: str,
    storage: str,
    objectives: Sequence[Objective],
    sampler: optuna.samplers.BaseSampler,
) -> optuna.Study:
    """Create or load the study on its storage.

    Args:
        study_name: Study name.
        storage: Optuna storage URL; relative SQLite paths resolve to the
            current working directory.
        objectives: Study objectives, giving the directions.
        sampler: Sampler for new trials.

    Returns:
        The study.
    """
    return optuna.create_study(
        study_name=study_name,
        storage=resolve_storage_url(storage),
        directions=[objective.direction for objective in objectives],
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_reference_trials(
    study: optuna.Study,
    search_space: LitePTSearchSpace,
    references: Mapping[str, Mapping[str, Any]] | None,
) -> list[str]:
    """Queue named reference configurations as the study's first trials.

    Args:
        study: Target study.
        search_space: Space the references must belong to.
        references: Reference name to parameter values.

    Returns:
        Names of the references queued now; ones already present are skipped.
    """
    queued = []
    existing = {
        trial.user_attrs.get(ATTR_REFERENCE)
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get(ATTR_REFERENCE)
    }
    for name, params in configured_references(references).items():
        normalized = search_space.validate_params(params)
        if name in existing:
            continue
        study.enqueue_trial(normalized, user_attrs={ATTR_REFERENCE: name}, skip_if_exists=True)
        queued.append(name)
    return queued


def reference_name_for_params(
    search_space: LitePTSearchSpace,
    references: Mapping[str, Mapping[str, Any]] | None,
    params: Mapping[str, Any],
) -> str | None:
    """Return the configured reference whose cell ``params`` falls in, if any."""
    cell = search_space.cell_key(params)
    for name, reference in configured_references(references).items():
        if search_space.cell_key(reference) == cell:
            return name
    return None


def configured_references(
    references: Mapping[str, Mapping[str, Any] | None] | None,
) -> dict[str, dict[str, Any]]:
    """Return the reference trials in force, as plain dicts.

    A study config inherits the reference trials of the config it extends;
    setting one to ``null`` drops it, for a search space the inherited
    cell no longer belongs to.
    """
    return {
        str(name): dict(_to_plain(params))
        for name, params in (references or {}).items()
        if params is not None
    }


def _to_plain(value: Any) -> Any:
    """Return an OmegaConf node or plain container as plain Python."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def trials_in_phase(study: optuna.Study, phase: str) -> list[FrozenTrial]:
    """Return running trials whose recorded phase is ``phase``, by number."""
    return sorted(
        (
            trial
            for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if trial.user_attrs.get(ATTR_PHASE) == phase
        ),
        key=lambda trial: trial.number,
    )
