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

A trial moves through three phases recorded in its user attributes:
``asked`` once sampled, ``trained`` once a checkpoint exists, and
``measured`` once accuracy and latency were reported to the sampler. Only
the last step tells Optuna anything, so the sampler sees every trial with
all of its objectives at once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.utils.mlflow_helpers import resolve_tracking_uri

ATTR_PHASE = "phase"
PHASE_ASKED = "asked"
PHASE_TRAINED = "trained"
PHASE_MEASURED = "measured"

ATTR_OVERRIDES = "overrides"
ATTR_STATIC = "static"
ATTR_REFERENCE = "reference"
ATTR_RUN_ID = "mlflow_run_id"
ATTR_HYDRA_DIR = "hydra_dir"
ATTR_CHECKPOINT = "checkpoint"
ATTR_TRAIN_LOG = "train_log"
ATTR_METRICS = "metrics"
ATTR_LATENCY = "latency"
ATTR_CONSTRAINTS = "constraints"
ATTR_ERROR = "error"
ATTR_MEASURE_ATTEMPTS = "measure_attempts"

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
            the benchmark's mean latency in milliseconds.
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
    """A soft bound on a validation metric the sampler is told about.

    Attributes:
        metric: Metric key.
        min: Lower bound, if any.
        max: Upper bound, if any.
    """

    metric: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        """Require at least one bound."""
        if self.min is None and self.max is None:
            raise ValueError(f"Constraint on {self.metric!r} needs a min or a max.")

    def violation(self, value: float) -> float:
        """Return how far ``value`` lies outside the bounds; feasible when at most zero."""
        worst = 0.0
        if self.min is not None:
            worst = max(worst, self.min - value)
        if self.max is not None:
            worst = max(worst, value - self.max)
        return worst


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
    """Parse the ``tuning.constraints`` config list."""
    return [
        Constraint(
            str(entry["metric"]),
            None if entry.get("min") is None else float(entry["min"]),
            None if entry.get("max") is None else float(entry["max"]),
        )
        for entry in entries or ()
    ]


def objective_values(
    objectives: Sequence[Objective], metrics: Mapping[str, float], latency_ms: float
) -> list[float]:
    """Assemble the objective vector in study order.

    Args:
        objectives: Study objectives.
        metrics: Validation metrics keyed by their logged name.
        latency_ms: Benchmarked mean latency.

    Returns:
        One float per objective.

    Raises:
        KeyError: If a metric objective was not logged.
    """
    values = []
    for objective in objectives:
        if objective.source == LATENCY_SOURCE:
            values.append(float(latency_ms))
        elif objective.name not in metrics:
            raise KeyError(
                f"Objective metric {objective.name!r} was not logged. Available: {sorted(metrics)}"
            )
        else:
            values.append(float(metrics[objective.name]))
    return values


def constraint_values(
    constraints: Sequence[Constraint], metrics: Mapping[str, float]
) -> list[float]:
    """Evaluate every constraint; values at most zero are feasible."""
    return [constraint.violation(float(metrics[constraint.metric])) for constraint in constraints]


def constraints_from_user_attrs(
    constraint_count: int,
) -> Callable[[optuna.trial.FrozenTrial], Sequence[float]]:
    """Build the sampler's ``constraints_func`` reading the stored violations.

    A trial that never reached measurement carries no violations; it is
    reported feasible so the sampler treats it by its state alone.
    """

    def read(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        stored = trial.user_attrs.get(ATTR_CONSTRAINTS)
        if stored is None:
            return [0.0] * constraint_count
        return [float(value) for value in stored]

    return read


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
    for name, params in (references or {}).items():
        normalized = search_space.validate_params(_to_plain(params))
        if name in existing:
            continue
        study.enqueue_trial(normalized, user_attrs={ATTR_REFERENCE: name}, skip_if_exists=True)
        queued.append(name)
    return queued


def _to_plain(value: Any) -> Any:
    """Return an OmegaConf node or plain container as plain Python."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def trials_in_phase(study: optuna.Study, phase: str) -> list[optuna.trial.FrozenTrial]:
    """Return running trials whose recorded phase is ``phase``, by number."""
    return sorted(
        (
            trial
            for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
            if trial.user_attrs.get(ATTR_PHASE) == phase
        ),
        key=lambda trial: trial.number,
    )
