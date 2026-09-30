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

"""Carry measured trials over from an earlier study.

A trained trial is expensive and its accuracy does not depend on which
study asked for it, so a new study with a different objective layout can
start from the old one's results instead of from the sampler's random
phase. Only the bench latency is missing, which the screen phase supplies.
Replicates of one architecture, deliberate or accidental, are merged into
one trial carrying the mean of their metrics: a single observation per cell
keeps the sampler from over-weighting the cell that happened to be trained
most often.
"""

from __future__ import annotations

import datetime as dt
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import optuna
from optuna.trial import FrozenTrial, TrialState

from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.study import (
    ATTR_CHECKPOINT,
    ATTR_IMPORT,
    ATTR_LATENCY,
    ATTR_METRICS,
    ATTR_OVERRIDES,
    ATTR_PHASE,
    ATTR_REFERENCE,
    ATTR_STATIC,
    PHASE_IMPORTED,
    reference_name_for_params,
    resolve_storage_url,
)

logger = logging.getLogger(__name__)
_EPOCH = dt.datetime(1970, 1, 1)


@dataclass(frozen=True)
class ImportedCell:
    """Replicates of one architecture merged into one observation.

    Attributes:
        params: Parameters in the target space's vocabulary.
        metrics: Mean of the replicates' validation metrics.
        checkpoint: Checkpoint of the replicate closest to the mean objective.
        proxy_latency: Mean eager proxy latency of the replicates, if any.
        sources: Source trial numbers.
        reference: Reference name the source trials carried, if any.
    """

    params: dict[str, Any]
    metrics: dict[str, float]
    checkpoint: str | None
    proxy_latency: dict[str, Any] | None
    sources: list[int]
    reference: str | None = None
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImportOutcome:
    """What happened to one source cell."""

    sources: list[int]
    status: str
    detail: str = ""
    trial_number: int | None = None


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = set(rows[0])
    for row in rows[1:]:
        keys &= set(row)
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in sorted(keys)}


def collect_cells(
    trials: Sequence[FrozenTrial],
    search_space: LitePTSearchSpace,
    objective_metric: str,
) -> tuple[list[ImportedCell], list[ImportOutcome]]:
    """Group the source study's measured trials by architecture.

    Args:
        trials: Trials of the source study.
        search_space: Target search space; parameters it does not sample are
            dropped, trials outside it are skipped.
        objective_metric: Metric used to pick the representative checkpoint.

    Returns:
        The merged cells, and the trials skipped with the reason.
    """
    groups: dict[tuple, list[FrozenTrial]] = {}
    skipped: list[ImportOutcome] = []
    for trial in sorted(trials, key=lambda item: (item.datetime_start or _EPOCH, item.number)):
        if trial.state != TrialState.COMPLETE or ATTR_METRICS not in trial.user_attrs:
            skipped.append(ImportOutcome([trial.number], "skipped", "not measured"))
            continue
        params = {
            name: value for name, value in trial.params.items() if name in search_space.param_names
        }
        try:
            key = search_space.cell_key(params)
        except ValueError as exc:
            skipped.append(ImportOutcome([trial.number], "skipped", str(exc)))
            continue
        groups.setdefault(key, []).append(trial)
    cells = []
    for key, members in groups.items():
        metric_rows = [dict(member.user_attrs[ATTR_METRICS]) for member in members]
        metrics = _mean_metrics(metric_rows)
        target = metrics.get(objective_metric)
        representative = members[0]
        if target is not None:
            representative = min(
                members,
                key=lambda member: abs(
                    float(member.user_attrs[ATTR_METRICS].get(objective_metric, float("inf")))
                    - target
                ),
            )
        latencies = [
            member.user_attrs[ATTR_LATENCY]
            for member in members
            if member.user_attrs.get(ATTR_LATENCY)
        ]
        proxy = None
        if latencies:
            proxy = dict(latencies[0])
            proxy["mean_ms"] = statistics.fmean(float(item["mean_ms"]) for item in latencies)
            proxy["replicates"] = len(latencies)
        references = [
            member.user_attrs[ATTR_REFERENCE]
            for member in members
            if member.user_attrs.get(ATTR_REFERENCE)
        ]
        cells.append(
            ImportedCell(
                params=dict(key),
                metrics=metrics,
                checkpoint=representative.user_attrs.get(ATTR_CHECKPOINT),
                proxy_latency=proxy,
                sources=[member.number for member in members],
                reference=references[0] if references else None,
            )
        )
    return cells, skipped


def existing_cells(
    study: optuna.Study, search_space: LitePTSearchSpace
) -> dict[tuple, FrozenTrial]:
    """Return the cells the target study already holds, with their first trial."""
    cells: dict[tuple, FrozenTrial] = {}
    for trial in sorted(study.get_trials(deepcopy=False), key=lambda item: item.number):
        if trial.state in (TrialState.FAIL, TrialState.WAITING):
            continue
        params = {k: v for k, v in trial.params.items() if k in search_space.param_names}
        try:
            cells.setdefault(search_space.cell_key(params), trial)
        except ValueError:
            continue
    return cells


def _attach_to_existing(
    study: optuna.Study, existing: FrozenTrial, attrs: Mapping[str, Any]
) -> bool:
    """Give a running trial without accuracy the imported accuracy of its cell.

    A queued reference trial is the usual case: the study created it before
    the import, and its accuracy is known from the source study, so it must
    not train again. Its phase becomes ``imported`` until the bench measures
    it; if the bench already has, the trainer tells it outright instead of
    training it.
    """
    if existing.state != TrialState.RUNNING or ATTR_METRICS in existing.user_attrs:
        return False
    live = optuna.Trial(study, existing._trial_id)
    for key, value in attrs.items():
        if key == ATTR_REFERENCE and existing.user_attrs.get(ATTR_REFERENCE):
            continue
        live.set_user_attr(key, value)
    if existing.user_attrs.get("bench") is None:
        live.set_user_attr(ATTR_PHASE, PHASE_IMPORTED)
    return True


def import_trials(
    study: optuna.Study,
    search_space: LitePTSearchSpace,
    *,
    source_storage: str | Sequence[str],
    source_study: str | Sequence[str],
    objective_metric: str,
    references: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ImportOutcome]:
    """Import the measured cells of other studies as ``imported`` trials.

    Each cell becomes a running trial whose parameters are fixed, so the
    sampler sees it like any other trial in flight until the screen phase
    measures its latency and tells its objective. Several sources are pooled
    before grouping, so replicates of one cell spread over studies merge.

    Args:
        study: Target study.
        search_space: Target search space.
        source_storage: Optuna storage URL(s) of the source studies.
        source_study: Name(s) of the source studies, one per storage.
        objective_metric: Metric used to pick each cell's checkpoint.
        references: Configured reference trials, so an imported cell that
            matches one takes its name.

    Returns:
        One outcome per source cell or skipped trial.
    """
    storages = [source_storage] if isinstance(source_storage, str) else list(source_storage)
    names = [source_study] if isinstance(source_study, str) else list(source_study)
    if len(storages) != len(names) or not storages:
        raise ValueError("Pass one source study name per source storage.")
    trials = []
    for storage, name in zip(storages, names, strict=True):
        source = optuna.load_study(study_name=name, storage=resolve_storage_url(storage))
        trials.extend(source.get_trials(deepcopy=False))
    cells, outcomes = collect_cells(trials, search_space, objective_metric)
    present = existing_cells(study, search_space)
    taken_references = {
        trial.user_attrs.get(ATTR_REFERENCE)
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get(ATTR_REFERENCE)
    }
    for cell in cells:
        key = search_space.cell_key(cell.params)
        reference = (
            reference_name_for_params(search_space, references, cell.params) or cell.reference
        )
        attrs: dict[str, Any] = {
            ATTR_METRICS: cell.metrics,
            ATTR_IMPORT: {
                "storage": storages,
                "study": names,
                "trials": cell.sources,
                "replicates": len(cell.sources),
            },
        }
        if cell.checkpoint:
            attrs[ATTR_CHECKPOINT] = cell.checkpoint
        if cell.proxy_latency:
            attrs[ATTR_LATENCY] = cell.proxy_latency
        if reference and reference not in taken_references:
            attrs[ATTR_REFERENCE] = reference
            taken_references.add(reference)
        if key in present:
            existing = present[key]
            if _attach_to_existing(study, existing, attrs):
                outcomes.append(
                    ImportOutcome(
                        cell.sources,
                        "attached",
                        f"accuracy attached to trial {existing.number}",
                        existing.number,
                    )
                )
            else:
                outcomes.append(ImportOutcome(cell.sources, "skipped", "cell already in the study"))
            continue
        attrs[ATTR_PHASE] = PHASE_IMPORTED
        study.enqueue_trial(dict(cell.params), user_attrs=attrs, skip_if_exists=False)
        trial = study.ask()
        proposal = search_space.suggest(trial)
        if search_space.cell_key(proposal.params) != key:
            raise RuntimeError(
                f"Imported trial {trial.number} did not receive its queued parameters."
            )
        trial.set_user_attr(ATTR_OVERRIDES, list(proposal.overrides))
        trial.set_user_attr(ATTR_STATIC, dict(proposal.static))
        present[key] = study._storage.get_trial(trial._trial_id)
        outcomes.append(
            ImportOutcome(
                cell.sources,
                "imported",
                f"{len(cell.sources)} replicate(s), {objective_metric}="
                f"{cell.metrics.get(objective_metric, float('nan')):.4f}",
                trial.number,
            )
        )
        logger.info("Imported source trials %s as trial %d", cell.sources, trial.number)
    return outcomes
