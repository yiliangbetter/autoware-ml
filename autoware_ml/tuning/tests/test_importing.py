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

"""Tests for importing measured trials from an earlier study."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from autoware_ml.tuning.importing import import_trials
from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.study import (
    ATTR_CHECKPOINT,
    ATTR_IMPORT,
    ATTR_LATENCY,
    ATTR_METRICS,
    ATTR_OVERRIDES,
    ATTR_PHASE,
    ATTR_REFERENCE,
    PHASE_IMPORTED,
    build_sampler,
    load_objectives,
    open_study,
)
from autoware_ml.tuning.tests.test_search_space import TINY

MIOU = "val/seg3d/mIoU"


def _source_study(tmp_path: Path) -> tuple[str, str]:
    """A three-objective study of the old layout with the rotary base searched."""
    storage = f"sqlite:///{tmp_path / 'source.db'}"
    study = optuna.create_study(
        study_name="old", storage=storage, directions=["minimize", "maximize", "maximize"]
    )
    space = LitePTSearchSpace(rope_base_range=(10.0, 1000.0))
    rows = [
        ({**TINY, "rope_base": 100.0}, 0.678, 0.40, 25.7, "tiny"),
        ({**TINY, "rope_base": 10.0, "mlp_ratio": 3}, 0.660, 0.38, 21.0, None),
        ({**TINY, "rope_base": 900.0, "mlp_ratio": 3}, 0.664, 0.42, 22.0, None),
        ({**TINY, "rope_base": 50.0, "mlp_ratio": 2}, 0.600, 0.30, 18.0, None),
    ]
    for params, miou, debris, latency, reference in rows:
        attrs = {} if reference is None else {ATTR_REFERENCE: reference}
        study.enqueue_trial(params, user_attrs=attrs)
        trial = study.ask()
        space.suggest(trial)
        trial.set_user_attr(ATTR_METRICS, {MIOU: miou, "val/seg3d/recall_debris": debris})
        trial.set_user_attr(ATTR_CHECKPOINT, f"/ckpt/{trial.number}/best.ckpt")
        trial.set_user_attr(ATTR_LATENCY, {"mean_ms": latency, "p95_ms": latency + 1})
        study.tell(trial, [latency, miou, debris])
    # A trial that never reached measurement carries no metrics.
    study.enqueue_trial({**TINY, "rope_base": 30.0, "mlp_ratio": 4, "patch_size": 256})
    trial = study.ask()
    space.suggest(trial)
    study.tell(trial, state=TrialState.FAIL)
    return storage, "old"


def test_import_merges_replicates_and_keeps_one_trial_per_cell(tmp_path: Path) -> None:
    storage, name = _source_study(tmp_path)
    objectives = load_objectives([{"name": MIOU, "direction": "maximize"}])
    study = open_study(
        "new", f"sqlite:///{tmp_path / 'new.db'}", objectives, build_sampler(None, 0)
    )
    space = LitePTSearchSpace()
    outcomes = import_trials(
        study,
        space,
        source_storage=storage,
        source_study=name,
        objective_metric=MIOU,
        references={"litept_tiny": TINY},
    )
    imported = [outcome for outcome in outcomes if outcome.status == "imported"]
    assert [outcome.sources for outcome in imported] == [[0], [1, 2], [3]]
    skipped = [outcome for outcome in outcomes if outcome.status == "skipped"]
    assert [outcome.sources for outcome in skipped] == [[4]]
    trials = study.get_trials(deepcopy=False)
    assert [trial.state for trial in trials] == [TrialState.RUNNING] * 3
    assert all(trial.user_attrs[ATTR_PHASE] == PHASE_IMPORTED for trial in trials)
    assert all("rope_base" not in trial.params for trial in trials)
    reference, merged, _ = trials
    assert reference.user_attrs[ATTR_REFERENCE] == "litept_tiny"
    assert reference.user_attrs[ATTR_METRICS][MIOU] == pytest.approx(0.678)
    assert merged.params["mlp_ratio"] == 3
    assert merged.user_attrs[ATTR_METRICS][MIOU] == pytest.approx(0.662)
    assert merged.user_attrs[ATTR_METRICS]["val/seg3d/recall_debris"] == pytest.approx(0.40)
    # The representative checkpoint is the replicate nearest the mean.
    assert merged.user_attrs[ATTR_CHECKPOINT] in ("/ckpt/1/best.ckpt", "/ckpt/2/best.ckpt")
    assert merged.user_attrs[ATTR_LATENCY]["mean_ms"] == pytest.approx(21.5)
    assert merged.user_attrs[ATTR_IMPORT]["trials"] == [1, 2]
    assert merged.user_attrs[ATTR_OVERRIDES]
    assert ATTR_REFERENCE not in merged.user_attrs
    # Importing again changes nothing.
    again = import_trials(
        study, space, source_storage=storage, source_study=name, objective_metric=MIOU
    )
    assert all(outcome.status == "skipped" for outcome in again)
    assert len(study.get_trials(deepcopy=False)) == 3


def test_import_attaches_accuracy_to_a_queued_reference_trial(tmp_path: Path) -> None:
    storage, name = _source_study(tmp_path)
    objectives = load_objectives([{"name": MIOU, "direction": "maximize"}])
    study = open_study(
        "new", f"sqlite:///{tmp_path / 'new.db'}", objectives, build_sampler(None, 0)
    )
    space = LitePTSearchSpace()
    # tune init queued the reference and the screen phase asked it already.
    study.enqueue_trial(space.validate_params(TINY), user_attrs={ATTR_REFERENCE: "litept_tiny"})
    reference = study.ask()
    proposal = space.suggest(reference)
    reference.set_user_attr(ATTR_OVERRIDES, list(proposal.overrides))
    reference.set_user_attr(ATTR_PHASE, "asked")
    outcomes = import_trials(
        study,
        space,
        source_storage=storage,
        source_study=name,
        objective_metric=MIOU,
        references={"litept_tiny": TINY},
    )
    attached = [outcome for outcome in outcomes if outcome.status == "attached"]
    assert [outcome.trial_number for outcome in attached] == [0]
    trial = study.trials[0]
    assert trial.user_attrs[ATTR_METRICS][MIOU] == pytest.approx(0.678)
    assert trial.user_attrs[ATTR_PHASE] == PHASE_IMPORTED
    assert trial.user_attrs[ATTR_REFERENCE] == "litept_tiny"
    assert trial.user_attrs[ATTR_IMPORT]["trials"] == [0]
    assert len(study.get_trials(deepcopy=False)) == 3


from autoware_ml.tuning.study import (  # noqa: E402
    ATTR_BENCH,
    ATTR_CONSTRAINTS,
    ATTR_ERROR,
    PHASE_REJECTED,
    PHASE_SCREENED,
    load_constraints,
)

RELATIVE = load_constraints(
    [{"source": "latency", "name": "latency_ms", "reference": "ptv3_tiny", "margin_ms": 0.0}]
)


def _bench(latency: float) -> dict:
    return {"latency_ms": latency, "metric": "gpu_total_ms", "ci95_ms": 0.05, "replicates": 7}


def _bench_source_study(tmp_path: Path) -> tuple[str, str]:
    """A bench-screened study: measured cells with latencies, and three untrained verdicts."""
    storage = f"sqlite:///{tmp_path / 'bench_source.db'}"
    study = optuna.create_study(study_name="bench_old", storage=storage, direction="maximize")
    space = LitePTSearchSpace()

    def add(params, **attrs):
        study.enqueue_trial(params, user_attrs=attrs)
        trial = study.ask()
        space.suggest(trial)
        return trial

    # Two replicates of the reference cell, timed once; one more measured cell.
    first = add(TINY, **{ATTR_REFERENCE: "litept_tiny", ATTR_BENCH: _bench(14.5)})
    first.set_user_attr(ATTR_METRICS, {MIOU: 0.700})
    study.tell(first, 0.700)
    second = add(TINY, **{ATTR_BENCH: _bench(14.7)})
    second.set_user_attr(ATTR_METRICS, {MIOU: 0.690})
    study.tell(second, 0.690)
    medium = add({**TINY, "width_preset": "medium"}, **{ATTR_BENCH: _bench(13.3)})
    medium.set_user_attr(ATTR_METRICS, {MIOU: 0.688})
    study.tell(medium, 0.688)
    # Rejected on the bench: too slow against the old bound, and unbuildable.
    slow = add(
        {**TINY, "patch_size": 1024}, **{ATTR_BENCH: _bench(30.0), ATTR_PHASE: PHASE_REJECTED}
    )
    study.tell(slow, state=TrialState.PRUNED)
    unbuilt = add(
        {**TINY, "width_preset": "base"},
        **{ATTR_PHASE: PHASE_REJECTED, ATTR_ERROR: "engines do not build on the target: Gather"},
    )
    study.tell(unbuilt, state=TrialState.PRUNED)
    # Screened under the old bound, never trained.
    add({**TINY, "mlp_ratio": 2}, **{ATTR_BENCH: _bench(12.0), ATTR_PHASE: PHASE_SCREENED})
    return storage, "bench_old"


def test_import_carries_bench_latencies_and_verdicts_once_the_bound_is_known(
    tmp_path: Path,
) -> None:
    storage, name = _bench_source_study(tmp_path)
    objectives = load_objectives(
        [
            {"name": "latency_ms", "direction": "minimize", "source": "latency"},
            {"name": MIOU, "direction": "maximize"},
        ]
    )
    study = open_study(
        "new", f"sqlite:///{tmp_path / 'new.db'}", objectives, build_sampler(None, len(RELATIVE))
    )
    space = LitePTSearchSpace()
    # Before the bound is measured: measured cells come in with their latency,
    # verdicts wait.
    outcomes = import_trials(
        study,
        space,
        source_storage=[storage],
        source_study=[name],
        objective_metric=MIOU,
        constraints=RELATIVE,
        reference_latency_ms=None,
    )
    by_status = {}
    for outcome in outcomes:
        by_status.setdefault(outcome.status, []).append(outcome)
    assert len(by_status["imported"]) == 2 and len(by_status["skipped"]) == 3
    assert all("waits for the reference latency" in o.detail for o in by_status["skipped"])
    tiny = next(t for t in study.trials if t.user_attrs.get(ATTR_REFERENCE) == "litept_tiny")
    assert tiny.user_attrs[ATTR_PHASE] == PHASE_SCREENED
    assert tiny.user_attrs[ATTR_BENCH]["latency_ms"] == pytest.approx(14.6)
    assert tiny.user_attrs[ATTR_BENCH]["imported_replicates"] == 2
    assert tiny.user_attrs[ATTR_METRICS][MIOU] == pytest.approx(0.695)
    # With the bound known (25 ms): the slow cell is rejected again, the
    # unbuildable one stays infeasible, the screened one joins the queue.
    outcomes = import_trials(
        study,
        space,
        source_storage=[storage],
        source_study=[name],
        objective_metric=MIOU,
        constraints=RELATIVE,
        reference_latency_ms=25.0,
        unbuildable_violation_ms=100.0,
    )
    statuses = sorted(o.status for o in outcomes)
    assert statuses == ["rejected", "rejected", "screened", "skipped", "skipped"]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    assert len(pruned) == 2
    slow = next(t for t in pruned if t.params["patch_size"] == 1024)
    assert slow.user_attrs[ATTR_CONSTRAINTS] == [pytest.approx(5.0)]
    assert "exceeds the bound 25.00 ms" in slow.user_attrs[ATTR_ERROR]
    unbuilt = next(t for t in pruned if t.params["width_preset"] == "base")
    assert unbuilt.user_attrs[ATTR_CONSTRAINTS] == [100.0] and ATTR_BENCH not in unbuilt.user_attrs
    queued = next(t for t in study.trials if t.params.get("mlp_ratio") == 2)
    assert queued.state == TrialState.RUNNING and queued.user_attrs[ATTR_PHASE] == PHASE_SCREENED
    assert ATTR_METRICS not in queued.user_attrs
    # Importing once more changes nothing.
    again = import_trials(
        study,
        space,
        source_storage=[storage],
        source_study=[name],
        objective_metric=MIOU,
        constraints=RELATIVE,
        reference_latency_ms=25.0,
    )
    assert {o.status for o in again} == {"skipped"}
    assert len(study.trials) == 5
