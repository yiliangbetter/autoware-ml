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

"""Tests for the study bookkeeping helpers."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
from omegaconf import OmegaConf

from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.study import (
    ATTR_CONSTRAINTS,
    ATTR_PHASE,
    ATTR_REFERENCE,
    PHASE_TRAINED,
    Constraint,
    Objective,
    build_sampler,
    constraint_values,
    constraints_from_user_attrs,
    enqueue_reference_trials,
    load_constraints,
    load_objectives,
    objective_values,
    open_study,
    resolve_storage_url,
    trials_in_phase,
)
from autoware_ml.tuning.tests.test_search_space import TINY

OBJECTIVES = [
    {"name": "latency_ms", "direction": "minimize", "source": "latency"},
    {"name": "val/seg3d/mIoU", "direction": "maximize"},
    {"name": "val/seg3d/recall_debris", "direction": "maximize", "source": "metric"},
]


def test_load_objectives_defaults_the_source_to_metric() -> None:
    objectives = load_objectives(OBJECTIVES)
    assert [objective.source for objective in objectives] == ["latency", "metric", "metric"]
    assert [objective.direction for objective in objectives] == ["minimize", "maximize", "maximize"]


def test_load_objectives_rejects_bad_entries() -> None:
    with pytest.raises(ValueError, match="direction"):
        Objective("x", "up", "metric")
    with pytest.raises(ValueError, match="source"):
        Objective("x", "minimize", "guess")
    with pytest.raises(ValueError, match="unique"):
        load_objectives([OBJECTIVES[1], OBJECTIVES[1]])
    with pytest.raises(ValueError, match="At least one"):
        load_objectives([])


def test_objective_values_follow_study_order() -> None:
    objectives = load_objectives(OBJECTIVES)
    metrics = {"val/seg3d/mIoU": 0.5, "val/seg3d/recall_debris": 0.6}
    assert objective_values(objectives, metrics, 12.5) == [12.5, 0.5, 0.6]
    with pytest.raises(KeyError, match="recall_debris"):
        objective_values(objectives, {"val/seg3d/mIoU": 0.5}, 12.5)


def test_constraints_report_violation_magnitude() -> None:
    constraints = load_constraints([{"metric": "r", "min": 0.4}, {"metric": "l", "max": 30.0}])
    assert constraint_values(constraints, {"r": 0.5, "l": 20.0}) == [0.0, 0.0]
    assert constraint_values(constraints, {"r": 0.1, "l": 35.0}) == pytest.approx([0.3, 5.0])
    with pytest.raises(ValueError, match="min or a max"):
        Constraint("r")


def test_constraints_func_reads_stored_violations_or_assumes_feasible() -> None:
    read = constraints_from_user_attrs(2)
    study = optuna.create_study()
    trial = study.ask()
    trial.set_user_attr(ATTR_CONSTRAINTS, [0.1, -0.2])
    study.tell(trial, 1.0)
    assert list(read(study.trials[0])) == [0.1, -0.2]
    bare = study.ask()
    study.tell(bare, 1.0)
    assert list(read(study.trials[1])) == [0.0, 0.0]


def test_build_sampler_wires_constraints_into_the_configured_sampler() -> None:
    cfg = OmegaConf.create(
        {"_target_": "optuna.samplers.TPESampler", "seed": 1, "constant_liar": True}
    )
    sampler = build_sampler(cfg, constraint_count=1)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._constraints_func is not None
    default = build_sampler(None, constraint_count=0)
    assert isinstance(default, optuna.samplers.TPESampler)
    assert default._constraints_func is None


def test_resolve_storage_url_makes_sqlite_paths_absolute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_storage_url("sqlite:///mlruns/optuna/study.db")
    assert resolved == f"sqlite:///{(tmp_path / 'mlruns/optuna/study.db').resolve()}"
    assert (tmp_path / "mlruns/optuna").is_dir()
    assert resolve_storage_url("postgresql://host/db") == "postgresql://host/db"


def test_open_study_is_idempotent_and_multi_objective(tmp_path: Path) -> None:
    objectives = load_objectives(OBJECTIVES)
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    sampler = build_sampler(None, 0)
    study = open_study("s", storage, objectives, sampler)
    again = open_study("s", storage, objectives, sampler)
    assert study.directions == again.directions
    assert [direction.name for direction in study.directions] == [
        "MINIMIZE",
        "MAXIMIZE",
        "MAXIMIZE",
    ]


def test_enqueue_reference_trials_is_idempotent_and_validated(tmp_path: Path) -> None:
    objectives = load_objectives(OBJECTIVES)
    study = open_study(
        "s", f"sqlite:///{tmp_path / 'study.db'}", objectives, build_sampler(None, 0)
    )
    space = LitePTSearchSpace()
    references = {
        "tiny": TINY,
        "base": {**TINY, "width_preset": "base", "attn_depth_preset": "6-2"},
    }
    assert enqueue_reference_trials(study, space, references) == ["tiny", "base"]
    first = study.ask()
    assert first.user_attrs[ATTR_REFERENCE] == "tiny"
    assert enqueue_reference_trials(study, space, references) == []
    with pytest.raises(ValueError, match="not one of the allowed"):
        enqueue_reference_trials(study, space, {"bad": {**TINY, "width_preset": "giant"}})


def test_trials_in_phase_lists_running_trials_by_number() -> None:
    study = optuna.create_study()
    first = study.ask()
    first.set_user_attr(ATTR_PHASE, PHASE_TRAINED)
    second = study.ask()
    second.set_user_attr(ATTR_PHASE, "asked")
    third = study.ask()
    third.set_user_attr(ATTR_PHASE, PHASE_TRAINED)
    assert [trial.number for trial in trials_in_phase(study, PHASE_TRAINED)] == [0, 2]
    study.tell(first, 1.0)
    assert [trial.number for trial in trials_in_phase(study, PHASE_TRAINED)] == [2]
