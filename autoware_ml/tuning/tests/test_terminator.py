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

"""Tests for the termination criterion on feasible trials."""

from __future__ import annotations

import optuna
import pytest

from autoware_ml.tuning.study import (
    ATTR_CONSTRAINTS,
    STUDY_ATTR_TERMINATED,
    STUDY_ATTR_TERMINATION,
    Objective,
)
from autoware_ml.tuning.terminator import StudyTerminator, TerminatorSettings, study_terminated

OBJECTIVE = Objective("val/seg3d/mIoU", "maximize", "metric")


def _study(values: list[float], infeasible: set[int] = frozenset()) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    for number, value in enumerate(values):
        trial = study.ask()
        trial.suggest_categorical("width", ["a", "b", "c"])
        trial.suggest_categorical("grid", [0.1, 0.12, 0.14])
        trial.set_user_attr(ATTR_CONSTRAINTS, [1.0 if number in infeasible else -0.1])
        study.tell(trial, value)
    return study


def test_settings_validate_the_evaluator_combination() -> None:
    assert TerminatorSettings.from_config(None) is None
    assert TerminatorSettings.from_config({"enabled": False}) is None
    settings = TerminatorSettings.from_config({"constant": 0.004, "min_n_trials": 5})
    assert settings is not None and settings.improvement == "regret_bound"
    with pytest.raises(ValueError, match="improvement"):
        TerminatorSettings(improvement="guess")
    with pytest.raises(ValueError, match="pairs with the emmr"):
        TerminatorSettings(error="median")
    with pytest.raises(ValueError, match="positive"):
        TerminatorSettings(min_n_trials=0)


def test_stagnation_terminates_only_after_the_patience_and_only_on_feasible_trials() -> None:
    values = [0.5] + [0.4] * 10 + [0.45, 0.46]
    terminator = StudyTerminator(
        TerminatorSettings(improvement="stagnation", max_stagnation_trials=10, min_n_trials=5)
    )
    # Over every trial the best is trial 0, unbeaten for twelve trials.
    report = terminator.check(_study(values), OBJECTIVE)
    assert report.n_trials == 13 and report.should_terminate
    assert "unchanged for 12" in report.reason
    # With trial 0 infeasible the feasible best is the last trial: no stagnation.
    report = terminator.check(_study(values, infeasible={0}), OBJECTIVE)
    assert report.n_trials == 12 and not report.should_terminate
    below = terminator.check(_study(values[:3]), OBJECTIVE)
    assert not below.should_terminate and "needed before" in below.reason


def test_regret_bound_records_the_verdict_on_the_study() -> None:
    values = [0.60, 0.62, 0.61, 0.65, 0.64, 0.66, 0.655, 0.66, 0.662, 0.658, 0.661, 0.66]
    study = _study(values)
    terminator = StudyTerminator(
        TerminatorSettings(constant=0.5, min_n_trials=10, top_trials_ratio=0.5, seed=0)
    )
    report = terminator.check_and_record(study, OBJECTIVE)
    assert report.n_trials == 12
    assert report.improvement is not None and report.improvement >= 0.0
    assert report.error == pytest.approx(0.5)
    assert report.should_terminate  # any regret bound is below an error of 0.5
    assert study.user_attrs[STUDY_ATTR_TERMINATION]["reason"] == report.reason
    assert study_terminated(study) == report.to_dict()
    assert study.user_attrs[STUDY_ATTR_TERMINATED]["should_terminate"]
    strict = StudyTerminator(TerminatorSettings(constant=0.0, min_n_trials=10, seed=0))
    assert not strict.check(study, OBJECTIVE).should_terminate
