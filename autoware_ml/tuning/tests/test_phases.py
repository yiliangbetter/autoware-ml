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

"""Tests for the train and measure phases with fake launchers and evaluators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from optuna.trial import TrialState

from autoware_ml.tuning.latency import LatencyReport
from autoware_ml.tuning.phases import (
    BEST_CHECKPOINT_NAME,
    CUDA_VISIBLE_DEVICES_ENV,
    STATUS_FAILED,
    STATUS_MEASURED,
    STATUS_PRUNED,
    STATUS_SKIPPED,
    STATUS_TRAINED,
    TrainLaunch,
    TrialEvaluation,
    TuningContext,
    format_status,
    run_measure_phase,
    run_train_phase,
)
from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.study import (
    ATTR_CHECKPOINT,
    ATTR_CONSTRAINTS,
    ATTR_ERROR,
    ATTR_LATENCY,
    ATTR_MEASURE_ATTEMPTS,
    ATTR_METRICS,
    ATTR_OVERRIDES,
    ATTR_PHASE,
    ATTR_RUN_ID,
    ATTR_STATIC,
    PHASE_MEASURED,
    PHASE_TRAINED,
    build_sampler,
    load_constraints,
    load_objectives,
    open_study,
)
from autoware_ml.tuning.tests.test_search_space import TINY
from autoware_ml.tuning.tests.test_study import OBJECTIVES
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_HYDRA_RUN_DIR_ENV,
    AUTOWARE_ML_RUN_ID_ENV,
)
from autoware_ml.utils.runtime import ConfigReference

CONFIG_NAME = "segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2"


@pytest.fixture
def context(tmp_path: Path, monkeypatch) -> TuningContext:
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create(
        {
            "logger": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"},
            "tuning": {
                "study_name": "unit",
                "storage": f"sqlite:///{tmp_path / 'optuna.db'}",
                "objectives": OBJECTIVES,
                "constraints": [{"metric": "val/seg3d/recall_debris", "min": 0.4}],
                "static_filter": {"max_encoder_params": None},
                "latency": {"frames": 4, "warmup": 1, "repeats": 1, "precision": "fp32"},
                "measure": {"max_attempts": 2},
                "train_overrides": ["trainer.max_epochs=1"],
            },
        }
    )
    objectives = load_objectives(OBJECTIVES)
    constraints = load_constraints(cfg.tuning.constraints)
    study = open_study(
        "unit", cfg.tuning.storage, objectives, build_sampler(None, len(constraints))
    )
    space = LitePTSearchSpace()
    study.enqueue_trial(space.validate_params(TINY))
    return TuningContext(
        config=ConfigReference.resolve(CONFIG_NAME),
        cfg=cfg,
        extra_overrides=("trainer.limit_train_batches=2",),
        search_space=space,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        study=study,
        log_dir=tmp_path / "logs",
    )


class _Finished:
    def __init__(self, code: int) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


def _launcher(exit_code: int = 0, write_checkpoint: bool = True):
    launches: list[TrainLaunch] = []

    def launch(item: TrainLaunch):
        launches.append(item)
        if write_checkpoint:
            item.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (item.checkpoint_dir / BEST_CHECKPOINT_NAME).write_bytes(b"ckpt")
        return _Finished(exit_code)

    launch.launches = launches
    return launch


def test_train_phase_launches_the_reference_trial_into_its_own_run(context: TuningContext) -> None:
    launcher = _launcher()
    outcomes = run_train_phase(context, n_trials=1, launcher=launcher, poll_interval=0.0)
    assert [outcome.status for outcome in outcomes] == [STATUS_TRAINED]
    (launch,) = launcher.launches
    assert launch.command[:3] == (sys.executable, "-m", "autoware_ml.scripts.train")
    assert launch.command[3:5] == ("--config-name", f"tasks/{CONFIG_NAME}")
    # Trial overrides first, then the config's fixed ones, then the command line's.
    assert "model.encoder.enc_channels=[32,64,128,256,512]" in launch.command
    assert launch.command.index("trainer.max_epochs=1") < launch.command.index(
        "trainer.limit_train_batches=2"
    )
    assert launch.env[AUTOWARE_ML_RUN_ID_ENV] == launch.run_id
    assert launch.env[AUTOWARE_ML_HYDRA_RUN_DIR_ENV] == str(launch.hydra_dir)
    assert CUDA_VISIBLE_DEVICES_ENV not in launch.env or launch.gpu is not None
    trial = context.study.trials[0]
    assert trial.state == TrialState.RUNNING
    assert trial.user_attrs[ATTR_PHASE] == PHASE_TRAINED
    assert trial.user_attrs[ATTR_RUN_ID] == launch.run_id
    assert Path(trial.user_attrs[ATTR_CHECKPOINT]).name == BEST_CHECKPOINT_NAME
    assert trial.user_attrs[ATTR_OVERRIDES]
    assert trial.user_attrs[ATTR_STATIC]["head_dim"] == 16


def test_train_phase_assigns_one_trial_per_gpu(context: TuningContext) -> None:
    launcher = _launcher()
    outcomes = run_train_phase(
        context, n_trials=3, gpus=["0", "1"], launcher=launcher, poll_interval=0.0
    )
    assert [outcome.status for outcome in outcomes] == [STATUS_TRAINED] * 3
    assert [launch.env[CUDA_VISIBLE_DEVICES_ENV] for launch in launcher.launches] == ["0", "1", "0"]
    with pytest.raises(ValueError, match="Duplicate"):
        run_train_phase(context, n_trials=1, gpus=["0", "0"], launcher=launcher)


def test_train_phase_fails_trials_without_a_checkpoint_or_with_an_exit_code(
    context: TuningContext,
) -> None:
    outcomes = run_train_phase(
        context, n_trials=1, launcher=_launcher(write_checkpoint=False), poll_interval=0.0
    )
    assert outcomes[0].status == STATUS_FAILED
    assert "no best.ckpt" in outcomes[0].detail
    outcomes = run_train_phase(
        context, n_trials=1, launcher=_launcher(exit_code=3), poll_interval=0.0
    )
    assert outcomes[0].status == STATUS_FAILED
    assert "code 3" in outcomes[0].detail
    states = [trial.state for trial in context.study.trials]
    assert states == [TrialState.FAIL, TrialState.FAIL]
    assert all(ATTR_ERROR in trial.user_attrs for trial in context.study.trials)


def test_train_phase_prunes_on_the_static_budget_without_a_launch(context: TuningContext) -> None:
    context.cfg.tuning.static_filter.max_encoder_params = 1000
    launcher = _launcher()
    outcomes = run_train_phase(
        context, n_trials=1, launcher=launcher, poll_interval=0.0, max_asks=2
    )
    assert launcher.launches == []
    assert [outcome.status for outcome in outcomes] == [STATUS_PRUNED, STATUS_PRUNED]
    assert all(trial.state == TrialState.PRUNED for trial in context.study.trials)
    assert "exceed the budget" in context.study.trials[0].user_attrs[ATTR_ERROR]


def _report(mean_ms: float) -> LatencyReport:
    return LatencyReport(
        mean_ms, mean_ms, mean_ms, 0.0, mean_ms, mean_ms, 4, 1, 1, "fp32", "cpu", [100.0, 25.0]
    )


def test_measure_phase_tells_objectives_and_constraints(context: TuningContext) -> None:
    run_train_phase(context, n_trials=2, launcher=_launcher(), poll_interval=0.0)
    seen: list[dict] = []
    # Trial 0 sits below the debris-recall bound, trial 1 above it.
    recalls = {0: 0.3, 1: 0.5}

    def evaluator(ctx, trial, latency_settings):
        seen.append(dict(latency_settings))
        assert trial.user_attrs[ATTR_PHASE] == PHASE_TRAINED
        return TrialEvaluation(
            metrics={
                "val/seg3d/mIoU": 0.5,
                "val/seg3d/recall_debris": recalls[trial.number],
                "val/loss": 1.0,
            },
            latency=_report(20.0),
        )

    outcomes = run_measure_phase(
        context, evaluator=evaluator, latency_overrides={"frames": 2, "warmup": None}
    )
    assert [outcome.status for outcome in outcomes] == [STATUS_MEASURED, STATUS_MEASURED]
    assert seen[0] == {"frames": 2, "warmup": 1, "repeats": 1, "precision": "fp32"}
    first, second = context.study.trials
    assert first.state == TrialState.COMPLETE
    assert first.values == [20.0, 0.5, 0.3]
    assert first.user_attrs[ATTR_PHASE] == PHASE_MEASURED
    assert first.user_attrs[ATTR_CONSTRAINTS] == pytest.approx([0.1])
    assert first.user_attrs[ATTR_METRICS]["val/loss"] == 1.0
    assert first.user_attrs[ATTR_LATENCY]["voxels_per_stage"] == [100.0, 25.0]
    assert second.user_attrs[ATTR_CONSTRAINTS] == [0.0]
    # Nothing is left to measure. The violating trial keeps its values for the
    # sampler but stays off the Pareto front; the feasible one defines it.
    assert run_measure_phase(context, evaluator=evaluator) == []
    assert [item.number for item in context.study.best_trials] == [1]
    assert "pareto #1" in format_status(context)


def test_measure_phase_retries_then_fails_a_broken_trial(context: TuningContext) -> None:
    run_train_phase(context, n_trials=1, launcher=_launcher(), poll_interval=0.0)

    def evaluator(ctx, trial, latency_settings):
        raise RuntimeError("out of memory")

    first = run_measure_phase(context, evaluator=evaluator)
    assert first[0].status == STATUS_SKIPPED
    assert "out of memory" in first[0].detail
    trial = context.study.trials[0]
    assert trial.state == TrialState.RUNNING
    assert trial.user_attrs[ATTR_MEASURE_ATTEMPTS] == 1
    second = run_measure_phase(context, evaluator=evaluator)
    assert second[0].status == STATUS_SKIPPED
    third = run_measure_phase(context, evaluator=evaluator)
    assert third[0].status == STATUS_FAILED
    assert context.study.trials[0].state == TrialState.FAIL


def test_status_lists_every_trial(context: TuningContext) -> None:
    run_train_phase(context, n_trials=2, launcher=_launcher(), poll_interval=0.0)
    text = format_status(context)
    assert "RUNNING" in text
    assert "width_preset=tiny" in text
    assert "2 trials, 0 complete" in text
