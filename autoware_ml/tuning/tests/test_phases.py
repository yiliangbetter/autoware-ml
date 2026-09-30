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
    assert second.user_attrs[ATTR_CONSTRAINTS] == pytest.approx([-0.1])
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


# --------------------------------------------------------------------------
# Bench mode: screen on the bench, train the queue, measure, terminate.

import datetime as dt  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from autoware_ml.tuning.bench import BenchMeasurement, BenchSettings  # noqa: E402
from autoware_ml.tuning.phases import (  # noqa: E402
    STATUS_BUDGET,
    drain_waiting_trials,
    STATUS_IDLE,
    STATUS_QUEUE_FULL,
    STATUS_REJECTED,
    STATUS_SCREENED,
    STATUS_TERMINATED,
    STATUS_WINDOW_CLOSED,
    run_screen_phase,
    screened_queue_length,
)
import optuna  # noqa: E402

from autoware_ml.tuning.study import (  # noqa: E402
    ATTR_BENCH,
    ATTR_BENCH_ATTEMPTS,
    ATTR_IMPORT,
    count_trained_here,
    ATTR_LATENCY_LIMIT,
    ATTR_TRAINED_HERE,
    PHASE_IMPORTED,
    PHASE_REJECTED,
    PHASE_SCREENED,
    STUDY_ATTR_TERMINATED,
    trial_is_feasible,
)
from autoware_ml.tuning.terminator import StudyTerminator, TerminatorSettings  # noqa: E402

MIOU = "val/seg3d/mIoU"
DEBRIS = "val/seg3d/recall_debris"
# Bench latencies by width preset: tiny is the reference at 13.5 ms.
BENCH_MS = {"tiny": 13.5, "medium": 12.0, "small": 10.0, "micro": 8.0, "base": 16.0}
TOKYO = ZoneInfo("Asia/Tokyo")
OPEN = dt.datetime(2026, 9, 30, 20, 0, tzinfo=TOKYO)
CLOSED = dt.datetime(2026, 9, 30, 12, 0, tzinfo=TOKYO)


class _FakeRunner:
    def __init__(self, latencies: dict[str, float] | None = None) -> None:
        self.calls: list[str] = []
        self.latencies = latencies or BENCH_MS

    def measure(self, name, package_dir, description):
        self.calls.append(name)
        width = description.split("width_preset=")[1].split()[0]
        latency = self.latencies[width]
        return BenchMeasurement(
            latency,
            "gpu_total_ms",
            "replicate_median_mean",
            7,
            [latency] * 7,
            [latency] * 7,
            0.05,
            latency - 0.5,
            latency + 0.5,
            50,
            116000.0,
            name,
            "s",
            str(package_dir),
            "sha",
            "onnx",
            "engine",
            "v12",
            "gpu",
            "trt",
            "now",
        )


def _fake_exporter(ctx, trial):
    package = ctx.package_dir(trial.number)
    package.mkdir(parents=True, exist_ok=True)
    (package / "ptv3_encoder.onnx").write_bytes(b"onnx")
    return package


@pytest.fixture
def bench_context(tmp_path: Path, monkeypatch) -> TuningContext:
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create(
        {
            "logger": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"},
            "tuning": {
                "study_name": "bench",
                "storage": f"sqlite:///{tmp_path / 'optuna.db'}",
                "max_trained_trials": 3,
                "objectives": [{"name": MIOU, "direction": "maximize"}],
                "constraints": [
                    {"metric": DEBRIS, "min": 0.35},
                    {"source": "latency", "reference": "litept_tiny", "margin_ms": 0.1},
                ],
                "static_filter": {"max_encoder_params": None},
                "latency": {"proxy": False},
                "measure": {"max_attempts": 2},
                "train_overrides": [],
                "reference_trials": {"litept_tiny": TINY},
                "bench": {
                    "host": "bench",
                    "exchange": {"local": str(tmp_path / "exchange"), "remote": "/remote"},
                    "harness": {"repo": "/repo", "command": "uv run"},
                    "variant": {"prefix": "t"},
                    "reference": "litept_tiny",
                    "window": {"start": "18:00", "end": "09:00", "timezone": "Asia/Tokyo"},
                    "queue": {"target": 2, "target_before_close": 4, "close_buffer_hours": 2},
                },
            },
        }
    )
    objectives = load_objectives(OmegaConf.to_container(cfg.tuning.objectives, resolve=True))
    constraints = load_constraints(OmegaConf.to_container(cfg.tuning.constraints, resolve=True))
    sampler = build_sampler(
        {"_target_": "optuna.samplers.TPESampler", "seed": 0, "n_startup_trials": 100},
        len(constraints),
    )
    study = open_study("bench", cfg.tuning.storage, objectives, sampler)
    space = LitePTSearchSpace()
    study.enqueue_trial(space.validate_params(TINY), user_attrs={"reference": "litept_tiny"})
    return TuningContext(
        config=ConfigReference.resolve(CONFIG_NAME),
        cfg=cfg,
        extra_overrides=(),
        search_space=space,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        study=study,
        log_dir=tmp_path / "logs",
        bench=BenchSettings.from_config(cfg.tuning.bench),
        terminator=StudyTerminator(
            TerminatorSettings(improvement="stagnation", max_stagnation_trials=1, min_n_trials=2)
        ),
    )


def test_screen_phase_measures_the_reference_first_and_applies_the_bound(
    bench_context: TuningContext,
) -> None:
    runner = _FakeRunner()
    outcomes = run_screen_phase(
        bench_context, runner, now=OPEN, queue_target=2, exporter=_fake_exporter, max_asks=30
    )
    assert runner.calls[0] == "t-t0000"
    study = bench_context.study
    reference = study.trials[0]
    assert reference.user_attrs[ATTR_BENCH]["latency_ms"] == pytest.approx(13.5)
    assert reference.user_attrs[ATTR_LATENCY_LIMIT] == pytest.approx(13.6)
    assert reference.user_attrs[ATTR_PHASE] == PHASE_SCREENED
    statuses = {o.trial_number: o.status for o in outcomes if o.trial_number >= 0}
    assert statuses[0] == STATUS_SCREENED
    assert screened_queue_length(bench_context) == 2
    rejected = [t for t in study.trials if t.user_attrs.get(ATTR_PHASE) == PHASE_REJECTED]
    screened = [t for t in study.trials if t.user_attrs.get(ATTR_PHASE) == PHASE_SCREENED]
    assert len(screened) == 2
    for trial in rejected:
        assert trial.state == TrialState.PRUNED
        assert trial.params["width_preset"] == "base"
        assert trial.user_attrs[ATTR_CONSTRAINTS][1] == pytest.approx(16.0 - 13.6)
        assert trial.user_attrs[ATTR_CONSTRAINTS][0] == 0.0
        # The sampler recorded the violation, so the trial counts as infeasible.
        assert sum(v for v in trial.system_attrs["constraints"] if v > 0) > 0
    for trial in screened:
        assert trial.state == TrialState.RUNNING
        assert BENCH_MS[trial.params["width_preset"]] <= 13.6
    # Every trial the bench saw carries its measurement; nothing was measured twice.
    measured = [t for t in study.trials if ATTR_BENCH in t.user_attrs]
    assert sorted(runner.calls) == sorted(f"t-t{t.number:04d}" for t in measured)
    # A second call finds the queue full and touches the bench for nothing.
    again = run_screen_phase(
        bench_context, runner, now=OPEN, queue_target=2, exporter=_fake_exporter
    )
    assert [o.status for o in again] == [STATUS_QUEUE_FULL]
    assert len(runner.calls) == len(measured)


def test_contended_measurements_are_retried_not_told(bench_context: TuningContext) -> None:
    settings = bench_context.bench
    assert settings is not None
    context = TuningContext(
        **{
            **bench_context.__dict__,
            "bench": settings.__class__(**{**settings.__dict__, "max_ci95_ms": 0.01}),
        }
    )
    runner = _FakeRunner()  # every fake measurement carries a 0.05 ms interval
    outcomes = run_screen_phase(context, runner, now=OPEN, n_trials=1, exporter=_fake_exporter)
    assert outcomes[0].status == STATUS_SKIPPED and "contended" in outcomes[0].detail
    trial = context.study.trials[0]
    assert trial.state == TrialState.RUNNING and ATTR_BENCH not in trial.user_attrs
    assert trial.user_attrs[ATTR_BENCH_ATTEMPTS] == 1
    # The second attempt fails the trial for good: max_attempts is 2 in the fixture.
    again = run_screen_phase(context, runner, now=OPEN, n_trials=1, exporter=_fake_exporter)
    assert again[0].status == STATUS_SKIPPED
    final = run_screen_phase(context, runner, now=OPEN, n_trials=1, exporter=_fake_exporter)
    assert final[0].status == STATUS_FAILED
    assert context.study.trials[0].state == TrialState.FAIL


def test_screen_phase_honours_the_window(bench_context: TuningContext) -> None:
    runner = _FakeRunner()
    closed = run_screen_phase(bench_context, runner, now=CLOSED, exporter=_fake_exporter)
    assert [o.status for o in closed] == [STATUS_WINDOW_CLOSED]
    assert "opens at 2026-09-30 18:00" in closed[0].detail
    assert runner.calls == []
    forced = run_screen_phase(
        bench_context, runner, now=CLOSED, ignore_window=True, n_trials=1, exporter=_fake_exporter
    )
    assert forced[0].status == STATUS_SCREENED and runner.calls == ["t-t0000"]


def test_queued_trials_train_in_order_measure_and_count_against_the_budget(
    bench_context: TuningContext,
) -> None:
    run_screen_phase(
        bench_context, _FakeRunner(), now=OPEN, queue_target=3, exporter=_fake_exporter, max_asks=40
    )
    study = bench_context.study
    queued = [t.number for t in study.trials if t.user_attrs.get(ATTR_PHASE) == PHASE_SCREENED]
    assert len(queued) == 3
    launcher = _launcher()
    outcomes = run_train_phase(bench_context, n_trials=2, launcher=launcher, poll_interval=0.0)
    assert [o.trial_number for o in outcomes] == queued[:2]
    assert all(o.status == STATUS_TRAINED for o in outcomes)
    assert all(study.trials[n].user_attrs[ATTR_TRAINED_HERE] for n in queued[:2])

    def evaluator(ctx, trial, latency_settings):
        assert latency_settings["proxy"] is False
        return TrialEvaluation(metrics={MIOU: 0.6 + trial.number / 100, DEBRIS: 0.5})

    measured = run_measure_phase(bench_context, evaluator=evaluator)
    assert [o.status for o in measured] == [STATUS_MEASURED, STATUS_MEASURED]
    for number in queued[:2]:
        trial = study.trials[number]
        assert trial.state == TrialState.COMPLETE
        assert trial.values == [pytest.approx(0.6 + number / 100)]
        assert trial.user_attrs[ATTR_CONSTRAINTS][0] == pytest.approx(-0.15)
        assert trial.user_attrs[ATTR_CONSTRAINTS][1] <= 0.0
        assert trial_is_feasible(trial)
        assert ATTR_LATENCY not in trial.user_attrs
    # The stagnation terminator with patience one fires after the second trial
    # only if the best did not improve; here the second is better, so it waits.
    assert STUDY_ATTR_TERMINATED not in study.user_attrs
    # One training slot is left in the budget of three, then nothing.
    third = run_train_phase(bench_context, n_trials=5, launcher=launcher, poll_interval=0.0)
    assert [o.status for o in third] == [STATUS_TRAINED]
    assert run_train_phase(bench_context, n_trials=1, launcher=launcher)[0].status == STATUS_BUDGET
    text = format_status(bench_context)
    assert "trainings paid for: 3 of 3" in text
    assert "latency bound: litept_tiny + 0.1 ms = 13.60 ms" in text
    assert "best feasible trial: #" in text


def test_screened_trial_with_imported_accuracy_is_told_instead_of_trained(
    bench_context: TuningContext,
) -> None:
    # The reference is screened first; then its accuracy arrives from an import.
    run_screen_phase(
        bench_context, _FakeRunner(), now=OPEN, n_trials=1, queue_target=0, exporter=_fake_exporter
    )
    study = bench_context.study
    assert study.trials[0].user_attrs[ATTR_PHASE] == PHASE_SCREENED
    optuna.Trial(study, study.trials[0]._trial_id).set_user_attr(
        ATTR_METRICS, {MIOU: 0.678, DEBRIS: 0.405}
    )
    launcher = _launcher()
    outcomes = run_train_phase(bench_context, n_trials=1, launcher=launcher, poll_interval=0.0)
    assert [o.status for o in outcomes] == [STATUS_MEASURED, STATUS_IDLE]
    assert launcher.launches == []
    told = study.trials[0]
    assert told.state == TrialState.COMPLETE and told.values == [pytest.approx(0.678)]
    assert trial_is_feasible(told)
    assert count_trained_here(study) == 0


def test_train_phase_idles_without_a_queue_and_stops_after_termination(
    bench_context: TuningContext,
) -> None:
    idle = run_train_phase(bench_context, n_trials=1, launcher=_launcher(), poll_interval=0.0)
    assert [o.status for o in idle] == [STATUS_IDLE]
    bench_context.study.set_user_attr(STUDY_ATTR_TERMINATED, {"reason": "done"})
    assert run_train_phase(bench_context, n_trials=1, launcher=_launcher())[0].status == (
        STATUS_TERMINATED
    )
    screened = run_screen_phase(
        bench_context, _FakeRunner(), now=OPEN, exporter=_fake_exporter, n_trials=5
    )
    # The reference is still measured (it anchors the bound); no new trial is asked.
    assert [o.status for o in screened] == [STATUS_SCREENED, STATUS_TERMINATED]


def test_imported_trials_are_told_outright_by_their_bench_latency(
    bench_context: TuningContext,
) -> None:
    study = bench_context.study
    space = bench_context.search_space
    # The queued reference must be asked before anything else is queued,
    # since the sampler hands out queued trials in order.
    assert [t.number for t in drain_waiting_trials(bench_context)] == [0]
    for params, miou, debris in (
        ({**TINY, "width_preset": "medium"}, 0.66, 0.40),  # feasible
        ({**TINY, "width_preset": "base"}, 0.69, 0.40),  # too slow
        ({**TINY, "width_preset": "small"}, 0.62, 0.30),  # debris too low
    ):
        study.enqueue_trial(
            space.validate_params(params),
            user_attrs={
                ATTR_PHASE: PHASE_IMPORTED,
                ATTR_METRICS: {MIOU: miou, DEBRIS: debris},
                ATTR_IMPORT: {"study": "old"},
            },
        )
        trial = study.ask()
        proposal = space.suggest(trial)
        trial.set_user_attr(ATTR_OVERRIDES, list(proposal.overrides))
    outcomes = run_screen_phase(
        bench_context, _FakeRunner(), now=OPEN, queue_target=0, exporter=_fake_exporter
    )
    statuses = {o.trial_number: o.status for o in outcomes if o.trial_number >= 0}
    assert statuses[0] == STATUS_SCREENED  # the reference, queued for training
    assert statuses[1] == STATUS_MEASURED and statuses[2] == STATUS_REJECTED
    assert statuses[3] == STATUS_REJECTED
    medium, base, small = study.trials[1], study.trials[2], study.trials[3]
    assert medium.state == base.state == small.state == TrialState.COMPLETE
    assert medium.values == [pytest.approx(0.66)] and trial_is_feasible(medium)
    assert base.values == [pytest.approx(0.69)] and not trial_is_feasible(base)
    assert base.user_attrs[ATTR_CONSTRAINTS] == pytest.approx([-0.05, 2.4])
    assert small.user_attrs[ATTR_CONSTRAINTS] == pytest.approx([0.05, -3.6])
    assert "best feasible trial: #1" in format_status(bench_context)
