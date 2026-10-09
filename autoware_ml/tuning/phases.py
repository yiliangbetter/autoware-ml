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

"""The phases of a batched study: screen, train, measure.

The screen phase asks the sampler for configurations, exports each with its
initial weights and times it on the bench; a trial slower than the latency
bound is told to the sampler as infeasible right there, the others queue up
for training. The train phase launches one training subprocess per queued
trial, each in its own pre-created MLflow run. The measure phase
re-validates each best checkpoint and only then tells the sampler the
objective, together with every constraint. Without a bench configured, the
train phase asks the sampler itself and the measure phase times the trial
with the eager proxy, as the first study did.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.trial import FrozenTrial, TrialState

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.tuning.bench import (
    BenchBusyError,
    BenchError,
    BenchJobError,
    BenchRunner,
    BenchSettings,
)
from autoware_ml.tuning.export import MAIN_CONTRACT, load_manifest
from autoware_ml.tuning.latency import LatencyReport, benchmark_latency
from autoware_ml.tuning.search_space import LitePTSearchSpace, TrialProposal
from autoware_ml.tuning.study import (
    ATTR_BENCH,
    ATTR_BENCH_ATTEMPTS,
    ATTR_CHECKPOINT,
    ATTR_CONSTRAINTS,
    ATTR_ERROR,
    ATTR_HYDRA_DIR,
    ATTR_LATENCY,
    ATTR_LATENCY_LIMIT,
    ATTR_MEASURE_ATTEMPTS,
    ATTR_METRICS,
    ATTR_OVERRIDES,
    ATTR_PACKAGE,
    ATTR_PHASE,
    ATTR_REFERENCE,
    ATTR_RUN_ID,
    ATTR_STATIC,
    ATTR_TRAIN_LOG,
    ATTR_TRAINED_HERE,
    PHASE_ASKED,
    PHASE_IMPORTED,
    PHASE_MEASURED,
    PHASE_REJECTED,
    PHASE_SCREENED,
    PHASE_TRAINED,
    STUDY_ATTR_TERMINATION,
    Constraint,
    Objective,
    best_feasible_trial,
    constraint_values,
    count_trained_here,
    find_reference_trial,
    latency_constraints,
    objective_values,
    reference_latency_ms,
    reference_model_latencies,
    set_reference_model_latency,
    trial_bench_latency_ms,
    trial_is_feasible,
    trials_in_phase,
)
from autoware_ml.tuning.terminator import StudyTerminator, study_terminated
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_HYDRA_RUN_DIR_ENV,
    AUTOWARE_ML_RUN_ID_ENV,
    prepare_run_context,
    resolve_tracking_uri,
)
from autoware_ml.utils.runtime import ConfigReference

logger = logging.getLogger(__name__)

TRAIN_ENTRYPOINT_MODULE = "autoware_ml.scripts.train"
BEST_CHECKPOINT_NAME = "best.ckpt"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
# Measurement and export run one trial on one device; the study's own
# parallelism is across trials, never inside one.
MEASURE_OVERRIDES = ("trainer.devices=1",)
# Where the task configs live, for reference models named by config.
TASK_CONFIG_PREFIX = "tasks"
# Per-trial attributes of the out-of-memory retry.
ATTR_RETRY_OVERRIDES = "retry_overrides"
ATTR_OOM_RETRIES = "oom_retries"
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUDA error: out of memory")
# The trainer and the screen phase share one GPU: the screen phase exports a
# package only while no training runs, and the trainer does not launch one
# while the screen phase holds the pause. Both are files in the study directory.
PAUSE_FILE_NAME = "PAUSE_TRAIN"
ACTIVE_FILE_NAME = "training.active"
# Force-added so they apply whether or not the datamodule config declares them.
BENCHMARK_OVERRIDES = (
    "++datamodule.validation_dataloader.batch_size=1",
    "++datamodule.validation_dataloader.persistent_workers=false",
)
STATIC_ENCODER_PARAMS = "encoder_params"

STATUS_TRAINED = "trained"
STATUS_MEASURED = "measured"
STATUS_FAILED = "failed"
STATUS_PRUNED = "pruned"
STATUS_SKIPPED = "skipped"
STATUS_SCREENED = "screened"
STATUS_REJECTED = "rejected"
STATUS_WINDOW_CLOSED = "window_closed"
STATUS_QUEUE_FULL = "queue_full"
STATUS_TERMINATED = "terminated"
STATUS_BUDGET = "budget_reached"
STATUS_IDLE = "idle"


class TrainerBusyError(RuntimeError):
    """The trainer kept the GPU for longer than the screen phase could wait."""


def study_dir(context: "TuningContext") -> Path:
    """Directory holding the study's coordination files, next to its logs."""
    return context.log_dir.parent


def mark_training_active(context: "TuningContext") -> None:
    """Record that this process runs a training on the GPU."""
    path = study_dir(context) / ACTIVE_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))


def clear_training_active(context: "TuningContext") -> None:
    """Forget the training this process ran."""
    (study_dir(context) / ACTIVE_FILE_NAME).unlink(missing_ok=True)


def training_active(context: "TuningContext") -> bool:
    """Whether a live trainer process holds the GPU, judged by its recorded pid."""
    path = study_dir(context) / ACTIVE_FILE_NAME
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_while_paused(
    context: "TuningContext", *, poll_s: float = 10.0, max_wait_s: float = 2 * 3600
) -> None:
    """Hold a training launch while the screen phase is exporting packages."""
    path = study_dir(context) / PAUSE_FILE_NAME
    started = time.monotonic()
    announced = False
    while path.exists() and time.monotonic() - started < max_wait_s:
        if not announced:
            logger.info("Training paused: %s exists (the screen phase is exporting)", path)
            announced = True
        time.sleep(poll_s)
    if announced:
        logger.info("Training resumes")


@contextlib.contextmanager
def trainer_paused(context: "TuningContext", max_wait_s: float, *, poll_s: float = 15.0):
    """Keep the trainer off the GPU while the body runs, once the current training ends.

    Raises:
        TrainerBusyError: If a training is still running after ``max_wait_s``.
    """
    pause = study_dir(context) / PAUSE_FILE_NAME
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text(dt.datetime.now().isoformat(timespec="seconds"))
    try:
        time.sleep(2.0)  # a trainer that passed the pause check gets to record its launch
        started = time.monotonic()
        announced = False
        while training_active(context):
            if time.monotonic() - started >= max_wait_s:
                raise TrainerBusyError(
                    f"a training held the GPU for more than {max_wait_s / 3600:.1f} h"
                )
            if not announced:
                logger.info("Waiting for the running training to finish before exporting")
                announced = True
            time.sleep(poll_s)
        yield
    finally:
        pause.unlink(missing_ok=True)


@dataclass(frozen=True)
class TuningContext:
    """Everything the phases need about one study.

    Attributes:
        config: Resolved config reference every compose and launch reuses.
        cfg: Composed configuration including the ``tuning`` block.
        extra_overrides: Command-line overrides applied to every compose and
            every launched training, after the trial's own.
        search_space: The search space instance.
        objectives: Study objectives in order.
        constraints: Study constraints.
        study: The open study.
        log_dir: Directory receiving per-trial logs.
        bench: Bench settings, or ``None`` for the proxy-only workflow.
        terminator: Termination criterion, or ``None`` to run to the budget.
    """

    config: ConfigReference
    cfg: DictConfig
    extra_overrides: tuple[str, ...]
    search_space: LitePTSearchSpace
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    study: optuna.Study
    log_dir: Path
    bench: BenchSettings | None = None
    terminator: StudyTerminator | None = None

    @property
    def config_name(self) -> str:
        """User-facing config name used for MLflow lineage."""
        return self.config.user_config_name

    @property
    def tuning(self) -> DictConfig:
        """The ``tuning`` config block."""
        return self.cfg.tuning

    @property
    def tracking_uri(self) -> str:
        """The MLflow tracking URI trials log to."""
        return str(self.cfg.logger.tracking_uri)

    @property
    def bench_mode(self) -> bool:
        """Whether trials are screened on the bench before training."""
        return self.bench is not None

    @property
    def max_trained_trials(self) -> int | None:
        """Cap on trainings this study pays for, if configured."""
        value = self.tuning.get("max_trained_trials")
        return None if value is None else int(value)

    def train_overrides(self) -> list[str]:
        """Fixed overrides the config applies to every trial's training."""
        return [str(value) for value in self.tuning.get("train_overrides") or []]

    def trial_overrides(self, trial_overrides: Sequence[str], *extra: str) -> list[str]:
        """Order the override layers: trial, then config, then command line."""
        return [*trial_overrides, *self.train_overrides(), *self.extra_overrides, *extra]

    def package_dir(self, trial_number: int) -> Path:
        """Directory an exported package of a trial lands in."""
        assert self.bench is not None
        return self.bench.packages_dir / self.bench.variant.variant_name(trial_number)


@dataclass(frozen=True)
class TrainLaunch:
    """One training subprocess prepared for a trial.

    Attributes:
        trial: The asked trial, kept so attributes can be set and it can be told.
        run_id: Pre-created MLflow run the training logs to.
        hydra_dir: Hydra output directory bound to that run.
        checkpoint_dir: Directory the run's checkpoints land in.
        command: Subprocess command line.
        env: Subprocess environment.
        log_path: File receiving the subprocess output.
        gpu: Device slot the launch occupies, if any.
    """

    trial: optuna.Trial
    run_id: str
    hydra_dir: Path
    checkpoint_dir: Path
    command: tuple[str, ...]
    env: dict[str, str]
    log_path: Path
    gpu: str | None = None

    @property
    def trial_number(self) -> int:
        """The trial's number in the study."""
        return self.trial.number


class TrialProcess(Protocol):
    """The subset of ``subprocess.Popen`` the scheduler polls."""

    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while running."""


Launcher = Callable[[TrainLaunch], TrialProcess]


class _LoggedProcess:
    """A subprocess whose output file is closed once it exits."""

    def __init__(self, process: subprocess.Popen[bytes], log_file: IO[bytes]) -> None:
        self._process = process
        self._log_file = log_file

    def poll(self) -> int | None:
        code = self._process.poll()
        if code is not None and not self._log_file.closed:
            self._log_file.close()
        return code


def launch_subprocess(launch: TrainLaunch) -> TrialProcess:
    """Start the training subprocess, appending its output to the trial log."""
    launch.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(launch.log_path, "ab")  # noqa: SIM115 - closed by _LoggedProcess
    header = f"# trial {launch.trial_number}\n# {' '.join(launch.command)}\n"
    log_file.write(header.encode("utf-8"))
    log_file.flush()
    process = subprocess.Popen(  # noqa: S603 - command assembled from the config
        list(launch.command), env=launch.env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return _LoggedProcess(process, log_file)


@dataclass(frozen=True)
class TrialOutcome:
    """What a phase did with one trial."""

    trial_number: int
    status: str
    detail: str = ""


def _set_attrs(trial: optuna.Trial, attrs: Mapping[str, Any]) -> None:
    for key, value in attrs.items():
        trial.set_user_attr(key, value)


def _running_trial(study: optuna.Study, frozen: FrozenTrial) -> optuna.Trial:
    """Re-open a running trial from its frozen view so attributes can be set."""
    return optuna.Trial(study, frozen._trial_id)


def build_train_command(context: TuningContext, overrides: Sequence[str]) -> tuple[str, ...]:
    """Assemble the training command for one override set.

    The config reference was resolved when the context was built, so a
    filesystem config path works exactly as it does for ``autoware-ml train``.
    """
    argv = [sys.executable, "-m", TRAIN_ENTRYPOINT_MODULE, *context.config.hydra_argv()]
    argv.extend(context.config.with_searchpath(overrides))
    return tuple(argv)


def ask_trial(context: TuningContext) -> tuple[optuna.Trial, TrialProposal] | None:
    """Ask the sampler for a configuration and record it on the trial.

    A proposal rejected by the static filter is told as pruned and ``None``
    is returned, so the caller can ask again without spending anything.
    """
    study = context.study
    trial = study.ask()
    proposal: TrialProposal = context.search_space.suggest(trial)
    _set_attrs(
        trial,
        {
            ATTR_PHASE: PHASE_ASKED,
            ATTR_OVERRIDES: list(proposal.overrides),
            ATTR_STATIC: dict(proposal.static),
        },
    )
    reason = static_rejection(context, proposal)
    if reason is not None:
        trial.set_user_attr(ATTR_ERROR, reason)
        study.tell(trial, state=TrialState.PRUNED)
        logger.info("Trial %d pruned before training: %s", trial.number, reason)
        return None
    return trial, proposal


def prepare_launch(
    context: TuningContext,
    trial: optuna.Trial,
    gpu: str | None = None,
    extra_train_overrides: Sequence[str] = (),
) -> TrainLaunch:
    """Prepare the training run of an asked or screened trial."""
    study = context.study
    reference = trial.user_attrs.get(ATTR_REFERENCE)
    run_context = prepare_run_context(
        context.tracking_uri,
        context.config_name,
        hydra_dir=None,
        stage="train",
        extra_tags={
            "optuna_study": study.study_name,
            "optuna_trial": str(trial.number),
            "optuna_reference": reference or "",
        },
    )
    overrides = context.trial_overrides(
        trial.user_attrs[ATTR_OVERRIDES],
        *(str(item) for item in trial.user_attrs.get(ATTR_RETRY_OVERRIDES) or ()),
        *extra_train_overrides,
    )
    env = dict(os.environ)
    env[AUTOWARE_ML_RUN_ID_ENV] = run_context.run_id
    env[AUTOWARE_ML_HYDRA_RUN_DIR_ENV] = str(run_context.hydra_dir)
    if gpu is not None:
        env[CUDA_VISIBLE_DEVICES_ENV] = gpu
    log_path = context.log_dir / f"trial_{trial.number:04d}.log"
    _set_attrs(
        trial,
        {
            ATTR_RUN_ID: run_context.run_id,
            ATTR_HYDRA_DIR: str(run_context.hydra_dir),
            ATTR_TRAIN_LOG: str(log_path),
            ATTR_TRAINED_HERE: True,
        },
    )
    return TrainLaunch(
        trial=trial,
        run_id=run_context.run_id,
        hydra_dir=run_context.hydra_dir,
        checkpoint_dir=run_context.checkpoints_dir,
        command=build_train_command(context, overrides),
        env=env,
        log_path=log_path,
        gpu=gpu,
    )


def prepare_trial_launch(
    context: TuningContext,
    gpu: str | None = None,
    extra_train_overrides: Sequence[str] = (),
) -> TrainLaunch | None:
    """Ask the sampler for a trial and prepare its training run (proxy workflow)."""
    asked = ask_trial(context)
    if asked is None:
        return None
    trial, _ = asked
    return prepare_launch(context, trial, gpu, extra_train_overrides)


def static_rejection(context: TuningContext, proposal: TrialProposal) -> str | None:
    """Return why a proposal fails the static budget, or ``None`` if it passes."""
    static_filter = context.tuning.get("static_filter") or {}
    limit = static_filter.get("max_encoder_params")
    if limit is None:
        return None
    estimate = proposal.static.get(STATIC_ENCODER_PARAMS)
    if estimate is not None and estimate > int(limit):
        return f"estimated encoder parameters {estimate} exceed the budget of {int(limit)}"
    return None


def oom_retry_overrides(context: TuningContext, launch: TrainLaunch) -> list[str] | None:
    """Return the overrides for one more attempt after a CUDA out-of-memory, if allowed.

    ``tuning.oom_retry.overrides`` names the smaller-batch recipe (the same
    clouds per optimizer step through accumulation keep the schedule) and
    ``max_retries`` how often a trial may take it. The retry only applies in
    bench mode, where the trainer re-reads the screened queue.
    """
    retry_cfg = context.tuning.get("oom_retry") or {}
    overrides = [str(item) for item in retry_cfg.get("overrides") or ()]
    retries = int(launch.trial.user_attrs.get(ATTR_OOM_RETRIES, 0))
    if not overrides or not context.bench_mode or retries >= int(retry_cfg.get("max_retries", 1)):
        return None
    try:
        tail = launch.log_path.read_text(errors="replace")[-200_000:]
    except OSError:
        return None
    if not any(marker in tail for marker in OOM_MARKERS):
        return None
    return overrides


def finish_trial_launch(
    context: TuningContext, launch: TrainLaunch, exit_code: int
) -> TrialOutcome:
    """Record a finished training subprocess on its trial."""
    trial = launch.trial
    if exit_code != 0:
        retry = oom_retry_overrides(context, launch)
        if retry is not None:
            detail = (
                f"training ran out of GPU memory (see {launch.log_path}); "
                f"queued again with {' '.join(retry)}"
            )
            _set_attrs(
                trial,
                {
                    ATTR_RETRY_OVERRIDES: retry,
                    ATTR_OOM_RETRIES: int(trial.user_attrs.get(ATTR_OOM_RETRIES, 0)) + 1,
                    ATTR_PHASE: PHASE_SCREENED,
                    ATTR_ERROR: detail,
                },
            )
            return TrialOutcome(trial.number, STATUS_SKIPPED, detail)
        detail = f"training exited with code {exit_code}, see {launch.log_path}"
        trial.set_user_attr(ATTR_ERROR, detail)
        context.study.tell(trial, state=TrialState.FAIL)
        return TrialOutcome(trial.number, STATUS_FAILED, detail)
    checkpoint = launch.checkpoint_dir / BEST_CHECKPOINT_NAME
    if not checkpoint.is_file():
        detail = f"training produced no {BEST_CHECKPOINT_NAME} in {launch.checkpoint_dir}"
        trial.set_user_attr(ATTR_ERROR, detail)
        context.study.tell(trial, state=TrialState.FAIL)
        return TrialOutcome(trial.number, STATUS_FAILED, detail)
    _set_attrs(trial, {ATTR_CHECKPOINT: str(checkpoint), ATTR_PHASE: PHASE_TRAINED})
    return TrialOutcome(trial.number, STATUS_TRAINED, str(checkpoint))


def tell_known_trials(context: TuningContext) -> list[TrialOutcome]:
    """Tell screened trials whose accuracy was imported instead of training them."""
    outcomes = []
    for frozen in trials_in_phase(context.study, PHASE_SCREENED):
        metrics = frozen.user_attrs.get(ATTR_METRICS)
        if metrics is None:
            continue
        values = tell_measured(context, frozen, metrics)
        told = context.study._storage.get_trial(frozen._trial_id)
        status = STATUS_MEASURED if trial_is_feasible(told) else STATUS_REJECTED
        outcomes.append(TrialOutcome(frozen.number, status, f"imported accuracy, told {values}"))
    return outcomes


def training_budget_left(context: TuningContext) -> int | None:
    """Return how many trainings the budget still allows, ``None`` for unlimited."""
    cap = context.max_trained_trials
    if cap is None:
        return None
    return max(0, cap - count_trained_here(context.study))


def run_train_phase(
    context: TuningContext,
    *,
    n_trials: int,
    gpus: Sequence[str] | None = None,
    extra_train_overrides: Sequence[str] = (),
    launcher: Launcher = launch_subprocess,
    poll_interval: float = 5.0,
    max_asks: int | None = None,
    ignore_termination: bool = False,
) -> list[TrialOutcome]:
    """Train ``n_trials`` trials, one subprocess per device slot.

    With a bench configured the trials come from the screened queue in
    order; otherwise the sampler is asked for new ones.

    Args:
        context: The study context.
        n_trials: Number of trainings to launch.
        gpus: Device ids, one concurrent training each; ``None`` runs one
            training at a time on whatever is visible.
        extra_train_overrides: Overrides appended to every launched training.
        launcher: Starts a prepared launch; injectable for tests.
        poll_interval: Seconds between completion polls.
        max_asks: Cap on sampler asks, bounding static rejections.
        ignore_termination: Train even after the study was terminated.

    Returns:
        One outcome per trial that was considered.
    """
    if n_trials <= 0:
        return []
    if gpus and len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate device ids in {list(gpus)}.")
    if not ignore_termination and study_terminated(context.study):
        return [TrialOutcome(-1, STATUS_TERMINATED, "the study was terminated")]
    budget = training_budget_left(context)
    if budget is not None:
        if budget == 0:
            return [TrialOutcome(-1, STATUS_BUDGET, f"{context.max_trained_trials} trainings")]
        n_trials = min(n_trials, budget)
    slots: deque[str | None] = deque(gpus) if gpus else deque([None])
    ask_budget = max_asks if max_asks is not None else 10 * n_trials
    outcomes: list[TrialOutcome] = []
    queue = None
    if context.bench_mode:
        outcomes.extend(tell_known_trials(context))
        queue = deque(trials_in_phase(context.study, PHASE_SCREENED))
    running: list[tuple[TrainLaunch, TrialProcess]] = []
    launched = 0
    asked = 0
    while True:
        while slots and launched < n_trials and asked < ask_budget:
            gpu = slots.popleft()
            if queue is not None:
                if not queue:
                    slots.appendleft(gpu)
                    break
                frozen = queue.popleft()
                launch = prepare_launch(
                    context, _running_trial(context.study, frozen), gpu, extra_train_overrides
                )
            else:
                asked += 1
                launch = prepare_trial_launch(context, gpu, extra_train_overrides)
                if launch is None:
                    slots.appendleft(gpu)
                    outcomes.append(TrialOutcome(asked - 1, STATUS_PRUNED, "static filter"))
                    continue
            logger.info(
                "Trial %d: training on %s, log %s",
                launch.trial_number,
                gpu if gpu is not None else "visible devices",
                launch.log_path,
            )
            wait_while_paused(context)
            running.append((launch, launcher(launch)))
            mark_training_active(context)
            launched += 1
        if not running:
            break
        still_running: list[tuple[TrainLaunch, TrialProcess]] = []
        for launch, process in running:
            code = process.poll()
            if code is None:
                still_running.append((launch, process))
                continue
            slots.append(launch.gpu)
            outcome = finish_trial_launch(context, launch, code)
            outcomes.append(outcome)
            logger.info("Trial %d: %s %s", outcome.trial_number, outcome.status, outcome.detail)
        running = still_running
        if running:
            time.sleep(poll_interval)
    clear_training_active(context)
    if queue is not None and not any(o.status in (STATUS_TRAINED, STATUS_FAILED) for o in outcomes):
        outcomes.append(TrialOutcome(-1, STATUS_IDLE, "no screened trial is waiting"))
    return outcomes


@dataclass(frozen=True)
class TrialEvaluation:
    """Accuracy and, optionally, proxy latency of one trained trial."""

    metrics: dict[str, float]
    latency: LatencyReport | None = None


Evaluator = Callable[[TuningContext, FrozenTrial, Mapping[str, Any]], TrialEvaluation]


def evaluate_trial(
    context: TuningContext, trial: FrozenTrial, latency_settings: Mapping[str, Any]
) -> TrialEvaluation:
    """Re-validate a trial's best checkpoint and, if enabled, time it eagerly.

    Validation runs with the trial's own configuration on one device, so the
    reported accuracy belongs to the checkpoint the latency belongs to.
    """
    import hydra
    import torch

    from autoware_ml.utils.checkpoints import apply_matching_weights
    from autoware_ml.utils.runtime import configure_torch_runtime, set_seed

    overrides = context.trial_overrides(trial.user_attrs[ATTR_OVERRIDES], *MEASURE_OVERRIDES)
    cfg = context.config.compose(overrides)
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model)
    model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))
    apply_matching_weights(
        model,
        trial.user_attrs[ATTR_CHECKPOINT],
        map_location=device,
        device=device,
        set_eval=True,
        enforce_full_coverage=True,
        logger=logger,
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=False,
        default_root_dir=str(context.log_dir / "measure"),
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.validate(model, datamodule=datamodule, verbose=False)
    metrics = {str(key): float(value) for key, value in trainer.callback_metrics.items()}
    if not latency_settings.get("proxy", True):
        return TrialEvaluation(metrics=metrics)

    try:
        benchmark_cfg = context.config.compose([*overrides, *BENCHMARK_OVERRIDES])
        benchmark_datamodule = hydra.utils.instantiate(benchmark_cfg.datamodule)
        benchmark_datamodule.setup("validate")
        model.to(device).eval()

        def forward(batch: ModelGTBatch) -> Any:
            prepared = model.on_after_batch_transfer(batch, 0)
            return model(**model.forward_inputs(prepared))

        report = benchmark_latency(
            forward,
            benchmark_datamodule.val_dataloader(),
            device=device,
            precision=str(latency_settings.get("precision", "fp32")),
            frames=int(latency_settings.get("frames", 50)),
            warmup=int(latency_settings.get("warmup", 10)),
            repeats=int(latency_settings.get("repeats", 5)),
            stage_counter=encoder_stage_counter(model),
        )
    except Exception:  # noqa: BLE001 - the proxy is informational, the accuracy is not
        logger.exception(
            "Trial %d: proxy latency failed; recording the accuracy only", trial.number
        )
        return TrialEvaluation(metrics=metrics)
    return TrialEvaluation(metrics=metrics, latency=report)


def encoder_stage_counter(model: Any) -> Callable[[ModelGTBatch], Sequence[int]] | None:
    """Return a per-stage point counter for PTv3-style encoders, else ``None``."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    from autoware_ml.models.segmentation3d.encoders.ptv3 import collect_encoder_stage_points

    def count(batch: ModelGTBatch) -> Sequence[int]:
        prepared = model.on_after_batch_transfer(batch, 0)
        inputs = model.forward_inputs(prepared)
        deepest = encoder(
            {key: inputs[key] for key in ("coord", "feat", "grid_coord", "offset") if key in inputs}
        )
        return [int(point.feat.shape[0]) for point in collect_encoder_stage_points(deepest)]

    return count


def trial_latency_ms(context: TuningContext, trial: FrozenTrial) -> float | None:
    """The latency the objectives and constraints read: bench first, proxy otherwise."""
    if context.bench_mode:
        return trial_bench_latency_ms(trial)
    proxy = trial.user_attrs.get(ATTR_LATENCY)
    return None if not proxy else float(proxy["mean_ms"])


def tell_measured(
    context: TuningContext, trial: FrozenTrial, metrics: Mapping[str, float]
) -> list[float]:
    """Tell the sampler a trial's objectives and constraints from its stored data."""
    latency = trial_latency_ms(context, trial)
    reference = reference_latency_ms(context.study, context.constraints)
    values = objective_values(context.objectives, metrics, latency)
    violations = constraint_values(context.constraints, metrics, latency, reference)
    live = _running_trial(context.study, trial)
    _set_attrs(live, {ATTR_CONSTRAINTS: violations, ATTR_PHASE: PHASE_MEASURED})
    context.study.tell(trial.number, values)
    return values


def record_measurement(
    context: TuningContext, trial: FrozenTrial, evaluation: TrialEvaluation
) -> list[float]:
    """Store the evaluation on the trial and tell the sampler its objectives.

    Returns:
        The objective values told.
    """
    live = _running_trial(context.study, trial)
    attrs: dict[str, Any] = {ATTR_METRICS: dict(evaluation.metrics)}
    if evaluation.latency is not None:
        attrs[ATTR_LATENCY] = evaluation.latency.to_dict()
    _set_attrs(live, attrs)
    frozen = context.study._storage.get_trial(trial._trial_id)
    return tell_measured(context, frozen, evaluation.metrics)


def log_measurement_to_mlflow(
    context: TuningContext, trial: FrozenTrial, evaluation: TrialEvaluation, values: Sequence[float]
) -> None:
    """Mirror the measurement onto the trial's MLflow run, best effort."""
    run_id = trial.user_attrs.get(ATTR_RUN_ID)
    if run_id is None:
        return
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=resolve_tracking_uri(context.tracking_uri))
        if evaluation.latency is not None:
            client.log_metric(run_id, "tune/proxy_latency_mean_ms", evaluation.latency.mean_ms)
            client.log_metric(run_id, "tune/proxy_latency_p95_ms", evaluation.latency.p95_ms)
        bench = trial.user_attrs.get(ATTR_BENCH)
        if bench:
            client.log_metric(run_id, "tune/bench_latency_ms", float(bench["latency_ms"]))
        for objective, value in zip(context.objectives, values, strict=True):
            client.log_metric(run_id, f"tune/objective/{objective.name}", value)
        client.set_tag(run_id, "optuna_phase", PHASE_MEASURED)
    except Exception as exc:  # noqa: BLE001 - the study is the source of truth
        logger.warning("Could not mirror trial %d to MLflow run %s: %s", trial.number, run_id, exc)


def run_measure_phase(
    context: TuningContext,
    *,
    evaluator: Evaluator = evaluate_trial,
    max_trials: int | None = None,
    max_attempts: int | None = None,
    latency_overrides: Mapping[str, Any] | None = None,
) -> list[TrialOutcome]:
    """Measure every trained trial and report it to the sampler.

    A trial whose evaluation raises stays trained so a later run can retry it,
    up to ``max_attempts`` after which it fails. After the batch the
    termination criterion, if configured, is evaluated and recorded.

    Args:
        context: The study context.
        evaluator: Produces the evaluation; injectable for tests.
        max_trials: Cap on trials measured in this call.
        max_attempts: Evaluation attempts before a trial is failed; defaults
            to the config's ``measure.max_attempts`` or 3.
        latency_overrides: Proxy benchmark settings overriding the config's.

    Returns:
        One outcome per trial considered.
    """
    measure_cfg = context.tuning.get("measure") or {}
    attempts_limit = (
        int(max_attempts) if max_attempts is not None else int(measure_cfg.get("max_attempts", 3))
    )
    latency_settings: dict[str, Any] = dict(
        OmegaConf.to_container(context.tuning.latency, resolve=True)
        if "latency" in context.tuning
        else {}
    )
    latency_settings.update(
        {key: value for key, value in (latency_overrides or {}).items() if value is not None}
    )
    trials = trials_in_phase(context.study, PHASE_TRAINED)
    if max_trials is not None:
        trials = trials[:max_trials]
    outcomes: list[TrialOutcome] = []
    relative = [c for c in latency_constraints(context.constraints) if c.reference is not None]
    if trials and relative and reference_latency_ms(context.study, context.constraints) is None:
        # Telling a trial needs the bound; until the reference is measured the
        # trained trials simply wait, which costs no measurement attempt.
        return [
            TrialOutcome(
                -1,
                STATUS_SKIPPED,
                f"{len(trials)} trained trial(s) wait: the latency of reference "
                f"{relative[0].reference!r} is still unknown",
            )
        ]
    for frozen in trials:
        attempts = int(frozen.user_attrs.get(ATTR_MEASURE_ATTEMPTS, 0))
        live = _running_trial(context.study, frozen)
        if attempts >= attempts_limit:
            detail = f"measurement failed {attempts} times: {frozen.user_attrs.get(ATTR_ERROR, '')}"
            live.set_user_attr(ATTR_ERROR, detail)
            context.study.tell(frozen.number, state=TrialState.FAIL)
            outcomes.append(TrialOutcome(frozen.number, STATUS_FAILED, detail))
            continue
        live.set_user_attr(ATTR_MEASURE_ATTEMPTS, attempts + 1)
        logger.info("Trial %d: measuring (attempt %d)", frozen.number, attempts + 1)
        try:
            evaluation = evaluator(context, frozen, latency_settings)
            values = record_measurement(context, frozen, evaluation)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            detail = f"{type(exc).__name__}: {exc}"
            live.set_user_attr(ATTR_ERROR, detail)
            logger.exception("Trial %d: measurement failed", frozen.number)
            outcomes.append(TrialOutcome(frozen.number, STATUS_SKIPPED, detail))
            continue
        log_measurement_to_mlflow(context, frozen, evaluation, values)
        detail = ", ".join(
            f"{objective.name}={value:.4g}"
            for objective, value in zip(context.objectives, values, strict=True)
        )
        logger.info("Trial %d: %s", frozen.number, detail)
        outcomes.append(TrialOutcome(frozen.number, STATUS_MEASURED, detail))
    if context.terminator is not None and any(o.status == STATUS_MEASURED for o in outcomes):
        report = context.terminator.check_and_record(context.study, context.objectives[0])
        logger.info("Termination check: %s", report.reason)
    return outcomes


# --------------------------------------------------------------------------
# Screen phase


@dataclass(frozen=True)
class ScreenOutcome(TrialOutcome):
    """A screen outcome carries the latency it measured."""

    latency_ms: float | None = None
    limit_ms: float | None = None


def export_contract(context: TuningContext) -> str:
    """The encoder export contract the bench packages follow."""
    return context.bench.export_contract if context.bench is not None else MAIN_CONTRACT


def trial_has_package(context: TuningContext, trial: FrozenTrial) -> bool:
    """Whether the trial's package exists under the current contract."""
    return load_manifest(context.package_dir(trial.number), export_contract(context)) is not None


def export_trial_package(context: TuningContext, trial: FrozenTrial) -> Path:
    """Export the trial's package unless a complete one already exists."""
    from autoware_ml.tuning.export import export_package

    if trial.state == TrialState.WAITING:
        drain_waiting_trials(context)
        trial = context.study._storage.get_trial(trial._trial_id)
    package_dir = context.package_dir(trial.number)
    contract = export_contract(context)
    manifest = load_manifest(package_dir, contract)
    if manifest is None:
        if package_dir.exists():
            shutil.rmtree(package_dir)  # incomplete, or exported under another contract
        overrides = context.trial_overrides(trial.user_attrs[ATTR_OVERRIDES], *MEASURE_OVERRIDES)
        cfg = context.config.compose(overrides)
        logger.info("Trial %d: exporting package to %s (%s)", trial.number, package_dir, contract)
        export_package(
            cfg,
            package_dir,
            config_name=context.config_name,
            params=dict(trial.params),
            overrides=overrides,
            contract=contract,
        )
    _running_trial(context.study, trial).set_user_attr(ATTR_PACKAGE, str(package_dir))
    return package_dir


def _describe_params(trial: FrozenTrial) -> str:
    return " ".join(f"{key}={_short(value)}" for key, value in sorted(trial.params.items()))


def screen_trial(
    context: TuningContext,
    trial: FrozenTrial,
    runner: BenchRunner,
    *,
    exporter: Callable[[TuningContext, FrozenTrial], Path] = export_trial_package,
) -> ScreenOutcome:
    """Time one trial on the bench and act on the latency bound.

    An imported trial is told outright, since its accuracy is known. A
    sampled trial slower than the bound is told as pruned with its violation
    so the sampler learns where the bound runs; a faster one joins the
    training queue. The reference trial passes by definition.
    """
    assert context.bench is not None
    study = context.study
    live = _running_trial(study, trial)
    attempts = int(trial.user_attrs.get(ATTR_BENCH_ATTEMPTS, 0))
    if attempts >= context.bench.max_attempts:
        detail = f"bench failed {attempts} times: {trial.user_attrs.get(ATTR_ERROR, '')}"
        live.set_user_attr(ATTR_ERROR, detail)
        study.tell(trial.number, state=TrialState.FAIL)
        return ScreenOutcome(trial.number, STATUS_FAILED, detail)
    live.set_user_attr(ATTR_BENCH_ATTEMPTS, attempts + 1)
    name = context.bench.variant.variant_name(trial.number)
    try:
        package_dir = exporter(context, trial)
        description = f"{study.study_name} trial {trial.number}: {_describe_params(trial)}"
        measurement = runner.measure(name, package_dir, description)
    except BenchBusyError as exc:
        # Someone else's job on the bench is not this trial's failure.
        live.set_user_attr(ATTR_BENCH_ATTEMPTS, attempts)
        detail = f"{exc}; measuring again later"
        live.set_user_attr(ATTR_ERROR, detail)
        logger.warning("Trial %d: %s", trial.number, detail)
        return ScreenOutcome(trial.number, STATUS_SKIPPED, detail)
    except BenchJobError as exc:
        if exc.unbuildable:
            return reject_unbuildable(context, trial, exc)
        detail = f"{type(exc).__name__}: {exc}"
        live.set_user_attr(ATTR_ERROR, detail)
        logger.exception("Trial %d: bench measurement failed", trial.number)
        return ScreenOutcome(trial.number, STATUS_SKIPPED, detail)
    except (BenchError, Exception) as exc:  # noqa: BLE001 - keep screening the rest
        detail = f"{type(exc).__name__}: {exc}"
        live.set_user_attr(ATTR_ERROR, detail)
        logger.exception("Trial %d: bench measurement failed", trial.number)
        return ScreenOutcome(trial.number, STATUS_SKIPPED, detail)
    limit_ci = context.bench.max_ci95_ms
    if limit_ci is not None and measurement.ci95_ms > limit_ci:
        detail = (
            f"replicate spread +-{measurement.ci95_ms:.2f} ms exceeds {limit_ci:.2f} ms; "
            "the bench was probably contended, measuring again later"
        )
        live.set_user_attr(ATTR_ERROR, detail)
        logger.warning("Trial %d: %s", trial.number, detail)
        return ScreenOutcome(trial.number, STATUS_SKIPPED, detail, measurement.latency_ms)
    live.set_user_attr(ATTR_BENCH, measurement.to_dict())
    latency = measurement.latency_ms
    # A trial that anchors the latency bound passes by definition; every
    # other trial is judged against the anchor, be it a trial or a model.
    anchors = {c.reference for c in latency_constraints(context.constraints) if c.reference}
    is_anchor = trial.user_attrs.get(ATTR_REFERENCE) in anchors
    reference = latency if is_anchor else reference_latency_ms(study, context.constraints)
    limit = None
    for constraint in latency_constraints(context.constraints):
        limit = constraint.upper_bound(reference)
    if limit is not None:
        live.set_user_attr(ATTR_LATENCY_LIMIT, limit)
    metrics = trial.user_attrs.get(ATTR_METRICS)
    if metrics is not None:
        frozen = study._storage.get_trial(trial._trial_id)
        values = tell_measured(context, frozen, metrics)
        told = study._storage.get_trial(trial._trial_id)
        status = STATUS_MEASURED if trial_is_feasible(told) else STATUS_REJECTED
        detail = f"{latency:.2f} ms" + (f" (limit {limit:.2f})" if limit is not None else "")
        detail += f", told {values}"
        return ScreenOutcome(trial.number, status, detail, latency, limit)
    violations = constraint_values(
        context.constraints, {}, latency, reference, unknown_metric_violation=0.0
    )
    if any(value > 0.0 for value in violations):
        _set_attrs(
            live,
            {
                ATTR_CONSTRAINTS: violations,
                ATTR_PHASE: PHASE_REJECTED,
                ATTR_ERROR: f"latency {latency:.2f} ms exceeds the bound {limit:.2f} ms",
            },
        )
        study.tell(trial.number, state=TrialState.PRUNED)
        return ScreenOutcome(
            trial.number,
            STATUS_REJECTED,
            f"{latency:.2f} ms > {limit:.2f} ms",
            latency,
            limit,
        )
    live.set_user_attr(ATTR_PHASE, PHASE_SCREENED)
    detail = f"{latency:.2f} ms" + (f" <= {limit:.2f} ms" if limit is not None else "")
    return ScreenOutcome(trial.number, STATUS_SCREENED, detail, latency, limit)


def drain_waiting_trials(context: TuningContext) -> list[FrozenTrial]:
    """Ask the sampler for every queued trial so it carries its overrides.

    Reference trials are queued by ``tune init`` with fixed parameters; the
    sampler returns them in order before proposing anything of its own.

    Returns:
        The trials asked, as frozen views.
    """
    study = context.study
    asked: list[FrozenTrial] = []
    waiting = study.get_trials(deepcopy=False, states=(TrialState.WAITING,))
    for _ in waiting:
        result = ask_trial(context)
        if result is None:
            continue
        trial, _ = result
        asked.append(study._storage.get_trial(trial._trial_id))
    return asked


def reject_unbuildable(
    context: TuningContext, trial: FrozenTrial, error: BenchJobError
) -> ScreenOutcome:
    """Tell an architecture whose engines do not build on the target as infeasible.

    It cannot be deployed, so this is a verdict rather than a failure to
    retry: the latency constraint takes a large violation, an imported trial
    keeps its known accuracy, a sampled one is pruned. Either way the sampler
    learns to stay away from that region.
    """
    assert context.bench is not None
    study = context.study
    live = _running_trial(study, trial)
    reason = "engines do not build on the target: " + error.detail.splitlines()[0][:200]
    live.set_user_attr(ATTR_ERROR, reason)
    metrics = trial.user_attrs.get(ATTR_METRICS)
    violations = []
    for constraint in context.constraints:
        if constraint.source == "latency":
            violations.append(context.bench.unbuildable_violation_ms)
        elif metrics is not None and constraint.metric in metrics:
            violations.append(constraint.violation(float(metrics[constraint.metric])))
        else:
            violations.append(0.0)
    _set_attrs(live, {ATTR_CONSTRAINTS: violations, ATTR_PHASE: PHASE_REJECTED})
    latency_objective = any(objective.source == "latency" for objective in context.objectives)
    if metrics is not None and not latency_objective:
        values = objective_values(context.objectives, metrics, None)
        study.tell(trial.number, values)
        return ScreenOutcome(trial.number, STATUS_REJECTED, f"{reason}; told {values}")
    study.tell(trial.number, state=TrialState.PRUNED)
    return ScreenOutcome(trial.number, STATUS_REJECTED, reason)


def reference_model_specs(context: TuningContext) -> dict[str, dict[str, Any]]:
    """Return the configured ``tuning.reference_models``: name to config and overrides.

    A reference model is an architecture outside the search space, such as
    the deployed PTv3, whose bench latency anchors the latency bound. It is
    never trained by the study; it is exported with its initial weights and
    timed like a trial, and its measurement lives on the study.
    """
    specs = context.tuning.get("reference_models") or {}
    plain = OmegaConf.to_container(specs, resolve=True) if OmegaConf.is_config(specs) else specs
    assert isinstance(plain, Mapping)
    return {str(name): dict(spec) for name, spec in plain.items()}


def reference_model_variant(context: TuningContext, name: str) -> str:
    """Return the harness variant name of a reference model."""
    assert context.bench is not None
    return f"{context.bench.variant.prefix}-ref-{name.replace('_', '-')}"


def export_reference_model_package(
    context: TuningContext, name: str, spec: Mapping[str, Any]
) -> Path:
    """Export a reference model's package unless a complete one already exists."""
    from autoware_ml.tuning.export import export_package

    assert context.bench is not None
    package_dir = context.bench.packages_dir / reference_model_variant(context, name)
    contract = export_contract(context)
    if load_manifest(package_dir, contract) is not None:
        return package_dir
    if package_dir.exists():
        shutil.rmtree(package_dir)
    config_name = str(spec["config_name"])
    reference = ConfigReference.resolve(config_name, TASK_CONFIG_PREFIX)
    overrides = [*(str(item) for item in spec.get("overrides") or ()), *MEASURE_OVERRIDES]
    cfg = reference.compose(overrides)
    logger.info("Reference model %s: exporting package to %s (%s)", name, package_dir, contract)
    export_package(
        cfg,
        package_dir,
        config_name=config_name,
        params={"reference_model": name},
        overrides=overrides,
        contract=contract,
    )
    return package_dir


def export_reference_models(context: TuningContext) -> list[TrialOutcome]:
    """Export the package of every reference model that has no bench latency yet."""
    assert context.bench is not None
    outcomes: list[TrialOutcome] = []
    known = reference_model_latencies(context.study)
    for name, spec in reference_model_specs(context).items():
        if name in known:
            continue
        try:
            export_reference_model_package(context, name, spec)
        except Exception as exc:  # noqa: BLE001 - reported as an outcome
            logger.exception("Reference model %s: export failed", name)
            outcomes.append(
                TrialOutcome(-1, STATUS_FAILED, f"reference model {name}: export failed: {exc}")
            )
    return outcomes


def measure_reference_models(context: TuningContext, runner: BenchRunner) -> list[TrialOutcome]:
    """Time every configured reference model whose package exists and that has no latency yet."""
    assert context.bench is not None
    outcomes: list[TrialOutcome] = []
    known = reference_model_latencies(context.study)
    for name, spec in reference_model_specs(context).items():
        if name in known:
            continue
        variant = reference_model_variant(context, name)
        package_dir = context.bench.packages_dir / variant
        if load_manifest(package_dir, export_contract(context)) is None:
            outcomes.append(
                TrialOutcome(-1, STATUS_SKIPPED, f"reference model {name}: no package yet")
            )
            continue
        try:
            description = (
                f"{context.study.study_name} reference model {name}: {spec['config_name']}"
            )
            measurement = runner.measure(variant, package_dir, description)
        except BenchBusyError as exc:
            detail = f"reference model {name}: {exc}; measuring again later"
            logger.warning(detail)
            outcomes.append(TrialOutcome(-1, STATUS_SKIPPED, detail))
            continue
        except Exception as exc:  # noqa: BLE001 - reported as an outcome
            logger.exception("Reference model %s: bench measurement failed", name)
            outcomes.append(
                TrialOutcome(
                    -1, STATUS_FAILED, f"reference model {name}: {type(exc).__name__}: {exc}"
                )
            )
            continue
        limit_ci = context.bench.max_ci95_ms
        if limit_ci is not None and measurement.ci95_ms > limit_ci:
            detail = (
                f"reference model {name}: replicate spread +-{measurement.ci95_ms:.2f} ms exceeds "
                f"{limit_ci:.2f} ms; the bench was probably contended, measuring again later"
            )
            logger.warning(detail)
            outcomes.append(TrialOutcome(-1, STATUS_SKIPPED, detail))
            continue
        record = measurement.to_dict()
        record.update(
            {
                "config_name": str(spec["config_name"]),
                "overrides": [str(item) for item in spec.get("overrides") or ()],
                "package_dir": str(package_dir),
                "contract": export_contract(context),
            }
        )
        set_reference_model_latency(context.study, name, record)
        outcomes.append(
            TrialOutcome(
                -1,
                STATUS_MEASURED,
                f"reference model {name}: {measurement.latency_ms:.2f} ms "
                f"(+-{measurement.ci95_ms:.2f} ms over {measurement.replicates} replicates)",
            )
        )
    return outcomes


def pending_screen_trials(context: TuningContext) -> list[FrozenTrial]:
    """Running trials that have no bench latency yet: the reference first, then by number."""
    assert context.bench is not None
    pending = [
        trial
        for trial in context.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if trial.user_attrs.get(ATTR_PHASE) in (PHASE_IMPORTED, PHASE_ASKED, None)
        and ATTR_OVERRIDES in trial.user_attrs
        and trial.user_attrs.get(ATTR_BENCH) is None
    ]
    pending.sort(key=lambda trial: trial.number)
    reference = context.bench.reference
    pending.sort(key=lambda trial: 0 if trial.user_attrs.get(ATTR_REFERENCE) == reference else 1)
    return pending


def screened_queue_length(context: TuningContext) -> int:
    """Number of latency-feasible trials waiting for training."""
    return len(trials_in_phase(context.study, PHASE_SCREENED))


def in_flight_count(context: TuningContext) -> int:
    """Trials waiting for training plus trials asked but not yet timed."""
    return screened_queue_length(context) + len(pending_screen_trials(context))


def run_screen_phase(
    context: TuningContext,
    runner: BenchRunner,
    *,
    n_trials: int | None = None,
    queue_target: int | None = None,
    now: dt.datetime | None = None,
    ignore_window: bool = False,
    ignore_termination: bool = False,
    exporter: Callable[[TuningContext, FrozenTrial], Path] = export_trial_package,
    max_asks: int | None = None,
) -> list[TrialOutcome]:
    """Screen trials on the bench until the queue target or the window is reached.

    Each round has two halves. First, with the trainer paused (it shares the
    GPU), the phase exports what needs exporting: the reference models,
    trials asked earlier that still lack a package, and new proposals until
    the queue target is reached. Then, GPU free again, it times the reference
    models and every pending trial on the bench; the reference trial goes
    first because the latency bound may derive from it. Rounds repeat while
    rejections leave the queue short of its target.

    Args:
        context: The study context, with a bench configured.
        runner: Bench runner.
        n_trials: Cap on bench measurements in this call.
        queue_target: Queue length to reach; defaults to the policy's.
        now: Current time, injectable for tests.
        ignore_window: Screen outside the bench window.
        ignore_termination: Ask for new trials after the study was terminated.
        exporter: Produces a trial's package; injectable for tests.
        max_asks: Cap on sampler asks, bounding static rejections.

    Returns:
        One outcome per trial considered.
    """
    settings = context.bench
    if settings is None:
        raise ValueError("The screen phase needs a bench configured under tuning.bench.")
    study = context.study
    window = settings.window
    current = (
        now if now is not None else (window.now() if window else dt.datetime.now().astimezone())
    )
    if window is not None and not ignore_window and not window.contains(current):
        return [
            TrialOutcome(
                -1,
                STATUS_WINDOW_CLOSED,
                f"bench window {window.describe()} opens at {window.opens_at(current):%Y-%m-%d %H:%M}",
            )
        ]
    if find_reference_trial(study, settings.reference) is None:
        raise ValueError(
            f"Reference trial {settings.reference!r} is not in the study; run tune init or tune "
            "import first."
        )
    outcomes: list[TrialOutcome] = []
    measured = 0
    limit_measurements = n_trials if n_trials is not None else 10**9
    drain_waiting_trials(context)
    outcomes.extend(tell_known_trials(context))
    target = queue_target if queue_target is not None else settings.queue.target_at(window, current)
    terminated = study_terminated(study) is not None and not ignore_termination
    budget = training_budget_left(context)
    if budget is not None:
        target = min(target, budget)
    asks = 0
    ask_budget = max_asks if max_asks is not None else 10 * max(1, target)
    relative = [c for c in latency_constraints(context.constraints) if c.reference is not None]
    anchors = {c.reference for c in relative}

    def window_open() -> bool:
        if ignore_window or window is None:
            return True
        return window.contains(window.now() if now is None else current)

    def reference_known() -> bool:
        return not relative or reference_latency_ms(study, context.constraints) is not None

    failed_exports: set[int] = set()

    def export_quietly(trial: FrozenTrial) -> bool:
        try:
            exporter(context, trial)
        except Exception as exc:  # noqa: BLE001 - the trial stays pending for the next round
            logger.exception("Trial %d: export failed", trial.number)
            outcomes.append(TrialOutcome(trial.number, STATUS_SKIPPED, f"export failed: {exc}"))
            failed_exports.add(trial.number)
            return False
        return True

    while True:
        progressed = False
        needs_export = (
            any(
                name not in reference_model_latencies(study)
                for name in reference_model_specs(context)
            )
            or any(not trial_has_package(context, t) for t in pending_screen_trials(context))
            or (not terminated and in_flight_count(context) < target and asks < ask_budget)
        )
        if needs_export:
            try:
                with trainer_paused(context, settings.export_max_wait_s):
                    outcomes.extend(export_reference_models(context))
                    for trial in pending_screen_trials(context):
                        if not trial_has_package(context, trial):
                            progressed |= export_quietly(trial)
                    while (
                        not terminated and in_flight_count(context) < target and asks < ask_budget
                    ):
                        asks += 1
                        asked = ask_trial(context)
                        if asked is None:
                            outcomes.append(TrialOutcome(-1, STATUS_PRUNED, "static filter"))
                            continue
                        trial, _ = asked
                        progressed |= export_quietly(study._storage.get_trial(trial._trial_id))
            except TrainerBusyError as exc:
                outcomes.append(TrialOutcome(-1, STATUS_SKIPPED, f"exports postponed: {exc}"))
        outcomes.extend(measure_reference_models(context, runner))
        for trial in pending_screen_trials(context):
            if measured >= limit_measurements or not window_open():
                break
            if trial.number in failed_exports:
                continue
            if not reference_known() and trial.user_attrs.get(ATTR_REFERENCE) not in anchors:
                continue
            outcomes.append(screen_trial(context, trial, runner, exporter=exporter))
            measured += 1
            progressed = True
        if not reference_known():
            outcomes.append(
                TrialOutcome(-1, STATUS_SKIPPED, "the reference latency is still unknown")
            )
            return outcomes
        if terminated:
            outcomes.append(TrialOutcome(-1, STATUS_TERMINATED, "the study was terminated"))
            return outcomes
        if (
            screened_queue_length(context) >= target
            or measured >= limit_measurements
            or asks >= ask_budget
            or not window_open()
            or not progressed
        ):
            break
    if screened_queue_length(context) >= target and not any(
        o.status in (STATUS_SCREENED, STATUS_REJECTED) for o in outcomes
    ):
        outcomes.append(
            TrialOutcome(
                -1, STATUS_QUEUE_FULL, f"{screened_queue_length(context)} >= {target} waiting"
            )
        )
    return outcomes


# --------------------------------------------------------------------------
# Status


def format_status(context: TuningContext) -> str:
    """Render every trial of the study as a text table with the study's verdicts."""
    study = context.study
    trials = study.get_trials(deepcopy=False)
    header = [
        "#",
        "state",
        "phase",
        "ref",
        "params",
        "bench_ms",
        "limit_ms",
        *[objective.name for objective in context.objectives],
        *[
            constraint.metric
            for constraint in context.constraints
            if constraint.source != "latency"
        ],
        "feasible",
    ]
    rows = [header]
    for trial in sorted(trials, key=lambda item: item.number):
        values = trial.values or []
        metrics = trial.user_attrs.get(ATTR_METRICS) or {}
        bench = trial_bench_latency_ms(trial)
        limit = trial.user_attrs.get(ATTR_LATENCY_LIMIT)
        stored = trial.user_attrs.get(ATTR_CONSTRAINTS)
        feasible = "" if stored is None else ("yes" if trial_is_feasible(trial) else "no")
        rows.append(
            [
                str(trial.number),
                trial.state.name,
                str(trial.user_attrs.get(ATTR_PHASE, "")),
                str(trial.user_attrs.get(ATTR_REFERENCE, "")),
                _describe_params(trial),
                "" if bench is None else f"{bench:.2f}",
                "" if limit is None else f"{float(limit):.2f}",
                *[f"{value:.4g}" for value in values],
                *[""] * (len(context.objectives) - len(values)),
                *[
                    _short(metrics[constraint.metric]) if constraint.metric in metrics else ""
                    for constraint in context.constraints
                    if constraint.source != "latency"
                ],
                feasible,
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    phases = {}
    for trial in trials:
        phases[str(trial.user_attrs.get(ATTR_PHASE, trial.state.name.lower()))] = (
            phases.get(str(trial.user_attrs.get(ATTR_PHASE, trial.state.name.lower())), 0) + 1
        )
    lines.append("")
    lines.append(
        f"study {study.study_name}: {len(trials)} trials, {len(completed)} complete, phases "
        + ", ".join(f"{name}={count}" for name, count in sorted(phases.items()))
    )
    reference = reference_latency_ms(study, context.constraints)
    for constraint in latency_constraints(context.constraints):
        if constraint.relative:
            lines.append(
                f"latency bound: {constraint.reference} + {constraint.margin:g} ms = "
                + (
                    "unknown"
                    if reference is None
                    else f"{constraint.upper_bound(reference):.2f} ms"
                )
            )
        elif constraint.max is not None:
            lines.append(f"latency bound: {constraint.max:.2f} ms")
    for name, record in reference_model_latencies(study).items():
        lines.append(
            f"reference model {name}: {float(record['latency_ms']):.2f} ms "
            f"({record.get('config_name', '')})"
        )
    cap = context.max_trained_trials
    lines.append(
        f"trainings paid for: {count_trained_here(study)}"
        + ("" if cap is None else f" of {cap}")
        + f", screened and waiting: {screened_queue_length(context)}"
    )
    if len(context.objectives) == 1:
        best = best_feasible_trial(study, context.objectives[0]) if completed else None
        if best is None:
            lines.append("best feasible trial: none yet")
        else:
            lines.append(
                f"best feasible trial: #{best.number} {context.objectives[0].name}="
                f"{best.values[0]:.4g} ({_describe_params(best)})"
            )
    elif completed:
        lines.append(f"{len(study.best_trials)} on the Pareto front")
        for trial in study.best_trials:
            lines.append(
                f"  pareto #{trial.number}: "
                + ", ".join(
                    f"{objective.name}={value:.4g}"
                    for objective, value in zip(context.objectives, trial.values, strict=True)
                )
            )
    termination = study.user_attrs.get(STUDY_ATTR_TERMINATION)
    if termination:
        lines.append(
            f"termination check ({termination.get('checked_at', '')}): {termination.get('reason', '')}"
        )
    verdict = study_terminated(study)
    if verdict:
        lines.append(f"TERMINATED at {verdict.get('checked_at', '')}: {verdict.get('reason', '')}")
    return "\n".join(lines)


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
