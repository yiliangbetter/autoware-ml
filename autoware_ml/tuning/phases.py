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

"""The two phases of a batched study: train trials, then measure them.

The train phase asks the sampler for configurations and launches one
training subprocess per trial, each in its own pre-created MLflow run so
the trial knows its run and checkpoint before training starts. The measure
phase re-validates each best checkpoint, benchmarks its latency, and only
then tells the sampler the full objective vector.
"""

from __future__ import annotations

import logging
import os
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

from autoware_ml.tuning.latency import LatencyReport, benchmark_latency, forward_inputs
from autoware_ml.tuning.search_space import LitePTSearchSpace, TrialProposal
from autoware_ml.tuning.study import (
    ATTR_CHECKPOINT,
    ATTR_CONSTRAINTS,
    ATTR_ERROR,
    ATTR_HYDRA_DIR,
    ATTR_LATENCY,
    ATTR_MEASURE_ATTEMPTS,
    ATTR_METRICS,
    ATTR_OVERRIDES,
    ATTR_PHASE,
    ATTR_REFERENCE,
    ATTR_RUN_ID,
    ATTR_STATIC,
    ATTR_TRAIN_LOG,
    PHASE_ASKED,
    PHASE_MEASURED,
    PHASE_TRAINED,
    Constraint,
    Objective,
    constraint_values,
    objective_values,
    trials_in_phase,
)
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
# Measurement runs one trial on one device; the study's own parallelism is
# across trials, never inside one.
MEASURE_OVERRIDES = ("trainer.devices=1",)
BENCHMARK_OVERRIDES = (
    "datamodule.val_dataloader_cfg.batch_size=1",
    "datamodule.val_dataloader_cfg.persistent_workers=false",
)
STATIC_ENCODER_PARAMS = "encoder_params"

STATUS_TRAINED = "trained"
STATUS_MEASURED = "measured"
STATUS_FAILED = "failed"
STATUS_PRUNED = "pruned"
STATUS_SKIPPED = "skipped"


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
        log_dir: Directory receiving per-trial training logs.
    """

    config: ConfigReference
    cfg: DictConfig
    extra_overrides: tuple[str, ...]
    search_space: LitePTSearchSpace
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    study: optuna.Study
    log_dir: Path

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

    def train_overrides(self) -> list[str]:
        """Fixed overrides the config applies to every trial's training."""
        return [str(value) for value in self.tuning.get("train_overrides") or []]

    def trial_overrides(self, trial_overrides: Sequence[str], *extra: str) -> list[str]:
        """Order the override layers: trial, then config, then command line."""
        return [*trial_overrides, *self.train_overrides(), *self.extra_overrides, *extra]


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


def prepare_trial_launch(
    context: TuningContext,
    gpu: str | None = None,
    extra_train_overrides: Sequence[str] = (),
) -> TrainLaunch | None:
    """Ask the sampler for a trial and prepare its training run.

    A trial rejected by the static filter is told as pruned and ``None`` is
    returned, so the caller can ask again without spending a training slot.
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
    overrides = context.trial_overrides(proposal.overrides, *extra_train_overrides)
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


def finish_trial_launch(
    context: TuningContext, launch: TrainLaunch, exit_code: int
) -> TrialOutcome:
    """Record a finished training subprocess on its trial."""
    trial = launch.trial
    if exit_code != 0:
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


def run_train_phase(
    context: TuningContext,
    *,
    n_trials: int,
    gpus: Sequence[str] | None = None,
    extra_train_overrides: Sequence[str] = (),
    launcher: Launcher = launch_subprocess,
    poll_interval: float = 5.0,
    max_asks: int | None = None,
) -> list[TrialOutcome]:
    """Train ``n_trials`` new trials, one subprocess per device slot.

    Args:
        context: The study context.
        n_trials: Number of trainings to launch.
        gpus: Device ids, one concurrent training each; ``None`` runs one
            training at a time on whatever is visible.
        extra_train_overrides: Overrides appended to every launched training.
        launcher: Starts a prepared launch; injectable for tests.
        poll_interval: Seconds between completion polls.
        max_asks: Cap on sampler asks, bounding static rejections.

    Returns:
        One outcome per trial that was asked.
    """
    if n_trials <= 0:
        return []
    slots: deque[str | None] = deque(gpus) if gpus else deque([None])
    if gpus and len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate device ids in {list(gpus)}.")
    ask_budget = max_asks if max_asks is not None else 10 * n_trials
    running: list[tuple[TrainLaunch, TrialProcess]] = []
    outcomes: list[TrialOutcome] = []
    launched = 0
    asked = 0
    while True:
        while slots and launched < n_trials and asked < ask_budget:
            asked += 1
            gpu = slots.popleft()
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
            running.append((launch, launcher(launch)))
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
    return outcomes


@dataclass(frozen=True)
class TrialEvaluation:
    """Accuracy and latency of one trained trial."""

    metrics: dict[str, float]
    latency: LatencyReport


Evaluator = Callable[[TuningContext, FrozenTrial, Mapping[str, Any]], TrialEvaluation]


def evaluate_trial(
    context: TuningContext, trial: FrozenTrial, latency_settings: Mapping[str, Any]
) -> TrialEvaluation:
    """Re-validate a trial's best checkpoint and benchmark its latency.

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

    benchmark_cfg = context.config.compose([*overrides, *BENCHMARK_OVERRIDES])
    benchmark_datamodule = hydra.utils.instantiate(benchmark_cfg.datamodule)
    benchmark_datamodule.setup("validate")
    model.to(device).eval()

    def forward(batch: Mapping[str, Any]) -> Any:
        prepared = model.on_after_batch_transfer(dict(batch), 0)
        return model(**forward_inputs(model, prepared))

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
    return TrialEvaluation(metrics=metrics, latency=report)


def encoder_stage_counter(model: Any) -> Callable[[Mapping[str, Any]], Sequence[int]] | None:
    """Return a per-stage point counter for PTv3-style encoders, else ``None``."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    from autoware_ml.models.segmentation3d.encoders.ptv3 import collect_encoder_stage_points

    def count(batch: Mapping[str, Any]) -> Sequence[int]:
        prepared = model.on_after_batch_transfer(dict(batch), 0)
        inputs = forward_inputs(model, prepared)
        deepest = encoder(
            {key: inputs[key] for key in ("coord", "feat", "grid_coord", "offset") if key in inputs}
        )
        return [int(point.feat.shape[0]) for point in collect_encoder_stage_points(deepest)]

    return count


def record_measurement(
    context: TuningContext, trial: FrozenTrial, evaluation: TrialEvaluation
) -> list[float]:
    """Store the evaluation on the trial and tell the sampler its objectives.

    Returns:
        The objective values told.
    """
    values = objective_values(context.objectives, evaluation.metrics, evaluation.latency.mean_ms)
    violations = constraint_values(context.constraints, evaluation.metrics)
    live = _running_trial(context.study, trial)
    _set_attrs(
        live,
        {
            ATTR_METRICS: evaluation.metrics,
            ATTR_LATENCY: evaluation.latency.to_dict(),
            ATTR_CONSTRAINTS: violations,
            ATTR_PHASE: PHASE_MEASURED,
        },
    )
    context.study.tell(trial.number, values)
    return values


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
        report = evaluation.latency
        client.log_metric(run_id, "tune/latency_mean_ms", report.mean_ms)
        client.log_metric(run_id, "tune/latency_p95_ms", report.p95_ms)
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
    up to ``max_attempts`` after which it fails.

    Args:
        context: The study context.
        evaluator: Produces the evaluation; injectable for tests.
        max_trials: Cap on trials measured in this call.
        max_attempts: Evaluation attempts before a trial is failed; defaults
            to the config's ``measure.max_attempts`` or 3.
        latency_overrides: Benchmark settings overriding the config's.

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
    return outcomes


def format_status(context: TuningContext) -> str:
    """Render every trial of the study as a text table."""
    study = context.study
    trials = study.get_trials(deepcopy=False)
    header = [
        "#",
        "state",
        "phase",
        "ref",
        "params",
        *[objective.name for objective in context.objectives],
    ]
    rows = [header]
    for trial in sorted(trials, key=lambda item: item.number):
        params = " ".join(f"{key}={_short(value)}" for key, value in sorted(trial.params.items()))
        values = trial.values or []
        rows.append(
            [
                str(trial.number),
                trial.state.name,
                str(trial.user_attrs.get(ATTR_PHASE, "")),
                str(trial.user_attrs.get(ATTR_REFERENCE, "")),
                params,
                *[f"{value:.4g}" for value in values],
                *[""] * (len(context.objectives) - len(values)),
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    lines.append("")
    lines.append(
        f"study {study.study_name}: {len(trials)} trials, {len(completed)} complete, "
        f"{len(study.best_trials) if completed else 0} on the Pareto front"
    )
    for trial in study.best_trials if completed else []:
        lines.append(
            f"  pareto #{trial.number}: "
            + ", ".join(
                f"{objective.name}={value:.4g}"
                for objective, value in zip(context.objectives, trial.values, strict=True)
            )
        )
    return "\n".join(lines)


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
