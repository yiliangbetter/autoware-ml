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

"""CLI-facing functions for the two-phase tuning commands."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from autoware_ml.tuning.phases import (
    TuningContext,
    format_status,
    run_measure_phase,
    run_train_phase,
)
from autoware_ml.tuning.study import (
    SQLITE_PREFIX,
    build_sampler,
    enqueue_reference_trials,
    load_constraints,
    load_objectives,
    open_study,
    resolve_storage_url,
)
from autoware_ml.utils.cli.helpers import adjust_argv
from autoware_ml.utils.mlflow_helpers import should_enable_logger
from autoware_ml.utils.runtime import ConfigReference

logger = logging.getLogger(__name__)

TASK_CONFIG_PREFIX = "tasks"


def resolve_log_dir(tuning: DictConfig) -> Path:
    """Return the per-trial log directory, next to a SQLite study by default."""
    configured = tuning.get("log_dir")
    if configured is not None:
        return Path(str(configured)).expanduser().resolve()
    storage = resolve_storage_url(str(tuning.storage))
    if storage.startswith(SQLITE_PREFIX):
        return Path(storage.removeprefix(SQLITE_PREFIX)).parent / str(tuning.study_name)
    return (Path.cwd() / "mlruns" / "optuna" / str(tuning.study_name)).resolve()


def load_context(config_name: str, extra_args: Sequence[str] = ()) -> TuningContext:
    """Compose the tuning config and open its study.

    Args:
        config_name: Tuning config name or YAML path.
        extra_args: Hydra overrides from the command line.

    Returns:
        The context the phases operate on.

    Raises:
        ValueError: If the config has no ``tuning`` block or no MLflow logger.
    """
    overrides = adjust_argv(extra_args)
    reference = ConfigReference.resolve(config_name, TASK_CONFIG_PREFIX)
    cfg = reference.compose(overrides)
    if "tuning" not in cfg:
        raise ValueError(
            f"Config '{config_name}' has no 'tuning' block; point --config-name at a tuning config."
        )
    if not should_enable_logger(cfg):
        raise ValueError("Tuning needs the MLflow logger: trials are tracked by their runs.")
    tuning = cfg.tuning
    search_space = hydra.utils.instantiate(tuning.search_space)
    objectives = load_objectives(OmegaConf.to_container(tuning.objectives, resolve=True))
    constraints = load_constraints(
        OmegaConf.to_container(tuning.get("constraints") or [], resolve=True)
    )
    sampler = build_sampler(tuning.get("sampler"), len(constraints))
    study = open_study(str(tuning.study_name), str(tuning.storage), objectives, sampler)
    return TuningContext(
        config=reference,
        cfg=cfg,
        extra_overrides=tuple(overrides),
        search_space=search_space,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        study=study,
        log_dir=resolve_log_dir(tuning),
    )


def init_study(config_name: str, extra_args: Sequence[str] = ()) -> None:
    """Create the study and queue its reference trials."""
    context = load_context(config_name, extra_args)
    references = context.tuning.get("reference_trials")
    queued = enqueue_reference_trials(context.study, context.search_space, references)
    print(
        f"Study '{context.study.study_name}' ready at {resolve_storage_url(str(context.tuning.storage))}"
    )
    print(f"Objectives: {', '.join(f'{o.direction} {o.name}' for o in context.objectives)}")
    if queued:
        print(f"Queued reference trials: {', '.join(queued)}")
    else:
        print("No new reference trials queued.")
    print(f"Trials so far: {len(context.study.get_trials(deepcopy=False))}")


def train_trials(
    config_name: str,
    extra_args: Sequence[str] = (),
    n_trials: int = 1,
    gpus: str | None = None,
) -> None:
    """Run the train phase."""
    context = load_context(config_name, extra_args)
    devices = [item.strip() for item in gpus.split(",") if item.strip()] if gpus else None
    outcomes = run_train_phase(context, n_trials=n_trials, gpus=devices)
    for outcome in outcomes:
        print(f"trial {outcome.trial_number}: {outcome.status} {outcome.detail}".rstrip())
    print(f"Logs: {context.log_dir}")


def measure_trials(
    config_name: str,
    extra_args: Sequence[str] = (),
    frames: int | None = None,
    max_trials: int | None = None,
) -> None:
    """Run the measure phase."""
    context = load_context(config_name, extra_args)
    outcomes = run_measure_phase(
        context, max_trials=max_trials, latency_overrides={"frames": frames}
    )
    if not outcomes:
        print("No trained trials waiting for measurement.")
    for outcome in outcomes:
        print(f"trial {outcome.trial_number}: {outcome.status} {outcome.detail}".rstrip())


def show_status(config_name: str, extra_args: Sequence[str] = ()) -> None:
    """Print the study's trials and Pareto front."""
    context = load_context(config_name, extra_args)
    print(format_status(context))
