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

"""CLI-facing functions for the batched tuning commands."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from autoware_ml.tuning.bench import BenchSettings, SshBenchRunner
from autoware_ml.tuning.importing import import_trials
from autoware_ml.tuning.phases import (
    TuningContext,
    export_trial_package,
    format_status,
    run_measure_phase,
    run_screen_phase,
    run_train_phase,
)
from autoware_ml.tuning.study import (
    SQLITE_PREFIX,
    build_sampler,
    enqueue_reference_trials,
    find_reference_trial,
    load_constraints,
    load_objectives,
    open_study,
    resolve_storage_url,
)
from autoware_ml.tuning.terminator import StudyTerminator, TerminatorSettings
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
    bench = BenchSettings.from_config(tuning.get("bench"))
    terminator_settings = TerminatorSettings.from_config(tuning.get("terminator"))
    if terminator_settings is not None and len(objectives) != 1:
        raise ValueError("The termination criterion applies to single-objective studies only.")
    if bench is not None and find_reference_trial is not None:
        references = tuning.get("reference_trials") or {}
        if bench.reference not in references:
            raise ValueError(
                f"bench.reference {bench.reference!r} is not one of the configured "
                f"reference_trials {sorted(references)}."
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
        bench=bench,
        terminator=None if terminator_settings is None else StudyTerminator(terminator_settings),
    )


def _print_outcomes(outcomes) -> None:
    for outcome in outcomes:
        number = "-" if outcome.trial_number < 0 else str(outcome.trial_number)
        print(f"trial {number}: {outcome.status} {outcome.detail}".rstrip())


def init_study(config_name: str, extra_args: Sequence[str] = ()) -> None:
    """Create the study and queue its reference trials."""
    context = load_context(config_name, extra_args)
    references = context.tuning.get("reference_trials")
    queued = enqueue_reference_trials(context.study, context.search_space, references)
    print(
        f"Study '{context.study.study_name}' ready at {resolve_storage_url(str(context.tuning.storage))}"
    )
    print(f"Objectives: {', '.join(f'{o.direction} {o.name}' for o in context.objectives)}")
    for constraint in context.constraints:
        if constraint.relative:
            bound = f"<= latency({constraint.reference}) + {constraint.margin:g} ms"
        else:
            bound = " and ".join(
                part
                for part in (
                    f">= {constraint.min:g}" if constraint.min is not None else "",
                    f"<= {constraint.max:g}" if constraint.max is not None else "",
                )
                if part
            )
        print(f"Constraint: {constraint.metric} {bound}")
    if context.bench is not None:
        window = context.bench.window
        print(
            f"Bench: {context.bench.ssh_target()}, window "
            + (window.describe() if window else "always open")
            + f", {context.bench.replicates} replicates of {context.bench.metric}"
        )
    if context.terminator is not None:
        settings = context.terminator.settings
        print(
            f"Terminator: {settings.improvement} vs {settings.error} error"
            + (f" {settings.constant:g}" if settings.error == "static" else "")
            + f", after {settings.min_n_trials} feasible trials"
        )
    if queued:
        print(f"Queued reference trials: {', '.join(queued)}")
    else:
        print("No new reference trials queued.")
    print(f"Trials so far: {len(context.study.get_trials(deepcopy=False))}")


def import_study(
    config_name: str,
    extra_args: Sequence[str] = (),
    from_storage: Sequence[str] = (),
    from_study: Sequence[str] = (),
) -> None:
    """Import the measured trials of other studies."""
    context = load_context(config_name, extra_args)
    if not from_storage or not from_study:
        raise ValueError("--from-storage and --from-study are required.")
    outcomes = import_trials(
        context.study,
        context.search_space,
        source_storage=from_storage,
        source_study=from_study,
        objective_metric=context.objectives[0].name,
        references=context.tuning.get("reference_trials"),
    )
    for outcome in outcomes:
        target = "" if outcome.trial_number is None else f" -> trial {outcome.trial_number}"
        print(f"source {outcome.sources}: {outcome.status}{target} {outcome.detail}".rstrip())
    imported = sum(1 for outcome in outcomes if outcome.status == "imported")
    attached = sum(1 for outcome in outcomes if outcome.status == "attached")
    print(f"Imported {imported} cell(s) and attached {attached} from {', '.join(from_study)}.")


def screen_trials(
    config_name: str,
    extra_args: Sequence[str] = (),
    n_trials: int | None = None,
    queue_target: int | None = None,
    ignore_window: bool = False,
    ignore_termination: bool = False,
) -> None:
    """Run the screen phase on the bench."""
    context = load_context(config_name, extra_args)
    if context.bench is None:
        raise ValueError("No bench is configured under tuning.bench.")
    runner = SshBenchRunner(context.bench, context.log_dir)
    outcomes = run_screen_phase(
        context,
        runner,
        n_trials=n_trials,
        queue_target=queue_target,
        ignore_window=ignore_window,
        ignore_termination=ignore_termination,
    )
    _print_outcomes(outcomes)


def export_trial(
    config_name: str,
    extra_args: Sequence[str] = (),
    trial: int | None = None,
    reference: str | None = None,
) -> None:
    """Export one trial's package without measuring it."""
    context = load_context(config_name, extra_args)
    if context.bench is None:
        raise ValueError("No bench is configured under tuning.bench.")
    if reference is not None:
        frozen = find_reference_trial(context.study, reference)
        if frozen is None:
            raise ValueError(f"No trial carries reference {reference!r}; run tune init first.")
    elif trial is not None:
        frozen = context.study._storage.get_trial(
            context.study._storage.get_trial_id_from_study_id_trial_number(
                context.study._study_id, trial
            )
        )
    else:
        raise ValueError("Pass --trial or --reference.")
    package_dir = export_trial_package(context, frozen)
    print(f"trial {frozen.number}: package at {package_dir}")


def train_trials(
    config_name: str,
    extra_args: Sequence[str] = (),
    n_trials: int = 1,
    gpus: str | None = None,
    ignore_termination: bool = False,
) -> None:
    """Run the train phase."""
    context = load_context(config_name, extra_args)
    devices = [item.strip() for item in gpus.split(",") if item.strip()] if gpus else None
    outcomes = run_train_phase(
        context, n_trials=n_trials, gpus=devices, ignore_termination=ignore_termination
    )
    _print_outcomes(outcomes)
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
    _print_outcomes(outcomes)


def show_status(config_name: str, extra_args: Sequence[str] = ()) -> None:
    """Print the study's trials and its verdicts."""
    context = load_context(config_name, extra_args)
    print(format_status(context))
