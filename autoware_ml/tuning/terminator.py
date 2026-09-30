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

"""Automatic termination of a single-objective study.

Optuna's terminator stops a study once the room left for improvement, an
upper bound on the regret estimated by a Gaussian process over the completed
trials, drops below the statistical error of the objective. Its built-in
``should_terminate`` looks at every completed trial; a constrained study
must judge only the feasible ones, otherwise a fast-but-useless model would
count as the incumbent, so the evaluators are driven here directly on the
feasible set and the verdict is recorded on the study.
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from autoware_ml.tuning.study import (
    STUDY_ATTR_TERMINATED,
    STUDY_ATTR_TERMINATION,
    Objective,
    feasible_complete_trials,
)

logger = logging.getLogger(__name__)

IMPROVEMENT_EVALUATORS = ("regret_bound", "emmr", "stagnation")
ERROR_EVALUATORS = ("static", "cross_validation", "median")


@dataclass(frozen=True)
class TerminatorSettings:
    """How and when the termination check runs.

    Attributes:
        improvement: ``regret_bound`` (Gaussian-process regret bound),
            ``emmr`` (expected minimum model regret) or ``stagnation``
            (trials since the best value last improved).
        error: ``static`` (a constant), ``cross_validation`` (needs per-trial
            cross-validation scores) or ``median`` (paired with ``emmr``).
        constant: The static error, in objective units. Improvements below
            it are not worth another trial.
        min_n_trials: Feasible completed trials before termination is considered.
        top_trials_ratio: Fraction of the best trials the regret bound fits.
        max_stagnation_trials: Patience of the stagnation evaluator.
        feasible_only: Judge only trials that satisfy every constraint.
        seed: Random seed of the evaluators.
    """

    improvement: str = "regret_bound"
    error: str = "static"
    constant: float = 0.0
    min_n_trials: int = 20
    top_trials_ratio: float = 0.5
    max_stagnation_trials: int = 30
    feasible_only: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the evaluator names and the combination."""
        if self.improvement not in IMPROVEMENT_EVALUATORS:
            raise ValueError(f"improvement must be one of {IMPROVEMENT_EVALUATORS}.")
        if self.error not in ERROR_EVALUATORS:
            raise ValueError(f"error must be one of {ERROR_EVALUATORS}.")
        if self.error == "median" and self.improvement != "emmr":
            raise ValueError("The median error evaluator pairs with the emmr improvement only.")
        if self.min_n_trials <= 0:
            raise ValueError("min_n_trials must be positive.")

    @classmethod
    def from_config(cls, cfg: DictConfig | Mapping[str, Any] | None) -> TerminatorSettings | None:
        """Parse the ``tuning.terminator`` block; ``None`` or ``enabled: false`` disables it."""
        if cfg is None:
            return None
        plain = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
        assert isinstance(plain, Mapping)
        if not plain.get("enabled", True):
            return None
        return cls(
            improvement=str(plain.get("improvement", "regret_bound")),
            error=str(plain.get("error", "static")),
            constant=float(plain.get("constant", 0.0)),
            min_n_trials=int(plain.get("min_n_trials", 20)),
            top_trials_ratio=float(plain.get("top_trials_ratio", 0.5)),
            max_stagnation_trials=int(plain.get("max_stagnation_trials", 30)),
            feasible_only=bool(plain.get("feasible_only", True)),
            seed=None if plain.get("seed") is None else int(plain["seed"]),
        )


@dataclass(frozen=True)
class TerminationReport:
    """Outcome of one termination check.

    Attributes:
        checked_at: ISO timestamp of the check.
        n_trials: Trials the evaluators saw.
        improvement: Room left for improvement, in objective units (for
            ``stagnation``: trials of patience left).
        error: Error the improvement is compared against.
        should_terminate: Whether the study should stop asking for trials.
        reason: Human-readable summary.
    """

    checked_at: str
    n_trials: int
    improvement: float | None
    error: float | None
    should_terminate: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


class StudyTerminator:
    """Optuna's termination criterion applied to the feasible trials of a study."""

    def __init__(self, settings: TerminatorSettings) -> None:
        """Build the evaluators the settings name."""
        self.settings = settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            from optuna import terminator

            if settings.improvement == "regret_bound":
                self._improvement = terminator.RegretBoundEvaluator(
                    top_trials_ratio=settings.top_trials_ratio,
                    min_n_trials=settings.min_n_trials,
                    seed=settings.seed,
                )
            elif settings.improvement == "emmr":
                self._improvement = terminator.EMMREvaluator(
                    min_n_trials=settings.min_n_trials, seed=settings.seed
                )
            else:
                self._improvement = terminator.BestValueStagnationEvaluator(
                    max_stagnation_trials=settings.max_stagnation_trials
                )
            if settings.error == "static":
                self._error = terminator.StaticErrorEvaluator(constant=settings.constant)
            elif settings.error == "cross_validation":
                self._error = terminator.CrossValidationErrorEvaluator()
            else:
                self._error = terminator.MedianErrorEvaluator(self._improvement)

    def trials_to_judge(self, study: optuna.Study) -> list[FrozenTrial]:
        """Return the completed trials the criterion is evaluated on."""
        if self.settings.feasible_only:
            return feasible_complete_trials(study)
        return sorted(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
            key=lambda trial: trial.number,
        )

    def check(self, study: optuna.Study, objective: Objective) -> TerminationReport:
        """Evaluate the criterion without recording it."""
        trials = self.trials_to_judge(study)
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        if len(trials) < self.settings.min_n_trials:
            return TerminationReport(
                now,
                len(trials),
                None,
                None,
                False,
                f"{len(trials)} of the {self.settings.min_n_trials} trials needed before "
                "termination is considered",
            )
        direction = (
            StudyDirection.MAXIMIZE
            if objective.direction == "maximize"
            else StudyDirection.MINIMIZE
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            improvement = float(self._improvement.evaluate(trials, direction))
            error = float(self._error.evaluate(trials, direction))
        should_terminate = improvement < error
        if self.settings.improvement == "stagnation":
            reason = (
                f"best value unchanged for {self.settings.max_stagnation_trials - improvement:.0f} "
                f"of {self.settings.max_stagnation_trials} allowed trials"
            )
        else:
            reason = (
                f"room for improvement {improvement:.4g} {'<' if should_terminate else '>='} "
                f"error {error:.4g} over {len(trials)} trials"
            )
        return TerminationReport(now, len(trials), improvement, error, should_terminate, reason)

    def check_and_record(self, study: optuna.Study, objective: Objective) -> TerminationReport:
        """Evaluate the criterion and store the verdict on the study.

        The first positive verdict is kept under ``terminated`` so later
        phases refuse to ask for or train new trials; every check is stored
        under ``termination`` for the status output.
        """
        report = self.check(study, objective)
        study.set_user_attr(STUDY_ATTR_TERMINATION, report.to_dict())
        if report.should_terminate and not study.user_attrs.get(STUDY_ATTR_TERMINATED):
            study.set_user_attr(STUDY_ATTR_TERMINATED, report.to_dict())
            logger.info("Study %s terminated: %s", study.study_name, report.reason)
        return report


def study_terminated(study: optuna.Study) -> Mapping[str, Any] | None:
    """Return the recorded termination verdict, if the study was terminated."""
    return study.user_attrs.get(STUDY_ATTR_TERMINATED)
