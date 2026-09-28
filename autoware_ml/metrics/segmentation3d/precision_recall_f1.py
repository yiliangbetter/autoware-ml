"""Precision, recall, and F1 metric."""

from __future__ import annotations

import torch

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter
from autoware_ml.metrics.segmentation3d.confusion import (
    ConfusionState,
    class_name_token,
    macro,
)


class PrecisionRecallF1(Metric[ConfusionState]):
    """Macro mean recall in every stage so validation has a recall headline. Test
    adds macro precision and F1 and the per-class breakdown.

    ``detail_stages`` widens that breakdown to other stages, so a tuning study
    can read a per-class recall at validation without every training run
    logging it.
    """

    def __init__(
        self,
        stages: tuple[str, ...] | list[str] = ("val", "test"),
        filter: MetricFilter | None = None,
        detail_stages: tuple[str, ...] | list[str] = ("test",),
    ) -> None:
        """Store the stages this metric runs in and where it breaks out classes.

        Args:
            stages: Stage names this metric reports for.
            filter: Optional selection axis, ``None`` for the whole scene.
            detail_stages: Stage names that also report macro precision and F1
                and the per-class recall, precision, and F1.
        """
        super().__init__(stages=stages, filter=filter)
        self.detail_stages: frozenset[EvalStage] = frozenset(
            EvalStage(stage) for stage in detail_stages
        )

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> dict[str, float]:
        """Compute precision, recall, and F1 metrics from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of precision/recall/F1 metric names to scalar values.
        """
        zeros = torch.zeros_like(state.true_positive)
        recall = torch.where(state.actual > 0, state.true_positive / state.actual, zeros)

        report = {"mRecall": macro(recall, state.has_support)}
        if stage not in self.detail_stages:
            return report

        precision = torch.where(state.predicted > 0, state.true_positive / state.predicted, zeros)
        f1_denominator = state.predicted + state.actual  # equals 2*TP + FP + FN
        f1 = torch.where(f1_denominator > 0, 2.0 * state.true_positive / f1_denominator, zeros)

        report["mPrecision"] = macro(precision, state.has_support)
        report["mF1"] = macro(f1, state.has_support)
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            name = class_name_token(class_index, state.class_names)
            report[f"recall_{name}"] = float(recall[class_index].item())
            report[f"precision_{name}"] = float(precision[class_index].item())
            report[f"f1_{name}"] = float(f1[class_index].item())
        return report
