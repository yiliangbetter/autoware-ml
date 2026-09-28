"""Intersection-over-union metric."""

from __future__ import annotations

import torch

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter
from autoware_ml.metrics.segmentation3d.confusion import (
    ConfusionState,
    class_name_token,
    macro,
)


class IoU(Metric[ConfusionState]):
    """Macro mean IoU in every stage. Test adds the frequency-weighted IoU and the
    per-class IoU for each class with ground-truth support.

    ``detail_stages`` widens that breakdown to other stages, so a tuning study
    can read a per-class value at validation without every training run
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
            detail_stages: Stage names that also report the frequency-weighted
                IoU and the per-class IoU.
        """
        super().__init__(stages=stages, filter=filter)
        self.detail_stages: frozenset[EvalStage] = frozenset(
            EvalStage(stage) for stage in detail_stages
        )

    def evaluate(self, state: ConfusionState, stage: EvalStage) -> dict[str, float]:
        """Compute IoU metrics from a confusion state.

        Args:
            state: Segmentation confusion state.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of IoU metric names to scalar values.
        """
        union = state.predicted + state.actual - state.true_positive
        zeros = torch.zeros_like(state.true_positive)
        iou = torch.where(union > 0, state.true_positive / union, zeros)

        report = {"mIoU": macro(iou, state.has_support)}
        if stage not in self.detail_stages:
            return report

        if bool(state.has_support.any()):
            report["fwIoU"] = float((state.frequency * iou)[state.has_support].sum().item())
        else:
            report["fwIoU"] = float("nan")
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            report[f"iou_{class_name_token(class_index, state.class_names)}"] = float(
                iou[class_index].item()
            )
        return report
