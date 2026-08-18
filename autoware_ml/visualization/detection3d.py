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

"""3D detection visualization adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from autoware_ml.visualization.colors import build_label_palette, labels_to_colors
from autoware_ml.visualization.common import (
    as_numpy,
    build_class_annotation_context,
    build_sample_metadata_events,
    ensure_xyz,
    format_class_label,
)
from autoware_ml.visualization.events import (
    Boxes3DEvent,
    PointCloud3DEvent,
    ScalarEvent,
    VisualizationEvent,
)


def build_detection3d_data_events(
    *,
    points: Any | None = None,
    gt_boxes: Any,
    gt_labels: Any,
    class_names: Sequence[str] | None = None,
    root_path: str = "detection3d",
    point_radius: float = 0.04,
    sample_name: str | None = None,
) -> list[VisualizationEvent]:
    """Build backend-neutral 3D detection events for transformed data only."""
    gt_boxes_np = as_numpy(gt_boxes, np.float32)
    gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1)
    if gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] < 7:
        raise ValueError(f"ground-truth boxes must have shape (N, >=7), got {gt_boxes_np.shape}")
    if gt_boxes_np.shape[0] != gt_labels_np.shape[0]:
        raise ValueError("ground-truth boxes and labels must have the same length")

    palette_size = int(gt_labels_np.max()) + 1 if gt_labels_np.size > 0 else 0
    palette = build_label_palette(palette_size)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
    annotation_context = build_class_annotation_context(root_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)

    if points is not None:
        point_positions = ensure_xyz(points)
        events.append(
            PointCloud3DEvent(
                path=f"{root_path}/points",
                positions=point_positions,
                radii=np.full((point_positions.shape[0],), point_radius, dtype=np.float32),
            )
        )

    gt_label_text = [format_class_label(int(label), class_names) for label in gt_labels_np]
    events.append(
        Boxes3DEvent(
            path=f"{root_path}/ground_truth",
            centers=gt_boxes_np[:, :3],
            sizes=gt_boxes_np[:, 3:6],
            yaws=gt_boxes_np[:, 6],
            colors=labels_to_colors(gt_labels_np, palette),
            labels=gt_label_text,
            class_ids=gt_labels_np,
        )
    )
    events.append(
        ScalarEvent(
            path=f"{root_path}/metrics/num_ground_truth",
            value=float(gt_boxes_np.shape[0]),
        )
    )

    return events


def normalize_detection_predictions(predictions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Normalize decoded 3D predictions into one shared visualization contract."""
    if {"bboxes_3d", "scores_3d", "labels_3d"} <= predictions.keys():
        boxes = as_numpy(predictions["bboxes_3d"], np.float32)
        scores = as_numpy(predictions["scores_3d"], np.float32).reshape(-1)
        labels = as_numpy(predictions["labels_3d"], np.int64).reshape(-1)
    elif {"bboxes", "scores", "labels"} <= predictions.keys():
        boxes = as_numpy(predictions["bboxes"], np.float32)
        scores = as_numpy(predictions["scores"], np.float32).reshape(-1)
        labels = as_numpy(predictions["labels"], np.int64).reshape(-1)
    else:
        raise KeyError(
            "Expected decoded predictions with either "
            "('bboxes_3d', 'scores_3d', 'labels_3d') or ('bboxes', 'scores', 'labels')."
        )

    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"decoded boxes must have shape (N, >=7), got {boxes.shape}")
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
        raise ValueError("decoded boxes, scores, and labels must have the same length")
    return {"boxes": boxes, "scores": scores, "labels": labels}


def build_detection3d_events(
    predictions: Mapping[str, Any],
    *,
    points: Any | None = None,
    gt_boxes: Any | None = None,
    gt_labels: Any | None = None,
    class_names: Sequence[str] | None = None,
    root_path: str = "detection3d",
    point_radius: float = 0.04,
    sample_name: str | None = None,
) -> list[VisualizationEvent]:
    """Build backend-neutral 3D detection visualization events for one sample."""
    normalized_predictions = normalize_detection_predictions(predictions)
    pred_boxes = normalized_predictions["boxes"]
    pred_scores = normalized_predictions["scores"]
    pred_labels = normalized_predictions["labels"]

    palette_size = int(pred_labels.max()) + 1 if pred_labels.size > 0 else 0
    if gt_labels is not None:
        gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1)
        if gt_labels_np.size > 0:
            palette_size = max(palette_size, int(gt_labels_np.max()) + 1)
    else:
        gt_labels_np = None

    palette = build_label_palette(palette_size)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
    annotation_context = build_class_annotation_context(root_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)

    if points is not None:
        point_positions = ensure_xyz(points)
        events.append(
            PointCloud3DEvent(
                path=f"{root_path}/points",
                positions=point_positions,
                radii=np.full((point_positions.shape[0],), point_radius, dtype=np.float32),
            )
        )

    pred_colors = labels_to_colors(pred_labels, palette) if pred_labels.size > 0 else None
    pred_label_text = [
        format_class_label(int(label), class_names, float(score))
        for label, score in zip(pred_labels, pred_scores, strict=False)
    ]
    events.append(
        Boxes3DEvent(
            path=f"{root_path}/prediction",
            centers=pred_boxes[:, :3],
            sizes=pred_boxes[:, 3:6],
            yaws=pred_boxes[:, 6],
            colors=pred_colors,
            labels=pred_label_text,
            class_ids=pred_labels,
        )
    )
    events.append(
        ScalarEvent(
            path=f"{root_path}/metrics/num_predictions",
            value=float(pred_boxes.shape[0]),
        )
    )
    if pred_scores.size > 0:
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/mean_score",
                value=float(pred_scores.mean()),
            )
        )

    if gt_boxes is not None and gt_labels_np is not None:
        gt_boxes_np = as_numpy(gt_boxes, np.float32)
        if gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] < 7:
            raise ValueError(
                f"ground-truth boxes must have shape (N, >=7), got {gt_boxes_np.shape}"
            )
        if gt_boxes_np.shape[0] != gt_labels_np.shape[0]:
            raise ValueError("ground-truth boxes and labels must have the same length")
        gt_label_text = [format_class_label(int(label), class_names) for label in gt_labels_np]
        events.append(
            Boxes3DEvent(
                path=f"{root_path}/ground_truth",
                centers=gt_boxes_np[:, :3],
                sizes=gt_boxes_np[:, 3:6],
                yaws=gt_boxes_np[:, 6],
                colors=labels_to_colors(gt_labels_np, palette),
                labels=gt_label_text,
                class_ids=gt_labels_np,
            )
        )
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/num_ground_truth",
                value=float(gt_boxes_np.shape[0]),
            )
        )

    return events
