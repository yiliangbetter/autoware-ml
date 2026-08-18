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

"""3D semantic-segmentation visualization adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.visualization.colors import (
    build_label_palette,
    labels_to_colors,
    scalar_to_heatmap_colors,
)
from autoware_ml.visualization.common import (
    as_numpy,
    build_class_annotation_context,
    build_sample_metadata_events,
    ensure_xyz,
    format_class_label,
)
from autoware_ml.visualization.events import (
    PointCloud3DEvent,
    ScalarEvent,
    VisualizationEvent,
)


def build_segmentation3d_data_events(
    points: Any,
    labels: Any,
    *,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = "segmentation3d",
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
) -> list[VisualizationEvent]:
    """Build backend-neutral events for transformed segmentation data only."""
    point_positions = ensure_xyz(points)
    labels_np = as_numpy(labels, np.int64).reshape(-1)
    if labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("labels must have the same length as points")

    palette_size = int(labels_np.max()) + 1 if labels_np.size > 0 else 0
    palette = build_label_palette(palette_size)
    radii = np.full((point_positions.shape[0],), point_radius, dtype=np.float32)
    label_text = _build_point_labels(labels_np, class_names, point_labels)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
    annotation_context = build_class_annotation_context(root_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)
    events.append(
        PointCloud3DEvent(
            path=f"{root_path}/data",
            positions=point_positions,
            colors=labels_to_colors(labels_np, palette, ignore_index=ignore_index),
            labels=label_text,
            radii=radii,
            class_ids=labels_np,
        )
    )
    events.append(ScalarEvent(f"{root_path}/metrics/num_points", float(point_positions.shape[0])))
    return events


def build_segmentation3d_events(
    points: Any,
    pred_labels: Any,
    *,
    pred_probs: Any | None = None,
    gt_labels: Any | None = None,
    class_names: Sequence[str] | None = None,
    ignore_index: int | None = None,
    root_path: str = "segmentation3d",
    point_radius: float = 0.04,
    point_labels: bool = False,
    sample_name: str | None = None,
) -> list[VisualizationEvent]:
    """Build backend-neutral segmentation visualization events for one sample."""
    point_positions = ensure_xyz(points)
    pred_labels_np = as_numpy(pred_labels, np.int64).reshape(-1)
    if pred_labels_np.shape[0] != point_positions.shape[0]:
        raise ValueError("predicted labels must have the same length as points")

    palette_size = int(pred_labels_np.max()) + 1 if pred_labels_np.size > 0 else 0
    if gt_labels is not None:
        gt_labels_np = as_numpy(gt_labels, np.int64).reshape(-1)
        if gt_labels_np.shape[0] != point_positions.shape[0]:
            raise ValueError("ground-truth labels must have the same length as points")
        valid_gt = gt_labels_np[gt_labels_np >= 0]
        if valid_gt.size > 0:
            palette_size = max(palette_size, int(valid_gt.max()) + 1)
    else:
        gt_labels_np = None

    palette = build_label_palette(palette_size)
    radii = np.full((point_positions.shape[0],), point_radius, dtype=np.float32)
    pred_label_text = _build_point_labels(pred_labels_np, class_names, point_labels)
    events: list[VisualizationEvent] = build_sample_metadata_events(root_path, sample_name)
    annotation_context = build_class_annotation_context(root_path, palette, class_names)
    if annotation_context is not None:
        events.insert(0, annotation_context)
    events.append(
        PointCloud3DEvent(
            path=f"{root_path}/prediction",
            positions=point_positions,
            colors=labels_to_colors(pred_labels_np, palette, ignore_index=ignore_index),
            labels=pred_label_text,
            radii=radii,
            class_ids=pred_labels_np,
        )
    )

    if gt_labels_np is not None:
        gt_label_text = _build_point_labels(gt_labels_np, class_names, point_labels)
        events.append(
            PointCloud3DEvent(
                path=f"{root_path}/ground_truth",
                positions=point_positions,
                colors=labels_to_colors(gt_labels_np, palette, ignore_index=ignore_index),
                labels=gt_label_text,
                radii=radii,
                class_ids=gt_labels_np,
            )
        )

    events.append(ScalarEvent(f"{root_path}/metrics/num_points", float(pred_labels_np.shape[0])))
    if pred_probs is not None:
        pred_probs_np = as_numpy(pred_probs, np.float32)
        if pred_probs_np.ndim != 2 or pred_probs_np.shape[0] != pred_labels_np.shape[0]:
            raise ValueError("pred_probs must have shape (N, C) aligned with points")
        events.append(
            ScalarEvent(
                path=f"{root_path}/metrics/mean_confidence",
                value=float(pred_probs_np.max(axis=1).mean()),
            )
        )
        num_classes = pred_probs_np.shape[1]
        if num_classes > 1:
            entropy = -(pred_probs_np * np.log(pred_probs_np + 1e-8)).sum(axis=1)
            entropy_norm = (entropy / np.log(num_classes)).astype(np.float32)
            events.append(
                PointCloud3DEvent(
                    path=f"{root_path}/entropy",
                    positions=point_positions,
                    colors=scalar_to_heatmap_colors(entropy_norm),
                    radii=radii,
                )
            )

    return events


def _build_point_labels(
    labels: np.ndarray,
    class_names: Sequence[str] | None,
    enabled: bool,
) -> list[str] | None:
    """Build optional per-point label text."""
    if not enabled:
        return None
    return [format_class_label(int(label), class_names) for label in labels]
