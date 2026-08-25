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

"""Tests for the 3D semantic-segmentation visualization adapter."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    PointCloud3DEvent,
    ScalarEvent,
)
from autoware_ml.visualization.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)

_TWO_POINTS = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def test_build_segmentation3d_events_logs_prediction_and_ground_truth() -> None:
    events = build_segmentation3d_events(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        pred_probs=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        gt_labels=np.array([1, 1], dtype=np.int64),
        class_names=["road", "car"],
    )

    point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
    point_paths = [event.path for event in point_events]
    assert "segmentation3d/prediction" in point_paths
    assert "segmentation3d/ground_truth" in point_paths
    assert any(isinstance(event, AnnotationContextEvent) for event in events)

    prediction = next(e for e in point_events if e.path == "segmentation3d/prediction")
    assert prediction.class_ids is not None
    assert prediction.labels is None


def test_build_segmentation3d_events_logs_entropy_cloud_and_confidence() -> None:
    events = build_segmentation3d_events(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        pred_probs=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    )

    point_paths = [event.path for event in events if isinstance(event, PointCloud3DEvent)]
    assert "segmentation3d/entropy" in point_paths

    metrics = {event.path: event.value for event in events if isinstance(event, ScalarEvent)}
    assert metrics["segmentation3d/metrics/num_points"] == 2.0
    assert metrics["segmentation3d/metrics/mean_confidence"] == pytest.approx(0.9)


def test_build_segmentation3d_events_emits_point_labels_on_request() -> None:
    events = build_segmentation3d_events(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        class_names=["road", "car"],
        point_labels=True,
    )

    prediction = next(
        event
        for event in events
        if isinstance(event, PointCloud3DEvent) and event.path == "segmentation3d/prediction"
    )
    assert prediction.labels == ["road", "car"]


def test_build_segmentation3d_events_legend_covers_every_declared_class() -> None:
    """A frame containing two classes must still name all classes in the legend."""
    events = build_segmentation3d_events(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        class_names=["road", "car", "building", "vegetation"],
    )

    context = next(event for event in events if isinstance(event, AnnotationContextEvent))
    assert [annotation.label for annotation in context.annotations] == [
        "road",
        "car",
        "building",
        "vegetation",
    ]


def test_build_segmentation3d_events_rejects_misaligned_predictions() -> None:
    with pytest.raises(ValueError, match="predicted labels must have the same length"):
        build_segmentation3d_events(_TWO_POINTS, np.array([0], dtype=np.int64))


def test_build_segmentation3d_events_rejects_misaligned_ground_truth() -> None:
    with pytest.raises(ValueError, match="ground-truth labels must have the same length"):
        build_segmentation3d_events(
            _TWO_POINTS,
            np.array([0, 1], dtype=np.int64),
            gt_labels=np.array([0], dtype=np.int64),
        )


def test_build_segmentation3d_events_rejects_misaligned_probabilities() -> None:
    with pytest.raises(ValueError, match=r"pred_probs must have shape \(N, C\)"):
        build_segmentation3d_events(
            _TWO_POINTS,
            np.array([0, 1], dtype=np.int64),
            pred_probs=np.zeros((1, 2), dtype=np.float32),
        )


def test_build_segmentation3d_data_events_logs_single_data_cloud() -> None:
    events = build_segmentation3d_data_events(
        _TWO_POINTS,
        np.array([0, 1], dtype=np.int64),
        class_names=["road", "car"],
    )

    point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
    assert [event.path for event in point_events] == ["segmentation3d/data"]
    assert any(isinstance(event, AnnotationContextEvent) for event in events)


def test_build_segmentation3d_data_events_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="labels must have the same length as points"):
        build_segmentation3d_data_events(_TWO_POINTS, np.array([0], dtype=np.int64))
