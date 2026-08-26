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

"""Tests for the sample-at-a-time visualization preview pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from autoware_ml.tests.visualization.conftest import (
    CalibrationPreviewModel,
    DetectionPreviewModel,
    PreviewDataModule,
    RecordingBackend,
    SegmentationPreviewModel,
    VoxelizedSegmentationPreviewModel,
)
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    Boxes3DEvent,
    ImageEvent,
    PointCloud3DEvent,
    Points2DEvent,
    TextEvent,
)
from autoware_ml.visualization.preview import (
    SEGMENTATION_LABEL_KEYS,
    VisualizationPreviewConfig,
    _has_segmentation_sample,
    resolve_preview_device,
    run_visualization_preview,
)

_NOOP_PREVIEW = VisualizationPreviewConfig(
    split="test", session=VisualizationSessionConfig(backend="noop")
)


@pytest.fixture
def preview_session(
    monkeypatch: pytest.MonkeyPatch, recording_backend: RecordingBackend
) -> RecordingBackend:
    """Route every preview session to the recording backend."""
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(recording_backend)),
    )
    return recording_backend


def _detection_sample() -> dict[str, Any]:
    """Build one detection sample with ground truth and class names."""
    return {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
        "class_names": ["pedestrian", "car"],
    }


_DETECTION_COLLATION = {
    "points": "concat",
    "gt_boxes": "concat",
    "gt_labels": "concat",
    "class_names": "list",
}
_SEGMENTATION_COLLATION = {"points": "concat", "segment": "concat"}


def _segmentation_sample() -> dict[str, Any]:
    """Build one segmentation sample with per-point ground truth."""
    return {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
    }


def test_preview_logs_a_calibration_sample(
    preview_session: RecordingBackend, preview_calibration_data: CalibrationData
) -> None:
    sample = {
        "calibration_data": preview_calibration_data,
        "points": np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32),
        "img": np.zeros((720, 1280, 3), dtype=np.uint8),
        "fused_img": np.zeros((5, 720, 1280), dtype=np.float32),
        "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
        "img_path": "sample.png",
    }

    visualized = run_visualization_preview(
        CalibrationPreviewModel(),
        PreviewDataModule(
            [sample],
            {
                "calibration_data": "list",
                "points": "concat",
                "img": "stack",
                "fused_img": "stack",
                "gt_calibration_status": "list",
                "img_path": "list",
            },
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    assert preview_session.steps == [0]
    assert "calibration_status/camera/fused" in preview_session.paths_of(ImageEvent)
    assert "calibration_status/camera/image/projected_points" in preview_session.paths_of(
        Points2DEvent
    )


def test_preview_logs_a_segmentation_sample(preview_session: RecordingBackend) -> None:
    visualized = run_visualization_preview(
        SegmentationPreviewModel(),
        PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    point_paths = preview_session.paths_of(PointCloud3DEvent)
    assert "segmentation3d/prediction" in point_paths
    assert "segmentation3d/ground_truth" in point_paths


def test_preview_reconstructs_points_for_voxelized_segmentation(
    preview_session: RecordingBackend,
) -> None:
    """PTv3 drops raw points, so positions come from ``coord[inverse]``."""
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        VoxelizedSegmentationPreviewModel(),
        PreviewDataModule(
            [sample],
            {"coord": "concat", "inverse": "index_concat", "origin_segment": "concat"},
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    prediction = next(
        event
        for event in preview_session.events
        if isinstance(event, PointCloud3DEvent) and event.path == "segmentation3d/prediction"
    )
    assert prediction.positions.shape == (3, 3)


def test_preview_logs_transformed_voxelized_data_without_a_model(
    preview_session: RecordingBackend,
) -> None:
    """PTv3 carries positions in ``coord``, so ``--mode data`` must still route.

    This path has no predictions to fall back on, so requiring a ``points`` key
    made transformed-data preview unusable for every PTv3 segmentation config.
    """
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        None,
        PreviewDataModule(
            [sample],
            {"coord": "concat", "inverse": "index_concat", "origin_segment": "concat"},
        ),
        VisualizationPreviewConfig(
            mode="data", split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    assert preview_session.paths_of(PointCloud3DEvent) == ["transformed/segmentation3d/data"]
    logged = next(event for event in preview_session.events if isinstance(event, PointCloud3DEvent))
    assert logged.positions.shape == (3, 3)


def test_preview_logs_a_detection_sample(preview_session: RecordingBackend) -> None:
    visualized = run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule([_detection_sample()], _DETECTION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    assert preview_session.paths_of(Boxes3DEvent) == [
        "detection3d/prediction",
        "detection3d/ground_truth",
    ]


def test_preview_logs_transformed_data_without_a_model(
    preview_session: RecordingBackend,
) -> None:
    visualized = run_visualization_preview(
        None,
        PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
        VisualizationPreviewConfig(
            mode="data", split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    assert preview_session.paths_of(PointCloud3DEvent) == ["transformed/segmentation3d/data"]
    assert "transformed/segmentation3d/meta/sample" in preview_session.paths_of(TextEvent)


def test_preview_scrubs_multiple_samples_on_the_shared_timeline(
    preview_session: RecordingBackend,
) -> None:
    samples = [_detection_sample() for _ in range(3)]

    visualized = run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule(samples, _DETECTION_COLLATION),
        VisualizationPreviewConfig(
            split="test",
            max_samples=3,
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 3
    assert preview_session.steps == [0, 1, 2]


def test_preview_starts_at_the_requested_sample_index(
    preview_session: RecordingBackend,
) -> None:
    samples = [_detection_sample() for _ in range(4)]

    run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule(samples, _DETECTION_COLLATION),
        VisualizationPreviewConfig(
            split="test",
            sample_index=2,
            max_samples=2,
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert preview_session.steps == [2, 3]


def test_preview_waits_for_the_viewer_after_logging(
    preview_session: RecordingBackend,
) -> None:
    """The preview must hand control to the backend instead of duck-typing it."""
    run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule([_detection_sample()], _DETECTION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert preview_session.waited is True


def test_preview_requires_a_model_for_prediction_mode() -> None:
    with pytest.raises(ValueError, match="Model must be provided"):
        run_visualization_preview(
            None,
            PreviewDataModule(
                [{"points": np.zeros((1, 4), dtype=np.float32)}], {"points": "concat"}
            ),
            VisualizationPreviewConfig(mode="predictions"),
        )


def test_preview_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unknown visualization mode: heatmap"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(mode="heatmap"),  # type: ignore[arg-type]
        )


def test_preview_rejects_a_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="max_samples must be greater than zero"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(max_samples=0),
        )


def test_preview_rejects_an_out_of_range_sample_index() -> None:
    with pytest.raises(IndexError, match="out of range"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(
                mode="data",
                sample_index=5,
                session=VisualizationSessionConfig(backend="noop"),
            ),
        )


def test_preview_reports_observed_keys_when_no_task_matches(
    preview_session: RecordingBackend,
) -> None:
    """An unroutable sample must name what it saw instead of guessing a task."""
    with pytest.raises(ValueError, match=r"Could not infer a visualization task.*points"):
        run_visualization_preview(
            None,
            PreviewDataModule(
                [{"points": np.zeros((1, 4), dtype=np.float32)}], {"points": "concat"}
            ),
            VisualizationPreviewConfig(
                mode="data", split="test", session=VisualizationSessionConfig(backend="noop")
            ),
        )


def test_preview_rejects_a_sample_matching_two_tasks(
    preview_session: RecordingBackend,
) -> None:
    """Multi-task samples must fail loudly rather than pick whichever check runs first."""
    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
        "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
    }

    with pytest.raises(ValueError, match="Ambiguous visualization task"):
        run_visualization_preview(
            None,
            PreviewDataModule(
                [sample],
                {
                    "points": "concat",
                    "segment": "concat",
                    "gt_boxes": "concat",
                    "gt_labels": "concat",
                },
            ),
            VisualizationPreviewConfig(
                mode="data", split="test", session=VisualizationSessionConfig(backend="noop")
            ),
        )


@pytest.mark.parametrize("position_key", ["points", "coord"])
@pytest.mark.parametrize("label_key", SEGMENTATION_LABEL_KEYS)
def test_segmentation_matcher_accepts_every_position_and_label_key(
    position_key: str, label_key: str
) -> None:
    """Both point-source keys pair with every supported per-point label key."""
    batch = {
        position_key: np.zeros((2, 3), dtype=np.float32),
        label_key: np.zeros((2,), dtype=np.int64),
    }

    assert _has_segmentation_sample(batch) is True


@pytest.mark.parametrize(
    "batch",
    [
        {"coord": np.zeros((2, 3), dtype=np.float32)},
        {"segment": np.zeros((2,), dtype=np.int64)},
        {},
    ],
    ids=["positions-only", "labels-only", "empty"],
)
def test_segmentation_matcher_needs_both_positions_and_labels(
    batch: dict[str, Any],
) -> None:
    assert _has_segmentation_sample(batch) is False


def test_resolve_preview_device_honors_explicit_devices() -> None:
    assert resolve_preview_device("cpu").type == "cpu"
    assert resolve_preview_device("auto").type in {"cpu", "cuda"}
