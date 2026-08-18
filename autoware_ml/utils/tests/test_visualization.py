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

"""Tests for the visualization scaffolding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.models.base import BaseModel
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus
from autoware_ml.visualization.backends import create_visualization_backend
from autoware_ml.visualization.calibration_status import build_calibration_status_events
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
    normalize_detection_predictions,
)
from autoware_ml.visualization.preview import VisualizationPreviewConfig, run_visualization_preview
from autoware_ml.visualization.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)
from autoware_ml.visualization.session import VisualizationSession
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ImageEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextEvent,
)
from autoware_ml.visualization.rerun_backend import RerunVisualizationBackend


class _DummyBackend:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.events: list[Any] = []

    def set_step(self, step: int) -> None:
        self.steps.append(step)

    def log_event(self, event: Any) -> None:
        self.events.append(event)

    def log_events(self, events: Any) -> None:
        self.events.extend(events)


class _PreviewDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_data_info(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class _PreviewDataModule(DataModule):
    def __init__(self, samples: list[dict[str, Any]], collation_map: dict[str, str]) -> None:
        super().__init__(collation_map=collation_map)
        self.samples = samples

    def _create_dataset(self, split: str, dataset_transforms: Any = None) -> Dataset:
        del split, dataset_transforms
        return _PreviewDataset(self.samples)


class _PreviewModelBase(BaseModel):
    def forward(self, **kwargs: Any) -> Any:
        del kwargs
        return None

    def compute_metrics(
        self, batch_inputs_dict: dict[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict, outputs
        return {"loss": torch.zeros(())}


class _CalibrationPreviewModel(_PreviewModelBase):
    def predict_step(self, batch_inputs_dict: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_inputs_dict, batch_idx
        return torch.tensor([[0.1, 0.9]], dtype=torch.float32)


class _SegmentationPreviewModel(_PreviewModelBase):
    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        points = batch_inputs_dict["points"]
        if isinstance(points, list):
            points = points[0]
        num_points = points.shape[0]
        pred_labels = torch.arange(num_points, dtype=torch.long) % 2
        pred_probs = torch.nn.functional.one_hot(pred_labels, num_classes=2).float()
        return {"pred_labels": pred_labels, "pred_probs": pred_probs}


class _VoxelizedSegmentationPreviewModel(_PreviewModelBase):
    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        inverse = batch_inputs_dict["inverse"].long()
        pred_labels = torch.arange(inverse.shape[0], device=inverse.device, dtype=torch.long) % 2
        pred_probs = torch.nn.functional.one_hot(pred_labels, num_classes=2).float()
        return {"pred_labels": pred_labels, "pred_probs": pred_probs}


class _DetectionPreviewModel(_PreviewModelBase):
    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> list[dict[str, torch.Tensor]]:
        del batch_inputs_dict, batch_idx
        return [
            {
                "bboxes_3d": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]],
                    dtype=torch.float32,
                ),
                "scores_3d": torch.tensor([0.9], dtype=torch.float32),
                "labels_3d": torch.tensor([1], dtype=torch.long),
            }
        ]


def _make_calibration_data() -> CalibrationData:
    return CalibrationData(
        camera_matrix=np.array(
            [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.zeros((5,), dtype=np.float32),
        lidar_to_camera_transformation=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_calibration_status_events_includes_frames_and_status() -> None:
    calibration_data = _make_calibration_data()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    events = build_calibration_status_events(
        calibration_data,
        points=np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32),
        image=image,
        fused_image=np.zeros((720, 1280, 5), dtype=np.float32),
        gt_status=CalibrationStatus.CALIBRATED.value,
        pred_status=CalibrationStatus.MISCALIBRATED.value,
        pred_score=0.9,
        sample_name="sample-1",
    )

    assert any(event.path == "calibration_status/camera" for event in events)
    assert any(event.path == "calibration_status/lidar/points" for event in events)
    assert any(event.path == "calibration_status/status/gt_label" for event in events)
    assert any(event.path == "calibration_status/status/pred_label" for event in events)
    assert any(
        isinstance(event, Points2DEvent)
        and event.path == "calibration_status/camera/image/projected_points"
        for event in events
    )
    assert any(
        isinstance(event, ScalarEvent) and event.path == "calibration_status/status/pred_score"
        for event in events
    )
    assert any(
        isinstance(event, TextEvent) and event.path == "calibration_status/status/summary"
        for event in events
    )


def test_build_segmentation3d_events_logs_prediction_and_ground_truth() -> None:
    points = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    pred_labels = np.array([0, 1], dtype=np.int64)
    pred_probs = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    gt_labels = np.array([1, 1], dtype=np.int64)

    events = build_segmentation3d_events(
        points,
        pred_labels,
        pred_probs=pred_probs,
        gt_labels=gt_labels,
        class_names=["road", "car"],
    )

    point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
    point_paths = [event.path for event in point_events]
    assert "segmentation3d/prediction" in point_paths
    assert "segmentation3d/ground_truth" in point_paths
    assert any(isinstance(event, AnnotationContextEvent) for event in events)
    pred_event = next(e for e in point_events if e.path == "segmentation3d/prediction")
    assert pred_event.class_ids is not None
    assert pred_event.labels is None


def test_build_segmentation3d_data_events_logs_single_data_cloud() -> None:
    events = build_segmentation3d_data_events(
        np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        class_names=["road", "car"],
    )

    point_events = [event for event in events if isinstance(event, PointCloud3DEvent)]
    assert [event.path for event in point_events] == ["segmentation3d/data"]
    assert any(isinstance(event, AnnotationContextEvent) for event in events)


def test_normalize_detection_predictions_accepts_both_prediction_key_sets() -> None:
    centerpoint_predictions = normalize_detection_predictions(
        {
            "bboxes_3d": np.ones((2, 9), dtype=np.float32),
            "scores_3d": np.array([0.8, 0.7], dtype=np.float32),
            "labels_3d": np.array([0, 1], dtype=np.int64),
        }
    )
    assert centerpoint_predictions["boxes"].shape == (2, 9)

    generic_predictions = normalize_detection_predictions(
        {
            "bboxes": np.ones((1, 7), dtype=np.float32),
            "scores": np.array([0.5], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
        }
    )
    assert generic_predictions["boxes"].shape == (1, 7)


def test_build_detection3d_events_logs_boxes_and_points() -> None:
    events = build_detection3d_events(
        {
            "bboxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        },
        points=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        gt_boxes=np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        gt_labels=np.array([1], dtype=np.int64),
        class_names=["pedestrian", "car"],
    )

    assert any(isinstance(event, PointCloud3DEvent) for event in events)
    box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
    assert [event.path for event in box_events] == [
        "detection3d/prediction",
        "detection3d/ground_truth",
    ]
    assert any(isinstance(event, AnnotationContextEvent) for event in events)
    assert box_events[0].class_ids is not None


def test_build_detection3d_data_events_logs_ground_truth_only() -> None:
    events = build_detection3d_data_events(
        points=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        gt_boxes=np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        gt_labels=np.array([1], dtype=np.int64),
        class_names=["pedestrian", "car"],
    )

    box_events = [event for event in events if isinstance(event, Boxes3DEvent)]
    assert [event.path for event in box_events] == ["detection3d/ground_truth"]
    assert box_events[0].class_ids is not None


def test_visualization_session_delegates_to_backend() -> None:
    backend = _DummyBackend()
    session = VisualizationSession(backend)

    session.set_step(7)
    session.log_detection3d(
        {
            "bboxes": np.zeros((0, 7), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }
    )

    assert backend.steps == [7]
    assert backend.events


def test_create_visualization_backend_noop() -> None:
    backend = create_visualization_backend(VisualizationSessionConfig(backend="noop"))
    backend.set_step(1)
    backend.log_events([])


def test_rerun_backend_logs_supported_events(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {"init": None, "serve_web": None, "logs": [], "steps": []}

    class _FakeRR:
        class TransformRelation:
            ChildFromParent = "ChildFromParent"

        @staticmethod
        def init(application_id: str, **kwargs: Any) -> None:
            calls["init"] = (application_id, kwargs)

        @staticmethod
        def serve_web(**kwargs: Any) -> None:
            calls["serve_web"] = kwargs

        @staticmethod
        def set_time_sequence(timeline: str, step: int) -> None:
            calls["steps"].append((timeline, step))

        @staticmethod
        def log(path: str, payload: Any, **kwargs: Any) -> None:
            calls["logs"].append((path, payload, kwargs))

        @staticmethod
        def Image(image: Any) -> tuple[str, Any]:
            return ("Image", image)

        @staticmethod
        def AnnotationContext(annotations: Any) -> tuple[str, Any]:
            return ("AnnotationContext", annotations)

        @staticmethod
        def ClassDescription(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("ClassDescription", kwargs)

        @staticmethod
        def AnnotationInfo(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("AnnotationInfo", kwargs)

        @staticmethod
        def Points3D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points3D", args, kwargs)

        @staticmethod
        def Points2D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points2D", args, kwargs)

        @staticmethod
        def Boxes3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Boxes3D", kwargs)

        @staticmethod
        def Transform3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Transform3D", kwargs)

        @staticmethod
        def Pinhole(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Pinhole", kwargs)

        @staticmethod
        def Scalar(value: Any) -> tuple[str, Any]:
            return ("Scalar", value)

        @staticmethod
        def TextLog(text: str, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
            return ("TextLog", text, kwargs)

    monkeypatch.setattr(
        "autoware_ml.visualization.rerun_backend._load_rerun_module",
        lambda: _FakeRR,
    )

    backend = RerunVisualizationBackend(
        VisualizationSessionConfig(web_port=9091, grpc_port=9877, wait=False)
    )
    backend.set_step(3)
    backend.log_events(
        build_detection3d_events(
            {
                "bboxes": np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
                "scores": np.array([0.8], dtype=np.float32),
                "labels": np.array([0], dtype=np.int64),
            }
        )
    )

    assert calls["init"][0] == "autoware-ml"
    assert calls["init"][1]["spawn"] is False
    assert calls["serve_web"] == {
        "open_browser": False,
        "web_port": 9091,
        "grpc_port": 9877,
        "server_memory_limit": "25%",
    }
    assert backend.web_url == (
        "http://localhost:9091?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9877%2Fproxy"
    )
    assert calls["steps"] == [("frame", 3)]
    assert calls["logs"]


def test_run_visualization_preview_logs_calibration_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend()
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(backend)),
    )

    sample = {
        "calibration_data": _make_calibration_data(),
        "points": np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32),
        "img": np.zeros((720, 1280, 3), dtype=np.uint8),
        "fused_img": np.zeros((5, 720, 1280), dtype=np.float32),
        "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
        "img_path": "sample.png",
    }

    visualized = run_visualization_preview(
        _CalibrationPreviewModel(),
        _PreviewDataModule(
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
        VisualizationPreviewConfig(
            split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    assert backend.steps == [0]
    assert any(
        isinstance(event, ImageEvent) and event.path == "calibration_status/camera/fused"
        for event in backend.events
    )
    assert any(
        isinstance(event, Points2DEvent)
        and event.path == "calibration_status/camera/image/projected_points"
        for event in backend.events
    )


def test_run_visualization_preview_logs_segmentation_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend()
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(backend)),
    )

    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
        "lidar_path": "sample.bin",
    }

    visualized = run_visualization_preview(
        _SegmentationPreviewModel(),
        _PreviewDataModule(
            [sample],
            {
                "points": "concat",
                "segment": "concat",
                "lidar_path": "list",
            },
        ),
        VisualizationPreviewConfig(
            split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    point_events = [event for event in backend.events if isinstance(event, PointCloud3DEvent)]
    point_paths = [event.path for event in point_events]
    assert "segmentation3d/prediction" in point_paths
    assert "segmentation3d/ground_truth" in point_paths


def test_run_visualization_preview_logs_voxelized_segmentation_without_raw_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend()
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(backend)),
    )

    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        _VoxelizedSegmentationPreviewModel(),
        _PreviewDataModule(
            [sample],
            {
                "coord": "concat",
                "inverse": "index_concat",
                "origin_segment": "concat",
            },
        ),
        VisualizationPreviewConfig(
            split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    point_events = [event for event in backend.events if isinstance(event, PointCloud3DEvent)]
    point_paths = [event.path for event in point_events]
    assert "segmentation3d/prediction" in point_paths
    assert "segmentation3d/ground_truth" in point_paths
    pred_event = next(e for e in point_events if e.path == "segmentation3d/prediction")
    assert pred_event.positions.shape == (3, 3)


def test_run_visualization_preview_logs_detection_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend()
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(backend)),
    )

    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
        "class_names": ["pedestrian", "car"],
    }

    visualized = run_visualization_preview(
        _DetectionPreviewModel(),
        _PreviewDataModule(
            [sample],
            {
                "points": "concat",
                "gt_boxes": "concat",
                "gt_labels": "concat",
                "class_names": "list",
            },
        ),
        VisualizationPreviewConfig(
            split="test", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert visualized == 1
    box_events = [event for event in backend.events if isinstance(event, Boxes3DEvent)]
    assert [event.path for event in box_events] == [
        "detection3d/prediction",
        "detection3d/ground_truth",
    ]


def test_run_visualization_preview_logs_transformed_segmentation_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend()
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(backend)),
    )

    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        None,
        _PreviewDataModule(
            [sample],
            {
                "points": "concat",
                "segment": "concat",
            },
        ),
        VisualizationPreviewConfig(
            mode="data",
            split="test",
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 1
    point_events = [event for event in backend.events if isinstance(event, PointCloud3DEvent)]
    assert [event.path for event in point_events] == ["transformed/segmentation3d/data"]
    assert any(
        isinstance(event, TextEvent) and event.path == "transformed/segmentation3d/meta/sample"
        for event in backend.events
    )


def test_run_visualization_preview_requires_model_for_prediction_mode() -> None:
    with pytest.raises(ValueError, match="Model must be provided"):
        run_visualization_preview(
            None,
            _PreviewDataModule(
                [{"points": np.zeros((1, 4), dtype=np.float32)}],
                {"points": "concat"},
            ),
            VisualizationPreviewConfig(mode="predictions"),
        )
