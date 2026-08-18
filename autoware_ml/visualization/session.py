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

"""High-level visualization session facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoware_ml.utils.calibration import CalibrationData
from autoware_ml.visualization.backends import create_visualization_backend
from autoware_ml.visualization.calibration_status import build_calibration_status_events
from autoware_ml.visualization.cameras import build_camera_events
from autoware_ml.visualization.contracts import VisualizationBackend, VisualizationSessionConfig
from autoware_ml.visualization.detection3d import (
    build_detection3d_data_events,
    build_detection3d_events,
)
from autoware_ml.visualization.segmentation3d import (
    build_segmentation3d_data_events,
    build_segmentation3d_events,
)


class VisualizationSession:
    """Wrap one backend and expose task-oriented logging helpers."""

    def __init__(self, backend: VisualizationBackend) -> None:
        """Initialize the visualization session from a backend instance."""
        self.backend = backend

    @classmethod
    def from_config(cls, config: VisualizationSessionConfig) -> VisualizationSession:
        """Construct a session from configuration."""
        return cls(create_visualization_backend(config))

    def set_step(self, step: int) -> None:
        """Advance the visualization timeline."""
        self.backend.set_step(step)

    def log_calibration_status(
        self,
        calibration_data: CalibrationData,
        *,
        points: Any | None = None,
        image: Any | None = None,
        fused_image: Any | None = None,
        gt_status: int | None = None,
        pred_status: int | None = None,
        pred_score: float | None = None,
        sample_name: str | None = None,
        root_path: str = "calibration_status",
    ) -> None:
        """Log one calibration-status sample."""
        self.backend.log_events(
            build_calibration_status_events(
                calibration_data,
                points=points,
                image=image,
                fused_image=fused_image,
                gt_status=gt_status,
                pred_status=pred_status,
                pred_score=pred_score,
                sample_name=sample_name,
                root_path=root_path,
            )
        )

    def log_segmentation3d(
        self,
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
    ) -> None:
        """Log one 3D segmentation sample."""
        self.backend.log_events(
            build_segmentation3d_events(
                points,
                pred_labels,
                pred_probs=pred_probs,
                gt_labels=gt_labels,
                class_names=class_names,
                ignore_index=ignore_index,
                root_path=root_path,
                point_radius=point_radius,
                point_labels=point_labels,
                sample_name=sample_name,
            )
        )

    def log_segmentation3d_data(
        self,
        points: Any,
        labels: Any,
        *,
        class_names: Sequence[str] | None = None,
        ignore_index: int | None = None,
        root_path: str = "segmentation3d",
        point_radius: float = 0.04,
        point_labels: bool = False,
        sample_name: str | None = None,
    ) -> None:
        """Log one transformed 3D segmentation sample without predictions."""
        self.backend.log_events(
            build_segmentation3d_data_events(
                points,
                labels,
                class_names=class_names,
                ignore_index=ignore_index,
                root_path=root_path,
                point_radius=point_radius,
                point_labels=point_labels,
                sample_name=sample_name,
            )
        )

    def log_detection3d(
        self,
        predictions: Mapping[str, Any],
        *,
        points: Any | None = None,
        gt_boxes: Any | None = None,
        gt_labels: Any | None = None,
        class_names: Sequence[str] | None = None,
        root_path: str = "detection3d",
        point_radius: float = 0.04,
        sample_name: str | None = None,
    ) -> None:
        """Log one 3D detection sample."""
        self.backend.log_events(
            build_detection3d_events(
                predictions,
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
            )
        )

    def log_cameras(
        self,
        images: dict[str, Any],
        *,
        root_path: str = "cameras",
    ) -> None:
        """Log per-camera transforms, intrinsics, and images for one sample."""
        self.backend.log_events(build_camera_events(images, root_path=root_path))

    def log_detection3d_data(
        self,
        *,
        points: Any | None = None,
        gt_boxes: Any,
        gt_labels: Any,
        class_names: Sequence[str] | None = None,
        root_path: str = "detection3d",
        point_radius: float = 0.04,
        sample_name: str | None = None,
    ) -> None:
        """Log one transformed 3D detection sample without predictions."""
        self.backend.log_events(
            build_detection3d_data_events(
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                root_path=root_path,
                point_radius=point_radius,
                sample_name=sample_name,
            )
        )
