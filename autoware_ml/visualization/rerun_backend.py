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

"""Rerun-backed visualization backend."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from importlib import import_module
from typing import Any

import numpy as np

from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextEvent,
    Transform3DEvent,
    VisualizationEvent,
)

logger = logging.getLogger(__name__)


def _load_rerun_module() -> Any:
    """Load the optional rerun dependency lazily."""
    return import_module("rerun")


def _yaw_to_quaternions(yaws: np.ndarray) -> np.ndarray:
    """Convert z-axis yaw angles to quaternions in xyzw order."""
    half_angles = yaws * 0.5
    quaternions = np.zeros((yaws.shape[0], 4), dtype=np.float32)
    quaternions[:, 2] = np.sin(half_angles)
    quaternions[:, 3] = np.cos(half_angles)
    return quaternions


class _RerunVisualizationBackendBase:
    """Shared Rerun event translation."""

    def _initialize_recording(self, config: VisualizationSessionConfig, *, spawn: bool) -> None:
        """Initialize one Rerun recording."""
        self.timeline = config.timeline
        self.rr = _load_rerun_module()
        self.rr.init(
            config.application_id,
            recording_id=config.recording_id,
            spawn=spawn,
        )

    def set_step(self, step: int) -> None:
        """Advance the rerun timeline to one integer step."""
        self.rr.set_time_sequence(self.timeline, int(step))

    def log_event(self, event: VisualizationEvent) -> None:
        """Translate one visualization event into rerun entities."""
        if isinstance(event, AnnotationContextEvent):
            logger.debug("Skipping Rerun annotation context for %s.", event.path)
            return

        if isinstance(event, ImageEvent):
            self.rr.log(event.path, self.rr.Image(event.image))
            return

        if isinstance(event, PointCloud3DEvent):
            self.rr.log(
                event.path,
                self.rr.Points3D(
                    event.positions,
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Points2DEvent):
            self.rr.log(
                event.path,
                self.rr.Points2D(
                    event.positions,
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Boxes3DEvent):
            self.rr.log(
                event.path,
                self.rr.Boxes3D(
                    centers=event.centers,
                    sizes=event.sizes,
                    quaternions=_yaw_to_quaternions(event.yaws),
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Transform3DEvent):
            self.rr.log(
                event.path,
                self.rr.Transform3D(
                    translation=event.translation,
                    mat3x3=event.rotation_matrix,
                    relation=self.rr.TransformRelation.ChildFromParent,
                ),
            )
            return

        if isinstance(event, PinholeEvent):
            width, height = event.resolution
            self.rr.log(
                event.path,
                self.rr.Pinhole(
                    image_from_camera=event.image_from_camera,
                    resolution=(width, height),
                ),
            )
            return

        if isinstance(event, ScalarEvent):
            self.rr.log(event.path, self.rr.Scalar(event.value))
            return

        if isinstance(event, TextEvent):
            self.rr.log(event.path, self.rr.TextLog(event.text, level=event.level))
            return

        raise TypeError(f"Unsupported visualization event: {type(event)!r}")

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log multiple visualization events."""
        for event in events:
            self.log_event(event)


class RerunVisualizationBackend(_RerunVisualizationBackendBase):
    """Emit visualization events through the Rerun web viewer."""

    def __init__(self, config: VisualizationSessionConfig) -> None:
        """Initialize one web-served Rerun recording."""
        self._initialize_recording(config, spawn=False)
        self.rr.serve_web(
            open_browser=False,
            web_port=config.web_port,
            grpc_port=config.grpc_port,
            server_memory_limit=config.server_memory_limit,
        )
        self.web_url = (
            f"http://localhost:{config.web_port}"
            f"?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{config.grpc_port}%2Fproxy"
        )
        self.wait = config.wait
        logger.info("Rerun web viewer: %s", self.web_url)

    def wait_until_interrupted(self) -> None:
        """Keep the web viewer server alive until interrupted."""
        if not self.wait:
            return
        logger.info("Rerun web viewer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)
