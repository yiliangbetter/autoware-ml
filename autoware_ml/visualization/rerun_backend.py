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
from autoware_ml.visualization.common import POINT_COLOR_MODES
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ClearEvent,
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


def _patch_class_id_array_protocol() -> None:
    """Make ``rerun.datatypes.ClassId`` convertible to NumPy under NumPy 1.x.

    ``rerun-sdk`` 0.23.1 declares ``numpy>=1.23`` but its generated
    ``__array__`` implementations forward their ``copy`` argument straight into
    ``numpy.asarray``.  NumPy only accepts that keyword from 2.0 onwards, and
    NumPy 1.x invokes ``__array__`` without it, so ``copy`` keeps its ``None``
    default and the forwarded call raises ``TypeError``.  Rerun swallows that
    error while serializing, which makes every ``AnnotationContext`` collapse to
    an empty list and silently strips class legends from the viewer.

    Dropping the keyword when it is ``None`` restores serialization on NumPy 1.x
    and leaves NumPy 2.x behaviour untouched, since NumPy 2 always passes an
    explicit ``copy`` value.
    """
    class_id_module = import_module("rerun.datatypes.class_id")
    class_id_type = class_id_module.ClassId

    def __array__(self: Any, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Convert one class id to a NumPy array across NumPy 1.x and 2.x."""
        if copy is None:
            return np.asarray(self.id, dtype=dtype)
        return np.asarray(self.id, dtype=dtype, copy=copy)

    class_id_type.__array__ = __array__


def _verify_annotation_context_support(rerun_module: Any) -> None:
    """Fail loudly when semantic legends cannot reach the viewer.

    Rerun reports serialization problems as warnings rather than exceptions, so
    a broken ``AnnotationContext`` would otherwise degrade silently into a
    viewer without class names or colors.
    """
    probe = rerun_module.AnnotationContext(
        [rerun_module.AnnotationInfo(id=0, label="probe", color=(255, 0, 0, 255))]
    )
    for batch in probe.as_component_batches():
        if "AnnotationContext#" not in str(batch.component_descriptor()):
            continue
        if batch.as_arrow_array().to_pylist() == [[]]:
            raise RuntimeError(
                "Rerun discarded a probe AnnotationContext, so class legends would be "
                "missing from the viewer. This indicates an incompatible "
                "rerun-sdk/numpy combination; expected rerun-sdk 0.23.1 with numpy 1.26.4."
            )
        return
    raise RuntimeError(
        "Rerun did not emit an AnnotationContext component for a probe legend; "
        "the installed rerun-sdk is incompatible with this backend."
    )


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
        _patch_class_id_array_protocol()
        _verify_annotation_context_support(self.rr)
        self.rr.init(
            config.application_id,
            recording_id=config.recording_id,
            spawn=spawn,
        )
        self._fused_blueprint_sent = False
        if config.point_color_mode not in POINT_COLOR_MODES:
            raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
        self.point_color_mode = config.point_color_mode
        self._observed_paths: set[str] = set()
        self._camera_paths: set[str] = set()
        self._scene_blueprint_signature: frozenset[str] = frozenset()

    def wait_until_interrupted(self) -> None:
        """Return immediately because no viewer is served by default."""

    def set_step(self, step: int) -> None:
        """Advance the rerun timeline to one integer step."""
        self.rr.set_time(self.timeline, sequence=int(step))

    def set_timestamp(self, timestamp: float) -> None:
        """Set the source sensor time on a second, timestamp-valued timeline."""
        self.rr.set_time("sensor_time", timestamp=float(timestamp))

    def log_event(self, event: VisualizationEvent) -> None:
        """Translate one visualization event into rerun entities."""
        if isinstance(event, AnnotationContextEvent):
            # Logged statically so one legend covers every frame on the timeline
            # instead of being resolved per step by latest-at semantics.
            self.rr.log(
                event.path,
                self.rr.AnnotationContext(
                    [
                        self.rr.AnnotationInfo(
                            id=annotation.id,
                            label=annotation.label,
                            color=annotation.color,
                        )
                        for annotation in event.annotations
                    ]
                ),
                static=True,
            )
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
            self.rr.log(event.path, self.rr.Scalars(event.value))
            return

        if isinstance(event, TextEvent):
            self.rr.log(event.path, self.rr.TextLog(event.text, level=event.level))
            return

        if isinstance(event, ClearEvent):
            self.rr.log(event.path, self.rr.Clear(recursive=event.recursive))
            return

        raise TypeError(f"Unsupported visualization event: {type(event)!r}")

    def _send_fused_blueprint_if_needed(self, events: list[VisualizationEvent]) -> None:
        """Put the fused image and its layers in one Rerun 2D view."""
        if self._fused_blueprint_sent:
            return
        fused_paths = [
            event.path
            for event in events
            if isinstance(event, ImageEvent) and event.path.endswith("/camera/fused")
        ]
        raw_paths = [
            event.path
            for event in events
            if isinstance(event, ImageEvent) and event.path.endswith("/camera/image")
        ]
        if not fused_paths:
            return
        fused_path = fused_paths[0]
        raw_path = raw_paths[0] if raw_paths else None
        self.rr.send_blueprint(
            self.rr.blueprint.Blueprint(
                self.rr.blueprint.Horizontal(
                    *(
                        [
                            self.rr.blueprint.Spatial2DView(
                                name="Raw camera",
                                origin=raw_path,
                                contents=[raw_path, f"{raw_path}/projected_points"],
                            )
                        ]
                        if raw_path
                        else []
                    ),
                    self.rr.blueprint.Spatial2DView(
                        name="Fused camera",
                        origin=fused_path,
                        contents=[
                            fused_path,
                            f"{fused_path}/depth",
                            f"{fused_path}/intensity",
                        ],
                    ),
                ),
                auto_views=False,
            )
        )
        self._fused_blueprint_sent = True

    def _scene_view(
        self,
        *,
        name: str,
        point_path: str | None,
        detection_path: str | None,
    ) -> Any:
        """Build one explicit scene view with camera frustums overlaid."""
        contents = []
        if point_path is not None and point_path in self._observed_paths:
            contents.append(point_path)
        if detection_path is not None and detection_path in self._observed_paths:
            contents.append(detection_path)
        if self._camera_paths:
            contents.append("scene/cameras/**")
        return self.rr.blueprint.Spatial3DView(
            name=name,
            origin="scene",
            contents=contents,
        )

    def _detection_statistics_views(self) -> list[Any]:
        """Build compact IoU quality and match-count charts when GT exists."""
        metrics_root = "scene/metrics/detection"
        quality_paths = [
            f"{metrics_root}/{name}"
            for name in ("precision", "recall", "mean_matched_iou")
            if f"{metrics_root}/{name}" in self._observed_paths
        ]
        count_paths = [
            f"{metrics_root}/{name}"
            for name in ("true_positives", "false_positives", "false_negatives")
            if f"{metrics_root}/{name}" in self._observed_paths
        ]
        views = []
        if quality_paths:
            views.append(
                self.rr.blueprint.TimeSeriesView(
                    name="3D IoU quality",
                    origin=metrics_root,
                    contents=quality_paths,
                    axis_y=self.rr.blueprint.ScalarAxis(range=(0.0, 1.0), zoom_lock=True),
                )
            )
        if count_paths:
            views.append(
                self.rr.blueprint.TimeSeriesView(
                    name="Detection matches",
                    origin=metrics_root,
                    contents=count_paths,
                )
            )
        return views

    def _comparison_tab(
        self,
        *,
        name: str,
        ground_truth_points: str | None,
        prediction_points: str | None,
    ) -> Any:
        """Build side-by-side GT and prediction scenes plus IoU statistics."""
        scene_views = []
        if any(
            path in self._observed_paths
            for path in (
                "scene/ground_truth/segmentation",
                "scene/ground_truth/detections",
            )
        ):
            scene_views.append(
                self._scene_view(
                    name=f"GT · {name}",
                    point_path=ground_truth_points,
                    detection_path="scene/ground_truth/detections",
                )
            )
        if any(
            path in self._observed_paths
            for path in (
                "scene/prediction/segmentation",
                "scene/prediction/detections",
            )
        ):
            scene_views.append(
                self._scene_view(
                    name=f"Prediction · {name}",
                    point_path=prediction_points,
                    detection_path="scene/prediction/detections",
                )
            )
        comparison = self.rr.blueprint.Horizontal(
            *scene_views,
            name=f"{name} comparison",
        )
        statistics = self._detection_statistics_views()
        if not statistics:
            return comparison
        return self.rr.blueprint.Vertical(
            comparison,
            self.rr.blueprint.Horizontal(*statistics, name="Detection statistics"),
            row_shares=[4.0, 1.0],
            name=f"{name} comparison",
        )

    def _camera_tab(self, camera_paths: list[str]) -> Any:
        """Build inspectable 2D camera views while 3D views show their frustums."""
        return self.rr.blueprint.Tabs(
            *[
                self.rr.blueprint.Spatial2DView(
                    name=path.rsplit("/", 1)[-1],
                    origin=path,
                    contents=[path],
                )
                for path in camera_paths
            ],
            name="Cameras",
        )

    def _send_scene_blueprint_if_needed(self) -> None:
        """Publish a requirements-driven scene blueprint as entities appear."""
        scene_paths = frozenset(
            path for path in self._observed_paths if path == "scene" or path.startswith("scene/")
        )
        signature = scene_paths | frozenset(f"@camera:{path}" for path in self._camera_paths)
        if not scene_paths or signature == self._scene_blueprint_signature:
            return

        tabs: list[tuple[str, Any]] = []
        if any(path.endswith("/segmentation") for path in scene_paths):
            tabs.append(
                (
                    "Semantic",
                    self._comparison_tab(
                        name="Semantic",
                        ground_truth_points="scene/ground_truth/segmentation",
                        prediction_points="scene/prediction/segmentation",
                    ),
                )
            )
        if "scene/lidar/intensity" in scene_paths:
            tabs.append(
                (
                    "Intensity",
                    self._comparison_tab(
                        name="Intensity",
                        ground_truth_points="scene/lidar/intensity",
                        prediction_points="scene/lidar/intensity",
                    ),
                )
            )
        if "scene/lidar/solid" in scene_paths:
            tabs.append(
                (
                    "Geometry",
                    self._comparison_tab(
                        name="Geometry",
                        ground_truth_points="scene/lidar/solid",
                        prediction_points="scene/lidar/solid",
                    ),
                )
            )
        if not tabs and any(path.endswith("/detections") for path in scene_paths):
            tabs.append(
                (
                    "Detections",
                    self._comparison_tab(
                        name="Detections",
                        ground_truth_points=None,
                        prediction_points=None,
                    ),
                )
            )
        if "scene/prediction/entropy" in scene_paths:
            tabs.append(
                (
                    "Uncertainty",
                    self.rr.blueprint.Horizontal(
                        self._scene_view(
                            name="Prediction · Entropy",
                            point_path="scene/prediction/entropy",
                            detection_path="scene/prediction/detections",
                        ),
                        self._scene_view(
                            name="Prediction · Semantic",
                            point_path="scene/prediction/segmentation",
                            detection_path="scene/prediction/detections",
                        ),
                        name="Uncertainty",
                    ),
                )
            )
        camera_paths = sorted(self._camera_paths)
        if camera_paths:
            tabs.append(("Cameras", self._camera_tab(camera_paths)))
        if not tabs:
            return

        preferred_tab = {
            "semantic": "Semantic",
            "intensity": "Intensity",
            "solid": "Geometry",
        }[self.point_color_mode]
        tab_names = [name for name, _ in tabs]
        active_tab = tab_names.index(preferred_tab) if preferred_tab in tab_names else 0
        self.rr.send_blueprint(
            self.rr.blueprint.Blueprint(
                self.rr.blueprint.Tabs(
                    *[tab for _, tab in tabs],
                    active_tab=active_tab,
                    name="Multi-task visualization",
                ),
                self.rr.blueprint.BlueprintPanel(expanded=False),
                self.rr.blueprint.SelectionPanel(expanded=False),
                self.rr.blueprint.TimePanel(expanded=True),
                auto_layout=False,
                auto_views=False,
            ),
            make_active=True,
            make_default=True,
        )
        self._scene_blueprint_signature = signature

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log multiple visualization events."""
        event_list = list(events)
        for event in event_list:
            self._observed_paths.add(event.path)
            if isinstance(event, PinholeEvent):
                self._camera_paths.add(event.path)
            self.log_event(event)
        self._send_fused_blueprint_if_needed(event_list)
        self._send_scene_blueprint_if_needed()


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
        """Flush all logged data, then keep the web viewer alive if requested."""
        recording = self.rr.get_global_data_recording()
        if recording is None:
            raise RuntimeError("Rerun did not create a global data recording.")
        recording.flush(blocking=True)
        if not self.wait:
            return
        logger.info("Rerun web viewer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)
