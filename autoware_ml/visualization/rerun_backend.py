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

    def wait_until_interrupted(self) -> None:
        """Return immediately because no viewer is served by default."""

    def set_step(self, step: int) -> None:
        """Advance the rerun timeline to one integer step."""
        self.rr.set_time(self.timeline, sequence=int(step))

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
