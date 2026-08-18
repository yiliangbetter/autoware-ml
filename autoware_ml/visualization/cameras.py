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

"""Camera visualization adapters.

Logs each camera as a Transform3D + Pinhole + Image triplet so that Rerun
can project 3D points onto camera images on hover.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.visualization.events import (
    ImageEvent,
    PinholeEvent,
    Transform3DEvent,
    VisualizationEvent,
)

logger = logging.getLogger(__name__)


def build_camera_events(
    images: dict[str, Any],
    *,
    root_path: str = "cameras",
) -> list[VisualizationEvent]:
    """Build visualization events for all cameras in one sample.

    Each camera gets a ``Transform3D`` (lidar-to-camera extrinsic expressed as
    child-from-parent), a ``Pinhole`` (intrinsic), and an ``Image`` logged
    under ``{root_path}/{cam_name}``.  Rerun uses the transform hierarchy to
    enable automatic point-to-image projection when hovering over 3D points.

    Args:
        images: Per-camera dict from a T4Dataset batch entry.  Each value must
            contain ``img_path`` (str), ``cam2img`` (3×3 array-like), and
            ``lidar2cam`` (4×4 array-like).
        root_path: Entity path prefix for all camera entities.

    Returns:
        List of ``Transform3DEvent``, ``PinholeEvent``, and ``ImageEvent``
        objects, one triplet per camera.
    """
    events: list[VisualizationEvent] = []
    for cam_name, cam_info in images.items():
        cam_path = f"{root_path}/{cam_name}"
        try:
            events.extend(_build_single_camera_events(cam_info, cam_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping camera %r: %s", cam_name, exc)
    return events


def _build_single_camera_events(
    cam_info: dict[str, Any],
    cam_path: str,
) -> list[VisualizationEvent]:
    """Build the three visualization events for one camera."""
    img_path = cam_info.get("img_path")
    cam2img = cam_info.get("cam2img")
    lidar2cam = cam_info.get("lidar2cam")

    if img_path is None or cam2img is None or lidar2cam is None:
        raise ValueError("missing img_path, cam2img, or lidar2cam")

    image_array = _load_image(str(img_path))
    height, width = image_array.shape[:2]

    intrinsic = np.asarray(cam2img, dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(lidar2cam, dtype=np.float32).reshape(4, 4)

    return [
        Transform3DEvent(
            path=cam_path,
            translation=extrinsic[:3, 3],
            rotation_matrix=extrinsic[:3, :3],
        ),
        PinholeEvent(
            path=cam_path,
            image_from_camera=intrinsic,
            resolution=(width, height),
        ),
        ImageEvent(
            path=cam_path,
            image=image_array,
        ),
    ]


def _load_image(img_path: str) -> npt.NDArray[np.uint8]:
    """Load an RGB image from disk using cv2."""
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
