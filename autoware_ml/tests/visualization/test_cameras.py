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

"""Tests for the multiview camera visualization adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from autoware_ml.visualization.cameras import build_camera_events
from autoware_ml.visualization.events import ImageEvent, PinholeEvent, Transform3DEvent

_INTRINSIC = [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
_EXTRINSIC = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
]


@pytest.fixture
def camera_image(tmp_path: Path) -> Path:
    """Write one small BGR image to disk and return its path."""
    image_path = tmp_path / "cam_front.png"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    return image_path


def _camera_entry(image_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build one camera info entry with optional overrides."""
    entry: dict[str, Any] = {
        "img_path": str(image_path),
        "cam2img": _INTRINSIC,
        "lidar2cam": _EXTRINSIC,
    }
    entry.update(overrides)
    return entry


def test_build_camera_events_emits_one_triplet_per_camera(camera_image: Path) -> None:
    events = build_camera_events(
        {
            "CAM_FRONT": _camera_entry(camera_image),
            "CAM_BACK": _camera_entry(camera_image),
        }
    )

    assert [event.path for event in events if isinstance(event, Transform3DEvent)] == [
        "cameras/CAM_FRONT",
        "cameras/CAM_BACK",
    ]
    assert [event.path for event in events if isinstance(event, PinholeEvent)] == [
        "cameras/CAM_FRONT",
        "cameras/CAM_BACK",
    ]
    assert [event.path for event in events if isinstance(event, ImageEvent)] == [
        "cameras/CAM_FRONT",
        "cameras/CAM_BACK",
    ]


def test_build_camera_events_reads_resolution_from_the_image(camera_image: Path) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)})

    pinhole = next(event for event in events if isinstance(event, PinholeEvent))
    assert pinhole.resolution == (64, 36)


def test_build_camera_events_splits_the_extrinsic_into_rotation_and_translation(
    camera_image: Path,
) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)})

    transform = next(event for event in events if isinstance(event, Transform3DEvent))
    np.testing.assert_allclose(transform.translation, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform.rotation_matrix, np.eye(3, dtype=np.float32))


def test_build_camera_events_honors_the_root_path(camera_image: Path) -> None:
    events = build_camera_events({"CAM_FRONT": _camera_entry(camera_image)}, root_path="multiview")

    assert all(event.path == "multiview/CAM_FRONT" for event in events)


@pytest.mark.parametrize("missing_key", ["img_path", "cam2img", "lidar2cam"])
def test_build_camera_events_raises_on_missing_calibration(
    camera_image: Path, missing_key: str
) -> None:
    """A camera missing calibration must fail loudly instead of being skipped."""
    entry = _camera_entry(camera_image, **{missing_key: None})

    with pytest.raises(ValueError, match=f"CAM_FRONT.*{missing_key}"):
        build_camera_events({"CAM_FRONT": entry})


def test_build_camera_events_raises_on_unreadable_image(tmp_path: Path) -> None:
    """A missing image must fail loudly rather than silently drop the camera."""
    entry = _camera_entry(tmp_path / "does_not_exist.png")

    with pytest.raises(FileNotFoundError, match="does_not_exist.png"):
        build_camera_events({"CAM_FRONT": entry})


def test_build_camera_events_is_empty_without_cameras() -> None:
    assert build_camera_events({}) == []
