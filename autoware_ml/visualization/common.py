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

"""Shared conversion helpers for visualization adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    AnnotationInfo,
    TextEvent,
    VisualizationEvent,
)


def as_numpy(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert tensors and sequences into NumPy arrays."""
    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        array = np.asarray(data)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def ensure_xyz(points: Any) -> np.ndarray:
    """Extract xyz coordinates from a point tensor or array."""
    array = as_numpy(points, np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {array.shape}")
    return array[:, :3]


def ensure_image_uint8(image: Any) -> np.ndarray:
    """Normalize image-like data to an HWC uint8 array."""
    array = as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"image must have shape (H, W, C), got {array.shape}")
    if array.shape[0] <= 8 and array.shape[1] > 8 and array.shape[2] > 8:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return array


def format_class_label(
    class_id: int,
    class_names: Sequence[str] | None = None,
    score: float | None = None,
) -> str:
    """Build a readable class label string."""
    if class_names is not None and 0 <= class_id < len(class_names):
        base = class_names[class_id]
    else:
        base = str(class_id)
    if score is None:
        return base
    return f"{base} ({score:.2f})"


def build_class_annotation_context(
    root_path: str,
    palette: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> AnnotationContextEvent | None:
    """Build one semantic legend for visualization descendants."""
    if palette.shape[0] == 0:
        return None
    return AnnotationContextEvent(
        path=root_path,
        annotations=[
            AnnotationInfo(
                id=index,
                label=format_class_label(index, class_names),
                color=tuple(int(channel) for channel in palette[index]),
            )
            for index in range(palette.shape[0])
        ],
    )


def build_sample_metadata_events(
    root_path: str, sample_name: str | None
) -> list[VisualizationEvent]:
    """Build lightweight per-sample metadata events."""
    if sample_name is None:
        return []
    return [TextEvent(f"{root_path}/meta/sample", sample_name)]
