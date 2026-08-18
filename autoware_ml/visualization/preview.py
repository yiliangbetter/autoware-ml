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

"""Sample-at-a-time visualization preview helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Subset

from autoware_ml.datamodule.base import DataModule
from autoware_ml.models.base import BaseModel
from autoware_ml.visualization.common import as_numpy
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.session import VisualizationSession

PreviewSplit = Literal["train", "val", "test", "predict"]
PreviewMode = Literal["auto", "predictions", "data"]
PreviewTask = Literal["calibration_status", "segmentation3d", "detection3d"]


@dataclass(frozen=True)
class VisualizationPreviewConfig:
    """Configure one visualization preview run."""

    mode: PreviewMode = "auto"
    split: PreviewSplit = "test"
    sample_index: int = 0
    max_samples: int = 1
    device: str = "auto"
    point_labels: bool = False
    session: VisualizationSessionConfig = VisualizationSessionConfig()


def resolve_preview_device(device: str) -> torch.device:
    """Resolve the preview execution device."""
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def run_visualization_preview(
    model: BaseModel | None,
    datamodule: DataModule,
    config: VisualizationPreviewConfig,
) -> int:
    """Render one or more task samples through a visualization session."""
    if config.max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    if config.mode not in {"auto", "predictions", "data"}:
        raise ValueError(f"Unknown visualization mode: {config.mode}")

    mode = _resolve_preview_mode(config.mode, model)
    if mode == "predictions" and model is None:
        raise ValueError("Model must be provided when visualization mode is 'predictions'.")

    device = resolve_preview_device(config.device)
    preview_dataloader, preview_indices = _build_preview_dataloader(
        datamodule=datamodule,
        split=config.split,
        sample_index=config.sample_index,
        max_samples=config.max_samples,
    )

    if model is not None:
        model.to(device)
        model.eval()
    session = VisualizationSession.from_config(config.session)
    visualized_count = 0

    dataset = getattr(datamodule, f"{config.split}_dataset")
    with torch.no_grad():
        for batch_idx, (dataset_index, batch) in enumerate(
            zip(preview_indices, preview_dataloader, strict=True)
        ):
            batch = _move_to_device(batch, device)
            if model is not None:
                batch = model.on_after_batch_transfer(batch, 0)
            predictions = (
                model.predict_step(batch, batch_idx)
                if mode == "predictions" and model is not None
                else None
            )
            raw_info = dataset.get_data_info(dataset_index)
            session.set_step(dataset_index)
            _log_preview_sample(session, batch, predictions, dataset_index, mode, config, raw_info)
            visualized_count += 1

    _wait_for_viewer(session.backend)
    return visualized_count


def _resolve_preview_mode(
    mode: PreviewMode, model: BaseModel | None
) -> Literal["predictions", "data"]:
    """Resolve automatic preview mode selection."""
    if mode == "auto":
        return "predictions" if model is not None else "data"
    return mode


def _wait_for_viewer(backend: Any) -> None:
    """Allow interactive backends to keep their viewer process alive."""
    wait = getattr(backend, "wait_until_interrupted", None)
    if wait is not None:
        wait()


def _build_preview_dataloader(
    datamodule: DataModule,
    split: PreviewSplit,
    sample_index: int,
    max_samples: int,
) -> tuple[DataLoader[Any], list[int]]:
    """Build a one-sample preview dataloader for the selected split."""
    stage_map: dict[PreviewSplit, str] = {
        "train": "fit",
        "val": "validate",
        "test": "test",
        "predict": "predict",
    }
    datamodule.setup(stage_map[split])
    dataset = getattr(datamodule, f"{split}_dataset")
    if dataset is None:
        raise ValueError(f"Split '{split}' is not available for visualization.")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is out of range for split '{split}' "
            f"with {len(dataset)} samples."
        )

    preview_indices = list(range(sample_index, min(len(dataset), sample_index + max_samples)))
    subset = Subset(dataset, preview_indices)
    return (
        DataLoader(
            dataset=subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=datamodule.collate_fn,
        ),
        preview_indices,
    )


def _move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensor-like preview data to the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    return data


def _log_preview_sample(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: Any,
    dataset_index: int,
    mode: PreviewMode,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Dispatch one preview sample to the task-appropriate visualization adapter."""
    task = _infer_preview_task(batch, predictions)
    sample_name = _infer_sample_name(batch, dataset_index)

    if task == "calibration_status":
        _log_calibration_preview(session, batch, predictions, sample_name, mode)
        return
    if task == "segmentation3d":
        if mode == "data":
            _log_segmentation_data_preview(session, batch, sample_name, config, raw_info)
            return
        _log_segmentation_preview(session, batch, predictions, sample_name, config, raw_info)
        return
    if task == "detection3d":
        if mode == "data":
            _log_detection_data_preview(session, batch, sample_name)
            return
        detection_predictions = _extract_single_detection_prediction(predictions)
        if detection_predictions is None:
            raise ValueError("Could not normalize detection predictions for visualization.")
        _log_detection_preview(session, batch, detection_predictions, sample_name)
        return

    raise ValueError(f"Unsupported visualization task: {task}")


def _infer_preview_task(batch: dict[str, Any], predictions: Any) -> PreviewTask:
    """Infer which task adapter should handle the preview sample."""
    if "calibration_data" in batch:
        return "calibration_status"
    if _has_segmentation_sample(batch) or _is_segmentation_predictions(predictions):
        return "segmentation3d"
    if (
        _has_detection_sample(batch)
        or _extract_single_detection_prediction(predictions) is not None
    ):
        return "detection3d"
    raise ValueError("Could not infer a visualization task from the current batch and predictions.")


def _has_segmentation_sample(batch: dict[str, Any]) -> bool:
    """Return whether the batch follows the segmentation3d schema."""
    return batch.get("points") is not None and (
        batch.get("segment") is not None or batch.get("pts_semantic_mask") is not None
    )


def _has_detection_sample(batch: dict[str, Any]) -> bool:
    """Return whether the batch follows the detection3d schema."""
    return batch.get("gt_boxes") is not None and batch.get("gt_labels") is not None


def _is_segmentation_predictions(predictions: Any) -> bool:
    """Return whether prediction outputs match the segmentation contract."""
    return isinstance(predictions, dict) and "pred_labels" in predictions


def _log_calibration_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: Any,
    sample_name: str,
    mode: PreviewMode,
) -> None:
    """Render one calibration-status preview sample."""
    calibration_data = _unwrap_single_item(batch["calibration_data"])
    pred_status: int | None = None
    root_path = "transformed/calibration_status" if mode == "data" else "calibration_status"
    if mode == "predictions":
        pred_probs = as_numpy(predictions)
        pred_probs = pred_probs[0] if pred_probs.ndim > 1 else pred_probs
        pred_status = int(pred_probs.argmax()) if pred_probs.size > 0 else None
        pred_score = float(pred_probs.max()) if pred_probs.size > 0 else None
    else:
        pred_score = None

    gt_status = batch.get("gt_calibration_status")
    if isinstance(gt_status, torch.Tensor):
        gt_status_value = int(gt_status.reshape(-1)[0].item())
    elif isinstance(gt_status, list) and gt_status:
        gt_status_value = int(gt_status[0])
    else:
        gt_status_value = None

    session.log_calibration_status(
        calibration_data,
        points=_unwrap_single_item(batch.get("points")),
        image=_unwrap_single_item(batch.get("img")),
        fused_image=_unwrap_single_item(batch.get("fused_img")),
        gt_status=gt_status_value,
        pred_status=pred_status,
        pred_score=pred_score,
        sample_name=sample_name,
        root_path=root_path,
    )


def _log_segmentation_data_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Render one transformed segmentation sample without predictions."""
    gt_labels = _get_segmentation_gt_labels(batch)
    session.log_segmentation3d_data(
        _get_segmentation_points(batch, gt_labels),
        _unwrap_single_item(gt_labels),
        class_names=_unwrap_single_item(batch.get("class_names")),
        point_labels=config.point_labels,
        sample_name=sample_name,
        root_path="transformed/segmentation3d",
    )
    _log_camera_preview(session, raw_info)


def _log_detection_data_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    sample_name: str,
) -> None:
    """Render one transformed detection sample without predictions."""
    session.log_detection3d_data(
        points=_unwrap_single_item(batch.get("points")),
        gt_boxes=_unwrap_single_item(batch.get("gt_boxes")),
        gt_labels=_unwrap_single_item(batch.get("gt_labels")),
        class_names=_unwrap_single_item(batch.get("class_names")),
        sample_name=sample_name,
        root_path="transformed/detection3d",
    )


def _log_segmentation_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Render one 3D segmentation preview sample."""
    gt_labels = _get_segmentation_gt_labels(batch)
    pred_labels = predictions["pred_labels"]
    session.log_segmentation3d(
        _get_segmentation_points(batch, pred_labels),
        pred_labels,
        pred_probs=predictions.get("pred_probs"),
        gt_labels=_unwrap_single_item(gt_labels),
        class_names=_unwrap_single_item(batch.get("class_names")),
        point_labels=config.point_labels,
        sample_name=sample_name,
    )
    _log_camera_preview(session, raw_info)


def _log_camera_preview(
    session: VisualizationSession,
    raw_info: dict[str, Any] | None,
    root_path: str = "cameras",
) -> None:
    """Log cameras for a sample using raw dataset info (bypasses collation)."""
    if raw_info is None:
        return
    images = raw_info.get("images")
    if isinstance(images, dict) and images:
        session.log_cameras(images, root_path=root_path)


def _log_detection_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: dict[str, Any],
    sample_name: str,
) -> None:
    """Render one 3D detection preview sample."""
    session.log_detection3d(
        predictions,
        points=_unwrap_single_item(batch.get("points")),
        gt_boxes=_unwrap_single_item(batch.get("gt_boxes")),
        gt_labels=_unwrap_single_item(batch.get("gt_labels")),
        class_names=_unwrap_single_item(batch.get("class_names")),
        sample_name=sample_name,
    )


def _extract_single_detection_prediction(predictions: Any) -> dict[str, Any] | None:
    """Normalize one-sample detection predictions for visualization."""
    if isinstance(predictions, dict):
        return predictions
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
        first_prediction = predictions[0]
        if isinstance(first_prediction, dict):
            return first_prediction
    return None


def _unwrap_single_item(value: Any) -> Any:
    """Unwrap the first (and only) item from a single-sample batch value.

    Handles both the old list-based collation format and the current
    tensor-based format where fixed-shape tensors are stacked along a new
    leading batch dimension.  Point-cloud tensors (ndim == 2) are left as-is
    since their leading dimension is the point count, not the batch size.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def _get_segmentation_gt_labels(batch: dict[str, Any]) -> Any:
    """Return the most detailed segmentation ground-truth labels available."""
    for key in ("origin_segment", "segment", "pts_semantic_mask"):
        labels = batch.get(key)
        if labels is not None:
            return labels
    return None


def _get_segmentation_points(batch: dict[str, Any], labels: Any) -> Any:
    """Return point positions aligned with segmentation labels.

    PTv3 pipelines may drop raw ``points`` after formatting and grid sampling.
    In that case, ``inverse`` maps original points to sampled voxel
    representatives, so ``coord[inverse]`` yields visualization positions with
    the same length as point-level predictions and ``origin_segment`` labels.
    """
    point_count = _first_dimension(labels)
    points = _unwrap_single_item(batch.get("points"))
    if points is not None and _first_dimension(points) == point_count:
        return points

    coord = _unwrap_single_item(batch.get("coord"))
    inverse = _unwrap_single_item(batch.get("inverse"))
    if coord is not None and inverse is not None and _first_dimension(inverse) == point_count:
        if isinstance(inverse, torch.Tensor):
            return coord[inverse.long()]
        return coord[inverse.astype(int)]

    if coord is not None and _first_dimension(coord) == point_count:
        return coord

    raise KeyError("Segmentation visualization requires 'points' or 'coord' aligned with labels.")


def _first_dimension(value: Any) -> int | None:
    """Return the leading dimension for tensor-like preview values."""
    value = _unwrap_single_item(value)
    if value is None or not hasattr(value, "shape") or len(value.shape) == 0:
        return None
    return int(value.shape[0])


def _infer_sample_name(batch: dict[str, Any], dataset_index: int) -> str:
    """Build a readable sample name for visualization metadata."""
    for key in ("name", "sample_token", "scene_token", "img_path", "lidar_path"):
        value = _unwrap_single_item(batch.get(key))
        if isinstance(value, str) and value:
            return value

    metadata = _unwrap_single_item(batch.get("metadata"))
    if isinstance(metadata, dict):
        token = metadata.get("token")
        if isinstance(token, str) and token:
            return token
        image_path = metadata.get("image", {}).get("img_path")
        if isinstance(image_path, str) and image_path:
            return Path(image_path).name

    return f"sample-{dataset_index}"
