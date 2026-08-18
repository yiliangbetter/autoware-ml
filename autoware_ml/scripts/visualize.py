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

"""Visualization preview entrypoint for Autoware-ML models."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig

from autoware_ml.utils.checkpoints import load_model_from_checkpoint
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    set_seed,
)
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.preview import (
    VisualizationPreviewConfig,
    resolve_preview_device,
    run_visualization_preview,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def _build_preview_config(cfg: DictConfig) -> VisualizationPreviewConfig:
    """Normalize optional visualization config values."""
    visualization_cfg = cfg.get("visualization", {})
    return VisualizationPreviewConfig(
        mode=str(visualization_cfg.get("mode", "auto")),
        split=str(visualization_cfg.get("split", "test")),
        sample_index=int(visualization_cfg.get("sample_index", 0)),
        max_samples=int(visualization_cfg.get("max_samples", 1)),
        device=str(visualization_cfg.get("device", "auto")),
        point_labels=bool(visualization_cfg.get("point_labels", False)),
        session=VisualizationSessionConfig(
            backend=str(visualization_cfg.get("backend", "rerun")),
            application_id=str(visualization_cfg.get("application_id", "autoware-ml")),
            recording_id=visualization_cfg.get("recording_id"),
            web_port=int(visualization_cfg.get("web_port", 9090)),
            grpc_port=int(visualization_cfg.get("grpc_port", 9876)),
            wait=bool(visualization_cfg.get("wait", True)),
            server_memory_limit=str(visualization_cfg.get("server_memory_limit", "25%")),
            timeline=str(visualization_cfg.get("timeline", "frame")),
        ),
    )


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Run one-sample visualization preview from a config and checkpoint."""
    log_configuration(cfg)
    configure_torch_runtime()
    set_seed(cfg)

    preview_config = _build_preview_config(cfg)

    logger.info("Instantiating datamodule...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    checkpoint_path = cfg.get("weights", None)
    device = resolve_preview_device(preview_config.device)
    model: L.LightningModule | None = None

    if preview_config.mode in {"auto", "predictions"} and checkpoint_path is not None:
        logger.info("Instantiating model...")
        model = hydra.utils.instantiate(cfg.model)
        model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))

        logger.info("Loading checkpoint: %s", checkpoint_path)
        load_model_from_checkpoint(
            model,
            Path(checkpoint_path),
            map_location=device,
            device=device,
            set_eval=True,
        )
    if preview_config.mode == "predictions" and checkpoint_path is None:
        raise ValueError("Checkpoint path must be provided for prediction visualization.")
    if preview_config.mode not in {"auto", "predictions", "data"}:
        raise ValueError(f"Unknown visualization mode: {preview_config.mode}")

    if preview_config.mode == "predictions":
        effective_mode = "predictions"
    elif preview_config.mode == "data":
        effective_mode = "data"
    else:
        effective_mode = "predictions" if checkpoint_path is not None else "data"

    if effective_mode == "predictions" and model is None:
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for prediction visualization.")
        raise RuntimeError("Prediction visualization requires an instantiated model.")

    logger.info(
        "Starting visualization preview: mode=%s split=%s sample_index=%s max_samples=%s device=%s point_labels=%s backend=%s web_port=%s grpc_port=%s wait=%s",
        effective_mode,
        preview_config.split,
        preview_config.sample_index,
        preview_config.max_samples,
        preview_config.device,
        preview_config.point_labels,
        preview_config.session.backend,
        preview_config.session.web_port,
        preview_config.session.grpc_port,
        preview_config.session.wait,
    )
    visualized_count = run_visualization_preview(model, datamodule, preview_config)
    logger.info("Visualization preview completed for %s sample(s).", visualized_count)


if __name__ == "__main__":
    main()
