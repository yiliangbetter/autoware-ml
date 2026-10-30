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

"""Export an untrained trial as a deployable package.

The package is what the inference node loads: the split ONNX modules plus
the ML-package parameter file that tells the node the model's geometry and
classes. The weights are the model's initialization, which is enough for a
latency measurement and useless for anything else, so the ONNX metadata says
so.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
ML_PACKAGE_PARAM_NAME = "ml_package_ptv3.param.yaml"
ENCODER_MODULE = "ptv3_encoder"
SEG_HEAD_MODULE = "ptv3_seg3d_head"
# A fixed, distinguishable colour ladder; the node only checks the length.
_PALETTE = (
    (0, 150, 245),
    (160, 32, 240),
    (255, 255, 0),
    (255, 192, 203),
    (255, 0, 0),
    (255, 240, 150),
    (255, 127, 0),
    (210, 180, 140),
    (255, 0, 255),
    (139, 137, 137),
    (0, 175, 0),
    (230, 230, 250),
    (153, 50, 204),
    (188, 143, 143),
    (64, 224, 208),
    (0, 100, 0),
    (128, 0, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (72, 61, 139),
    (255, 105, 180),
    (46, 139, 87),
    (205, 133, 63),
    (112, 128, 144),
)


@dataclass(frozen=True)
class PackageManifest:
    """What an exported package contains.

    Attributes:
        directory: Package directory.
        files: File name to SHA-256.
        inputs: Declared ONNX inputs per module.
        params: Trial parameters the package realizes.
        overrides: Hydra overrides the package was exported with.
    """

    directory: Path
    files: dict[str, str]
    inputs: dict[str, list[str]]
    params: dict[str, Any]
    overrides: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "directory": str(self.directory),
            "files": self.files,
            "inputs": self.inputs,
            "params": self.params,
            "overrides": self.overrides,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def default_palette(count: int) -> list[int]:
    """Return ``count`` RGB triplets flattened, cycling the fixed ladder."""
    flat: list[int] = []
    for index in range(count):
        flat.extend(_PALETTE[index % len(_PALETTE)])
    return flat


def render_ml_package_params(cfg: DictConfig | Mapping[str, Any]) -> str:
    """Render the node's ML-package parameter file for a composed config.

    The values are the ones the deploy config stamps into the ONNX metadata,
    read from the same places, so the package and the artifact agree.

    Args:
        cfg: Composed training config of the trial.

    Returns:
        YAML text.
    """
    # Only the nodes needed are resolved: the full config carries Hydra
    # runtime interpolations that exist only inside a Hydra job.
    if OmegaConf.is_config(cfg):
        modules = OmegaConf.to_container(OmegaConf.select(cfg, "deploy.onnx.modules"), resolve=True)
        dec_patch_size = OmegaConf.to_container(
            OmegaConf.select(cfg, "model.seg3d_head.dec_patch_size"), resolve=True
        )
    else:
        modules = cfg["deploy"]["onnx"]["modules"]
        dec_patch_size = cfg["model"]["seg3d_head"].get("dec_patch_size")
    assert isinstance(modules, Mapping)
    encoder_meta = modules[ENCODER_MODULE]["metainfo"]
    head_meta = modules[SEG_HEAD_MODULE]["metainfo"]
    voxel_size = [float(value) for value in encoder_meta["voxel_size"]]
    class_names = [str(name) for name in head_meta["class_names"]]
    if dec_patch_size is None:
        dec_patch_size = [512] * len(head_meta["dec_depths"])
    lines = [
        "/**:",
        "  ros__parameters:",
        "    encoder:",
        f"      point_cloud_range: {_yaml_list([float(v) for v in encoder_meta['point_cloud_range']])}",
        f"      voxel_size: {_yaml_list(voxel_size)}",
        f"      serialization_orders: {_yaml_list([str(o) for o in encoder_meta['serialization_orders']])}",
        f"      pooling_strides: {_yaml_list([int(s) for s in encoder_meta['pooling_strides']])}",
        f"      enc_channels: {_yaml_list([int(c) for c in encoder_meta['enc_channels']])}",
        "    segmentation3d:",
        f"      dec_depths: {_yaml_list([int(d) for d in head_meta['dec_depths']])}",
        f"      dec_patch_size: {_yaml_list([int(p) for p in dec_patch_size])}",
        f"      class_names: {_yaml_list(class_names)}",
        f"      palette: {_yaml_list(default_palette(len(class_names)))}",
        "",
    ]
    return "\n".join(lines)


def export_package(
    cfg: DictConfig,
    output_dir: Path,
    *,
    config_name: str,
    params: Mapping[str, Any],
    overrides: Sequence[str],
) -> PackageManifest:
    """Export the configured model with its initial weights into ``output_dir``.

    Runs the same per-module pipeline as ``autoware-ml deploy`` minus the
    checkpoint and the tracking run: ONNX export, graph modifiers, precision
    conversion and the metadata stamp.

    Args:
        cfg: Composed config of the trial.
        output_dir: Package directory, created if needed.
        config_name: User-facing config name recorded in the metadata.
        params: Trial parameters recorded in the manifest.
        overrides: Overrides recorded in the manifest.

    Returns:
        The manifest, also written as ``manifest.json``.
    """
    import hydra
    import onnx
    import torch

    from autoware_ml.utils.deploy import (
        export_to_onnx,
        merge_module_onnx_cfg,
        modify_onnx_graph,
        resolve_export_specs,
        should_modify_graph,
        validate_cuda_available,
    )
    from autoware_ml.utils.mlflow_helpers import get_git_sha
    from autoware_ml.utils.onnx_meta import stamp_onnx_meta
    from autoware_ml.utils.runtime import configure_torch_runtime, set_seed

    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))
    model.to(device).eval()
    export_specs = resolve_export_specs(datamodule, model, device)
    git_sha = get_git_sha()
    files: dict[str, str] = {}
    inputs: dict[str, list[str]] = {}
    for module_name, spec in export_specs.items():
        module_cfg = merge_module_onnx_cfg(cfg.deploy.onnx, module_name)
        final_path = output_dir / f"{module_name}.onnx"
        # The precision helpers pull in the fp16 converter, which not every
        # environment carries; an fp32 export, the default, needs neither.
        precision = str(module_cfg.get("precision", "fp32")).lower()
        converts_precision = precision != "fp32"
        if converts_precision:
            from autoware_ml.utils.onnx_precision import validate_module_onnx_precision

            validate_module_onnx_precision(spec.module, module_cfg)
        else:
            for submodule in spec.module.modules():
                required = getattr(submodule, "required_onnx_precision", None)
                if required is not None and str(required) != precision:
                    raise ValueError(
                        f"{type(submodule).__name__} requires deploy.onnx.precision="
                        f"'{required}', but the module config uses '{precision}'."
                    )
        export_to_onnx(
            spec.module,
            spec.args,
            module_cfg,
            spec.input_names,
            spec.output_names,
            spec.dynamic_axes,
            final_path,
        )
        path = final_path
        modify_graph_cfg = module_cfg.get("modify_graph", None)
        if should_modify_graph(modify_graph_cfg):
            path = modify_onnx_graph(path, modify_graph_cfg)
        if converts_precision:
            from autoware_ml.utils.onnx_precision import (
                convert_onnx_precision,
                resolve_onnx_precision,
            )

            path = convert_onnx_precision(path, resolve_onnx_precision(module_cfg))
        if path != final_path:
            path.replace(final_path)
        metainfo_cfg = module_cfg.get("metainfo", None)
        metainfo = (
            OmegaConf.to_container(metainfo_cfg, resolve=True) if metainfo_cfg is not None else {}
        )
        assert isinstance(metainfo, dict)
        metainfo["weights"] = "initialization only: exported for latency screening"
        stamp_onnx_meta(
            final_path,
            config_name=config_name,
            module=module_name,
            release=None,
            export_git_sha=git_sha,
            metainfo=metainfo,
            tracker=None,
            run_id=None,
        )
        graph = onnx.load(str(final_path), load_external_data=False).graph
        inputs[module_name] = [item.name for item in graph.input]
        files[final_path.name] = _sha256(final_path)
        logger.info("Exported %s with %d inputs", final_path, len(inputs[module_name]))

    params_path = output_dir / ML_PACKAGE_PARAM_NAME
    params_path.write_text(render_ml_package_params(cfg))
    files[params_path.name] = _sha256(params_path)
    manifest = PackageManifest(
        directory=output_dir,
        files=files,
        inputs=inputs,
        params=dict(params),
        overrides=list(overrides),
    )
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return manifest


def load_manifest(package_dir: Path) -> PackageManifest | None:
    """Return the manifest of an exported package, if it is complete."""
    path = package_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    for name in data.get("files", {}):
        if not (package_dir / name).is_file():
            return None
    return PackageManifest(
        directory=package_dir,
        files=dict(data["files"]),
        inputs={key: list(value) for key, value in data.get("inputs", {}).items()},
        params=dict(data.get("params", {})),
        overrides=list(data.get("overrides", [])),
    )
