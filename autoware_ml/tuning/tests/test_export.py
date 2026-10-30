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

"""Tests for the package export helpers that need no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from autoware_ml.tuning.export import (
    MANIFEST_NAME,
    PackageManifest,
    default_palette,
    load_manifest,
    render_ml_package_params,
)
from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.tests.test_search_space import TINY, TUNE_CONFIG
from autoware_ml.utils.runtime import compose_config


def test_ml_package_params_follow_the_trial_config() -> None:
    params = {
        **TINY,
        "grid_size": 0.16,
        "width_preset": "medium",
        "dec_depth_preset": "none",
        "patch_size": 256,
    }
    proposal = LitePTSearchSpace().proposal_from_params(params)
    cfg = compose_config(TUNE_CONFIG, list(proposal.overrides))
    parsed = yaml.safe_load(render_ml_package_params(cfg))["/**"]["ros__parameters"]
    encoder = parsed["encoder"]
    assert encoder["voxel_size"] == [0.16, 0.16, 0.16]
    assert encoder["point_cloud_range"] == list(cfg.point_cloud_range)
    assert encoder["serialization_orders"] == ["z", "z-trans"]
    assert encoder["pooling_strides"] == [2, 2, 2, 2]
    assert encoder["enc_channels"] == [36, 72, 144, 216, 432]
    head = parsed["segmentation3d"]
    assert head["dec_depths"] == [0, 0, 0, 0]
    assert head["dec_patch_size"] == [256, 256, 256, 256]
    # The taxonomy is the database's since #143, so mirror it rather than pin a count;
    # debris is asserted by name because the study constrains its recall.
    taxonomy = list(OmegaConf.select(cfg, "database.taxonomy.segmentation3d.class_names"))
    assert head["class_names"] == taxonomy
    assert "debris" in head["class_names"]
    assert len(head["palette"]) == 3 * len(taxonomy)


def test_default_palette_has_three_values_per_class() -> None:
    assert len(default_palette(20)) == 60
    assert default_palette(30)[:3] == default_palette(30)[75:78]
    assert all(0 <= value <= 255 for value in default_palette(30))


def test_manifest_round_trips_and_requires_every_file(tmp_path: Path) -> None:
    manifest = PackageManifest(
        directory=tmp_path,
        files={"ptv3_encoder.onnx": "a", "ml_package_ptv3.param.yaml": "b"},
        inputs={"ptv3_encoder": ["grid_coord", "feat"]},
        params=dict(TINY),
        overrides=["grid_size=0.12"],
    )
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict()))
    assert load_manifest(tmp_path) is None
    for name in manifest.files:
        (tmp_path / name).write_bytes(b"x")
    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded.files == manifest.files and loaded.inputs == manifest.inputs
    assert loaded.params == manifest.params and loaded.overrides == manifest.overrides
    assert load_manifest(tmp_path / "missing") is None
