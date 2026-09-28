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

"""Tests for the LitePT search space and its Hydra overrides."""

from __future__ import annotations

import hydra
import optuna
import pytest

from autoware_ml.models.segmentation3d.encoders.ptv3 import LitePTEncoder
from autoware_ml.tuning.search_space import (
    PARAM_NAMES,
    LitePTSearchSpace,
    format_override,
)
from autoware_ml.utils.runtime import compose_config

TUNE_CONFIG = "segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2"

TINY = {
    "grid_size": 0.12,
    "width_preset": "tiny",
    "conv_depth_preset": "2-2-2",
    "attn_depth_preset": "2-2",
    "dec_depth_preset": "light",
    "patch_size": 512,
    "mlp_ratio": 4,
    "rope_base": 100.0,
    "stem_type": "linear",
}


def test_format_override_renders_hydra_syntax() -> None:
    assert format_override("grid_size", 0.12) == "grid_size=0.12"
    assert format_override("model.encoder.enc_depths", (2, 2, 2, 6, 2)) == (
        "model.encoder.enc_depths=[2,2,2,6,2]"
    )
    assert format_override("flag", True) == "flag=true"
    assert format_override("model.encoder.stem_type", "conv") == "model.encoder.stem_type=conv"
    assert format_override("a.b", 3, force_add=True) == "++a.b=3"


def test_reference_params_expand_to_the_shipped_tiny_config() -> None:
    proposal = LitePTSearchSpace().proposal_from_params(TINY)
    overrides = set(proposal.overrides)
    assert "grid_size=0.12" in overrides
    assert "model.encoder.enc_channels=[32,64,128,256,512]" in overrides
    assert "model.encoder.enc_depths=[2,2,2,2,2]" in overrides
    assert "model.encoder.enc_patch_size=[512,512,512,512,512]" in overrides
    assert "model.seg3d_head.dec_depths=[0,0,1,1]" in overrides
    assert "model.seg3d_head.dec_patch_size=[512,512,512,512]" in overrides
    assert "++model.encoder.stem_type=linear" in overrides
    assert not any("stem_kernel_size" in item for item in overrides)
    assert proposal.static["head_dim"] == 16
    assert proposal.static["rope_span"] == 12
    assert proposal.static["encoder_params"] > 9_000_000


def test_conv_stem_adds_the_kernel_size() -> None:
    proposal = LitePTSearchSpace().proposal_from_params({**TINY, "stem_type": "conv"})
    assert "++model.encoder.stem_type=conv" in proposal.overrides
    assert "++model.encoder.stem_kernel_size=3" in proposal.overrides


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({**TINY, "width_preset": "huge"}, "not one of the allowed"),
        ({**TINY, "rope_base": 5.0}, "outside"),
        ({key: value for key, value in TINY.items() if key != "patch_size"}, "missing="),
        ({**TINY, "bogus": 1}, "unknown="),
    ],
)
def test_validate_params_rejects_values_outside_the_space(params: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LitePTSearchSpace().validate_params(params)


def test_constructor_rejects_unknown_presets_and_bad_ranges() -> None:
    with pytest.raises(ValueError, match="Unknown width preset"):
        LitePTSearchSpace(width_presets=["base", "giant"])
    with pytest.raises(ValueError, match="rope_base_range"):
        LitePTSearchSpace(rope_base_range=(100.0, 10.0))


def test_suggest_samples_every_parameter_inside_the_space() -> None:
    space = LitePTSearchSpace()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        trial = study.ask()
        proposal = space.suggest(trial)
        assert set(proposal.params) == set(PARAM_NAMES)
        assert set(trial.params) == set(PARAM_NAMES)
        # The stored params round-trip through validation unchanged.
        assert space.validate_params(trial.params) == proposal.params
        assert len(proposal.overrides) in (12, 13)
        study.tell(trial, 0.0)


def test_enqueued_reference_is_what_suggest_returns() -> None:
    space = LitePTSearchSpace()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.enqueue_trial(space.validate_params(TINY))
    proposal = space.suggest(study.ask())
    assert proposal.params == space.validate_params(TINY)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sampled_overrides_compose_and_build_the_model(seed: int) -> None:
    # A proposal is only useful if the training config it produces builds:
    # every list must have the right length and every width must divide.
    space = LitePTSearchSpace()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    proposal = space.suggest(study.ask())
    cfg = compose_config(TUNE_CONFIG, list(proposal.overrides))
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    assert isinstance(encoder, LitePTEncoder)
    assert list(cfg.model.encoder.enc_channels) == proposal.static["enc_channels"]
    assert list(cfg.model.encoder.enc_depths) == proposal.static["enc_depths"]
    assert cfg.grid_size == proposal.params["grid_size"]
    # The head follows the encoder's rotary base and MLP ratio by interpolation.
    assert cfg.model.seg3d_head.dec_rope_base == proposal.params["rope_base"]
    assert cfg.model.seg3d_head.mlp_ratio == proposal.params["mlp_ratio"]
    assert cfg.model.encoder.stem_type == proposal.params["stem_type"]
    head = hydra.utils.instantiate(cfg.model.seg3d_head)
    assert head.dec_depths == list(cfg.model.seg3d_head.dec_depths)
