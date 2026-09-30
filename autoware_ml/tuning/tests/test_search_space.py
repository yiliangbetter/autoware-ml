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
    CATEGORICAL_PARAM_NAMES,
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
    "stem_type": "linear",
}
ROPE_RANGE = (10.0, 1000.0)


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
        ({key: value for key, value in TINY.items() if key != "patch_size"}, "missing="),
        ({**TINY, "bogus": 1}, "unknown="),
        # The rotary base is not a parameter of a space that does not search it.
        ({**TINY, "rope_base": 100.0}, "unknown="),
    ],
)
def test_validate_params_rejects_values_outside_the_space(params: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LitePTSearchSpace().validate_params(params)


def test_rope_base_is_searched_only_when_a_range_is_given() -> None:
    fixed = LitePTSearchSpace()
    assert not fixed.searches_rope_base
    assert fixed.param_names == CATEGORICAL_PARAM_NAMES
    assert not any("rope_base" in item for item in fixed.proposal_from_params(TINY).overrides)
    searched = LitePTSearchSpace(rope_base_range=ROPE_RANGE)
    assert searched.searches_rope_base
    assert set(searched.param_names) == set(PARAM_NAMES)
    proposal = searched.proposal_from_params({**TINY, "rope_base": 100.0})
    assert "model.encoder.enc_rope_base=100.0" in proposal.overrides
    with pytest.raises(ValueError, match="outside"):
        searched.validate_params({**TINY, "rope_base": 5.0})
    with pytest.raises(ValueError, match="missing="):
        searched.validate_params(TINY)


def test_cell_key_ignores_the_rotary_base() -> None:
    searched = LitePTSearchSpace(rope_base_range=ROPE_RANGE)
    one = searched.cell_key({**TINY, "rope_base": 10.0})
    other = searched.cell_key({**TINY, "rope_base": 900.0})
    assert one == other == LitePTSearchSpace().cell_key(TINY)
    assert one != LitePTSearchSpace().cell_key({**TINY, "mlp_ratio": 2})


def test_constructor_rejects_unknown_presets_and_bad_ranges() -> None:
    with pytest.raises(ValueError, match="Unknown width preset"):
        LitePTSearchSpace(width_presets=["base", "giant"])
    with pytest.raises(ValueError, match="rope_base_range"):
        LitePTSearchSpace(rope_base_range=(100.0, 10.0))


@pytest.mark.parametrize("rope_base_range", [None, ROPE_RANGE])
def test_suggest_samples_every_parameter_inside_the_space(rope_base_range) -> None:
    space = LitePTSearchSpace(rope_base_range=rope_base_range)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    rope_override = 1 if rope_base_range else 0
    for _ in range(20):
        trial = study.ask()
        proposal = space.suggest(trial)
        assert set(proposal.params) == set(space.param_names)
        assert set(trial.params) == set(space.param_names)
        # The stored params round-trip through validation unchanged.
        assert space.validate_params(trial.params) == proposal.params
        assert len(proposal.overrides) in (11 + rope_override, 12 + rope_override)
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
    # The rotary base stays at the config default and the head follows the
    # encoder's base and MLP ratio by interpolation.
    assert cfg.model.encoder.enc_rope_base == 100.0
    assert cfg.model.seg3d_head.dec_rope_base == cfg.model.encoder.enc_rope_base
    assert cfg.model.seg3d_head.mlp_ratio == proposal.params["mlp_ratio"]
    assert cfg.model.encoder.stem_type == proposal.params["stem_type"]
    head = hydra.utils.instantiate(cfg.model.seg3d_head)
    assert head.dec_depths == list(cfg.model.seg3d_head.dec_depths)
