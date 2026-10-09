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

"""Tests for the LitePT presets and their analytic parameter count."""

from __future__ import annotations

import pytest

from autoware_ml.models.segmentation3d.encoders import ptv3 as ptv3_module
from autoware_ml.models.segmentation3d.encoders.ptv3 import LitePTEncoder
from autoware_ml.tuning import presets
from autoware_ml.tuning.presets import (
    ATTN_DEPTH_PRESETS,
    CONV_DEPTH_PRESETS,
    DEC_DEPTH_PRESETS,
    WIDTH_PRESETS,
    WidthPreset,
    encoder_depths,
    estimate_encoder_parameters,
    rope_frequencies_per_axis,
    rope_span,
)


@pytest.mark.parametrize("head_dim", [6, 12, 16, 18, 24, 32])
def test_rope_span_matches_the_encoder_module(head_dim: int) -> None:
    # The presets module avoids torch, so its copy of the rotary arithmetic
    # must track the real one.
    assert rope_span(head_dim) == ptv3_module.rope_span(head_dim)


def test_rope_frequencies_per_axis() -> None:
    assert rope_frequencies_per_axis(16) == 2
    assert rope_frequencies_per_axis(18) == 3
    assert rope_frequencies_per_axis(24) == 4


@pytest.mark.parametrize("name", sorted(WIDTH_PRESETS))
def test_width_presets_keep_one_head_dim_and_rotate_most_of_it(name: str) -> None:
    preset = WIDTH_PRESETS[name]
    assert preset.head_dim >= presets.ROPE_MIN_HEAD_DIM
    # tiny is the PTv3 comparability anchor and knowingly leaves 4 dims unrotated;
    # large shares its 16-channel heads because those are the ones that build.
    assert preset.fully_rotated or name in ("tiny", "large")


def test_shipped_configs_are_expressible() -> None:
    base = WIDTH_PRESETS["base"]
    assert base.enc_channels == (36, 72, 144, 252, 504)
    assert base.head_dim == 18
    tiny = WIDTH_PRESETS["tiny"]
    assert tiny.enc_channels == (32, 64, 128, 256, 512)
    assert tiny.head_dim == 16
    assert encoder_depths("2-2-2", "6-2") == (2, 2, 2, 6, 2)
    assert encoder_depths("2-2-2", "2-2") == (2, 2, 2, 2, 2)
    assert DEC_DEPTH_PRESETS["light"] == (0, 0, 1, 1)


def test_width_preset_rejects_non_dividing_heads() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        WidthPreset(
            "bad", (32, 64, 128, 256, 512), (3, 4, 8, 16, 32), (64, 64, 128, 256), (4, 4, 8, 16)
        )


def test_width_preset_rejects_mixed_head_dims() -> None:
    with pytest.raises(ValueError, match="mixes head dimensions"):
        WidthPreset(
            "bad", (32, 64, 128, 256, 512), (2, 4, 8, 16, 16), (64, 64, 128, 256), (4, 4, 8, 16)
        )


def test_width_preset_rejects_tiny_heads() -> None:
    with pytest.raises(ValueError, match="rotary minimum"):
        WidthPreset("bad", (4, 8, 16, 32, 64), (1, 2, 4, 8, 16), (8, 8, 16, 32), (2, 2, 4, 8))


def _count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("name", sorted(WIDTH_PRESETS))
@pytest.mark.parametrize("attn_depth", sorted(ATTN_DEPTH_PRESETS))
def test_estimate_matches_the_real_encoder(name: str, attn_depth: str) -> None:
    # The static filter and the analysis rely on the estimate being exact
    # for the modules as built; drift here must fail.
    preset = WIDTH_PRESETS[name]
    depths = encoder_depths("2-2-2", attn_depth)
    encoder = LitePTEncoder(
        in_channels=presets.ENCODER_INPUT_CHANNELS,
        enc_channels=preset.enc_channels,
        enc_num_head=preset.enc_num_head,
        enc_depths=depths,
        mlp_ratio=3,
    )
    assert _count(encoder) == estimate_encoder_parameters(preset.enc_channels, depths, 3)


def test_encoder_input_width_matches_the_shipped_config() -> None:
    # The estimate's input width must follow the width the task configs pass
    # to the encoder, which is the sweep-split voxel feature encoder's output.
    from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder

    assert presets.ENCODER_INPUT_CHANNELS == SweepSplitVoxelFeatureEncoder.out_channels


@pytest.mark.parametrize("conv_depth", sorted(CONV_DEPTH_PRESETS))
def test_estimate_matches_the_conv_stem(conv_depth: str) -> None:
    preset = WIDTH_PRESETS["small"]
    depths = encoder_depths(conv_depth, "2-2")
    encoder = LitePTEncoder(
        in_channels=presets.ENCODER_INPUT_CHANNELS,
        enc_channels=preset.enc_channels,
        enc_num_head=preset.enc_num_head,
        enc_depths=depths,
        mlp_ratio=4,
        stem_type="conv",
        stem_kernel_size=3,
    )
    expected = estimate_encoder_parameters(
        preset.enc_channels, depths, 4, stem_type="conv", stem_kernel_size=3
    )
    assert _count(encoder) == expected


def test_estimate_rejects_unknown_stem() -> None:
    with pytest.raises(ValueError, match="Unknown stem type"):
        estimate_encoder_parameters((12, 24, 48, 96, 192), (1, 1, 1, 1, 1), 2, stem_type="dense")
