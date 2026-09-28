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

"""LitePT architecture presets for hyperparameter search.

The Optuna sampler draws scalars, but the LitePT encoder and decoder take
per-stage lists that only make sense together: channel widths and head
counts must divide, widths must grow with depth, and the rotary embedding
wants a head dimension that is a multiple of six. A preset names one
well-formed choice of those lists so the sampler decides between ladders
instead of assembling one element at a time.

This module deliberately imports no torch so the CLI can read it cheaply;
the tests cross-check its arithmetic against the real modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

STAGE_COUNT = 5
DECODER_STAGE_COUNT = STAGE_COUNT - 1
# LitePT gating: convolution-only early stages, attention-only late stages.
CONV_STAGE_COUNT = 3
ATTN_STAGE_COUNT = STAGE_COUNT - CONV_STAGE_COUNT
ROPE_AXES = 3
ROPE_MIN_HEAD_DIM = 6

LINEAR_STEM = "linear"
CONV_STEM = "conv"
STEM_TYPES = (LINEAR_STEM, CONV_STEM)


def rope_span(head_dim: int) -> int:
    """Return how many leading head dimensions the axis-split 3D RoPE rotates.

    Mirrors ``autoware_ml.models.segmentation3d.encoders.ptv3.rope_span``: the
    span splits into three per-axis chunks of paired dimensions, so it is the
    largest multiple of six that fits.

    Args:
        head_dim: Per-head channel count.

    Returns:
        Rotated prefix length; the remaining ``head_dim % 6`` pass through.
    """
    return (head_dim // ROPE_MIN_HEAD_DIM) * ROPE_MIN_HEAD_DIM


def rope_frequencies_per_axis(head_dim: int) -> int:
    """Return how many rotary frequencies each spatial axis receives.

    Args:
        head_dim: Per-head channel count.

    Returns:
        Number of rungs on the per-axis frequency ladder.
    """
    return (rope_span(head_dim) // ROPE_AXES) // 2


@dataclass(frozen=True)
class WidthPreset:
    """Channel widths and head counts for the encoder and the decoder head.

    Attributes:
        name: Preset name used by the search space.
        enc_channels: Encoder channel widths per stage.
        enc_num_head: Encoder attention head counts per stage.
        dec_channels: Decoder channel widths per stage.
        dec_num_head: Decoder attention head counts per stage.
    """

    name: str
    enc_channels: tuple[int, ...]
    enc_num_head: tuple[int, ...]
    dec_channels: tuple[int, ...]
    dec_num_head: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the ladder shape and the head-dimension invariants."""
        if len(self.enc_channels) != STAGE_COUNT or len(self.enc_num_head) != STAGE_COUNT:
            raise ValueError(f"Width preset {self.name!r} needs {STAGE_COUNT} encoder stages.")
        if (
            len(self.dec_channels) != DECODER_STAGE_COUNT
            or len(self.dec_num_head) != DECODER_STAGE_COUNT
        ):
            raise ValueError(
                f"Width preset {self.name!r} needs {DECODER_STAGE_COUNT} decoder stages."
            )
        pairs = list(zip(self.enc_channels, self.enc_num_head, strict=True)) + list(
            zip(self.dec_channels, self.dec_num_head, strict=True)
        )
        for channels, heads in pairs:
            if heads <= 0 or channels % heads != 0:
                raise ValueError(
                    f"Width preset {self.name!r}: {channels} channels do not divide into "
                    f"{heads} heads."
                )
        head_dims = {channels // heads for channels, heads in pairs}
        if len(head_dims) != 1:
            raise ValueError(
                f"Width preset {self.name!r} mixes head dimensions {sorted(head_dims)}; "
                "the rotary embedding wants one per preset."
            )
        if self.head_dim < ROPE_MIN_HEAD_DIM:
            raise ValueError(
                f"Width preset {self.name!r}: head dimension {self.head_dim} is below the "
                f"rotary minimum of {ROPE_MIN_HEAD_DIM}."
            )
        if any(a > b for a, b in zip(self.enc_channels, self.enc_channels[1:], strict=False)):
            raise ValueError(f"Width preset {self.name!r}: encoder widths must not shrink.")

    @property
    def head_dim(self) -> int:
        """Channels per attention head, identical at every stage."""
        return self.enc_channels[0] // self.enc_num_head[0]

    @property
    def rope_span(self) -> int:
        """Rotated head dimensions under axis-split 3D RoPE."""
        return rope_span(self.head_dim)

    @property
    def fully_rotated(self) -> bool:
        """Whether every head dimension carries rotary position."""
        return self.rope_span == self.head_dim


WIDTH_PRESETS: Mapping[str, WidthPreset] = {
    preset.name: preset
    for preset in (
        # The published LitePT parametrization: 18 channels per head everywhere.
        WidthPreset(
            "base", (36, 72, 144, 252, 504), (2, 4, 8, 14, 28), (72, 72, 144, 252), (4, 4, 8, 14)
        ),
        WidthPreset(
            "medium", (36, 72, 144, 216, 432), (2, 4, 8, 12, 24), (72, 72, 144, 216), (4, 4, 8, 12)
        ),
        # PTv3-tiny's widths with the LitePT gating: the comparability anchor.
        # 16 channels per head rotates 12 and passes 4 through unrotated.
        WidthPreset(
            "tiny", (32, 64, 128, 256, 512), (2, 4, 8, 16, 32), (64, 64, 128, 256), (4, 4, 8, 16)
        ),
        WidthPreset(
            "small", (18, 36, 72, 144, 288), (1, 2, 4, 8, 16), (36, 36, 72, 144), (2, 2, 4, 8)
        ),
        WidthPreset(
            "micro", (12, 24, 48, 96, 192), (1, 2, 4, 8, 16), (24, 24, 48, 96), (2, 2, 4, 8)
        ),
    )
}

# Depth presets split where LitePT splits: convolution stages cost the same
# arithmetic per block at every stage, attention stages cost parameters and
# kernel launches instead, so the two are sampled independently.
CONV_DEPTH_PRESETS: Mapping[str, tuple[int, ...]] = {
    "2-2-2": (2, 2, 2),
    "1-1-1": (1, 1, 1),
}
ATTN_DEPTH_PRESETS: Mapping[str, tuple[int, ...]] = {
    "6-2": (6, 2),
    "4-2": (4, 2),
    "2-2": (2, 2),
    "2-1": (2, 1),
}
# The decoder keeps LitePT's rule of no attention at the fine stages: ``none``
# is the published base decoder (unpooling only) and ``light`` is PTv3-tiny's.
DEC_DEPTH_PRESETS: Mapping[str, tuple[int, ...]] = {
    "none": (0, 0, 0, 0),
    "light": (0, 0, 1, 1),
}


def encoder_depths(conv_preset: str, attn_preset: str) -> tuple[int, ...]:
    """Assemble the five-stage encoder depth list from the two depth presets.

    Args:
        conv_preset: Name in :data:`CONV_DEPTH_PRESETS`.
        attn_preset: Name in :data:`ATTN_DEPTH_PRESETS`.

    Returns:
        Blocks per encoder stage, convolution stages first.
    """
    return CONV_DEPTH_PRESETS[conv_preset] + ATTN_DEPTH_PRESETS[attn_preset]


def _linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _layer_norm_params(channels: int) -> int:
    return 2 * channels


def _batch_norm_params(channels: int) -> int:
    return 2 * channels


def conv_block_parameters(channels: int) -> int:
    """Parameters of one convolution-only LitePT block.

    The block is a 3x3x3 submanifold convolution with bias, a linear layer,
    and a layer norm; no attention branch exists so no MLP does either.

    Args:
        channels: Block width.

    Returns:
        Trainable parameter count.
    """
    return (
        27 * channels * channels
        + channels
        + _linear_params(channels, channels)
        + _layer_norm_params(channels)
    )


def attention_block_parameters(channels: int, mlp_ratio: float) -> int:
    """Parameters of one attention-only LitePT block.

    Entry norm, attention norm, packed QKV projection with bias, output
    projection, MLP norm, and the two MLP linears. Rotary buffers carry no
    parameters.

    Args:
        channels: Block width.
        mlp_ratio: Hidden expansion of the block MLP.

    Returns:
        Trainable parameter count.
    """
    hidden = int(channels * mlp_ratio)
    return (
        3 * _layer_norm_params(channels)
        + _linear_params(channels, 3 * channels)
        + _linear_params(channels, channels)
        + _linear_params(channels, hidden)
        + _linear_params(hidden, channels)
    )


def pooling_parameters(in_channels: int, out_channels: int) -> int:
    """Parameters of one serialized pooling: a linear projection and a batch norm."""
    return _linear_params(in_channels, out_channels) + _batch_norm_params(out_channels)


def stem_parameters(in_channels: int, embed_channels: int, stem_type: str, kernel_size: int) -> int:
    """Parameters of the embedding stem.

    Args:
        in_channels: Input feature width.
        embed_channels: Stage-0 width.
        stem_type: ``linear`` (per-point linear with bias) or ``conv``
            (bias-free submanifold convolution).
        kernel_size: Convolution kernel size, ignored for the linear stem.

    Returns:
        Trainable parameter count including the stem batch norm.
    """
    if stem_type == LINEAR_STEM:
        stem = _linear_params(in_channels, embed_channels)
    elif stem_type == CONV_STEM:
        stem = kernel_size**3 * in_channels * embed_channels
    else:
        raise ValueError(f"Unknown stem type {stem_type!r}, expected one of {STEM_TYPES}.")
    return stem + _batch_norm_params(embed_channels)


def estimate_encoder_parameters(
    enc_channels: Sequence[int],
    enc_depths: Sequence[int],
    mlp_ratio: float,
    stem_type: str = LINEAR_STEM,
    stem_kernel_size: int = 3,
    in_channels: int = 4,
) -> int:
    """Estimate the LitePT encoder's trainable parameter count analytically.

    The count is exact for the modules as built today and is used as a cheap
    static feature and budget filter before a trial spends a training slot.

    Args:
        enc_channels: Encoder widths per stage.
        enc_depths: Blocks per stage.
        mlp_ratio: MLP hidden expansion.
        stem_type: Embedding stem variant.
        stem_kernel_size: Stem convolution kernel size.
        in_channels: Input feature width.

    Returns:
        Trainable parameter count of the encoder.
    """
    if len(enc_channels) != STAGE_COUNT or len(enc_depths) != STAGE_COUNT:
        raise ValueError(f"Expected {STAGE_COUNT} encoder stages.")
    total = stem_parameters(in_channels, enc_channels[0], stem_type, stem_kernel_size)
    for stage in range(STAGE_COUNT):
        channels = enc_channels[stage]
        if stage < CONV_STAGE_COUNT:
            block = conv_block_parameters(channels)
        else:
            block = attention_block_parameters(channels, mlp_ratio)
        total += block * enc_depths[stage]
        if stage + 1 < STAGE_COUNT:
            total += pooling_parameters(channels, enc_channels[stage + 1])
    return total
