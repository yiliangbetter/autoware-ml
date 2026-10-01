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

"""LitePT search space: Optuna parameters in, Hydra overrides out.

The space samples a handful of scalars and preset names and expands them
into the per-stage lists the model config needs. Reference trials are
expressed in the same parameter vocabulary so they can be enqueued and
compared on the same footing as sampled ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import optuna

from autoware_ml.tuning.presets import (
    ATTN_DEPTH_PRESETS,
    CONV_DEPTH_PRESETS,
    CONV_STEM,
    DEC_DEPTH_PRESETS,
    DECODER_STAGE_COUNT,
    ENCODER_INPUT_CHANNELS,
    STAGE_COUNT,
    STEM_TYPES,
    WIDTH_PRESETS,
    encoder_depths,
    estimate_encoder_parameters,
    rope_frequencies_per_axis,
)

PARAM_GRID_SIZE = "grid_size"
PARAM_WIDTH = "width_preset"
PARAM_CONV_DEPTH = "conv_depth_preset"
PARAM_ATTN_DEPTH = "attn_depth_preset"
PARAM_DEC_DEPTH = "dec_depth_preset"
PARAM_PATCH_SIZE = "patch_size"
PARAM_MLP_RATIO = "mlp_ratio"
PARAM_ROPE_BASE = "rope_base"
PARAM_STEM = "stem_type"
PARAM_NAMES = (
    PARAM_GRID_SIZE,
    PARAM_WIDTH,
    PARAM_CONV_DEPTH,
    PARAM_ATTN_DEPTH,
    PARAM_DEC_DEPTH,
    PARAM_PATCH_SIZE,
    PARAM_MLP_RATIO,
    PARAM_ROPE_BASE,
    PARAM_STEM,
)
# The categorical parameters every space samples; ``rope_base`` is optional.
CATEGORICAL_PARAM_NAMES = tuple(name for name in PARAM_NAMES if name != PARAM_ROPE_BASE)


def _format_value(value: Any) -> str:
    """Render one scalar or list as a Hydra override value."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def format_override(key: str, value: Any, *, force_add: bool = False) -> str:
    """Render ``key=value`` in Hydra's command-line override syntax.

    Args:
        key: Dotted config path.
        value: Scalar or list value.
        force_add: Emit Hydra's ``++`` prefix so the override also applies
            to a key the config does not declare, such as an optional
            constructor argument left at its default.

    Returns:
        Override string accepted by ``hydra.compose`` and the CLI.
    """
    prefix = "++" if force_add else ""
    return f"{prefix}{key}={_format_value(value)}"


@dataclass(frozen=True)
class TrialProposal:
    """One sampled configuration and everything derived from it.

    Attributes:
        params: Sampled values keyed by parameter name.
        overrides: Hydra overrides that realize the configuration.
        static: Features computable without training, such as the estimated
            encoder parameter count, for filtering and analysis.
    """

    params: dict[str, Any]
    overrides: tuple[str, ...]
    static: dict[str, Any] = field(default_factory=dict)


class LitePTSearchSpace:
    """Sample LitePT architectures and translate them into Hydra overrides.

    Every list argument is the set of allowed values for the corresponding
    categorical parameter. The rotary base is drawn log-uniformly from
    ``rope_base_range`` when one is given; with ``None`` it is not searched
    and the model config's default applies, which the rope ablation of
    2026-09-29 showed to be the right call (mIoU flat over two decades).
    Narrow a study by editing the lists in config, not the code.
    """

    def __init__(
        self,
        grid_sizes: Sequence[float] = (0.10, 0.12, 0.14, 0.16, 0.18),
        width_presets: Sequence[str] = ("base", "medium", "tiny", "small", "micro"),
        conv_depth_presets: Sequence[str] = ("2-2-2", "1-1-1"),
        attn_depth_presets: Sequence[str] = ("6-2", "4-2", "2-2", "2-1"),
        dec_depth_presets: Sequence[str] = ("none", "light"),
        patch_sizes: Sequence[int] = (256, 512, 1024),
        mlp_ratios: Sequence[int] = (2, 3, 4),
        rope_base_range: Sequence[float] | None = None,
        stem_types: Sequence[str] = ("linear", "conv"),
        conv_stem_kernel_size: int = 3,
        encoder_key: str = "model.encoder",
        head_key: str = "model.seg3d_head",
        in_channels: int = ENCODER_INPUT_CHANNELS,
    ) -> None:
        """Initialize the search space.

        Args:
            grid_sizes: Allowed voxel sizes in meters.
            width_presets: Allowed :data:`WIDTH_PRESETS` names.
            conv_depth_presets: Allowed :data:`CONV_DEPTH_PRESETS` names.
            attn_depth_presets: Allowed :data:`ATTN_DEPTH_PRESETS` names.
            dec_depth_presets: Allowed :data:`DEC_DEPTH_PRESETS` names.
            patch_sizes: Allowed attention patch sizes, applied to every
                attention stage of the encoder and the decoder.
            mlp_ratios: Allowed MLP expansion ratios.
            rope_base_range: ``(low, high)`` of the log-uniform rotary base,
                or ``None`` to leave the base at the config default.
            stem_types: Allowed embedding stems.
            conv_stem_kernel_size: Kernel size used when the conv stem is drawn.
            encoder_key: Config path of the encoder node.
            head_key: Config path of the segmentation head node.
            in_channels: Input feature width, for the parameter estimate.
        """
        self.grid_sizes = tuple(float(value) for value in grid_sizes)
        self.width_presets = tuple(_check_names(width_presets, WIDTH_PRESETS, "width"))
        self.conv_depth_presets = tuple(
            _check_names(conv_depth_presets, CONV_DEPTH_PRESETS, "convolution depth")
        )
        self.attn_depth_presets = tuple(
            _check_names(attn_depth_presets, ATTN_DEPTH_PRESETS, "attention depth")
        )
        self.dec_depth_presets = tuple(
            _check_names(dec_depth_presets, DEC_DEPTH_PRESETS, "decoder depth")
        )
        self.patch_sizes = tuple(int(value) for value in patch_sizes)
        self.mlp_ratios = tuple(int(value) for value in mlp_ratios)
        self.rope_base_range: tuple[float, float] | None = None
        if rope_base_range is not None:
            low, high = (float(value) for value in rope_base_range)
            if not 0 < low < high:
                raise ValueError(f"rope_base_range must satisfy 0 < low < high, got {low}, {high}.")
            self.rope_base_range = (low, high)
        self.stem_types = tuple(_check_names(stem_types, dict.fromkeys(STEM_TYPES), "stem"))
        self.conv_stem_kernel_size = int(conv_stem_kernel_size)
        self.encoder_key = encoder_key
        self.head_key = head_key
        self.in_channels = int(in_channels)

    @property
    def searches_rope_base(self) -> bool:
        """Whether the rotary base is a sampled parameter of this space."""
        return self.rope_base_range is not None

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters this space samples, in ``PARAM_NAMES`` order."""
        return tuple(
            name for name in PARAM_NAMES if name != PARAM_ROPE_BASE or self.searches_rope_base
        )

    def cell_key(self, params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        """Return a hashable identity of the categorical cell ``params`` falls in.

        The rotary base is excluded on purpose: it is continuous, and the
        ablation showed it does not separate architectures, so trials that
        differ only in it are replicates of one cell.
        """
        normalized = self.validate_params(params)
        return tuple((name, normalized[name]) for name in CATEGORICAL_PARAM_NAMES)

    def suggest(self, trial: optuna.Trial) -> TrialProposal:
        """Draw one configuration from the trial and expand it.

        Args:
            trial: Optuna trial to sample from.

        Returns:
            The proposal with its overrides and static features.
        """
        params = {
            PARAM_GRID_SIZE: trial.suggest_categorical(PARAM_GRID_SIZE, list(self.grid_sizes)),
            PARAM_WIDTH: trial.suggest_categorical(PARAM_WIDTH, list(self.width_presets)),
            PARAM_CONV_DEPTH: trial.suggest_categorical(
                PARAM_CONV_DEPTH, list(self.conv_depth_presets)
            ),
            PARAM_ATTN_DEPTH: trial.suggest_categorical(
                PARAM_ATTN_DEPTH, list(self.attn_depth_presets)
            ),
            PARAM_DEC_DEPTH: trial.suggest_categorical(
                PARAM_DEC_DEPTH, list(self.dec_depth_presets)
            ),
            PARAM_PATCH_SIZE: trial.suggest_categorical(PARAM_PATCH_SIZE, list(self.patch_sizes)),
            PARAM_MLP_RATIO: trial.suggest_categorical(PARAM_MLP_RATIO, list(self.mlp_ratios)),
            PARAM_STEM: trial.suggest_categorical(PARAM_STEM, list(self.stem_types)),
        }
        if self.rope_base_range is not None:
            params[PARAM_ROPE_BASE] = trial.suggest_float(
                PARAM_ROPE_BASE, *self.rope_base_range, log=True
            )
        return self.proposal_from_params(params)

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Check a fixed parameter set, such as a reference trial, against the space.

        Args:
            params: Values keyed by parameter name.

        Returns:
            The normalized parameters.

        Raises:
            ValueError: If a name is unknown, missing, or outside the space.
        """
        expected = set(self.param_names)
        unknown = sorted(set(params) - expected)
        missing = sorted(expected - set(params))
        if unknown or missing:
            raise ValueError(
                f"Trial parameters mismatch the search space: unknown={unknown}, missing={missing}."
            )
        normalized: dict[str, Any] = {
            PARAM_GRID_SIZE: float(params[PARAM_GRID_SIZE]),
            PARAM_WIDTH: str(params[PARAM_WIDTH]),
            PARAM_CONV_DEPTH: str(params[PARAM_CONV_DEPTH]),
            PARAM_ATTN_DEPTH: str(params[PARAM_ATTN_DEPTH]),
            PARAM_DEC_DEPTH: str(params[PARAM_DEC_DEPTH]),
            PARAM_PATCH_SIZE: int(params[PARAM_PATCH_SIZE]),
            PARAM_MLP_RATIO: int(params[PARAM_MLP_RATIO]),
            PARAM_STEM: str(params[PARAM_STEM]),
        }
        if self.rope_base_range is not None:
            normalized[PARAM_ROPE_BASE] = float(params[PARAM_ROPE_BASE])
        choices = {
            PARAM_GRID_SIZE: self.grid_sizes,
            PARAM_WIDTH: self.width_presets,
            PARAM_CONV_DEPTH: self.conv_depth_presets,
            PARAM_ATTN_DEPTH: self.attn_depth_presets,
            PARAM_DEC_DEPTH: self.dec_depth_presets,
            PARAM_PATCH_SIZE: self.patch_sizes,
            PARAM_MLP_RATIO: self.mlp_ratios,
            PARAM_STEM: self.stem_types,
        }
        for name, allowed in choices.items():
            if normalized[name] not in allowed:
                raise ValueError(
                    f"{name}={normalized[name]!r} is not one of the allowed values {allowed}."
                )
        if self.rope_base_range is not None:
            low, high = self.rope_base_range
            if not low <= normalized[PARAM_ROPE_BASE] <= high:
                raise ValueError(
                    f"{PARAM_ROPE_BASE}={normalized[PARAM_ROPE_BASE]} is outside [{low}, {high}]."
                )
        return normalized

    def proposal_from_params(self, params: Mapping[str, Any]) -> TrialProposal:
        """Expand a parameter set into overrides and static features.

        Args:
            params: Values keyed by parameter name, sampled or fixed.

        Returns:
            The proposal.
        """
        normalized = self.validate_params(params)
        return TrialProposal(
            params=normalized,
            overrides=self.build_overrides(normalized),
            static=self.static_features(normalized),
        )

    def build_overrides(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        """Translate parameters into Hydra overrides.

        Args:
            params: Normalized parameter values.

        Returns:
            Override strings for the training and evaluation configs.
        """
        width = WIDTH_PRESETS[params[PARAM_WIDTH]]
        depths = encoder_depths(params[PARAM_CONV_DEPTH], params[PARAM_ATTN_DEPTH])
        patch = int(params[PARAM_PATCH_SIZE])
        enc = self.encoder_key
        head = self.head_key
        overrides = [
            format_override(PARAM_GRID_SIZE, params[PARAM_GRID_SIZE]),
            format_override(f"{enc}.enc_channels", width.enc_channels),
            format_override(f"{enc}.enc_num_head", width.enc_num_head),
            format_override(f"{enc}.enc_depths", depths),
            format_override(f"{enc}.enc_patch_size", (patch,) * STAGE_COUNT),
            format_override(f"{enc}.mlp_ratio", params[PARAM_MLP_RATIO]),
            # The stem keys are constructor defaults the shipped configs do
            # not declare, so they are force-added rather than overridden.
            format_override(f"{enc}.stem_type", params[PARAM_STEM], force_add=True),
            format_override(f"{head}.dec_channels", width.dec_channels),
            format_override(f"{head}.dec_num_head", width.dec_num_head),
            format_override(f"{head}.dec_depths", DEC_DEPTH_PRESETS[params[PARAM_DEC_DEPTH]]),
            format_override(f"{head}.dec_patch_size", (patch,) * DECODER_STAGE_COUNT),
        ]
        if PARAM_ROPE_BASE in params:
            overrides.append(format_override(f"{enc}.enc_rope_base", params[PARAM_ROPE_BASE]))
        if params[PARAM_STEM] == CONV_STEM:
            overrides.append(
                format_override(
                    f"{enc}.stem_kernel_size", self.conv_stem_kernel_size, force_add=True
                )
            )
        return tuple(overrides)

    def static_features(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Compute training-free features of a parameter set.

        Args:
            params: Normalized parameter values.

        Returns:
            Estimated encoder parameter count, head geometry, and rotary
            coverage, all JSON-serializable.
        """
        width = WIDTH_PRESETS[params[PARAM_WIDTH]]
        depths = encoder_depths(params[PARAM_CONV_DEPTH], params[PARAM_ATTN_DEPTH])
        return {
            "encoder_params": estimate_encoder_parameters(
                width.enc_channels,
                depths,
                params[PARAM_MLP_RATIO],
                stem_type=params[PARAM_STEM],
                stem_kernel_size=self.conv_stem_kernel_size,
                in_channels=self.in_channels,
            ),
            "enc_channels": list(width.enc_channels),
            "enc_depths": list(depths),
            "head_dim": width.head_dim,
            "rope_span": width.rope_span,
            "rope_frequencies_per_axis": rope_frequencies_per_axis(width.head_dim),
        }


def _check_names(names: Sequence[str], table: Mapping[str, Any], kind: str) -> list[str]:
    """Return ``names`` after checking each is a key of ``table``."""
    if not names:
        raise ValueError(f"At least one {kind} preset is required.")
    unknown = [name for name in names if name not in table]
    if unknown:
        raise ValueError(f"Unknown {kind} preset(s) {unknown}; known: {sorted(table)}.")
    return [str(name) for name in names]
