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

"""Tests for the eager latency benchmark."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.tuning.latency import benchmark_latency, forward_inputs


def _batches(count: int):
    for index in range(count):
        yield {"feat": torch.full((4, 3), float(index)), "extra": index}


def test_benchmark_times_only_the_frames_after_warmup() -> None:
    seen: list[float] = []

    def forward(batch):
        seen.append(float(batch["feat"][0, 0]))
        return batch["feat"].sum()

    report = benchmark_latency(
        forward,
        _batches(10),
        device=torch.device("cpu"),
        precision="fp32",
        frames=3,
        warmup=2,
        repeats=2,
        stage_counter=lambda batch: [int(batch["feat"].shape[0]), 2],
    )
    # Two warmup frames once each, then three frames twice each; the rest untouched.
    assert seen == [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert report.frames == 3
    assert report.repeats == 2
    assert report.warmup == 2
    assert report.voxels_per_stage == [4.0, 2.0]
    assert report.min_ms <= report.median_ms <= report.max_ms
    assert report.mean_ms > 0.0
    assert report.to_dict()["precision"] == "fp32"


def test_benchmark_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="precision"):
        benchmark_latency(
            lambda batch: None, _batches(3), device=torch.device("cpu"), precision="int8"
        )
    with pytest.raises(ValueError, match="no frames"):
        benchmark_latency(
            lambda batch: None, _batches(1), device=torch.device("cpu"), precision="fp32", warmup=1
        )


def test_forward_inputs_follows_the_forward_signature() -> None:
    class Model(torch.nn.Module):
        def forward(self, feat, offset):
            return feat, offset

    batch = {"feat": 1, "offset": 2, "segment": 3}
    assert forward_inputs(Model(), batch) == {"feat": 1, "offset": 2}
