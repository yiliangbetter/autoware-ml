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

"""Eager PyTorch latency benchmark.

This is a training-GPU proxy for the deployed TensorRT latency on the target
ECU: absolute numbers differ, but they rank candidate architectures the
same way, which is what the sampler needs. Winners still get measured on
the vehicle hardware.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from time import perf_counter
from typing import Any

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch

PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class LatencyReport:
    """Latency statistics over the timed frames.

    Attributes:
        mean_ms: Mean of the per-frame latencies.
        median_ms: Median of the per-frame latencies.
        p95_ms: 95th percentile of the per-frame latencies.
        std_ms: Standard deviation of the per-frame latencies.
        min_ms: Fastest frame.
        max_ms: Slowest frame.
        frames: Number of timed frames.
        repeats: Timed repetitions per frame; each frame reports its median.
        warmup: Untimed frames run first.
        precision: Autocast precision used.
        device: Device the benchmark ran on.
        voxels_per_stage: Mean point count per encoder stage over the warmup
            frames, when a stage counter was given.
    """

    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    frames: int
    repeats: int
    warmup: int
    precision: str
    device: str
    voxels_per_stage: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of ``values``."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _time_call(function: Callable[[], Any], use_cuda: bool) -> float:
    """Run ``function`` once and return its wall time in milliseconds."""
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    started = perf_counter()
    function()
    return (perf_counter() - started) * 1000.0


def benchmark_latency(
    forward: Callable[[ModelGTBatch], Any],
    batches: Iterable[ModelGTBatch],
    *,
    device: torch.device,
    precision: str = "fp32",
    frames: int = 50,
    warmup: int = 10,
    repeats: int = 5,
    stage_counter: Callable[[ModelGTBatch], Sequence[int]] | None = None,
) -> LatencyReport:
    """Time ``forward`` over single-frame batches.

    Args:
        forward: Callable running the model on one batch already on ``device``.
        batches: Batches as the dataloader yields them; the first ``warmup``
            are untimed, the next ``frames`` are timed.
        device: Device to move batches to.
        precision: ``fp32`` runs the model as trained; ``fp16`` autocasts on
            CUDA, which spconv's implicit-GEMM tuner rejected for the full
            LitePT graph on the environment this was developed on, so it is
            opt-in.
        frames: Timed frame count.
        warmup: Untimed warmup frame count.
        repeats: Timed repetitions per frame; the frame reports its median so
            a single scheduler hiccup does not skew it.
        stage_counter: Optional callable returning per-stage point counts for
            a batch, evaluated on the warmup frames only.

    Returns:
        The report.

    Raises:
        ValueError: On a bad precision or an empty timed set.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}.")
    if frames <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("frames and repeats must be positive and warmup non-negative.")
    use_cuda = device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda and precision == "fp16"
        else nullcontext()
    )
    per_frame: list[float] = []
    stage_counts: list[Sequence[int]] = []
    with torch.inference_mode(), autocast:
        for index, batch in enumerate(islice(batches, warmup + frames)):
            batch = batch.to_device(device)
            if index < warmup:
                forward(batch)
                if stage_counter is not None:
                    stage_counts.append(list(stage_counter(batch)))
                continue
            timings = [_time_call(lambda: forward(batch), use_cuda) for _ in range(repeats)]
            per_frame.append(statistics.median(timings))
    if not per_frame:
        raise ValueError("The dataloader yielded no frames to time after warmup.")
    voxels_per_stage = None
    if stage_counts:
        voxels_per_stage = [statistics.fmean(counts) for counts in zip(*stage_counts, strict=True)]
    return LatencyReport(
        mean_ms=statistics.fmean(per_frame),
        median_ms=statistics.median(per_frame),
        p95_ms=_percentile(per_frame, 0.95),
        std_ms=statistics.pstdev(per_frame) if len(per_frame) > 1 else 0.0,
        min_ms=min(per_frame),
        max_ms=max(per_frame),
        frames=len(per_frame),
        repeats=repeats,
        warmup=warmup,
        precision=precision,
        device=str(device),
        voxels_per_stage=voxels_per_stage,
    )
