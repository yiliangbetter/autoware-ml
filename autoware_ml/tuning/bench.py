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

"""Deployed-latency screening on the benchmark host.

A TensorRT engine's latency depends on the graph and the input cloud, not on
the weight values, so an architecture can be timed on the target hardware
before it is trained: export it with its initial weights, ship the ONNX to
the bench, build the engines there and run the inference harness. A trial
that misses the latency bound never spends a training slot.

The bench is shared and only ours during a nightly window, so every call
that touches it goes through :class:`BenchWindow`. Files travel over a NAS
directory both machines mount (the exchange): packages in, results out. SSH
carries only the control command, which runs :mod:`bench_job` on the bench.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import shlex
import shutil
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

JOB_SCRIPT = Path(__file__).with_name("bench_job.py")
PACKAGES_DIR = "packages"
RESULTS_DIR = "results"
JOBS_DIR = "jobs"
VARIANT_TEMPLATE_NAME = "variant.template.toml"
ML_PACKAGE_PARAM_NAME = "ml_package_ptv3.param.yaml"
NODE_PARAM_NAME = "ptv3.param.yaml"
STATISTICS = ("replicate_median_mean", "replicate_mean_mean", "pooled_median")
SSH_CONNECTION_FAILURE = 255


class BenchError(RuntimeError):
    """A bench measurement could not be completed."""


ENGINE_BUILD_FAILURE = "engine_build"


class BenchJobError(BenchError):
    """The bench-side job reported a failure.

    Attributes:
        stage: Job stage that failed.
        detail: What the job reported, with the harness log tails.
        kind: Failure class; ``engine_build`` means the engines did not build
            on the target, which is a verdict on the architecture.
    """

    def __init__(self, stage: str, detail: str, kind: str | None = None) -> None:
        super().__init__(f"bench job failed at {stage}: {detail}")
        self.stage = stage
        self.detail = detail
        self.kind = kind or stage

    @property
    def unbuildable(self) -> bool:
        """Whether the architecture's engines could not be built on the target."""
        return self.kind == ENGINE_BUILD_FAILURE


def _parse_time(value: str) -> dt.time:
    hours, minutes = (int(part) for part in str(value).split(":")[:2])
    return dt.time(hours, minutes)


@dataclass(frozen=True)
class BenchWindow:
    """The daily interval during which the bench may be used.

    An interval whose end is not after its start wraps past midnight, so
    ``18:00`` to ``09:00`` is the evening and the following morning.

    Attributes:
        start: Opening time of day.
        end: Closing time of day.
        zone: Time zone both times are expressed in.
    """

    start: dt.time
    end: dt.time
    zone: ZoneInfo

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> BenchWindow | None:
        """Build a window from ``{start, end, timezone}``; ``None`` means always open."""
        if not cfg:
            return None
        return cls(
            _parse_time(cfg["start"]),
            _parse_time(cfg["end"]),
            ZoneInfo(str(cfg.get("timezone", "Asia/Tokyo"))),
        )

    def now(self) -> dt.datetime:
        """Return the current time in the window's zone."""
        return dt.datetime.now(self.zone)

    def _localize(self, at: dt.datetime) -> dt.datetime:
        return at.astimezone(self.zone) if at.tzinfo is not None else at.replace(tzinfo=self.zone)

    def contains(self, at: dt.datetime) -> bool:
        """Whether ``at`` falls inside the window."""
        local = self._localize(at).time().replace(tzinfo=None)
        if self.start < self.end:
            return self.start <= local < self.end
        return local >= self.start or local < self.end

    def closes_at(self, at: dt.datetime) -> dt.datetime:
        """Return the next closing instant at or after ``at``."""
        local = self._localize(at)
        close = local.replace(hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0)
        if close <= local:
            close += dt.timedelta(days=1)
        return close

    def opens_at(self, at: dt.datetime) -> dt.datetime:
        """Return the next opening instant at or after ``at``."""
        local = self._localize(at)
        if self.contains(local):
            return local
        opening = local.replace(
            hour=self.start.hour, minute=self.start.minute, second=0, microsecond=0
        )
        if opening <= local:
            opening += dt.timedelta(days=1)
        return opening

    def remaining(self, at: dt.datetime) -> dt.timedelta:
        """Return how long the window stays open after ``at``; zero when closed."""
        local = self._localize(at)
        if not self.contains(local):
            return dt.timedelta(0)
        return self.closes_at(local) - local

    def describe(self) -> str:
        """Render the window for logs."""
        return f"{self.start:%H:%M}-{self.end:%H:%M} {self.zone.key}"


@dataclass(frozen=True)
class QueuePolicy:
    """How many latency-feasible trials to keep waiting for the trainer.

    The trainer runs around the clock while the bench is only open at night,
    so the queue is kept short while the bench is available (every proposal
    then sees the freshest results) and filled up shortly before the window
    closes, enough to keep the trainer busy until it reopens.

    Attributes:
        target: Queue length to maintain while the window is open.
        target_before_close: Queue length to reach before the window closes.
        close_buffer: How long before closing the larger target applies.
    """

    target: int = 3
    target_before_close: int = 10
    close_buffer: dt.timedelta = dt.timedelta(hours=2, minutes=30)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> QueuePolicy:
        """Build the policy from ``{target, target_before_close, close_buffer_hours}``."""
        cfg = cfg or {}
        return cls(
            int(cfg.get("target", 3)),
            int(cfg.get("target_before_close", 10)),
            dt.timedelta(hours=float(cfg.get("close_buffer_hours", 2.5))),
        )

    def target_at(self, window: BenchWindow | None, at: dt.datetime) -> int:
        """Return the queue target that applies at ``at``."""
        if window is None:
            return self.target
        remaining = window.remaining(at)
        if remaining <= dt.timedelta(0):
            return 0
        if remaining <= self.close_buffer:
            return max(self.target, self.target_before_close)
        return self.target


def _toml_string(value: str) -> str:
    return json.dumps(str(value))


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    return _toml_string(value)


@dataclass(frozen=True)
class VariantTemplate:
    """What every trial's harness variant has in common.

    The fields mirror the harness's ``variant.toml`` for a model-kind variant.
    Only the ML package differs between trials, so the template renders with
    a placeholder the bench-side job fills in once the package is ingested.

    Attributes:
        prefix: Variant name prefix; trial numbers are appended.
        firmware: Harness firmware profile, the base image.
        adapter: Benchmark adapter generation matching the universe ref.
        sources: Overlay sources, ``autoware_universe`` first.
        mcap: Input cloud, relative to the harness repository.
        node_param: Node parameter file: a file inside the package, or
            ``<source>:<path>`` inside an overlay source tree.
        params: ML-package parameter files inside the package.
        vars: Launch variables the node parameter file refers to.
        warmup: Untimed inference cycles per replicate.
        iterations: Timed inference cycles per replicate.
        args: Extra benchmark binary arguments.
        nsys_trace: Trace domains for ``--profile`` runs.
    """

    prefix: str = "tune"
    firmware: str = "oss-universe-devel-cuda-20260902"
    adapter: str = "v12"
    sources: Mapping[str, Any] = field(default_factory=dict)
    mcap: str = "assets/mcaps/sample_pointcloud.mcap"
    node_param: str = NODE_PARAM_NAME
    params: tuple[str, ...] = (ML_PACKAGE_PARAM_NAME,)
    vars: Mapping[str, str] = field(
        default_factory=lambda: {
            "model_variant": "ptv3",
            "use_seg3d_head": "true",
            "use_det3d_head": "false",
        }
    )
    warmup: int = 10
    iterations: int = 50
    args: tuple[str, ...] = ()
    nsys_trace: tuple[str, ...] = ("cuda", "nvtx", "osrt")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> VariantTemplate:
        """Build the template from the ``bench.variant`` block."""
        cfg = dict(cfg or {})
        defaults = cls()
        return cls(
            prefix=str(cfg.get("prefix", defaults.prefix)),
            firmware=str(cfg.get("firmware", defaults.firmware)),
            adapter=str(cfg.get("adapter", defaults.adapter)),
            sources=dict(cfg.get("sources") or {}),
            mcap=str(cfg.get("mcap", defaults.mcap)),
            node_param=str(cfg.get("node_param", defaults.node_param)),
            params=tuple(str(item) for item in cfg.get("params") or defaults.params),
            vars={str(k): str(v) for k, v in (cfg.get("vars") or defaults.vars).items()},
            warmup=int(cfg.get("warmup", defaults.warmup)),
            iterations=int(cfg.get("iterations", defaults.iterations)),
            args=tuple(str(item) for item in cfg.get("args") or ()),
            nsys_trace=tuple(str(item) for item in cfg.get("nsys_trace") or defaults.nsys_trace),
        )

    def variant_name(self, trial_number: int) -> str:
        """Return the harness variant name of a trial."""
        return f"{self.prefix}-t{trial_number:04d}"

    def render(self, name: str, description: str, ml_package: str) -> str:
        """Render the variant file for one package.

        Args:
            name: Variant name.
            description: One-line description recorded in the file.
            ml_package: ML store handle, or a placeholder the job replaces.

        Returns:
            TOML text.
        """
        lines = [
            f"# Variant: {name}",
            "# Written by autoware-ml tune screen; the ML package carries random weights",
            "# because only the latency of this architecture is measured.",
            "",
            "[variant]",
            f"name = {_toml_string(name)}",
            'kind = "model"',
            f"description = {_toml_string(description)}",
            "",
            "[model]",
            'model = "ptv3"',
            f"adapter = {_toml_string(self.adapter)}",
            f"firmware = {_toml_string(self.firmware)}",
            f"ml_package = {_toml_string(ml_package)}",
            f"node_param = {_toml_string(self.node_param)}",
            f"params = {_toml_value(list(self.params))}",
            f"mcap = {_toml_string(self.mcap)}",
            f"warmup = {self.warmup}",
            f"iterations = {self.iterations}",
            f"args = {_toml_value(list(self.args))}",
            "",
            "[model.vars]",
            *[f"{key} = {_toml_string(value)}" for key, value in self.vars.items()],
        ]
        if self.sources:
            lines += ["", "[model.sources]"]
            lines += [f"{key} = {_toml_value(value)}" for key, value in self.sources.items()]
        lines += [
            "",
            "[nsys]",
            "enabled = true",
            f"trace = {_toml_value(list(self.nsys_trace))}",
            "",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BenchSettings:
    """Everything the screen phase needs to know about the bench.

    Attributes:
        host: SSH host or address of the bench.
        user: SSH user on the bench.
        ssh_key: Private key file, if not the default identity.
        ssh_options: Extra ``ssh`` options.
        exchange_local: Exchange directory as this machine mounts it.
        exchange_remote: The same directory as the bench mounts it.
        harness_repo: Benchmark harness checkout on the bench.
        harness_state_dir: Harness state directory on the bench.
        harness_command: Shell prefix that runs harness commands on the
            bench, ending with the ``uv run`` that precedes ``bench``.
        remote_python: Python interpreter running the job on the bench.
        variant: The variant template.
        replicates: Replicates per measurement, each a fresh container.
        metric: Harness metric that is the trial's latency.
        statistic: How replicates pool into one number.
        reference: Name of the reference trial anchoring the latency bound.
        node_param_template: Node parameter file copied into every package,
            or ``None`` to use the one the variant names.
        window: Nightly window, or ``None`` for always open.
        queue: Queue policy for the trainer.
        timeout_s: Wall-clock limit for one bench job.
        max_attempts: Bench attempts per trial before it is failed.
        keep_variants: Leave the variant file in the harness repository.
        keep_packages: Leave the package in the exchange after measuring.
        max_ci95_ms: Widest replicate confidence interval accepted; a wider
            one means the bench was contended and the measurement is retried.
        unbuildable_violation_ms: Latency violation recorded for an
            architecture whose engines do not build on the target, so the
            sampler treats it as far outside the bound.
    """

    host: str
    user: str
    ssh_key: str | None
    ssh_options: tuple[str, ...]
    exchange_local: Path
    exchange_remote: str
    harness_repo: str
    harness_state_dir: str
    harness_command: str
    remote_python: str
    variant: VariantTemplate
    replicates: int
    metric: str
    statistic: str
    reference: str
    node_param_template: Path | None
    window: BenchWindow | None
    queue: QueuePolicy
    timeout_s: int
    max_attempts: int
    keep_variants: bool
    keep_packages: bool
    max_ci95_ms: float | None = None
    unbuildable_violation_ms: float = 100.0

    @classmethod
    def from_config(cls, cfg: DictConfig | Mapping[str, Any] | None) -> BenchSettings | None:
        """Parse the ``tuning.bench`` block; ``None`` or ``enabled: false`` disables it."""
        if cfg is None:
            return None
        plain = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
        assert isinstance(plain, Mapping)
        if not plain.get("enabled", True):
            return None
        exchange = plain.get("exchange") or {}
        harness = plain.get("harness") or {}
        job = plain.get("job") or {}
        statistic = str(plain.get("statistic", "replicate_median_mean"))
        if statistic not in STATISTICS:
            raise ValueError(f"bench.statistic must be one of {STATISTICS}, got {statistic!r}.")
        template = plain.get("node_param_template")
        ssh_key = plain.get("ssh_key")
        return cls(
            host=str(plain["host"]),
            user=str(plain.get("user", "")),
            ssh_key=None if ssh_key is None else str(Path(str(ssh_key)).expanduser()),
            ssh_options=tuple(str(item) for item in plain.get("ssh_options") or ()),
            exchange_local=Path(str(exchange["local"])).expanduser(),
            exchange_remote=str(exchange["remote"]),
            harness_repo=str(harness["repo"]),
            harness_state_dir=str(harness.get("state_dir", "/var/tmp/bench-state")),
            harness_command=str(harness["command"]),
            remote_python=str(harness.get("python", "python3")),
            variant=VariantTemplate.from_config(plain.get("variant")),
            replicates=int(plain.get("replicates", 7)),
            metric=str(plain.get("metric", "gpu_total_ms")),
            statistic=statistic,
            reference=str(plain["reference"]),
            node_param_template=None if template is None else Path(str(template)).expanduser(),
            window=BenchWindow.from_config(plain.get("window")),
            queue=QueuePolicy.from_config(plain.get("queue")),
            timeout_s=int(job.get("timeout_s", 3600)),
            max_attempts=int(job.get("max_attempts", 2)),
            keep_variants=bool(job.get("keep_variants", False)),
            keep_packages=bool(job.get("keep_packages", True)),
            max_ci95_ms=None if plain.get("max_ci95_ms") is None else float(plain["max_ci95_ms"]),
            unbuildable_violation_ms=float(plain.get("unbuildable_violation_ms", 100.0)),
        )

    @property
    def packages_dir(self) -> Path:
        """Local view of the package drop directory."""
        return self.exchange_local / PACKAGES_DIR

    @property
    def results_dir(self) -> Path:
        """Local view of the results directory."""
        return self.exchange_local / RESULTS_DIR

    def remote_path(self, local: Path) -> str:
        """Translate a path under the local exchange into the bench's view of it."""
        relative = local.resolve().relative_to(self.exchange_local.resolve())
        return f"{self.exchange_remote.rstrip('/')}/{relative.as_posix()}"

    def ssh_target(self) -> str:
        """Return the ``user@host`` SSH destination."""
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_command(self) -> list[str]:
        """Return the ``ssh`` invocation prefix."""
        command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.ssh_key:
            command += ["-i", self.ssh_key]
        command += list(self.ssh_options)
        command.append(self.ssh_target())
        return command


@dataclass(frozen=True)
class BenchMeasurement:
    """One trial's latency as measured on the bench.

    Attributes:
        latency_ms: The number the study uses, pooled per ``statistic``.
        metric: Harness metric it was read from.
        statistic: How the replicates were pooled.
        replicates: Number of successful replicates.
        replicate_medians: Per-replicate median of the metric.
        replicate_means: Per-replicate mean of the metric.
        ci95_ms: Half width of the 95% interval over replicate medians.
        min_ms: Fastest timed iteration over all replicates.
        max_ms: Slowest timed iteration over all replicates.
        iterations: Timed iterations per replicate.
        num_voxels: Mean input voxel count, when the adapter reports it.
        variant: Harness variant name.
        series: Harness series tag.
        result_dir: Local directory holding the replicate results.
        ml_package: ML store hash of the package that was measured.
        onnx_sha256: Encoder ONNX hash the harness recorded.
        engine_sha256: Encoder engine hash the harness recorded.
        adapter: Benchmark adapter version.
        gpu: GPU the bench reported.
        tensorrt: TensorRT version the image carried.
        measured_at: ISO timestamp of the aggregation.
    """

    latency_ms: float
    metric: str
    statistic: str
    replicates: int
    replicate_medians: list[float]
    replicate_means: list[float]
    ci95_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    num_voxels: float | None
    variant: str
    series: str
    result_dir: str
    ml_package: str
    onnx_sha256: str
    engine_sha256: str
    adapter: str
    gpu: str
    tensorrt: str
    measured_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _ci95_half_width(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate_results(
    result_files: Sequence[Path],
    metric: str = "gpu_total_ms",
    statistic: str = "replicate_median_mean",
    *,
    variant: str = "",
    series: str = "",
    ml_package: str = "",
) -> BenchMeasurement:
    """Pool the harness's per-replicate results into one latency.

    Mirrors the harness's own report: replicates are independent processes,
    iterations within one are not, so the number is the mean over replicate
    medians by default.

    Args:
        result_files: ``result.json`` of every replicate.
        metric: Metric to read from each replicate's summary.
        statistic: One of :data:`STATISTICS`.
        variant: Variant name recorded on the measurement.
        series: Series tag recorded on the measurement.
        ml_package: Package hash recorded on the measurement.

    Returns:
        The measurement.

    Raises:
        BenchError: If no replicate succeeded or the metric is missing.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}.")
    medians: list[float] = []
    means: list[float] = []
    pooled: list[float] = []
    voxels: list[float] = []
    iterations = 0
    onnx_sha = engine_sha = adapter = gpu = tensorrt = ""
    failures = []
    for path in sorted(result_files):
        result = json.loads(Path(path).read_text())
        if result.get("status") != "success":
            failures.append(f"{path}: {result.get('status')} (exit {result.get('exit_code')})")
            continue
        benchmark = result.get("benchmark") or {}
        summary = (benchmark.get("summary") or {}).get(metric)
        if summary is None:
            raise BenchError(f"{path} carries no summary for metric {metric!r}.")
        medians.append(float(summary["median"]))
        means.append(float(summary["mean"]))
        pooled.extend(float(item[metric]) for item in benchmark.get("iterations") or [])
        iterations = max(iterations, int(summary.get("count", 0)))
        voxel_summary = (benchmark.get("summary") or {}).get("num_voxels") or {}
        if voxel_summary.get("mean"):
            voxels.append(float(voxel_summary["mean"]))
        onnx_sha = onnx_sha or str(benchmark.get("onnx_sha256", ""))
        engine_sha = engine_sha or str(benchmark.get("engine_sha256", ""))
        adapter = adapter or str((benchmark.get("adapter") or {}).get("version", ""))
        provenance = result.get("provenance") or {}
        gpu = gpu or str((provenance.get("host") or {}).get("gpu", ""))
        tensorrt = tensorrt or str((provenance.get("image") or {}).get("tensorrt", ""))
    if not medians:
        raise BenchError("No replicate succeeded: " + "; ".join(failures or ["no result files"]))
    if statistic == "replicate_median_mean":
        latency = statistics.fmean(medians)
    elif statistic == "replicate_mean_mean":
        latency = statistics.fmean(means)
    else:
        latency = statistics.median(pooled) if pooled else statistics.fmean(medians)
    return BenchMeasurement(
        latency_ms=latency,
        metric=metric,
        statistic=statistic,
        replicates=len(medians),
        replicate_medians=medians,
        replicate_means=means,
        ci95_ms=_ci95_half_width(medians),
        min_ms=min(pooled) if pooled else min(medians),
        max_ms=max(pooled) if pooled else max(medians),
        iterations=iterations,
        num_voxels=statistics.fmean(voxels) if voxels else None,
        variant=variant,
        series=series,
        result_dir=str(Path(result_files[0]).parent.parent) if result_files else "",
        ml_package=ml_package,
        onnx_sha256=onnx_sha,
        engine_sha256=engine_sha,
        adapter=adapter,
        gpu=gpu,
        tensorrt=tensorrt,
        measured_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )


class BenchRunner(Protocol):
    """Measures one exported package on the bench."""

    def measure(self, name: str, package_dir: Path, description: str) -> BenchMeasurement:
        """Time the package and return its latency.

        Args:
            name: Harness variant name, unique per trial.
            package_dir: Package directory under the local exchange.
            description: One-line description recorded on the variant.
        """


ML_PACKAGE_PLACEHOLDER = "__ML_PACKAGE__"


def stage_package_files(
    settings: BenchSettings, name: str, package_dir: Path, description: str
) -> None:
    """Add the variant template and the node parameters to a package.

    Args:
        settings: Bench settings.
        name: Variant name.
        package_dir: The package directory.
        description: Description recorded on the variant.
    """
    template = settings.variant.render(name, description, ML_PACKAGE_PLACEHOLDER)
    (package_dir / VARIANT_TEMPLATE_NAME).write_text(template)
    if settings.node_param_template is not None:
        shutil.copyfile(settings.node_param_template, package_dir / NODE_PARAM_NAME)


class SshBenchRunner:
    """Runs the bench job over SSH and reads the results back from the exchange."""

    def __init__(self, settings: BenchSettings, log_dir: Path) -> None:
        """Initialize the runner.

        Args:
            settings: Bench settings.
            log_dir: Directory receiving one log per bench job.
        """
        self.settings = settings
        self.log_dir = log_dir

    def _stage_job_script(self) -> str:
        jobs_dir = self.settings.exchange_local / JOBS_DIR
        jobs_dir.mkdir(parents=True, exist_ok=True)
        target = jobs_dir / JOB_SCRIPT.name
        shutil.copyfile(JOB_SCRIPT, target)
        return self.settings.remote_path(target)

    def build_remote_command(self, name: str, package_dir: Path, series: str) -> str:
        """Assemble the shell command the bench executes for one measurement."""
        settings = self.settings
        argv = [
            settings.remote_python,
            self._stage_job_script(),
            "run",
            "--repo",
            settings.harness_repo,
            "--state",
            settings.harness_state_dir,
            "--command",
            settings.harness_command,
            "--package",
            settings.remote_path(package_dir),
            "--name",
            name,
            "--tag",
            series,
            "--replicates",
            str(settings.replicates),
            "--results",
            f"{settings.remote_path(settings.results_dir)}/{name}",
        ]
        if settings.keep_variants:
            argv.append("--keep-variant")
        return shlex.join(argv)

    def measure(self, name: str, package_dir: Path, description: str) -> BenchMeasurement:
        """Time the package on the bench, retrying lost SSH connections."""
        settings = self.settings
        stage_package_files(settings, name, package_dir, description)
        series = "tune-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        command = self.build_remote_command(name, package_dir, series)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"bench_{name}.log"
        attempts = 0
        while True:
            attempts += 1
            with log_path.open("a") as log:
                log.write(
                    f"# {dt.datetime.now().isoformat(timespec='seconds')} attempt {attempts}\n"
                )
                log.write(f"# {' '.join(settings.ssh_command())} {command}\n")
                log.flush()
                completed = subprocess.run(  # noqa: S603 - command assembled from the config
                    [*settings.ssh_command(), command],
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    timeout=settings.timeout_s,
                    check=False,
                )
                log.write(completed.stdout)
            if completed.returncode == SSH_CONNECTION_FAILURE and attempts < settings.max_attempts:
                logger.warning("SSH to %s failed; retrying in 60 s", settings.host)
                time.sleep(60)
                continue
            break
        report = _last_json_line(completed.stdout)
        if report is None:
            raise BenchJobError(
                "ssh", f"no report from the bench (exit {completed.returncode}), see {log_path}"
            )
        if report.get("status") != "ok":
            raise BenchJobError(
                str(report.get("stage", "?")),
                str(report.get("detail", "")),
                kind=None if report.get("kind") is None else str(report["kind"]),
            )
        result_dir = settings.results_dir / name / series
        result_files = sorted(result_dir.glob("replicate-*/result.json"))
        if not result_files:
            raise BenchJobError("results", f"no replicate results under {result_dir}")
        return aggregate_results(
            result_files,
            settings.metric,
            settings.statistic,
            variant=name,
            series=series,
            ml_package=str(report.get("ml_package", "")),
        )


def _last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
