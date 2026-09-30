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

"""Tests for the bench client: window, queue policy, variant files, aggregation."""

from __future__ import annotations

import datetime as dt
import json
import shlex
import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from autoware_ml.tuning.bench import (
    ML_PACKAGE_PLACEHOLDER,
    NODE_PARAM_NAME,
    VARIANT_TEMPLATE_NAME,
    BenchError,
    BenchSettings,
    BenchWindow,
    QueuePolicy,
    SshBenchRunner,
    VariantTemplate,
    aggregate_results,
    stage_package_files,
)

TOKYO = ZoneInfo("Asia/Tokyo")


def _at(hour: int, minute: int = 0, day: int = 30) -> dt.datetime:
    return dt.datetime(2026, 9, day, hour, minute, tzinfo=TOKYO)


def test_window_wraps_past_midnight() -> None:
    window = BenchWindow.from_config({"start": "18:00", "end": "09:00", "timezone": "Asia/Tokyo"})
    assert window is not None
    assert window.contains(_at(18, 0))
    assert window.contains(_at(23, 59))
    assert window.contains(_at(3, 0))
    assert window.contains(_at(8, 59))
    assert not window.contains(_at(9, 0))
    assert not window.contains(_at(12, 0))
    assert not window.contains(_at(17, 59))
    assert window.closes_at(_at(20, 0)) == dt.datetime(2026, 10, 1, 9, 0, tzinfo=TOKYO)
    assert window.opens_at(_at(12, 0)) == _at(18, 0)
    assert window.opens_at(_at(20, 0)) == _at(20, 0)
    assert window.remaining(_at(7, 0)) == dt.timedelta(hours=2)
    assert window.remaining(_at(12, 0)) == dt.timedelta(0)
    # A naive time is read in the window's zone; another zone converts.
    assert window.contains(dt.datetime(2026, 9, 30, 19, 0))
    assert window.contains(dt.datetime(2026, 9, 30, 10, 0, tzinfo=dt.timezone.utc))
    assert window.describe() == "18:00-09:00 Asia/Tokyo"
    assert BenchWindow.from_config(None) is None


def test_window_within_one_day() -> None:
    window = BenchWindow(dt.time(9, 0), dt.time(17, 0), TOKYO)
    assert window.contains(_at(12, 0))
    assert not window.contains(_at(18, 0))
    assert window.remaining(_at(16, 30)) == dt.timedelta(minutes=30)


def test_queue_policy_fills_the_queue_before_the_window_closes() -> None:
    window = BenchWindow(dt.time(18, 0), dt.time(9, 0), TOKYO)
    policy = QueuePolicy.from_config(
        {"target": 3, "target_before_close": 10, "close_buffer_hours": 2}
    )
    assert policy.target_at(window, _at(20, 0)) == 3
    assert policy.target_at(window, _at(7, 30)) == 10
    assert policy.target_at(window, _at(12, 0)) == 0
    assert policy.target_at(None, _at(12, 0)) == 3


def test_variant_template_renders_valid_toml() -> None:
    template = VariantTemplate.from_config(
        {
            "prefix": "lt2",
            "adapter": "v12",
            "sources": {
                "autoware_universe": "autowarefoundation/autoware_universe@abc",
                "autoware_core": {"url": "autowarefoundation/autoware_core", "ref": "def"},
            },
        }
    )
    assert template.variant_name(7) == "lt2-t0007"
    text = template.render("lt2-t0007", 'trial 7: width="tiny"', ML_PACKAGE_PLACEHOLDER)
    parsed = tomllib.loads(text)
    assert parsed["variant"] == {
        "name": "lt2-t0007",
        "kind": "model",
        "description": 'trial 7: width="tiny"',
    }
    model = parsed["model"]
    assert model["model"] == "ptv3" and model["adapter"] == "v12"
    assert model["ml_package"] == ML_PACKAGE_PLACEHOLDER
    assert model["node_param"] == NODE_PARAM_NAME
    assert model["params"] == ["ml_package_ptv3.param.yaml"]
    assert model["vars"] == {
        "model_variant": "ptv3",
        "use_seg3d_head": "true",
        "use_det3d_head": "false",
    }
    assert model["sources"]["autoware_universe"] == "autowarefoundation/autoware_universe@abc"
    assert model["sources"]["autoware_core"] == {
        "url": "autowarefoundation/autoware_core",
        "ref": "def",
    }
    assert model["warmup"] == 10 and model["iterations"] == 50
    assert parsed["nsys"]["trace"] == ["cuda", "nvtx", "osrt"]


def _write_result(path: Path, *, status: str, median: float, mean: float, voxels: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [{"gpu_total_ms": median + delta} for delta in (-0.5, 0.0, 0.5)]
    payload = {
        "status": status,
        "exit_code": 0 if status == "success" else 1,
        "provenance": {
            "host": {"gpu": "NVIDIA RTX 4000 SFF Ada Generation"},
            "image": {"tensorrt": "10.8.0.43"},
        },
        "benchmark": {
            "onnx_sha256": "onnx",
            "engine_sha256": "engine",
            "adapter": {"version": "v12"},
            "summary": {
                "gpu_total_ms": {"count": 3, "median": median, "mean": mean, "min": 0, "max": 0},
                "num_voxels": {"mean": voxels},
            },
            "iterations": iterations,
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_aggregate_results_pools_replicate_medians_and_skips_failures(tmp_path: Path) -> None:
    files = [
        _write_result(
            tmp_path / "s" / "replicate-00" / "result.json",
            status="success",
            median=13.0,
            mean=13.2,
            voxels=116000,
        ),
        _write_result(
            tmp_path / "s" / "replicate-01" / "result.json",
            status="success",
            median=14.0,
            mean=14.1,
            voxels=116000,
        ),
        _write_result(
            tmp_path / "s" / "replicate-02" / "result.json",
            status="failed",
            median=99.0,
            mean=99.0,
            voxels=0,
        ),
    ]
    measurement = aggregate_results(files, variant="lt2-t0001", series="s", ml_package="sha")
    assert measurement.latency_ms == pytest.approx(13.5)
    assert measurement.replicates == 2
    assert measurement.replicate_medians == [13.0, 14.0]
    assert measurement.ci95_ms == pytest.approx(1.96 * 0.7071067811865476 / 2**0.5)
    assert measurement.min_ms == pytest.approx(12.5) and measurement.max_ms == pytest.approx(14.5)
    assert measurement.num_voxels == pytest.approx(116000)
    assert measurement.gpu.startswith("NVIDIA") and measurement.tensorrt == "10.8.0.43"
    assert measurement.adapter == "v12" and measurement.engine_sha256 == "engine"
    assert measurement.result_dir == str(tmp_path / "s")
    assert aggregate_results(files, statistic="replicate_mean_mean").latency_ms == pytest.approx(
        13.65
    )
    assert aggregate_results(files, statistic="pooled_median").latency_ms == pytest.approx(13.5)
    assert set(measurement.to_dict()) >= {"latency_ms", "replicate_medians", "measured_at"}
    with pytest.raises(BenchError, match="No replicate succeeded"):
        aggregate_results(files[2:])
    with pytest.raises(BenchError, match="no summary"):
        aggregate_results(files[:1], metric="gpu_encoder_ms")


def _settings(tmp_path: Path, **overrides) -> BenchSettings:
    cfg = {
        "host": "10.0.0.1",
        "user": "bench",
        "ssh_key": str(tmp_path / "key"),
        "exchange": {"local": str(tmp_path / "exchange"), "remote": "/mnt/nas/exchange"},
        "harness": {
            "repo": "/mnt/nas/harness",
            "command": "export A=1 && uv run --frozen",
        },
        "variant": {"prefix": "lt2"},
        "reference": "litept_tiny",
        "window": {"start": "18:00", "end": "09:00"},
        "job": {"timeout_s": 10, "max_attempts": 2},
        "node_param_template": str(tmp_path / "node.yaml"),
    }
    cfg.update(overrides)
    settings = BenchSettings.from_config(cfg)
    assert settings is not None
    return settings


def test_settings_translate_paths_and_build_the_ssh_command(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.packages_dir == tmp_path / "exchange" / "packages"
    package = settings.packages_dir / "lt2-t0003"
    package.mkdir(parents=True)
    assert settings.remote_path(package) == "/mnt/nas/exchange/packages/lt2-t0003"
    assert settings.ssh_command()[-1] == "bench@10.0.0.1"
    assert "-i" in settings.ssh_command()
    assert settings.window is not None and settings.window.describe() == "18:00-09:00 Asia/Tokyo"
    assert settings.harness_state_dir == "/var/tmp/bench-state"
    assert BenchSettings.from_config(None) is None
    assert BenchSettings.from_config({"enabled": False, "host": "x"}) is None
    with pytest.raises(ValueError, match="statistic"):
        _settings(tmp_path, statistic="mode")


def test_runner_stages_the_package_and_quotes_the_remote_command(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (tmp_path / "node.yaml").write_text("/**:\n  ros__parameters: {}\n")
    package = settings.packages_dir / "lt2-t0003"
    package.mkdir(parents=True)
    stage_package_files(settings, "lt2-t0003", package, "trial 3")
    assert ML_PACKAGE_PLACEHOLDER in (package / VARIANT_TEMPLATE_NAME).read_text()
    assert (package / NODE_PARAM_NAME).read_text().startswith("/**:")
    runner = SshBenchRunner(settings, tmp_path / "logs")
    command = runner.build_remote_command("lt2-t0003", package, "tune-1")
    argv = shlex.split(command)
    assert argv[:3] == ["python3", "/mnt/nas/exchange/jobs/bench_job.py", "run"]
    assert argv[argv.index("--command") + 1] == "export A=1 && uv run --frozen"
    assert argv[argv.index("--package") + 1] == "/mnt/nas/exchange/packages/lt2-t0003"
    assert argv[argv.index("--results") + 1] == "/mnt/nas/exchange/results/lt2-t0003"
    assert argv[argv.index("--replicates") + 1] == "7"
    assert "--keep-variant" not in argv
    assert (tmp_path / "exchange" / "jobs" / "bench_job.py").is_file()
