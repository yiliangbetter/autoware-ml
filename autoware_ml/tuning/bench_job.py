#!/usr/bin/env python3
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

"""Measure one exported package on the benchmark host.

This script runs on the bench under its system Python and imports nothing
outside the standard library: the bench has no autoware-ml environment. The
screen phase copies it into the exchange directory and invokes it over SSH:

    python3 bench_job.py run --repo <harness checkout> --state <state dir>
        --command '<shell prefix ending in "uv run --frozen">'
        --package <package dir> --name <variant> --tag <series>
        --replicates 7 --results <output dir> [--keep-variant]

The package directory holds the ONNX files, the node and ML-package parameter
files and ``variant.template.toml``. The job ingests the package into the
harness's ML store, writes the variant, runs the harness, copies the results
into the output directory, removes what it created on the bench and prints
one JSON line describing the outcome as its last line of output.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

PLACEHOLDER = "__ML_PACKAGE__"
TEMPLATE_NAME = "variant.template.toml"
INGEST_HELPER = "_ml_store_helper.py"

HELPER_SOURCE = """
import json, sys
from pathlib import Path
from pa_tools import ml_store

verb = sys.argv[1]
if verb == "ingest":
    asset = ml_store.ingest(Path(sys.argv[2]), sys.argv[3])
    print(json.dumps({"sha256": asset.sha256, "label": asset.label, "root": str(ml_store.store_root())}))
elif verb == "forget":
    sha = sys.argv[2]
    root = ml_store.store_root()
    index_path = root / "index.json"
    removed = []
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        for label in [k for k, v in index.items() if v.get("sha256") == sha]:
            index.pop(label)
            removed.append(label)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\\n")
    print(json.dumps({"root": str(root), "removed": removed}))
"""


def log(message: str) -> None:
    print(f"[bench_job {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def run_shell(command: str, cwd: pathlib.Path, log_path: pathlib.Path | None = None) -> str:
    """Run ``command`` through bash in ``cwd`` and return its stdout."""
    log(f"$ {command}")
    with open(log_path, "ab") if log_path else open(os.devnull, "wb") as sink:
        sink.write(f"$ {command}\n".encode())
        proc = subprocess.run(
            ["bash", "-lc", command], cwd=str(cwd), stdout=subprocess.PIPE, stderr=sink, check=False
        )
        sink.write(proc.stdout)
    if proc.returncode != 0:
        tail = proc.stdout.decode(errors="replace").splitlines()[-20:]
        if log_path is not None:
            tail += tail_file(log_path, 30).splitlines()
        raise RuntimeError(f"command exited with {proc.returncode}: {command}\n" + "\n".join(tail))
    return proc.stdout.decode(errors="replace")


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("no JSON line in helper output")


def tail_file(path: pathlib.Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


class Job:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = pathlib.Path(args.repo)
        self.state = pathlib.Path(args.state)
        self.command = args.command
        self.package = pathlib.Path(args.package)
        self.name = args.name
        self.tag = args.tag
        self.replicates = args.replicates
        self.results = pathlib.Path(args.results)
        self.keep_variant = args.keep_variant
        self.sha: str | None = None
        self.store_root: pathlib.Path | None = None
        self.stage = "start"
        self.results.mkdir(parents=True, exist_ok=True)
        self.job_log = self.results / f"{self.tag}.job.log"

    def helper(self, *argv: str) -> dict:
        # The helper lives beside the results, never inside the package: every
        # file in the package is part of its content hash.
        helper = self.results / INGEST_HELPER
        helper.write_text(HELPER_SOURCE)
        quoted = " ".join(f"'{item}'" for item in (str(helper), *argv))
        return last_json(run_shell(f"{self.command} python {quoted}", self.repo, self.job_log))

    def ingest(self) -> None:
        self.stage = "ingest"
        report = self.helper("ingest", str(self.package), self.name)
        self.sha = report["sha256"]
        self.store_root = pathlib.Path(report["root"])
        log(f"ingested {self.package} as {self.sha[:12]} ({report['label']})")

    def write_variant(self) -> None:
        self.stage = "variant"
        template = (self.package / TEMPLATE_NAME).read_text()
        assert self.sha is not None
        variant_dir = self.repo / "variants" / self.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "variant.toml").write_text(template.replace(PLACEHOLDER, self.sha))
        log(f"wrote {variant_dir / 'variant.toml'}")

    def run_bench(self) -> None:
        self.stage = "bench"
        run_shell(
            f"{self.command} bench run '{self.name}' --replicates {self.replicates} --tag '{self.tag}'",
            self.repo,
            self.job_log,
        )

    def collect(self) -> list[str]:
        self.stage = "collect"
        variant_state = self.state / "variants" / self.name
        series_dir = variant_state / "results" / self.tag
        if not series_dir.is_dir():
            raise RuntimeError(f"no results under {series_dir}")
        target = self.results / self.tag
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(series_dir, target)
        for extra in ("engines/engines.log", "image.json"):
            source = variant_state / extra
            if source.is_file():
                shutil.copyfile(source, target / source.name)
        copied = sorted(str(p) for p in target.glob("replicate-*/result.json"))
        log(f"collected {len(copied)} replicate result(s) into {target}")
        return copied

    def cleanup(self) -> None:
        self.stage = "cleanup"
        variant_state = self.state / "variants" / self.name
        if variant_state.exists():
            shutil.rmtree(variant_state, ignore_errors=True)
        if self.sha and self.store_root:
            for path in self.store_root.glob(f"work/{self.sha}-*"):
                shutil.rmtree(path, ignore_errors=True)
            shutil.rmtree(self.store_root / "objects" / self.sha, ignore_errors=True)
            try:
                self.helper("forget", self.sha)
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                log(f"could not drop the store index entry: {exc}")
        if not self.keep_variant:
            shutil.rmtree(self.repo / "variants" / self.name, ignore_errors=True)
        helper = self.results / INGEST_HELPER
        if helper.exists():
            helper.unlink()

    def failure_kind(self) -> str:
        """Classify a failure: ``engine_build`` when the harness could not build the engines.

        An engine that does not build on the target at the deployed profile
        means the architecture cannot be deployed, which the study treats as a
        verdict rather than as something to retry.
        """
        variant_state = self.state / "variants" / self.name
        engines_log = variant_state / "engines" / "engines.log"
        if self.stage == "bench" and engines_log.is_file():
            text = engines_log.read_text(errors="replace")
            if not (variant_state / "results" / self.tag).is_dir() and (
                "Failed to setup" in text or "Error Code" in text
            ):
                return "engine_build"
        return self.stage

    def failure_detail(self, exc: Exception) -> str:
        """Summarize a failure, keeping the full harness logs beside the results."""
        parts = [str(exc)]
        variant_state = self.state / "variants" / self.name
        keep = self.results / f"{self.tag}-failed"
        keep.mkdir(parents=True, exist_ok=True)
        for candidate in (
            variant_state / "engines" / "engines.log",
            *sorted(variant_state.glob(f"results/{self.tag}/replicate-*/run.log")),
        ):
            if not candidate.is_file():
                continue
            shutil.copyfile(candidate, keep / f"{candidate.parent.name}-{candidate.name}")
            tail = tail_file(candidate)
            if tail:
                parts.append(f"--- {candidate} (kept under {keep}):\n{tail}")
        return "\n".join(parts)


def cmd_run(args: argparse.Namespace) -> int:
    job = Job(args)
    started = time.time()
    report: dict = {"status": "failed", "variant": job.name, "series": job.tag}
    try:
        job.ingest()
        job.write_variant()
        job.run_bench()
        results = job.collect()
        report = {
            "status": "ok",
            "variant": job.name,
            "series": job.tag,
            "ml_package": job.sha,
            "results": results,
            "seconds": round(time.time() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        report.update(
            {
                "stage": job.stage,
                "kind": job.failure_kind(),
                "detail": job.failure_detail(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            job.cleanup()
        except Exception as exc:  # noqa: BLE001 - cleanup must not hide the outcome
            log(f"cleanup failed: {exc}")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    run = sub.add_parser("run", help="measure one package")
    run.add_argument("--repo", required=True, help="benchmark harness checkout on this host")
    run.add_argument("--state", required=True, help="harness state directory on this host")
    run.add_argument("--command", required=True, help="shell prefix ending in the uv run command")
    run.add_argument("--package", required=True, help="package directory (ONNX + parameter files)")
    run.add_argument("--name", required=True, help="variant name")
    run.add_argument("--tag", required=True, help="series tag")
    run.add_argument("--replicates", type=int, default=7)
    run.add_argument("--results", required=True, help="directory receiving the series results")
    run.add_argument("--keep-variant", action="store_true", help="leave the variant in the repo")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
