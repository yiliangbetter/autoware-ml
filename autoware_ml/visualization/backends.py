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

"""Visualization backend abstractions and factory helpers."""

from __future__ import annotations

from collections.abc import Iterable

from autoware_ml.visualization.contracts import VisualizationBackend, VisualizationSessionConfig
from autoware_ml.visualization.events import VisualizationEvent
from autoware_ml.visualization.rerun_backend import RerunVisualizationBackend


class NoOpVisualizationBackend:
    """Drop all events while keeping the public backend API stable."""

    def set_step(self, step: int) -> None:
        """Ignore timeline updates."""
        del step

    def log_event(self, event: VisualizationEvent) -> None:
        """Ignore one visualization event."""
        del event

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Ignore multiple visualization events."""
        for _ in events:
            continue


def create_visualization_backend(
    config: VisualizationSessionConfig,
) -> VisualizationBackend:
    """Create a concrete visualization backend from configuration."""
    if config.backend == "noop":
        return NoOpVisualizationBackend()
    if config.backend == "rerun":
        return RerunVisualizationBackend(config)
    raise ValueError(f"Unknown visualization backend: {config.backend}")
