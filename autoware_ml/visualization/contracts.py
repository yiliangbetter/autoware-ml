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

"""Shared visualization backend contracts and configuration types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from autoware_ml.visualization.events import VisualizationEvent


@dataclass(frozen=True)
class VisualizationSessionConfig:
    """Configure one visualization recording."""

    backend: Literal["rerun", "noop"] = "rerun"
    application_id: str = "autoware-ml"
    recording_id: str | None = None
    web_port: int = 9090
    grpc_port: int = 9876
    wait: bool = True
    server_memory_limit: str = "25%"
    timeline: str = "frame"


class VisualizationBackend(Protocol):
    """Interface implemented by visualization backends."""

    def set_step(self, step: int) -> None:
        """Advance the backend timeline to one integer step."""

    def log_event(self, event: VisualizationEvent) -> None:
        """Log one visualization event."""

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log multiple visualization events."""
