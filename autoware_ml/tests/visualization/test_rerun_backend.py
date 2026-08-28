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

"""Tests for the Rerun visualization backend."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    AnnotationInfo,
    Boxes3DEvent,
    ImageEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextEvent,
    Transform3DEvent,
)
from autoware_ml.visualization.rerun_backend import (
    RerunVisualizationBackend,
    _patch_class_id_array_protocol,
    _verify_annotation_context_support,
)

_ANNOTATION_DESCRIPTOR = (
    "rerun.archetypes.AnnotationContext:rerun.components.AnnotationContext#context"
)


class _FakeArrowArray:
    """Expose only the ``to_pylist`` surface the compatibility probe uses."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def to_pylist(self) -> Any:
        """Return the recorded payload."""
        return self._payload


class _FakeComponentBatch:
    """Mimic one Rerun component batch."""

    def __init__(self, descriptor: str, payload: Any) -> None:
        self._descriptor = descriptor
        self._payload = payload

    def component_descriptor(self) -> str:
        """Return the component descriptor string."""
        return self._descriptor

    def as_arrow_array(self) -> _FakeArrowArray:
        """Return the payload wrapped as an arrow-like array."""
        return _FakeArrowArray(self._payload)


class _FakeAnnotationContext:
    """Mimic ``rr.AnnotationContext`` including its serialized payload."""

    #: When true, mimic the serialization failure that silently drops legends.
    serializes_empty = False

    def __init__(self, context: Any) -> None:
        self.context = context

    def as_component_batches(self) -> list[_FakeComponentBatch]:
        """Return an indicator batch plus the annotation-context batch."""
        payload: Any = (
            [[]]
            if type(self).serializes_empty
            else [[{"class_id": info["id"], "label": info["label"]} for info in self.context]]
        )
        return [
            _FakeComponentBatch("rerun.components.AnnotationContextIndicator", [None]),
            _FakeComponentBatch(_ANNOTATION_DESCRIPTOR, payload),
        ]


def _build_fake_rerun(calls: dict[str, Any]) -> Any:
    """Build a fake ``rerun`` module that records every call it receives."""

    class _FakeRR:
        AnnotationContext = _FakeAnnotationContext

        class TransformRelation:
            ChildFromParent = "ChildFromParent"

        @staticmethod
        def init(application_id: str, **kwargs: Any) -> None:
            calls["init"] = (application_id, kwargs)

        @staticmethod
        def serve_web(**kwargs: Any) -> None:
            calls["serve_web"] = kwargs

        @staticmethod
        def set_time(timeline: str, *, sequence: int) -> None:
            calls["steps"].append((timeline, sequence))

        @staticmethod
        def log(path: str, payload: Any, **kwargs: Any) -> None:
            calls["logs"].append((path, payload, kwargs))

        @staticmethod
        def AnnotationInfo(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        @staticmethod
        def Image(image: Any) -> tuple[str, Any]:
            return ("Image", image)

        @staticmethod
        def Points3D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points3D", args, kwargs)

        @staticmethod
        def Points2D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points2D", args, kwargs)

        @staticmethod
        def Boxes3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Boxes3D", kwargs)

        @staticmethod
        def Transform3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Transform3D", kwargs)

        @staticmethod
        def Pinhole(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Pinhole", kwargs)

        @staticmethod
        def Scalars(value: Any) -> tuple[str, Any]:
            return ("Scalars", value)

        @staticmethod
        def TextLog(text: str, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
            return ("TextLog", text, kwargs)

    return _FakeRR


@pytest.fixture
def rerun_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``rerun`` module and return its recorded calls."""
    calls: dict[str, Any] = {"init": None, "serve_web": None, "logs": [], "steps": []}
    monkeypatch.setattr(
        "autoware_ml.visualization.rerun_backend._load_rerun_module",
        lambda: _build_fake_rerun(calls),
    )
    return calls


@pytest.fixture
def backend(rerun_calls: dict[str, Any]) -> RerunVisualizationBackend:
    """Build one Rerun backend wired to the fake module."""
    return RerunVisualizationBackend(
        VisualizationSessionConfig(web_port=9091, grpc_port=9877, wait=False)
    )


def _logged(calls: dict[str, Any], path: str) -> list[Any]:
    """Return the payloads logged to one entity path."""
    return [payload for logged_path, payload, _ in calls["logs"] if logged_path == path]


def test_backend_initializes_and_serves_the_web_viewer(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    assert rerun_calls["init"][0] == "autoware-ml"
    assert rerun_calls["init"][1]["spawn"] is False
    assert rerun_calls["serve_web"] == {
        "open_browser": False,
        "web_port": 9091,
        "grpc_port": 9877,
        "server_memory_limit": "25%",
    }
    assert backend.web_url == (
        "http://localhost:9091?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9877%2Fproxy"
    )


def test_backend_forwards_timeline_steps(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.set_step(3)

    assert rerun_calls["steps"] == [("frame", 3)]


def test_backend_logs_annotation_context_statically(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    """Legends must be static so one context covers every frame on the timeline."""
    backend.log_event(
        AnnotationContextEvent(
            path="detection3d",
            annotations=[AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255))],
        )
    )

    path, payload, kwargs = rerun_calls["logs"][0]
    assert path == "detection3d"
    assert kwargs == {"static": True}
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert "AnnotationContext#" in str(payload[0].component_descriptor())


def test_backend_uses_the_supported_scalars_api(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    """``rr.Scalar`` is deprecated since rerun 0.23, so ``rr.Scalars`` must be used."""
    backend.log_event(ScalarEvent(path="detection3d/metrics/num_predictions", value=4.0))

    assert _logged(rerun_calls, "detection3d/metrics/num_predictions") == [("Scalars", 4.0)]


def test_backend_translates_every_supported_event(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_events(
        [
            ImageEvent(path="cameras/front", image=np.zeros((4, 4, 3), dtype=np.uint8)),
            PointCloud3DEvent(path="lidar/points", positions=np.zeros((2, 3), dtype=np.float32)),
            Points2DEvent(path="cameras/front/overlay", positions=np.zeros((2, 2), np.float32)),
            Boxes3DEvent(
                path="detection3d/prediction",
                centers=np.zeros((1, 3), dtype=np.float32),
                sizes=np.ones((1, 3), dtype=np.float32),
                yaws=np.zeros((1,), dtype=np.float32),
            ),
            Transform3DEvent(
                path="cameras/front",
                translation=np.zeros((3,), dtype=np.float32),
                rotation_matrix=np.eye(3, dtype=np.float32),
            ),
            PinholeEvent(
                path="cameras/front",
                image_from_camera=np.eye(3, dtype=np.float32),
                resolution=(64, 36),
            ),
            TextEvent(path="meta/sample", text="sample-1"),
        ]
    )

    logged_kinds = [payload[0] for _, payload, _ in rerun_calls["logs"]]
    assert logged_kinds == [
        "Image",
        "Points3D",
        "Points2D",
        "Boxes3D",
        "Transform3D",
        "Pinhole",
        "TextLog",
    ]


def test_backend_converts_yaw_to_a_z_axis_quaternion(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_event(
        Boxes3DEvent(
            path="detection3d/prediction",
            centers=np.zeros((1, 3), dtype=np.float32),
            sizes=np.ones((1, 3), dtype=np.float32),
            yaws=np.array([np.pi / 2], dtype=np.float32),
        )
    )

    _, payload, _ = rerun_calls["logs"][0]
    quaternion = payload[1]["quaternions"]
    np.testing.assert_allclose(
        quaternion, [[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]], atol=1e-6
    )


def test_backend_rejects_unknown_events(backend: RerunVisualizationBackend) -> None:
    with pytest.raises(TypeError, match="Unsupported visualization event"):
        backend.log_event(object())  # type: ignore[arg-type]


def test_verify_annotation_context_support_raises_when_legends_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silently emptied AnnotationContext must fail loudly, not lose the legend."""
    monkeypatch.setattr(_FakeAnnotationContext, "serializes_empty", True)
    fake_rerun = _build_fake_rerun({"init": None, "serve_web": None, "logs": [], "steps": []})

    with pytest.raises(RuntimeError, match="discarded a probe AnnotationContext"):
        _verify_annotation_context_support(fake_rerun)


def test_installed_rerun_serializes_annotation_contexts() -> None:
    """Regression guard for the rerun-sdk 0.23.1 / NumPy 1.x ``__array__`` break.

    Rerun reports the failure as a warning and emits an empty context, so
    without the compatibility patch every class legend disappears from the
    viewer while the recording still looks healthy.
    """
    rerun = pytest.importorskip("rerun")
    _patch_class_id_array_protocol()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        context = rerun.AnnotationContext(
            [
                rerun.AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255)),
                rerun.AnnotationInfo(id=1, label="pedestrian", color=(0, 255, 0, 255)),
            ]
        )
        payload = next(
            batch.as_arrow_array().to_pylist()
            for batch in context.as_component_batches()
            if "AnnotationContext#" in str(batch.component_descriptor())
        )

    entries = payload[0]
    assert [entry["class_id"] for entry in entries] == [0, 1]
    assert [entry["class_description"]["info"]["label"] for entry in entries] == [
        "car",
        "pedestrian",
    ]


def test_installed_rerun_passes_the_compatibility_probe() -> None:
    rerun = pytest.importorskip("rerun")
    _patch_class_id_array_protocol()

    _verify_annotation_context_support(rerun)
