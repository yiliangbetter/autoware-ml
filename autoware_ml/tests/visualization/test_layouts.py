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

"""Tests for backend-neutral task layout declarations."""

from __future__ import annotations

import numpy as np

from autoware_ml.visualization.events import ImageEvent, LayoutGroup, ViewSpec
from autoware_ml.visualization.layouts import (
    build_calibration_blueprint,
    build_scene_blueprint,
)


def test_calibration_layout_is_expressed_only_as_neutral_specs() -> None:
    blueprint = build_calibration_blueprint(
        [
            ImageEvent(
                path="calibration_status/camera/image",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            ImageEvent(
                path="calibration_status/camera/fused",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
        ]
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    assert [child.name for child in blueprint.layout.children] == [
        "Raw camera",
        "Fused camera",
    ]


def test_scene_layout_includes_semantic_camera_and_statistics_views() -> None:
    paths = {
        "scene/ground_truth/segmentation",
        "scene/prediction/segmentation",
        "scene/prediction/entropy",
        "scene/ground_truth/detections",
        "scene/prediction/detections",
        "scene/lidar/intensity",
        "scene/cameras/front",
        "scene/cameras/front/projected/ground_truth/segmentation",
        "scene/cameras/front/projected/prediction/detections",
        "scene/metrics/detection/precision",
        "scene/metrics/detection/true_positives",
    }

    blueprint = build_scene_blueprint(
        paths,
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    serialized = repr(blueprint)
    assert "Semantic comparison" in serialized
    assert "Prediction · Entropy" in serialized
    assert "front · GT" in serialized
    assert "front · Prediction" in serialized
    assert "3D IoU quality" in serialized
    assert "Detection matches" in serialized
    assert blueprint.blueprint_panel_expanded is True
    assert [child.name for child in blueprint.layout.children] == [
        "Semantic comparison",
        "Intensity comparison",
        "Uncertainty",
        "Cameras",
    ]

    semantic_layout = blueprint.layout.children[0]
    assert isinstance(semantic_layout, LayoutGroup)
    camera_switch = semantic_layout.children[0]
    assert isinstance(camera_switch, LayoutGroup)
    assert camera_switch.kind == "tabs"
    assert camera_switch.name == "Semantic comparison"
    assert camera_switch.active == 0
    assert [child.name for child in camera_switch.children] == [
        "Camera projections OFF",
        "Camera projections ON",
    ]
    assert "show_labels=True" in serialized


def test_scene_layout_camera_switch_respects_visible_initial_state() -> None:
    blueprint = build_scene_blueprint(
        {
            "scene/prediction/segmentation",
            "scene/prediction/detections",
            "scene/cameras/front",
        },
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=True,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    camera_switch = blueprint.layout.children[0]
    assert isinstance(camera_switch, LayoutGroup)
    assert camera_switch.kind == "tabs"
    assert camera_switch.name == "Semantic comparison"
    assert camera_switch.active == 1


def test_unknown_scene_adapter_gets_a_generic_view() -> None:
    blueprint = build_scene_blueprint(
        {"scene/new_task/result"},
        [],
        point_color_mode="solid",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    view = blueprint.layout.children[0]
    assert isinstance(view, ViewSpec)
    assert view.name == "Scene"
    assert view.contents == ("scene/new_task/result",)
