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

"""Backend-neutral viewer layouts for the built-in task adapters.

This module owns task paths and presentation choices. Concrete backends only
translate :class:`BlueprintEvent` primitives, so adding a task-specific layout
does not require importing or modifying a backend SDK integration.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from autoware_ml.visualization.events import (
    BlueprintEvent,
    ImageEvent,
    LayoutGroup,
    ViewOverride,
    ViewSpec,
    VisualizationEvent,
)

_DETECTION_METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "mean_best_iou": "GT mean best IoU",
    "max_iou": "Frame max IoU",
    "mean_matched_iou": "Matched mean IoU (>=0.5)",
    "true_positives": "True positives",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
}


def build_calibration_blueprint(
    events: Iterable[VisualizationEvent],
) -> BlueprintEvent | None:
    """Build the raw/fused calibration image comparison when available."""
    fused_paths = [
        event.path
        for event in events
        if isinstance(event, ImageEvent) and event.path.endswith("/camera/fused")
    ]
    raw_paths = [
        event.path
        for event in events
        if isinstance(event, ImageEvent) and event.path.endswith("/camera/image")
    ]
    if not fused_paths:
        return None
    fused_path = fused_paths[0]
    children: list[ViewSpec] = []
    if raw_paths:
        raw_path = raw_paths[0]
        children.append(
            ViewSpec(
                kind="spatial2d",
                name="Raw camera",
                origin=raw_path,
                contents=(raw_path, f"{raw_path}/projected_points"),
            )
        )
    children.append(
        ViewSpec(
            kind="spatial2d",
            name="Fused camera",
            origin=fused_path,
            contents=(fused_path, f"{fused_path}/depth", f"{fused_path}/intensity"),
        )
    )
    return BlueprintEvent(
        layout=LayoutGroup(kind="horizontal", children=tuple(children)),
        auto_views=False,
    )


def _scene_view(
    observed_paths: set[str],
    camera_paths: Sequence[str],
    *,
    name: str,
    point_path: str | None,
    detection_path: str | None,
    camera_frustums_visible: bool,
) -> ViewSpec:
    """Build one 3D scene specification."""
    contents = []
    if point_path is not None and point_path in observed_paths:
        contents.append(point_path)
    if detection_path is not None and detection_path in observed_paths:
        contents.append(detection_path)
    overrides = []
    if camera_paths:
        contents.append("scene/cameras/**")
        overrides.append(ViewOverride(path="/scene/cameras", visible=camera_frustums_visible))
    if detection_path is not None and detection_path.startswith("scene/prediction/"):
        overrides.append(ViewOverride(path=f"/{detection_path}", show_labels=True))
    return ViewSpec(
        kind="spatial3d",
        name=name,
        origin="scene",
        contents=tuple(contents),
        overrides=tuple(overrides),
    )


def _detection_statistics_views(observed_paths: set[str], timeline: str) -> list[ViewSpec]:
    """Build compact IoU quality and match-count plot specifications."""
    metrics_root = "scene/metrics/detection"
    quality_paths = [
        f"{metrics_root}/{name}"
        for name in (
            "precision",
            "recall",
            "mean_best_iou",
            "max_iou",
            "mean_matched_iou",
        )
        if f"{metrics_root}/{name}" in observed_paths
    ]
    count_paths = [
        f"{metrics_root}/{name}"
        for name in ("true_positives", "false_positives", "false_negatives")
        if f"{metrics_root}/{name}" in observed_paths
    ]

    def series_overrides(paths: list[str]) -> tuple[ViewOverride, ...]:
        return tuple(
            ViewOverride(
                path=path,
                series_name=_DETECTION_METRIC_LABELS[path.rsplit("/", 1)[-1]],
                marker_size=8.0,
            )
            for path in paths
        )

    views = []
    if quality_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="3D IoU quality",
                origin=metrics_root,
                contents=tuple(quality_paths),
                overrides=series_overrides(quality_paths),
                y_range=(0.0, 1.0),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    if count_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="Detection matches",
                origin=metrics_root,
                contents=tuple(count_paths),
                overrides=series_overrides(count_paths),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    return views


def _comparison_tab(
    observed_paths: set[str],
    camera_paths: Sequence[str],
    *,
    name: str,
    ground_truth_points: str | None,
    prediction_points: str | None,
    camera_frustums_visible: bool,
    timeline: str,
) -> LayoutGroup:
    """Build side-by-side GT and prediction scenes plus IoU plots."""

    def scene_comparison(
        camera_visible: bool,
        *,
        layout_name: str,
    ) -> LayoutGroup:
        scene_views = []
        if any(
            path in observed_paths
            for path in (
                "scene/ground_truth/segmentation",
                "scene/ground_truth/detections",
            )
        ):
            scene_views.append(
                _scene_view(
                    observed_paths,
                    camera_paths,
                    name=f"GT · {name}",
                    point_path=ground_truth_points,
                    detection_path="scene/ground_truth/detections",
                    camera_frustums_visible=camera_visible,
                )
            )
        if any(
            path in observed_paths
            for path in (
                "scene/prediction/segmentation",
                "scene/prediction/detections",
            )
        ):
            scene_views.append(
                _scene_view(
                    observed_paths,
                    camera_paths,
                    name=f"Prediction · {name}",
                    point_path=prediction_points,
                    detection_path="scene/prediction/detections",
                    camera_frustums_visible=camera_visible,
                )
            )
        return LayoutGroup(
            kind="horizontal",
            children=tuple(scene_views),
            name=layout_name,
        )

    if camera_paths:
        comparison = LayoutGroup(
            kind="tabs",
            children=(
                scene_comparison(False, layout_name="Camera projections OFF"),
                scene_comparison(True, layout_name="Camera projections ON"),
            ),
            name=f"{name} comparison",
            active=int(camera_frustums_visible),
        )
    else:
        comparison = scene_comparison(
            camera_frustums_visible,
            layout_name=f"{name} comparison",
        )
    statistics = _detection_statistics_views(observed_paths, timeline)
    if not statistics:
        return comparison
    return LayoutGroup(
        kind="vertical",
        children=(
            comparison,
            LayoutGroup(
                kind="horizontal",
                children=tuple(statistics),
                name="Detection statistics",
            ),
        ),
        name=f"{name} comparison",
        shares=(2.0, 1.0),
    )


def _camera_tab(observed_paths: set[str], camera_paths: Sequence[str]) -> LayoutGroup:
    """Build per-camera GT/prediction views with persistent projections."""
    camera_tabs: list[ViewSpec | LayoutGroup] = []
    for path in camera_paths:
        camera_name = path.rsplit("/", 1)[-1]
        ground_truth_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/ground_truth/")
        )
        prediction_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/prediction/")
        )
        shared_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/")
            and observed_path not in ground_truth_paths
            and observed_path not in prediction_paths
        )
        comparison_views = []
        if ground_truth_paths:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=f"{camera_name} · GT",
                    origin=path,
                    contents=(path, *shared_paths, *ground_truth_paths),
                )
            )
        if prediction_paths:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=f"{camera_name} · Prediction",
                    origin=path,
                    contents=(path, *shared_paths, *prediction_paths),
                )
            )
        if not comparison_views:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=camera_name,
                    origin=path,
                    contents=(path, *shared_paths),
                )
            )
        camera_tabs.append(
            comparison_views[0]
            if len(comparison_views) == 1
            else LayoutGroup(
                kind="horizontal",
                children=tuple(comparison_views),
                name=f"{camera_name} comparison",
            )
        )
    return LayoutGroup(kind="tabs", children=tuple(camera_tabs), name="Cameras")


def _uncertainty_tab(
    observed_paths: set[str],
    camera_paths: Sequence[str],
    *,
    camera_frustums_visible: bool,
) -> LayoutGroup:
    """Build entropy/semantic views with a visible camera projection switch."""

    def comparison(camera_visible: bool, *, layout_name: str) -> LayoutGroup:
        return LayoutGroup(
            kind="horizontal",
            children=(
                _scene_view(
                    observed_paths,
                    camera_paths,
                    name="Prediction · Entropy",
                    point_path="scene/prediction/entropy",
                    detection_path="scene/prediction/detections",
                    camera_frustums_visible=camera_visible,
                ),
                _scene_view(
                    observed_paths,
                    camera_paths,
                    name="Prediction · Semantic",
                    point_path="scene/prediction/segmentation",
                    detection_path="scene/prediction/detections",
                    camera_frustums_visible=camera_visible,
                ),
            ),
            name=layout_name,
        )

    if not camera_paths:
        return comparison(
            camera_frustums_visible,
            layout_name="Uncertainty",
        )
    return LayoutGroup(
        kind="tabs",
        children=(
            comparison(False, layout_name="Camera projections OFF"),
            comparison(True, layout_name="Camera projections ON"),
        ),
        name="Uncertainty",
        active=int(camera_frustums_visible),
    )


def build_scene_blueprint(
    observed_paths: Iterable[str],
    camera_paths: Iterable[str],
    *,
    point_color_mode: str,
    camera_frustums_visible: bool,
    timeline: str,
) -> BlueprintEvent | None:
    """Build the complete built-in scene layout from neutral entity paths."""
    paths = set(observed_paths)
    cameras = sorted(camera_paths)
    scene_paths = {path for path in paths if path == "scene" or path.startswith("scene/")}
    if not scene_paths:
        return None

    tabs: list[tuple[str, ViewSpec | LayoutGroup]] = []
    comparison_arguments = {
        "observed_paths": paths,
        "camera_paths": cameras,
        "camera_frustums_visible": camera_frustums_visible,
        "timeline": timeline,
    }
    if any(path.endswith("/segmentation") for path in scene_paths):
        tabs.append(
            (
                "Semantic",
                _comparison_tab(
                    **comparison_arguments,
                    name="Semantic",
                    ground_truth_points="scene/ground_truth/segmentation",
                    prediction_points="scene/prediction/segmentation",
                ),
            )
        )
    if "scene/lidar/intensity" in scene_paths:
        tabs.append(
            (
                "Intensity",
                _comparison_tab(
                    **comparison_arguments,
                    name="Intensity",
                    ground_truth_points="scene/lidar/intensity",
                    prediction_points="scene/lidar/intensity",
                ),
            )
        )
    if "scene/lidar/solid" in scene_paths:
        tabs.append(
            (
                "Geometry",
                _comparison_tab(
                    **comparison_arguments,
                    name="Geometry",
                    ground_truth_points="scene/lidar/solid",
                    prediction_points="scene/lidar/solid",
                ),
            )
        )
    if not tabs and any(path.endswith("/detections") for path in scene_paths):
        tabs.append(
            (
                "Detections",
                _comparison_tab(
                    **comparison_arguments,
                    name="Detections",
                    ground_truth_points=None,
                    prediction_points=None,
                ),
            )
        )
    if "scene/prediction/entropy" in scene_paths:
        tabs.append(
            (
                "Uncertainty",
                _uncertainty_tab(
                    paths,
                    cameras,
                    camera_frustums_visible=camera_frustums_visible,
                ),
            )
        )
    if cameras:
        tabs.append(("Cameras", _camera_tab(paths, cameras)))
    if not tabs:
        # A custom adapter that uses the conventional scene root still gets a
        # useful view even before it supplies a tailored BlueprintEvent.
        tabs.append(
            (
                "Scene",
                ViewSpec(
                    kind="spatial3d",
                    name="Scene",
                    origin="scene",
                    contents=tuple(sorted(scene_paths)),
                ),
            )
        )

    preferred_tab = {
        "semantic": "Semantic",
        "intensity": "Intensity",
        "solid": "Geometry",
    }[point_color_mode]
    tab_names = [name for name, _ in tabs]
    active_tab = tab_names.index(preferred_tab) if preferred_tab in tab_names else 0
    return BlueprintEvent(
        layout=LayoutGroup(
            kind="tabs",
            children=tuple(tab for _, tab in tabs),
            active=active_tab,
            name="Multi-task visualization",
        ),
        blueprint_panel_expanded=bool(cameras),
        selection_panel_expanded=False,
        time_panel_expanded=False,
        auto_layout=False,
        auto_views=False,
        make_active=True,
        make_default=True,
    )
