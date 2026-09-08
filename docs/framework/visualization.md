---
icon: lucide/binoculars
---

# Visualization Design

Autoware-ML now includes an isolated visualization scaffold intended for
prediction preview, calibration debugging, and dataset inspection. The initial
backend target is [Rerun](https://rerun.io/), but the task-facing API is kept
backend-neutral so future backends or offline exporters can reuse the same
task adapters.

## Design Goals

- Keep visualization out of model and transform core logic.
- Make visualization callable from tests, scripts, and future CLI commands.
- Reuse one neutral scene-event layer across calibration status, segmentation3d,
  and detection3d.
- Keep the first integration narrow: sample-at-a-time preview rather than
  framework-wide logging during training.

## Layering

```mermaid
flowchart LR
    Task[Task code or test] --> Session[VisualizationSession]
    Session --> Adapter[Task adapter]
    Adapter --> Events[Visualization events]
    Events --> Backend[Visualization backend]
    Backend --> Rerun[Rerun SDK]
```

### 1. Visualization Session

`autoware_ml.visualization.session.VisualizationSession` is the public entrypoint. It
owns a backend and exposes task-oriented helpers:

- `log_calibration_status(...)`
- `log_segmentation3d(...)`
- `log_segmentation3d_data(...)`
- `log_detection3d(...)`
- `log_detection3d_data(...)`

This keeps later CLI or test integration simple and avoids leaking Rerun calls
into the rest of the framework.

### 2. Task Adapters

Task adapters turn native Autoware-ML data into neutral scene events:

- `visualization/calibration_status.py`
- `visualization/segmentation3d.py`
- `visualization/detection3d.py`

These adapters understand task semantics, for example:

- calibration camera intrinsics and lidar-to-camera transforms
- segmentation point labels and confidences
- detection decoded 3D boxes and class scores
- LiDAR point-cloud intensity and semantic point coloring
- pointwise prediction logits for uncertainty visualization

### 3. Visualization Events

The shared event layer in `visualization/events.py` defines primitives such as:

- `AnnotationContextEvent`
- `ImageEvent`
- `PointCloud3DEvent`
- `Points2DEvent`
- `Boxes3DEvent`
- `Transform3DEvent`
- `PinholeEvent`
- `ScalarEvent`
- `TextEvent`

This is the isolation boundary between task code and any concrete backend.

### 4. Backend

`visualization/rerun_backend.py` is the only place that knows the Rerun SDK.
It converts the neutral events into `rerun.Image`, `rerun.Points3D`,
`rerun.Boxes3D`, `rerun.Transform3D`, and related entities.

There is also a `NoOpVisualizationBackend` for disabled or test-only paths. The
Rerun backend is imported lazily, so the `noop` backend and the smoke tests
built on it stay usable without the Rerun SDK installed.

### Dependency compatibility

`rerun-sdk` 0.23.1 declares `numpy>=1.23`, but its generated `__array__`
implementations forward a `copy` argument that only NumPy 2.0 and later accept.
Under the pinned `numpy==1.26.4` this makes every `AnnotationContext` serialize
to an empty list, and because Rerun reports the failure as a warning rather than
an exception, class legends disappear from the viewer while the recording still
looks healthy.

`rerun_backend.py` therefore patches `rerun.datatypes.ClassId.__array__` to drop
the keyword when it is `None`, which restores NumPy 1.x support and leaves
NumPy 2.x behaviour untouched. Backend startup then logs a probe legend and
raises if it still serializes empty, so an incompatible dependency bump fails
loudly instead of quietly dropping every legend. Remove the patch once the
repository moves to NumPy 2.x or a Rerun release that fixes the conversion.

### Class name resolution

Legends and per-instance labels read as class names rather than integer ids only
when class names reach the adapters, and they are looked up from three sources in
order:

1. an explicit `visualization.class_names` override
2. the collated batch's `class_names` key
3. the raw dataset info returned by `dataset.get_data_info(...)`

The batch is checked but rarely carries anything: split transform pipelines drop
`class_names` before collation for every task, so detection recovers names from
the raw dataset info and segmentation depends on the configured value. Each task
dataset config therefore exposes a top-level `class_names`, which the metric
suites reference as well so the viewer and the metrics cannot disagree.

When no source carries them, `format_class_label` falls back to the stringified
class id, so an unnamed legend means the resolution chain came up empty rather
than that the recording is broken.

## Task Coverage

### Calibration Status

The initial calibration adapter can log:

- camera intrinsics
- lidar-to-camera extrinsics
- raw camera image
- projected lidar points overlaid on the image plane
- optional fused image preview
- optional 3D lidar points
- predicted and ground-truth calibration status
- prediction confidence and a readable status summary

This is enough to preview calibration state later from a test run or dedicated
CLI command without baking visualization into the existing preview transform.

### Segmentation 3D

The segmentation adapter can log:

- semantic class legend
- point cloud positions
- predicted semantic labels as per-point colors
- optional ground-truth labels
- sample metadata and point counts
- optional mean confidence from `pred_probs`
- pointwise normalized entropy computed from `pred_logits` (softmax followed by
  `-sum(p * log(p)) / log(num_classes)`)
- selectable point coloring: semantic labels, LiDAR intensity, or a solid color

This matches the current segmentation prediction contract, which already
returns `pred_labels` and `pred_probs`.

Point positions are read from `points` when present, and from `coord` otherwise.
PTv3 pipelines drop raw points during grid sampling, so their samples are
matched and rendered through `coord`, with `coord[inverse]` restoring
point-level positions that align with `origin_segment` labels.

### Combined Detection and Segmentation

The existing `multi/ptv3` configurations are supported by the preview pipeline.
A combined sample is logged below one `scene` hierarchy. The Rerun blueprint
presents ground truth and prediction side by side, with the matching detection
boxes overlaid on each point cloud. Prediction previews reuse the model's
decoded detection output and pointwise segmentation logits; no visualization
code is added to the model.

The explicit scene layout contains:

- a **Semantic** comparison with GT and predicted pointwise classes
- an **Intensity** comparison using normalized LiDAR return intensity
- a **Geometry** comparison with a neutral point color
- an **Uncertainty** comparison between logit entropy and predicted semantics
- a **Cameras** tab for the image streams, while every 3D comparison also
  includes the calibrated camera frustums
- compact 3D IoU quality and match-count plots below GT/PD detection comparisons

The three point-cloud tabs make the coloring mode selectable in the viewer.
`--point-color-mode` only chooses which one is initially active.

### Detection 3D

The detection adapter can log:

- semantic class legend
- optional lidar points
- decoded predicted 3D boxes
- optional ground-truth boxes
- class-colored boxes with class/score labels
- sample metadata and prediction/ground-truth counts
- the same point-cloud color modes, including LiDAR intensity from column 4
- same-class, yaw-aware 3D IoU matching at threshold 0.5, with per-frame
  true-positive, false-positive, false-negative, precision, recall, and mean
  matched-IoU scalars

Detection and segmentation use sibling `scene/prediction` and
`scene/ground_truth` entities. A frame with no detection annotations remains a
segmentation-only preview; no synthetic detection objects are created.

### Camera and timing behavior

Each available camera is logged as a transform, pinhole calibration, and image.
The Rerun blueprint places the camera frustums in the same 3D scene as the
LiDAR points and boxes, enabling live 3D camera views and projection inspection.
Every explicit 3D view uses `scene` as its origin. Automatic views are disabled,
and the Blueprint, Selection, and Time panels start collapsed, so a root `/`
text view or stream is not shown in the working layout. The Time panel can be
expanded when the 10 Hz timeline needs to be scrubbed.
Camera overlays are pointwise LiDAR segmentation only; camera/pixel
segmentation is outside this scope.

For T4 multi-task datasets, one annotated keyframe acts as a preview anchor. At
the default `--prediction-frequency-hz 10`, the provider resolves the nine
numbered LiDAR and camera files between adjacent 1 Hz keyframes. The anchor is
rendered with GT and prediction, while intermediate frames contain prediction
only. Missing source files are skipped, scene boundaries are never crossed, and
GT is neither interpolated nor held from the previous frame.

Before each timeline step, the dynamic LiDAR, prediction, ground-truth, camera,
metric, and metadata entities are cleared recursively. Intermediate prediction
records also force `sweeps=[]`, so only the current source cloud is inferred and
displayed; points from previous frames cannot accumulate.

The preview setting `VisualizationPreviewConfig.point_color_mode` accepts
`semantic` (default), `intensity`, or `solid`. For `intensity`, the fourth point
feature is normalized per frame and rendered with the shared scalar heatmap.

It normalizes both existing decoded output styles:

- `bboxes_3d` / `scores_3d` / `labels_3d`
- `bboxes` / `scores` / `labels`

## Current Integration

Visualization is now wired into a dedicated CLI preview command:

```bash
autoware-ml visualize \
    --config-name <task>/<model>/<config> \
    --weights <path/to/model.ckpt> \
    --mode predictions \
    --split test \
    --sample-index 0 \
    --max-samples 1 \
    --prediction-frequency-hz 10 \
    --point-color-mode semantic
```

This path:

1. Instantiates the configured datamodule.
2. Optionally instantiates the configured model and loads `--weights`.
3. Builds a one-sample preview dataloader for the selected split.
4. Runs the split-specific dataset transforms and collation path; when weights are provided, the model-owned preprocessing and prediction path is used.
5. Either logs transformed model inputs directly or runs `predict_step(...)` and logs predictions.
6. Emits task-specific visualization events through `VisualizationSession`.

That keeps visualization out of the Lightning evaluation loop while making the
feature immediately usable.

When no weights are provided, the same command falls back to transformed-data
preview and logs the actual model inputs after the selected split pipeline,
without running prediction.

When multiple samples are previewed, the Rerun backend logs them on the shared
`frame` timeline and records their source timestamps on `sensor_time`, so the
viewer scrubber can move between prediction frames directly.

## Planned Integration Points

Possible later integration steps are:

1. Add a narrow `--show` path for `autoware-ml test` and/or `predict`.
2. Allow selected transforms or test helpers to emit visualization events
   through `VisualizationSession` instead of saving ad hoc preview files.

## Recommended Usage Pattern

The preferred future call site is:

```python
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.session import VisualizationSession

session = VisualizationSession.from_config(
    VisualizationSessionConfig(
        backend="rerun",
        recording_id="calibration-debug",
        web_port=9090,
        grpc_port=9876,
    )
)
session.set_step(sample_index)
session.log_calibration_status(...)
```

That keeps the dependency on Rerun confined to one backend module. Use
`rerun` for browser-based Docker workflows, and `noop` for smoke tests that
only validate the preview pipeline.
