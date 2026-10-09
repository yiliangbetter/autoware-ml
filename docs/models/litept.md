---
icon: lucide/feather
---

# LitePT

LitePT is a PTv3-compatible model that reduces latency by turning off early
attention and late convolution blocks. It is fully compatible with PTv3's
export contract, task models and heads.

It is implemented as `LitePTEncoder`, a subclass of `PointTransformerV3Encoder`:
the gating and the rotary embedding are options on the shared components, and
PTv3's defaults leave both off. Everything not listed below is identical to
[PointTransformerV3](ptv3.md).

## Summary

| Property     | Value                                                           |
| ------------ | --------------------------------------------------------------- |
| Task         | 3D semantic segmentation, 3D object detection                   |
| Modality     | LiDAR                                                           |
| Input        | Single point cloud of x, y, z and intensity                     |
| Output       | Point-wise semantic labels or 3D boxes/scores/classes           |
| Architecture | PTv3 hierarchy with per-stage conv/attention gating and 3D RoPE |
| Datasets     | T4Dataset                                                       |

## What differs from PTv3

| Aspect              | PTv3                                | LitePT                                              |
| ------------------- | ----------------------------------- | --------------------------------------------------- |
| Block composition   | Every block: conv + attention + MLP | Stages 0-2 conv only, stages 3-4 attention only     |
| Positional encoding | Submanifold conv per block          | Conv early; axis-split 3D RoPE on `grid_coord` late |
| Decoder             | `dec_depths` blocks per stage       | Unpooling only at `base`; gated blocks at `tiny`    |
| Encoder blocks      | 14 conv + 14 attention              | 6 conv + 8 attention at `base`                      |
| Serialization       | Every stage sorts for attention     | Only the two coarsest stages read an order          |

## Rotary position embedding

`Point3DRoPE` splits the head dimension into three equal per-axis chunks and
rotates each with a shared frequency ladder over integer `grid_coord` positions.
Each chunk pairs dimension `i` with `i + chunk // 2` as the two components one
shared `(cos, sin)` rotates together, so the rotated span must be a multiple of
six: `rope_span` returns the largest one that fits and any trailing dimensions
pass through unrotated. `base` sizes every stage at 18 channels per head and so
rotates the whole head; `tiny` inherits PTv3-tiny's head counts for a `head_dim`
of 16, rotating 12 dimensions and passing 4 through.

The default `enc_rope_base` of 100.0 mirrors what LitePT's reference CUDA
operator calls `rope_freq`. This is a hyperparameter and needs tuning.

## Available Configurations

| Config Name                                                                           | Task                         | Dataset   | Head        | Range | Purpose                        |
| ------------------------------------------------------------------------------------- | ---------------------------- | --------- | ----------- | ----- | ------------------------------ |
| `segmentation3d/litept/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg` | segmentation3d               | T4Dataset | Seg decoder | 122 m | T4Dataset segmentation         |
| `multi/litept/t4dataset/voxel012_tiny_b8_100e_t4dataset_122m_j6gen2_det3d_seg3d`      | segmentation3d + detection3d | T4Dataset | TransFusion | 122 m | Joint segmentation + detection |

Each inherits dataset, transforms, optimizer and deploy settings from the PTv3
config of the same name and swaps only the encoder and the decoder head.

Both come in the two scales [PTv3](ptv3.md) uses. `base` is the published LitePT
parametrization: 18 channels per attention head and a decoder that only unpools.
`tiny` is PTv3-tiny with the gating applied and nothing else changed, so the two
stay directly comparable - its decoder keeps PTv3-tiny's blocks and runs them
under LitePT's rule. The configs above are at `tiny` scale, following the PTv3
configs they inherit from.

```bash
autoware-ml train --config-name segmentation3d/litept/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg
```

## Hyperparameter tuning

`segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2` defines a
batched Optuna study over the LitePT architecture, driven by the `tune`
commands described in
[Optuna](../user-guide/optuna.md#batched-tuning-with-a-bench-in-the-loop).
Every trial trains a fixed 12 epochs without early stopping, so the study
compares architectures rather than schedules, and every trial is timed on
the target ECU with its initial weights before it trains, so no training is
spent on an architecture that is too slow.

### Search space

The sampler draws scalars, so the per-stage lists are named presets in
`autoware_ml/tuning/presets.py`. Every preset keeps each width divisible by
its head count and, wherever possible, 18 channels per head so the rotary
embedding rotates the whole head. The exceptions are `tiny` and `large`,
which use 16 channels per head: `tiny` because it is PTv3-tiny's ladder,
`large` because 16 is the head size whose stage-4 TensorRT engine builds on
the target at every patch size (the 18-channel presets fail beyond patch 256
at the deployed profile), and a rung above the published widths is only
worth sampling if it can be deployed.

| Parameter           | Values                           | Notes                                                                                                                                                                                                                         |
| ------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grid_size`         | 0.10 to 0.18 m                   | The largest latency lever; coarse grids collapse small classes such as debris.                                                                                                                                                |
| `width_preset`      | large, base, medium, tiny, small, micro | Encoder and decoder widths and head counts. `tiny` is PTv3-tiny's ladder, kept as the comparability anchor; `large` (48 to 768 channels) sits one rung above the published widths for studies whose latency bound allows it. |
| `conv_depth_preset` | 2-2-2, 1-1-1                     | Blocks in the three convolution stages. Each block costs the same arithmetic at every stage because width doubles as the point count quarters.                                                                                |
| `attn_depth_preset` | 6-2, 4-2, 2-2, 2-1               | Blocks in the two attention stages. A block here carries far more parameters and kernel launches than a convolution block. A stage of depth d reads d serialization orders, so even depths match the two the export declares. |
| `dec_depth_preset`  | none, light                      | Decoder blocks: unpooling only, or PTv3-tiny's two attention blocks.                                                                                                                                                          |
| `patch_size`        | 256, 512, 1024                   | Attention window at the attention stages and in the decoder. Past the stage's point count the window is already global.                                                                                                       |
| `mlp_ratio`         | 2, 3, 4                          | Feed-forward expansion. Only attention blocks carry an MLP, so this touches the two deepest stages alone.                                                                                                                     |
| `stem_type`         | linear, conv                     | Embedding stem; the convolution stem uses a 3x3x3 kernel.                                                                                                                                                                     |

The rotary base is not searched. The first study sampled it log-uniformly
over 10 to 1000, and a five-point ablation on the best architecture found
mIoU flat across those two decades, inside the replicate noise: attention
runs only at the two coarsest stages, whose grid coordinates span a few
dozen voxels, so bases above a few tens only idle low-frequency channels.
It stays at the config default of 100; `rope_base_range` in the search
space config re-enables it.

Deliberately fixed: the gating (which stages convolve and which attend), the
serialization orders, and the regularization and numerics settings
(`drop_path`, the dropouts, `shuffle_orders`, `qkv_bias`, `qk_scale`,
`pre_norm`, flash attention and the upcast flags). None of them changes the
cost of the deployed graph, and the regularizers only show over schedules
longer than a trial's. `drop_path`, which now interacts with depth, and
`shuffle_orders`, since training sees four orders while deployment reads two,
are candidates for a follow-up study once an architecture is chosen.

### Objective and constraints

The study maximizes mIoU under a deployed-latency constraint, with an
optional accuracy guard. The latency bound is relative to a reference
measured by the same harness on the same bench, so the question the study
answers is the most accurate model that is no slower than a known one. The
reference is either a reference trial inside the search space (LitePT-tiny,
the bound of the second study, plus 0.1 ms) or a reference model outside it
(`tuning.reference_models`: the third study bounds every trial by the
deployed PTv3-tiny, exported and timed the same way, with no margin). A
debris recall bound (0.35 in the second study: the class is the small-object
proxy, and the bound sits 1.5 standard deviations of the run-to-run noise
below the best architecture of the first study, where 0.4 rejected that
architecture two runs in seven) is a `metric` constraint the study config
may add; the third study records debris recall, the far-range slices and the
grouped-vehicle IoU without enforcing them. An infeasible trial keeps its
value for the sampler but never counts as the best.

Latency is measured before training with the model's initial weights, on
the ADM-AL30 bench through the `ptv3-inference-benchmarks` harness, seven
replicates of fifty inference cycles each. A trial over the bound is told to
the sampler as pruned with its violation, so the sampler learns where the
bound runs at the cost of a few bench minutes instead of an hour of training.

Per-class metrics are only broken out at test by default, so the tuning config
sets the dataset's `detail_stages` to `[val, test]` to log them every epoch.
With a single objective the study stops itself: Optuna's terminator bounds
the improvement a further trial could bring with a Gaussian process over the
feasible trials and stops once that bound falls under 0.004 mIoU, two
standard deviations of the replicate noise.

```bash
autoware-ml tune init --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2
autoware-ml tune import --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2 --from-storage sqlite:///mlruns/litept_tune/optuna.db --from-study litept_seg3d_j6gen2_v1
autoware-ml tune screen --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2
autoware-ml tune train --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2 --n-trials 1
autoware-ml tune measure --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2
autoware-ml tune status --config-name segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2
```

## Export contract

Declared identically to [PTv3](ptv3.md#onnx-preprocessing-contract), but the
gated graph reads fewer of the inputs - nothing consumes the `head_indices` of the
stages that carry no convolution - and
`torch.onnx.export` drops whatever the traced graph never consumes. A LitePT
artifact therefore exposes a subset of the PTv3 inputs.

The deployed node binds the inputs the artifact declares rather than the full
PTv3 list, so a subset is a drop-in engine. Narrowing the declaration to match
is future work.

## Implementation

| Path                                                  | Description                                   |
| ----------------------------------------------------- | --------------------------------------------- |
| `autoware_ml/models/segmentation3d/encoders/ptv3.py`  | `LitePTEncoder`, `Point3DRoPE`, block gating  |
| `autoware_ml/models/segmentation3d/ptv3_base.py`      | `PTv3EncoderExportBase`, shared export inputs |
| `autoware_ml/models/segmentation3d/heads/ptv3.py`     | Decoder gating (`dec_conv`, `dec_attn`)       |
| `autoware_ml/configs/tasks/*/litept/`                 | Experiment configurations                     |
| `autoware_ml/tuning/`                                 | Presets, search space, bench, tuning phases   |
| `autoware_ml/models/detection3d/tests/test_litept.py` | Rotary, gating, and export tests              |

## Acknowledgment

LitePT follows the published topology of the LitePT point-transformer variant,
reimplemented on top of the Autoware-ML PTv3 components. The reference
implementation's kernel-5 submanifold stem is not adopted: it exports poorly
and showed no accuracy advantage over the linear stem.
