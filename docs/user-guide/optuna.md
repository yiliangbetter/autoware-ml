---
icon: lucide/sliders-horizontal
---

# Optuna

Autoware-ML integrates [Optuna](https://optuna.org/) for automated hyperparameter optimization. Optuna intelligently searches the hyperparameter space, finding optimal configurations faster than grid search or random search.

## How It Works

Optuna sits above Hydra in the configuration hierarchy:

```text
Optuna (suggests hyperparameters)
    ↓
Hydra (configures training)
    ↓
Lightning (runs training)
    ↓
MLflow (logs results)
```

Each trial:

1. Optuna suggests hyperparameter values
2. Hydra builds the config with those values
3. Training runs and reports the objective metric
4. Optuna uses the result to inform the next trial

## Running a Hyperparameter Search

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.direction=minimize
```

This launches 50 trials, minimizing the objective metric (default: validation loss).

## Defining Search Spaces

Define hyperparameter ranges in your config:

```yaml title="configs/tasks/my_task/my_model/my_optuna.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - _self_

hydra:
  sweeper:
    params:
      model.optimizer.lr: interval(0.0001, 0.01)
      model.optimizer.weight_decay: interval(0.001, 0.1)
      datamodule.train_dataloader.batch_size: choice(2, 4, 8, 16)
      trainer.max_epochs: range(10, 50, step=10)
```

### Search Space Types

| Type          | Syntax                          | Example                          |
| ------------- | ------------------------------- | -------------------------------- |
| Continuous    | `interval(low, high)`           | `interval(0.0001, 0.01)`         |
| Log-scale     | `interval(low, high, log=true)` | `interval(1e-5, 1e-2, log=true)` |
| Integer range | `range(start, end, step)`       | `range(10, 100, step=10)`        |
| Categorical   | `choice(a, b, c)`               | `choice(adam, sgd, adamw)`       |

### Log-Scale for Learning Rates

Learning rates are typically searched on a log scale:

```yaml
hydra:
  sweeper:
    params:
      model.optimizer.lr: interval(1e-5, 1e-2, log=true)
```

## Configuring the Sweeper

Configure the Optuna sweeper in your config:

```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.OptunaSweeper
    study_name: my_optimization
    direction: minimize
    n_trials: 100

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
```

Common samplers: `TPESampler` (default), `CmaEsSampler`, `RandomSampler`, `GridSampler`.

## Objective Metrics

By default the training entrypoint returns `val/loss`. Set `optimized_metric`
to any logged metric and match the sweeper direction:

```yaml
optimized_metric: val/seg3d/mIoU

hydra:
  sweeper:
    direction: maximize
```

## Parallel Trials

Run multiple trials in parallel:

```bash
autoware-ml train --config-name <task>/<model>/<optuna-config> \
    --multirun \
    hydra.sweeper.n_jobs=4
```

!!! warning "GPU Considerations"
    Each parallel trial needs GPU memory. Ensure sufficient VRAM or use multiple GPUs.

Multiple workers can connect to the same study.

## Viewing Results

### MLflow Integration

All trials are logged to MLflow. Use the MLflow UI to compare trials as regular training runs.

## Best Practices

- **Start with wide ranges**, then narrow based on results
- **Use pruning** to avoid wasting compute on bad configurations
- **Fix unimportant parameters** to reduce search space

## Example: Full Optimization Config

```yaml title="configs/tasks/my_task/my_model/my_optuna.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - override /hydra/sweeper: optuna
  - _self_

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.OptunaSweeper
    study_name: my_optimization
    direction: minimize
    n_trials: 100

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42

    params:
      model.optimizer.lr: interval(1e-5, 1e-2, log=true)
      model.optimizer.weight_decay: interval(1e-4, 1e-1, log=true)
      datamodule.train_dataloader.batch_size: choice(2, 4, 8)

trainer:
  max_epochs: 20  # Shorter for faster trials
```

Run it:

```bash
autoware-ml train --config-name <task>/<model>/<optuna-config> --multirun
```

## Batched Tuning With a Bench in the Loop

The sweeper above runs each trial to completion before the next one starts,
and the objective has to exist when training ends. A study whose constraint
is the deployed latency does not fit that shape: latency is measured on other
hardware, and it is known before training. A TensorRT engine's timing depends
on the graph and the input cloud, not on the weight values, so an
architecture can be exported with its initial weights and timed on the target
before a single epoch is spent on it. The `tune` commands split the loop into
phases that each run as their own batch:

1. `tune init` creates the study on its storage and queues the reference
   trials, the known-good configurations the study has to beat.
2. `tune import` carries the measured trials of an earlier study over, one per
   architecture with replicates averaged, so the sampler starts from real
   observations instead of its random phase. They only lack a bench latency.
3. `tune screen` asks the sampler for trials, exports each with its initial
   weights, ships the package to the bench and runs the inference harness
   there. The reference trial goes first because the latency bound derives
   from it. A trial slower than the bound is told to the sampler as an
   infeasible pruned trial and never trains; a faster one joins the training
   queue. The phase only touches the bench inside its configured window.
4. `tune train` trains the queued trials, each in its own subprocess and
   MLflow run. A trial that finishes with a `best.ckpt` is marked `trained`;
   one that fails is reported to the sampler as failed.
5. `tune measure` re-validates every trained checkpoint with its own
   configuration, stores the metrics, tells the sampler the objective and
   every constraint, and then evaluates the termination criterion.
6. `tune status` prints the trials, the latency bound, the best feasible
   trial and the termination verdict.

Phases can be repeated in any order; each picks up where the study is. Since
trials reach the sampler only after measurement, the config enables
`constant_liar` on the TPE sampler so configurations still in flight are not
proposed again. Without a `bench` block the workflow is the earlier one:
`tune train` asks the sampler itself and `tune measure` times the checkpoint
with the eager PyTorch proxy on the training GPU.

### The Tuning Config

A tuning config inherits a training config, pins the per-trial budget, and
adds a `tuning` block:

```yaml title="configs/tasks/segmentation3d/litept/t4dataset/tune_voxel012_tiny_12e_t4dataset_122m_j6gen2_semaseg.yaml"
trainer:
  max_epochs: 12
callbacks:
  early_stopping: null

tuning:
  study_name: litept_seg3d_voxel012_tiny_12e_t4dataset_122m_j6gen2
  storage: sqlite:///mlruns/optuna/litept_seg3d_voxel012_tiny_12e_t4dataset_122m_j6gen2.db
  max_trained_trials: 80
  sampler:
    _target_: optuna.samplers.TPESampler
    multivariate: true
    constant_liar: true
  search_space:
    _target_: autoware_ml.tuning.search_space.LitePTSearchSpace
  objectives:
    - { name: val/seg3d/mIoU, direction: maximize }
  constraints:
    - { metric: val/seg3d/recall_debris, min: 0.35 }
    - { source: latency, name: latency_ms, reference: litept_tiny, margin_ms: 0.1 }
  latency: { proxy: true, frames: 50, warmup: 10, repeats: 5, precision: fp32 }
  bench:
    host: sensing-bench.local
    exchange: { local: /nas_litept/bench, remote: /mnt/fdd_nas/.../bench }
    harness: { repo: /mnt/fdd_nas/.../ptv3-inference-benchmarks, command: "... uv run --frozen" }
    variant: { prefix: lt2, adapter: v12, sources: { autoware_universe: owner/repo@sha } }
    reference: litept_tiny
    window: { start: "18:00", end: "09:00", timezone: Asia/Tokyo }
    queue: { target: 3, target_before_close: 10, close_buffer_hours: 2.5 }
  terminator: { improvement: regret_bound, error: static, constant: 0.004, min_n_trials: 20 }
  reference_trials:
    litept_tiny: { grid_size: 0.12, width_preset: tiny, ... }
```

| Key                                | Meaning                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_trained_trials`               | Trainings the study pays for. Screened-out and imported trials are free and do not count.                                                                                                                                                                                                                                |
| `objectives`                       | Ordered objectives. `source: metric` reads a logged validation metric, `source: latency` the trial's latency in milliseconds. One objective enables the termination criterion.                                                                                                                                           |
| `constraints`                      | Bounds the sampler is told about. A `metric` entry takes `min` and `max`. A `source: latency` entry takes `max`, or `reference` and `margin_ms`: the bound is that reference trial's measured latency plus the margin. A violating trial keeps its value, so nothing measured is discarded, but it never counts as best. |
| `static_filter.max_encoder_params` | Rejects a proposal before it trains when its estimated encoder size exceeds the budget.                                                                                                                                                                                                                                  |
| `latency`                          | Eager proxy settings: untimed warmup frames, timed frames, repetitions per frame, autocast precision. `proxy: false` skips it. With a bench it is recorded for comparison only.                                                                                                                                          |
| `bench`                            | The bench, see below. Omit it or set `enabled: false` for the proxy-only workflow.                                                                                                                                                                                                                                       |
| `terminator`                       | Optuna's termination criterion, see below. Omit it to run to the budget.                                                                                                                                                                                                                                                 |
| `measure.max_attempts`             | Evaluation attempts before a trial is failed. A crashed measurement is retried by running `tune measure` again.                                                                                                                                                                                                          |
| `train_overrides`                  | Overrides applied to every trial's training.                                                                                                                                                                                                                                                                             |
| `reference_trials`                 | Named parameter sets queued first. Values must lie inside the search space. The one `bench.reference` names anchors the latency bound.                                                                                                                                                                                   |

Every trial is a regular MLflow run tagged with `optuna_study` and
`optuna_trial`, and its measurement is mirrored onto the run as `tune/*`
metrics. Per-trial training and bench logs are written next to the study
database.

### The Bench

The bench is the target ECU running the inference harness of the
`ptv3-inference-benchmarks` repository. The screen phase and the bench
exchange files through a directory both machines mount (`bench.exchange`):
packages go in, results come back. SSH carries only the control command,
which runs `autoware_ml/tuning/bench_job.py` on the bench with its system
Python. That job ingests the package into the harness's ML store, writes the
harness variant from the template the screen phase rendered, runs
`bench run` for the configured replicates, copies the results into the
exchange and removes what it created on the bench.

| Key                                                    | Meaning                                                                                                                                                                                                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `host`, `user`, `ssh_key`                              | SSH destination. The container running the study needs an SSH client and a key the bench accepts.                                                                                                                                                      |
| `exchange.local`, `exchange.remote`                    | The exchange directory as this machine and as the bench mount it.                                                                                                                                                                                      |
| `harness.repo`, `harness.command`, `harness.state_dir` | The harness checkout on the bench, the shell prefix that runs its commands (ending in the `uv run` that precedes `bench`) and its state directory.                                                                                                     |
| `variant`                                              | What every trial's variant has in common: firmware profile, adapter generation, overlay sources (the `autoware_universe` ref must match this repository's export contract), input cloud, node parameter file, launch variables, warmup and iterations. |
| `node_param_template`                                  | A node parameter file copied into every package, when the universe's own does not fit, such as a class mapping for a dataset with more classes.                                                                                                        |
| `replicates`, `metric`, `statistic`                    | Replicates per measurement and how they pool: by default the mean of per-replicate medians of `gpu_total_ms`, the harness's own convention.                                                                                                            |
| `max_ci95_ms`                                          | Widest replicate interval accepted. The bench is shared; a wider spread means another job ran alongside, so the measurement is repeated instead of told.                                                                                               |
| `unbuildable_violation_ms`                             | Latency violation recorded for an architecture whose engines do not build on the target at the deployed profile. It cannot be deployed, so it is told as infeasible rather than retried; an imported trial keeps its accuracy.                         |
| `reference`                                            | Name of the reference trial whose latency anchors the relative bound. It is measured first.                                                                                                                                                            |
| `window`                                               | Daily interval during which the bench may be used; an end before the start wraps past midnight. Outside it `tune screen` returns at once.                                                                                                              |
| `queue`                                                | How many screened trials to keep waiting for the trainer while the window is open (`target`) and to reach before it closes (`target_before_close`, applied during the last `close_buffer_hours`).                                                      |
| `job`                                                  | Per-job timeout, attempts before a trial is failed, and whether variants and packages are kept.                                                                                                                                                        |

The package a trial ships is the same split ONNX pair `autoware-ml deploy`
produces, stamped `unversioned` and marked as initialization-only in its
metadata, plus the node's ML-package parameter file rendered from the trial's
config: voxel size, point cloud range, encoder widths, decoder depths and the
class list.

### Termination

With one objective the study can stop itself. Optuna's terminator bounds the
regret, the improvement a better trial could still bring, with a Gaussian
process fit on the completed trials, and stops the study once that bound
falls under the statistical error of the objective. Its built-in check looks
at every completed trial; a constrained study must judge only the feasible
ones, otherwise a fast but inaccurate trial would count as the incumbent, so
the evaluators run on the feasible set here. `tune measure` records every
check on the study, and once a check says stop, `tune screen` and
`tune train` refuse new trials unless told to ignore the verdict.

| Key                     | Meaning                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `improvement`           | `regret_bound` (the Gaussian-process bound), `emmr` (expected minimum model regret) or `stagnation` (trials since the best value last improved). |
| `error`                 | `static` (a constant in objective units), `cross_validation` (needs per-trial cross-validation scores) or `median` (pairs with `emmr`).          |
| `constant`              | The static error. Set it to the run-to-run noise of the objective; improvements below it are not worth another trial.                            |
| `min_n_trials`          | Feasible completed trials before termination is considered.                                                                                      |
| `feasible_only`         | Judge only trials that satisfy every constraint.                                                                                                 |
| `max_stagnation_trials` | Patience of the stagnation evaluator.                                                                                                            |

### Latency Proxy

The eager proxy runs the model in PyTorch on the training GPU with
single-frame batches and CUDA-event timing, in fp32 by default. Without a
bench it is the study's latency: absolute numbers differ from the deployed
ones, so the Pareto candidates must be re-timed on the vehicle hardware
before choosing. With a bench it is still recorded, since the pair of numbers
per trial tells how well the proxy ranks architectures. `precision: fp16`
autocasts the forward, which spconv's implicit-GEMM tuner rejected for the
full LitePT graph in the development environment, so it is opt-in.

### Viewing the Study

```bash
optuna-dashboard sqlite:///mlruns/optuna/<study>.db
```

## Learn More

For detailed Optuna documentation, see the [official Optuna documentation](https://optuna.org/).
