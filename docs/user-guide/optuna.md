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

## Two-Phase Batch Tuning

The sweeper above runs each trial to completion before the next one starts,
and the objective has to exist when training ends. A study whose objectives
include deployed latency does not fit that shape: latency is measured after
training, ideally as its own batch and often on other hardware. The `tune`
commands split the loop into phases that can each run overnight:

1. `tune init` creates the study on its storage and queues the reference
   trials, the known-good configurations the study has to reproduce.
2. `tune train` asks the sampler for new trials and trains each one in its
   own subprocess and MLflow run. A trial that finishes with a `best.ckpt`
   is marked `trained`; one that fails is reported to the sampler as failed.
3. `tune measure` re-validates every trained checkpoint with its own
   configuration, benchmarks its latency, stores both on the trial, and only
   then tells the sampler the full objective vector.
4. `tune status` prints the trials and the Pareto front.

Phases can be repeated in any order; each picks up where the study is. Since
trials reach the sampler only after measurement, the config enables
`constant_liar` on the TPE sampler so configurations still in flight are not
proposed again.

### The Tuning Config

A tuning config inherits a training config, pins the per-trial budget, and
adds a `tuning` block:

```yaml title="configs/tasks/segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2.yaml"
trainer:
  max_epochs: 12
callbacks:
  early_stopping: null

tuning:
  study_name: litept_seg3d_voxel012_122m_t4dataset_j6gen2
  storage: sqlite:///mlruns/optuna/litept_seg3d_voxel012_122m_t4dataset_j6gen2.db
  sampler:
    _target_: optuna.samplers.TPESampler
    seed: 42
    multivariate: true
    constant_liar: true
  search_space:
    _target_: autoware_ml.tuning.search_space.LitePTSearchSpace
  objectives:
    - { name: latency_ms, direction: minimize, source: latency }
    - { name: val/seg3d/mIoU, direction: maximize, source: metric }
    - { name: val/seg3d/recall_debris, direction: maximize, source: metric }
  constraints:
    - { metric: val/seg3d/recall_debris, min: 0.4 }
  latency: { frames: 50, warmup: 10, repeats: 5, precision: fp32 }
  reference_trials:
    litept_tiny: { grid_size: 0.12, width_preset: tiny, ... }
```

| Key                                | Meaning                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `objectives`                       | Ordered objectives. `source: metric` reads a logged validation metric, `source: latency` the benchmark's mean latency in milliseconds.                 |
| `constraints`                      | Soft bounds the sampler is told about. A violating trial keeps its objectives, so nothing measured is discarded, but it never enters the Pareto front. |
| `static_filter.max_encoder_params` | Rejects a proposal before it trains when its estimated encoder size exceeds the budget.                                                                |
| `latency`                          | Benchmark settings: untimed warmup frames, timed frames, repetitions per frame, and autocast precision.                                                |
| `measure.max_attempts`             | Evaluation attempts before a trial is failed. A crashed measurement is retried by running `tune measure` again.                                        |
| `train_overrides`                  | Overrides applied to every trial's training.                                                                                                           |
| `reference_trials`                 | Named parameter sets queued first. Values must lie inside the search space.                                                                            |

Every trial is a regular MLflow run tagged with `optuna_study` and
`optuna_trial`, and its measurement is mirrored onto the run as `tune/*`
metrics. Per-trial training logs are written next to the study database.

### Latency Proxy

The benchmark runs the model eagerly in PyTorch on the training GPU with
single-frame batches and CUDA-event timing, in fp32 by default. It is a proxy
for the deployed TensorRT latency on the target ECU: absolute numbers differ,
but the ranking of architectures is what the sampler needs. Measure the Pareto
candidates on the vehicle hardware before choosing one. `precision: fp16`
autocasts the forward, which spconv's implicit-GEMM tuner rejected for the full
LitePT graph in the development environment, so it is opt-in.

### Viewing the Study

```bash
optuna-dashboard sqlite:///mlruns/optuna/<study>.db
```

## Learn More

For detailed Optuna documentation, see the [official Optuna documentation](https://optuna.org/).
