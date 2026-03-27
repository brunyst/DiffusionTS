# Diffusion Models for Synthetic Time Series Generation: A Study on Stochastic Processes

This repository contains our project for the **Advanced Deep Learning** course in the **MVA Master's program**. It studies the use of diffusion models for synthetic time series generation, with a focus on stochastic-process-based data such as Ornstein-Uhlenbeck and Heston-type dynamics.

The codebase provides an end-to-end pipeline for:

- simulating synthetic time series from known stochastic processes,
- training score-based diffusion models,
- generating new synthetic trajectories,
- evaluating the quality of the generated samples with quantitative metrics and visual diagnostics.

## Project Overview

Synthetic time series generation is useful for benchmarking, stress-testing, privacy-preserving data sharing, and understanding whether generative models can recover the statistical structure of dynamical systems. In this project, we explore diffusion-based generative modeling in controlled settings where the ground-truth process is known.

The repository is organized around a full experimental workflow:

1. simulate training and test sets from configurable stochastic processes;
2. define a diffusion noise schedule;
3. train a score model;
4. sample synthetic trajectories from the learned model;
5. compute evaluation metrics and save figures for analysis.

## Repository Structure

- [`main.py`](main.py): main entry point for running the full pipeline
- [`config/`](config): Hydra configuration files
- [`pipelines/`](pipelines): data, schedule, training, sampling, and metrics pipelines
- [`models/`](models): diffusion schedules, training utilities, sampler, and losses
- [`networks/`](networks): neural network backbones for score estimation
- [`metrics/`](metrics): evaluation metrics and analysis tools
- [`utils/`](utils): simulation, preprocessing, plotting, and utility functions

## Supported Data Families

The project includes simulation and evaluation utilities for several synthetic time series families, including:

- `OUrange`
- `OU`
- `OUmodes`
- `BM`
- `GBM`
- `CIR`
- `CIRrange`
- `Heston`
- `HestonRange`
- `Lines`
- `Sines`
- `LinearODEs`
- `PDV2factor`

The available dataset labels, schedules, and model configurations are declared in [`pipelines/dicts.py`](pipelines/dicts.py).

## Installation

### Requirements

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/) recommended for dependency management
- optional: a [Weights & Biases](https://wandb.ai/) account for experiment tracking

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### CPU setup

```bash
uv sync --extra cpu
```

### NVIDIA GPU setup

For CUDA 11.8:

```bash
uv sync --extra cu118
```

For CUDA 12.4:

```bash
uv sync --extra cu124
```

### Development setup

```bash
uv sync --extra cpu --extra dev
```

## Quick Start

From the repository root, run:

```bash
uv run python main.py
```

This uses the default Hydra configuration in [`config/config.yaml`](config/config.yaml), which is currently set up for a multivariate `HestonRange` experiment.

To run the provided Ornstein-Uhlenbeck configuration instead:

```bash
uv run python main.py --config-name config_OUrange
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Parameters can be overridden directly from the command line.

Examples:

```bash
uv run python main.py data.data_label=OUrange data.d=1 training.n_epochs=200
```

```bash
uv run python main.py --config-name config_OUrange wandb.mode=disabled sampling.M_gen=200
```

```bash
uv run python main.py data.data_label=HestonRange data.d=2 training.model_label="ScoreNet1D Adapt"
```

### Key configuration fields

- `data.data_label`: type of synthetic process to simulate
- `data.M_train`: number of training time series
- `data.L`: time series length
- `data.d`: number of channels kept for training
- `data.transforms`: preprocessing transforms applied before training
- `schedule.schedule_label`: diffusion schedule type
- `training.model_label`: score network architecture
- `training.n_epochs`: number of training epochs
- `training.batch_size`: batch size
- `sampling.M_gen`: number of generated samples
- `sampling.nb_steps`: number of reverse diffusion steps
- `wandb.mode`: `online`, `offline`, or `disabled`

## Example Commands

### Lightweight sanity check

```bash
uv run python main.py \
  --config-name config_OUrange \
  wandb.mode=disabled \
  data.M_train=128 \
  training.n_epochs=20 \
  sampling.M_gen=64 \
  metrics.compute_predictive=false \
  metrics.compute_discriminative=false
```

### More complete OUrange experiment

```bash
uv run python main.py \
  --config-name config_OUrange \
  wandb.mode=offline \
  data.M_train=3000 \
  training.n_epochs=1000 \
  sampling.nb_steps=1000
```

### Multivariate HestonRange experiment

```bash
uv run python main.py \
  data.data_label=HestonRange \
  data.d=2 \
  data.transforms='["R", null]' \
  training.in_channels=2 \
  training.out_channels=2 \
  wandb.mode=offline
```

## Outputs

Each run creates a dedicated directory:

```text
runs/<run_id>/
```

Typical contents include:

- `weights/`: trained model checkpoints
- `losses/`: saved training and validation losses
- `data_gen/`: generated synthetic samples
- `figures/losses/`: training curves
- `figures/gen/`: generated time series plots
- `figures/MLE/`: parameter-estimation comparison plots
- `figures/memorization/`: nearest-neighbor memorization diagnostics
- `figures/autocorr/`: time-time correlation matrix visualizations
- `config.yaml`: resolved run configuration snapshot

Additional top-level folders may also appear:

- `data_sim/`: cached simulated datasets
- `figures/data_sim/`: figures for simulated training data
- `outputs/`: Hydra outputs
- `wandb/`: local Weights & Biases logs

## Experiment Tracking

Weights & Biases integration is enabled in the main pipeline. If you only want local execution:

```bash
uv run python main.py wandb.mode=disabled
```

To keep logs locally without syncing:

```bash
uv run python main.py wandb.mode=offline
```

## Notes on Reproducibility

- Runs should be launched from the repository root.
- The code automatically selects `cuda`, then `mps`, then `cpu`, depending on hardware availability.
- For quick experiments, it is recommended to reduce `data.M_train`, `training.n_epochs`, and possibly disable the most expensive metrics.
- For multivariate experiments, keep `training.in_channels` and `training.out_channels` consistent with `data.d`.
- In some configurations, `sampling.M_gen=null` means “generate as many samples as in the training set”.

## Current Scope

This repository is intended as a course project and research codebase rather than a polished production package. The main focus is experimental flexibility and analysis of diffusion-based generation on synthetic stochastic processes.

At the moment:

- the main interface is `python main.py`;
- there is no dedicated test suite yet;
- the documentation reflects the current experimental pipeline and should evolve with the project.

## Acknowledgment

This work was developed as part of the **Advanced Deep Learning** course in the **MVA Master's program**.
