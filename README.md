# caloxtreme-clf

Binary classifier for calorimeter showers, built with PyTorch Lightning and configured via LightningCLI.

Ce projet entraine un classifieur binaire sur des donnees calorimetriques (generees vs truth) avec un workflow base sur des fichiers YAML.

## TL;DR

- Main training entrypoint: [main.py](main.py)
- Most up-to-date config: [config/mlp_caloinn300_test_trainer.yaml](config/mlp_caloinn300_test_trainer.yaml)
- Model class used by that config: [module/lightning.py](module/lightning.py)
- Data module used by that config: [data/datamodule.py](data/datamodule.py)

## Quick Start

### 1) Environment setup

Python requirement (from [pyproject.toml](pyproject.toml)): `>=3.14`.

```bash
uv sync
```

### 2) Smoke run (fast sanity check)

Use the lightweight synthetic-data config to validate your environment:

```bash
uv run python main.py fit --config config/mlp_test.yaml
```

This config sets `use_synthetic_data: true`, so it does not depend on external HDF5 files.

### 3) Main training run (current reference)

```bash
uv run python main.py fit --config config/mlp_caloinn300_test_trainer.yaml
```

The reference config currently points to absolute HPC paths. Adapt them to your environment if needed:

- `data.init_args.gen_data_path`
- `data.init_args.truth_data_path`
- `data.init_args.test_data_path`
- `data.init_args.xml_path`
- `trainer.stage_dir`

## How the training command works

[main.py](main.py) instantiates `LightningCLI` with:

- a custom trainer class: `training_utils.trainer.Trainer`
- `subclass_mode_model=True` (model selected from YAML)
- config saving enabled (`save_config_kwargs={"overwrite": True}`)

In practice, the YAML controls:

- model class and hyperparameters
- data module and data paths
- optimizer and scheduler
- trainer options (accelerator/devices/epochs/logger/callbacks)

## Core code map

- Entrypoint CLI: [main.py](main.py)
- Data pipeline: [data/datamodule.py](data/datamodule.py)
- Feature helpers: [data/utils.py](data/utils.py), [data/HighLevelFeatures.py](data/HighLevelFeatures.py)
- Lightning modules: [module/lightning.py](module/lightning.py)
- Network definition: [module/classifier.py](module/classifier.py)
- Custom trainer/root-dir logic: [training_utils/trainer.py](training_utils/trainer.py)
- WandB logger wrapper: [training_utils/wandb.py](training_utils/wandb.py)
- Checkpoint callback: [training_utils/callbacks/checkpoint.py](training_utils/callbacks/checkpoint.py)

## Data contract (important)

`SimpleMLPDataModule` expects raw source HDF5 files with at least:

- `showers`
- `incident_energies`

During `prepare_data`, the module creates cached split files:

- `*_train.hdf5`
- `*_val.hdf5`
- `*_test.hdf5`

These files include labels and processed fields:

- `y` (0 for generated, 1 for truth/test)
- `X_proc` (flattened and scaled shower)
- `cond_proc` (incident energy in GeV)
- `X_hlf` (high-level physics features)

Energy filtering is controlled by `data.init_args.min_energy`.

## Output locations

The effective run directory is managed by [training_utils/trainer.py](training_utils/trainer.py):

- non-interactive Slurm: `${trainer.stage_dir}/${SLURM_JOB_ID}`
- otherwise: `${trainer.stage_dir}/YYYY-MM-DD--HH-MM`

With the default checkpoint callback ([training_utils/callbacks/checkpoint.py](training_utils/callbacks/checkpoint.py)), checkpoints are saved under:

- `${stage_dir}/artifacts/`

WandB logging is configured in the YAML through [training_utils/wandb.py](training_utils/wandb.py).

## Common commands

Train:

```bash
uv run python main.py fit --config config/mlp_caloinn300_test_trainer.yaml
```

Test (using same config):

```bash
uv run python main.py test --config config/mlp_caloinn300_test_trainer.yaml
```

Override options from CLI (example):

```bash
uv run python main.py fit \
	--config config/mlp_caloinn300_test_trainer.yaml \
	--trainer.max_epochs 5 \
	--data.init_args.batch_size 2048
```

Run through Slurm helper scripts (example):

```bash
bash batch/debug-gpu.sh uv run python main.py fit --config config/mlp_caloinn300_test_trainer.yaml
```

## Troubleshooting

1. File-not-found at startup

Check every absolute path in [config/mlp_caloinn300_test_trainer.yaml](config/mlp_caloinn300_test_trainer.yaml). Most onboarding failures come from environment-specific paths.

2. Very slow data loading

Try `use_lazy_hdf5: true` (already enabled in the reference config) and ensure `tmp_dir` points to a fast local filesystem for staged copies.

3. Out-of-memory

Reduce `data.init_args.batch_size` and/or switch to fewer workers.

4. No artifacts where expected

Confirm `trainer.stage_dir` and whether you are running under Slurm interactive vs batch mode, since output folder naming differs.

## Notes

- There is currently no Makefile in this repository. Use direct `uv run ...` commands.
- For interactive allocations on NERSC-like systems, see [request_interactive.sh](request_interactive.sh).
- For batch submission templates, see [batch/debug-gpu.sh](batch/debug-gpu.sh) and [batch/pm_submit_batch_40GB_1GPU.sh](batch/pm_submit_batch_40GB_1GPU.sh).
