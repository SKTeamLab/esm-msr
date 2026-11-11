# Retraining Baselines on the Default Capped Split

Scripts to retrain all six baseline models (`ThermoMPNN` single & epistatic, `MutateEverything` single & double, and `SPURS` single & multi) on the default ESM-MSR capped split (`data/splits_oct06_capped.pkl` -> `oct06_capped_allbl`). Each baseline trains on measured valid-label rows only, and every script runs from the repository root either locally (`bash`) or on Slurm (`sbatch`).

| Path | Description |
|---|---|
| `slurm_thermompnn_train.sh` | ThermoMPNN (`MODEL=single`) and ThermoMPNN-D (`MODEL=epistatic`) |
| `slurm_mutate_everything_train.sh` | MutateEverything Stage 1 (`STAGE=single`) and Stage 2 (`STAGE=double`) |
| `slurm_spurs_train.sh`, `spurs/spurs_retrain.patch` | SPURS (`MODEL=single`) and SPURS-multi (`MODEL=multi`); auto-bootstraps patched training tree |
| `spurs/inference_configs/` | Hydra model configs used by `inference_scripts/baselines` when scoring retrained SPURS checkpoints |
| `requirements/` | Separate Python/CUDA 12.8 environment specs for baseline retraining (not needed for ESM-MSR itself) |

## Prerequisites

1. Clone upstream baseline repositories and weights via [`inference_scripts/baselines/setup_third_party.sh`](../../inference_scripts/baselines/setup_third_party.sh).
2. Create the baseline training environments documented in [`requirements/README.md`](./requirements/README.md) (`ThermoMPNN-D`, `spurs`, and `mutate_everything_cu128`).
3. Ensure Tsuboyama raw tables/PDBs (`data/tsuboyama/`) and Rosetta single-mutant features (`FINAL_results/` on an SSD, used by ThermoMPNN-D double augmentation) are available; see [`inference_scripts/baselines/paths.py`](../../inference_scripts/baselines/paths.py) for path overrides.

## 1. Prepare Training Tables (`oct06_capped_allbl`)

Using the main repository environment (`msr_venv`), generate the shared item tables for `splits_oct06_capped` and build MutateEverything's directory layout:

```bash
DATA=$PWD/outputs/baselines/oct06_capped_allbl/data
python inference_scripts/baselines/prepare/prepare_baseline_data.py \
  --split splits_oct06_capped --keep_all_doubles
# Index MSAs (see inference_scripts/baselines/prepare/index_msas.py) and build MutateEverything inputs:
ln -sf /path/to/msa_index.json "$DATA/msa_index.json"
MSR_BASELINE_DATA="$DATA" python inference_scripts/baselines/prepare/build_mutate_everything_data.py --mode build
```

This populates `outputs/baselines/oct06_capped_allbl/data` (used by ThermoMPNN-D and SPURS) and `$MUTATE_EVERYTHING_DIR/data/oct06_capped_allbl` (used by MutateEverything).

## 2. ThermoMPNN & ThermoMPNN-D (`single` and `epistatic`)

```bash
DATA=$PWD/outputs/baselines/oct06_capped_allbl/data
for s in 1 2 3; do
  for m in single epistatic; do
    MODEL=$m SEED=$s SPLIT=splits_oct06_capped MSR_BASELINE_DATA="$DATA" \
      bash training_scripts/baselines/slurm_thermompnn_train.sh   # or `sbatch`
  done
done
```

Outputs are written to `outputs/baselines/oct06_capped_allbl/thermompnn/<model>_seed<s>/`, with the selected checkpoint path recorded in `best_checkpoint.txt`.

## 3. MutateEverything (`single` Stage 1 -> `double` Stage 2)

```bash
for s in 1 2 3; do
  STAGE=single SEED=$s SPLIT_NAME=oct06_capped_allbl bash training_scripts/baselines/slurm_mutate_everything_train.sh
  STAGE=double SEED=$s SPLIT_NAME=oct06_capped_allbl bash training_scripts/baselines/slurm_mutate_everything_train.sh
done
```

`STAGE=both` (the default) runs Stage 1 (20 epochs) followed immediately by Stage 2 (100 epochs); GPU count (`NPROC`) is detected automatically. Checkpoints are saved in `$MUTATE_EVERYTHING_DIR/logs/af_{single,double}_oct06_capped_allbl_seed<s>/` (`checkpoint-99.pth` is benchmarked).

## 4. SPURS & SPURS-multi (`single` -> `multi`)

```bash
DATA=$PWD/outputs/baselines/oct06_capped_allbl/data
for s in 1 2 3; do
  MODEL=single SEED=$s MSR_BASELINE_DATA="$DATA" bash training_scripts/baselines/slurm_spurs_train.sh
  MODEL=multi  SEED=$s MSR_BASELINE_DATA="$DATA" bash training_scripts/baselines/slurm_spurs_train.sh
done
```

On first run, `slurm_spurs_train.sh` creates `outputs/baselines/oct06_capped_allbl/spurs_train` from `$SPURS_DIR`, applies `spurs/spurs_retrain.patch`, and stages ProteinMPNN `v_48_020.pt`. Checkpoints are written to `spurs_train/logs/oct06_capped_allbl/{leg_seed<s>,leg_multi_seed<s>}/checkpoints/best.ckpt` and evaluated via [`inference_scripts/baselines`](../../inference_scripts/baselines/README.md).
