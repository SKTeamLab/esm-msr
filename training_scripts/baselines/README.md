# Retraining the baselines on the ESM-MSR split

How to get from this repository to the corrected ThermoMPNN-D, MutateEverything and SPURS checkpoints, which
[`inference_scripts/baselines`](../../inference_scripts/baselines/README.md) then benchmarks. All three train on the capped split
(`data/splits_oct06_capped.pkl`) with valid-label rows only: no bounded or censored row, every measured double kept, no ESM-MSR
loss tweaks. Checkpoints and data are not in the repository (too large); everything else needed is.

| here | what |
|---|---|
| `slurm_thermompnn_train.sh` | ThermoMPNN-D: `thermompnn_inference.py train` (single or epistatic network) |
| `slurm_mutate_everything_train.sh` | MutateEverything, upstream's two-stage recipe on 3 GPUs |
| `slurm_spurs_train.sh`, `spurs/spurs_retrain.patch` | SPURS and SPURS-multi: the training code is a patch on the pinned SPURS |
| `spurs/inference_configs/` | the model configs SPURS inference needs for a retrained checkpoint |

Set-up: `inference_scripts/baselines/setup_third_party.sh` (pinned upstream checkouts) and the environments in
`inference_scripts/baselines/envs/`. Commands are run from the repository root.

## 1. Training tables

```bash
# ThermoMPNN-D and SPURS items tables (needs data/tsuboyama and the splits pickle); msr_venv
python inference_scripts/baselines/prepare/prepare_oct06_data.py --split data/splits_oct06_capped.pkl --keep_all_doubles \
    --out outputs/baselines_capped_allbl/data
# MutateEverything's on-disk format; the MSAs are data (msa_index.json comes from prepare/index_msas.py)
ln -s <dir>/msa_index.json outputs/baselines_capped_allbl/data/msa_index.json
MSR_BASELINE_DATA=$PWD/outputs/baselines_capped_allbl/data python inference_scripts/baselines/prepare/build_mutate_everything_data.py \
    --mode build --name oct06_capped_allbl
```
Without `--keep_all_doubles` the same script writes the older `outputs/baselines_capped` tables (doubles without both singles
measured, and the ones ESM-MSR would train rank-only, dropped); ME stage 1 is trained on those (its single mutations are the same
rows): build them the same way with `--name oct06_capped`.

## 2. ThermoMPNN-D (seeds 1-3, both networks)

```bash
for s in 1 2 3; do for m in single epistatic; do
  MODEL=$m SEED=$s MSR_BASELINE_DATA=$PWD/outputs/baselines_capped_allbl/data BASE_OUT=$PWD/outputs/baselines_corrected \
    sbatch training_scripts/baselines/slurm_thermompnn_train.sh
done; done
```
Each run writes `best_checkpoint.txt`; its content is the `--single_ckpt` / `--epistatic_ckpt` of the inference script.

## 3. MutateEverything (seeds 1-3)

```bash
SEED=1 STAGE=single SPLIT_NAME=oct06_capped sbatch training_scripts/baselines/slurm_mutate_everything_train.sh
SEED=1 STAGE=double SPLIT_NAME=oct06_capped_allbl STAGE1_SPLIT=oct06_capped sbatch training_scripts/baselines/slurm_mutate_everything_train.sh
```
Stage 2 (`logs/af_double_oct06_capped_allbl_seed<s>/checkpoint-99.pth` in the ME checkout) is the checkpoint to benchmark.

## 4. SPURS and SPURS-multi (seeds 1-3, from the retraining of 2026-10-28; not re-run when this was written)

```bash
W=<work dir>; R=$PWD
cp -r $THIRD_PARTY_DIR/SPURS $W/SPURS_train && (cd $W/SPURS_train && patch -p1 < $R/training_scripts/baselines/spurs/spurs_retrain.patch)
mkdir -p $W/SPURS_train/data/checkpoints/ThermoMPNN/vanilla_model_weights
cp $THIRD_PARTY_DIR/ProteinMPNN/vanilla_model_weights/v_48_020.pt $W/SPURS_train/data/checkpoints/ThermoMPNN/vanilla_model_weights/
COMMON="datamodule._target_=items +datamodule.data_dir=$R/outputs/baselines_capped_allbl/data trainer.max_epochs=80 \
  callbacks.model_checkpoint.save_last=false model.encoder.pretrained_weights=data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt \
  model.legacy_esm_forward=true"
for s in 1 2 3; do
  NAME=oct06/leg_seed$s SPURS_TRAIN_DIR=$W/SPURS_train OVERRIDES="$COMMON seed=$s train.seed=$s" sbatch training_scripts/baselines/slurm_spurs_train.sh
done
# once a single model has finished: the same seed's multi model starts from it
NAME=oct06/leg_multi_seed$s CONFIG_NAME=train_multi SPURS_TRAIN_DIR=$W/SPURS_train \
  OVERRIDES="$COMMON seed=$s train.seed=$s model.init_from=$W/SPURS_train/logs/oct06/leg_seed$s/checkpoints/best.ckpt" \
  sbatch training_scripts/baselines/slurm_spurs_train.sh
```
`legacy_esm_forward=true` keeps the ESM structural adapter at its random initialisation, as in the released checkpoint, whose adapter
never received a gradient; `false` trains it. Singles train on single + native_cond rows, SPURS-multi on the doubles. Only about 50
training libraries have doubles and 9 the validation split, so SPURS-multi's selected epoch (best around epoch 2) is noisy.

Benchmark them (the same-seed pair, never mixed): copy or link `logs/oct06/leg_seed<s>/checkpoints/best.ckpt` and
`.../leg_multi_seed<s>/...` into a directory as `oct06_leg_seed<s>.ckpt` / `oct06_legmulti_seed<s>.ckpt`, then
```bash
MODEL=spurs MODELS=both SPURS_CKPT_DIR=<that dir> SPURS_SINGLE=oct06_leg_seed1.ckpt SPURS_MULTI=oct06_legmulti_seed1.ckpt OUT=runs/spurs_s1 \
  inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
```
(`MODELS=single` for the singles alone; SPURS-multi still needs `SPURS_SINGLE`, whose config it reads.)

## Scoring cautions

* Score Tsuboyama on **measured rows only** (`cens == 0`): the inference scripts predict the bounded rows too (flagged in `cens`), and
  including them inflates SPURS by about 0.005-0.007.
* The published SPURS checkpoint is leaky on this split (24 of the 33 capped test libraries were in its training data): the retrained
  checkpoints are the held-out comparison, the published one a reference.
