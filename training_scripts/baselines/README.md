# Retraining the baselines on an ESM-MSR split

How to get from this repository to retrained ThermoMPNN-D, MutateEverything and SPURS checkpoints on **any split** (a pickle under
`data/` with `train` / `val` / `test` keys), which [`inference_scripts/baselines`](../../inference_scripts/baselines/README.md)
then benchmarks. They train on valid-label rows only: no bounded or censored row, every measured double kept, no ESM-MSR loss
tweaks. The commands below use the capped split (`splits_oct06_capped`), the one the corrected baselines were trained on; change
`SPLIT` for another. Checkpoints and data are not in the repository (too large); everything else needed is.

| here | what |
|---|---|
| `slurm_thermompnn_train.sh` | ThermoMPNN-D: `thermompnn_inference.py train` (single or epistatic network) |
| `slurm_mutate_everything_train.sh` | MutateEverything, upstream's two-stage recipe on 3 GPUs |
| `slurm_spurs_train.sh`, `spurs/spurs_retrain.patch` | SPURS and SPURS-multi: the training code is a patch on the pinned SPURS |
| `spurs/inference_configs/` | the model configs SPURS inference needs for a retrained checkpoint |

Set-up: `inference_scripts/baselines/setup_third_party.sh` (pinned upstream checkouts) and the environments in
`inference_scripts/baselines/envs/`. Commands are run from the repository root.

## 1. Training tables

A split and two choices define a table variant, named from them (`paths.variant_name`): the split's stem without `splits_`, plus
`_allbl` with `--keep_all_doubles` (every measured double is a row; without it doubles lacking both singles, and the ones
ESM-MSR would train rank-only, are dropped), plus `_nobb` with `--mut_backbones exclude` (mutant-background libraries left out of
train and val). `splits_oct06_capped` gives `oct06_capped`, `oct06_capped_allbl`, `oct06_capped_allbl_nobb`, ...

```bash
SPLIT=splits_oct06_capped; V1=oct06_capped; V=oct06_capped_allbl     # V1: without --keep_all_doubles, V: with
# ThermoMPNN-D and SPURS items tables (needs data/tsuboyama and data/$SPLIT.pkl); msr_venv
python inference_scripts/baselines/prepare/prepare_baseline_data.py --split $SPLIT --keep_all_doubles      # -> outputs/baselines/$V/data
python inference_scripts/baselines/prepare/prepare_baseline_data.py --split $SPLIT                         # -> outputs/baselines/$V1/data
# MutateEverything's on-disk format (data/<variant> in its checkout); the MSAs are data (msa_index.json comes from prepare/index_msas.py)
for v in $V1 $V; do
  ln -s <dir>/msa_index.json outputs/baselines/$v/data/msa_index.json
  MSR_BASELINE_DATA=$PWD/outputs/baselines/$v/data python inference_scripts/baselines/prepare/build_mutate_everything_data.py --mode build
done
```
Each prepared directory has a `variant.json` (name, split, flags): the training and inference scripts read it and stop if they are
asked for another split. ME stage 1 is trained on the `$V1` tables (its single mutations are the same rows as `$V`'s). ColabFold
output for library sequences without an MSA goes in `<data dir>/new_msas/<library>.a3m` (`--mode fasta` writes the sequences to
search; `MSR_ME_NEW_MSAS` moves the directory).

## 2. ThermoMPNN-D (seeds 1-3, both networks)

```bash
for s in 1 2 3; do for m in single epistatic; do
  MODEL=$m SEED=$s MSR_BASELINE_DATA=$PWD/outputs/baselines/$V/data SPLIT=$SPLIT sbatch training_scripts/baselines/slurm_thermompnn_train.sh
done; done
```
Each run (in `outputs/baselines/$V/thermompnn/<model>_seed<s>`, `BASE_OUT` moves it) writes `best_checkpoint.txt`; its content is the
`--single_ckpt` / `--epistatic_ckpt` of the inference script.

## 3. MutateEverything (seeds 1-3)

```bash
SEED=1 STAGE=single SPLIT_NAME=$V1 sbatch training_scripts/baselines/slurm_mutate_everything_train.sh
SEED=1 STAGE=double SPLIT_NAME=$V STAGE1_SPLIT=$V1 sbatch training_scripts/baselines/slurm_mutate_everything_train.sh
```
Stage 2 (`logs/af_double_${V}_seed<s>/checkpoint-99.pth` in the ME checkout) is the checkpoint to benchmark.

ME imports OpenFold from the copy installed in its conda environment, not from the checkout. The environment used here has an
OpenFold built in January 2025 whose Python source equals the checkout's except `openfold/data/data_pipeline.py`, which carries
diagnostic lines only (comments and a `print` of the deletion-matrix lengths when they disagree; no effect on any output).
`mutate_everything/openfold_debug_print.patch` is that difference: to build the identical package, apply it in the ME checkout
(`cd $MUTATE_EVERYTHING_DIR && patch -p1 < <repo>/training_scripts/baselines/mutate_everything/openfold_debug_print.patch`) before
`cd openfold && python setup.py install`; leaving it out gives the same numbers.

## What deviates from upstream, checked 2026-10-28

* **ThermoMPNN-D** (`df9a75a`, upstream main): the checkout has no modified file and the environment holds no copy of the code
  (`thermompnn` is imported from the checkout). The deviations are in this repository: `OctItemsDataset` replaces the dataset's
  `__init__` (our items tables instead of the raw CSV) and `load_cfg` fills in two paths; models, trainers, losses and configs are
  upstream's. Mutant-background Tsuboyama libraries: `prepare/prepare_baseline_data.py` writes `pdbs/<library>.pdb`, the wild-type
  AlphaFold backbone with the background residue renamed, so the sequence the structure implies is the background sequence;
  `OctItemsDataset` asserts that sequence equals the table's wild-type sequence for every library, in training, validation and
  inference alike (checked for all 15 libraries of the capped split: the structure differs from the wild type at exactly the
  background position).
* **MutateEverything** (`034dd91`, upstream master): no modified tracked file; the only deviation is the OpenFold install above.
  `inference_scripts/baselines/mutate_everything_inference.py` replaces upstream's evaluation entry (no detection metric,
  one process, a protein that runs out of memory is reported and skipped, not fatal), as `predict_dms.py` did.
  The older fork (`~/software/MutateEverything`, upstream `c380ea9` plus changes to `data.py`, `engine_train.py`, `metrics.py`,
  `misc.py`, `main_train.py`, `modeling/`) is not used by any of these runs. Its mutant-backbone support (a `mut_structure` column,
  the wild-type protein's MSA with the query residue overwritten at the background position, predictions keyed by library
  code) is replaced by data: each mutant-background library is its own protein (own `pdb_id`, fasta and MSA whose query is the
  background sequence; verified for all 15 libraries of the capped split), which upstream's `data.py` handles unchanged. Its
  `--additive_mode` is replaced by scoring the component singles (`testcomp`). Its OOM-swallowing multi decoder and NaN-tolerant
  metrics are not needed (see above; metrics are computed afterwards).
* **SPURS** (`70cbb99`, a non-default upstream branch, `fix/py311-inference-beta`): inference uses it unmodified; training uses it
  plus `spurs/spurs_retrain.patch` (a patch is a unified diff: per file, the changed lines with their surrounding context, applied
  with `patch -p1` inside a checkout; this one was made with `diff -ruN <pinned> <retraining tree>`, so applying it to the pinned
  checkout gives the retraining tree exactly).
* Not checked: other packages in the environments (`envs/` records their versions, not their contents).

## 4. SPURS and SPURS-multi (seeds 1-3, from the retraining of 2026-10-28; not re-run when this was written)

```bash
W=<work dir>; R=$PWD
cp -r $THIRD_PARTY_DIR/SPURS $W/SPURS_train && (cd $W/SPURS_train && patch -p1 < $R/training_scripts/baselines/spurs/spurs_retrain.patch)
mkdir -p $W/SPURS_train/data/checkpoints/ThermoMPNN/vanilla_model_weights
cp $THIRD_PARTY_DIR/ProteinMPNN/vanilla_model_weights/v_48_020.pt $W/SPURS_train/data/checkpoints/ThermoMPNN/vanilla_model_weights/
COMMON="datamodule._target_=items +datamodule.data_dir=$R/outputs/baselines/$V/data trainer.max_epochs=80 \
  callbacks.model_checkpoint.save_last=false model.encoder.pretrained_weights=data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt \
  model.legacy_esm_forward=true"
for s in 1 2 3; do
  NAME=$V/leg_seed$s SPURS_TRAIN_DIR=$W/SPURS_train OVERRIDES="$COMMON seed=$s train.seed=$s" sbatch training_scripts/baselines/slurm_spurs_train.sh
done
# once a single model has finished: the same seed's multi model starts from it
NAME=$V/leg_multi_seed$s CONFIG_NAME=train_multi SPURS_TRAIN_DIR=$W/SPURS_train \
  OVERRIDES="$COMMON seed=$s train.seed=$s model.init_from=$W/SPURS_train/logs/$V/leg_seed$s/checkpoints/best.ckpt" \
  sbatch training_scripts/baselines/slurm_spurs_train.sh
```
`legacy_esm_forward=true` keeps the ESM structural adapter at its random initialisation, as in the released checkpoint, whose adapter
never received a gradient; `false` trains it. Singles train on single + native_cond rows, SPURS-multi on the doubles. Only about 50
training libraries have doubles and 9 the validation split, so SPURS-multi's selected epoch (best around epoch 2) is noisy.

Benchmark them (the same-seed pair, never mixed): copy or link `logs/$V/leg_seed<s>/checkpoints/best.ckpt` and
`.../leg_multi_seed<s>/...` into a directory as `${V}_leg_seed<s>.ckpt` / `${V}_legmulti_seed<s>.ckpt`, then
```bash
MODEL=spurs MODELS=both SPLIT=$SPLIT SPURS_CKPT_DIR=<that dir> SPURS_SINGLE=${V}_leg_seed1.ckpt SPURS_MULTI=${V}_legmulti_seed1.ckpt OUT=runs/spurs_s1 \
  inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
```
(`MODELS=single` for the singles alone; SPURS-multi still needs `SPURS_SINGLE`, whose config it reads.)

## Scoring cautions

* Score Tsuboyama on **measured rows only** (`cens == 0`): the inference scripts predict the bounded rows too (flagged in `cens`), and
  including them inflates SPURS by about 0.005-0.007.
* The published SPURS checkpoint is leaky on this split (24 of the 33 capped test libraries were in its training data): the retrained
  checkpoints are the held-out comparison, the published one a reference.
