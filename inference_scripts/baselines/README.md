# Baseline inference on every benchmark, with resource tracking

One script per baseline, each runs **every** benchmark (Tsuboyama val + test, the 12 external sets, the 7 DMS, Domainome,
ProteinGym), records per-dataset and per-protein wall time, CPU, RAM, VRAM and I/O of the inference calls
(`../resource_tracker.py`, [`docs/INFERENCE_RESOURCES.md`](../../docs/INFERENCE_RESOURCES.md)), and writes predictions in the
same row order for the same rows, so any two baselines can be scored against ESM-MSR's with the same tools. ESM-MSR itself is
`../esm_msr_testing.py`.

| script | model | what one file covers |
|---|---|---|
| `thermompnn.py` | ThermoMPNN / ThermoMPNN-D (also `train`) | `--single_ckpt` and/or `--epistatic_ckpt`; `--mutants single\|multi\|both`; modes `all`, `test`, `external`, `dms`, `domainome`, `pgym` |
| `mutate_everything.py` | MutateEverything | any checkpoint (`--resume`: stage 1 singles, stage 2 doubles); `--mutants`; `--sets all` or any group / set names |
| `spurs.py` | SPURS and SPURS-multi | `--models single\|multi\|both`; `--mutants`; `--checkpoint_single/multi` |
| `proteinmpnn.py` | ProteinMPNN `v_48_020` | `--mutants`; `--n_samples`, `--coord_noise` for the test-time ensemble |

`--mutants single|multi|both` chooses which rows are scored (the others are NaN and cost nothing). With one network or model
missing, the readouts that need it are NaN and the combined readout falls back to what exists (ThermoMPNN: additive from the
single network; SPURS: its own four columns).

## Reproducing

1. **Upstream code**, unmodified, at the commits in `third_party.lock`: `./setup_third_party.sh` (clones into
   `$THIRD_PARTY_DIR`, default `~/software`, and checks out the pinned commit; warns if a tracked file was modified). These
   scripts are the only additions to those repositories: nothing is patched.
2. **Environments** in `envs/` (what the October runs used): `msr_venv.yml` (ESM-MSR and ProteinMPNN), `spurs_higher.yml`,
   `mutate_everything.yml`, `thermompnn_venv_requirements.txt` (a venv in the ThermoMPNN-D checkout).
3. **Where things are**: every location comes from `paths.py` and can be moved with an environment variable
   (`THIRD_PARTY_DIR`, `THERMOMPNN_DIR`, `MUTATE_EVERYTHING_DIR`, `SPURS_DIR`, `PROTEINMPNN_DIR`, `MSR_DATA_DIR`,
   `MSR_STRUCTURES_DIR`, `MSR_DOMAINOME_TABLE`, `MSR_BASELINE_DATA`, `MSR_PGYM_INPUTS`, `MSR_PGYM_STRUCTURES`). The
   preprocessed tables (`data/preprocessed/*.csv`) carry the structure prefix of the machine that built them;
   `paths.fix_pdb_path` points them at `MSR_STRUCTURES_DIR`.
4. **Inputs that are data, not code**, and so not in the repository: MutateEverything's weights
   (`models/finetuning_ptm_2.pt`, from its README) and its prepared inputs under `$MUTATE_EVERYTHING_DIR/data/`
   (`prepare/build_*_me_data.py` write them from the preprocessed tables; the MSAs they reuse are ColabFold output, searched
   by `prepare/find_dms_msas.py` / `index_msas.py`, locations in `MSR_ME_*` variables), SPURS's weights (Hugging Face hub),
   the ThermoMPNN-D item tables (`prepare/prepare_oct06_data.py`, `MSR_BASELINE_DATA`) and checkpoints, the ProteinGym
   inputs (`preprocessing/pgym_preprocess.py`) and structures.
5. **Run**, in the fixed gpu4 slot (1 GPU, 6 CPUs, 60 GB), one job per model:

```bash
MODEL=thermompnn SINGLE=/path/single.ckpt EPISTATIC=/path/epistatic.ckpt OUT=runs/thermompnn \
    inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
MODEL=mutate_everything CKPT=logs/af_double_oct06_capped_allbl_seed1/checkpoint-99.pth OUT=runs/me \
    inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
MODEL=spurs OUT=runs/spurs inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
MODEL=proteinmpnn OUT=runs/proteinmpnn inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
MODEL=esm_msr CKPT=release-baseline/s1/epoch=04-val_rho_combined_avg=0.855.ckpt OUT=runs/esm_msr \
    inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
python analysis_scripts/resource_report.py runs/*/resources --out runs/summary.csv     # the resource table
```

Each script also runs by hand (`python thermompnn.py all --help`).

## Outputs

`$OUT/predictions` per benchmark (`external_<set>.csv`, `dms_<assay>.csv`, `domainome.csv`, `tsuboyama_{val,test}.csv`, ProteinGym's
per-DMS CSVs and `summary.csv`; ME writes upstream's `results_<set>.csv`) and `$OUT/resources/<run_id>/`. Positive means
stabilizing in every prediction column, except that MutateEverything's `ddg_pred` is upstream's ddG (positive = destabilizing,
`ddg_pred = -scores`), as in its October files.

## What changed from the October scripts (so older predictions are not directly the same runs)

* **ThermoMPNN**: batches never span proteins (`--mixed_protein_batches` restores the old behaviour); epistatic checkpoint
  optional; one process for `all`.
* **MutateEverything**: one GPU, one process (upstream's `--dist_eval` sharded over 3); the loader runs in-process; the metric
  that cannot run on the DMS is dropped, as in the October `predict_dms.py`.
* **SPURS**: the four per-block copies of the scoring loop are one function; DMS rows come from the complete tables (the old
  script read the copy of GB1 that lacks V39A/V39C/V39D); Tsuboyama mutant-background libraries are scored on their own
  background sequence, as in the harness rerun; Domainome's target is `scaled_fitness` as in `esm_msr_testing.py` (the old
  script negated it before flipping the prediction).
* **ProteinMPNN**: every block uses the batched `score_unmasked_fast` that ProteinGym used (one background order per position,
  seed 0) instead of the row-by-row scorer; mutant-background libraries are scored on their background sequence; the legacy
  `functional` block (data outside the repository) is gone. The row-by-row scorers stay in the file as the reference.

## Not in this directory

Training launchers (`train_oct06.sh`, `slurm_thermompnn_train.sh`; ThermoMPNN training itself is `thermompnn.py train`), the
scoring and epistasis scripts that turn predictions into the tables (`score_baselines.py`, `epi_*.py`) and the audit scripts
live outside the repository on the cluster (`~/esm-msr-devel/baselines/`) and are not needed to produce predictions.
