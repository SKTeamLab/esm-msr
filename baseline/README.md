# Baseline: the released code on the capped split, with the new validation stats

Goal: track how the released training code (this branch's `main` objective, defaults and data pipeline, unchanged) performs when scored with the
epistasis hierarchy of `devel` (`docs/validation_metrics.md` there; `src/esm_msr/epi_hierarchy.py` here is the same file).

* Recipe: `recipe_args.txt` = the `esm-msr-small` hparams (`LoRA_models/esm-msr-small/hparams.yaml`, run `msr_dual_ensemble_small_large_r1a4`):
  dual adapters, rank 1 / alpha 4, the teacher-forced ensemble objective (`--lora_mode ensemble`, combined regression + rank), singles + doubles,
  micro-batch 16, bf16-mixed, seed 1. The key shipped in that file is not used.
* Split: `data/splits_oct06_capped.pkl` of `devel` (identity- and TM-capped), via `--split_file`.
* Cache: `cache_main_capped/` = symlinks to the old `cache_may22_twostep` (never modified) plus the libraries of the split that it lacked, built with the
  released dataset class (`build_cache_sharded.py`); every file holds all four subsets, as the old ones do.
* Transplanted (validation side only): `epi_hierarchy.py`, `MSRModel.forward_cycle`, `--val_cycle_passes`, pooled `val_epi_<level>_<head>` logging and
  a `val_dump_<tag>.npz` per epoch. Nothing in training, in the existing metrics or in the schedulers is touched.
* Predictions are taken on the observed scale as they are (this model has no link).
* Run: `baseline/run_baseline.sh NAME EPOCHS SEED [flags]`, or `baseline/run_with_retry.sh` to retry crashes.
