# Baseline: the released code on the capped split, with devel's link (optional) and validation metrics

Goal: track how the released training code (this branch's `main` objective, defaults and data pipeline) performs when scored with devel's
trusted metrics, and test conservative augmentations of it, starting with devel's monotone link.

* Recipe: `recipe_args.txt` = the `esm-msr-small` hparams (`LoRA_models/esm-msr-small/hparams.yaml`) with the canonical ranks WT 2 / alpha 4,
  MT 16 / alpha 16: dual adapters, the teacher-forced ensemble objective (`--lora_mode ensemble`, combined regression + rank), singles + doubles,
  micro-batch 16, bf16-mixed. The basis run is `bl_w2a4_m16a16_capped` (seed 1, 8 epochs, `--val_cycle_passes`, 2026-10-07).
* Split: `data/splits_oct06_capped.pkl` of `devel` (identity- and TM-capped), via `--split_file`.
* Cache: `cache_main_capped/` = symlinks to the old `cache_may22_twostep` (never modified) plus the libraries of the split that it lacked, built
  with the released dataset class (`build_cache_sharded.py`). The cache is unchanged by the link transplant (`dG_wt` is stamped at load time).
* Run: `baseline/run_baseline.sh NAME EPOCHS SEED [flags]`, or `baseline/run_with_retry.sh` to retry crashes.

## Transplanted from devel (2026-10-13)

* **Link** (`src/esm_msr/link.py`, `--link softclamp`, default `none` = the released objective exactly): observed ddG = h(dG_wt + latent) -
  dG_wt, h a learned soft floor / ceiling shared by every library (initial -1 / 5, temperature 0.5, `--link_lr` 5e-3, no warm-up, never cut by
  the plateau). Applied to every regression whose target is a measurement of the item: the WT regression on singles and the combined
  regression on singles and doubles. The two heads are combined on the LATENT scale, 0.5 (WT) + 0.5 (MT), and the link is applied once to that
  average, never to each head separately. The WT regression of a double onto the sum of its singles stays latent (a sum of two measurements is
  not a measurement of the double). Rank losses are untouched (h is monotone). Items whose library has no measured dG_wt leave the
  regressions. Approximation: teacher forcing puts the measured additive sum (observed singles) into the WT half of the combined latent;
  near the floor an observed single is a compressed version of its latent (main drops out-of-range variants, so few are affected).
  Not supported with the link: `--lambda_epi_combined`, mut_ctx and reversion items (config errors).
* **dG_wt** per library: `MegaScaleDatasetPreprocessor._wt_dG_by_code` (mean dG_ML of the 'wt' rows, keyed like `code`), stamped on every
  item by `Dataset.set_dG_wt` and collated as `batch['dG_wt']` (NaN for benchmarks).
* **Metrics** (`src/esm_msr/epi_metrics.py`, devel's module made symmetric): `val_epi_<metric>_<head>` for every metric (naive_rho,
  global_err, global_bias, beyond_rho, pair_rho, line_rho, cell_mag, cell_rank, cell_flipacc, cell_sigsd) and every head comb ((WT + ~MT)/2),
  mt (~MT), ctx ((WT + ~WT)/2, the WT-in-context control, needs `--val_cycle_passes`), plus add (WT) on the naive and global metrics. Guards
  as in devel: `val_rho_combined_avg`, `val_rmse_combined_avg` (observed scale under the link; benchmarks stay latent), `val_rho_wt_valid_avg`
  (WT on singles, mutant-background libraries left out). The released ddG metrics are still logged unchanged. Full definitions and the
  reading guide: `docs/validation_metrics.md` on devel. This replaces the earlier `epi_hierarchy.py` transplant.
* **Micro-batch size** changes speed and memory only (since `0330f78`: each ranking list is normalised by its own valid length;
  before, at more than one list per micro-batch, WT lists with masked doubles were reweighted). Micro 64 runs about 2.9x faster than
  the released 16 on the RTX 5090 (0.43 vs 0.15 it/s, GPU 88% vs 29%, about 26 GB). `tests/test_microbatch.py` checks 16 vs 64.
* **Dumps**: `val_dump_<tag>.npz` gains `dG_wt`, `subset_type` and `link_summary`. `baseline/epi_from_dump.py` rescores any dump of this branch
  (old ones get dG_wt from the raw file), e.g. the basis run, for a like-for-like comparison.
