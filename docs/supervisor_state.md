# Supervisor state (read this first after a context reset)

Written 2026-10-04 by the supervising instance. It records what exists, what is only designed or untested, and the traps. Where a statement is a
measurement it says so and where it came from; where it is inference it says that too. Details live in the documents named in each section.

## 0. Update 2026-10-05 (read `docs/hparam_review.md` first)

The hyper-parameter surface was audited and the dead parts removed on branch **`claude/cleanup-dead-params`** (one commit on top of `3f18100`; **not yet merged** into
`claude/censored-margin-ranking-devel` because `queue_lam.sh` was still running: bash re-reads a running script and `i3_lam300` loads the code at launch). The training
numerics are unchanged (bit-identical on 14 of 15 synthetic scenarios); retired flags are still accepted. Everything in sections 1-8 below that names a removed flag
(`--flip_align_units`, `--flip_pair_groups`, `--lora_mode`, `--detach_*`, `--subset_size`, `--freeze_wt_*`, `--reg_loss`, ...) describes the old CLI; `esm_msr.config.RETIRED_FLAGS`
says what replaced each one. Facts established since this file was written: GPU runs exist (W0, W2, L1, I1, I2; see §4 of the review), the plateau scheduler cut W2's learning
rate, and the adapters use rsLoRA (scale = alpha / sqrt(rank)).

## 1. Where the code is

| item | state |
|---|---|
| my branch | `claude/censored-margin-ranking-devel` in worktree `repo/.claude/worktrees/censored-margin-ranking-9b239c`, head `8cebcc1`, **14 commits ahead of `devel`** |
| `devel` | head `56c14ce` (someone else's work, unchanged by me since `43265f0`). **My 14 commits are NOT merged and NOT pushed.** Merging needs the user's go-ahead |
| what is on `devel` already | censored `ListMLELoss(score_mask)`, `--censor_floor`, cache v6 (`dG_meas`), `val_rho_epi_fast/_full`, FINDINGS 6c correction, first ANOVA report |
| what is only on my branch | consolidated censoring (`esm_msr/censoring.py`, cache v7), `val_rho_flip_pair`, `--link softclamp`, interaction loss + pair-group sampler + aligned micro-batches, `decompose_predictions.py`, ANOVA report corrections |
| tests | `PYTHONPATH=src /home/sareeves/miniconda3/envs/msr_venv/bin/python -m unittest discover -s tests` -> 87 pass (about 2 s, CPU only) |
| GPU | RTX 5090, idle at the time of writing. **Nothing from my branch has been run on a GPU.** Every claim about training behaviour is a design expectation |

Interpreter: `/home/sareeves/miniconda3/envs/msr_venv/bin/python`. Training env: `export PYTHONPATH=repo/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
from `/home/sareeves/playground/esm-msr-devel`. The canonical command is in `docs/epistasis_training_handoff.md` §2 (now with `--cache_path cache_v7`).

## 2. Traps (these have each cost time already)

1. **Cache path.** `CACHE_VERSION` is `v7`; the file name carries it. All 14 run scripts in the repo root (`_run_*.sh`, untracked, the run manager's) still say
   `--cache_path cache_v6`. Run against code with v7 and they will find nothing and silently **regenerate into the v6 directory** (CPU, minutes). Use `cache_v7`
   (`/home/sareeves/playground/esm-msr-devel/cache_v7`, 164 files, built and sanity-checked). The first epoch's log must say "Loading cached data", not "Generating".
2. **Background jobs.** The "completed" notification for a command started with `nohup ... &` is the *launcher shell* exiting, not the job. Wait on the job's log file
   (`until grep -aq 'DONE|Traceback' LOG; do sleep 15; done`), never on `pgrep -f` (it matches its own shell).
3. **Resumed runs split their log.** `training_logs/<run>/0/metrics.csv` holds only the resumed part; the earlier epochs are in `metrics_backup_ep2.csv`. Epoch-matched
   comparisons need both (this once made a no-flip control look like it was at epoch 0 when it was at epoch 3). The first validation row (step 0) is the untrained baseline.
4. **`--min_additive_dG`/`reg_ok`/`--subfloor_rank_only` are one mechanism**, not two (threshold + what to do with flagged items). Its per-library log line ("marked 0 of 965") is per library.
5. **Validation never sees training-side flags.** `--censor_floor`, `--min_additive_dG`, `--subfloor_rank_only` and `--flip_pair_groups` do not change the validation sets. `--include_out_of_range`
   does add censored items to validation, but every existing metric is computed on uncensored items only.
6. **Old checkpoints** lack the new hparams; the code reads them with `.get(..., default)`. Loading a link-trained checkpoint for inference ignores the link (keys `link_head.*`): inference returns the latent ddG.
7. `esm_msr_testing.py` reads `ddG_ML` through `MegaScaleDatasetPreprocessor`, whose default (`include_out_of_range=False`) keeps its truth exactly as before; training turns it on.

## 3. What each new flag does (all default off; flags off = previous behaviour, bit-identical rank loss, tested)

| flag | effect | requires / notes |
|---|---|---|
| `--include_out_of_range` | keep variants the assay reports only as `<-1` (dead) / `>5` (hyperstable) as **censored** items (singles for the WT head, doubles -> `cond` items for the MT head) | `--rank_loss listmle`; cache v7 |
| `--censor_reg_weight W` (1.0) | weight of the one-sided hinge for censored items in regression; 0 = rank only | |
| `--censor_floor F` | numeric flip-column items with measured dG <= F become lower-censored in the MT flip loss (rank tie at the bottom) | regression unchanged unless `--censor_floor_hinge` |
| `--link softclamp` | latent ddG -> observed dG through one monotone soft floor/ceiling; regression on the observed scale | **no `--shared_bias_init`**; legacy `--lambda_*_combined` raise; `reg_ok` ignored; items without `dG_wt` leave the regression |
| `--link_lo/-hi/-tau`, `--link_learn_bounds`, `--link_lr` | link init and optimizer group (own group `link`, no weight decay) | defaults -1, 5, 0.5, True, 5e-3 |
| `--lambda_int_mt L` | interaction-only loss: double-centred predictions vs double-centred measurements per position-pair matrix | needs `--flip_pair_groups` > 0 and `--flip_align_units` to be useful |
| `--flip_pair_groups N` | sampler packs up to N columns of one position pair together (balanced sizes) | 3 keeps a unit inside one 64-row micro-batch |
| `--flip_align_units` | MT micro-batches cut at pair boundaries, never mid-column | also fixes the ~10% of within-column pairs lost to fixed cuts (estimate) |
| `--int_min_rows/--int_min_cols` (4, 2) | minimum complete block for the interaction loss | |

**Validation metrics (trimmed 2026-10-05, commit 1b59176; runs before that carry the older, larger set).** Per protein: `rho_combined`, `rmse_combined`, `rho_flip_pair` (doubles only). Library-equal averages (`val_*_avg`): `rho_combined` (checkpoint name, plateau scheduler), `rmse_combined`, `rho_wt_valid` (WT guard), `rho_mt_valid`, `rho_epi_full`, `rho_colrank`, `rho_flip`, `rho_flip_pair` (early-stop monitor), `auc_dead_wt`, `auc_dead_mt`. Pooled over all libraries (`val_*_pooled`): `rho_combined`, `rmse_combined`, `rho_epi_full`, `rho_pair_offset`, `rho_row_effect`, `rho_col_effect`, `rho_colrank`, `rho_flip`, `rho_flip_pair`, `auc_dead_wt`, `auc_dead_mt`; plus `val_n_flip_pair_matrices`. Dropped from logging (still computed in `stats.py`): all `rho_epi_fast*`, `*_latent`, `rho_wt_all`, `rho_mt_all`, `auc_hyper_*` (n = 2 and about 25 positives: not interpretable).
Two ladders. A (nested, saturation-free): `rho_epi_full` -> `rho_colrank` (within-column rank, nothing centred; what the rank loss optimises) -> `rho_flip` (pair-specific row effects + interaction) -> `rho_flip_pair` (interaction only). B (isolations on measured dddG): observed-scale `rmse_combined` (saturation, the link) -> `rho_pair_offset` -> `rho_row_effect` / `rho_col_effect` -> `rho_flip_pair`. Zero-shot (step 0) pooled: pair offset 0.31, row 0.42, column 0.46, flip_pair 0.12, colrank 0.13, epi_full 0.49.
`rho_epi_fast` under the link was on a saturated baseline and went negative at step 0 (-0.11 pooled); its latent twin equals the no-link value (0.579). `link/floor` and `link/ceiling` are no longer logged (ceiling == hi exactly, floor == lo to ~1e-3).

How to tell from a run's log that a feature is active: `train/cens_lower_items`, `train/cens_upper_items`, `train/L_reg_wt_cens`, `train/L_reg_mt_cens` (censoring); `link/lo`, `link/hi`, `link/floor`,
`link/ceiling`, `link/tau_*` and group `link` in `lr/` and `norm_grad/` (link); `train/L_int_mt` (interaction loss; zero or absent means no complete matrix reached it).

## 4. Run plan (nothing launched yet) — see `docs/epistasis_training_handoff.md` §2e and §2f

All with the canonical command and `--cache_path cache_v7`, 3 epochs first, same seed, then replicate across seeds.

| arm | extra flags | purpose |
|---|---|---|
| W0 | none | regression guard: should reproduce `v6_anchor100` on the v7 cache |
| W1 / W2 / W3 | `--include_out_of_range --censor_reg_weight 0` / `--include_out_of_range` / W2 + `--censor_floor 0.5` | censored dead/hyperstable singles and doubles: rank only, then + hinge, then + floor |
| L0 / L1 | `--link softclamp` (no `--shared_bias_init`) / L0 + `--include_out_of_range` | link alone, then with censoring |
| I0 / I1 | L1 + `--flip_pair_groups 3 --flip_align_units` / I0 + `--lambda_int_mt 1.0` | sampler change alone, then the interaction loss |

Guards (must not regress against W0): `val_rho_wt_valid_avg`, `val_rmse_combined_avg`. Targets: `val_rho_flip_pair_avg`; `val_auc_dead_wt`/`val_auc_hyper_wt` above chance and above W0.
Use at least 2 seeds per arm before believing a difference: epoch-to-epoch noise of `val_rho_flip_avg` is about +/-0.02-0.03.

## 5. Measured results so far (source in brackets)

* **Rank loss and censoring are unproven.** Epoch-matched `val_rho_flip_avg`: no-flip control 0.204/0.207/0.205/0.228 (epochs 0-3); flip arm seed 1: 0.174/0.176/0.213; seed 2: 0.142/0.162/0.173/0.219. Late-epoch values overlap
  (0.18-0.23). Censored arms reach 0.22-0.24 at epochs 4-7, but the no-flip control was never run that long [training_logs, FINDINGS 6c].
* **Capability test** (forward only, epoch-2 checkpoints; reproduces logged validation values): flip loss vs none, validation 0.214 vs 0.199 (paired +0.015, 6 of 16 libraries better); training libraries 0.359 vs 0.310
  (paired +0.049, 19 of 21 better, p=0.004). The loss works on data it trained on and barely transfers [scratchpad `capability_test.py`, reported in chat and the ANOVA report].
* **Gradient scale.** `norm_grad/lora_mt` is about 1-86 (typically ~30) with the rank loss and about 0.3-0.5 without; `L_rank_mt` about 19-35 vs `L_reg_mt` about 0.2-1.7 [logs]. The rank loss is a per-list sum, so at `--lambda_rank_mt 1.0`
  it is effectively about 14x a per-item loss. A per-member normalisation of the flip loss was proposed, **not implemented**. A log-scale lambda sweep (0.1, 0.03) is the cheaper test.
* **ANOVA of measured doubles** (`docs/anova_epistasis_report.md`, 127,476 doubles; 166,996 with out-of-range variants clipped): identity-independent structure explains about 90% of dddG variance out-of-sample
  (saturation 60%, +pair offset 16%, +row/column 13%); the leftover is 10.4% (8.6% clipped). Noise: replicate rows of the same mutant SD 0.27 (4,552 mutants) agrees with trypsin-vs-chymotrypsin (0.22-0.31); synonymous wild-type copies (0.115)
  understate it (deep coverage). So identity-specific interaction is about 0-5% of the variance, most likely about 2%. Earlier reports in this session gave 0-5%, then 0-9%; the current text is the corrected one.
* **The pooled flip metric is partly earnable without the partner residue.** A predictor knowing the scored substitution and partner position earns 0.137 (3 random splits), trained models 0.20-0.21; per-pair version: exactly 0.
* **Data census** (v7, training libraries): singles 104,775 ordinary / 5,114 dead / 4,312 hyperstable; `cond` 111,364 / 6,796 / 14. Validation singles 23,759 / 1,451 / 283; validation doubles 9,419 / 3,752 dead.
  Four libraries (2KRS, 2KT8, 2LYP, 5GU9) have a wild type that is itself out of range: no numeric `dG_wt`, so their `>5` variants are rank-only anchors with no hinge.
  Unmeasured singles (`-` or absent) are **not** treated as dead: positions with them are not enriched for dead neighbours (4.5% vs 4.0%).
* **Interaction-loss yield on real batches** (40 batches, `--flip_pair_groups 3 --flip_align_units --censor_floor 0.5`): about 51 flip rows per micro-batch, 0.74 complete matrices and 30 cells reach the loss, 58% of flip rows land in a usable
  block. A thin signal per step.
* **Bias terms** in earlier runs were small (WT about +0.06-0.08, MT drifting to -0.08 to -0.21 kcal/mol); a no-bias arm (`v6_censor05_anchor100_nobias`) had logged one epoch (0.207) when last looked at.

## 6. Design facts a reader would otherwise have to rediscover

* Censoring is one mechanism (`esm_msr/censoring.py`): per item `cens` (-1 / 0 / +1), `cens_src` (1 assay range, 2 floor), `cens_bound` on the item's own ddG scale (`dG_bound + (ddG - dG_meas)`). Lower-censored items tie at the bottom of a list in forward Plackett-Luce,
  upper-censored items at the top in reverse Plackett-Luce, both-sided lists average the two (`ListMLELoss.forward_censored`). Only plain singles censor the WT head; a censored single never gives a double an additive expectation.
* The link is `h(z) = hi - tau_hi*softplus((hi - u)/tau_hi)` with `u = lo + tau_lo*softplus((z - lo)/tau_lo)`; the plateaus differ slightly from `lo`/`hi` when a temperature is not small against the span (reported as `link/floor`, `link/ceiling`).
  A conditional item is scored as its double: `h(dG_wt + bg_offset + latent) - dG_wt`, `bg_offset = ddG_AB - ddG(A|B)`, target `ddG + bg_offset`.
* The interaction loss is a function of total predictions only (double-centring), so no second output head exists; it does not remove identity-independent effects from the model, which are still learned through the ordinary regression.
  Not built: projecting the MT output onto the interaction subspace (it would need another head to carry the rest).
* The WT head trains on singles only in the canonical command (the `double` subset cap is 0).

## 7. Open items and decisions that are the user's

1. Merge and push my 14 commits to `devel`? (not done; needs go-ahead). After that, the run scripts must change `cache_v6` -> `cache_v7`.
2. Which arms to run first, and whether to flip `--link` to the default after L0 vs W0.
3. Per-member normalisation of the flip loss and a lambda sweep: not done.
4. A double-specific noise estimate: open (raw sequencing counts in the Zenodo record, `Raw_NGS_count_tables.zip`, were not opened).
5. Not implemented, and worth deciding on: oversampling censored singles in WT lists (about one censored item per 16-item list now); top-tied handling beyond the two-pass approximation; `--link` effects on inference (it returns latent ddG).

## 8. Where things are

* Reports: `docs/anova_epistasis_report.md` (+ figures `docs/anova/`), `FINDINGS.md`, `docs/epistasis_training_handoff.md`.
* Scripts: `analysis_notebooks/anova/` (`anova_variance_partition.py`, `flip_splithalf.py`, `make_figures.py`, `decompose_predictions.py`).
* Scratch (not in git, session-specific): cache build `build_v7.py`, capability test `capability_test.py`, real-data smoke `smoke_real.py` under the session scratchpad.
* Logs: `/home/sareeves/playground/esm-msr-devel/training_logs/<run>/0/`. Caches: `/home/sareeves/playground/esm-msr-devel/cache_v7` (the v6 directory still exists for older code; v4 and v5 were no longer present when this was written).
