# Handoff: training runs for the interaction objective

Operational brief. Read `FINDINGS.md` for *why*; this is *what to run, what to watch, what to
decide*. Builds on `docs/training_handoff.md`, which still describes the environment, cache,
splits and cost model — none of that changed.

## 1. What changed in the code

Four behavioural changes. Two of them alter results at default settings, so a run launched
with the old command will not reproduce the old numbers.

| flag | old | new | effect |
|---|---|---|---|
| `--lambda_rank_mt` | did not exist | **1.0** | New within-column rank loss on the MT pass. **Changes training at defaults.** |
| `--lambda_reg_mt` | 0.0 | **1.0** | Was always passed as 1.0 by the canonical command, so this only makes the defaults self-consistent. |
| `--flip_list_min` | — | 4 | Minimum members for a flip column to contribute. |
| `--flip_group_units` | — | **on** | Orders MT work-unit rows by flip column so micro-batches hold whole columns. Measured: 2.9× more items reach the loss. |
| `--subfloor_rank_only` | — | **on** | Sub-floor doubles are now *marked* rather than *dropped*. **Changes the item count**: `min_additive_dG` no longer removes items, so train size goes back up to roughly the unfiltered ~233k. |

Also new, non-behavioural: `val_rho_flip` validation metric, `epistasis_pred` inference
column, `flip_key` and `reg_ok` fields on cached items.

> **The cache must be rebuilt**, and `CACHE_VERSION` was bumped `v4` → `v5` so this happens
> on its own: the version is part of each cache filename, so a stale `v4` pickle can no longer
> be loaded silently by a run expecting the new item schema. Point `--cache_path cache_v7` and
> it regenerates; ~4 minutes for the full 404 libraries (0.4–0.8 s/library). The old `v4`
> files are untouched and still serve runs on the previous commit.
>
> **Verify anyway before a long run:** the first epoch's logs must show a non-zero
> `train/rank_mt` and a finite `val_rho_flip_avg`. Non-zero `train/rank_mt` is the single
> check that the new objective is actually active.

## 2. Canonical command

As in `docs/training_handoff.md`, with these changes: new cache path, the two new loss terms
made explicit, and the monitor switched.

```bash
cd /home/sareeves/playground/esm-msr-devel
export PYTHONPATH=repo/src HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python

"$PY" repo/src/esm_msr/training.py \
  --experiment_name RUN_NAME --version 0 \
  --raw_data_file '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv' \
  --af_model_folder '/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs' \
  --split_file '/home/sareeves/software/esm-msr/data/hyperopt_splits.pkl' \
  --cache_path cache_v7 \
  --benchmark_data_path repo/data/preprocessed \
  --checkpoint_path training_checkpoints --log_dir training_logs \
  --num_epochs 8 --seed 1 \
  --dataloading cycle --loader_strategy all \
  --subset_caps single=None cond=None native_cond=None \
  --min_additive_dG -1.0 --subfloor_rank_only \
  --lora_rank_wt 2  --lora_alpha_wt 4  --lora_dropout_wt 0.1 --target_mode_wt expanded \
  --lora_rank_mt 16 --lora_alpha_mt 16 --lora_dropout_mt 0.1 --target_mode_mt expanded \
  --incl_sequence_head_wt --incl_sequence_head_mt \
  --adapter_mode dual --lora_mode ensemble \
  --lambda_reg_wt 1.0 --lambda_rank_wt 1.0 \
  --lambda_reg_mt 1.0 --lambda_rank_mt 1.0 --flip_list_min 4 \
  --mt_single_anchor_weight 0.5 --cond_weight 0.5 --native_cond_weight 1.0 \
  --reg_loss mse --precision bf16-mixed \
  --batch_size 256 --micro_batch_size 64 --subset_size 16 \
  --learning_rate 2e-4 --lr_warmup_steps 500 \
  --shared_scale_init 0.3 --shared_bias_init 0 --detach_ensemble_input \
  --num_workers 4 --log_every_n_steps 25 --save_top_k 3 \
  --monitor_metric val_rho_flip_avg --monitor_mode max
```

### Why the monitor changes

`val_rho_combined_avg` cannot see the interaction channel. In simulation, conventional ΔΔΔG
correlation returns **+0.90 when neither side has any interaction at all** — it is dominated
by the assay's saturating response, which both the measurement and any competent model share.
Selecting checkpoints on it is selecting for ΔΔG calibration, which is fine for the ΔΔG output
and uninformative about the MT adapter.

`val_rho_flip_avg` is exactly zero for an additive readout and unaffected by any monotone
assay distortion, so it is the only logged number specific to what the MT adapter exists for.

**Keep watching `val_rho_combined_avg` and `val_rmse_combined_avg` anyway.** They are the ΔΔG
product metrics, and a run that gains flip ρ while losing ΔΔG calibration is not an
improvement. If they diverge, say so rather than picking one.

## 2b. RESOLVED: Column-aware sampler boosts flip throughput

**Implemented & Verified.** The batch sampler (`ProteinCyclingBatchSampler` in `esm_msr/data.py`)
now uses a dual-queue column-aware bin-packing algorithm:
1. Groups items surviving caps into whole flip columns (`flip_key != ''`) and non-column items (singles).
2. Calculates each library's natural flip share `flip_budget = round(batch_size * (N_flip / N_grand))`.
3. Packs whole columns intact into each batch up to `flip_budget` (or more if singles are depleted), and fills the exact remainder with non-column singles to hit `batch_size = 256` exactly.
4. Preserves batch count deterministically (`N_grand // batch_size`) with 0.00% column splitting across libraries.

**Measured Comparison** (RTX 5090):

| config | steps with flip | columns/step | mean column length | **items/step** | Epoch 1 `val_rho_flip_avg` |
|---|---|---|---|---|---|
| `--no-flip_group_units` | 25 / 29 | 1.00 | 4.08 | **4.1** | — |
| `--flip_group_units` (old sampler) | **29 / 29** | 2.59 | 4.65 | **11.8** | ~0.141 baseline |
| **Column-Aware Sampler** (new) | **All steps** | **2.5 – 4.0** | **16.0 – 19.0** | **35 – 64** | **0.169** |

**Consequences for tuning & validation:**
* Mean column length jumped from 4.65 to **16–19**, matching actual biological column sizes (~19).
* Items entering the flip loss per step rose by ~4× (up to 64 of 64 micro-batch rows).
* First-epoch validation on the canonical run (`v5_flip_baseline`) reached **`val_rho_flip_avg = 0.169`**, while preserving ΔΔG headline performance (`val_rho_combined_avg = 0.800`, `val_rmse_combined_avg = 0.740` kcal/mol).
* Thresholds like `--flip_list_min 4` can now safely be retained or raised without starving the loss.

Diagnostics logged every step to watch: `train/flip_cols` (columns used),
`train/flip_len` (mean members per column), `train/flip_items` (rows actually entering the
loss).

## 2c. The MT single anchor interacts with this, and is untested

Anchored singles have no `flip_key`, so they never contribute to the flip loss — but they
occupy roughly half of all MT backbone rows at the canonical
`--mt_single_anchor_weight 0.5`, which directly reduces items/step above. There is also a
conceptual tension: the anchor teaches the MT pass to reproduce wild-type-context single
effects, which is the *additive* component — precisely what the flip metric double-centres
away. It should not fight the flip objective, but it does spend capacity and slots on
something the flip objective does not need.

**I did not get to measure this.** The runs were queued and did not finish. The mechanical
prediction is that lowering the anchor raises `train/flip_items` roughly in proportion to the
freed slots; whether that converts into better `val_rho_flip_avg` is the open question, and
whether it costs `val_rho_mt_valid_avg` (which the anchor exists to protect) is the risk.

**Add this to the run matrix** as a first-class sweep rather than treating 0.5 as settled:

| run | override | watch |
|---|---|---|
| `v5_anchor050` | `--mt_single_anchor_weight 0.5` (canonical) | the reference |
| `v5_anchor025` | `--mt_single_anchor_weight 0.25` | `train/flip_items` should rise |
| `v5_anchor000` | `--mt_single_anchor_weight 0.0` | also ~1.9× faster; check `val_rho_mt_valid_avg` for the cost |

If 0.25 or 0 wins on `val_rho_flip_avg` without hurting `val_rho_mt_valid_avg` or
`val_rmse_combined_avg`, lower the canonical default. Note `--mt_single_anchor_weight 0`
requires nothing else to change, but the assertion tying it to `--lambda_reg_mt > 0` means
you cannot zero both.

## 2d. Censored ranking (`--censor_floor`) — implemented, NOT yet run on GPU

`--subfloor_rank_only` withholds sub-floor items from the regression but leaves them in the
flip rank loss, where `ListMLELoss` trains the model to reproduce their mutual order. That order
is noise (pinned or unidentifiable fits). `--censor_floor F` fixes it with a censored
Plackett–Luce likelihood:

* A flip-column item is **censored** when its *measured* dG (`dG_meas` = library `dG_wt` +
  total ddG, cached since **v6**) is `<= F`. Items without a finite `dG_meas` are never censored.
* Censored items stay in each column's denominators, so every above-floor item is still pushed
  above every floor item, but they contribute no numerator term: **their order among themselves
  costs nothing** (exact in the loss; unit-tested by permuting them).
* A column needs >= `--flip_list_min` members in total *and* >= 1 uncensored member.
* Measured, not additive: genuine compensators (additive prediction below the floor, measured
  above it) keep their ordering. `--subfloor_rank_only` is unchanged and independent.
* Scope: only the MT flip loss. `ListMLELoss.score_mask` is opt-in (default `None`); the WT /
  combined rank losses never pass it and are bit-identical to before (verified against the
  previous implementation on loss and gradient, and by a test that only `_compute_flip_loss`
  passes it). Requires `--rank_loss listmle`.
* Default is **off** (`None`), so runs without the flag reproduce v5 behaviour on the v6 cache.
* New diagnostic `train/flip_uncensored`: members that actually carry order information.
  `train/flip_items` still counts all column members and now overstates signal when censoring.

Floor choice is open. Share of `cond` items censored in cache_v6: **0.0% at -1.0**, **11.8% at 0.0**, **28.0% at +0.5** (practical floor, §5 of
FINDINGS). Suggested sweep once a GPU slot is free, vs. run **A**:

| run | override | question |
|---|---|---|
| **F0** | `--censor_floor 0.0` | Does ignoring floor-pinned order beat A on `val_rho_flip_avg`? |
| ~~F1~~ | ~~`--censor_floor -1.0`~~ | **Useless**: 0.0% of cond items in cache_v6 have measured dG <= -1.0 (the fit's bound is never reached in the stored values). Use 0.0 and 0.5. |
| **F2** | `--censor_floor 0.5` | Aggressive: the practical floor. |

Decision rule: F > A on flip with `val_rmse_combined_avg` flat -> adopt the best floor.

## 2e. Consolidated censoring (`--include_out_of_range`, `--censor_floor`) — implemented, not yet run on GPU

Everything that "only bounds" a measurement now goes through one mechanism, `esm_msr/censoring.py`. Each item carries
`cens` (-1 lower-bounded, +1 upper-bounded, 0 ordinary), `cens_src` (1 assay range, 2 floor) and `cens_bound` (the bound on
the item's own ddG scale). Losses read only those fields.

| source | what it is | flag | rank losses | regression |
|---|---|---|---|---|
| assay range, `<-1` | variant confidently below the assay range ("dead"); only a bound | `--include_out_of_range` | tied at the bottom of their list | hinge: penalised only if predicted above the bound |
| assay range, `>5` | variant confidently above it ("hyperstable"); only a bound | `--include_out_of_range` | tied at the top | hinge: penalised only if predicted below the bound |
| measured floor | numeric dG at or below `F` (flip-column items) | `--censor_floor F` | tied at the bottom of the flip column | unchanged (regresses on the value) unless `--censor_floor_hinge` |
| additive floor | double predicted below `--min_additive_dG` | `--subfloor_rank_only` (unchanged) | untouched | withheld (`reg_ok=False`) |

* **Rank loss.** `ListMLELoss.forward_censored`: lower-censored members are exact in forward Plackett-Luce (drop their own
  terms, keep them in the denominators); upper-censored members are exact in *reverse* Plackett-Luce (worst first). A list with
  both averages the two passes. With no censored member it equals the old loss exactly (verified bit-for-bit on loss and
  gradient, and by tests). It applies to the WT head's lists (singles only), the MT flip columns, and nothing else.
* **WT head.** Only single mutants censor the WT head: its target for a double is the additive sum of its singles, and a censored
  single never contributes to that sum (a censored single is only a bound, so doubles built on it have no additive expectation
  and no dddG). With the flags off, every WT code path is unchanged.
* **MT head.** A dead or hyperstable *double* makes its two conditional (`cond`) items censored with the bound shifted by the
  partner single, so they join the flip columns as tied-bottom / tied-top members.
* **Weights.** `--censor_reg_weight W` scales the hinge terms relative to ordinary items (0 leaves censored items in the rank
  losses only).
* **Cache.** `CACHE_VERSION` is now **v7**; use `--cache_path cache_v7`. It always contains the out-of-range items, and the dataset drops
  them at load unless `--include_out_of_range`, so one cache serves runs with and without the flag. Evaluation scripts that read
  `ddG_ML` through `MegaScaleDatasetPreprocessor` are unaffected (its default drops those rows).
* **Not changed:** `-` (no estimate) and singles with no row at all are *not* treated as dead. Positions with such missing singles
  are not enriched for dead neighbours (4.5% of their other substitutions are dead vs 4.0% elsewhere).
* **Validation** scores every existing metric on the uncensored items only, so it stays comparable. New: `val_auc_dead_wt`,
  `val_auc_hyper_wt` (singles) and `val_auc_dead_mt`, `val_auc_hyper_mt` (conditional items): P(a random ordinary item is
  scored above a dead one) and P(a random hyperstable item is scored above an ordinary one), 0.5 = chance.
* **New diagnostics:** `train/cens_lower_items`, `train/cens_upper_items` per batch; `train/L_reg_wt_cens` and `train/L_reg_mt_cens`
  for the hinge terms.

Requires `--rank_loss listmle`. Floor censoring (`--censor_floor`) still applies only to MT flip columns.

**What the v7 cache holds** (training libraries, items by censoring): singles 104,775 ordinary, **5,114 dead**, **4,312 hyperstable**;
conditional items 111,364 ordinary, 6,796 lower-censored, 14 upper-censored (conditional items inherit a double's censoring, with the bound
shifted by the partner single); native-conditional 17,435 / 254 / 89. Validation: singles 23,759 ordinary, 1,451 dead, 283 hyperstable; doubles
9,419 ordinary and 3,752 dead.
Four libraries (2KRS, 2KT8, 2LYP, 5GU9) have a wild type that is itself above the assay range, so there is no numeric `dG_wt` and no bound on the
ddG scale. Their `>5` variants (3,406 of the hyperstable singles) are kept as **rank-only** anchors: they outrank every numeric variant of their
library, they get a placeholder label that no loss reads, and they get no hinge (`cens_bound` is NaN).

Suggested arms (3 epochs first, same seed and cache), each differing from the previous by one switch:

| arm | name | flags beyond the canonical command | question |
|---|---|---|---|
| W0 | `v7_ref` | *(none)* | Reproduces the v6 reference (`v6_anchor100`) on the v7 cache: a pure regression guard. |
| W1 | `v7_oor_rank` | `--include_out_of_range --censor_reg_weight 0` | Do dead and hyperstable singles help as rank anchors alone? |
| W2 | `v7_oor_full` | `--include_out_of_range` | Does the one-sided regression add to it (WT scale at the extremes)? |
| W3 | `v7_oor_full_floor` | W2 + `--censor_floor 0.5` | Does floor censoring of the MT flip columns stack on top? |

Decision rules: `val_rho_wt_valid_avg` and `val_rmse_combined_avg` must not regress against W0 (the WT head is the thing to
protect); the gain to look for is `val_auc_dead_wt` / `val_auc_hyper_wt` above chance and above W0's, with `val_rho_wt_all_avg`
flat or up. Note that dead and hyperstable singles are about 3.9% and 3.2% of training singles, so an average 16-item WT list holds about one
censored item; the sampler does not oversample them.

## 2f. Changes motivated by the ANOVA report — implemented, not yet run on GPU

All default off; with every flag off nothing changes (tests cover the loss composition, and `forward()` of the rank loss is bit-identical).
Background and numbers: `docs/anova_epistasis_report.md`.

### Per-pair flip metric: `val_rho_flip_pair`
`val_rho_flip` builds each matrix from the scored position alone, so its columns pool partners at *different positions*. A predictor that knows the
scored substitution and the partner **position** but not the partner **residue** earns 0.137 of the models' 0.20-0.21 on validation (measured). `val_rho_flip_pair`
uses one matrix per (scored position, partner position): rows are scored substitutions, columns partner residues, so double-centring removes everything that
does not depend on the specific *combination*, and that predictor scores exactly 0. Select checkpoints on it with `--monitor_metric val_rho_flip_pair_avg`
(and `val_flip_pair_matrices/<loader>` for how many matrices it used). It is also written by `esm_msr_testing.py` as `rho_flip_pair`.

### Monotone saturating link: `--link softclamp`
The calibrated LLR sum is treated as a **latent** ddG, and regression is done on the observed scale, `h(dG_wt + background + latent) - dG_wt`, with one soft
floor/ceiling `h` shared by all libraries (`esm_msr/link.py`; four parameters, own optimizer group, `--link_lr`). Saturation (60-66% of the variance of measured
dddG) lives in `h` instead of being imitated by the adapters. Ranking is untouched (h is monotone). A conditional item is scored as the double it came from.
Validation reports `rmse_combined` and `rho_epi_*` on the observed scale (so they stay comparable with runs without the link) and the latent versions as `*_latent`.

What it makes redundant, with `--link softclamp`:

| setting | status | why |
|---|---|---|
| `--min_additive_dG`, `--subfloor_rank_only` (`reg_ok`) | **ignored** by the regression | they existed to keep floor-pinned values out of a linear regression; `h` absorbs them |
| `--shared_bias_init` | **redundant, leave unset** | the link fixes the absolute levels; a bias would shift the latent zero point (a no-op mutation must be 0) |
| calibration-head **scale** | still needed | it is the latent scale; only its bias is redundant |
| `--censor_floor_hinge` | redundant | `h` already treats floor-pinned values as observations at the plateau |
| `--censor_floor` (rank tie of floor-pinned items) | still useful | the *order* among pinned items is still noise; the link does not touch ranking |
| `--censor_reg_weight` / hinge for `<-1`, `>5` | keep | a bound is not a value: the hinge is computed on the observed scale and is still the right treatment |
| `--lambda_*_combined` (legacy) | **not supported** (raises) | defined on the unsaturated additive target |
| `--cond_weight`, `--mt_single_anchor_*`, `--lambda_reg_mt` | unchanged | same terms, now on the observed scale |

Items without a known `dG_wt` (four libraries whose wild type is itself out of range) are left out of the regression and keep their rank terms.
The learned plateaus are logged as `link/floor` and `link/ceiling` (with `link/lo`, `link/hi`, `link/tau_lo`, `link/tau_hi`); without out-of-range items
the floor settles above -1 because only variants measured above it are kept. Inference is unchanged and returns the latent (unsaturated) ddG.

### Interaction-only loss: `--lambda_int_mt`, `--flip_pair_groups`, `--flip_align_units`
For each position-pair matrix in a micro-batch (trimmed to a complete block) the predictions and the measurements are **double-centred** separately and regressed on
each other. The pair offset, each substitution's own effect and each partner's own effect are annihilated on both sides, so the loss pressures only the interaction.
**It adds to the ordinary regression and does not replace it**, so the identity-independent effects (which can be real biology) keep being learned there:
saturation by the link, per-substitution severity by the WT head and the MT conditional regression, pair offsets and row/column effects by the MT conditional regression.
It needs several columns of one pair in a micro-batch: `--flip_pair_groups 3` packs up to 3 columns of a pair together in the sampler, and `--flip_align_units` cuts MT
micro-batches at pair boundaries (this also fixes the ~10% of within-column pairs lost to fixed-size cuts). Recommended: `--lambda_int_mt 1.0 --flip_pair_groups 3 --flip_align_units`.
Check what a model captured with `analysis_notebooks/anova/decompose_predictions.py` (Spearman of its predicted dddG with each *measured* component: saturation, pair offset,
row+column effects, interaction): the first three should stay put and the last should rise.

Suggested arms (3 epochs, same seed and `cache_v7`; monitor `val_rho_flip_pair_avg`, guard `val_rho_wt_valid_avg` and `val_rmse_combined_avg`):

| arm | name | flags beyond the canonical command (no `--shared_bias_init`) | question |
|---|---|---|---|
| L0 | `v7_link` | `--link softclamp` | Does the link keep or improve ΔΔG quality and reduce the work the adapters do for saturation? |
| L1 | `v7_link_oor` | L0 + `--include_out_of_range` | Do dead / hyperstable items help once the link carries saturation? |
| I0 | `v7_pairgroups` | L1 + `--flip_pair_groups 3 --flip_align_units` | The sampler/micro-batch change alone (no new loss). |
| I1 | `v7_int` | I0 + `--lambda_int_mt 1.0` | Does the interaction-only loss raise `val_rho_flip_pair_avg`? |

## 3. What to watch

| metric | meaning | expectation |
|---|---|---|
| `train/rank_mt` | the new flip loss | must be non-zero from step one. Zero means a stale cache, or `flip_list_min` set too high (§2b). |
| `train/flip_cols` | flip columns used per step | ~2.6 grouped at `micro_batch_size 64`. Falling toward 0 is the alarm. |
| `train/flip_uncensored` | of those, rows whose order is scored (only with `--censor_floor`) | Fraction of `flip_items` is the effective-signal share; if tiny, lower the floor. |
| `train/flip_items` | rows actually entering the flip loss | ~12 of 64 at the canonical settings (§2b). Rises if the anchor is lowered. |
| `train/flip_len` | mean members per column | ~4.65. If this approaches `flip_list_min` the loss is running on scraps. |
| `val_flip_pairs/<loader>` | usable position pairs per validation library | 0 for libraries without designed doubles — expected, not a failure |
| `val_rho_flip_avg` | **the monitored metric** | reference: the released checkpoint scores 0.141 on the test docket and 0.146 on verified held-out proteins. Validation libraries differ, so treat the *first run's* value as the baseline to beat, not these numbers. |
| `val_rho_epi_fast_avg` / `val_rho_epi_full_avg` | Spearman vs measured ΔΔΔG of `0.5*(mt-wt)` (needs only the double) and of `comb_AB-comb_A-comb_B` (needs both singles in the loader) | `fast` is polluted by MT/WT disagreement on singles, so it moves with `--mt_single_anchor_*`; judge anchor arms on `full`. Neither isolates identity-specific epistasis (FINDINGS §1); `val_rho_flip_avg` does. Formerly logged as `val_rho_epi`. |
| `val_rho_combined_avg` | ΔΔG headline | must not regress materially |
| `val_rmse_combined_avg` | ΔΔG calibration, kcal/mol | must not regress; this is what `lambda_reg_mt` protects |
| `norm_grad/lora_mt` vs `norm_grad/lora_wt` | gradient balance | if the MT group's norm jumps after adding the rank term, lower `--lambda_rank_mt` before touching anything else |

## 4. Runs to do, in order

Sequential; one GPU. 3 epochs each unless stated.

| # | name | overrides | question |
|---|---|---|---|
| **A** | `v5_flip_baseline` | *(none — the command above)* | Does the flip loss raise `val_rho_flip` over the previous objective? This is the reference. |
| **B** | `v5_noflip` | `--lambda_rank_mt 0.0` | The control for A. Same cache, same everything, flip loss off. **Run this even if A looks good** — without it you cannot attribute any gain to the new term. |
| **C** | `v5_flip_strong` | `--lambda_rank_mt 3.0` | Is the loss under-weighted at 1.0? Watch `val_rmse_combined_avg` for the trade. |
| **D** | `v5_nosubfloor` | `--no-subfloor_rank_only` | Is keeping sub-floor doubles for ordering better than dropping them? Changes train size, so compare per-epoch not per-step. |

If wall-clock is tight, A and B are the pair that must happen. Add
`--mt_single_anchor_frac 0.25` to all four for ~1.6× speedup at unchanged expected anchor
contribution.

Optional fifth, only if A clearly wins:

| **E** | `v5_flip_hpo` | sweep `--lora_rank_mt {8,16,32}` × `--cond_weight {0.25,0.5,1.0}` | These are MT-specific hyperparameters that have never been tuned against a metric that can see the MT adapter. Select on `val_rho_flip_avg`. |

## 5. Decision rules

* **A > B on `val_rho_flip_avg`, with `val_rmse_combined_avg` flat** → the flip loss works;
  make it default and go to E.
* **A ≈ B** → the loss is not biting. Most likely causes in order: stale cache (check
  `train/rank_mt`), too few flip columns per batch (check `val_flip_pairs`; the batch sampler
  draws per library, so a library with few designed pairs yields few columns), or the weight
  being too low (try C).
* **A > B on flip but `val_rmse_combined_avg` regresses** → the two objectives are competing.
  Lower `--cond_weight` on the regression rather than lowering `--lambda_rank_mt`; the
  regression's targets are the noisy derived conditionals and are the right thing to give up.
* **Anything NaN in `val_rho_flip`** → not a failure in itself. It is NaN wherever a loader
  has no usable complete pair matrix, which is most benchmark loaders and any library without
  designed doubles. `_avg` skips NaNs. It *is* a failure if every loader is NaN.

## 6. Evaluating a finished checkpoint

Score it with the conditional-ordering test on the real held-out docket, not on ProteinGym
subsets that overlap training. The analysis scripts are in the session scratchpad
(`ordering_test.py`, `testdocket_seeds.py`); the data is
`analysis_notebooks/predictions/hyperopt_splits-test/`.

Three requirements, all of which have bitten already:

1. **Trim each position-pair matrix to a fully complete sub-block before ranking.** Missing
   cells arise because the double failed to measure, so missingness depends on both
   substitution effects and is shared between the measured and predicted matrix. Centring an
   incomplete matrix admits ρ up to **+0.95** with no interaction present at all.
2. **Score epistasis from `epistasis_pred` (MT pass), not `combined_pred`.** Worth
   0.141 → 0.158 on the test docket; the WT pass is additive and reorders within columns.
3. **Report signature SD beside ρ.** A model with signature SD at the floor (~0.010) has an
   additive readout and the ρ is not meaningful. Three independent additive controls
   (ESM-MSR's WT pass, its masked-marginal rescoring, ThermoMPNN-D's additive head) return
   exactly zero — use any of them as a pipeline check.

Competitor numbers to beat, same rows, three seeds: Mutate Everything 0.083 ± 0.011,
ProteinMPNN (zero-shot) 0.079, Rosetta 0.044 ± 0.005, ThermoMPNN-D 0.044 ± 0.016,
SPURS 0.044 ± 0.023.

## 7. Do not change these without reading FINDINGS.md

* **`--mask_strategy` stays off.** Masked-marginal scoring drops the flip signature to
  *exactly zero* — the readout becomes additive and cannot express interaction at all. This is
  now measured, not inferred.
* **`--lambda_reg_mt` stays > 0.** It is the only term giving the MT pass an absolute scale.
  Without it `combined_pred = 0.5·WT + 0.5·MT` is uncalibrated and the ΔΔG output breaks. The
  assertion tying `mt_single_anchor_weight` to it remains.
* **Do not read epistasis from `combined_pred`.** §4 of FINDINGS.md.
* **Structure-encoder adaptation stays off** by request, despite the evidence that the
  capability lives in the structure channel. Revisit deliberately, not accidentally.

## 8. Reporting back

Per run: final and best `val_rho_flip_avg`, `val_rho_combined_avg`, `val_rmse_combined_avg`,
`val_rho_mt_valid_avg`; the epoch each peaked; whether `train/rank_mt` was non-zero
throughout; median `val_flip_pairs` across loaders; wall-clock per epoch; peak VRAM; and any
assertion or NaN event with 20 surrounding log lines.

For A vs B, report the difference in `val_rho_flip_avg` **and** state whether
`val_rmse_combined_avg` moved — a flip gain bought with ΔΔG calibration is a trade, not a win,
and the decision about whether to take it is not yours or mine to make silently.
