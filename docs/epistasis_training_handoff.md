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
> be loaded silently by a run expecting the new item schema. Point `--cache_path cache_v5` and
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
  --cache_path cache_v5 \
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

## 2b. KNOWN ISSUE: the loss only sees a fraction of each micro-batch

**Read this before tuning anything.** Measured on a 4-protein, 29-step run (`--skip_val`,
`micro_batch_size 64`, `subset_size 16`, anchor 0.5):

| config | steps with flip | columns/step | mean column length | **items/step** |
|---|---|---|---|---|
| `--no-flip_group_units` | 25 / 29 | 1.00 | 4.08 | **4.1** |
| `--flip_group_units` (default) | **29 / 29** | 2.59 | 4.65 | **11.8** |

Grouping is a clear win and is on by default. But note the last column: even grouped, only
**~12 of 64 micro-batch rows contribute to the flip loss**, and the mean column holds 4.65
members against the ~19 that exist in the data. Two causes:

1. **Anchored singles occupy MT rows and have no column.** At `--mt_single_anchor_weight 0.5`
   they are roughly half of all MT backbone rows. They sort to the end of the unit so they do
   not split a column, but they still consume slots.
2. **A 256-item batch drawn from one library need not contain a whole column.** The sampler
   groups by library and by `subset_size` lists for the WT ListMLE term; it knows nothing about
   flip columns. `subset_size` does **not** control flip columns — the flip loss groups by
   `flip_key` and ignores `subset_size` entirely, so changing it will not help here.

**Consequences for tuning:**

* **Do not raise `--flip_list_min` above 4** without fixing the sampler first. With a mean
  column length of 4.65, a threshold of 6 or 8 would discard most columns and the loss would
  go quiet. If you see `train/flip_cols` near zero, this is the first thing to check.
* **The proper fix is a column-aware sampler** — draw whole flip columns into a batch rather
  than relying on them co-occurring. That is the highest-value follow-up on this code and was
  not attempted here. It should raise items/step from ~12 toward ~50 and make the loss roughly
  4× more efficient per forward pass.
* **A cheap partial mitigation** is raising `--micro_batch_size` (more rows per unit, so more
  whole columns land together) at the cost of VRAM. Untested.

Diagnostics logged every step to help: `train/flip_cols` (columns used),
`train/flip_len` (mean members per column), `train/flip_items` (rows actually entering the
loss). Watch all three.

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

## 3. What to watch

| metric | meaning | expectation |
|---|---|---|
| `train/rank_mt` | the new flip loss | must be non-zero from step one. Zero means a stale cache, or `flip_list_min` set too high (§2b). |
| `train/flip_cols` | flip columns used per step | ~2.6 grouped at `micro_batch_size 64`. Falling toward 0 is the alarm. |
| `train/flip_items` | rows actually entering the flip loss | ~12 of 64 at the canonical settings (§2b). Rises if the anchor is lowered. |
| `train/flip_len` | mean members per column | ~4.65. If this approaches `flip_list_min` the loss is running on scraps. |
| `val_flip_pairs/<loader>` | usable position pairs per validation library | 0 for libraries without designed doubles — expected, not a failure |
| `val_rho_flip_avg` | **the monitored metric** | reference: the released checkpoint scores 0.141 on the test docket and 0.146 on verified held-out proteins. Validation libraries differ, so treat the *first run's* value as the baseline to beat, not these numbers. |
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
