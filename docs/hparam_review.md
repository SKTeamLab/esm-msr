# Hyper-parameter review, dead-code removal and run plan

Written 2026-10-05 on branch `claude/cleanup-dead-params` (from `claude/censored-margin-ranking-devel` @ 3f18100).
**Measured** means read from a log, a cache or a test; **inferred** means reasoning that no run has checked.

## 1. What changed (and what did not)

* **Training numerics are unchanged.** The real loss composition, run on a synthetic batch under 15 configurations (rank only,
  link, interaction loss, anchors, hinge, censored and uncensored batches, micro-batch 12 / 16 / 32 / 64, WT only), returns bit-identical
  losses and gradients before and after the cleanup in 14 of them (max |difference| = 0; harness: `diff_losses.py` in the session scratchpad).
  The one that differs is the old *unaligned* chunking at a micro-batch that cuts columns, which is the intended change (§3, `flip_align_units`).
* **Retired flags still parse.** `esm_msr.config.RETIRED_FLAGS` lists them. A retired flag given its old harmless value is ignored with a
  warning, so older commands and the ones in `docs/` keep running; given a value that used to change behaviour (`--lora_mode corrector`,
  `--reg_loss huber`, `--lambda_reg_combined 1`, `--no-dedup_backbone`, `--detach_regression`, `--subset_caps double=0.5`, ...) it is an error.
  Retired flags never reach `hparams.yaml`.
* **Old checkpoints still load** (inference reads `hparams.yaml` with defaults; `lora_mode` is simply no longer read).
* **Validation, the WT head and every metric name are unchanged.** One new option changes protocol only if you set it (`--lr_plateau_metric`),
  and one is new and off by default (`--wt_early_stop_patience`).

### Removed
| removed | why |
|---|---|
| `lambda_{rank,reg,epi}_combined`, `mt_reg_mask`, `detach_ensemble_input`, `zero_epistasis_for_singles`, `lora_mode`, the `combined` work-unit path, `crit_rank_combined` | the legacy teacher-forced ensemble objective; `--link` already refused it; ~120 lines of `_compose_losses_streaming_and_backward` |
| `double_weight`, `reversion_weight`, `incl_doubles/reversions` (as flags), the WT "ensemble" block | doubles reach training as their two `cond` items; reversions never had a head. Doubles stay in the *validation* sets |
| `huber_delta`, `reg_loss`, `AsymmetricHuberLoss`, `rank_loss`, `invert_list_loss`, `ListMLELoss_enhanced` | MSE and ListMLE are the only ones ever run |
| `detach_calibration`, `detach_regression` | see §3 |
| `dedup_backbone` | always on (class attribute) |
| `flip_group_units`, `flip_align_units`, `flip_pair_groups`, `int_min_rows`, `int_min_cols` | derived or constant, see §3 |
| `freeze_wt_after_epoch`, `freeze_wt_on_convergence`, `wt_convergence_patience`, `wt_convergence_metric` | replaced by `--wt_early_stop_patience` |
| `use_plddt` (training, `MSRModel`), `residual_wd`, `residual_lr_mult`, `incl_{singles,cond,native_cond}` (as flags) | never read / derived from `--subset_caps` |
| the `grad_clip_norm` branch of `training_step` | the flag never existed, so the branch was unreachable: **no gradient clipping is applied, and none ever was** |
| `scripts/queue_runner.sh` | superseded |

### Renamed / derived / new
* `--subset_size` -> `--wt_list_size` (the old name still works). It is the WT head's ListMLE list length, not a count of subsets (§3).
* `flip_pair_groups` is derived: with `--lambda_int_mt > 0`, `micro_batch_size // 19` columns of a pair travel together (3 at the default 64);
  `--lambda_int_mt` with `micro_batch_size < 38` is an error because a pair cannot then fit two columns. MT micro-batches are cut at pair boundaries
  whenever `lambda_rank_mt` or `lambda_int_mt` is on.
* `--wt_early_stop_patience N` (0 = off): WT-head early stopping on `val_rho_wt_valid_avg`, which **restores the best WT state** before freezing.
* `--lr_plateau_metric {rho_combined,rho_wt_valid,rho_flip_pair}`: what drives ReduceLROnPlateau (default unchanged).
* `scripts/run_with_retry.sh NAME EPOCHS SEED [flags]`: the crash-retry runner, now generic. `run_i1.sh` is a wrapper over it.

## 2. Bugs and implementation issues found while reading

1. **LR plateau scheduler keyed to `rho_combined`** (measured). `ReduceLROnPlateau(patience=1, factor=0.1)` steps on `val_rho_combined_avg`,
   which is dominated by the WT head and flat after epoch 2 (0.816), and it cuts the rate of **every** group, the MT head and the link included.
   It fired in `w2_cens` at step 4599 (after the epoch-4 validation), so that run's last epoch trained at 2e-5 while `val_rho_flip_pair_avg` was
   still rising. No link run has been cut yet (their `rho_combined` kept creeping up), but any longer run will be. It also means "stop the WT head
   when it converges" has so far happened only as this side effect. Fix: selectable (`--lr_plateau_metric`); default left alone so the I-series stays comparable.
2. **The MT unit existed only when `lambda_reg_mt > 0`** (measured by test). With the MT regression off, the MT rank and interaction losses silently
   trained nothing. Now any of the three MT losses creates MT units (new test). No past run is affected (all had `lambda_reg_mt 1`).
3. **WT "convergence" freezing froze the weights *after* the patience window, not the best ones**, and was off in every arm (`freeze_wt_on_convergence`
   defaults False and no script sets it). Now it restores the best snapshot (new test).
4. **`--detach_calibration` was never read.** What reached the model was `--detach_regression`, which cuts the adapters off from *every* term that goes
   through the calibrated prediction, including `--lambda_int_mt` and the link regression. It could only ever have been "regression trains the
   scale/bias, ranking trains the adapter". Removed.
5. **`micro_batch_size` was silently rounded down to a multiple of `subset_size`** (so 40 became 32). Removed; no run used a non-multiple.
6. **The learned link bounds drift in the direction the censored hinge prefers** (measured, §3 `link_learn_bounds`).
7. **Out-of-range items change the validation sets too** (inferred from `preprocess_megascale.setup_dataloaders`: val loaders get
   `include_out_of_range`). All ordinary metrics exclude censored items, so they stay comparable; the two `auc_dead_*` metrics exist only with the flag.
8. **`min_additive_dG` / `subfloor_rank_only` do nothing under `--link`**, in training or validation (measured: `reg_ok` is read only on the
   non-link branch, and validation loaders only mark, never drop). They are kept for `--link none`.
9. Not touched: `inference_scripts/msr_inference.py` forwards `--lora_mode` / `--adapter_mode` to `inference.py`, which defines neither (pre-existing).

## 3. Your list, item by item

**`censor_floor`.** Off in every arm since the I-series (`None`). It marks a numerically measured flip-column item whose dG is at or below F as
lower-censored *for the rank losses only*: it joins the tied-bottom group of its column. Under the link the regression side needs nothing
(`h` saturates), but the *order* among floor-pinned items is still noise, which is the case for it. How much it would touch (measured, 120 training
libraries, ordinary items): conditional items with dG <= 0.0: 13,298 (11.9%), <= 0.5: 31,304 (28%); native-conditional: 1,023 (5.9%) / 2,274 (13%).
The planned W3 arm (`--censor_floor 0.5`) was never run. **Worth one arm at 0.0** (0.5 would censor over a quarter of the MT items).

**`censor_floor_hinge`.** Applies the one-sided regression to those floor-censored items too, instead of regressing on their measured value.
`docs/epistasis_training_handoff.md` §2f already marks it redundant under the link (`h` treats pinned values as observations at the plateau), and I
agree: a hinge discards the numeric value (-0.9 vs -0.2) that the link can use. The sub-floor targets that *are* in play now are the `<-1` items
(`include_out_of_range`), which always get the hinge. Kept, not planned.

**`censor_reg_weight`.** Yes, the weight of the hinge term for censored items relative to ordinary ones (it does not touch the ordinary regression).
`0` = censored items stay in the rank losses only, which makes it the clean ablation of "do the dead/hyperstable items help as rank anchors alone?".
Kept at 1; low priority.

**`cond_weight` (0.5), `native_cond_weight` (1.0), and where `native_cond` fits.** Both are *regression* weights only (the rank and interaction
losses are unweighted). `cond` = the two conditional items derived from each measured double, ddG(A|B) = ddG_AB - ddG_B. `native_cond` = measurements in
15 mutant-background libraries (codes like `1A0N_L7S`): a directly measured conditional effect, the cleanest MT label there is, but only 17,435 items
(+254 dead, +89 hyperstable) against 111,364 `cond` items. So native items are 13.5% of the MT labels and, at weight 1 vs 0.5, 24% of the MT regression weight.
They enter the rank loss as flip columns (`code|pos|native`, about 1/8 of the columns) but **not** the interaction loss or the pair metrics (no partner
position). The 2:1 weighting is inverse variance (a derived item is a difference of two measurements); under the link it has a second justification: both
`cond` items of a double are re-scorings of the *same* measured ddG_AB, so at 0.5 each the double counts once. Acceptable prior; not planned.

**`dedup_backbone`.** Always on now. It existed to prove the dedup was exact and to A/B memory. One property to know: rows that share an input share one
LoRA-dropout draw, so all WT-pass singles of a protein in a batch see the same dropout mask.

**`detach_calibration`.** Dead (bug 4). Originated in the first public release as "ranking trains the adapter, regression trains only scale/bias".
Incompatible with the link and the interaction loss. Removed with `detach_regression`.

**`double_weight`, `reversion_weight`, `incl_doubles`, `incl_reversions`.** Dead for training, removed. Doubles remain in validation (`rho_epi_full`).

**`flip_align_units`, `flip_group_units`, `flip_pair_groups`.** You were right on all three. Alignment is now automatic whenever a flip loss is on (it also
stops fixed-size cuts from splitting columns, which cost about 10% of within-column pairs); `flip_group_units` was superseded by it and is gone;
`flip_pair_groups` is `micro_batch_size // 19` (19 = the longest possible column), which is 3 at 64, so the I-series behaviour is reproduced exactly.
The only behavioural change is for rank-only arms (W/L series): their chunking is now aligned, which is what the I-series used.

**`freeze_wt_after_epoch`.** Removed, replaced by `--wt_early_stop_patience` (see `wt_convergence_patience`).

**`huber_delta`.** No reason left that anything has checked: saturation is the link's job and censoring the hinge's, which were the outliers Huber was
protecting against. Removed. (Heavy-tailed noise in the derived conditionals is plausible but untested.)

**`incl_doubles`, `incl_reversion`.** See above.

**`include_out_of_range`.** Open, and not testable from existing runs: W0 -> W2 changed it but also changed micro-batch (64 -> 32) and early-stop metric,
and W2's trajectory (0.184, 0.163, 0.206, 0.180, 0.212, 0.215) moves as much between epochs as W0 and W2 differ. Late-stage arm (one run, flag off).

**`int_min_cols`, `int_min_rows`.** Not what you guessed: they apply only to the interaction loss. After a pair matrix is trimmed to a complete block
it must have at least 4 rows (scored substitutions) and 2 columns (partner residues), because double-centring needs two of each and four rows keep a
row mean from being one cell. The rank loss has its own threshold, `flip_list_min` (4 members per column). Now constants.

**`lambda_rank_mt`.** Agreed it is the important one, and the scale mismatch is large (measured, i2): rank loss ~25 per column against `L_int_mt`
~0.03 per cell (x30 = ~1). The *value* is not the *gradient*, so I do not know the gradient share; a one-off probe (per-term gradient norms of the MT
LoRA at a checkpoint) would settle it and is in the plan. A rank-off arm is also in the plan.

**`link_learn_bounds`.** I could not find a reason once out-of-range items are in. Its docstring reason ("without dead items the floor sits above -1
because only variants measured above it are kept") no longer applies, and the logs show the bounds moving, in i2 in the direction the hinge wants:
`lo` ended at -1.16 (it was -0.74 at epoch 0), `hi` 5.09 -> 5.39 over four epochs (L1: `lo` -0.50..-0.88, `hi` 5.13..5.29). A ceiling of 5.39 lets the hinge for `>5` items be
satisfied at a finite latent, which a fixed 5.0 never allows. The assay's own bounds are -1 and 5. Plan: one arm with `--no-link_learn_bounds` (the knee
softness `tau` stays learned); if it matches, remove the flag.

**`lora_alpha_mt`, `lora_rank_mt`.** Correction to something I implied earlier: the adapters use **rsLoRA** (`use_rslora=True`), so the output scale is
alpha / sqrt(rank), not alpha / rank (MT: 16/4 = 4.0; WT: 4/sqrt(2) = 2.83). rsLoRA exists to make the update size rank-independent at fixed alpha, so a
**rank sweep should hold alpha at 16** and is then a clean capacity test; alpha is not "adjusted alongside rank". Under Adam the step size is about the
learning rate whatever the gradient scale, and the output scale multiplies the weight change, so `lora_alpha_mt` behaves like a learning-rate multiplier for
the MT adapter alone (inferred; the 8 / 32 arms would test it). Both are plan items.

**`lora_mode`.** Dead, removed (it was only read by the combined objective).

**`mask_mutated_structure`.** Careful: this is the *cache-build* option (masks the partner site in the cached `cond` structure; default off, so cache_v7 is
unmasked). The run-time option is `--mask_structure`, which blanks coordinates and structure tokens at every position the MT sequence differs from the
structure, i.e. the scored site and the partner site, on the MT pass only, with no cache rebuild. Only a two-epoch v4 test exists (`mt_valid` 0.569 / 0.586
masked vs 0.542 / 0.583 unmasked: indistinguishable) and it predates the flip metrics. Known limitation: neighbouring structure tokens still carry the masked
residue's geometry. Plan item.

**`min_additive_dG`.** No longer decides which doubles are included: with `--subfloor_rank_only` (what we run) nothing is dropped, items are only marked
`reg_ok=False`, and under `--link` even that flag is ignored (bug 8). Inert for every link run, in training and validation.

**`mt_reg_mask`.** Not which types train: it only changed which items the legacy combined objective used. Removed.

**`mt_single_anchor_weight`, `mt_single_anchor_frac`.** Kept. Anchored singles are 48% of the MT backbone rows, so `frac` is mostly a **cost lever**: 0.25
saves about a third of training time (weights are scaled 1/frac). It is a plan item because, if it is neutral, every later arm gets cheaper.

**`native_cond`, `single` (null).** `--subset_caps single=None cond=None native_cond=None`: None means every item of that subset, 0 means none, a number
is a fraction of the unrestricted size. Comet shows None as null. So yes, included.

**`reversion_weight`.** Dead, removed.

**`subfloor_rank_only`.** The alternatives for double-derived items whose *additive prediction* is below -1 are (i) drop them (`--no-subfloor_rank_only`,
the old default), (ii) regress on the measured value anyway (what the flag exists to avoid), or (iii) keep them rank-only (default). Under the link none of
this applies: they regress on the observed scale through `h`. Inert for link runs.

**`subset_size 16`.** Not an accumulation count. Each work unit is back-propagated on its own (streaming), so nothing accumulates across "subsets". It is
the length of the WT head's ListMLE lists: the WT block of a batch (all singles of one protein; batches are single-protein) is cut into consecutive lists
of 16, roughly 7 lists per batch, with the remainder (<16 rows) left out of the rank loss. Setting it to the micro-batch would turn 16-item lists into 64-item
lists, a change to the WT head's objective, which the standing rule says must be flagged; so it is renamed `--wt_list_size`, decoupled from the micro-batch,
and left at 16. A 16 -> 64 arm would be a legitimate WT-head experiment but is not in the plan.

**`use_plddt`.** Dead (it only logged a warning). Removed from training and `MSRModel`; `inference.py` still accepts the flag.

**`wt_convergence_patience`.** You were right: it is not what happens in any arm. `freeze_wt_on_convergence` was off in all of them, so the WT head trained to the
end of every run. What did happen is bug 1 (a plateau cut of everything). With the flag on, the old code froze at the first non-improving validation
(patience 1) in the state one epoch past the best. Replaced by `--wt_early_stop_patience N`, which restores the best WT adapter and calibration head, freezes
them and keeps training the MT head. Off by default, so no existing arm changes; it *is* a WT-head change when used.

**What the interaction loss is doing (measured, i2).** In-sample, the share of double-centred target variance left unexplained fell from 0.95 (steps 0-500)
to 0.70 (3700-5000) while validation `rho_flip_pair` is 0.19 and `val_rho_flip_pair_pooled` 0.20. The target's own variance (~0.04-0.05) is at the measurement-noise level
(SD 0.27 gives 0.073, and double-centring a typical 8 x 3 block keeps 58% of it: 0.043), so most of the in-sample gain is plausibly noise being fitted. That is the capacity / ill-posedness argument,
and why the rank sweep is first in the plan. (Inferred; the validation counterpart `L_int` is not logged.)

## 4. Where the runs stand (single seed; epoch-to-epoch noise in `val_rho_flip_pair_avg` is about +-0.02 to 0.03)

| arm | micro-batch | out-of-range | link | `lambda_int_mt` | `val_rho_flip_pair_avg` by epoch (0 = after the first epoch) |
|---|---|---|---|---|---|
| w0_base | 64 | no | no | 0 | 0.142, 0.168, 0.183 (3 epochs) |
| w2_cens | 32 | yes | no | 0 | 0.184, 0.163, 0.206, 0.180, 0.212, 0.215 |
| l1_link_cens | 32 | yes | yes | 0 | 0.159, 0.158, 0.179, 0.191, 0.199, 0.209 |
| i1 (stopped) | 64 | yes | yes | 1 | 0.176 (1 epoch) |
| i2_lam30 | 64 | yes | yes | 30 | 0.134, 0.140, 0.178, 0.186, ... (running) |

Nothing here separates the arms. What is measured: the link lowers observed-scale RMSE (about 0.65-0.67 vs 0.70-0.73), the WT head is unchanged in all arms, and
`val_rho_colrank_avg` rose from 0.13 to 0.59 under the interaction loss (the identity-independent effects are being learned; `val_rho_flip_pair_pooled` also rose,
0.12 -> 0.20). At lambda 30 the first two epochs show **no** `flip_pair` progress at all, then it catches up to the rank-only arms (inferred: early on the
interaction gradient is mostly noise and Adam normalises it up; not checked).

## 5. Run plan

**Cost.** i2 runs at 0.57 it/s, 915 steps per epoch: about 27 min per epoch plus ~4 min validation and ~6 min initial validation, so **~2.5-3.5 h per arm**
with the early stop (patience 2, min 3 epochs, cap 6). Ten arms are 1-1.5 days of GPU.

**Reading results.** Use the mean of the last three validations of `val_rho_flip_pair_avg` plus `val_rho_flip_pair_pooled`, `val_rho_colrank_avg`,
`val_rmse_combined_avg` and `val_rho_wt_valid_avg` (the WT guard). With one seed, only differences of about 0.04 or more are believable until A2 measures the
seed spread. State that in every readout.

**Base** (all arms unless stated): canonical `run_arm.sh` + `--link softclamp --include_out_of_range`, no interaction loss, micro-batch 64, seed 1.
(The interaction arms are i2 / i3, already queued; their code path is the same as the base plus `--lambda_int_mt`.)

**Gate (before any arm):** merge this branch when the queue is idle, then a 50-step GPU smoke of the new code (WT unit, MT unit, link, validation, and
`--wt_early_stop_patience 1` with `--learning_rate 0` to force the restore/freeze path). A CPU end-to-end smoke of the real module on this code has already passed (measured, 2026-10-05, 2 training steps on 2 libraries, fp32): constructor
with the new `MSRModel` signature, WT and MT units on the real backbone with link, hinge and anchors, a real validation pass, and the WT early-stop path
(restore, save 293 WT tensors, freeze WT, thaw MT) through the real `PEFTStateManager`. It stopped afterwards only because `ModelCheckpoint` monitors
`val_rho_flip_pair_avg`, which a single validation library without doubles cannot produce. It does not cover bf16 autocast, the memory behaviour on the GPU, or the
interaction loss with the real backbone (the latter is covered on a stub).

### Phase A (fixed; `scripts/queue_plan1.sh`)
| # | name | flags beyond the base | question | decision rule |
|---|---|---|---|---|
| A1 | `p1_base_s1` | none | the reference for everything below, on the cleaned code (aligned chunking, micro-batch 64) | compare with L1 (0.209 at epoch 5) |
| A2 | `p1_base_s2` | `--seed 2` | seed spread | defines "believable" for the rest |
| A3 | `p2_mtr4` | `--lora_rank_mt 4` (alpha stays 16) | does less MT capacity help (your ill-posedness argument)? | flip_pair >= base and pooled/colrank not worse -> capacity is the lever, go lower |
| A4 | `p2_mtr2` | `--lora_rank_mt 2` | the WT head's rank, which works | same |

### Phase B (after reading A; `scripts/queue_plan2.sh`, edit its BASE line to the best of A)
| # | name | flags | question |
|---|---|---|---|
| B1 | `p3_mask` | `--mask_structure` | does telling the MT pass "geometry unknown here" help? (never tested with the flip metric) |
| B2 | `p3_fixbounds` | `--no-link_learn_bounds` | are the learned link bounds anything but slack for the hinge? |
| B3 | `p4_rank0_int30` | `--lambda_rank_mt 0 --lambda_int_mt 30` | can the interaction loss replace the rank loss, or are they complementary? (i2 is the same without `--lambda_rank_mt 0`) |
| B4 | `p4_floor0` | `--censor_floor 0.0` | does tying floor-pinned order help? (touches 12% of `cond` items) |
| B5 | `p5_nooor` | drop `--include_out_of_range` | do dead / hyperstable items help at all? |
| B6 | `p2_a8` / `p2_a32` | `--lora_alpha_mt 8` / `32` at the best rank | is the MT adapter's effective step size right? (only if A shows an effect) |
| B7 | `p6_frac025` | `--mt_single_anchor_frac 0.25` | cost lever: if neutral, later arms cost about a third less |
| B8 | `p7_confirm` | best combination, seeds 1 and 2 | confirm before any default changes |

### Diagnostics that need no training run (GPU minutes, between arms)
* **Gradient-share probe:** per-term gradient norms of the MT LoRA (rank / regression / interaction at lambda 1) on ~20 batches at the i2 checkpoint. Answers "is the
  interaction loss actually minuscule" directly and says what lambda would put it on par.
* **`analysis_notebooks/anova/decompose_predictions.py`** on the W2 / L1 / i2 checkpoints: which measured component each model's predicted dddG tracks.
* **Native-background pair benchmark** (the ~2,516 matrices from the native-conditional libraries) before using native pairs for training.

### Not planned, and why
`wt_list_size`, `wt_early_stop_patience` (WT-head changes: separate decision), `censor_floor_hinge` (redundant under the link), `cond_weight` / `native_cond_weight`
(low value, plausible priors), LoRA dropout and weight decay on the MT adapter (not on your list; the obvious next regularisers *if* A shows capacity matters).

## 6. Decisions for you
1. Merge `claude/cleanup-dead-params` into `claude/censored-margin-ranking-devel`? I will not touch the working tree while the queue runs (bash re-reads
   a running script, and `i3_lam300` loads the code when it starts); I intend to fast-forward after `i3_lam300` finishes unless you say otherwise.
2. Flip `--link` to the default? If yes, the next deletions are the non-link branch: `reg_ok`, `subfloor_rank_only`, `min_additive_dG`, `censor_floor_hinge`, the
   `hinge_src` logic, `shared_bias_init`.
3. Change `--lr_plateau_metric` default to `rho_flip_pair` (or retire the plateau scheduler)? I left it alone for comparability.
