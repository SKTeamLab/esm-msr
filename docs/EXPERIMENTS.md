# Experiment log and findings (through 2026-10-18)

Chronological record of what was tried, what was measured, and what was concluded. Measurements are copied
from each run's `metrics.csv` (mean of the last 3 epochs unless stated) and from the launchers' own headers,
which are the primary source for *why* each arm existed. **Older runs log older metric names** — rescore
their dumps with `scripts/epi_from_dump.py` before comparing (see `STATE.md` §5).

The live state and operating instructions are in `STATE.md`; this file is the record of how we got here.

---

## 1. Phase A (2026-10-04/05): does the flip metric respond at all?

| arm | flags | epochs | best `val_rho_flip_pair_avg` | `rho_wt_valid` | `rmse_combined` |
|---|---|---|---|---|---|
| `w0_base` | none (micro 64) | 3 (fixed) | 0.182 (ep 2) | 0.811 | 0.694 (latent scale) |
| `w2_cens` | `--include_out_of_range` | 6 (cap) | **0.215** (ep 5) | 0.808 | 0.702 |
| `l1_link_cens` | `--link softclamp --include_out_of_range` | 5 of 6 | 0.199 (ep 4) | 0.806 | **0.650** (observed scale) |
| `i1` | link + censoring + pair groups + `--lambda_int_mt 1.0` | **never ran** | — | — | — |

**Conclusions.** (i) The flip metric did **not** show a demonstrated gain: W2 was +0.03 and L1 +0.02 over W0,
both inside the ±0.02-0.03 epoch noise of a single seed, and the arms also differed by micro-batch (32 vs 64
— micro 32 was forced by VRAM, a confound). (ii) **The link's one clear effect was calibration**, not
epistasis ranking: observed-scale `rmse_combined` 0.65-0.67 vs 0.70-0.73 without it. (iii) A fair test needs
equal lengths and ≥2 seeds — a lesson that has had to be re-learned twice since.

**Infrastructure lessons from this phase** (all still relevant): three crashes in two hours, two of them a
bug in `on_validation_epoch_end` (misaligned pooled observed-scale arrays), the rest VRAM exhaustion that
**WSL reports as `CUDA driver error: device not ready`**, not as an OOM; `dmesg` showed
`dxgkio_make_resident: Ioctl failed: -12`. Fix was a smaller micro-batch. Also: `SIGTERM` exits rc=0, so
killing a run looks like a clean finish to a queue.

## 2. Phase B (2026-10-06/08): replace the metrics, understand the variance

The flip metric was replaced by the **20-metric epistasis suite** (`src/esm_msr/epi_metrics.py`,
`docs/validation_metrics.md`), which separates levels and removes saturation. That made the earlier
questions answerable, and produced the analysis that still drives design:

* **The variance of measured dddG**: 56% is a function of the additive expectation alone, 22% pair offsets
  (70% of it between libraries), 14% line effects (mostly shared measurement noise of the singles), 6% the
  specific combination. Training-set ANOVA agrees (60/16/13/10%, cross-validated).
* **Saturation leaks into every statistic on raw dddG.** In simulation, a predictor that knows *only* the
  saturation scores 0.58 on the naive dddG correlation — more than one that knows every epistasis component
  but not the saturation (0.42). Two devices remove it exactly for an additive model: within-column order of
  ddG_AB, and residuals beyond each side's own global curve.
* **The link's floor is ~1 kcal/mol too low.** Doubles far below the floor measure near dG +0.03 (median of
  61 libraries), the learned floor was −0.99, and every model under-predicted the nonspecific epistasis by
  0.5-0.7 kcal/mol; a nonlinear recalibration independently put the floor at +0.29.
* **MT capacity and MT weight both mattered**: MT rank 16 reached `cell_rank_mt` 0.192/0.188/0.190 at epochs
  3-5 (≈+0.02 over rank-2 references), reversal accuracy 0.83-0.84 with guards unchanged; `o6g1_bal111`
  (MT components 1/1/1 at master 1.0) beat every other devel arm on every level. The MT adapter's own
  contribution over the WT adapter read in context was only ≈+0.02 (rank-2 arms) to +0.06 (baseline).
* **4 epochs is too short** for arms that change MT capacity; queues moved to 6, then 8-10.

## 3. Phase C (2026-10-09/13): queue r2, the `o6` arms

18 arms × 6 epochs on re-measured gradient-balance constants (`reg_wt 28.6, reg_mt 76.5, comp_off 55.3,
comp_subst 111.7, comp_int 742.3, flip_mt 9.1`). Screening results (mean of last 3 epochs; arms differ in
length/resume state, so read this as screening, not measurement):

| arm | what it added | cell_rank_mt | cell_mag_comb |
|---|---|---|---|
| `o6s21_mt8_s2` | MT rank 8, seed 2 | **0.2088** | 0.285 |
| `o6s09_mt8` | MT rank 8 | **0.2043** | 0.290 |
| `o6s02_mt16` / `o6s14_mt16_s2` | MT rank 16 | 0.1939 / 0.1927 | 0.282 / 0.287 |
| `o6q01_mt16` | rank 16, old constants | 0.1886 | 0.280 |
| `o6s23_flip1_pack` | packed matrices + reversal loss λ1 | 0.1902 | 0.264 |
| `o6s04_pack` | packed matrices, no reversal loss | 0.1561 | 0.239 |
| `o6s22_flip03_pack` / `o6s24_flip3_pack` | reversal λ0.3 / λ3 | 0.1703 / 0.1383 | 0.251 / 0.176 |
| `o6s25_flip1_nopack` | reversal loss without packing | 0.1180 | 0.169 |
| `o6s29/30/32` | component weights down (int / subst / all) | 0.1838 / 0.1788 / 0.1814 | 0.252 / 0.251 / 0.236 |
| `o6s33_no_colrank` | colrank loss off | 0.1731 | 0.239 |
| `o6s01_ref` | the r2 reference | 0.1692 | 0.270 |
| `o6s16_mt32` | MT rank 32 | 0.0064 | 0.000 — **degenerate run, not a result** |

`o6s20_mt64` (0.1718) and `o6s26/27/28` (floor censoring at +0.3 / 0.0 / −0.5) did not help.

**Conclusions.** MT **rank 8** was adopted as the basis (git: "winning rank 8 basis"), and rank 32 never
produced a usable run. **Packing alone hurt**; the reversal (confident-reversal) loss only looked good when
packed, and the two packing arms bracket it widely — this is the one place where a config change has an
apparently large effect that nobody has confirmed with seeds.

**Bugs this phase found** (each had invalidated runs before it was caught):
`--pack_pair_matrices` left autocast-cached weight casts without autograd history under bf16, so **packed
training gave the MT adapter no gradient at all** (fixed `9d629d1`, regression test under CPU autocast);
packed batches OOMed until the ceiling settled at micro 64 (via 38/48) after the autograd-graph leak was
fixed (`7d6d775`, `9e268a6`); the link head leaked gradient through the packed matrix context; the sampler
needed its epoch advanced on resume.

## 4. Phase D (2026-10-13/15): the link series on the released basis

Same basis (`bl_w2a4_m16a16_capped`), seed 1, 8 epochs, `--link softclamp`, micro 64 — one change each:

| arm | cell_rank_mt | cell_mag_comb | flipacc_mt | beyond_rho_mt | beyond_rho_comb | rho_doubles | rho_comb |
|---|---|---|---|---|---|---|---|
| `+link` (control) | 0.2017 | 0.2844 | 0.8652 | +0.5048 | +0.4833 | 0.7134 | 0.8383 |
| **`+nonspecific_term fx`** | **0.2126** | **0.3038** | **0.8707** | **+0.5639** | **+0.5499** | 0.7172 | 0.8401 |
| `+link floor` (per-library, from 134 libraries) | 0.1972 | 0.2721 | 0.8389 | +0.5147 | +0.4959 | 0.7123 | 0.8367 |
| `--tf_singles latent_link` | 0.1841 | 0.2694 | 0.8559 | +0.4752 | +0.4752 | 0.6939 | 0.8380 |
| `--tf_singles predicted` | 0.1902 | 0.2789 | 0.8387 | +0.5276 | +0.4903 | 0.7003 | 0.8381 |

**The `--nonspecific_term fx` arm is the best cell-level result on record and the reason it is worth
revisiting — see `docs/nonspecific_epistasis.md`.** The per-library floor arm (data floors, 96 own
estimates) was not better than the control, and its non-circular train-only control (`_tr`, floors estimated
from 45 training libraries only) was stopped at the user's request after 10 minutes. The `--lambda_reg_wt 5`
arm never ran.

## 5. Phase E (2026-10-16/18): the `dc` schedule series

A conservative rebuild on `devel-revision`, pinned per series, testing the **training schedule** rather than
the architecture. Early arms: `dc1/dc2` floor-link exploration, `dc3` gap censoring + learned global floor
(0.1988/+0.5472), `dc4/dc5` `native_cond` as a single (dc5 retired: confounded, then a CUDA fault),
`dc6` range censoring + 4-parameter link, `dc7` 4-parameter link (0.1968/+0.5293), then **`dc8`** the
floor-only link canonical triplicate.

**The plateau bug (the most important methodological lesson of this series).** `dc12` tested an MT plateau,
but the *shared* `ReduceLROnPlateau` on `rho_combined` owned every parameter group, so one patience-1 trigger
at a time drove all rates down together to ~5e-7 by epoch 7. The MT head was effectively frozen shortly
after its peak; the arm was retired as `_invalid_..._inert_plateau`. After the shared scheduler was removed
(`baf2f76`: each head plateaus on its own metric), the **dc12r triplicate** measured the real effect:
`beyond_rho_mt` +0.0101 for the plateau treatment, **t = 1.66 on 2 df — not significant**.

| arm | cell_rank_mt | cell_mag_comb | flipacc_mt | beyond_rho_mt | rho_wt_valid | cell score | epochs |
|---|---|---|---|---|---|---|---|
| `dc8` triplicate (s1/s2/s3) | 0.1981 / 0.1855 / 0.1921 | 0.246-0.248 | 0.825-0.862 | +0.502…+0.537 | 0.834-0.836 | — | 8 |
| `dc9` reg ×5 | 0.1872 | 0.2411 | 0.8235 | +0.4967 | 0.8353 | — | 4 |
| `dc12r` triplicate | 0.1999 / 0.1875 / 0.1846 | 0.244-0.248 | 0.794-0.848 | +0.522…+0.536 | 0.835-0.836 | — | 8 (resumed) |
| **`dc14`** (both plateaus p1/f0.3, 10 ep) | 0.1987 | 0.2470 | 0.8328 | +0.5389 | 0.8348 | 52.6 | 10 |
| **`dc15`** (+`--no-wt_trains_doubles`) | 0.1975 | 0.2672 | 0.8280 | +0.5324 | 0.8397 | 53.4 | 8 of 10 |
| **`dc17`** (new canonical, 5 changes) | 0.1975 | **0.2704** | **0.8513** | +0.5270 | 0.8386 | **54.3** | 8 of 10 (self-stopped) |
| `dc18` (floor pinned to 0) | 0.1916 | 0.2670 | 0.8592 | **+0.4857** | 0.8394 | — | 7 (self-stop) |

`dc17`'s five changes, taken together by decision (so no single one is attributable): both heads get
patience 1 / factor 0.3 / **three** decays before freezing and the run stops itself when every group is at
lr 0; `--no-wt_trains_doubles` (already the default); `--per_mutation_bias`; and a link in which **nothing is
learned**. It stopped at epoch 8 of the 10 cap when both heads froze — the cap is now a ceiling, not a
target.

**Noise, measured from the two triplicates** (at epoch 7, across seeds): `cell_rank_mt` SD 0.0069-0.0090,
`cell_mag_comb` 0.0023-0.0082, `cell_flipacc_mt` 0.0237-0.0296, `beyond_rho_mt` 0.0082-0.0177. This
supersedes the Oct-8 estimates and is why `dc14 → dc15 → dc17` should be read as **one believable gain**,
`cell_mag_comb` (+0.023, several SD) plus `rho_wt_valid` (+0.004), and **no credible gain** on
`cell_rank_mt` (0.1987 → 0.1975, well inside noise) or `flipacc_mt`.

**dc18 verdict (completed 2026-10-18 22:39, 7 epochs).** Pinning the link floor to 0 **hurt the MT side**:
`beyond_rho_mt` 0.4857 against dc17's 0.5270 is −0.041, two to four times the replicate SD, with rank/mag/flip flat.
That is the launcher's predicted mechanism — where a library's measured floor is below 0, a censored item's hinge
target sits below anything the link can emit, so the hinge can never be satisfied. **The WT guard, however, was
marginally better** (0.8394 vs 0.8386; its epoch-1 peak 0.8414 is the highest seen anywhere). So the knee is good
for the WT half and bad for the MT objective — the tension the WT sweep's new `floor` stage exists to resolve,
judged on the WT head's own metric and stacked only where it belongs.

**What has never moved**: `val_epi_global_bias_comb` sits at ≈ −0.50 in *every* arm (dc3 through dc18,
including the ones that changed the floor). Whatever produces that half-kcal/mol residual is not the floor
parameter.

## 6. Failures, retirements, and races (the triage record)

| log | why it does not count |
|---|---|
| `_invalid_dc12_s1_inert_plateau` | the shared scheduler compounded the lr; the MT plateau was never really tested (fixed in `baf2f76`, re-run as dc12r) |
| `_invalid_dc17_s1_25loaders` | `NameError: assay_floor` in the pinned snapshot: 3 benchmark loaders failed, **25 of 28 validation loaders** came up (fixed `b34093f`; good runs log loader 27 of 27) |
| `_confounded_dc5_rangecens_1605` | two changes at once — not attributable |
| `_cudafault_dc5_c4_1656` | died of a CUDA fault after one epoch |
| `_aborted_dc17_s1_stopped_e0` | stopped by request at epoch 0 |
| `_aborted_dc18_s1_queue_held` | aborted at queue handover; restarted cleanly at 18:14 |
| `_truncated_dc15_s1_e8of10` | queue moved on after epoch 7; not a crash |
| `dc16_s1` | killed 8 minutes in, at epoch 0: **dc17 was launched on top of it**. The GPU-busy guard samples `pgrep -x python` and reads `/proc/<pid>/cmdline`; a killed-but-unreaped python has an empty cmdline and is invisible. dc16 (`dc15 + --lambda_rank_mt 0.2`) was never redone — **`dc19` re-tests that variable on the dc17 basis** |
| `bl_w2a4_m16a16_capped_s1` (baseline triplicate, 2026-10-17 22:34) | the log stops after Comet initialisation, no `metrics.csv`, no epochs: the set was abandoned when the queue was repointed at dc14. **The released-model control at matched epochs is therefore still missing** |
| `o6s16_mt32` | degenerate metrics (`cell_mag` 0.000) — treat as a failed run, do not read the rank figure as a capacity result |

## 6b. 2026-10-19 — the floor knee, measured at the base configuration

The rescheduled WT sweep put the knee first. The first floor=0 arm was accidentally run at the **base** settings
rather than the standing stack (rank 2 / a4 / dropout 0.1 — the sweep's start-up truncated `wt_stack.txt`; see
STATE pitfall 12), which turned it into a clean matched pair against the dawn `floor` arm and is why it is kept as
**`wt_floor0_base`** (Comet: `wt_floor0_base-0`, previously misnamed `wt_floor0_drop02`).

| arm | config | best `val_rho_wt_valid_avg` | epochs |
|---|---|---|---|
| `wt_floor_data` | base + per-library floors (knee -1.0..0.2) | 0.8391 | 8 (pre-patch, ran to the cap) |
| `wt_floor0_base` | base + one asserted knee at 0 | **0.8403** | 6 (post-patch, self-stopped) |

Per-epoch the two curves are almost the same (largest gap 0.0017; identical at e4: 0.8391 both). Against the
~0.0005 between-seed SD on this metric the difference is ~2 SD, and it favours floor=0; the data-knee arm also had
two more epochs to find its peak. **Conclusion: at matched config, the knee moves nothing measurable and costs
nothing.** That is the empirical support for the parsimony prior (a single asserted knee, no per-library estimate
needed for 123 of 170 libraries), and it makes the zero knee the default unless something contradicts it.

Consequence for the schedule: the arm now running (`wt_floor0_drop02b`, rank16/a32/dropout0.2 + zero knee) was meant
to be a **confirmation that the knee is still neutral-or-better under the new capacity/dropout stack**, compared
against `wt_drop02` (0.8447, same stack, data knee) — not first evidence. **It never got a verdict:** it was stopped
at epoch 1 (12:09, `val_rho_wt_valid_avg` 0.8335 / 0.8307) to free the GPU for the reference run, and archived as
`logs/_aborted_wt_floor0b_oldmetric_38min`. The alpha-32 seed-2 pair was dropped as superseded at 12:00. So the knee
call rests on the base-config matched pair above plus the parsimony prior; the general evidence rule that governs such
calls from now on is STATE §5.2 (a change that is not clearly outside measurement error earns one extra seed for both
arms before it stacks; the external sets are a guard, not a criterion).

## 6c. 2026-10-19 — the validation metric basis changes (OOD excluded)

The three benchmark libraries (`ptmuld`, `s461`, `ssym`) are out of the training distribution, so they no longer enter
any validation aggregate: `val_*_avg`, the pooled global ranking and the whole `val_epi_*` hierarchy are now
in-distribution only. They are still logged per library and summarised as `val_rho_combined_external` /
`val_rmse_combined_external`; the list is the `--external_val_sets` flag and is carried into generated frozen-WT blocks
(STATE §5.1, `docs/validation_metrics.md`).

**Verified on the first run under the new basis** (reference epoch 0, 2026-10-19 12:50, `check_metric_basis.py` exit 0,
not the diff): `val_rho_combined_avg` = **+0.838533** equals the nanmean over the 25 in-distribution per-library columns
exactly, and the all-28 mean would have been **+0.822933** — so the three sets are genuinely out of the aggregate.
`val_rho_combined_external` = +0.692935 / `val_rmse_combined_external` = 1.386. The pooled ranking recomputed from
`val_dump_e0.npz` (34,957 items, `SRC_RANGE` held back) gives **+0.860621**, matching `val_rho_ungrouped_combined`, and
the dump contains **zero** OOD items — the external loaders are skipped before pooling, not merely down-weighted.

**This is a hard boundary for comparisons.** Any aggregate quoted before this date includes the OOD sets and is not
comparable to one after it — recompute from the per-library columns (`val_<metric>/<library>`), which are unchanged.
The runs that straddle the boundary: the alpha-32 zero-knee arm (`wt_floor0_drop02b`) ran under the OLD
definition (and has no verdict), and the gold-standard baseline (`wt_ref_floor0_r16a16_d02`, floor 0 / r16 / a16 /
dropout 0.2, started 12:29 on 2026-10-19) is the first under the NEW one. Comparing anything across that line requires
the manual recomputation from the per-library columns, which is expected and cheap.

## 6d. 2026-10-19 — the new basis could not run at all: `ext_per_loader` was never initialised

The OOD-exclusion commit added both the per-loader assignment
(`ext_per_loader[name] = {k: metrics[k] for k in EXTERNAL_SUMMARY_METRICS if k in metrics}`) and the
`summarise_external_benchmarks(ext_per_loader, …)` call, but never initialised the dict. Every run on the new basis
therefore died at the **pre-training zero-shot validation** — the first `on_validation_epoch_end`, before any training
— with `NameError: name 'ext_per_loader' is not defined` (first victim: the reference run, 2026-10-19 12:22, 7 min in).
The pure helper was unit-tested and the calling method was not, which is why the suite was green. Fixed in `75d3e34`
(one line); `309c27e` adds a test that drives `on_validation_epoch_end` with one in-distribution and one OOD loader and
fails with the exact `NameError` if the initialisation is removed. A second, environment-only failure in the relaunch
(`PermissionError` on `multiprocessing.SemLock`: the launching shell's sandbox denies `/dev/shm`) is recorded in
STATE §4 — it is not a code bug and must not be "fixed" by lowering `num_workers`.

## 6e. 2026-10-19 — the frozen-WT generator was about to freeze the wrong head (three preflight bugs)

`make_fwt_block.py` builds the frozen-WT / tf-mix arms from an explicit checkpoint. A preflight against the new
reference checkpoint (`--verify`, run before anything was queued) found three defects, all of the stale-pointer or
silent-override class:

1. **`--verify` imported `esm_msr` from the retired pin** (`dc_runs/pin_fwt_series/src`), which predates
   `--external_val_sets` and rejected every generated arm. It now imports `TREE/src`, the tree that actually runs.
2. **The checkpoint's architecture was inserted BEFORE the template's own `--lora_rank_wt 2 / --lora_alpha_wt 4`**, and
   argparse takes the last value — so the frozen model would have been built at **WT rank 2** while the log printed
   "frozen WT checkpoint" and the summary showed the gold head's rank 16. Now appended at the END of the command, and
   `--verify` reports `verified ... rank_wt=16` for all five arms.
3. **Checkpoint names carry literal single quotes.** `--checkpoint_filename` was passed through
   `$(cat wt_base_args.txt)`, and command substitution does not strip the shell quoting, so the file is actually named
   `'epoch=02-val_rho_wt_valid_avg=0.8504'.ckpt`. Consequences: `epoch=*.ckpt` globs match **nothing**, and an
   unquoted interpolation of the path loses the quotes. Fixed three ways: `gate_capacity.best_ckpt` globs `*.ckpt` and
   matches the epoch inside the name; `make_fwt_block` `shlex.quote`s the path and its `--verify` parses with
   `shlex.split(..., comments=True)` while skipping the template's backtick-wrapped comment lines; and
   `wt_base_args.txt` no longer carries the quotes, so arms launched from now on get clean names.

Any one of them would have produced a frozen-WT block that trained a different architecture against the gold head —
the "the architecture must travel with the weights" rule broken silently, at the gate that feeds the whole tf-mix/ns
stage.

## 6f. 2026-10-19 — the reference was killed by the driver, then resumed (with one record casualty)

At ~14:24, during epoch 5's validation, the reference process and its queue runner both died abruptly with **no
traceback in either log** — the signature of a driver-level kill (an exception would have printed). The GPU came back
idle on its own. Epochs 0–4 had completed:

| epoch | 0 | 1 | 2 | 3 | 4 (best) |
|---|---|---|---|---|---|
| `val_rho_wt_valid_avg` | 0.8389 | 0.8390 | 0.8504 | 0.8457 | **0.8513** |

The epoch-4 checkpoint and `last.ckpt` were intact, so the run was **resumed** from end-of-epoch-4 into the same
experiment name and version 0 (`--skip_val` to avoid a duplicate zero-shot row). The restore is confirmed
(`Restoring states from the checkpoint path …`, then `Epoch 5`).

**Casualty**: Lightning's `CSVLogger` rewrites `metrics.csv` from scratch on resume, so epochs 0–4 were erased from the
file even though the checkpoints survived (STATE pitfall 13). The numbers above are recoverable from the archived
attempt log, the checkpoint filenames (4 dp) and `val_dump_e0..e4.npz`. `gate_capacity.py` now takes an arm's best as
the max over `metrics.csv` and the checkpoint names, so the capacity comparison still uses the true 0.8513.

The resumed run then stopped cleanly at **epoch 5 (0.8500)**, the third plateau decay, so the reference's final
record is 0.8389 / 0.8390 / 0.8504 / 0.8457 / **0.8513** / 0.8500 — best at epoch 4, which is the WT reference the
capacity arms are compared against on the new basis.

A second parser bug surfaced while recovering the record: once `wt_base_args.txt` stopped carrying the quotes,
**new** checkpoints are clean-named (`epoch=05-val_rho_wt_valid_avg=0.8500.ckpt`) and the value regex
`([0-9.]+)` swallowed the dot before `ckpt` (`0.8500.`), which would have crashed the capacity gate at decision
time. All value parsers now use `(\d+\.\d+)` and are tested against a directory mixing quoted and clean names.

Also note the resume is a **continuation, not a replay**: RNG state is not carried by the checkpoint, so epochs 5+
can differ from what an uninterrupted run would have produced. Epochs 0–4 are the original measurements.

## 6g. 2026-10-19 — a WT-only arm ran its last epoch with no loss (a pin patch the consolidation dropped)

`wt_r8a11` was caught by the user, not by a check: the WT path converged (best 0.8510 at epoch 4) and froze after its
third plateau decay at epoch 6, and then epoch 7 started anyway — **4.9 GB reserved instead of 24.7, 1.9 it/s instead
of 0.55, `lr/lora_mt` 2e-4 and no loss**. The run was spending its epoch budget re-scoring fixed weights.

Cause: `_transition_to_mt()` unfreezes the MT path unconditionally, and `_stop_if_nothing_is_training()` reads
**learning rates**, not `requires_grad` — a thawed MT group at its base LR looks live forever. The pin had the guard
(`if freeze_mt_adapter: freeze_mt_components()`) as a recorded hand patch from 2026-10-18 23:35; it was never
upstreamed, so the consolidation onto `devel-revision` silently lost it. That is why the pin-era WT arms stopped
correctly and why the **resumed** reference did too (a resume skips warm-up; the warm-up `LambdaLR` had been undoing
the `on_train_start` zeroing in fresh arms). Fixed in `9b73d3a`, pinned both ways by
`tests/test_transition_freeze.py`. Cost so far: one wasted epoch on `wt_r8a11`; the same arm would have cost it again
on `wt_r4a8` and on every frozen-WT / tf-mix arm.

## 7. Open questions (in priority order)

1. **Does the nonspecific term help on the current basis?** The best lead in the project, single seed —
   plan in `docs/nonspecific_epistasis.md` §7.
2. **The missing released-model control.** A matched-epoch, same-libraries comparison of the released
   objective against the current canonical is needed; the last attempt was abandoned at launch. `dc20`
   (scoring-only pass on external doubles) covers the external side but not this.
3. **Does cell-level capacity respond at all?** Screening said MT rank 8 > 16 > 32 (32 unusable), one seed
   each, on superseded constants. `dc23` re-tests rank 8 on the dc17 basis.
4. **The floor.** Data floors, zero, and learned floors have all been tried and none moved
   `global_bias_comb`. Either the residual is not the floor, or the metric measures something else — worth a
   dedicated diagnostic before another floor arm. *Update 2026-10-19*: at matched config the knee costs nothing
   on `val_rho_wt_valid_avg` (§6b), so the remaining question is not whether floor=0 is safe but whether it buys
   anything at the MT level on a frozen WT head (that is the A/B `gate1_tfmix` emits).
5. **The WT sweep's effect on the MT phase.** `claude_queue.txt` runs the WT head alone first (doubles
   removed from the batches, ~1/3 the cost) and stacks winners at a 0.001 margin. Every MT arm after it will
   be built on a different WT head than dc17 — so dc17 vs dc19+ must not be compared across that boundary
   without care.
6. **Inference plumbing.** Nothing trained with `--link` or `--nonspecific_term` can be shipped until
   `inference_scripts/` reads `link_head.*` and `ns_head.*`.
7. **Per-item noise weighting at both ends of the assay (idea, 2026-10-20, not scheduled).** Measurement noise is
   heteroscedastic: the two proteases' estimates of the same single disagree by a median 0.32 kcal/mol in range
   (dG 1-3) but 0.79 at dG -1..0 and 0.52 at 4.5-5 (15% disagree by more than 1 kcal/mol there, 10% at the floor).
   Every loss currently weights all uncensored items equally. The data carry per-item uncertainty
   (`deltaG_95CI`, `deltaG_t_95CI`, `deltaG_c_95CI` in `meas`, see `data.MEAS_COLUMNS`), so the regression could
   use inverse-variance weights, and the rank loss could down-weight near-noise pairs. This is the principled
   alternative to adding ceiling parameters: the top of the assay shows rising noise and censoring (`>5` tags), not
   compression (no pile-up below 5; trypsin and chymotrypsin do not bend apart below 5; the WT head's top-end
   residual is mostly the selection that drops `>5` rows). Analysis in the 2026-10-20 review session.
   *Update 2026-10-22: implemented as `--ci_weight`, see `docs/ci_weighting.md`. The reported interval (`deltaG_95CI`) turned out to be
   irrelevant to the model's held-out error; the label's uncertainty is its library-centred protease disagreement (calibrated coefficient
   0.263 ± 0.015 against the theoretical 1/4), with the double's additive expectation as a depth prior. With the floor bound kept the realistic
   gain over `dthr0` is within seed noise (weighting efficiency 1.065). First test queued: `mt_frozenwt_ciw_r16a16_s1`.*

## 8. 2026-10-22 — the training code was trimmed, and the arms that test what the audit found untried

`docs/trim_audit.md` audited every flag against 131 completed runs' `hparams.yaml`, the launchers and the code (98 of 147 behavioural flags never
took a second value; ~460 lines of `utils.py` were unreachable), the plan was approved whole, and it was implemented on
`claude/trim-training-2026-10-22` with loss parity (bit-identical, 11 recipes) and data parity (identical libraries, items, censoring and all 411
sampler batches on a subset) against the pre-trim code. The label-uncertainty weighting (`--ci_weight`, §7 item 7) was removed with it: its one
arm, `mt_frozenwt_ciw_r16a16_s1`, runs from its own pin and is the feature's only test.

The new arms (project `esm-msr-trim-oct22`, seed 1, frozen WT = best epoch of `wt_r2a4_f0_cthr05_s1`, one variable each against `trim_ctrl_s1`):
`trim_ncmt_s1` -> `trim_ncmt_late_s1` (cancelled and requeued: `--native_cond_for_mt` adds no examples, it swaps the MT input for 1.7% of the stream), `trim_joint_dual_s1` / `trim_joint_fused_s1` (one shared adapter against two, joint
training), `trim_mask_mt_ind_s1` (independent masking on the MT pass), `trim_huber_s1` (robust regression), `trim_rank0_s1` / `trim_rank025_s1` /
`trim_list64_s1` / `trim_reg10_s1` (MT loss balance and list length), `trim_tfmix0_s1` (never teacher-force). Single-seed screens: per the promotion
rule a lead needs a second seed for both arms before anything stacks.

