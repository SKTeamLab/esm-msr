# The frozen-WT paradigm: train the MT head against a fixed, pre-trained WT head

**Status 2026-10-18: implemented and pinned; arms prepared, NOT started.** Code on
`feature/nonspecific-term` @ `cf2b450`, pinned as `dc_runs/pin_fwt_series`; the launchers
`dc_runs/launch_fwt_*.sh` and `dc_runs/queue_fwt.txt` exist. The from-scratch ns set
(`queue_ns.sh`, `pin_ns_series` @ `7db75a7`) is running and unaffected.

## 1. What it is

Every epistasis run loads **its WT half from an existing checkpoint**, freezes it, and trains **only the MT path**
(plus the link and, when enabled, the nonspecific term) **on multi-mutants alone**. The WT head is then a fixed
additive-expectation oracle: whatever it predicts for a double's two singles is the background that double is
scored against, for the whole run.

Three existing primitives already covered most of this — `--freeze_wt_adapter`, `--no-incl_singles`,
`--no-wt_trains_doubles`. What was missing was **loading a pre-trained WT path**; that is what `cf2b450` adds,
together with a named flag for the combination:

| flag | what it does |
|---|---|
| `--init_wt_from_checkpoint PATH` | copies the **whole WT model** — adapter, calibration head **and link** (`link_head.*`) — out of a full Lightning `.ckpt` or a `converged_wt_*.pt`. Nothing else is loaded. A path typo **raises** instead of quietly training a random WT head; a checkpoint without `link_head.*` leaves the link as configured (logged, not an error). |
| `--freeze_link` | freezes every link parameter. The link is the last stage of the WT readout (latent → observed ddG): its floors, temperature and global floor shape the observed-scale prediction the WT loss and the rmse guard read, so a frozen WT adapter paired with a link that still moves would not be the model being frozen. `--frozen_wt_mt_only` turns this on and requires `--link`. |
| `--frozen_wt_mt_only` | the paradigm as one flag. Requires a checkpoint; derives `--freeze_wt_adapter`, `--freeze_link`, `--no-incl_singles`, `--no-wt_trains_doubles`, `--lambda_reg_wt 0 --lambda_rank_wt 0`, logging each change. Because it is derived in `parse_arguments`, the run's `hparams.yaml` records the *effective* settings. |

Deliberately **not** loaded from the checkpoint: the MT adapter (always fresh) and `ns_head.*` (the nonspecific term
is the thing those arms train; it stays configured by its own flags). Validation is untouched —
`setup_dataloaders` hardcodes singles+doubles for the val loaders — so `val_rho_wt_valid_avg` stays a guard that the
frozen head is the one you think it is.

## 2. Why

1. **The WT head is separable.** Since `f21c913` it trains on singles only, and for teacher forcing a double only
   needs the WT prediction where a measured additive label is missing. Nothing about the MT objective requires the
   WT head to move *during* the run.
2. **A moving WT half is a moving target.** The dc17 launcher records the diagnosis: the WT adapter "was taking
   full-rate steps on a metric it had stopped improving - and moving the target the MT path teacher-forces
   against." dc17 then froze the WT head after three decays anyway (effectively from epoch 5), so the frozen-WT
   paradigm is largely making explicit what the best current recipe already converged to.
3. **The nonspecific term gets a stationary `x`.** `f(x)` is a function of the additive expectation; with the WT
   half fixed, the distribution of `x` does not drift while `f` learns. That removes one confound from the ns test
   (see `fwt_ns_s1` below).
4. **Cheaper epochs, on the same item coverage.** Measured on the real train split (172 libraries, 170 cached;
   **156,265 singles, 61,331 doubles**, cond excluded as always):

   | batch size | dc17-style (singles+doubles) | doubles-only |
   |---|---|---|
   | 256 | 771 steps/epoch, **95.0%** of doubles seen | 217 steps/epoch, **90.6%** of doubles seen |
   | 128 | 1601 steps, 97.5% | 454 steps, 94.8% |
   | 64  | 3300 steps, 98.8% | 933 steps, 97.4% |

   **All doubles are still seen once per epoch** — every item lands in exactly one batch — but two things change:
   the per-library chunk remainder now discards 9.4% of the doubles instead of 5.0%, and each epoch has **~3.6x
   fewer optimiser steps** (217 vs 771 at batch 256), each carrying 256 doubles instead of ~49. Per-epoch double
   exposure is therefore about the same (55.6k vs 58.3k instances), while the number of updates is not: 10 epochs
   is ~2,170 MT steps against dc17's ~6,930. The plateau schedule is per epoch, so it decays after that many fewer
   updates. If the MT head looks undertrained at the cap, **raise `--num_epochs`** rather than the learning rate.
   The remainder also argues for a smaller `--batch_size` on these arms if a library's doubles regularly fall just
   short of a multiple of 256.

## 3. Choosing the frozen WT head

Chosen: **dc17's converged WT model**, `dc_runs/logs/dc17_3dec_nowtdbl_pmb_w2a4_m16a16_s1/0/epoch=08-val_rho_combined_avg=0.840.ckpt`.

Rationale, and the caveat that goes with it:

* dc17 is the current canonical and its WT head is the one the MT arms have been trained against;
* its WT plateau decayed three times (e1 → e3 → e4 → e5) and **froze from epoch 5**, so epochs 4-8 carry the same
  frozen WT weights. `rho_wt_valid` is 0.8386 from e4 to e8;
* **the peak was e1 at 0.8406** — about 4x the between-seed SD — but that checkpoint was not kept (top-5 is ranked
  by `val_rho_combined_avg`, and e1 was not in the saved set). If the frozen WT head is to be the *best* WT by its
  own guard, that needs a separate decision: re-save on `rho_wt_valid`, or drive the WT-only sweep
  (`wtsweep` in `claude_queue.txt`) and freeze its winner. The paradigm and the sweep compose naturally, and that
  is the better long-run route to "the best frozen WT adapter".
* dc18 does not offer a better WT: it is dc17 plus the floor pin, same seed, so its WT recipe is identical.

The link travels with it. Verified against the artifact, not assumed: the checkpoint carries
`link_head.{raw_tau,floor_min,floor_max,global_floor}` and the loader reproduces its exact summary —
**tau 0.59, floor band −1.0…0.2, global_floor −0.136** — which is then frozen (`--freeze_link`).

To use a different head, change the `--init_wt_from_checkpoint` path in the launchers (one line each).

## 4. Acceptance check — do this before trusting an fwt arm

Two independent signals that the WT head actually loaded, both in the first minute of the log:

1. `WT path warm-started from ...: 586 tensors loaded` **and** `Link warm-started from ...: 4/4 tensor(s)
   (raw_tau, floor_min, floor_max, global_floor)`;
2. at the step-0 validation, `val_rho_wt_valid ≈ 0.839`. If it is ≈ 0.55, the WT head did **not** load (that is
   the untrained value) — stop the arm. The logged `link/*` values should also read tau 0.59, floor −1.0…0.2,
   global_floor −0.136 (dc17's), not this run's configured defaults.

## 5. The prepared series (`dc_runs/queue_fwt.txt`, not started)

| arm | what it is | what it answers |
|---|---|---|
| `fwt_ctrl_s1` | dc17 objective, frozen dc17 WT, MT-only, doubles-only | what the paradigm itself costs or gains against `dc17_s1` |
| `fwt_ns_s1` | + `--nonspecific_term fx` | the ns term against a **fixed** `x` |
| `fwt_ns_s2` | seed 2 of the above | the replicate; judged against `fwt_ctrl_s1` as `ns1_s2` is against `ctrl_s2` |

**Comparability.** `fwt_ctrl_s1` vs `dc17_s1` changes three things at once (WT head fixed, stream doubles-only,
WT losses off), so it is a paradigm-level A/B and not an attribution. The clean single-variable comparison is
`fwt_ns_*` vs `fwt_ctrl_s1`. Arms from the two code pins are otherwise comparable: with the new flags off the
pin's code is behaviourally identical to `pin_ns_series`.

## 6. The floor knee is A/B'd here, not assumed

The frozen head carries the knee it was trained with, and `make_fwt_block.py` reproduces it from the checkpoint's
hparams. But *what the knee does to the MT objective* is only measurable with the WT head frozen: during WT training
the MT columns are meaningless (the MT adapter is untrained, so its symmetrised levels read 0.0), and dc18's
−0.041 on `beyond_rho_mt` came from re-censoring the **ensemble's** training targets, not from a frozen head.

So `gate1_tfmix` emits one extra arm after the WT sweep: the same frozen head, the same mix (1.0), but the **other**
knee (`make_fwt_block.py --knee data|zero`). Its matched counterpart is the block's own `fwt_ctrl_s1`, so the pair
differs only in the knee. The gate reads the checkpoint's `assay_floor` to decide which knee is "the other" one, so
the arm is never a duplicate. Read `beyond_rho_mt` with the usual margin (0.03); guards unchanged.

## 7. Risks and open questions

* **The stream's library population does not change, but its remainder does.** 118 of 170 training libraries hold
  no doubles at all (they contribute nothing either way); the 52 that do are all still visited every epoch, and no
  library's doubles fall below one batch. What does change: the per-library remainder discards 9.4% of the doubles
  per epoch instead of 5.0%, and the batch composition goes from ~19% doubles to 100%. Both are visible in the
  step count, and `--batch_size 128` recovers most of the lost remainder at 2x the steps.
* **Fewer, larger updates.** ~217 MT steps per epoch instead of ~771: the per-epoch plateau therefore decays after
  roughly a third of the updates dc17 got. That is the most plausible way the paradigm could under-train the MT head.
* **The MT head never sees a single in training.** Singles still appear in validation, so the guards hold, but any
  MT behaviour on singles is untested by design.
* **Teacher forcing depends on measured additive labels.** For doubles whose singles are missing or measured at the
  floor, the WT pass supplies the background — a frozen WT therefore contributes a *fixed* bias for exactly those
  items. Worth watching in `global_bias_comb` (which no floor device has moved so far anyway).
* **`--wt_plateau_*` is inert** with the WT frozen (its group's lr is zeroed by `enforce_freezing`).
* **Speed has a second, unused lever**: with the WT frozen and teacher forcing on, the WT forward could be skipped
  for doubles that have an additive label (the `phase == 'mt'` path in `--alternate_heads` does exactly that). That
  would roughly halve the remaining training cost; it changes which items the MT losses cover, so it needs its own
  arm, not a default.
* **Adoption is not automatic.** As with the ns term, `inference_scripts` read neither `link_head.*` nor
  `ns_head.*`; and nothing here changes the released model, only how epistasis runs are started.
