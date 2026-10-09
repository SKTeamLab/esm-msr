# Summary of recent work (2026-10-08)

Branch `claude/epistasis-metrics-logging-cfd318`, merged into `devel` and pushed. The analysis is in `docs/epistasis_metrics_report.html`
(also published as a private artifact: https://claude.ai/artifact/TGASgegY2H9x72EGcjBrSb); every metric is documented in
`docs/validation_metrics.md`; how to run and read the queue is in `docs/handoff_oct08.md`.

## 1. Findings

* **Where the variance of measured dddG lives** (validation doubles: 24 position-pair matrices in 9 libraries, 8,494 doubles): global, a
  function of the additive expectation x = dG_wt + ddG_A + ddG_B alone, 56%; pair offsets 22% (70% of it between libraries); line effects 14%
  (mostly the shared measurement noise of the singles); the specific combination 6% (about the doubles' own noise). The training-set ANOVA
  agrees (60 / 16 / 13 / 10%, cross-validated).
* **Saturation leaks into every statistic on raw dddG**, because the floor makes even an additive model non-additive on the observed scale.
  In simulation a predictor that knows only the saturation scores 0.58 on the naive dddG correlation, more than one that knows every epistasis
  component but not the saturation (0.42). Doubles below the floor (x < -1, 18% of doubles, 34% of the dddG sum of squares) carry almost no
  specific information except rescue. Two devices remove saturation exactly for an additive model: within-column order of ddG_AB, and residuals
  beyond each side's own global curve.
* **The link's floor is about 1 kcal/mol too low.** Doubles far below the floor are measured near dG +0.03 (median of 61 libraries, range -0.55
  to +0.70); the learned floor is -0.99. Every trained model under-predicts the nonspecific epistasis by 0.5-0.7 kcal/mol, and the MT head
  absorbs part of it. A nonlinear recalibration fitted to the validation set independently puts the floor at +0.29.
* **The released baseline (MT rank 16, no link) led every devel arm at the cell level** at matched epochs (cell_rank_mt 0.197 vs 0.146-0.167 at
  epoch 2). The no-link devel arm was no better than the link arms. MT rank 16 on devel (o6q01_mt16, old balance constants) reached
  cell_rank_mt 0.192 / 0.188 / 0.190 at epochs 3-5 (mean 0.190, about +0.02 over the rank-2 references at the same epochs, 0.154-0.176),
  reversal accuracy 0.83-0.84, +0.055 over the WT-in-context control, guards unchanged (rho_wt_valid 0.841, rmse_combined 0.601). It plateaus
  by epoch 3-5, so 6 epochs is enough.
* **MT regression weight matters across the board.** o6g1_bal111 (MT components 1/1/1 at master 1.0, WT regression 1.0) beat every other devel
  arm on every level without hurting the guards. The devel arms order by master x component weight; the gain comes with lower plain and censored
  regression losses, not a better-fitted interaction component.
* **The MT adapter adds little over the WT adapter read in context** in the rank-2 devel arms (about +0.02 on cell_rank), +0.06 in the baseline.
* **4 epochs is too short** for arms that change MT capacity (o6q01 gained +0.023 in its 4th epoch). Queues use 6.

## 2. What was built

| piece | where | default |
|---|---|---|
| 20 validation metrics by level (3 ddG guards, 17 epistasis metrics with saturation and WT-in-context controls) | `src/esm_msr/epi_metrics.py`, `training.py`; `docs/validation_metrics.md` | on |
| checkpoints and the MT plateau on `val_epi_cell_rank_mt` | `scripts/run_arm.sh`, `training.py` | on |
| offline rescoring of any validation dump with intervals | `scripts/epi_from_dump.py` | |
| the same metrics plus the raw (confounded) levels in testing; RMSE columns | `inference_scripts/esm_msr_testing.py` | on |
| per-dataset recalibration (linear / nonlinear link), proof of concept | `esm_msr_testing.py --recalibrate` | off |
| confident-reversal loss on the MT latent interaction contrast | `--lambda_mt_flip`, `--flip_delta`, `--flip_scale` | off |
| whole pair matrices per batch + prediction cache, so components and reversals are computed on whole matrices (exact gradients, tested) | `--pack_pair_matrices` (needs `--batch_size` >= 361) | off |
| measured gradient-balance constants from a file, separate constants for packed components and the reversal loss | `--reg_balance_file`, `scripts/grad_share_probe.py`, `scripts/balance_from_probe.py` | |
| queue tooling: file-driven runner, resume, probe-then-queue | `scripts/queue_runner.sh`, `resume_arm.sh`, `probe_then_queue.sh` | |
| `--val_cycle_passes` (the ctx control) on by default | `config.py` | on |

## 3. Running now

`scripts/probe_then_queue.sh` (started 2026-10-08 22:17): after o6q01_mt16's resume, gradient-share probes on the GPU (the reference checkpoint
with micro-batch slices and packed, the rank-16 checkpoint with slices), constants written to
`/home/sareeves/playground/esm-msr-devel/probes/reg_balance_oct08.json`, then queue r2 (`scripts/queue_r2.txt`, 18 arms of 6 epochs, about 3
days): the reference under the new constants, MT rank 16 / 8 / 32, MT master weight 1 and 3, whole-matrix packing with and without the reversal
loss, floor censoring, colrank off, combinations and second seeds.

## 4. Things that could still be implemented

1. **Masked components on incomplete matrices.** On whole matrices every missing or censored cell makes the complete-block trimming drop a row
   or a column (355 vs 390 cells per 1,000 items reach the components); a two-way decomposition with missing cells would keep them.
2. **A partial last batch per library.** At batch 400 the sampler drops 11.6% of the items each epoch (8.6% at 256): the remainder that does not
   fill a batch.
3. **The floor.** If floor censoring (o6s10) helps, make it the default, or give the link a higher, learned or library-aware floor; judge by
   `val_epi_global_bias_comb` and `global_err_add`.
4. **The partner single's noise in conditional regression.** A conditional item is scored through the link with the MEASURED partner single as
   its background, so that single's measurement noise becomes an unpredictable column effect of the error; use the model's own single or
   regress the double directly.
5. **Retire or shrink the interaction component** of the MT regression if the reversal loss carries the cell level (its target is at the noise floor).
6. **A soft-margin reversal loss**: weight each candidate reversal by its probability of being real under the noise model instead of a hard 0.6.
7. **The WT-in-context control in inference** (the reverse WT leg), so `esm_msr_testing.py` can report the ctx head too.
8. **A global recalibration** (for example a calibration head trained on all external ddG data) to replace the per-dataset proof of concept.
9. **A larger or cross-validated validation set for the pair level** (24 pairs from 9 libraries cannot rank configurations).
10. **Promote winners to defaults** in `scripts/run_arm.sh` and `config.py` once r2 has two seeds of them (batch 400 only with packing).
