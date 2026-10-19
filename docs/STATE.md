# Current state (last updated 2026-10-18 21:00 EDT)

Read this first, then `docs/README.md` for what every other document is. Facts here are dated where they
matter; anything older than the date below is marked as history in `docs/EXPERIMENTS.md`.

## 1. What this project is, and where the work stands

The repo ships the published **ESM-MSR** model (ESM3-small + LoRA, mutant stability ddG). The research line
on this checkout extends it to **double mutants and epistasis**: a dual-adapter architecture (WT head for
singles, MT head for multi-mutants), a link that models the readout with a saturation floor, censoring for
bounded rows, and a cell-level epistasis metric suite. The target is `val_epi_cell_rank_mt` plus its two
companions (`val_epi_cell_mag_comb`, `val_epi_cell_flipacc_mt`) with `val_rho_wt_valid_avg` /
`val_rmse_combined_avg` as guards; `val_epi_beyond_rho_*` is the saturation-free interaction level.

Current best cell-level result on record: the **`--nonspecific_term fx`** arm (one seed) —
see `docs/nonspecific_epistasis.md`. Current best *canonical* arm: **dc17**.

## 2. Code: branches, worktrees, and the pinning rule

| ref | path | what it is |
|---|---|---|
| `devel-experimental` (workspace) | `repo/` | the dev branch; epistasis metric suite, packing, flip loss, grad-balance |
| `devel-revision` | `.claude/worktrees/devel-main-conservative-merge-5e554f` | conservative rebuild of the released objective + floor link |
| `claude/main-baseline` | `.claude/worktrees/main-baseline` | released-behaviour line; the basis of the Oct 13-14 link series |
| `claude/val-doubles-and-masking` | `.claude/worktrees/practical-snyder-52312a` | val doubles + masking work |
| `claude/ext-val-and-masks` | `.claude/worktrees/ext-val` | external validation, mask strategies |
| `feature/nonspecific-term` (`7266661` + OOD commit) | `.claude/worktrees/nonspecific-term` | ns term, frozen-WT, `--tf_mix`, `--val_wt_only`, `incl_*` fix, OOD metric exclusion (parent `b34093f`); queue plan written, not run — `docs/queue_takeover_ns.md` |

The trees, their bases, the running pin's patch list and the commit state at handoff are in **`docs/HANDOFF.md` §1, §4**
(the docs home is `repo/` on `devel-experimental`; the feature worktree's `docs/` is an older, different set).

**Rule: a queued series runs from a pinned snapshot of the code, not from a live worktree.** The worktrees
are shared and move mid-session. Examples: `/home/sareeves/playground/esm-msr-devel/dc_runs/pin_dc14_series`
(`e76fbbb`) and `pin_dc17_series` (`b34093f`), each with a `PINNED_FROM.txt`. A run whose log does not name
its snapshot cannot be reproduced.

The **canonical recipe** is not repeated here — it is the argument list of
`/home/sareeves/playground/esm-msr-devel/dc_runs/launch_dc17_s1.sh`, whose header explains every choice.
The earlier canonical command (still what `scripts/run_arm.sh` embeds) is in
`docs/epistasis_training_handoff.md` §2.

## 3. Where things live

| what | where |
|---|---|
| training code the dc queue runs | `dc_runs/pin_dc17_series/src/` (a snapshot, not a branch) |
| queue file (edit freely) | `dc_runs/claude_queue.txt`; runner `dc_runs/claude_queue.sh` |
| queue history (append every start/stop/change) | `/home/sareeves/playground/esm-msr-devel/queue_history.txt` |
| per-arm launch scripts (the lab notebook) | `dc_runs/launch_*.sh` — header carries hypothesis, evidence, and what changed from the previous arm |
| run logs | `dc_runs/<arm>.log`, `run_logs/<arm>.log` (`DONE rc=0` = clean) |
| metrics, checkpoints, val dumps | `dc_runs/logs/<arm>/0/` and `training_logs/<arm>/0/`: `metrics.csv`, `last.ckpt`, `val_dump_e{0..N}.npz` |
| gradient-balance constants | `/home/sareeves/playground/esm-msr-devel/probes/reg_balance_oct08.json` |
| cached preprocessed data | `/home/sareeves/playground/esm-msr-devel/cache_devel_conservative` (`--cache_version c5`) |
| split file | `repo/data/splits_oct06_capped.pkl` |
| Comet | project `esm-msr-agent-oct06-capped` (key read from `~/.comet.config` by `run_arm.sh`) |
| offline rescoring of any dump | `scripts/epi_from_dump.py` (also reads the released-code dump format) |
| interpreter | `/home/sareeves/miniconda3/envs/msr_venv/bin/python` (has **pytest**, installed 2026-10-18) |
| tests | `PYTHONPATH=src python -m pytest tests -q` — currently **288 passed, 1 skipped**, CPU only, ~17 s. (The parent `b34093f` is 234 passed, 1 skipped; the suite is pytest-style, so `unittest discover` is no longer the right entry point.) |
| GPU | one RTX 5090, 32.6 GB. The Windows display holds ~5 GB when idle; a training run takes 22-25 GB **reserved**. One job at a time. |

## 4. Running now / queued (2026-10-19 12:00)

Runner `dc_runs/queue_ns.sh` -> `dc_runs/queue_ns.txt`, which holds `wtsweep` and `gate1_tfmix`; the gates append
everything after that. Full orientation, incl. the trees and commit state: **`docs/HANDOFF.md`**.

* **`wtsweep` is running, stage `floor0`, arm `wt_floor0_drop02b`** - WT rank 16 / alpha 32 / dropout 0.2 with the zero
  knee (`--assay_floor 0.0 --link_floor_min 0.0 --link_floor_max 0.001`), matched against `wt_drop02` (same stack, data
  knee, 0.8447). It self-stops after three plateau decays (~epoch 5).
  Completed stages: floor (`wt_floor_data` 0.8391; `wt_floor_zero` died at construction - this pin rejects a 0/0 band),
  capacity (`wt_r16a32` 0.8435 beat rank 2's 0.8391 by 0.0044; rank 8 tied it), dropout (`wt_drop02` 0.8447, +0.0010).
  The base-config knee pair showed no cost: `wt_floor_data` 0.8391 vs `wt_floor0_base` 0.8403 (EXPERIMENTS §6b).
* **Next, automatically:** the `floor0` hook `floor_decide.py` adopts the zero knee by parsimony (seed-1 pair only;
  `--retest` re-enables a second seed) and rewrites the knee lines of `wt_stack.txt`; then **`capacity2` runs
  `wt_r16a16` = floor 0 / r16 / alpha16 / dropout 0.2 under the new metric definition - the new WT gold standard**.
  `wt_best.txt` is `0.0` so that run becomes the standing best rather than being compared to an old-definition 0.8447.
  Then `lr` -> `balance` -> `reach`.
* **Metric basis changed 2026-10-19** (§5.1): the OOD sets (`ptmuld`, `s461`, `ssym`) are out of every aggregate and
  summarised as `val_*_combined_external`; aggregates before and after that date are not comparable. Every run from
  `wt_r16a16` onward uses the new basis.
* **After the sweep:** `gate1_tfmix` generates the frozen-WT/tf-mix block from the winner (1 knee A/B arm + 5
  `--tf_mix` points) and appends `gate2_ns`, which picks the mix (`mix_pick.py`, margin 0.03 on `beyond_rho_mt`, else
  keeping 1.0) and adds `--nonspecific_term fx` at it with two seeds plus a seed-2 control. `make_fwt_block.py` reads
  the winner's weights, LoRA shape **and** link/censoring semantics (including `--external_val_sets`) from its
  checkpoint, so a stack cannot be mistyped or silently lost at the gate.
* **Deferred, not queued:** the from-scratch ns arms (`ctrl_s2`, `ns1_s1/s2`, `ns1_coarse_s1`, pinned at
  `pin_ns_series` @ `7db75a7`). `claude_queue.txt` still lists `wtsweep` and `dc19`-`dc23`; they predate the frozen-WT
  paradigm and both sweeps and must be re-stated before that runner is ever restarted.
* **Frozen-WT paradigm** (implemented: code `cf2b450`-`71d21fe`, pin `pin_fwt_series`): every future epistasis run is
  meant to freeze a chosen WT head and train only the MT path on doubles (`docs/frozen_wt_paradigm.md`).

## 5. How to read results

* **Target**: `val_epi_cell_rank_mt` (with `cell_mag_comb`, `cell_flipacc_mt`). Judge on the **mean of the
  last 2-3 epochs**, never the best epoch — epoch-to-epoch jitter is about the replicate SD.
* **Cell score** (for ranking arms): mean of `cell_rank_mt/0.0047`, `cell_mag_comb/0.0071`,
  `cell_flipacc_mt/0.0103`, averaged over the last 2-3 epochs.
* **Noise (measured, replaces the older estimates).** From the two triplicates, at epoch 7 across seeds:
  `cell_rank_mt` SD **0.007-0.009**, `cell_mag_comb` 0.002-0.008, `cell_flipacc_mt` **0.024-0.030**,
  `beyond_rho_mt` 0.008-0.018. Consequence: a single-seed difference needs roughly ≥0.02 on
  `cell_rank_mt` (≈2.8 SD) or ≥0.05 on flipacc before it is worth a sentence, and **two seeds are the
  minimum for any claim**. The Oct-8 handoff's 0.005 / 0.007 / 0.010 were optimistic.
* **Guards**: `val_rmse_combined_avg` and `val_rho_wt_valid_avg` must not regress; `val_epi_beyond_rho_comb`
  should not fall when the cell level rises.
* **Attribution**: `cell_rank_mt - cell_rank_ctx` = what the MT adapter adds over the WT adapter read in
  context (≈+0.06 in the modern arms).
* **WT-only runs (`--val_wt_only`, the WT sweep)**: the MT adapter is untrained there, so validation runs the WT
  forward only plus the WT-rev cycle pass (2 forwards per batch instead of 4). What is then readable, measured on
  `wt_floor_data/0/metrics.csv`:
  * **`ctx` is the real diagnostic** (`val_epi_cell_rank_ctx`, `val_epi_beyond_rho_ctx`) — it is built from `wt_rev`
    and does not need an MT head;
  * the raw levels (`val_epi_naive_rho_*`) and the `val_rho_*` scores for `mt`/`comb` mirror the WT head, because
    `mt` and `comb` are set equal to `wt` in this mode;
  * the **symmetrised** `val_epi_*_mt` and `val_epi_*_comb` levels read **0.0000**: they are built from the MT-reverse
    pass, which does not exist here, so 0.0 is the "no data" value and not a measurement. Do not read those zeros as
    a score, and never compare a `comb` column from a WT-only run with one from an ensemble run.
  * (If those levels are wanted in a WT-only run later, the honest construction is `mt_fwd := wt_rev`, which makes
    `comb` identical to `ctx` — the same statement the raw columns already make.)
* **Do not optimise**: `val_epi_naive_rho_*` (saturation leaks into them — the `fx` term moves them +0.21),
  `val_epi_pair_rho_comb` (24 pairs only), `val_epi_cell_sigsd_mt`, and anything named
  `val_auc_hyper_*` (≈2-25 items; see `EXPERIMENTS.md` §4).
* **Compare within a constant set and a code snapshot.** Arms from different `--reg_balance` constants or
  different pinned snapshots are not comparable; residuals are in `EXPERIMENTS.md`.
* **Old runs log old metric names.** Rescore any dump with the current definitions:
  `PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e5.npz [--boot 200] [--csv out.csv]`.
  The released baseline's cell-level numbers were produced this way (`training_logs/bl_w2a4_m16a16_capped/0/rescored_oct13.csv`).

### 5.1 Metric definition change, 2026-10-19: OOD sets are out of every aggregate

The three external benchmark libraries (`ptmuld`, `s461`, `ssym`) are **out of distribution** and are no longer
inside any validation aggregate. They are still scored and logged per loader (`val_<metric>/<name>`) for
diagnostics, and summarised on their own as:

* `val_rho_combined_external` — mean `rho_combined` over those loaders
* `val_rmse_combined_external` — mean `rmse_combined` over those loaders

Everything else — `val_*_avg`, the pooled global ranking (`val_rho_ungrouped_*`), the NDCG@k aggregates and the
whole `val_epi_*` hierarchy — now covers in-distribution libraries only. The list is a flag,
`--external_val_sets` (default `ptmuld,s461,ssym`), so it travels with each run and with the frozen-WT blocks that
`make_fwt_block.py` generates; `make_fwt_block.py` now carries it in its semantics.

**Consequence: aggregates are not comparable across the boundary.** A run before 2026-10-19 has the OOD sets inside
its `val_*_avg`; a run after does not. When an old run must be compared, recompute from the per-library columns
(`val_<metric>/<library>`), all of which are still written.

**The new WT gold standard** is the run that establishes the new basis: floor 0, WT rank 16 / alpha 16, dropout 0.2
(arm `wt_r16a16` in the rescheduled sweep). Future WT comparisons are against it.

## 6. Operating the queue

* **Change upcoming arms**: edit the queue file atomically (write a temp file, `mv` it). Never edit a bash
  script that is running: bash re-reads it while executing.
* **Stop after the current arm**: `touch` the queue's `.stop` file. To stop now: runner first, then
  `run_with_retry.sh` (or it retries), then `run_arm.sh`/`launch_*.sh`, then python (kill the dataloader
  workers if they linger).
* **Start a queue**: `setsid nohup <runner>.sh <queue>.txt [WAIT_PID] > run_logs/<queue>.out 2>&1 < /dev/null &`.
  The `[1] Done` line is the setsid launcher, not the runner — find the real pid with `ps`. Log it in
  `queue_history.txt`.
* **Retries**: `run_with_retry.sh` retries twice and **deletes `training_logs/<NAME>` first** — never reuse
  the name of a run you want to keep. Add the `_invalid_` / `_aborted_` / `_confounded_` prefix when retiring
  a log; that is this project's triage convention.
* **GPU-busy guards** wait while >4-12 GB is in use (the display alone holds ~5 GB). They sample `pgrep -x
  python` and read `/proc/<pid>/cmdline`; a killed-but-unreaped python has an **empty cmdline** and is
  invisible to them, which is how one arm once started on top of another (dc16 → dc17, §6 of EXPERIMENTS).

## 7. Pitfalls (each has cost time already)

1. **Autocast + no_grad.** Anything that runs a forward under `torch.no_grad()` inside a training step must
   call `torch.clear_autocast_cache()` afterwards, or the step's later forwards lose their gradient
   (`--pack_pair_matrices` gave the MT adapter *no* gradient this way; fixed in devel `9d629d1`). CPU stub
   tests without autocast do not catch it — test under CPU autocast.
2. **`--pack_pair_matrices`** needs `--batch_size` ≥ 361 and a micro-batch ≤ ~64; a batch of 400 drops
   ~11.6% of the items per epoch (the per-library remainder).
3. **Resumed runs split their log.** `metrics.csv` holds only the post-resume rows (the earlier ones are in
   `metrics_backup_ep<N>.csv`), and the plateau counters restart. Epoch-matched comparisons need both.
4. **`SIGTERM` exits rc=0.** Killing a training process looks like a clean finish to the queue; stop the
   queue first.
5. **WSL reports VRAM exhaustion as `CUDA driver error: device not ready`**, not as an OOM; check `dmesg`
   and the VRAM sample log before hunting for a code bug.
6. **Cache version.** `--cache_version` must match the directory (`c5` for the dc series); a mismatch
   silently regenerates the cache (CPU, minutes). The first epoch's log must say "Loading cached data".
7. **`--incl_*` and `--subset_caps` are one mechanism, in both directions.** A subset capped to 0 is never loaded,
   and an explicitly passed `--no-incl_doubles` / `--no-incl_singles` now forces that cap to 0.0 (before
   2026-10-18 it was parsed and then silently discarded, leaving the subset in). The WT sweep drops doubles by
   `--subset_caps single=None double=0.0`; the frozen-WT arms by `--frozen_wt_mt_only`, which sets both the flag
   and the cap. Never express "no doubles" with only one of the two spellings in a launcher you intend to reuse.
8. **Validation never sees training-side flags** (`--censor_floor`, `--min_additive_dG`,
   `--subfloor_rank_only`, `--flip_pair_groups`). `--include_out_of_range` does add censored items, but every
   metric is computed on uncensored items only.
9. **Inference ignores the link** (`link_head.*` keys): loading a link-trained checkpoint returns the latent
   ddG. It also ignores `ns_head.*` (the nonspecific term), so such a checkpoint must not be shipped until
   the inference plumbing is extended.
10. **Do not change without reading the design doc**: `--subset_caps` must leave one family `None`;
   `--incl_reversions` stays off; `--mask_structure` must match between training and inference;
   `--mask_strategy` (sequence masking) stays off; do not reintroduce density capping or target-conditioned
   resampling.
11. **`--lambda_reg_mt` must be > 0** for `--mt_single_anchor_weight > 0`; `--nonspecific_term fx`
    requires `--link`.
12. **Restarting `launch_wtsweep.sh` used to wipe the stack it was resuming.** It truncated `wt_stack.txt` and
    deleted `wt_best.txt` at start, which is right for a fresh sweep and wrong for a *rescheduled* one: on
    2026-10-19 that silently dropped rank16/α32/dropout 0.2 before the floor0 arm, which then ran at the base
    settings (rank 2, α4, dropout 0.1) and produced a 0.0044 "knee deficit" that was really the missing capacity
    stack — and, because `wt_best.txt` was gone, `wt_pick` saw no standing best and auto-stacked the one-arm
    stage. It now initialises from `wt_stack_seed.txt` / `wt_best_seed.txt` when present, so a rescheduled sweep
    inherits what the finished stages won. **After any restart, diff the running arm's `hparams.yaml` against the
    arm it is supposed to be matched to** — that check is what caught this.

## 8. Design facts a reader would otherwise have to rediscover

* **Censoring** is one mechanism (`src/esm_msr/censoring.py`): per item `cens` (−1 / 0 / +1), `cens_src`
  (1 assay range, 2 floor), `cens_bound` on the item's own ddG scale. Lower-censored items tie at the bottom
  of a list in forward Plackett-Luce, upper-censored at the top in reverse Plackett-Luce, both-sided lists
  average the two. Only plain singles censor the WT head; a censored single never supplies a double's
  additive expectation.
* **The link** is `h(z) = hi − tau_hi·softplus((hi−u)/tau_hi)` with `u = lo + tau_lo·softplus((z−lo)/tau_lo)`;
  when a temperature is not small against the span the plateaus differ from `lo`/`hi` (logged as
  `link/floor`, `link/ceiling`). A conditional item is scored as its double: `h(dG_wt + bg_offset + latent) − dG_wt`.
* **The interaction loss** is a function of total predictions only (double-centring), so no second output
  head exists; identity-independent effects are still learned through the ordinary regression.
* **Per-library floors are data**, not parameters: only ~20-47 of 198 libraries yield a floor estimate, most
  print `NO floor evidence`. `--learn_global_floor` was found to drift to −0.37/−0.50 and is pinned instead.
* **The WT head trains on singles only** (`--no-wt_trains_doubles` is the default since `f21c913`), which is
  what makes the WT-only sweep in `claude_queue.txt` affordable (≈1/3 of the cost).
