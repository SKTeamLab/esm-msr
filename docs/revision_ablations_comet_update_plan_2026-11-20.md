# Comet project `revision-ablations`: link correction and the pooled correlations (plan, 2026-11-20)

Two changes to what validation logs, and how every existing experiment of the project gets them.

## 1. What changed

**Pooled correlations (new, `esm_msr.pooled_stats`).** `val_[unmasked_]rho_combined_pooled` and `val_[unmasked_]rho_singles_wt_pooled`: one Spearman over every
in-distribution validation item at once (the canonical population, the external sets never in it), next to the existing per-library means `val_rho_combined_avg` and
`val_rho_singles_wt_avg`. Logged natively by every run that starts on code with `src/esm_msr/pooled_stats.py` (this change) on; older runs get them offline (below). The same function does both.

**Link rule (pushed 8bd32d2..379f7cc).** The Tobit expected-reading link only touches the observed-scale statistics (every RMSE and `epi_naive_rho_comb`). It is applied to a
run trained with the Tobit likelihood, with its own constants, on the in-distribution libraries; never to an external set; never to a run trained with MSE
(`--reg_nll none`). Before that, the three MSE runs `FINAL_p0_mse_s{1,2,3}` were read through it. No other run is affected (a Tobit run's link was right; the external sets
carry no dG_wt, so they were never linked).

## 2. Tools (this commit)

| step | tool | what it does |
|---|---|---|
| rescore | `scripts/revision_ablations_rescore.py RUN_DIR... --out DIR --reg-nll none\|tobit [--link auto\|summary\|none] [--check]` | offline, from `val_dump_*.npz` + `metrics.csv` (never `hparams.yaml`: it holds the Comet key). Recomputes the per-library suite, `rmse_*_avg`, `epi_naive_rho_comb` and the pooled series per epoch and regime; `--check` proves the run's own link reproduces the logged values; writes `<experiment>.json` with `corrected` and `new` points |
| inventory | `scripts/revision_ablations_comet.py list` | read-only: experiment, key, running / idle, which already have pooled series |
| pooled series | `... append --points DIR [--execute]` | adds the new series in place (`ExistingExperiment`); sends once, polls with back-off (uncached API), only sends what is absent |
| correction | `... correct --points DIR [--execute] [--target-project P]` | rebuilds an experiment (same name, parameters, tags + `link_corrected_2026-11-20`, code assets, every point at its original step/epoch except the corrected ones + pooled series), verifies it value by value, then RETIRES the original (`SUPERSEDED_<name>`, tags `superseded`, `link_misapplied`). Never deletes unless `--delete-original` |

Dry run is the default for both Comet commands. Checked so far (no uploads): reproduction OK (max |diff| <= 2e-4 over 2,800-3,300 re-derived values per run) on the three MSE runs and on a
Tobit control (`FINAL_p0_pw0_s2`: 0 corrections, 36 pooled points); dry runs against the real project resolve `FINAL_p0_mse_s1-0` and `-s3-0` (points 8,936 -> 8,988 and
8,307 -> 8,355, steps aligned) and `FINAL_p0_pw0_s2-0` (36 pooled points to send).

## 3. Order of work

1. **Correct the MSE runs.** `FINAL_p0_mse_s1` and `_s3` are finished. `_s2` was still running when checked (Comet flag RUNNING, newest metric 6 minutes old; cluster job 390192):
   wait for it, re-fetch its dumps, then rescore. Rehearse on ONE run first with `--target-project` (a scratch project; the original stays untouched), look at the copy, then run
   for real. The copy takes minutes to show in Comet; rerun the same command to resume checking, never re-send by hand.
2. **Append the pooled series to every other experiment.** Same command, all names. Idempotent: rerunning only sends what is still absent.
3. **Runs that finish later** (deepfloor, floor0, norange_s3, ...) either log the pooled series natively (code after this commit) or are appended once they end. The pins of
   the already-started jobs predate the pooled code, so those get it by `append`.
4. **Verify** with `list` (every experiment shows the pooled series; `link-sensitive` counts unchanged), and by re-reading each touched experiment with the uncached client.

## 4. Inventory: where the dumps are (47 experiments, Comet names carry `-0`)

| experiments | dumps | notes |
|---|---|---|
| `FINAL_p0_s{1-3}`, `FINAL_p1_s{1-3}`, `FINAL_p1_marg_s{1-3}`, `FINAL_p0_last4_s{1-3}`, `FINAL_p0_subcal_s1`, `FINAL_p0_norange_s1`, `FINAL_p0_mse_s{1-3}`, (`FINAL_p0_deepfloor_*`, `FINAL_p0_floor0_*` when finished) | subhak-gpu `~/esm-msr-devel/runs_final_ablations/<arm>/0/` (10-18 dumps each) | fetch only `val_dump_*.npz` + `metrics.csv` |
| `FINAL_p0_mega_s{1-3}` | subhak-gpu `~/esm-msr-devel/outputs/mega_retrain/esm_msr/FINAL_p0_mega_s*/0/` (11-12) | |
| `FINAL_p0_pw0_s2`, `FINAL_p0_norange_s2` | local `cluster_arbiter/local_runs/<arm>/0/` (9, 14) | norange_s2 is the interrupted run |
| `FINAL_p0_norange_s3` | local `cluster_arbiter/local_runs_training/` while training, then `local_runs/` | |
| `FINAL_p0_pw0_s1`, `_s3`, `FINAL_p0_norank_s{1-3}`, `_r2a4_s{1-3}`, `_rcthr0_s{1-3}`, `_noseqhwm_s{1-3}` (13 experiments) | fir `/scratch/sareeves/esm-msr-devel/runs_revision_ablations/<arm>/0/` | needs the Duo login; the only blocker for "everything" |
| `bl_w2a4_m16a16_capped_s{1-3}` | `training_logs/bl_w2a4_m16a16_capped_s*/0/` (local, kept) | main-baseline code, flat dump format (no `subset_type` / `cens_src` / `dG_wt`: the script derives them); use `--link summary` |
| `bl_w2a4_m16a16_capped_rank05_reg15_s{1-3}` | subhak-gpu `~/esm-msr-devel/sweep_baseline_nov07/runs/<arm>/0/` (9 each) | same, `--link summary` |
| `tfl_T_p0_ctl_s1` | subhak-gpu `~/esm-msr-devel/sweep_block_nov02/v12/runs/tfl_T_p0_ctl_s1/0/` | |

Everything except the 13 fir experiments can be done without a login. Per-run commands (example for the three MSE runs once `_s2` ends):

    PYTHONPATH=src python scripts/revision_ablations_rescore.py final_rescore_nov20/FINAL_p0_mse_s{1,2,3} --reg-nll none --out final_rescore_nov20/points
    PYTHONPATH=src python scripts/revision_ablations_comet.py correct --points final_rescore_nov20/points --target-project <scratch> --names FINAL_p0_mse_s1   # rehearsal
    PYTHONPATH=src python scripts/revision_ablations_comet.py correct --points final_rescore_nov20/points --execute

and for a Tobit run `--reg-nll tobit` (no corrections, only pooled points), then `append --execute`.

## 5. Risks and what the tools do about them

* **Comet cannot overwrite a point or remove one**: corrections are copies, originals are retired, so a mistake costs a copy, not a run.
* **Stale reads / late ingestion** (two earlier traps): uncached client; each point sent once; absent ones re-sent only after polling.
* **A run whose flag says running**: refused (also when its newest metric is under 90 minutes old); `--ignore-running-flag` only after checking its job is over.
* **Alignment**: pooled points are placed at the steps of the validations Comet already holds and a points file that does not line up is refused.
* **Not touched**: the external-set statistics (never linked), the rank metrics, the training metrics, and any `sys.*` series (they cannot be written to a copy; the retired original keeps them).
