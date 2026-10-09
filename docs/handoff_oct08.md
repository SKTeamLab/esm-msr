# Handoff for a run-managing agent (2026-10-08)

Read this first. Background: `docs/summary_oct08.md` (what changed and why), `docs/validation_metrics.md` (every metric), the report
`docs/epistasis_metrics_report.html`. Facts below are dated 2026-10-08 unless marked.

## 1. Where things are

| what | where |
|---|---|
| code the queue runs | worktree `/home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/epistasis-metrics-logging-cfd318`, branch `claude/epistasis-metrics-logging-cfd318` (== `devel` at handoff) |
| queue file (edit freely) | `scripts/queue_r2.txt` in that worktree |
| queue history (every start / end / change, append to it) | `/home/sareeves/playground/esm-msr-devel/queue_history.txt` |
| run logs | `/home/sareeves/playground/esm-msr-devel/run_logs/<NAME>.log` (ends with `DONE rc=0` when finished cleanly) |
| metrics, checkpoints, validation dumps | `/home/sareeves/playground/esm-msr-devel/training_logs/<NAME>/0/` (`metrics.csv`, `last.ckpt`, `val_dump_<zs|eN>.npz`) |
| gradient-balance constants | `/home/sareeves/playground/esm-msr-devel/probes/reg_balance_oct08.json` (probe files and logs beside it) |
| Comet | project `esm-msr-agent-oct06-capped` (key read from `~/.comet.config` by `run_arm.sh`) |
| interpreter | `/home/sareeves/miniconda3/envs/msr_venv/bin/python`; tests: `PYTHONPATH=src python -m unittest discover -s tests` (161, CPU, seconds) |
| GPU | one RTX 5090 (32 GB). The Windows display holds about 5.3 GB on an idle card; a training run takes 20-25 GB, so only one job at a time |

## 2. What is running at handoff

`scripts/probe_then_queue.sh` (pid in `queue_history.txt`): waits for o6q01_mt16's resume, runs three gradient-share probes (about 30-60 min),
writes the constants file, then `exec`s `scripts/queue_runner.sh scripts/queue_r2.txt`. The runner takes the first uncommented line of the
queue file before every arm, comments it out (`# started <time>: ...`) and runs it with `scripts/run_with_retry.sh` (or `resume_arm.sh` for a
`resume ...` line). 18 arms of 6 epochs, about 3.5 h each (packed arms somewhat longer).

Check on it: `tail queue_history.txt`; `ps -eo pid,cmd | grep -E "queue_runner|probe_then_queue|training.py --experiment_name"`;
`tail -c 300 run_logs/<NAME>.log`; `nvidia-smi`.

**Update (2026-10-08, 23:45).** The first packed probe recorded no gradient for the MT rank loss. Cause: under bf16 autocast, the no-grad cache
pass of `--pack_pair_matrices` left cached weight casts without autograd history, so packed training gave the MT adapter no gradient at all
(fixed in devel `9d629d1`, regression test under CPU autocast). No packed arm had run. `scripts/reprobe_packed_then_queue.sh` now replaces the
runner: after o6s01_ref it re-runs the packed probe, rewrites `probes/reg_balance_oct08.json` (plain constants unchanged; `*_packed` added;
the previous file is kept as `reg_balance_oct08_before_packed_fix.json`) and continues `queue_r2.txt` with o6s02. Constants measured on 2026-10-08
(plain, geometric mean of the rank-2 and rank-16 probes): reg_wt 28.6, reg_mt 76.5, comp_off 55.3, comp_subst 111.7, comp_int 742.3,
flip_mt 9.1 (the rank-2 and rank-16 probes agree within about 1.3x, flip_mt 14.4 vs 5.8). Check the packed values in `queue_history.txt`.

## 3. Operating the queue

* **Change upcoming arms**: edit `scripts/queue_r2.txt` (atomically: write a temp file and `mv`/`os.replace` it). Never edit a bash script
  that is running (`queue_runner.sh`, `probe_then_queue.sh`, `run_*.sh`): bash re-reads scripts while running them.
* **Stop after the current arm**: `touch scripts/queue_r2.txt.stop`. To stop now: kill the runner first, then the arm's `run_with_retry.sh`
  (or it retries), then `run_arm.sh`, then the python processes (`kill -9` the dataloader workers if they linger).
* **Start a queue**: `setsid nohup scripts/queue_runner.sh scripts/<queue>.txt [WAIT_PID] > run_logs/<queue>.out 2>&1 < /dev/null &`, then find
  the runner's pid with `ps` (the `[1] Done` line is the setsid launcher exiting, not the queue). Log it in `queue_history.txt`.
* **Line format**: `NAME EPOCHS SEED [flags]` (flags are appended to `run_arm.sh`'s canonical command and override it), or
  `resume NAME EPOCHS SEED COMET_KEY [the run's flags]` to continue `training_logs/NAME/0/last.ckpt` to EPOCHS epochs in total (the Comet key
  is in the run log's comet.com URL; earlier rows are kept in `metrics_backup_e<N>.csv`, the log in `run_logs/<NAME>_to_e<N>.log`).
* **Crashes**: `run_with_retry.sh` retries the same configuration twice (intermittent WSL GPU errors) and DELETES `training_logs/<NAME>` before
  each attempt; never reuse the name of a run you want to keep.

## 4. Reading results

* **The target is the cell level**: `val_epi_cell_rank_mt`, confirmed by `val_epi_cell_mag_comb` and `val_epi_cell_flipacc_mt`. For ranking
  arms use the **cell score** = mean(cell_rank_mt / 0.0047, cell_mag_comb / 0.0071, cell_flipacc_mt / 0.0103) averaged over the last 2-3 epochs.
* **Noise**: replicate SD (same configuration, different seed or rerun) about 0.005 (cell_rank_mt), 0.007 (cell_mag), 0.010 (flipacc); a
  single-seed difference needs about 2.8 x that to be believed (0.014 on cell_rank_mt). r2 has seed-2 replicates of the reference, MT rank 16,
  master 1 and pack+flip: use them to re-estimate the noise under the new constants.
* **Guards**: `val_rmse_combined_avg` and `val_rho_wt_valid_avg` must not regress; `val_epi_beyond_rho_comb` should not fall when the cell level rises.
* **Attribution**: `cell_rank_mt - cell_rank_ctx` is what the MT adapter adds over the WT adapter read in context.
* **Do not optimise** `val_epi_naive_rho_*` (mostly saturation under the link), `val_epi_pair_rho_comb` (24 pairs), `val_epi_cell_sigsd_mt`.
* **Compare within a constant set.** r2 arms use the re-measured `--reg_balance` constants (stored in each run's `hparams.yaml` as
  `reg_balance_values`); the o6g / o6q / o6h runs used the old ones, so weight-dependent differences between the two sets are not comparable.
  o6s01_ref is the r2 reference.
* **Old runs log old metric names.** Rescore any dump with the current definitions:
  `PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e5.npz [--boot 200] [--csv out.csv]`.

## 5. Decisions the user expects (from the r2 design)

1. MT capacity: o6s02_mt16 vs o6s01_ref (and o6s14 seed 2); o6s09_mt8 separates capacity from adapter scale; o6s16_mt32 tests beyond 16.
2. MT regression weight: o6s03_m1, o6s06_m3 vs the reference.
3. Whole-matrix losses: o6s04_pack (components on whole matrices) and o6s05_pack_flip1 / o6s12_pack_flip3 (+ the reversal loss); watch
   `train/flip_rev` (about 76 per batch with a matrix) and `train/flip_rev_acc` (should rise above its starting value).
4. Floor: o6s10_floor03 (`global_bias_comb` toward 0, cell level unchanged or better).
5. Colrank: o6s11_colrank0.
6. Combinations: o6s07, o6s08, o6s15.
Report per arm: the cell score and its three parts (mean of the last 3 epochs), the guards, `cell_rank_ctx`, `global_bias_comb`, wall time per
epoch, peak memory (`mem/peak_allocated_gb`), any crash with 20 log lines around it. Do not change defaults in code without the user.

## 6. Traps

* A hook forbids writing files in other worktrees from a session; work inside this worktree.
* Anything that runs a forward under `torch.no_grad()` inside a training step must call `torch.clear_autocast_cache()` afterwards, or the
  step's later forwards lose their gradient (see the update in section 2). CPU stub tests without autocast do not catch it.
* `scripts/run_arm.sh` runs the code of the worktree it lives in (set `WT=` to override).
* `--pack_pair_matrices` needs `--batch_size` >= 361 (400 in r2); a batch of 400 drops 11.6% of the items each epoch (the per-library remainder).
* `grad_share_probe.py --extra` takes a value that starts with `--` only as `--extra="--flag ..."`.
* The queue's busy-card guard waits while more than 12 GB is in use (the display alone holds about 5.3 GB).
* Resumed runs: the learning-rate plateau counters restart; `metrics.csv` holds only the resumed rows.
* Epoch-to-epoch jitter of the cell metrics is about the replicate SD; judge on the mean of the last epochs, not the best epoch.
