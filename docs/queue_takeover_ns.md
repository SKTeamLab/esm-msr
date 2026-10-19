# Queue takeover plan: test `--nonspecific_term fx` first

**Status: EXECUTED 2026-10-18 21:07 EDT.** The idle runner was retired and `queue_ns.sh` (pid 23863) started
waiting on `dc18` (pid 1803, untouched); the set `ctrl_s2, ns1_s1, ns1_s2, ns1_coarse_s1` runs from
`pin_ns_series` @ `7db75a7`. Everything below is kept as the record and as the procedure for replaying or
undoing it. The only deviation from the plan as written: `claude_queue.sh` was killed while waiting (no arm in
flight), which is the safe window §2 identified.

Written 2026-10-18 21:00 EDT.

## 1. Why preempt the queue

`wtsweep` (eleven 1-1.5 h WT-only arms) is the next thing `claude_queue.txt` will do. The nonspecific term
is the project's best single-seed cell-level result (`docs/nonspecific_epistasis.md`: `beyond_rho_mt`
+0.059 over an otherwise identical control, and it beats every arm of its series), it is now ported to the
line the queue actually runs, and confirming or killing it is worth ~24 h of GPU more than starting an
eleven-arm sweep whose winner would change the WT basis *again* before the term has been tested on the
current one. Test the term first, then sweep.

This is a **queue takeover, not a run takeover**: `dc18` is left alone to finish. Only the runner is
replaced.

## 2. Where things stand at the moment of writing

| | |
|---|---|
| training now | `dc18_floor0_w2a4_m16a16_s1`, pid **1803**, epoch 3 of a 10 cap (will self-stop when both heads freeze, ~epoch 6-8) |
| runner now | `dc_runs/claude_queue.sh`, pid **4103**, waiting on pid 1803; its first arm is `wtsweep` |
| queue file | `dc_runs/claude_queue.txt` — `wtsweep`, `dc19_s1`, `dc20_extval_score_dc17_s1`, `dc21_maskladder_s1`, `dc22_dropout0_s1`, `dc23_mt8_s1` |
| GPU | one card, 32.6 GB, one job at a time (~25 GB reserved per run) |
| disk | 401 GB free; ~1.3 GB per run |

The runner is a loop over `claude_queue.txt`: it re-reads the file before each arm, skips a name already
recorded as `finished` in `claude_queue.log`, gates on `gpu_busy()`, and waits out `HANDOVER_PID` (1803)
first. It has **no stop file** — killing it is the only way to replace it, and that is safe *right now*
because it is idle between arms.

## 3. The code to run

* Branch **`feature/nonspecific-term`**, commit **`7db75a7`**, worktree
  `.claude/worktrees/nonspecific-term` (parent `b34093f` = the dc17/dc18/dc19 basis).
* `--nonspecific_term fx` (+ `--ns_knot_lo/-hi/-step`, `--ns_lr`), implemented as in the archived
  `70be32e`; see `docs/nonspecific_epistasis.md` §2 and the commit message for the two adaptations
  (`--per_mutation_bias` accounting in x; no `--tf_singles` on this branch).
* Tests: `tests/test_nonspecific_term.py`, 14 tests. Full suite on the branch:
  `PYTHONPATH=src python -m pytest tests -q` → **248 passed, 1 skipped** (parent `b34093f`: 234 passed,
  1 skipped), ~17 s, CPU. `pytest` is now installed in `msr_venv`.
* **Pin it before queueing** (project convention — a shared worktree must not move under a run):

```bash
cp -a /home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/nonspecific-term/src \
      /home/sareeves/playground/esm-msr-devel/dc_runs/pin_ns_series/
cat > /home/sareeves/playground/esm-msr-devel/dc_runs/pin_ns_series/PINNED_FROM.txt <<'TXT'
Snapshot of feature/nonspecific-term 7db75a7 (clean tree), taken 2026-10-18.
Parent basis: b34093f (== pin_dc17_series). Sole change: --nonspecific_term fx (port of the archived
main-baseline 70be32e) + tests. With the flag off the code is behaviourally identical to pin_dc17_series.
TXT
```

## 4. The arms

Copy `dc_runs/launch_dc17_s1.sh` for each; change **only** the marked lines. Everything else stays
byte-identical to dc17 (same basis, `--per_mutation_bias`, `--no-wt_trains_doubles`, both plateaus at
patience 1 / factor 0.3 / 3 decays, the data-only link, `--censor_floor`/`--censor_gap`, batch 256 /
micro 64, `--val_cycle_passes --val_dump`, 10-epoch cap).

| arm (launch + queue name) | line to change vs `launch_dc17_s1.sh` | seed | why |
|---|---|---|---|
| `ctrl_s2` | `cd $D/pin_ns_series`; `--experiment_name ctrl_s2_w2a4_m16a16`; `--seed 2` | 2 | the matched control **on the new pin**; with dc17_s1 it makes dc17 a two-seed baseline |
| `ns1_s1` | same cd; `--experiment_name ns1_s1_w2a4_m16a16`; add `--nonspecific_term fx` | 1 | direct test against the existing dc17_s1 |
| `ns1_s2` | same cd; `--experiment_name ns1_s2_w2a4_m16a16`; add `--nonspecific_term fx`; `--seed 2` | 2 | the matched pair that decides: `ns1_s2` vs `ctrl_s2` |
| `ns1_coarse_s1` | same cd; `--experiment_name ns1_coarse_s1_w2a4_m16a16`; add `--nonspecific_term fx --ns_knot_step 2.0` | 1 | ablation: with 5 knots instead of 17, does the gain survive? Separates "a curve" from "a floor/ceiling offset" |

`ns1_coarse_s1` is the cheapest useful ablation and is deliberately last: skip it if the first three answer
the question.

Queue file `dc_runs/queue_ns.txt` (control first, so a mid-queue read has its reference, then alternating
seeds):

```
ctrl_s2
ns1_s1
ns1_s2
ns1_coarse_s1
```

## 5. Runner

`claude_queue.sh` hardcodes `LIST=claude_queue.txt`, so add a sibling rather than edit it (bash re-reads a
running script, and this one may still be alive at the moment of takeover). `dc_runs/queue_ns.sh` = a copy
of `claude_queue.sh` with three changes:

1. `LIST=${1:?usage: queue_ns.sh <queue-file> [wait-pid]}` and `LOG=$D/${1%.txt}.log`;
2. `HANDOVER_PID=${2:-}`;
3. a stop file: `[ -e "$LIST.stop" ] && { echo "$(date '+%F %T') stopped by $LIST.stop" >> $LOG; break; }`
   at the top of the `while true` loop (the missing stop file is the one operational gap in the current
   runner).

Keep its `gpu_busy()` as is — it already treats a killed-but-unreaped python (empty `/proc/<pid>/cmdline`)
as busy, which is what let dc17 start on top of dc16.

## 6. Execution sequence (for the operator — NOT DONE)

```bash
D=/home/sareeves/playground/esm-msr-devel/dc_runs
# 0. preconditions: dc18 still pid 1803; claude_queue.sh pid 4103 idle (waiting on it)
ps -o pid,etime,cmd -p 1803,4103 | cut -c1-80

# 1. retire the current runner (idle between arms: it is waiting on 1803)
kill 4103
echo "$(date '+%F %T') claude_queue.sh (pid 4103) retired by request; queue taken over by queue_ns.sh
  first: nonspecific-term test set (ctrl_s2, ns1_s1, ns1_s2, ns1_coarse_s1); wtsweep/dc19+ stay in
  claude_queue.txt and run after" >> /home/sareeves/playground/esm-msr-devel/queue_history.txt

# 2. create the pin, the four launch scripts, queue_ns.txt, queue_ns.sh (§3-§5), then
chmod +x $D/launch_ctrl_s2.sh $D/launch_ns1_s1.sh $D/launch_ns1_s2.sh $D/launch_ns1_coarse_s1.sh $D/queue_ns.sh

# 3. start it, waiting out dc18 so nothing is wasted and two jobs never share the card
cd $D
setsid nohup ./queue_ns.sh queue_ns.txt 1803 > $D/ns_queue.out 2>&1 < /dev/null &
ps -eo pid,cmd | grep -E "queue_ns" | grep -v grep          # the real pid (the [1] Done is the launcher)
echo "$(date '+%F %T') QUEUED ns set via queue_ns.sh (runner pid <PID>, waits on 1803): ctrl_s2, ns1_s1,
  ns1_s2, ns1_coarse_s1; pin_ns_series @ 7db75a7; ~5 h/arm; NOTE the ns arms may not self-stop early
  (the nonspecific group never decays), so budget the full 10-epoch cap" >> /home/sareeves/playground/esm-msr-devel/queue_history.txt

# 4. after the set finishes, hand the queue back
setsid nohup ./claude_queue.sh > $D/claude_queue.out 2>&1 < /dev/null &
```

`dc18` keeps running throughout step 1-3; `queue_ns.sh` blocks on pid 1803 before its first arm. Nothing is
killed mid-run and no epoch is lost.

## 7. Cost, displacement, and how to get the displaced work back

* ~5 h per arm (dc17: 4 h 58 for 9 epochs), so **~20 h for four**, plus whatever remains of dc18 (~2-4 h).
* Displaced: `wtsweep` (≈14 h) and everything behind it (`dc19` ≈5 h, `dc20` ≈15 min, `dc21`, `dc22`,
  `dc23`). They are untouched in `claude_queue.txt` and resume with a plain `claude_queue.sh` restart
  (§6 step 4) — the runner skips anything already logged as finished.
* Caveat on resuming `wtsweep`: it **truncates `wt_stack.txt`** at start (`: > $D/wt_stack.txt`). That is
  fine only if it has not yet run a stage. If it has, the stack must be saved first — check
  `dc_runs/wt_sweep.log` and `wt_stack.txt` before restarting it.

## 8. Reading protocol and the decision rule

Compare, all mean of the **last 3 epochs**, never the best epoch:

1. **`ns1_s2` vs `ctrl_s2`** (matched: same pin, same seed, one flag) — the primary comparison.
2. **`ns1_s1` vs `dc17_s1`** (same seed, flag off vs on, different pin; the pin difference is a no-op with
   the flag off) — the replication of the Oct-14 direction.
3. Metrics, in order of what the term is expected to move: `val_epi_beyond_rho_mt`, `val_epi_beyond_rho_comb`,
   `val_epi_cell_mag_comb`, then `val_epi_cell_rank_mt`, `val_epi_cell_flipacc_mt`.
4. Noise to beat (replicate SD from the triplicates, `STATE.md` §5): `beyond_rho_mt` 0.008-0.018,
   `cell_mag_comb` 0.002-0.008, `cell_rank_mt` 0.007-0.009, `flipacc_mt` 0.024-0.030. **Adopt only if
   `beyond_rho_mt` improves by ≥0.03 in both seeds with the guards flat**; anything smaller is not separable
   from run noise at this set size.
5. Guards must not regress: `val_rho_wt_valid_avg`, `val_rmse_combined_avg`, and `val_epi_beyond_rho_comb`
   must hold while the cell level rises.
6. **Shape check** (a replication, not a formality): the learned curve `ns/f_at_x*` should be monotone
   decreasing with x, ≈ +0.4…+0.5 near x = −1 and ≈ −0.15…−0.2 at x ≥ +2, as it was on 2026-10-14. A flat or
   wildly oscillating curve means it learned something other than the nonspecific shape; report it as such.
   `ns_term` is in every `val_dump_e*.npz` for offline inspection.
7. `val_epi_naive_rho_*` will rise and must not be read as an improvement (saturation leaks into it — that
   is the whole reason the term is interesting and the reason the metric is not a target).
8. Expect `val_epi_global_bias_comb` to stay ≈ −0.50; the Oct-14 run did not improve it either. If it does
   move, that is new information about where that residual lives.

Decision: **adopt** (make `fx` part of the canonical recipe) only with two seeds agreeing and the guards
intact; **iterate** (knots, freeze after epoch 2, with/without the floor pin) if the term helps but the curve
is unstable; **drop** if `beyond_rho_mt` is flat or the guards fall.

## 9. Risks and watch items

* **The term's optimizer group never decays** (by design: `lr/nonspecific` is constant at `--ns_lr 5e-3`).
  It also means `_stop_if_nothing_is_training()` may never fire while the term can still move, so an arm may
  run the full 10-epoch cap instead of self-stopping near epoch 8 — budget ~5.5 h per ns arm.
* **Late absorption**: if `beyond_rho_mt` peaks around epoch 4-5 and then declines while |f| grows, the term is
  fitting the objective rather than the physics. The `ns/f_at_x*` trace is the diagnostic (it is logged every
  step); `ns1_coarse_s1` is the structural check.
* Do not run two jobs: `queue_ns.sh`'s GPU gate is the only guard, and it samples every 60 s.
* Do not edit `queue_ns.sh` or any `launch_*.sh` while a run is going; edit `queue_ns.txt` atomically if the
  arm list must change and use `queue_ns.txt.stop` to stop between arms.
* Every ns arm writes `ns_head.*` into its checkpoints. `inference_scripts` still ignore it (`docs/STATE.md`
  §7.8), so **do not ship** these weights until that plumbing exists — the external-benchmark numbers would be
  scored without the term.

## 10. Rollback

* Wrong arm set: create `queue_ns.txt.stop`, let the current arm end, start `claude_queue.sh` again.
* Bad pin: delete `dc_runs/pin_ns_series` and rebuild from `7db75a7`; the branch is intact and no run
  depends on the live worktree.
* Nothing here touches `pin_dc17_series`, so dc17/dc18/dc19 remain reproducible and comparable.
