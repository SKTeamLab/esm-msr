# HANDOFF — read this first

Written 2026-10-19 for an agent with elevated (danger-full-access) permissions taking over mid-queue. It states where
the code and the documentation actually live, which tree runs what, what is running right now, what was learned in the
last two days, and the immediate next actions. Everything else is in `STATE.md` (facts, pitfalls, how to read
results) and `EXPERIMENTS.md` (measured history).

## 0. In one paragraph

ESM-MSR predicts stability changes for single and double mutants (ddG, and the epistasis of doubles) by adding LoRA
adapters to ESM3-small on top of pre-trained base weights. Two adapters exist: **WT** (trained on singles, rank 2/alpha
4 historically) and **MT** (rank 16/alpha 16, trained on doubles). The project's central rule is that gains must
**stack**: every winning configuration is appended to a stack file and every later run inherits it. The current campaign
first builds the best **WT** head (a 16-arm sweep, currently running), then freezes it and trains **only the MT head on
doubles** against it, sweeping how much of the measured additive label to pass as teacher forcing, and finally testing
the nonspecific-epistasis term (`--nonspecific_term fx`), which is the best single-seed result on record.

## 1. Where things live, and which tree runs

**Documentation lives in `repo/` on branch `devel-experimental`** (`/home/sareeves/playground/esm-msr-devel/repo/docs`).
It is NOT in the feature worktree: `.claude/worktrees/nonspecific-term/docs/` contains an older, different, three-file
set with no `STATE.md`. Read docs from `repo/docs`.

**The queue runs an unversioned snapshot, not any branch.** `/home/sareeves/playground/esm-msr-devel/dc_runs/pin_dc20_series`
is the only code that trains. It is a copy of `claude/ext-val-and-masks e075ed8`, plus **deliberate hand patches recorded
in `PINNED_FROM.txt`** (that file is authoritative — read it before touching the pin). The current patches are:

1. `--val_wt_only` (WT-only validation, 2026-10-18 23:20) — upstream commit `7266661`.
2. `_transition_to_mt()` respects `--freeze_mt_adapter` (23:35) — this is what lets WT-only arms early-stop.
3. OOD validation exclusion + `_external` metrics (2026-10-19 12:00) — upstream commit in the feature branch, see §4.

Trees:

| path | branch / state | what it is for |
|---|---|---|
| `repo/` | `devel-experimental` @ `006078b` (ahead 23 / behind 31 of origin) | **documentation home**; docs commits land here |
| `repo/.claude/worktrees/nonspecific-term` | `feature/nonspecific-term` @ `7266661` + one uncommitted-then-committed OOD patch | **code home**: ns term, frozen-WT, `--tf_mix`, `--val_wt_only`, `incl_*` fix, OOD exclusion. Based on `main` (`c34c315`), NOT on `devel-experimental` |
| `repo/.claude/worktrees/ext-val` | `claude/ext-val-and-masks` @ `e075ed8` | the pin's upstream source |
| `repo/.claude/worktrees/devel-main-conservative-merge-5e554f` | `devel-revision` @ `b34093f` | the dc17–dc19 code (dc17's results come from here) |
| `.claude/worktrees/{main-baseline,delink-above-tau,scale-mismatch-improvements-a44582,practical-snyder-52312a,branch-strategy-reorganization-5cfa71}` | assorted `claude/*` | unused by the current campaign |

> **Which tree is which, in one line each.** *Execution* = `dc_runs/pin_dc20_series`, a snapshot of the
> **`devel-revision`** line (`b34093f` + 3 commits) plus the hand patches in `PINNED_FROM.txt` — not
> `devel-experimental`. *Code* = the `feature/nonspecific-term` worktree, which is based on **`main` (`c34c315`)** —
> neither `devel-revision` nor `devel-experimental`. *Docs* = `repo/` on **`devel-experimental`**.
>
> **`devel-experimental` contains none of this campaign's code.** Verified at handoff: `--nonspecific_term`,
> `--frozen_wt_mt_only`, `--tf_mix`, `--val_wt_only` and `--external_val_sets` all return zero matches in
> `repo/src/esm_msr/config.py`, and the pin has only those flags its patches added. So branching or merging from
> `devel-experimental` gets the documentation with none of the features; the feature branch is the only tree with both
> the code and the tests, and it does not carry `devel-experimental`'s (longer) history. Reconciling those two lines is
> a deliberate decision, not a formality.

Working rules that come from this layout:

* Code changes go in the **feature worktree**, with tests (`PYTHONPATH=src python -m pytest tests -q`; currently
  **288 passed, 1 skipped**), then the same patch is applied to the **pin** deliberately and recorded in `PINNED_FROM.txt`.
* Never leave the pin and the feature branch silently divergent: that is how a run trains code nobody can reproduce.
* `dc_runs/` is outside the workspace; writes there need elevated permission.
* **A running bash script must not be edited** (the sweep is read incrementally). Stop the queue, edit, restart.

## 2. What is running right now

`dc_runs/queue_ns.sh` drives `dc_runs/queue_ns.txt`, which currently contains only `wtsweep` and `gate1_tfmix`; the
gates append everything after that.

The sweep (`launch_wtsweep.sh`, pid visible in `wt_sweep.log`) is mid-`floor0`: arm **`wt_floor0_drop02b`** =
WT rank 16 / alpha 32 / dropout 0.2 with the zero knee, matched against `wt_drop02` (same stack, data knee, 0.8447).
It self-stops after three plateau decays (~epoch 5). Then:

1. `floor_decide.py` runs as a hook. It **adopts the zero knee by parsimony** (seed-1 pair only; `--retest` re-enables
   the second seed) and rewrites the knee lines of `wt_stack.txt`.
2. `capacity2` runs **`wt_r16a16`** = floor 0 / rank 16 / alpha 16 / dropout 0.2 under the **new metric definition**.
   That run is the **new WT gold standard**; `wt_best.txt` is set to `0.0` so it becomes the standing best rather than
   being compared against an old-definition number.
3. `lr` → `balance` → `reach` follow, stacking onto the gold standard.
4. `gate1_tfmix` generates the frozen-WT block from the sweep winner (6 arms: 1 knee A/B + 5 `--tf_mix` points), then
   `gate2_ns` picks the mix and appends the `fx` arms (2 seeds + a seed-2 control).

## 3. What the last two days established

* **The link knee (floor) costs nothing on the WT side.** Matched at the base config: `wt_floor_data` 0.8391 (8 epochs)
  vs `wt_floor0_base` 0.8403 (6 epochs), per-epoch curves within 0.0017. The zero knee is the simpler prior — one
  asserted knee instead of a per-library floor that only ~47 of 170 libraries can measure — so it is adopted. An
  earlier "0.0044 deficit" was **not** the knee: see the next bullet.
* **`launch_wtsweep.sh` used to wipe the stack it was resuming.** It truncated `wt_stack.txt` and deleted `wt_best.txt`
  at start, so a *rescheduled* restart silently dropped rank16/alpha32/dropout0.2 before the floor arm, which then ran at
  rank 2/alpha 4/dropout 0.1 — producing a fake knee deficit — and, with no standing best, `wt_pick` auto-stacked a
  one-arm stage. Fixed: the sweep now seeds both files from `wt_stack_seed.txt` / `wt_best_seed.txt`. The diagnostic
  habit that caught it: **diff the running arm's `hparams.yaml` against the arm it is supposed to be matched to.**
  Recorded as STATE pitfall 12.
* **OOD benchmarking sets are out of every validation aggregate** (2026-10-19). `ptmuld`, `s461`, `ssym` are out of
  distribution; they are still logged per library for diagnostics and summarised as `val_rho_combined_external` /
  `val_rmse_combined_external`, but `val_*_avg`, the pooled ranking (`val_rho_ungrouped_*`) and the whole `val_epi_*`
  hierarchy are in-distribution only. **Aggregates are therefore not comparable across that date** — recompute from
  the per-library columns when an older run must be compared. The list is the `--external_val_sets` flag and is carried
  into generated frozen-WT blocks.
* **`val_epi_cell_rank_ctx` is the only WT-side cell metric** (cell levels exist for comb/mt/ctx only; in a WT-only arm
  comb and mt read 0.0). It is a genuine held-out generalization probe — the WT head never sees a double — but it is
  evidence of *context dependence*, not of an interaction model: pair/library recognition and imperfect saturation
  removal can both contribute. Numerically it is a WT-adapter property: WT-only arms span 0.133–0.160 while the
  full dc17 model's ctx control reads 0.1407. Use it to select a frozen WT or to catch offset-fitting, not as an
  objective, and ignore differences under ~0.02 (replicate SD 0.007–0.009, single seed, best-epoch selection).
* **The ns term (`--nonspecific_term fx`) is still the best single-seed result**: `beyond_rho_mt` +0.059 and
  `cell_mag` +0.019 over a matched control, with a learned f(x) that decreases in the additive expectation. It is
  queued last, on the stacked recipe, with two seeds — that is what `gate2_ns` exists to do.

## 4. Commit / branch state at handoff

* `devel-experimental` (`repo/`): documentation commits only. HEAD at handoff is `d822891` (this file, the STATE §4
  rewrite, EXPERIMENTS §6c, the metrics doc and the README index).
* `feature/nonspecific-term` (`nonspecific-term` worktree): `7db75a7` ns term → `cf2b450` frozen WT → `71d21fe` link
  loading → `18b1ede` `--tf_mix` → `384ec80` `incl_*` fix → `7266661` `--val_wt_only` → the OOD-exclusion commit
  (training.py, config.py, `tests/test_external_val_sets.py`).
* `pin_dc20_series`: no git history of its own; `PINNED_FROM.txt` is the record and now lists the OOD patch too.
* Nothing is pushed. `devel-experimental` is ahead 23 / behind 31 of `origin/devel-experimental` — a merge is a
  separate decision and has not been attempted.

## 5. Immediate next actions (in order)

1. **Let `wt_floor0_drop02b` finish** (~13:00–13:30 on 2026-10-19). Do not kill it; it is the last old-definition run
   and the seed-1 leg of the knee confirmation.
2. **Verify the new metric on `wt_r16a16`'s `metrics.csv`** (do not trust the diff):
   * `val_rho_combined_external` and `val_rmse_combined_external` exist;
   * `val_rho_combined_avg` equals the `nanmean` over the 25 non-OOD per-library columns;
   * `val_rho_ungrouped_combined` matches a pooled recomputation that excludes the three OOD libraries.
3. **Confirm the gold standard** is recorded as floor 0 / r16 / alpha16 / dropout 0.2 (arm `wt_r16a16`), and that
   `wt_best.txt` and `wt_stack.txt` reflect it.
4. Then let `lr` → `balance` → `reach` run, and check `gate1_tfmix` when the sweep ends (it generates six arms and
   verifies every command line with `--verify`).
5. Deferred, not queued: the from-scratch ns arms (`ctrl_s2`, `ns1_s1/s2`, `ns1_coarse_s1`); `claude_queue.txt`
   still lists `wtsweep` and dc19–dc23 and must be restated before that runner is ever restarted; and **inference does
   not read `link_head.*` or `ns_head.*`**, so nothing trained with `--link` or `--nonspecific_term` can ship yet.

## 6. Where the numbers are

* `dc_runs/queue_history.txt` — the operational record: every launch, kill, confound, repair, decision and its reason.
  Read the tail of this file first; it is the fastest way to learn what actually happened.
* `dc_runs/wt_sweep.log` — per-stage arm results, `WINNER`/`STACKED` lines, margins.
* `dc_runs/floor_decide.log` — the knee decision, with both seed-1 numbers.
* `dc_runs/logs/<arm>/0/metrics.csv` — per-epoch, per-library validation metrics (the source for any manual recompute).
* `dc_runs/logs/<arm>/0/hparams.yaml` — what an arm actually ran; the check that catches stack/confound errors.
* `docs/EXPERIMENTS.md` — measured history and open questions; `docs/STATE.md` — facts, pitfalls, design invariants.
