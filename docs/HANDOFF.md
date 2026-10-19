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

**One branch carries both the code and the documentation: `devel-revision`** (`04c707b` at handoff), checked out
at `repo/.claude/worktrees/devel-main-conservative-merge-5e554f`, which is the **execution tree**. Contents:

* `b34093f` — devel-revision proper: the dc17–dc19 code;
* **seven campaign commits**, and nothing else: `--nonspecific_term fx` (ns term), `--frozen_wt_mt_only` +
  `--init_wt_from_checkpoint` (train MT only against a frozen WT), link loading/freezing for that frozen model,
  `--tf_mix`, the `--no-incl_*` fix, `--val_wt_only`, and the OOD exclusion from every validation aggregate;
* `docs/` — this documentation set (imported from `devel-experimental`, which is no longer the docs home).

Deliberately **NOT** in it: `--ext_val_doubles`, per-pass masking (`--mask_strategy_wt/mt`) and `--double_batch_frac`.
Those came from the `claude/ext-val-and-masks` side branch and were carried into the old pin; none of them belongs to
this campaign. The *separation* commits are the point: WT/MT separation (frozen WT, MT-only training, WT-only
validation) and external-vs-in-distribution separation (the `_external` metrics).

**No pinning.** The queue used to run `dc_runs/pin_dc20_series`, a *directory copy* of the `claude/ext-val-and-masks`
snapshot plus hand-applied patches — because worktrees move mid-session and a running queue must not have code changed
under it. As of `04c707b` the launchers `cd` straight into the tree above and **log the commit they started from**
(`tree=... commit=04c707b branch=devel-revision` is the first line of every arm log). The corresponding rule is:
**do not commit to `devel-revision` while the queue is running.**

Two caveats about past numbers: every run before `04c707b` came from the pin (i.e. from devel-revision **plus** the
three ext-val commits, plus the hand patches). That lineage is tagged `archive/pin-lineage-2026-10-19` and the pin
directories are still on disk, so those runs remain reproducible; for the canonical recipe the three dropped commits
are inert (mask strategies off, `--ext_val_doubles` unused, `--double_batch_frac` off), but the code is not
byte-identical to what ran. `devel-experimental` still holds a copy of the docs from the same import point.

Other worktrees exist (`main-baseline`, `delink-above-tau`, `scale-mismatch-improvements`, `practical-snyder`,
`branch-strategy-reorganization`, `ext-val`) and are **not** used by the campaign.

Working rules that come from this layout:

* Code and docs both change on **`devel-revision`**, with tests (`PYTHONPATH=src python -m pytest tests -q`; currently
  **288 passed, 1 skipped**). There is no second tree to keep in sync any more — the old failure mode (pin silently
  divergent from the branch) is gone by construction.
* Keep the branch's scope narrow: campaign changes only. Unrelated side-branch work (masking, `--ext_val_doubles`,
  `--double_batch_frac`) does not belong on it, however harmless it looks — it makes the code that ran ambiguous.
* `dc_runs/` is outside the workspace; writes there need elevated permission. Launchers live there and point at the
  execution tree above.
* **A running bash script must not be edited** (the sweep is read incrementally). Stop the queue, edit, restart.

## 2. What is running right now

The reference run. `dc_runs/queue_ns.sh` drives `dc_runs/queue_ref.txt`, which contains one arm:
**`wt_ref_floor0_r16a16_d02`** — the new WT **gold standard**: floor 0 (`--assay_floor 0.0`, link band 0…0.001),
WT rank 16 / alpha 16, dropout 0.2, singles only, under the **new metric definition**. Its launcher does not read
`wt_stack.txt`, so no stale stack can change it, and its log's first line records `commit=04c707b`.

It started 2026-10-19 12:15 from `devel-revision`; expect ~1.5 h (≈17 min/epoch, self-stopping after three plateau
decays). When it lands, its `val_rho_wt_valid_avg` **is the WT reference** for everything after it, and its
`val_rho_combined_external` / `val_rmse_combined_external` are the external-set baseline.

After it: the queue is otherwise empty. The WT sweep's remaining stages (`lr`, `balance`, `reach`) were **not**
carried over — they were designed against the old metric basis and the old pin. Re-state them if they are still wanted,
and continue with `gate1_tfmix` (frozen-WT block: 1 knee A/B arm + 5 `--tf_mix` points) → `gate2_ns` → the `fx` arms.
The current `wt_stack.txt` / `wt_best.txt` belong to the retired sweep; treat them as history, not state.

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

* **`devel-revision` is the single branch**: `b34093f` + seven campaign commits + `04c707b` (docs import and a comment
  fix). HEAD at handoff is `04c707b`, and `devel-experimental` is no longer the docs home.
* `feature/nonspecific-term` (`nonspecific-term` worktree): `7db75a7` ns term → `cf2b450` frozen WT → `71d21fe` link
  loading → `18b1ede` `--tf_mix` → `384ec80` `incl_*` fix → `7266661` `--val_wt_only` → the OOD-exclusion commit
  (training.py, config.py, `tests/test_external_val_sets.py`).
* The pin directories remain on disk as the record of what ran before `04c707b`; `PINNED_FROM.txt` says so and lists
every patch. They are archives now, not inputs.
* Nothing is pushed. `devel-revision` has never been pushed; `devel-experimental` is ahead 23 / behind 31 of
  `origin/devel-experimental` — a merge is a
  separate decision and has not been attempted.

## 5. Immediate next actions (in order)

1. **Let `wt_ref_floor0_r16a16_d02` finish** (~1.5 h from 12:15 on 2026-10-19) and record its `val_rho_wt_valid_avg` as
   the WT reference.
2. **Verify the new metric on its `metrics.csv`** (do not trust the diff):
   * `val_rho_combined_external` and `val_rmse_combined_external` exist;
   * `val_rho_combined_avg` equals the `nanmean` over the 25 non-OOD per-library columns;
   * `val_rho_ungrouped_combined` matches a pooled recomputation that excludes the three OOD libraries.
3. **Confirm the gold standard** is recorded as floor 0 / r16 / alpha16 / dropout 0.2 (arm `wt_ref_floor0_r16a16_d02`),
   and decide whether the retired `wt_stack.txt` / `wt_best.txt` should be cleared rather than reused.
4. Then re-state the remaining hyperparameter arms (lr / balance / reach) on the new basis, and run `gate1_tfmix` (it
   generates six arms and verifies every command line with `--verify`).
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
