# Stacking: how a win propagates into later runs

This project's central rule is **stack gains from wins**: a later arm inherits whatever an earlier one proved, and
nothing is inherited on the strength of a number nobody re-read. This document says what stacks onto what, where the
decision points are, and how to execute one.

## 1. What stacks onto what

| win | stacks into | does **not** affect | how it propagates |
|---|---|---|---|
| **ns test** (`ctrl_s2` vs `ns1_s1`/`ns1_s2`) | every later epistasis arm: `--nonspecific_term fx` becomes part of the recipe | `wtsweep` — it is WT-only (doubles dropped by `--subset_caps single=None double=0.0`, MT frozen and both MT lambdas 0, so the MT forward never runs and `f(x)` acts on multi-mutants only: it cannot bind) | generation argument `--ns yes` |
| **WT sweep** (`wtsweep`) | the frozen WT model of every later epistasis arm: its **weights** and its **architecture** | the from-scratch ns arms (already pinned, already queued) | `make_fwt_block.py --ckpt <winner>` |
| **tf-mix sweep** | the canonical `--tf_mix` value, and the ns block built on it | — | `gate2_ns` regenerates with the winning `--mix` |
| **floor knee** | (a) the WT sweep's own arms, through `wt_stack.txt`; (b) the frozen-WT block, through the checkpoint's `assay_floor`/band read by `make_fwt_block.py`; (c) one A/B arm at the frozen stage | — | the WT-side choice is the sweep's floor pair; the MT-side value is measured only on the frozen head (`--knee data\|zero`), because a WT arm's MT columns are meaningless |
| **dc-series schedule/recipe** (dc17's plateaus, `--per_mutation_bias`, data-only link) | already the base of every dc/fwt arm | — | it is in the template |

The two propagation mechanisms already in the repo are `wt_stack.txt` (the WT sweep appends the flags of each stage
winner; later arms inherit them through `launch_wt_arm.sh`) and `--nonspecific_term`/`--tf_mix` flags. The third,
added for the frozen-WT family, is the generator below.

## 2. Why the architecture has to stack with the weights

A checkpoint cannot be loaded into a differently shaped adapter. `--init_wt_from_checkpoint` copies tensors of
matching name and shape and reports mismatches; a model built at rank 2 cannot accept the rank-16 adapter that won
the capacity stage. So the frozen-WT block **must** be built with the winning WT architecture.

That is why `make_fwt_block.py` reads the architecture out of the checkpoint's own `hyper_parameters`
(`lora_rank_wt`, `lora_alpha_wt`, `lora_dropout_wt`, `target_mode_wt`, `last_n_layers_wt`,
`incl_sequence_head_wt`, `use_dora_wt`, `unfreeze_layernorms_wt`, **`per_mutation_bias`**) and emits them as
explicit flags, rather than trusting a hand-typed stack. `--per_mutation_bias` belongs on that list for a subtler
reason: it changes what the calibration bias *means*, so loading a checkpoint trained with it into a run without it
silently mis-scales the additive half.

It also refuses to run when those hparams are missing, instead of guessing.

## 3. The decision points

The queue is deliberately split so that nothing is built on a superseded choice:

The order is **wtsweep → tfmix → ns** (reversed 2026-10-18 at the user's direction): the WT head is locked down
first so everything after it is measured against the best synthetic additive background available.

```
queue_ns.txt (running)     wtsweep                 <- 13 arms; stage 0 is the floor knee, stages 1-5 the WT shape
        |
        |  GATE 1 (gate1_tfmix): which WT head won, with what architecture and knee?
        v
       tfmix arms          fwt_ctrl, tfmix_75/50/25/00   <- how much the optimized WT is worth to the MT head
        |
        |  GATE 2 (gate2_ns): which mix won?
        v
       ns arms             same recipe + --nonspecific_term fx, 2 seeds, plus a seed-2 control (2v2)
```

Both gates run as queue entries and append their block to the queue, so the ladder runs unattended.

The generated block is **not** in `queue_ns.txt`: it would otherwise commit ~20 h to a frozen WT head and a recipe
that the two sweeps may have replaced. It is generated once, from artifacts, and appended — the runner re-reads its
list before every arm, so appending while `wtsweep` is still running is the normal way to do it.

## 4. Executing a gate

Read the sweep first: `dc_runs/wt_sweep.log` ends each stage with its results and a `STACKED:` line, and
`wt_stack.txt` holds the union of the winning flags. The arm to freeze is the **final stage's winner** — it is the
only one that carries the whole stack. Its checkpoint lives in `dc_runs/logs/<arm>/0/`, ranked by
`val_rho_wt_valid_avg` in the filename.

```bash
cd /home/sareeves/playground/esm-msr-devel/dc_runs
# 1. the frozen WT model (final stage winner; the run dir is per arm)
CKPT=$(ls -1 logs/wt_<final-stage-winner>/0/epoch=*-val_rho_wt_valid_avg=*.ckpt | sort | tail -1)

# 2. generate the block, stacking the ns win if it won, and verify every command line
python make_fwt_block.py --ckpt "$CKPT" --ns yes --mix 1.0,0.75,0.5,0.25,0.0 --verify

# 3. append it to the running queue (the runner re-reads the list before each arm)
cat queue_fwt_tfmix.txt >> queue_ns.txt
```

Then log both decisions in `queue_history.txt` (the ns verdict, the WT winner and the architecture it carried) and
update `docs/STATE.md` §4. If a gate says "no win", the corresponding stack is simply left out: `--ns no`, or
`--ckpt` pointing at the current canonical head (dc17's), which is what the block was last generated with.

## 5. Arm parity: everything else must match the reference

A stage may change its own variable and nothing else. Before an arm family runs, diff it against the reference run
(dc18 for the current recipe) and account for every difference:

```bash
python dc_runs/arm_parity.py launch_dc18_s1.sh wt_base_args.txt launch_fwt_ctrl_s1.sh launch_tfmix_50_s1.sh
```

That audit found the WT sweep carrying `--no-val_cycle_passes` while dc18 ran it — an unintended deviation that
would have left the sweep without the ctx diagnostics. The differences that remain are deliberate: the WT sweep
turns the MT half off (`--lambda_*_mt 0`, `--freeze_mt_adapter`), drops doubles (`--subset_caps double=0.0`), watches
`val_rho_wt_valid_avg`, runs 8 epochs, and validates the WT head only (`--val_wt_only`); the floor knee is its
stage 0. The frozen-WT arms differ by the paradigm itself (`--frozen_wt_mt_only`, the stacked WT architecture and
link/censoring semantics, `--lambda_*_wt 0`) plus their own `--tf_mix`.

**When does a WT-only arm stop?** The WT path decays on `val_rho_wt_valid_avg` (patience 1, ×0.3) and freezes after
`--wt_freeze_after_decays` (3 in the sweep base, not 2), at which point `_stop_if_nothing_is_training` ends the run —
but only if no group is left at lr > 0 **with trainable parameters**. It used to not stop: `_transition_to_mt()`
unfroze the MT path unconditionally, which in a WT-only arm (MT frozen on purpose) left a live group behind, so the
arm ran its whole epoch cap re-scoring identical frozen weights. The pin now respects `--freeze_mt_adapter` there;
arms started before 2026-10-18 23:35 keep the old behaviour. `--early_stopping_patience` is absent from the sweep
base (default 0, no callback), so the cap and the freeze are the only two stop conditions.

**A pin may be patched, but provenance wins.** When a pinned snapshot needs a change (2026-10-18: `--val_wt_only`
added to `pin_dc20_series`), record it in that pin's `PINNED_FROM.txt` and verify it *against the pin's own idioms* —
the pin resolves masking per pass (`_per_pass(self._pass_masks(), 'wt')`), which the feature branch does not, and a
patch copied between them failed exactly there. `dc_runs/check_pin_val_wt_only.py` is that check for this case.

## 6. Rules of thumb

* **One decision, one artifact.** A stack is a file (`wt_stack.txt`, the generated queue file) or a flag, never a
  remembered value. The generator writes the checkpoint path and the architecture into the queue file's header, so a
  later reader can see what an arm was built on.
* **Do not hand-edit a generated block.** Change the gate's arguments and regenerate; the header records when and
  from what.
* **Parsimony prior for the floor knee** (user, 2026-10-19). Every other stage stacks only on a > 0.001 win, but the
  knee is a *data-derived* hyperparameter — only ~47 of 170 training libraries can measure a floor at all — so one
  asserted knee for every library is the simpler and more defensible prior, and it wins ties. The call rests on the
  **base-config matched pair** (`wt_floor_data` 0.8391 vs `wt_floor0_base` 0.8403, per-epoch curves within 0.0017) plus
  that prior; the capacity-level confirmation (`wt_floor0_drop02b`) was stopped at epoch 1 and the alpha-32 seed-2 pair
  was dropped as superseded (STATE §4, EXPERIMENTS §6b). `dc_runs/floor_decide.py` was the sweep's hook; it now
  defaults to the parsimony call on the seed-1 pair, with `--retest` opt-in. The external benchmark sets are **not** a
  decision input — the "floor0 wins 2/3" read came from the confounded arm generation.
* **Parsimony is the default; an addition must earn its place** (user, 2026-10-19 — the general rule, which
  supersedes the knee-specific wording above). Stacking is unchanged: only a winner stacks. The bar is **asymmetric
  in complexity**: "+0.002 is required to justify non-parsimony additions."
  * A **simpler** change (a smaller WT adapter, one asserted knee) is adopted whenever it is **not clearly worse**
    (`diff >= -band`). Accepting a model that is not worse is not a gain claim, so it needs **no second seed**.
  * A **more complex** change is an addition and needs a clear gain (`diff > +band`). Below that it is not adopted.
  * Among adoptable arms the data decides (highest value); parsimony makes a simpler arm adoptable, it does not
    prefer a worse one.
  The external OOD sets are a **generalization guard**: read `val_rho_combined_external` /
  `val_rmse_combined_external` to see whether a decision travels, and never decide anything from them.
  **Implemented** for the WT head as `dc_runs/gate_capacity.py` (queue entry `gate_capacity`): best-epoch
  `val_rho_wt_valid_avg`, band 0.002, complexity = `lora_rank_wt`. The decision is written to `dc_runs/wt_gold.txt`,
  which `gate1_tfmix` consumes — no stack file, no remembered value. A symmetric seed-2 pair remains available
  (`gate_capacity_final`) for confirming a *non-parsimonious* addition, but it is not queued automatically.
  **rsLoRA caveat**: `use_rslora=True`, so the effective scale is `alpha/sqrt(rank)`. A capacity arm must hold that
  constant: reference r16·a16 = 4.00, r4·a8 = 4.00, r8·a11 = 3.89 (nearest integer to 4·√8). The retired sweep's
  r16·a32 was 8.00 — twice the reference — so those capacity numbers mixed rank with scale.
* **Margin before stacking.** The sweep already refuses to stack a gain under 0.001 (~2 between-seed SD on
  `rho_wt_valid`). Do the same for the ns term: `beyond_rho_mt` ≥ 0.03 with guards flat, two seeds, or it does not
  stack.
* **Restate stale arms before they run.** `claude_queue.txt` still holds `dc19`–`dc23`, which were designed before
  the frozen-WT paradigm and before the ns result. Under this rule they should be re-stated (or replaced by their
  frozen-WT equivalents) rather than run as written.
