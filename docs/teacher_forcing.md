# The teacher-forced ensemble, reimplemented (`--tf_ensemble`)

## Why it exists again

The legacy combined objective — `(WT + MT)/2` regressed against the measured ddG — was removed from this branch;
`RETIRED_FLAGS` still lists `lambda_reg_combined`, `lambda_rank_combined`, `lambda_epi_combined`,
`detach_ensemble_input`, `mt_reg_mask` and `zero_epistasis_for_singles` as belonging to it. This is **not** that
objective brought back. It is a separate option that differs in one deliberate way, which is the reason for
rebuilding it rather than reverting.

The old objective took the WT half of the combined prediction from the **measured additive label**
(`ddG_A + ddG_B`) whenever it had one. That is a regime the MT head is never in at inference: there is no label
then, only the WT head's own prediction, which carries an error the MT head was never shown. So the correction it
learns is a correction to a quantity it will not be given.

`--tf_label_prob` is the ratio that fixes that. Per item, per step, the WT half is the measured label with
probability `p` and the WT head's own prediction with probability `1 - p`:

* `p = 1.0` — pure teacher forcing, i.e. the legacy regime exactly. This is the default when the option is on, so
  that turning the option on alone changes nothing about the regime and isolates the reimplementation itself.
* `p = 0.0` — fully free-running: the MT head only ever corrects what the WT head actually says.
* in between — the head sees both, resampled every step, and cannot come to depend on the label.

An item with no finite additive label always takes the prediction, whatever `p` is. The realised fraction is
logged as `train/tf_label_frac`, because `p` only sets it in expectation, and a run should show the regime it
actually trained in rather than the one it was configured with.

## What it costs, and why it is off by default

This branch runs the WT and MT passes as **separate work units**, so inside an MT unit there is no WT prediction
in hand — unlike the old objective, where both passes ran over the same micro-batch. The implementation therefore
takes one extra **no-grad** WT forward over the MT unit's rows. That is close to doubling the forward cost of
every MT unit, which is why `--tf_ensemble` defaults to off. The forward is skipped entirely when `p >= 1` and
every item in the unit has a label, since nothing would read it.

The WT half is detached either way: it is either a label (a constant) or a no-grad forward. Gradient reaches the
MT pass only.

## Open questions, for whoever picks this up

1. **The scale mismatch is real and unresolved.** `ddG_A + ddG_B` is on the **observed** scale — it is a sum of
   two saturated readings — while `mt_pred_cal` is **latent**, and the combined latent then goes through the link.
   So at `p = 1` the WT half is a saturated quantity being treated as an unsaturated one. The old objective had
   the same flaw and guarded only the worst of it, by dropping doubles where either single was measured at the
   floor. Options: invert the link on the label, restrict to items whose singles are comfortably off the floor,
   or lower `p` so the term is mostly the (correctly latent) WT prediction. Note that the mismatch enters at
   weight 0.5 here; it would enter at weight 1.0 under a sum-style combination.
2. **Where the ratio should sit.** A fixed `p` is the simplest thing. A schedule — start at 1.0 and anneal to 0 —
   is the usual answer to this exact problem in sequence models, and would let the MT head learn the easy
   correction first. Nothing here implements a schedule.
3. **Doubles-only batches.** The motivation for moving this work to this branch was its batch construction: on
   `devel-revision` an `--mt_reg_mask doubles` arm starves about 47% of its rank lists, because the median
   training library there has **no doubles at all**. On this branch doubles enter as `cond` items
   (`esm_msr.routing`), so the sampler can be asked for batches that actually contain them. That is the next
   piece of work and it is not started.
4. **Interaction with the component losses.** `--mt_comp_offset/_subst/_int` split the MT regression over
   complete blocks. The combined regression added here is a plain weighted MSE and does not participate in that
   split. Whether the combined term should also be decomposed is untested.

## Flags

| flag | default | meaning |
|---|---|---|
| `--tf_ensemble` | off | master switch for the combined objective |
| `--tf_label_prob` | 1.0 | P(WT half is the measured label) per item per step |
| `--lambda_tf_reg` | 1.0 | weight of the combined regression |
| `--lambda_tf_rank` | 0.0 | weight of a within-column rank loss on the combined prediction |

`tests/test_tf_ensemble.py` pins the ratio's behaviour: `p=1` reproduces the legacy choice, `p=0` is free-running,
an unlabelled item never takes a label, the realised fraction matches the requested one, the draw is resampled
between steps rather than being a fixed per-item split, and the half is always exactly one source and never a
blend of the two.
