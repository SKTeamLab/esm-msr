# The teacher-forced ensemble, reimplemented (`--tf_ensemble`)

## Why it exists again

The legacy combined objective — `(WT + MT)/2` regressed against the measured ddG — was removed from this branch;
`RETIRED_FLAGS` still lists `lambda_reg_combined`, `lambda_rank_combined`, `lambda_epi_combined`,
`detach_ensemble_input`, `mt_reg_mask` and `zero_epistasis_for_singles` as belonging to it. This is **not** that
objective brought back. It is a separate option that differs in one deliberate way, which is the reason for
rebuilding it rather than reverting.

The old objective took the WT half of the combined prediction from a label whenever it had one. That is a regime
the MT head is never in at inference: there is no label then, only the WT head's own prediction, which carries an
error the MT head was never shown. So the correction it learns is a correction to a quantity it will not be given.

`--tf_label_prob` is the ratio that fixes that. Per item, per step, the WT half is the label with probability `p` and
the WT head's own prediction with probability `1 - p`:

* `p = 1.0` — pure teacher forcing. This is the default when the option is on.
* `p = 0.0` — fully free-running: the MT head only ever corrects what the WT head actually says.
* in between — the head sees both, resampled every step, and cannot come to depend on the label.
* `--tf_label_prob_end` with `--tf_anneal_steps` — a linear anneal from `--tf_label_prob` to the end value, then
  held. It starts at `--mt_lora_delay_steps`, where the MT adapter's learning rate starts, so the anneal is not
  spent on steps the MT head does not train in. Being keyed on `global_step`, it resumes correctly.

An item with no usable label always takes the prediction, whatever `p` is. The realised fraction is logged as
`train/tf_label_frac` (over the items that enter the term), because `p` only sets it in expectation, and it sits
below `p` wherever labels are missing; with an anneal, `train/tf_label_prob` logs the scheduled value too.

## What the label is

The first version of this option used `ddG_additive = ddG_A + ddG_B` as the label, copied from the legacy objective.
**That was wrong on this branch, and by far more than the scale issue it was flagged for.** The legacy objective ran
on `double` items, whose WT pass scores both mutations on the real wild type, so `ddG_A + ddG_B` was the WT pass's own
target. Here the term runs on MT units, which hold `cond` items (and `native_cond`, and anchored singles). A `cond`
item ddG(A|B) has `wt_seq = _seq_with(B)`: its WT pass scores A on the B background, which is exactly what
`forward_batch` averages with the MT pass at validation. Its label is the item's own target ddG(A|B). Using
`ddG_A + ddG_B` instead was off by `ddG_B - dddG`, a median of **2.0 kcal/mol** (mean 2.3, p90 4.6) over the 262k
`cond` items the raw Tsuboyama table yields (every double with both singles measured, before the dataset's own
filters), against a target SD of 0.9. At `p = 1` without a link the term's minimiser
was `mt = 2 ddG(A|B) - ddG_A - ddG_B`, so it pulled the MT head away from the target the MT regression pulls it to.

So the label is now **the item's own target** (`tf_label` in `training.py`), for every item type. Two consequences:

* A labelled item's combined regression is `(½ t + ½ m - t)² = ¼ (m - t)²`: **a quarter-weight copy of the MT
  regression** (exact with MSE; the same minimiser with any criterion). Teacher forcing adds no information on this
  branch — the MT head already gets its target directly from the `cond` items. What `--tf_ensemble` adds is the
  free-running half, and `p` is best read as how much of the term is a copy of `reg_mt` versus the ensemble loss.
  That is still a reason to anneal: early on the WT head's predictions are poor and moving, and a fixed target keeps
  the MT head from learning to absorb an error that is about to change.
* Every item has a finite label without the link, so at `p = 1` the extra WT forward is skipped altogether.

### The scale mismatch (resolved)

With the link, `mt_pred_cal` is latent and the target is observed. The label is now the target taken back through the
link: the `z` with `h(dG_wt + bg + z) - dG_wt = ddG + bg`, the MT regression's own convention, via `link.inverse`
(detached), as the latent censored hinge already does. It is used only where `z` lies strictly between the link's
knees, where `h`'s slope is above about ½; beyond them an observation fixes the latent only to within its noise times
`1/slope`, and on the plateaus not at all, so those items take the WT prediction. A labelled item's term therefore
has exactly the MT regression's minimiser under the link as well. Items whose `ddG` the regression treats as a bound
(`reg_cens != 0`) have no label.

### Other fixes in the same pass

* The combined regression uses the same rows, scale and criterion as the MT regression: `self.crit_reg` rather than
  a hard-coded square, and `reg_ok` (`--subfloor_rank_only`) without the link. Before, sub-floor items the MT
  regression withholds still entered this term.
* The rank term mixed units: it ranked `½ wt_half + ½ mt_pred_raw`, a calibrated kcal/mol half plus a raw LLR half.
  It now ranks raw against raw — the WT head's `pred_raw`, or the label taken back through the WT calibration head,
  as the legacy objective did. Note that at `p = 1` it adds the true target into the scores being ranked, so it is a
  weakened column rank rather than a copy of it; `--lambda_tf_rank` stays 0 by default.

## What it costs, and why it is off by default

This branch runs the WT and MT passes as **separate work units**, so inside an MT unit there is no WT prediction
in hand — unlike the old objective, where both passes ran over the same micro-batch. The implementation therefore
takes one extra **no-grad** WT forward over the MT unit's rows. That is close to doubling the forward cost of
every MT unit, which is why `--tf_ensemble` defaults to off. The forward is skipped entirely when `p >= 1` and
every item a term reads has a label, since nothing would read the prediction. Without the link that is every unit;
with it, a unit holding an item beyond the link's knees still pays for it.

The WT half is detached either way: it is either a label (a constant) or a no-grad forward. Gradient reaches the
MT pass only.

## Open questions, for whoever picks this up

1. **Doubles-only batches.** The motivation for moving this work to this branch was its batch construction: on
   `devel-revision` an `--mt_reg_mask doubles` arm starves about 47% of its rank lists, because the median
   training library there has **no doubles at all**. On this branch doubles enter as `cond` items
   (`esm_msr.routing`), so the sampler can be asked for batches that actually contain them. Not started. With the
   label fixed it matters less for this option specifically, since the term runs on every MT-unit item; it matters
   for how much of each batch is conditional at all.
2. **Interaction with the component losses.** `--mt_comp_offset/_subst/_int` split the MT regression over complete
   position-pair blocks; the combined regression is a plain weighted error outside that split. Now that the labelled
   part is a quarter-weight copy of `reg_mt`, this has a concrete consequence: under, say, `--mt_comp_int 0` the
   copy re-fits the interaction the component weights switched off, at `λ_tf_reg · p / 4`. Decomposing the combined
   residual the same way would close it on the per-unit path; `--pack_pair_matrices` would need cached combined
   predictions for the rest of each matrix, which do not exist. Until then, prefer low `p` when the component weights
   are not all 1.
3. **A pair-level ensemble.** The term here is per `cond` item, on what validation reports for that item. The
   quantity the architecture is built around is the double: `½ (WT_A + WT_B) + ½ (MT(A|B) + MT(B|A))`, with the WT
   passes on the real wild type. Training that directly would pair a double's two `cond` items in one MT unit and
   run the WT pass on the wild-type sequence rather than the item's `wt_seq`; its teacher-forced label would then be
   `ddG_A + ddG_B` legitimately. Untried.

## Flags

| flag | default | meaning |
|---|---|---|
| `--tf_ensemble` | off | master switch for the combined objective |
| `--tf_label_prob` | 1.0 | P(WT half is the label) per item per step; the start of an anneal |
| `--tf_label_prob_end` | unset | anneal `--tf_label_prob` linearly to this; unset = fixed |
| `--tf_anneal_steps` | 0 | length of the anneal in optimizer steps, from `--mt_lora_delay_steps` |
| `--lambda_tf_reg` | 1.0 | weight of the combined regression |
| `--lambda_tf_rank` | 0.0 | weight of a within-column rank loss on the combined prediction |

`tests/test_tf_ensemble.py` pins the ratio (`p = 1` always takes a label, `p = 0` never does, an unlabelled item never
does, the realised fraction matches, the draw is resampled, the half is one source and never a blend), the label (the
target without a link; with one, the latent that reproduces the observation, and none beyond the knees or for a bound),
the schedule, the flag checks, and three properties through the real training step on a stub model: a prediction that
satisfies the MT regression also satisfies the teacher-forced term, with and without the link; without the link the
labelled term is exactly a quarter of the MT regression's gradient; and a fully labelled unit skips the WT forward.
The first two fail with `ddG_additive` as the label, and the link case fails with the un-inverted observed target.
