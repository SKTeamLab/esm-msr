# `--tf_mix`: how much of a double's WT half is the measured label

Status 2026-10-18: implemented (`18b1ede`, pin `dc_runs/pin_fwt_series`), **queued** at the end of the running
queue as a five-point sweep (`fwt_ctrl_s1`, `tfmix_75_s1`, `tfmix_50_s1`, `tfmix_25_s1`, `tfmix_00_s1`).

## 1. What it tunes

Teacher forcing replaces the WT half of a multi-mutant's combined prediction with the **measured additive
expectation**, `ddG_A + ddG_B`, so the MT head only has to learn the correction. Today that is an all-or-nothing
choice per item: use the label where one is usable, otherwise use the WT head's own prediction.

`--tf_mix m` blends the two on the calibrated scale:

```
forced_wt_half = m * (measured additive label)  +  (1 - m) * (WT head prediction)
```

| m | meaning |
|---|---|
| **1.0** (default) | the label wherever one is usable — the released behaviour, bit-identical (tested) |
| 0.75 / 0.5 / 0.25 | a blend; 0.5 is the arithmetic mean |
| 0.0 | never teacher-force: the WT half is always the WT head |

Mixed too, since 2026-10-19: **singles**, so `m=0` is a pure no-teacher-forcing arm for the whole stream while
`m=1` leaves a single's forced half as its own measurement (which is also its target). Without that, a `m<1` sweep
would still teacher-force the singles and would not measure the compensator/interaction balance it claims. Not
mixed: **items with no usable label** (they fall back to the WT head for any m), and items already dropped because
one of their singles is measured at the floor. Under `--tf_delink` the labels are de-linked one single at a time
first and the blend happens after that, and the raw-scale half is derived from the calibrated one, so the regression
and rank objectives stay consistent.

## 2. Why it is worth tuning

Both candidates are noisy estimates of the *same* quantity — the additive expectation — and their errors are
different in kind:

* **The measured label** carries the measurement noise of two independent singles, the floor clipping of either
  (a single measured at its library's floor reports the floor, not its effect), and censoring. Its error is
  item-specific and roughly independent across doubles.
* **The WT head** is smooth and shared: its bias is exactly in the additive direction (and, under the frozen-WT
  paradigm, it is a *fixed* function — the same error for every double with a similar sequence), but it is not
  inflated by two extra measurements.

The Oct-8 variance analysis put 56% of measured dddG on the additive expectation alone, so this is not a detail:
the target the MT head is scored against is dominated by a term whose estimator we get to choose. m trades variance
for bias, and where the optimum sits is an empirical question.

Under the **frozen-WT paradigm** the question is sharpest: the WT half is constant for the whole run, so m sweeps
the target from "the measured singles" to "one fixed model's prediction", with nothing else moving. With the mix
applied to singles, the sweep is also the balance between the MT head as a **compensator** (m=0: it corrects the
frozen WT head's error on singles and doubles alike) and as a **total/interaction predictor** (m=1: the additive
half is the measurement, so the MT half supplies the total, and the interaction is what the ensemble's average
adds beyond the additive expectation).

## 3. The queued sweep

Same frozen WT model for all five arms — now the **capacity-ladder gold head** (`gate_capacity.py` → `wt_gold.txt`),
not dc17 — so **m is the only variable**. The frozen head's own architecture (WT rank/alpha, knee, censoring) comes
from its checkpoint's hparams.

**The MT head is the canonical r16a16 in the family**, and the block runs the **released `mean` ensemble with the
teacher-forcing corrections** (2026-10-19): `--combined_mode mean`, `--tf_delink`, `--tie_mt_scale`, mandatory
`--per_mutation_bias`, and **singles are kept in the training stream**.

* **`mean`, and why `sum` was removed.** The MT pass produces a *sum of per-position LLRs in the mutant context* —
  a total ddG-like quantity, not a difference. `mean` averages two such totals, which is a valid ensemble and
  exactly what each head computes. `sum` asked the MT head for `y − wt` (the epistasis term): a pure difference
  that the LLR sum cannot produce, and on a single the target would be 0 while the LLR sum is not. `--combined_mode
  sum` was removed from the branch (2026-10-19); `route` remains the no-ensemble alternative.
* **singles are included.** `--frozen_wt_mt_only` no longer forces `--no-incl_singles`: the WT head learns
  contextual effects from singles, doubles are floor-limited and not intrinsically more informative, and the MT
  head's target on a single is that single's own ddG. `--no-incl_singles` still gives the old doubles-only stream.
* **`tf_delink`**: the additive label is a sum of two *saturated* single readings, so each single is inverted through
  the link and summed (the inverse is nonlinear; inverting the sum is a different, wrong number). Singles at or
  below their library floor have no inverse and drop out of teacher forcing.
* **`tie_mt_scale`**: the MT calibration shares the frozen WT head's **scale** (bias separate), so both calibrated
  halves are on one latent scale and the raw-scale rank loss weights the halves as the calibrated combination does.
* **`per_mutation_bias`** is mandatory: the calibrated LLR sum is a sum over positions, so a scalar bias puts a
  double one bias *below* the sum of its two singles (0.1065 kcal/mol).

**`fx` is compatible.** `--nonspecific_term fx` requires `--combined_mode mean`, which this block uses, so `gate2_ns`
generates the ns arms (control seed-2, ns arms at the winning mix, and the `_mtaw` deviation arm) as before.

**One deviation arm, not the family** (user, 2026-10-19). `gate2_ns` also generates a single arm at the winning mix
with `--mt-alpha wt`: the MT alpha becomes the frozen WT gold's alpha (`r16a11` for an `r8a11` gold), tagged `_mtaw`
so it cannot collide with the canonical arm. This is deliberately a **scale** test, not a capacity one — under rsLoRA
it moves the MT scale to 2.75 — and alpha is entangled with the learning rate.

Table (names are the launcher/queue names; experiments are suffixed `_w<WTrank>a<WTalpha>_m16a<MTalpha>_<mode>`, e.g. `_w8a11_m16a16_mean`):

| arm | m | note |
|---|---|---|
| `fwt_ctrl_s1` | 1.0 | the control **and** the paradigm's own A/B against `dc17_s1` |
| `tfmix_75_s1` | 0.75 | mostly the label |
| `tfmix_50_s1` | 0.50 | midpoint |
| `tfmix_25_s1` | 0.25 | mostly the WT head |
| `tfmix_00_s1` | 0.00 | the WT head alone — the lower bound, and the arm that measures what the measured singles were worth |

Order: the control, then descending away from it, so a mid-queue read always has its reference. Cost ~3-4 h per
arm (full singles+doubles stream, validation dominating), so **~15-20 h for the block**, after the capacity
ladder decides the gold head and before `gate2_ns`.

After `gate2_ns`, **`gate3_native_cond`** appends the native_cond routing arms at the winning mix: `nc_mt`
(`--no-native_cond_for_wt --native_cond_for_mt`, MT only, the MT pass given the WT pass's input), `nc_both` (both
heads on the same input) and `nc_none` (excluded). The winning arm and the gate2 control seed 2 are the baseline.
See `docs/dual_adapter_design.md` section 2 for the four-way routing.

## 4. What to read

* **`val_epi_beyond_rho_epi`** (authoritatively defined as ~MT(AB) - WT(AB)) and `val_epi_beyond_rho_pairavg_epi` first — the interaction level is the MT head's job and
  the only level that cannot be won by getting the additive part right.
* Then `cell_rank_epi`, `cell_mag_epi`, `cell_flipacc_epi` (mean of the last 3 epochs), against `fwt_ctrl_s1`.
* Guards: `val_rho_wt_valid_avg` should be identical across arms (the WT head is frozen and loaded from one
  checkpoint - if it is not, an arm failed the load check), `val_rmse_combined_avg` must not regress, and
  `val_epi_beyond_rho_epi` must not fall.
* Two readings would be findings in their own right: **m = 0 not hurting** means the measured additive labels add
  nothing to the WT head; **m = 1 not hurting** means the WT head's additive error is not what limits us. A peak in
  the middle is the interesting case.
* `ns/f_at_x*` is irrelevant here (no ns term in these arms); do not confuse the two sweeps.

## 5. Risks

* **Self-reference at low m.** With m < 1 the MT head's target partly *is* the WT half of its own combined
  prediction, so it can reduce the loss by agreeing with the additive model instead of learning the interaction.
  The signature to watch: `beyond_rho_epi` flat or falling while `rho_combined`/`rmse` hold or improve — the arm has
  collapsed toward the additive model. `beyond_rho_epi` is the metric that cannot be gamed that way, which is why it
  is the primary read.
* **The rank objective inherits the blend** (it uses `combined_pred_raw`), so m changes list ordering too, not only
  the squared error. That is intended, but it means an m arm is a change to both losses at once.
* **No per-item weighting**: m is global, so a double whose singles are clean and one whose singles were at the
  floor get the same trust. That is the obvious next refinement (see below).

## 6. Not implemented (natural follow-ups)

* **A schedule** (warm from m = 1 toward the tuned value, so early training is anchored by measurements and late
  training by the model).
* **Per-item confidence**: weight the label by the uncertainty of its two singles (distance above the floor, censoring
  status, replicate disagreement). The pieces exist — `floor_stats`, `cens`, the two-protease disagreement — but the
  mixing law would need to be defined.
* **Mixing on the rank lists only**, leaving the regression target as the measured label.
