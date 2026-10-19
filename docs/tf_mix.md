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

Not mixed, by design: **singles** (their forced half is their own measurement, which is also their target),
**items with no usable label** (they fall back to the WT head for any m), and items already dropped because one of
their singles is measured at the floor. Under `--tf_delink` the labels are de-linked one single at a time first and
the blend happens after that, and the raw-scale half is derived from the calibrated one, so the regression and rank
objectives stay consistent.

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
the target from "the measured singles" to "one fixed model's prediction", with nothing else moving.

## 3. The queued sweep

Same frozen WT model for all five arms — now the **capacity-ladder gold head** (`gate_capacity.py` → `wt_gold.txt`),
not dc17 — so **m is the only variable**. The frozen head's own architecture (WT rank/alpha, knee, censoring) comes
from its checkpoint's hparams.

**The MT head is the canonical r16a16 in the family.** With `use_rslora=True` the adapter scale is `alpha/sqrt(r)`,
not `alpha/r`. The capacity ladder held that scale at the reference's 4.0, and a scale-preserving MT head at rank 16
is `16/sqrt(16) = 4.0` — the canonical value. So the tf-mix block keeps r16a16: the mix stays the only variable and
the block remains comparable with the dc17 line.

**One deviation arm, not the family** (user, 2026-10-19). `gate2_ns` also generates a single arm at the winning mix
with `--mt-alpha wt`: the MT alpha becomes the frozen WT gold's alpha (`r16a11` for an `r8a11` gold), tagged `_mtaw`
so it cannot collide with the canonical arm. This is deliberately a **scale** test, not a capacity one — under rsLoRA
it moves the MT scale to 2.75 — and alpha is entangled with the learning rate, so it should be read as "does a smaller
MT scale help at this mix", not as a drop-in replacement.

Table (names are the launcher/queue names; each experiment is suffixed `_w<WTrank>a<WTalpha>_m16a<MTalpha>`, e.g. `_w8a11_m16a16` canonical and `_w8a11_m16a11` for the deviation arm):

| arm | m | note |
|---|---|---|
| `fwt_ctrl_s1` | 1.0 | the control **and** the paradigm's own A/B against `dc17_s1` |
| `tfmix_75_s1` | 0.75 | mostly the label |
| `tfmix_50_s1` | 0.50 | midpoint |
| `tfmix_25_s1` | 0.25 | mostly the WT head |
| `tfmix_00_s1` | 0.00 | the WT head alone — the lower bound, and the arm that measures what the measured singles were worth |

Order: the control, then descending away from it, so a mid-queue read always has its reference. Cost ~3-4 h per
arm (doubles-only stream, ~217 steps/epoch, validation dominating), so **~15-20 h for the block**, after the capacity
ladder decides the gold head and before `gate2_ns`.

## 4. What to read

* **`val_epi_beyond_rho_mt`** and `val_epi_beyond_rho_comb` first — the interaction level is the MT head's job and
  the only level that cannot be won by getting the additive part right.
* Then `cell_rank_mt`, `cell_mag_comb`, `cell_flipacc_mt` (mean of the last 3 epochs), against `fwt_ctrl_s1`.
* Guards: `val_rho_wt_valid_avg` should be identical across arms (the WT head is frozen and loaded from one
  checkpoint - if it is not, an arm failed the load check), `val_rmse_combined_avg` and
  `val_epi_beyond_rho_comb` must not fall.
* Two readings would be findings in their own right: **m = 0 not hurting** means the measured additive labels add
  nothing to the WT head; **m = 1 not hurting** means the WT head's additive error is not what limits us. A peak in
  the middle is the interesting case.
* `ns/f_at_x*` is irrelevant here (no ns term in these arms); do not confuse the two sweeps.

## 5. Risks

* **Self-reference at low m.** With m < 1 the MT head's target partly *is* the WT half of its own combined
  prediction, so it can reduce the loss by agreeing with the additive model instead of learning the interaction.
  The signature to watch: `beyond_rho_mt` flat or falling while `rho_combined`/`rmse` hold or improve — the arm has
  collapsed toward the additive model. `beyond_rho_mt` is the metric that cannot be gamed that way, which is why it
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
