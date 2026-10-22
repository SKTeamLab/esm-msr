# CI-weighted training (`--ci_weight`) in place of censoring on doubles

Written 2026-10-22. Implements open question 7 of `EXPERIMENTS.md` §7 and queues its first test. **Status: implemented, tested on
CPU (356 passed), real-cache smoke-tested; the arm `mt_frozenwt_r2a4_ciw_s1` is queued, not yet run.** Everything here is off by
default: with the new flags absent every loss and every dataset is what it was (the 333 pre-existing tests are untouched).

## 1. What was built

| flag | default | what it does |
|---|---|---|
| `--ci_weight` | off | weight each **uncensored** item of `--ci_weight_subsets` by the precision of its own label |
| `--ci_weight_subsets` | `double` | which subsets are weighted; the rest keep weight 1 |
| `--ci_weight_tau` | `0.02` | kcal/mol of variance that is not the label's, added to every item (§2) |
| `--ci_weight_losses` | `both` | `reg`, `rank` or `both`: which losses see the weights, so a result can be traced to one |
| `--no-censor_doubles` | censor | no floor, threshold or gap rule reaches a measured double or cond item; singles are censored as before; a double the assay only **bounded** (`<-1` / `>5`) stays a bound |

The weight is `w = [1/(σ² + τ²)] / mean(1/(σ² + τ²))` with `σ = deltaG_95CI / 3.92`, the mean taken over the training items it
applies to (computed once at start-up and logged, `args.ci_weight_ref`), so the weights average 1 and the regression keeps its
scale. The ranking loss uses a second normalisation of the same weights, geometric mean 1, for a reason given in §3.
`deltaG_95CI` is the *full width* of the 95% interval, and for every numeric row `dG_ML == deltaG`, so it is the interval of the
training label's own fit. No cache change: `meas` already carries the column (`data.MEAS_COLUMNS`), so `--cache_version c5` is
untouched and no rebuild happens.

Code: `src/esm_msr/ci_weights.py` (the weights), `losses.py` (`ListMLELoss(weights=)`), `censoring.py` / `data.py` /
`preprocess_megascale.py` (`censor_multi` / `--no-censor_doubles`, training loaders only; validation scores the canonical
population whatever training censors), `training.py` (`ci_item_weights`, the composer), `config.py`.
Tests: `tests/test_ci_weighting.py`. Evidence: `analysis_scripts/ci_weighting_stats.py`, `analysis_scripts/ci_weighting_sim.py`.

## 2. What the interval is on this data (`ci_weighting_stats.py`)

* **It is small.** The CI-implied label SD is a median **0.035 kcal/mol** on in-range doubles. The two proteases disagree about
  the same variant by a median 0.30 (a per-protease SD of ≈ 0.35), ten times larger, and the model's held-out error is
  `val_rmse_combined_avg` ≈ **0.49**. The interval is the fit's statistical precision, not the assay's error.
* **It barely tracks the real error.** Spearman(CI width, protease gap) = **0.12** in range (0.15 over all doubles).
* **It rises at the destabilised end, but gently.** Doubles, median CI width by dG band: 0.12-0.15 across dG 0-4.5, 0.17 at
  -0.5..0, **0.30** below -0.5. The protease gap over the same bands goes 0.3 → 0.76 → **2.4**.
* **So it cannot stand in for floor censoring.** 33.3% of doubles read at or below the 0.5 threshold. Under *pure* inverse
  variance (τ = 0) they would still keep **29.2%** of the total weight (their CI is 0.167 median against 0.139 for the rest).
  What the floor does to a reading is bias (it reports the unfolded-state composition error of that protease and library), not
  variance, and a confidence interval measures variance. Weighting is not a softer censoring; it is a different thing.

Weights over the doubles, by τ (`w95/w05` is the 95th over the 5th percentile weight; ESS is the effective sample size):

| τ | w95/w05 | ESS | max / mean |
|---|---|---|---|
| 0 (pure inverse variance) | 30× | 47% | 28.7 |
| **0.02 (default)** | **14×** | **71%** | **3.7** |
| 0.04 | 6× | 86% | 1.9 |
| 0.10 | 1.9× | 97% | 1.2 |
| 0.40 (≈ the model's residual) | 1.1× | 100% | 1.0 |

`τ` is the variance that is not the label's: the inverse-variance weight of a regression whose residual is label noise plus the
model's own misfit is exactly `1/(σ² + τ²)`. **The statistically correct τ is therefore the model's residual SD (≈ 0.4), where
the weights are flat and `--ci_weight` does nothing.** Any τ that makes the weights discriminate is a deliberate emphasis choice
(trust the precisely measured items more than the model's error would justify), not a maximum-likelihood weighting. The default
keeps most of the contrast and stops a few tiny-interval items from carrying 29× the average.

## 3. Can it be weighted appropriately in both losses?

**Regression: yes, exactly.** Inverse-variance weighting is the maximum-likelihood fit under heteroscedastic Gaussian noise. The
weights multiply the numerator only (`global_w_sum` stays the sum of the *subset* weights), so a batch of poorly measured items
carries less in total, which is the right behaviour, and the loss keeps its scale (`tests/test_ci_weighting.py`,
`test_regression_numerator_is_weighted_and_the_denominator_stays_the_subset_sum`). A censored item is not weighted: its label is
a bound, and a range-bounded double's `deltaG_95CI` belongs to the unbounded fit the assay declined to report (median 0.65,
four times a fitted row's).

**Ranking (ListMLE): only approximately, and the obvious way is wrong.** Plackett-Luce has no noise parameter. It models
*choices* (the k-th term is "this item is next, out of those left"), and a label's noise reaches the loss through the *order* it
induces, which depends on pairs of items, not on one. The first thing one writes, scaling each choice term by its chosen item's
weight, is measurably harmful. Simulation (`ci_weighting_sim.py`; 24 seeds, paired; change in Spearman against the unweighted
fit; noise ×3 is a median label SD of 1.5 against a unit signal):

| noise | misfit | regression, weighted | rank: scale the chosen term only | rank: offset only | rank: **offset + term (shipped)** |
|---|---|---|---|---|---|
| ×1 | 0 | +0.0025 | **−0.0107** | +0.0010 | **+0.0021** |
| ×3 | 0 | +0.0216 | **−0.0774** | +0.0035 | **+0.0118** |
| ×3 | 0.5 | +0.0159 | **−0.0867** | +0.0012 | **+0.0099** |
| ×0.25 | 0.5 | −0.0019 | −0.0023 | −0.0012 | −0.0009 |

Why: each Plackett-Luce term is zero-sum in the scores (+1 to the chosen item, −1 shared among those left). Weight only the
chosen term and a low-weight item keeps its full share of the negatives from other items' terms while its own positive term
shrinks, so low-weight items are pushed down for no reason the features can explain. `test_uninformative_labels_exert_no_net_force_on_any_item`
shows it exactly: with equal scores and labels that say nothing (averaged over all 24 orders of four items) the term-only loss
pushes the lowest-weight item with +0.70 and the highest with −1.09, and the shipped loss exerts 0.0 on all of them. The fix is
to treat `log w` as a known offset on every score, `exp(s_j + log w_j)`, in numerator and every denominator, and to scale each
term by `w` as well. That restores the balance and lets a noisy competitor weigh less against the winners above it
(`ListMLELoss._offset`). It is not a likelihood of the noisy labels.

*The normalisation that makes the balance hold in the real lists.* The MT stream mixes singles (weight 1, offset 0) and weighted
doubles in one list of 16, and the balance needs every item's term weight to be the **same multiple of `exp(offset)`**. Weights
normalised to arithmetic mean 1 sit beside offsets `log w`, which average `-0.256` over the doubles at the default τ (a systematic
0.26 shift of every double's score against the singles of its list once labels are informative). Centring the offsets removes the shift
but, with arithmetic-mean-1 term weights, breaks the balance again (a residual force of 0.03-0.07 per item in the test, with a
direction). The consistent form is the geometric-mean-1 normalisation for the *ranking* loss only: offsets `log w - mean(log w)`,
term weight `exp(offset)` (`ci_weights.item_offsets`; `test_a_mixed_list_is_balanced_only_when...`). Its price is stated, not hidden: a
ranking term is `e^{0.256} = 1.29×` larger on a weighted item at the default τ (1.09× at τ = 0.05), logged at start-up. The
regression keeps arithmetic mean 1 and its scale. `--ci_weight_losses reg` removes the 1.29× entirely. A pairwise form (weight a pair by how far apart the labels are
against their combined SD, the "near-noise pairs" of §7) is the principled alternative; it needs an O(n²) list term and was not built.

**Two cautions the table carries.** (1) The effect is small and scales with how much noise there is to see: at noise ×0.25
with model misfit, *every* weighting scheme is slightly negative, including regression (−0.0019). This is the regime the real
data sits in (label SD 0.04 against a model error of 0.49), so the honest prior for `--ci_weight` alone is **no gain, perhaps a
small loss**. (2) The simulation's weights see the label noise exactly; here the interval sees only a tenth of it (§2).

## 4. The arm: `mt_frozenwt_r2a4_ciw_s1`

The `mt_frozenwt_r2a4_s1` recipe (frozen best-epoch WT of `wt_r2a4_f0_cthr05_s1`, MT-only, rank 2 / alpha 4, seed 1) with
`--no-censor_doubles --ci_weight --ci_weight_tau 0.02`. Reference: `mt_frozenwt_r2a4_s1`, a paired comparison at the same seed
(identical WT head, identical data order: no item is added or dropped, only whether a double is censored).

What is **kept** from the reference, deliberately: `--censor_range` (range-bounded doubles stay bounds, because they have no
value to weight and dropping them re-selects the doubles on their own outcome), `--censor_threshold 0.5` for **singles** (so the
teacher-forcing cut is unchanged: a double whose single reads at or below 0.5 still takes the WT pass, and a single still
supplies no label for itself), and everything else. To read "instead of censoring" more strictly (drop the bounded doubles too),
add `--no-censor_range`; that changes the population further and is a separate arm.

**This is a large population change, not a reweighting.** In the four libraries checked on the real cache, the reference
censors 38% (2N88), 85% (1PSE), 29% (2RU9) and 0% (5JRT) of the doubles by threshold; the arm trains on all of them as values.
Across the table, 33% of doubles read at or below 0.5. Whatever the arm shows is mostly "doubles uncensored", with the CI
weights a second-order modifier (§2: they keep 29% of the weight on that population). Read it next to
`mt_frozenwt_dthr0_r16a16_s1` (doubles threshold 0 instead of 0.5), the nearest precedent. If a clean attribution to the weights
is wanted, the follow-up is `--no-censor_doubles` **without** `--ci_weight` (the same arm, flat weights), and `--ci_weight` **on top
of** censoring; each is one more run, ~4.5 h.

Reading it: target `val_epi_cell_rank_mt` with `cell_mag_comb` and `cell_flipacc_mt`, guards `val_rho_wt_valid_avg` (fixed, the WT
is frozen) and `val_rmse_combined_avg` (reference ≈ 0.49). One seed is screening, not a finding (`STATE.md` §5: a single-seed
difference needs ≈ 0.02 on `cell_rank_mt`); promote only per the triplicate rule. The log carries `train/ci_w_mean` (should sit at
1.0) and, at start-up, the weight distribution and its reference.

## 5. Known simplifications

* The weight uses the double's **own** interval. With teacher forcing (`--tf_mix 1.0`) the combined prediction also inherits the
  noise of the two singles' labels, `¼(σ_A² + σ_B²)` on top of `σ_AB²`; the singles' intervals are not carried on the double and
  adding them would need a load-time lookup. With label SDs of 0.04 it would change the weights' ordering very little.
* The interval is `deltaG_95CI` of the combined fit; the protease gap (§2: 10× larger, and what actually predicts a bad
  reading) is **not** in the weight. A weight built on it is a different experiment, closer to the existing `--censor_gap`.
* Replicate rows are collapsed by taking the mean of the interval widths, which slightly overstates the width of a replicated
  variant (rare).
