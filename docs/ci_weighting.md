# Label-uncertainty weighting (`--ci_weight`) in place of censoring

> **Status 2026-10-22 (trim): the code was removed from the working branch** (`docs/trim_audit.md` §2.2). It lives at the tag
> `archive/ci-weighting-2026-10-22` (commit `448dfcd`); `analysis_scripts/ci_weighting_{fit,sim,stats}.py` and `censoring_band_check.py` need that tag
> checked out (they import `esm_msr.ci_weights`). The one arm that used it, `mt_frozenwt_ciw_r16a16_s1`, runs from the locked pin
> `dc_runs/pin_ciw_f666366` and is the only test the feature will get. The measurements below stand.

Written 2026-10-22, redesigned the same day. Implements open question 7 of `EXPERIMENTS.md` §7. **Status: implemented and tested on CPU
(365 passed, 333 of them the pre-existing suite), smoke-tested on real cached libraries. The arm `mt_frozenwt_ciw_r16a16_s1` is queued
behind `mt_frozenwt_r2a4_s3`, not yet run.** Everything is off by default: with the new flags absent every loss and every dataset is what it
was.

The first version (commit `cb49643`) weighted by the interval the assay reports, `deltaG_95CI`. That was the wrong uncertainty (§2) and is
superseded; this document describes the replacement.

## 1. What exists

| flag | default | what it does |
|---|---|---|
| `--ci_weight` | off | weight **every uncensored item, of every subset alike**, by the inverse of its label's variance plus `--ci_weight_tau`², normalised to mean 1 over the training items |
| `--ci_weight_tau` | `0.5` | SD (kcal/mol) of the part of the residual that is the **model's**, added to every item (§3) |
| `--ci_weight_losses` | `both` | `reg`, `rank` or `both`, so a result can be traced to one loss |
| `--no-censor_doubles` | censor | no floor/threshold/gap rule reaches a measured double. **Not used by the queued arm**: a double read below the link's floor knee is then a regression target the link cannot reach (`training.reachable_bounds` exists for this reason). Kept for a stricter second shot |

Code: `src/esm_msr/ci_weights.py` (the uncertainty model and weights), `data.py` (`label_var` stamped on every item at load from `meas`, no cache
change: `--cache_version c5` is untouched), `losses.py` (`ListMLELoss(weights=, offsets=)`), `training.py` (`ci_item_weights`, the composer),
`preprocess_megascale.py` (the normaliser, from the training items), `censoring.py` (`censor_multi`), `config.py`.
Tests: `tests/test_ci_weighting.py`. Evidence: `analysis_scripts/ci_weighting_fit.py` (everything in §2-3), `ci_weighting_sim.py` (§4),
`ci_weighting_stats.py` (the raw interval statistics).

## 2. What the label uncertainty actually is

Every variant has two independent estimates of its stability, one per protease (`deltaG_t`, `deltaG_c`), and `dG_ML` is their joint fit.
I tested each candidate measure of label noise against the **held-out squared error of a trained model** (`mt_frozenwt_r2a4_s1`, epoch 3,
34,957 validation items joined to their raw measurement columns): if a measure is label noise, the model's error should rise with it, with a
coefficient of about 1 once the model's own baseline error is subtracted.

| candidate | what the held-out error does |
|---|---|
| `deltaG_95CI` (the interval; median implied SD 0.035) | flat for 99% of items (MSE 0.245-0.27), rising only in the last 0.5%. Coefficient on CI² in a regression: −0.4 ± 1.5 |
| per-protease fitting error, K50 intervals | flat |
| raw protease gap, \|dG_t − dG_c\| | flat up to 1.2 kcal/mol (MSE 0.27-0.32), because **a whole library shares a protease offset** |
| **library-centred** gap (the gap minus the median signed gap of the library's in-range variants) | rises steadily: MSE **0.21 → 0.26 → 0.29 → 0.35 → 0.52 → 0.72** across its quantile bins (medians 0.06 → 1.22) |

The library offset cancels in `ddG = dG − dG_wt`, so only a variant's own deviation `g` from it is label error. For two independent protease
errors the mean of the readings has variance `g²/4` (E[g²] = 2u², variance of the mean u²/2), so the coefficient on `g²` should be **1/4 with
nothing fitted**. In range (readings 1-4.5) it is **0.263 ± 0.015** (singles 0.256 ± 0.014, doubles 0.226 ± 0.048). That is the whole model:
`L = (CI/3.92)² + g²/4`, and `g` (capped at 3: `dG_ML` is clamped to [−1, 5], so no label is wrong by more than a few kcal/mol) is statistically
independent of the error of the mean for equal variances, so weighting by it does not select on the label's own noise. Weighting by the
**reading** would: a variant truly at the floor whose noise puts it a little higher would count for more, pulling the fit upward.

**Depth below the floor.** For a double the additive expectation `x = dG_wt + ddG_A + ddG_B` comes from other measurements, and the
deviation grows steeply with depth. The mean capped squared deviation `m(x)`:

| x (additive dG) | −4.5 | −3.5 | −2.5 | −1.5 | −0.75 | −0.25 | 0.25 | 0.75 | 1.5 | 2.5 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m(x) | 4.52 | 4.25 | 3.61 | 2.36 | 1.35 | 0.58 | 0.27 | 0.15 | 0.09 | 0.09 | 0.16 |

It holds at a fixed reading too (reading −0.5..0.5: the uncapped deviation² goes 7.8 → 17 → 40 → 58 as x goes from >0 to < −3). On held-out
doubles, error `= 0.272 + 0.678·m(x)` and the item's own deviation adds nothing once `m(x)` is in (−0.03 ± 0.03). So `m(x)` is the prior for a
double's own `g²` (an average of the two: an empirical-Bayes estimate with a prior worth one observation). Singles have no such expectation and
use their own deviation. **What the data do not separate** is label error from model difficulty at depth: held-out error there is 2-3 against a
label variance of 0.25·m ≈ 1, so label error is the minority share of the residual, and the weights use only the label share.

## 3. The weights

`w = [1/(τ² + L)] / mean[1/(τ² + L)]`, mean over the training items. This is the optimal weight of a regression whose residual is label
error plus a model error common to all items; **τ is that model error**, 0.46-0.57 on clean validation items, so the default is 0.5. Smaller
τ discriminates harder. Weights on the real training split (`ci_weighting_fit.py`):

| items | τ | w05 / w50 / w95 | ESS | weighting efficiency | weight share of doubles with x < 0 (count share) |
|---|---|---|---|---|---|
| every item that is not a range bound (207,546) | 0.5 | 0.44 / 1.10 / 1.16 | 95% | 1.19 | 6.6% (11.9%) |
| | 0.25 | 0.18 / 1.13 / 1.39 | 87% | 1.80 | 3.6% |
| **kept as values by the `dthr0` recipe (181,013)** | **0.5** | **0.58 / 1.07 / 1.12** | **97%** | **1.065** | 5.7% (9.3%) |
| | 0.25 | 0.28 / 1.10 / 1.32 | 91% | 1.34 | 3.2% |

*Weighting efficiency* `E[τ²+L]·E[1/(τ²+L)]` is the most estimator variance that optimal weights can remove against uniform ones (1.0 =
nothing). On the items the queued recipe keeps it is **1.065**: at the statistically right τ the weights are nearly flat, and the whole effect is
a soft down-weighting of the ~9% of items that are deep doubles. The hard protease-gap rule would censor only **0.21% of singles and 1.79% of
doubles** among them. See §6 for what that implies.

## 4. Applying the weights to both losses

**Regression: exact.** Inverse-variance weighting is the maximum-likelihood fit under noise that differs per item. The weights multiply the
numerator only (`global_w_sum` stays the sum of the *subset* weights), so a batch of poorly measured items carries less in total and the loss keeps
its scale. A censored item is not weighted: its label is a bound.

**Ranking (ListMLE): approximate, and the obvious way is wrong.** Plackett-Luce models *choices*; a label's noise reaches it through the *order*
it induces, which depends on pairs. Scaling each choice term by its chosen item's weight is measurably harmful (`ci_weighting_sim.py`; 24 seeds,
paired; change in Spearman against the unweighted fit; noise ×3 is a median label SD of 1.5 against a unit signal):

| noise | misfit | regression, weighted | rank: scale the chosen term only | rank: offset only | rank: **offset + term (shipped)** |
|---|---|---|---|---|---|
| ×1 | 0 | +0.0025 | **−0.0107** | +0.0010 | **+0.0021** |
| ×3 | 0 | +0.0216 | **−0.0774** | +0.0035 | **+0.0118** |
| ×3 | 0.5 | +0.0159 | **−0.0867** | +0.0012 | **+0.0099** |
| ×0.25 | 0.5 | −0.0019 | −0.0023 | −0.0012 | −0.0009 |

Each Plackett-Luce term is zero-sum in the scores. Weight only the chosen term and a low-weight item keeps its full share of the negatives from
other items' terms while its own positive term shrinks, so low-weight items are pushed down for no reason the features can explain
(`test_uninformative_labels_exert_no_net_force_on_any_item`: with equal scores and labels that say nothing, the term-only loss pushes the
lowest-weight item with +0.70 and the highest with −1.09; the shipped loss exerts 0). The fix is to offset every score by `log w − mean(log w)`
(a known offset, in numerator and every denominator) **and** scale each term by `w`. The balance holds only if every item that has a term carries
a term weight that is the same multiple of `exp(offset)`; that is why the weighting must be **uniform over the uncensored items** (the only ones
with terms), and why leaving a subset out breaks it (`test_leaving_one_termed_item_unweighted_breaks_the_balance`). Censored members sit in the
denominators at offset 0, the population's typical weight. It is an approximation of a noisy-label likelihood, not the likelihood. The last row is
the regime this data is in: when label noise is small against model error, no weighting helps, and a hand-set strength hurts slightly.

## 5. The arm: `mt_frozenwt_ciw_r16a16_s1`

The `mt_frozenwt_dthr0_r16a16_s1` recipe (frozen best-epoch WT of `wt_r2a4_f0_cthr05_s1`, MT-only, rank 16 / alpha 16, doubles floor-bounded at
≤ 0, singles at ≤ 0.5, seed 1) with the hard protease-gap censoring replaced by the weights: `--no-censor_gap --ci_weight --ci_weight_tau 0.5`.
Its parsed arguments differ from `dthr0`'s in exactly `censor_gap`, `ci_weight` and the experiment name. **Reading "instead of censoring" this
way is my interpretation**; the floor bounds are kept because below the link's floor knee a value is an unreachable target, and the hinge there
is the same as a clamped value for the regression (§6).

Same seed and WT head as `dthr0_r16a16_s1`, so the comparison is paired (identical data order: no item is added or dropped). Code runs from a
detached, locked worktree pinned at the commit (`dc_runs/pin_ciw_<hash>`), so nothing edited later can change it while it waits.
Read the epistasis hierarchy (`cell_rank_mt`, `cell_mag_comb`, `cell_flipacc_mt`) with the guards `val_rho_wt_valid_avg` (fixed) and
`val_rmse_combined_avg`. The log carries `train/ci_w_mean` (≈ 1) and, at start-up, the weight distribution and the weighting efficiency. One
seed is screening only (`STATE.md` §5: a single-seed difference needs ≈ 0.02 on `cell_rank_mt`).

## 6. Reflection: can weighting beat censoring?

* **At the floor, censoring is already the right model, and weighting cannot improve on it.** For the regression, a hinge bounded at the link's
  floor knee is a squared error to the knee (the link cannot predict below it), i.e. the same as a clamped value; there is nothing for a weight to
  add. For the ranking, the order *within* the dead class is noise (protease rank agreement 0.20 below 0), and censoring ties it exactly, whereas
  weights can only make the noisy order count less. Replacing the floor bound by weights is at best neutral and at worst puts noise into the rank
  loss; that is why the queued arm keeps it.
* **Outside the floor the label noise is small against the model's, so there is little to exploit.** In the retained set the efficiency bound is
  1.065 at the right τ (1.34 at the aggressive τ = 0.25): optimal weights could remove at most 6% (25%) of the variance, equivalent to 6% (34%)
  more data, and the realistic fraction of that is smaller. The hard gap rule touches 0.2% / 1.8% of singles / doubles. A paired single-seed
  run cannot see an effect of that size (the noise floor is ≈ 0.02 on `cell_rank_mt`).
* **I expected soft censoring of the boundary band to be the better lever, tested the premise offline, and it does not hold up.** The idea was that
  the hard threshold lumps two populations in the (0, 0.5] reading band: 4,437 training doubles (8.4% of all doubles) that read slightly "alive" while
  their additive expectation says dead, and 5,026 that are plausible low values, so a per-item rule (or a Tobit-style posterior) should beat one global
  cut. `analysis_scripts/censoring_band_check.py` reproduces three checks:
  1. **The `dthr0` gain is not established.** Paired at seed 1 and matched epoch 3 (`mt_frozenwt_r16a16_s1` against `mt_frozenwt_dthr0_r16a16_s1`):
     `cell_rank_mt` +0.008 (about one SD; −0.004, +0.001, +0.015 at epochs 0-2), `rho_combined` −0.0006, `rmse_combined` +0.0016, `cell_flipacc` +0.043
     (about 1.5 SD). I had read a larger gain off a cross-seed comparison; it was wrong to say `dthr0` "already shows it".
  2. **Where the two models differ, there is no rank signal.** On the 8,494 validation doubles, 55% of the total squared-error change is in the
     expected-dead band (mse 2.29 → 1.99) and 48% in the (0.5, 1.5] reading band, but the within-library Spearman of prediction against label in the
     (0, 0.5] and ≤ 0 groups is ≈ 0 for both models (−0.010 → −0.006, 0.082 → 0.071, −0.060 → −0.042) and unchanged. The model that treats the band as
     values fits its piled-up readings better; it does not rank anything better.
  3. **The boundary decision is insensitive elsewhere too.** The WT-only sweep of the singles' threshold (0.3 / 0.5 / 0.6, seed 2) gives best
     `val_rho_wt_valid_avg` 0.8460 / 0.8459 / 0.8466.
  The protease agreement that does exist near the floor is mostly dead-versus-alive separation (rank agreement of the two proteases' readings,
  grouped by the exogenous expectation x: 0.18 for x < −3, 0.56 for x in (−2, −1], 0.86 above 1), which is what bounds and ties already extract.
  So a soft or per-item rule would reclassify mainly the 4.4k expected-dead doubles, which carry no rank signal; I do **not** expect it to move the
  epistasis metrics and have not queued it. It stays a hypothesis for a different kind of evidence (for example a triplicate showing `dthr0` itself
  helps, followed by a reg-only versus rank-only split of the threshold). The singles' threshold has the same property.
* **My expectation for the queued arm is a null result within the seed noise**, with the deep-double share of the loss moving from 9.3% to 5.7%.
  A null is still informative: it says the label noise outside the floor is not what limits the epistasis metrics.

## 7. Known simplifications

* The weight uses the double's own deviation and its depth prior; with teacher forcing (`--tf_mix 1.0`) its prediction also inherits the noise of
  the two singles' labels (`¼(σ_A² + σ_B²)` on top). Adding it needs the singles' readings on the double; at these variances it moves the weights
  little.
* `τ` is the model's residual on clean validation items from one trained head; a better head has a smaller τ and would warrant sharper weights.
* The depth prior `m(x)` is a tabulated mean fitted on the training split, shipped as constants in `ci_weights.py`; `ci_weighting_fit.py`
  refits it and prints the difference (zero at the time of writing).
* Replicate rows are collapsed by taking the mean of the interval widths (`preprocess_megascale._collapse_replicates`), which slightly overstates
  a replicated variant's interval (rare; the interval is a small part of `L`).
