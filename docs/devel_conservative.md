# `devel_conservative`

Branched from `main` (`c34c315`), not from `devel`. It adds one thing — an objective that knows where each library's
assay floor is — plus the data that objective needs, and deliberately nothing else. `devel` could not beat `main` in
any configuration, so the starting point is `main` and every addition has to justify itself on its own.

## 1. What changed against `main`

| # | Change | Where |
|---|--------|-------|
| 1 | 2D subsampling of doubles deleted (it was **on by default**) | `data.py` |
| 2 | `reversion` gone; `mut_ctx` → `cond`; `native_cond` added; every item carries its raw measurement row | `data.py` |
| 3 | Per-library floor, accepted only on evidence of pinning; `Nth_percentile_dG` kept as a diagnostic | `data.py` |
| 4 | Floor link `dG_obs = y_floor + τ·softplus((dG_latent − y_floor)/τ)`, τ=0.59 | `link.py` (new) |
| 5 | Hinge regression: a sub-floor measurement is a bound, not a value | `losses.py` |
| 6 | Censored ranking: sub-floor items tie at the bottom, their mutual order unscored | `losses.py` |
| 7 | No non-specific epistasis term; no epistasis loss at all | `config.py`, `training.py` |
| 8 | `devel`'s flat `compute_metrics` + the full `epi_metrics` hierarchy (`epi_cell_*`, `beyond_rho`, …) | `stats.py`, `epi_metrics.py`, `training.py` |
| 9 | Dead args and dead code removed; no backwards compatibility | everywhere |

### The 2D subsampler (item 1)

`ProteinCyclingBatchSampler` defaulted to `subset_balance_configs = {'double': {bins: 15, cap_percentile: 75,
missing_cap_fraction: 0.20}}`, so **every run on `main` was training on 2D-subsampled doubles** unless it passed
something else. It binned doubles over (additive expectation, dddG) and capped each bin at the 75th percentile of
populated-bin counts. That reshapes the ddG distribution the model sees: it preferentially discards the common,
well-measured, in-range doubles and keeps the sparse bins, which are mostly the saturated ones. Deleted outright.
Proportional per-subset capping (`--subset_caps`) stays, since it mixes subsets rather than reshaping one.

### Subsets (item 2)

`mut_ctx` was **not** "almost the same as `native_cond`". It is the same as `devel`'s `cond`: the conditional effect
of one mutation of a double given the other, `ddG_AB − ddG_partner`, derived arithmetic, two items per double.
`native_cond` is a different and genuinely *measured* thing: a single in a mutant-background library (`1A0N_L7S`),
which is already a conditional effect `ddG(X | L7S)`. Both now exist under `devel`'s names; `_label_native_cond`
assigns the latter at load time from the library code, so one cache serves runs that want them and runs that do not.

Reversions are gone: their before-state is a mutant sequence, which violates "the WT pass sees the real wild type",
and their after-state is the wild type, which is the WT pass's job already.

`cond` items reuse their parent double's structure (the unchanged WT backbone). The alternative — loading or masking
and re-encoding a structure of the partner mutation per item — measured ~0.4 s/item, about **50 hours** over the full
dataset. The partner's side chain is shown as wild type, which is wrong, but it is what the MT adapter is given at
inference time, so training and inference agree.

Each item now carries `meas`, a 24-column float32 array of the raw row: `dG_ML`, `ddG_ML`, the two-protease
`deltaG` and the per-protease `deltaG_t` / `deltaG_c`, all four 95% CI columns for each, both `log10_K50`s and their
unfolded references, and the two fitting errors. Read it with `data.meas_value(item, 'deltaG_t_95CI')`.

One subtlety this forced: replicate collapsing used to group by "every column except `ddG_ML`", which was the same as
grouping by the identity columns while only four columns were kept — but silently stops collapsing replicates once
the confidence intervals ride along, because replicates differ in those. `_collapse_replicates` now names the
identity columns explicitly and averages every measurement column.

### The link (item 4)

```
dG_obs = h(dG_latent),   h(z) = y_floor + τ · log(1 + exp((z − y_floor)/τ)),   dG_latent = dG_wt + latent
```

Softplus with its knee at the library's own floor: increasing everywhere, asymptotically the identity above the
floor, asymptotically flat at `y_floor` below it. Regression goes through it; ranking does not, because it is
monotone, so the latent scale stays unsaturated and can say a double is 7 kcal/mol down even where the assay can
only report the floor.

`τ` defaults to 0.59 kcal/mol and is **fixed** unless `--learn_link_tau`. `y_floor` is **data, not a parameter** —
each item carries its library's estimate, so the link adds no per-library parameters. The estimate is clipped into
`[--link_floor_min, --link_floor_max]` = `[-1.0, 0.2]`, the band the assay's reported floor occupies.

**Where it is applied.** One rule, everywhere: *every prediction is assembled on the latent scale, and the link is
applied once, at the very end. Labels are never transformed.*

| path | assembly (latent) | then |
|---|---|---|
| WT, single | the pass's LLR for the one mutation | `h` → compare to measured `ddG_X` |
| WT, double | the pass's LLR **sum** over both positions | `h` → compare to measured `ddG_AB` |
| combined | `0.5·WT_latent + 0.5·MT_latent` | `h` → compare to measured `ddG_AB` |

So the WT head is regressed onto the item's own measurement whatever the item is. The architecture, not the label,
is what makes it additive: the WT pass can only emit a sum of per-position terms, so it cannot represent
pair-specific epistasis even when trained against a double's measurement — the residual stays residual, and with
many doubles per single the pull on each single term averages out.

This replaced an earlier target of `ddG_A + ddG_B` for doubles, which could not survive the link. Each term of that
sum is an observed value, but **their sum is not anything the readout can report**: two singles each measured at
−2.5 give a label of −5.0, while a linked prediction is floored near `y_floor − dG_wt` ≈ −3.3. No prediction could
ever match it, and the gradient would drive the latent down without limit. The two self-consistent options were
"latent prediction vs sum-of-singles" (the old behaviour) or "linked prediction vs the measured double" (now); the
mixture of the two is the one combination that is broken. The old target was also the worse of the two even without
the link, because `ddG_A` is the *floored* value of single A, so for any floored single it taught the WT head a
compressed latent — exactly the contamination the link exists to remove.

Items whose library has no measured `dG_wt` (the external benchmarks) leave the linked regression and keep their
rank terms, which the link would not have changed anyway, since it is monotone.

**One residual impurity, handled.** In `ensemble` mode the WT half of the combined latent is teacher-forced with the
measured `ddG_A + ddG_B` so the MT head only has to learn the correction. That injects an *observed* sum where a
*latent* belongs. It is harmless while both singles sit inside the dynamic range (`h` is near-identity there, so
observed ≈ latent), and wrong exactly when a single is itself floored. Those doubles are therefore excluded from
teacher forcing and fall back to the WT head's own latent (`train/tf_saturated_dropped` counts them per batch).

There is **no ceiling term**. The `>5` hyperstable end has no numeric value to regress onto, so a second soft clamp
would be fitting a shape to data that is not there.

### Hinge and censored ranking (items 5, 6)

An item is censored when its measured dG is at or below its library's (clipped) floor. Then:

* **Regression** scores it against its bound and *only when the prediction is on the wrong side* — predicting "at or
  below the floor" for something measured at the floor is free, and only claiming it is *better* than the floor costs
  anything (`losses.hinge_regression_loss`).
* **Ranking** ties it at the bottom of its list, keeping it in the Plackett-Luce denominators so the model still
  learns it ranks below the measurable variants, while never scoring its order against its equally-dead peers
  (`ListMLELoss.forward_censored`).

Bounds are stored on each item's own ddG scale, derived as `dG_bound + (ddG − dG_meas)`, so one library floor serves
singles, doubles and conditionals alike: a double measured at the floor bounds `ddG_AB` at `y_floor − dG_wt`, and the
`ddG(A|B)` derived from it is bounded `ddG_B` higher.

One knob controls both: `--no-censor_floor` censors nothing and the losses degrade to the uncensored forms.

### Validation (item 8)

`stats.compute_metrics` is `devel`'s: a flat dict, with `rho_wt_valid` / `rho_wt_all` / `rho_mt_valid` /
`rho_mt_all` / `rho_combined` / `rmse_combined`, NDCG for both heads, and all of them computed on **uncensored items
only** so a library's score is not inflated by variants whose measured value is only its floor. The censored items
get their own metrics — `auc_dead_{wt,mt}`, the probability a measurable variant is ranked above a floored one —
plus `frac_censored`. `rmse_combined` is computed on the observed scale when the link is on, so it stays comparable
with a run that has none. `--val_dump` writes per-item predictions to `val_dump_<loader>.npz`.

`val_rho_combined_avg`, the checkpoint monitor, is unchanged in name and meaning.

## 2. Against `claude/main-baseline`

`main-baseline` is the released objective plus an optional link. The overlap is real but the link is a different
object:

| | `main-baseline` | here |
|---|---|---|
| Link shape | soft floor **and** soft ceiling (`MonotoneLink`) | soft floor only (`FloorLink`) |
| Knee position | learned global `lo`, plus a per-library **offset** read from `--link_floor_file` (JSON) | the library's own floor, cached per item; no parameters |
| `τ` | one learned `tau_lo`/`tau_hi` | fixed 0.59, learnable on request |
| Floor estimator | `baseline/estimate_floors.py`: median dG of doubles whose additive expectation < −1.5, ≥20 of them, shrunk toward a global median with 5 pseudo-counts | `Nth_percentile_dG`, N=5 (the doubles-conditioned figure is *recorded* as `floor_cond`, not used) |
| Censoring | none | the floor drives hinge regression and censored ranking |
| Teacher forcing | `--tf_singles latent_link` puts measured singles back on the latent scale via `h⁻¹` | not carried over; no inverse is needed |
| Non-specific epistasis | `--nonspecific_term fx` (`NonspecificTerm`, piecewise-linear f(x)) | absent (item 7) |
| Epistasis metrics | `epi_metrics.py` (464 lines) | absent |

`main-baseline` also carries `--val_cycle_passes`, `--premask_coords`, `--use_plddt`, `--resume_global_step`,
`--detach_calibration`, `--invert_list_loss`, `--lambda_epi_combined`, `--zero_epistasis_for_singles`,
`--incl_reversions` and `--incl_mut_ctx`, none of which exist here.

## 3. Against `devel`

`devel` is a much larger program: +14,807 lines over `main`, with an identity-dependent interaction objective
(`flipkeys.py`, flip columns, packed pair matrices), MT-side regression and anchor singles, automatic batch sizing
(`auto_batch.py`), and epistasis metrics. What this branch took from it:

* the `cond` / `native_cond` item construction and names, and `cond`'s structure-reuse default;
* `routing.py`, the subset→head taxonomy (trimmed: no `reversion`, no legacy aliases);
* the *idea* of `censoring.py`, and `ListMLELoss.forward_censored` essentially verbatim;
* the flat `compute_metrics` and `safe_rmse`.

What it deliberately did not take (item 7): `--lambda_mt_flip`, `--lambda_mt_cell`, `--lambda_mt_colrank`,
`--flip_delta/_scale/_list_min`, `--mt_single_anchor_weight/_frac`, `--lambda_rank_mt`, `--lambda_reg_mt`,
`--mt_comp_*`, `--pack_pair_matrices`, `--reg_balance`, `--mask_structure`, `--mask_mutated_structure`,
`--premask_mt_structure`, `--wt_early_stop_patience`, `--wt_list_size`, `auto_batch.py`, `epi_metrics.py`,
`flipkeys.py`.

Two differences worth naming because they look like the same feature but are not:

* **`devel`'s link is the two-sided one**, like `main-baseline`'s, with no per-library floor. The per-library floor
  is `main-baseline`'s contribution; the floor-as-cached-data design is new here.
* **`devel` censors on a global constant** (`--censor_floor F`, applied only to flip-column items). Here the
  threshold is each library's own floor and applies to every subset.

`devel`'s `delta_single_diagnostics` and `flip_signature_rho` are not ported: the first needs inference-output
columns this branch does not produce, the second needs the flip keys of the interaction objective.

## 4. Is the 5th percentile a good floor estimate?

**Short answer: it is a usable proxy and the clip band keeps it safe, but it is not estimating the floor — it is
estimating a quantile of the library's own sensitivity, and those two things come apart badly in exactly the
libraries where the floor matters least.** I would keep it as the default only until the doubles-conditioned
estimator can be wired in; the ingredients for the swap are already cached.

**What the floor actually is.** The assay infers dG by comparing a variant's measured K50 against a K50 predicted
for its *unfolded* sequence from composition. A variant that is unfolded anyway has K50 equal to its true unfolded
K50, so its inferred dG is the error of that composition prediction for that sequence — a constant of the library and
of the protease. That constant is the floor. It is a property of the readout, not of the variants.

**Why a quantile is the wrong shape for it.** A quantile of the measured dG distribution only lands on the floor if
enough of that distribution is actually sitting on the floor. So the estimate is confounded with how destabilising
the library's variants happen to be. Measured on this dataset, during the cache build:

* `1A32`: 5th percentile = **+1.005** kcal/mol. Clipped to +0.2. Nothing in that library is anywhere near a floor;
  its mildest 5% are simply healthy proteins.
* `2N88`: 5th percentile = **−0.331**, doubles-conditioned = **+0.020**.
* `2L7F`: 5th percentile = **−0.351**, doubles-conditioned = **+0.346** — a 0.70 kcal/mol disagreement.
* `HHH_rd4_0676`: 5th percentile = **+0.236**, clipped to +0.2, and no deep doubles at all to check it against.

`test_floor_censoring.py::test_estimate_tracks_library_composition_not_only_the_assay` pins this behaviour so it
cannot change silently: two libraries with the *same* true floor of −1 get estimates of ">1.0" and "−1.0" purely
from their variant composition.

**Across the whole dataset** (from the cache build, `floor estimates (...)` in the run log):

| split | libraries | median | IQR | range | clipped into the band | vs doubles-conditioned |
|---|---|---|---|---|---|---|
| train | 170/170 | +0.052 | [−0.336, +0.931] | [−0.742, **+3.347**] | **74 of 170 (44%)** | median Δ −0.312, SD 0.300, r = **0.40** (n=45) |
| val | 25/25 | +0.350 | [+0.139, +0.579] | [−0.229, +1.785] | **17 of 25 (68%)** | median Δ −0.114, SD 0.148, r = 0.56 (n=5) |

Three things follow from that table, and together they are the case for replacing the estimator:

1. **The clip band is load-bearing for half the dataset.** Nothing fell below −1.0, so every one of those 74 + 17
   libraries was clipped at the *top*, to exactly +0.2. For 44% of training libraries and 68% of validation
   libraries the "per-library floor" is therefore not per-library at all — it is the constant +0.2. The upper
   quartile of the raw estimate is +0.93 and the worst is +3.35, which is not a floor under any reading.
2. **Where there IS a floor population, the percentile lands below it.** On the 45 libraries with enough deep
   doubles to check, the percentile sits a median 0.31 kcal/mol *below* the doubles-conditioned estimate. That is
   the expected behaviour of a 5th-percentile statistic applied to a pile-up: it reports the noise tail beneath the
   floor rather than the floor's centre, so the knee goes too low and the censoring under-fires.
3. **The two estimators correlate only 0.40.** The percentile is carrying limited information about the quantity it
   is standing in for, and the two biases run in opposite directions — too high in mild libraries, too low in
   saturated ones — so no single choice of N fixes both.

**The structural objection.** A fixed Nth percentile censors about N% of every library *by construction*, whatever
the data says. The fraction of the library declared unidentifiable is set by the estimator, not discovered. The clip
band is what rescues this in practice — where the percentile lands above +0.2 it is pulled down and the censored
fraction collapses (`1A32`: 7 of 1195 items, 0.6%) — but where the percentile lands *inside* the band while still
being above the true floor, ~5% of good singles get censored anyway (`1AOY`: `y_floor` = 0.126, 65 of 1287 items, all
singles). Those are measurements the hinge then refuses to regress onto.

**Two smaller points.** The censoring tags (`<-1`) carry no number and are excluded from the percentile, so it is the
floor of the population the regression loss actually sees; `frac_cens_lo` is recorded per library because a library
that reports most of its dead variants as `<-1` has a truncated numeric tail and therefore a percentile biased
upwards. (In the libraries inspected so far this is 0.0%, so it is not currently biting.) And the clipped floor, not
the raw estimate, is what *both* the link's knee and the censoring rule use — they disagreed in the first draft,
which is how the `1A32` case surfaced.

### The two-protease disagreement

The assay fits dG twice, once per protease. Two independent estimates of one variant's stability means a
per-variant, doubles-free consistency check — which is what the floor estimate is really missing, since only 63 of
479 libraries contain enough deeply destabilised doubles to locate a floor at all.

**The obvious version does not work.** Normalising the gap by its confidence intervals (`z = |dG_t − dG_c| / σ`)
flags *in-range* variants hardest, because the reported CIs are tiny (median 95% width 0.147 kcal/mol) while
genuine protease-specific differences are ~0.3:

| mean of the two channels | median \|t−c\| | median z | fraction z>3 |
|---|---|---|---|
| −0.5 … 0.0 | 0.553 | 2.08 | 36% |
| 1.0 … 2.0 | **0.312** | **6.28** | 74% |
| 2.0 … 3.0 | 0.323 | 8.58 | 80% |
| > 5.0 | 0.740 | 1.71 | 21% |

The *absolute* gap is U-shaped as the mechanism predicts (0.55 at the bottom, 0.31 mid-range, 0.74 at the top), but
in σ units the systematics swamp it. Likewise `deltaG_95CI` is useless as an unidentifiability flag: **zero**
numeric rows have a CI wider than 1 kcal/mol.

**The gross gap does work, as an identifier.** Among the numeric rows training actually sees, `|t−c| > 2` flags
2.53%, and 92% of those sit below dG 0.5 (median dG of the flagged −0.37). The population is bimodal — thresholds
of 2 and 3 select 17.7k and 15.3k rows — so the cut is not delicate. For reference, the rows the assay already
refuses to score have a median gap of **13.1** and a maxed-out CI of 20.0, so the signal is unambiguous where we
can check it against the assay's own verdict.

**But it is a biased locator.** Per-library floors built from the gap-flagged variants have excellent coverage (237
libraries at n≥10, against 63 for the doubles-conditioned estimator; 183 libraries reachable *only* this way) and
real per-library signal (**split-half r = 0.912**). They are nonetheless **0.39 kcal/mol below** the
doubles-conditioned estimate (r = 0.62): gap-based median −0.489 against −0.068. The doubles-conditioned figure is
the trustworthy one, because it conditions on the *prediction* that a variant should be dead, whereas the gap
criterion correlates with the fit having landed low — and −0.068 is what the independent published estimate of the
floor (+0.02, SD 0.31) says it should be.

**So the two signals do different jobs, and both are used:**

| question | signal | why |
|---|---|---|
| *which* items are unidentifiable | the two-protease gap (`--censor_gap_max 2.0`) | per-variant, needs no doubles and no floor; reaches every library |
| *where* the link's knee sits | the doubles-conditioned floor, else the dataset median | conditions on prediction not outcome, so it is unbiased; agrees with the published floor |

A gap-censored item is bounded by **its own reported value** — a fit that cannot identify a stability still bounds
it from above — so censoring no longer depends on having located a floor at all. The two sources are a union, not
alternatives: the gap rule is specific but not sensitive (it catches only ~15% of the rows below dG 0.5), while the
floor rule is sensitive within the libraries where a floor is known. Measured on the generated cache the gap rule
censors 0.98% of items overall, 0–4.8% per library, and correctly censors **nothing** in `1A32` — the library whose
5th percentile was +1.005 because it contains nothing destabilised at all.

**Refinements, in the order I would try them.**

1. **Condition on being dead instead of assuming the tail is dead.** Use `floor_cond` — the median measured dG of
   doubles whose additive expectation `dG_wt + ddG_A + ddG_B` is below −1.5 — where at least ~20 such doubles exist,
   and shrink toward the global median otherwise. This is `main-baseline`'s estimator, it has measured reliability
   there (split-half 0.97, SE ≈ 0.05 against a between-library spread of 0.31), and it estimates the floor itself
   rather than a quantile of sensitivity. It is already computed and cached per library as `floor_cond` with its `n`,
   so this is a one-line change of which field feeds `y_floor` plus a shrinkage step — **no cache regeneration**.
2. **Make the censoring threshold a margin, not an equality.** Censor on `dG ≤ y_floor + δ` with δ from the item's
   own `deltaG_95CI`, which is now cached. A measurement 0.05 above a floor estimated with SE 0.05 is not meaningfully
   distinguishable from it, and conversely a precisely measured 0.3 above the floor should not be censored just
   because the library's quantile landed high.
3. **Use the two-protease disagreement as a deadness signal.** `deltaG_t` and `deltaG_c` are both cached. A variant
   at the floor should pin in both proteases and its two estimates should agree at the (library-specific) unfolded
   baseline; a variant with a genuine low-but-real dG need not. This is an independent check on any floor estimate
   and does not depend on the library having deep doubles.
4. **Only trust a floor where there is a floor population.** `n_floor_cond`, `frac_cens_lo` and the percentile's
   distance from the band are all cached; a library with no evidence of any pinning should arguably get no floor
   censoring at all rather than the band's edge.

## 5. Known gaps

* `cond` and `native_cond` are generated and cached but **not trained on** (`--subset_caps cond=0.0 native_cond=0.0`)
  — the MT-head routing for them is not built yet. Note that excluding `native_cond` is a deliberate departure from
  the released model, which trained on those items because they were labelled `single` there. They remain in
  validation, where they are direct measurements, so `val_rho_combined` stays comparable.
* The out-of-range rows (`<-1`, `>5`) are still dropped upstream by `preprocess_megascale`, so `cens_src = SRC_RANGE`
  is never set. Admitting them is a change to the training population, not just to the loss; the censoring and loss
  machinery is two-sided throughout so that doing it later needs no rework.
* The cache is ~11 GB (vs 6 GB for `cache_main_capped`) because `cond` items each carry their own copy of their
  library's structure tensors. Deduplicating the structure per library would fix most of that and is orthogonal.
