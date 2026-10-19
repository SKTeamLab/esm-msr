Tsuboyama 2023 · MegaScale · 131,143 double mutants

# What the double-mutant data can and cannot tell you about epistasis

An audit of the thermodynamic coupling values derived from the MegaScale dataset. The published ΔΔG values are robust. The problem with derived epistasis is not noise but a systematic bias, concentrated in the fifth of double mutants whose stability the assay cannot reach, where apparent stabilizing coupling of several kcal/mol is manufactured by the measurement.

0.93reliability of ΔΔG_ML itself, from duplicate measurements

0.80reliability of dddG derived from it

+2.06kcal/mol mean dddG where the assay cannot reach — a bias, not noise

0.47kcal/mol reproducible residue-specific coupling

## 01Where ΔG stops being a measurement

Stability is inferred from K50, the protease concentration at which a domain is half digested, read against a sequence-specific prediction for the fully unfolded state (K50,U). When a domain is unfolded, K50 collapses onto K50,U and ΔG is no longer identifiable. The authors say so plainly:

“ΔG values become unreliable if K50 approaches K50,F or K50,U … ΔG becomes very sensitive to K50 and its uncertainty increases relative to the uncertainty in K50.”

mean ΔG (kcal/mol)median 95% CI on ΔG, worst protease

**ΔG is a function of the K50 gap, and the gap closing destroys the estimate.** Binned over all 700k measured variants. Where log₁₀K50 − log₁₀K50,U falls below about 0.2, the reported confidence interval runs to tens of kcal/mol — the fit is unconstrained, not merely noisy. The nominal −1 kcal/mol bound is never the operative limit.

Worked example — 2KVT

Wild type sits at ΔG 4.0–4.2 with a K50 gap of 2.5–3.1 and CIs near 0.1. The singles S35L and H37K have gaps of 0.10 and 0.15, CIs of 0.44–0.66, and reported ΔG of −0.61 and −0.11. The double S35L:H37K has a gap of 0.20 and ΔG +0.07 — nominally more folded than either single.

Subtracting these yields dddG +4.95 kcal/mol of apparent stabilizing epistasis. All three states are unfolded and all three ΔG values are unconstrained. Every value is numeric and passes the authors' tabulated filters.

## 02The authors filtered; the filter does not cover this

21.7% of ΔΔG entries are withheld as "-", but for a different reason: those are libraries whose *wild type* fell outside the dynamic range, and their mean ΔG is *higher* (2.69) than that of retained rows (1.76). Variants that individually became unmeasurable keep their numeric ΔΔG.

The authors also publish a confidence criterion — a 95% CI above 0.5 kcal/mol marks a low-confidence ΔG. Two practical traps follow.

- The *combined* deltaG_95CI is anti-conservative. For the 2KVT singles it reads 0.16 while each protease's own CI is 0.44–0.46: averaging two independently unconstrained fits narrows the interval without adding information. Filtering on it removes 2% of doubles and changes nothing.
- The per-protease CIs are the informative ones, and nothing downstream is obliged to use them. A pipeline that reads only ddG_ML inherits every unidentifiable measurement silently.

## 03Two ways to ask how good the labels are, and they disagree

ΔG is fit twice, independently, from trypsin and chymotrypsin, so dddG can be computed twice and compared. That comparison is brutal: Pearson 0.13 unfiltered, rising to 0.83 once variants with a wide per-protease interval are removed.

Pearson rSpearman ρfraction of doubles kept

**Agreement between the two single-protease estimates of dddG.** Threshold is the worst per-protease 95% CI across the double and both of its singles. Pearson climbs from 0.13 to 0.83 by discarding 12% of the data; Spearman starts at 0.69, so ranks were never as broken as the magnitudes.

**The excluded variants carry no reproducible single-protease coupling.** dddG from trypsin against dddG from chymotrypsin; axes matched. Discarded variants (left) scatter to ±30 kcal/mol at r = 0.05; retained variants (right) sit on the diagonal at r = 0.83.

But that is the wrong label to judge

Those per-protease values are reconstructed from one protease each and are unbounded, which is why they reach 30 kcal/mol. The value actually published and trained on, ddG_ML, pools both proteases in a bounded fit, so its error is bounded too: the largest dddG built from it is 6.9 kcal/mol, not 33.

About 0.8% of variants were measured twice, which gives the trained label's reliability with no modelling assumption at all: **0.93, with 0.24 kcal/mol of noise**. Propagating that through a three-term difference puts dddG at reliability **0.80** — far from the 0.13 the surrogate suggests.

So the pessimistic reading is an artifact of the surrogate. Filtering on the per-protease interval moves the trained labels only a little:

| Trained label | terms | σ obs | σ noise | reliability now | filtered | ceiling now → filtered |
| --- | --- | --- | --- | --- | --- | --- |
| ΔΔG, single mutant | 1 | 1.02 | 0.24 | 0.945 | 0.955 | 0.972 → 0.977 |
| ΔΔG, double mutant | 1 | 1.08 | 0.24 | 0.950 | 0.961 | 0.975 → 0.980 |
| ΔΔG(A\|B) = AB − B | 2 | 0.92 | 0.34 | 0.864 | 0.894 | 0.930 → 0.945 |
| dddG = AB − A − B | 3 | 0.92 | 0.42 | 0.798 | 0.829 | 0.893 → 0.911 |

Noise is the duplicate-measured 0.24 kcal/mol (0.20 among retained variants), propagated as √k for a k-term difference. The attenuation ceiling — the best correlation any model can reach against a label this noisy — moves by one or two points. Label noise is not what limits a model here.

## 04The real defect is bias, which reliability cannot detect

Two replicates of an unidentifiable fit agree with each other while both are pinned to the same wrong value. Replicate agreement among the variants a CI filter would drop is still 0.78 — respectable — because their errors are shared, not random. What gives them away is that their epistasis is impossible.

| Additive prediction for the double | n | share | mean dddG | dropped by CI ≤ 1.0 |
| --- | --- | --- | --- | --- |
| ΔG > +0.5 — comfortably measurable | 55,605 | 42% | +0.33 | 1% |
| ΔG in \[−1, +0.5\] | 48,384 | 37% | +0.89 | 16% |
| ΔG \< −1 — the assay cannot reach it | 27,154 | 21% | **+2.06** | 51% |

Mean coupling rises six-fold as the predicted double moves out of the measurable range. Two surface substitutions do not stabilise a domain by 2 kcal/mol on average; this is the measurement reporting a folded-state ΔG for an unfolded protein. Note the last column: the CI filter catches only half of this group, so it is a partial proxy. A criterion on the predicted ΔG itself targets the population directly.

## 04How much epistasis is actually there

On the retained 108,610 doubles, the observed dddG spread is 0.84 kcal/mol. Splitting it three ways: fit a stability-only model — dddG from the additive expectation, the wild-type ΔG, and the two single effects, with whole domains held out — and treat its residual as candidate residue-specific coupling, then ask how much of that residual reproduces across proteases.

global (stability-only)residue-specific, reproduciblemeasurement noise

**Just over half of epistasis is a smooth function of stability; a real specific component remains.** Variance decomposition of dddG. The stability-only model reaches held-out R² = 0.54 without seeing which residues are involved. Its residual has spread 0.57 and reproduces at r = 0.68 across proteases, leaving **0.47 kcal/mol** of coupling that no function of total stability can explain. The authors independently report typical couplings of 0.5–1.0 kcal/mol.

**The diminishing-returns trend is real, and it is not the artifact.** Measured ΔG of the double against its additive prediction, with the interquartile band. The curve flattens as the prediction becomes destabilizing — but the slope of dddG against the additive expectation survives every filter at about −0.4, so the flattening is reproducible physics (a stability threshold), not only a failure of measurement. Points below roughly −1 on the x-axis are predictions the assay cannot test at all.

## 05What a per-protease CI filter is actually worth

Adding the filter is still the right call, but for the second reason rather than the first, and the expected effect is specific.

- **Variance: almost nothing.** The attenuation ceiling for single-mutant ΔΔG moves 0.972 → 0.977, and for the conditional target 0.930 → 0.945. A validation Spearman near 0.82 is far below either, so the labels are not the binding constraint.
- **Bias: this is the gain.** It removes half of the 21% of doubles whose stability the assay cannot reach, where mean coupling is inflated to +2.06 kcal/mol. Those items are 17% of rows but carry 27–34% of the squared-label mass that a regression objective chases.
- **Cost: a truncated tail.** The filter is not random with respect to the label. It drops 40% of the most destabilizing decile of doubles against 3% of the mildest, so the model sees a narrower range of destabilization than it will be asked about at screening time.
- **Coverage: 15 of 153 libraries lose more than half their doubles**, and 7 are left with fewer than 20 (up from 2), which makes those libraries' per-library metrics noisier. No library loses all of them.

A criterion on the predicted ΔG of the double targets the biased population directly rather than at 51% efficiency, and discards fewer well-measured destabilizing variants. The two are complementary: the interval catches variants that failed for other reasons, the range criterion catches variants that were never answerable.

## 06Choosing the threshold, and retiring the density cap

The bounded fit cannot report a ΔG below −1 kcal/mol: the measured distribution has **zero mass** there, for singles and doubles alike. Meanwhile 21% of doubles have an additive prediction below −1. For that fifth the reported value is not uncertain, it is arithmetically obliged to be too high, and the error lands in dddG as apparent stabilising coupling.

**Coupling is flat where the double is measurable and rises steeply where it is not.** Mean dddG with its interquartile band, against where the additive prediction puts the double. Right of about +0.5 the curve is level near +0.3 kcal/mol. Left of the −1 floor it climbs past +2. Nothing physical distinguishes those pairs; the assay simply stops resolving them.

mean dddG (bias + selection)slope of dddG vs additivedoubles kept

**The slope plateaus; the mean does not.** Sweeping a minimum on the additive ΔG. The slope against the additive expectation falls from −0.43 to about −0.23 and then stops moving — that residual is the genuine diminishing-returns trend, and the excess above it was measurement. Mean dddG keeps falling past the plateau because a stricter threshold also selects milder mutation pairs, so it mixes bias with selection and should not be used to pick the cut.

mean dddGshare still unreachable

**Density capping discards a third of the data and leaves the bias worse than doing nothing.** Clipped pairs sit in the sparse tail of the (additive, dddG) histogram, so per-bin capping thins the dense well-measured core and spares them: the unreachable share rises from 21% to 24% and mean dddG from +0.89 to +0.96. The per-protease interval helps a little. Only a criterion on the predicted ΔG removes the population by construction.

Recommendation

Filter double-derived items on the additive prediction, at a threshold between −1 (keeps 79%, removes the guaranteed-biased fifth, slope −0.29) and 0 (keeps 57%, slope fully plateaued at −0.24). Retire the density cap: it costs 34% of the data and moves the bias the wrong way. Keep the per-protease interval as a cheap secondary filter for variants that failed for unrelated reasons.

The cost is borne entirely by the conditional targets, not the wild-type path: 0% of single mutants sit below the floor, so the WT adapter keeps its full range of destabilisation either way.

### Limits of this audit

- Test–retest separates reproducible from random, never real physics from an error shared by both proteases — and in the unreachable region the error is shared, which is exactly why the bias survives every reliability check. The surviving stability trend could still be partly common-mode compression.
- The duplicate-based noise estimate rests on 4,534 pairs, 0.8% of variants and all of them double mutants. If re-measured variants are better behaved than average, 0.24 kcal/mol understates the noise and the reliabilities above are optimistic.
- The distance test is weak here *by dataset design*: the double mutants were chosen as structural contacts, so 83% of even long-range pairs sit within 5 Å and there is almost no non-contact control group. Residual coupling does fall from 0.47 kcal/mol at contact to about 0.23 beyond 8 Å, on a few thousand pairs.
- The dataset contains only single and double mutants — no triples at all — so any conditional target derived from it is a two-mutation quantity, while screening applications ask about far more.
- The variance split depends on one global model class. A stronger stability-only model would claim more of the 0.47.
- A better distance control exists and is not used here: three domains (2K5H, 1H8K, 1OPS) were each scanned in their wild-type form and in two mutant backgrounds, giving 5,512 single mutations measured in both contexts — directly measured conditional effects with a full distance axis, rather than effects derived by subtraction from designed contact pairs.

Source: Tsuboyama2023_Dataset2_Dataset3_20230416.csv, 776,298 rows; insertions and deletions excluded; doubles restricted to those whose two constituent singles are both measured in the same library. dddG = ΔΔG(AB) − ΔΔG(A) − ΔΔG(B). Per-protease ΔΔG computed against each library's own wild-type rows. Reliability is Pearson or Spearman between the trypsin-derived and chymotrypsin-derived dddG; bootstrap 95% intervals on the Pearson values span ±0.01 at these sample sizes. Quotations from Tsuboyama et al., *Mega-scale experimental analysis of protein folding stability in biology and design*.