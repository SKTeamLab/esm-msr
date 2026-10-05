ESM-MSR · methods note

# Does the model learn identity-dependent interaction epistasis?

A test that answers this without having to model the assay, correct for saturation, or estimate a noise ceiling — because it is built to be blind to all three. Written from the beginning, with the reasoning for every control.

**Headline.** Yes. On the full held-out test docket — 36 domains, 10,732 doubles in 35 near-complete position-pair matrices — ESM-MSR reaches ρ = 0.141 ± 0.003 over three training seeds — 0.164 once unreliable sub-floor measurements are dropped — against 0.083 ± 0.011 for Mutate Everything, 0.079 for ProteinMPNN, 0.044 for ThermoMPNN-D, Rosetta `cartesian_ddg` and SPURS alike, and 0.037 for its own backbone with the adapter off. Three independent additive readouts return **exactly zero**. On the 26-library validation split the same ranking holds with wider separations — 0.198 against 0.129 for Mutate Everything — and on the two proteins verified held out by sequence identity it reaches 0.146, **2.8× the best of 95 ProteinGym reference models**. Of those 95, **54 cannot express identity-dependent interaction at all** — their readout is additive, so the statistic is structurally zero. Agreement declines with Cβ distance, as a contact-mediated mechanism predicts.

> **Update, 2026-10-04 (see §16).** Every number in §§05–12 describes models trained *before* the changes in §16. No retrained model has been scored yet, so this note reports **no improved results**. What changed is (a) the training objective (§16.2), (b) a per-pair version of the validation flip metric that is the one comparable with the tables here (§16.1), and (c) a variance analysis showing how much of the signal is available (§16.3). One claim in §06 was wrong and is corrected there.

**The distinction this test is built around.** Non-additivity comes in two kinds. *Identity-independent* epistasis applies the same correction to every substitution pairing at two positions — a fixed bias for that pair, or anything that depends only on how large the two individual effects are. *Identity-dependent* interaction is the part that changes when you swap one tryptophan for one alanine. Conventional ΔΔΔG metrics are dominated by the first, so a model can score well on them while predicting nothing about specific pairings. This test returns **exactly zero** for every identity-independent mechanism, by construction, and that is demonstrated numerically below rather than argued.

## 01The question, stated precisely

Two mutations can be non-additive for reasons that have nothing to do with each other's chemistry. What we want to know is whether the model has learned the part that *does*: whether putting a tryptophan rather than an alanine at one position changes the effect of what sits at a second position, in a way the model anticipates. Call that **identity-dependent interaction**.

Everything below works inside a single pair of positions (p, q). Tsuboyama's double mutants were designed as near-complete 20×20 matrices at a handful of position pairs per domain, so for a given pair we have most combinations of a substitution X at p and Y at q measured in the same experiment. That matrix is the experiment we need.

## 02Why the obvious approach cannot answer it

The classical measure of epistasis is

```
ΔΔΔG  =  (double)  −  (single X)  −  (single Y)
```

and the obvious test is to correlate measured ΔΔΔG against predicted ΔΔΔG. That number is the one the harness reports, and it is uninformative, for three separate reasons.

### Problem 1 — nonspecific epistasis dominates

Assays are not linear in stability. Past a point, a protein is unfolded and more destabilization cannot be seen; a growth assay saturates; Tsuboyama's ΔG becomes unidentifiable near its dynamic-range floor. So even if two mutations never touch, their combined effect falls short of the sum, and ΔΔΔG is large. This depends only on *how big* the two effects are, never on *which residues* they are. On GRB2-SH3 abundance it is 73% of ΔΔΔG variance. A model that has learned nothing but the shape of the assay response scores well.

### Problem 2 — the singles' measurement error is shared, not independent

This one is easy to miss. ΔΔΔG subtracts the measured single effects, so it inherits their error. But every double containing substitution X subtracts the *same* measurement of X. In a 20-column matrix row, all 19 doubles carry an identical error. The noise is therefore not scattered — it has exactly the shape of a row effect plus a column effect, and it does not average away within a pair. In simulation it inflated ΔΔΔG variance by 1.7× and pulled the recoverable correlation from 0.66 down to 0.45.

### Problem 3 — position coupling is not identity coupling

Two positions packed against each other may couple more than two surface positions, on average over all amino acids. That is a real interaction, but it says nothing about chemistry, and a model that has only learned "buried pairs interact" would score for it.

Each of these has a correction, and I tried several. Subtracting a monotone fit of the additive sum, partialling it out, fitting a two-way additive model and taking the residual — all of them leak. The fundamental obstacle is that nonspecific epistasis lives on a **latent** scale, while every correction is applied on the **observed** scale, and the map between them is the unknown thing. In simulation, a two-way interaction residual reports ρ = 0.88 between two models that share *no* specific interaction at all, purely because each one saturates. Correcting the estimator was the wrong strategy.

## 03The test

Instead of modelling the assay, use a statement that is true no matter what the assay does.

**If X and Y do not interact in an identity-dependent way, then whichever of two substitutions at p is more destabilizing must be the same regardless of what sits at q.**

Why this is safe: if the latent stability is additive, the gap between X and X′ is `α_X − α_X′`, which carries no reference to Y. The assay then applies some unknown response curve on top — but a monotone function cannot change which of two numbers is larger. Saturation, a hard floor, a sigmoid, a log: all of them preserve order. So under the null, the *ranking* of substitutions at p is identical in every column q, whatever the assay did to the values.

Identity-dependent interaction is the only thing that can flip that ranking. So we look for flips, and ask whether the model predicts where they are.

What the test looks at

Two substitutions at position p, measured against four different partners at position q

On the left the ordering holds in every column — consistent with additivity plus any assay distortion, and the test returns exactly zero. On the right the ordering flips at two partners: V is worse than L with most partners but better with others. No additive model and no monotone response curve can produce that. The test asks whether the model's predicted matrix flips in the same places.

### The statistic, step by step

Work inside one position pair. Lay the double mutants out as a matrix: rows are the substitution at p, columns are the partner at q, each cell is the measured phenotype of that double. Then three steps, applied identically to the measured matrix and to the model's predicted matrix:

1. **Rank within each column.** Take one column — one fixed partner residue — and replace the values in it by their percentile rank among the substitutions present. Repeat for every column independently. *This is the step that throws the assay away.* A column's values may have been squashed by saturation, clipped at a floor, or log-transformed; ranks within that column are unchanged by any of it.
2. **Double-centre.** Subtract each row's mean, then each column's mean. Row means carry how destabilizing each substitution is on average; column means carry the same for the partners; the grand mean carries the pair's overall coupling. Removing them leaves only the part that depends on the *combination*.
3. **Correlate.** Spearman between the measured and predicted centred matrices, pooled over all position pairs.

Call the result of steps 1–2 the **flip signature**. Here is why it works. Suppose there is no identity-dependent interaction. Then within any column, the ordering of substitutions is set by their own latent effects, which do not reference the partner — so *every column gives the same ranking*. A matrix whose columns are all identical is constant along rows, and double-centring a matrix that is constant along rows gives exactly zero. Not small: zero.

Two quantities come out of this, and keeping them apart matters:

- **Signature SD** — how much flip structure a model predicts *at all*. A model whose predicted epistasis is additive in the two substitutions, or a fixed bias per position pair, has a signature of exactly zero, hence signature SD ≈ 0. It is not that such a model scores badly; it is that it has nothing to say about identity.
- **ρ** — how well the structure it does predict matches the measurement.

A model needs both. High signature SD with ρ ≈ 0 means it confidently predicts specific pairings and gets them wrong; signature SD ≈ 0 means it never made a specific prediction.

Worked example on a 4 × 4 matrix

Measured values on top, flip signature below; a saturating assay response is applied in all three

Left: purely additive latent effects. The assay response has compressed the values badly — the bottom-right corner is nearly flat at −3.0 — yet the signature is identically zero, because within every column the ordering L > A > V > P still holds. **Middle: identity-independent epistasis.** A fixed −0.5 bias is applied to every cell of the pair, which is exactly the mechanism that lets a model appear to capture epistasis without capturing interactions. The signature is again identically zero. Right: three cells carry genuine identity-specific interaction, and the signature becomes non-zero (SD 0.133) precisely at and around them. The test responds to the third case and only the third. In that third panel the L row reads zero throughout because none of the three interactions involves L — a zero cell means "no ordering flip here", not missing data, and every cell is populated.

| Confound | How it is controlled | Residual risk |
| --- | --- | --- |
| Nonspecific / global epistasis | Annihilated. Any monotone response preserves within-column order. | None |
| ΔG dynamic-range floor | Annihilated — a floor is monotone. | None |
| Shared model/assay saturation | Annihilated on both sides for the same reason. | None |
| Each mutation's own effect | Removed by double-centring (row and column means). | None |
| Position-pair coupling | A constant within a pair; removed by centring. | None |
| Error in the single mutants | Never used. The test reads only the double-mutant matrix. | None |
| Error in the doubles | Produces random flips, uncorrelated with the model. | Attenuates only — so the estimate is a floor |
| Non-monotone assay response | Not controlled. | Would break the test; no evidence of it here |
| Indirect (allosteric) coupling | Not separated from contact-mediated coupling. | Limits the claim to "interaction", not "contact" |

The last row of "None" entries is the point of the design: these are not corrections whose adequacy you have to argue, they are consequences of the construction. The only thing left that can move the number is noise, and noise can only push it *down*. **Every value below is a lower bound.**

### The null distribution

Significance comes from a permutation that keeps each side's own flip structure and destroys only the claim about which pairing flips: within a position pair the predicted signature is re-indexed by independent random permutations of the row and column identities. The model keeps exactly as much predicted interaction as it had; only its alignment to the measurement is scrambled. 200 draws per DMS.

## 04Validating the test before using it

A simulated 19×19 pair matrix with known additive effects, a known specific interaction, a strongly saturating assay response, and a *different* saturating response for the model, so that any shared-saturation artefact would show up.

Six nulls and four signal cases

Simulated data; the test should return zero above the line and track the truth below it

All six nulls return ≤ 0.022, including a hard floor applied to the measurement (0.005) and a cubic monotone rescaling (exactly 0.000). The null that matters most — both sides carrying genuine but *unrelated* interaction — gives 0.022. Where interaction is genuinely shared, the test recovers 0.78 against a true value of 0.90, and degrades smoothly as measurement noise rises. So it is conservative in both directions: it does not manufacture signal, and it understates real signal by roughly 15% before noise.

### The same scenarios scored conventionally

Running the identical simulated data through ρ(ΔΔΔG measured, ΔΔΔG predicted) — the quantity the benchmark harness reports — shows what the conventional estimator does with these nulls.

Conventional ΔΔΔG correlation against the conditional-ordering test

Same simulated data, same scenarios, two estimators; the top six rows are nulls where the honest answer is zero

In the first null neither side has *any* specific interaction — each merely saturates — and conventional ΔΔΔG returns **+0.903**, which is essentially the value it returns when the interaction is genuinely shared (+0.878). The two situations are indistinguishable. It also returns +0.459 when only the measurement interacts, +0.374 when both interact but independently, and **−0.251** under a monotone rescaling of the measurement — the right answer with the wrong sign. Worst null: 0.903 for ΔΔΔG against **0.012** for the ordering test, a factor of 75. The last row adds measurement noise to the singles, which in real data is shared by every double containing that single; conventional ΔΔΔG inherits it, the ordering test never reads the singles at all.

## 05Core result: the held-out test docket

The 36-domain hyperopt test split, held out from training entirely: 43,700 variants, of which 10,732 are doubles concentrated in 35 near-complete 20 × 20 position-pair matrices across 7 domains. Every model below is scored on byte-identical rows joined on `uid`, so nothing differs but the predictions.

Two of the entries are controls rather than competitors. The **WT pass only** readout is a sum of per-position log-likelihood ratios in the wild-type background — additive by construction — so the test must return a flat signature for it. The **masked-marginal** readout scores each mutated position with that position masked, which also cannot see the partner residue. Both are real-data checks on the claim that the statistic is blind to identity-independent prediction.

Full held-out test docket, every model

35 position pairs, 10,125 scored cells, identical rows; bars show the mean over training seeds and the range across them

Each fine-tuned model was scored at three training seeds independently; Rosetta's three `cartesian_ddg` replicates play the same role. Zero-shot entries have no fine-tuning seed and contribute one value. ESM-MSR's lead is far larger than any seed spread — 0.147 ± 0.002 against 0.078 ± 0.011 for the next best — and the three additive controls sit at the floor. Note that **ProteinMPNN unmasked matches Mutate Everything** at 0.078 with no supervision at all, and that Rosetta has the largest signature SD of any entry (0.120) while converting the least of it into agreement.

| Model | seeds | ρ ± sd, all cells | ρ ± sd, measured ΔG ≥ 0 | signature SD | z |
| --- | --- | --- | --- | --- | --- |
| **ESM-MSR, MT pass only** | 3 | +0.158 ± 0.007 | +0.181 ± 0.010 | 0.0996 | +14.3 |
| **ESM-MSR, unmasked (released)** | 3 | +0.141 ± 0.003 | +0.164 ± 0.004 | 0.0567 | +13.3 |
| ESM-MSR, independence-masked | 3 | +0.130 ± 0.005 | +0.149 ± 0.009 | 0.0600 | +11.4 |
| Mutate Everything | 3 | +0.083 ± 0.011 | +0.085 ± 0.010 | 0.0996 | +8.2 |
| ProteinMPNN (unmasked) | 1 | +0.079 | +0.085 | 0.0921 | +7.5 |
| ProteinMPNN (masked 0.20) | 1 | +0.060 | +0.069 | 0.0930 | +5.3 |
| Rosetta `cartesian_ddg` | 3 reps | +0.044 ± 0.005 | +0.049 ± 0.003 | 0.1220 | +4.0 |
| ThermoMPNN-D (epistatic) | 3 | +0.044 ± 0.016 | +0.046 ± 0.019 | 0.0351 | +4.3 |
| SPURS | 3 | +0.044 ± 0.023 | +0.049 ± 0.019 | 0.0494 | +4.0 |
| ESM3 base, adapter off | 1 | +0.037 | +0.042 | 0.0505 | +3.3 |
| ESM-MSR, masked-marginal *(control)* | 3 | +0.000 ± 0.000 | +0.000 ± 0.000 | 0.0000 | 0.0 |
| ESM-MSR, WT pass only *(control)* | 3 | +0.000 ± 0.000 | +0.000 ± 0.000 | 0.0000 | 0.0 |
| ThermoMPNN-D additive *(control)* | 3 | −0.000 ± 0.001 | −0.003 ± 0.003 | 0.0016 | −0.1 |

The three additive controls return **exactly zero, with signature SD exactly zero**: ESM-MSR's WT pass, its masked-marginal rescoring, and ThermoMPNN-D's additive checkpoint. Not approximately — the flip signature of an additive prediction on a complete block is identically flat, so the statistic is undefined and reported as 0. Three independent sources, and the test behaves exactly as the construction says it must.

**Removing sub-floor measurements raises ESM-MSR and barely moves the others.** Dropping doubles whose measured ΔG is below 0 discards 16.8% of cells and takes ESM-MSR from 0.141 to 0.164, a 16% gain, while Mutate Everything goes 0.083 → 0.085 (+2%) and ProteinMPNN 0.079 → 0.085. The margin widens from 1.7× to 1.9×. That is the direction consistent with real signal: the unreliable measurements were *diluting* ESM-MSR's apparent accuracy, not inflating it. Note this filter is itself value-based truncation, so it is only safe with the complete-block estimator in place.

Seed variance changes two readings. **SPURS and ThermoMPNN-D have the largest spreads** (±0.023 and ±0.016); single-seed figures for either are unreliable. ESM-MSR is the most stable entry at ±0.003, and its margin over the field is many times any spread present. Two further results: **ProteinMPNN unmasked reaches 0.079 with no supervision at all**, close to Mutate Everything's supervised model, and Rosetta has the highest signature SD of any entry (0.122) while converting the least of it into agreement.

**Our ProteinMPNN is not ProteinGym's ProteinMPNN, and the two figures must not be cross-read.** On the two overlapping assays (1UFM and 3DKM) the two score columns agree at only ρ = 0.81–0.85 on identical variants — if the decoding were the same this would be \~0.99. Their flip signatures differ more sharply still: ours has SD 0.061–0.094 against ProteinGym's 0.214–0.240, so ProteinGym's decoding emits three to four times as much identity-dependent structure. On the same cells, ours reaches ρ = 0.146 and 0.041 on 1UFM and 3DKM where ProteinGym's reaches 0.052 and −0.004. Our masked-0.20 and unmasked variants are near-identical to each other, so the masking fraction is not the difference — the decoding scheme is. The 0.078 in this table and the 0.024 in the ProteinGym figure are different methods that share a name.

**The masked-marginal result is worth its own line.** The same checkpoint, rescored with the mutated position masked, drops from 0.141 to **exactly zero**, signature SD 0.0567 to 0.0000. Masking the scored position removes the model's view of how the partner residue changes this position's identity preference, and the readout becomes additive — not merely worse, but incapable. The independence-masked variant retains most of it (0.130), so it is masking the *scored* position specifically that is fatal.

## 06The same comparison on the validation split

The 26-library validation split, scored identically: 9,419 doubles in 27 usable position-pair matrices, 7,232 cells, byte-identical rows across models, three training seeds where they exist. This is the table to compare a retrained model against, **using the per-pair metric `val_rho_flip_pair_avg` (§16.1)**.

> **Correction (2026-10-04).** An earlier version of this paragraph said the monitored metric `val_rho_flip_avg` sits on the same axis as the entries below. It does not. `val_rho_flip` builds each matrix from the key `code|scored position|partner position+residue`, so its matrix is indexed by the scored position and **pools columns across different partner positions**; the tables here use one matrix per position pair. The two numbers can be close (the released model's 0.198 here and its logged `val_rho_flip_avg` of about 0.20–0.21) without measuring the same thing: see §16.1.

Validation split, every model

27 position pairs, 7,232 cells; bars are the mean over training seeds, whiskers the range, hollow rings the measured ΔG ≥ 0 subset

The ranking is the same as on the test docket and the separations are wider, because the validation libraries carry deeper pair matrices. ESM-MSR's MT pass reaches 0.248 against 0.129 for Mutate Everything — ahead on 22 of 27 position pairs, p \< 0.001 — and the three additive controls again return exactly zero. Note that validation values run higher than test-docket values throughout (0.198 vs 0.141 for the released readout), so the two dockets are not interchangeable: compare a new model against this table, not against the test one.

| Model | seeds | ρ ± sd, all cells | range | ρ, measured ΔG ≥ 0 | signature SD | z |
| --- | --- | --- | --- | --- | --- | --- |
| **ESM-MSR, MT pass only** | 3 | +0.248 ± 0.001 | 0.247–0.249 | +0.259 | 0.1140 | +19.3 |
| ESM-MSR, independence-masked | 3 | +0.204 ± 0.004 | 0.201–0.209 | +0.220 | 0.0721 | +16.4 |
| **ESM-MSR, unmasked (released)** | 3 | +0.198 ± 0.012 | 0.184–0.206 | +0.220 | 0.0656 | +17.0 |
| ESM3 base, adapter off | 1 | +0.132 | — | +0.141 | 0.0712 | +10.8 |
| Mutate Everything | 3 | +0.129 ± 0.007 | 0.121–0.134 | +0.136 | 0.1033 | +11.0 |
| ProteinMPNN (masked 0.20) | 1 | +0.116 | — | +0.136 | 0.0920 | +9.9 |
| ProteinMPNN (unmasked) | 1 | +0.108 | — | +0.126 | 0.0909 | +8.9 |
| Rosetta `cartesian_ddg` | 3 reps | +0.071 ± 0.002 | 0.069–0.072 | +0.068 | **0.1359** | +6.0 |
| ThermoMPNN-D (epistatic) | 3 | +0.042 ± 0.009 | 0.035–0.052 | +0.044 | 0.0329 | +3.5 |
| SPURS | 3 | +0.037 ± **0.036** | 0.002–0.073 | +0.043 | 0.0474 | +3.0 |
| ESM-MSR, WT pass only *(control)* | 3 | +0.000 ± 0.000 | — | +0.000 | 0.0000 | 0.0 |
| ESM-MSR, masked-marginal *(control)* | 3 | +0.000 ± 0.000 | — | +0.000 | 0.0000 | 0.0 |
| ThermoMPNN-D additive *(control)* | 3 | +0.000 ± 0.000 | — | +0.000 | 0.0000 | 0.0 |

Paired per position pair against Mutate Everything (seed-averaged, n = 27): MT pass better on 22 (p \< 0.001), released readout on 19 (p = 0.010), independence-masked on 20 (p = 0.016). ProteinMPNN and base ESM3 are statistically indistinguishable from Mutate Everything here (p = 0.89 and 0.41) — three different routes to roughly 0.11–0.13.

Two cautions for anyone reading a retrained model against this table. **SPURS has a seed spread of ±0.036 on a mean of 0.037**, so a single seed of it is uninformative; assume the same could be true of a new model and run more than one. And the measured ΔG ≥ 0 column moves ESM-MSR up by about 0.02 while barely moving the weaker entries, so quote whichever column you use consistently.

## 07Sub-floor doubles: bad data, biased data, and which one breaks the test

A double mutant whose two substitutions together push the protein below the assay's identifiability floor is a different object from a double whose substitutions genuinely compensate and leave it folded. The first is **bad data** — the maximum-likelihood fit cannot determine ΔG and reports a number anyway. The second is **real data under survivorship bias** — a true measurement, but you only see it because it survived. Confusing them matters, and as it turns out they affect this test in completely different ways.

### What the released table does and does not tell you

There is no quality flag. The only flag-like columns in the Tsuboyama release are `fitting_error_t` / `fitting_error_c`, which are fit residuals rather than identifiability measures, and `Stabilizing_mut`, which is a classification. Across the 64 domains used here, of 47,967 doubles with both singles measured:

- **61.8%** have an additive-predicted ΔG below the practical floor (+0.5)
- **47.6%** below 0
- **23.4%** below the release's own stated −1 bound

Every one of them carries a numeric `ddG_ML` with nothing to mark it. So yes — doubles whose combined effect definitely falls below the assay appear as ordinary entries.

### The two populations can be partly separated, but not by ΔΔΔG

Within the sub-floor set (additive ΔG \< 0, n = 22,840), the per-protease confidence interval plus the measured ΔG splits it three ways:

| Class | criterion | n | share | mean ΔΔΔG | median CI |
| --- | --- | --- | --- | --- | --- |
| Genuine compensation | CI ≤ 0.5 and measured ΔG ≥ 0.5 | 7,456 | 32.6% | +1.86 | 0.25 |
| Unidentifiable (bad data) | CI > 0.5 | 10,298 | 45.1% | +1.42 | 1.15 |
| Genuinely unfolded | CI ≤ 0.5 and measured ΔG \< 0.5 | 5,086 | 22.3% | +1.40 | 0.34 |

The CI does discriminate: well-measured doubles (additive ΔG > 1) have median CI 0.141 with 2.8% above 0.5, while sub-floor doubles have median CI 0.552 with 54.4% above 0.5. But note the ΔΔΔG column — **all three classes show a large positive mean**, +1.40 to +1.86. Filtering on confidence removes the fabricated class and leaves the survivorship bias completely intact. Apparent stabilising epistasis at the floor is not simply an artefact to be filtered away; part of it is real compensation that you are only able to see *because* it is real.

*Update (§16.2):* variants the assay reports only as `<-1` or `>5` are now kept as censored items rather than dropped; that is the rank-only treatment suggested above, extended to both ends of the range. Not yet run.

## 08The same test on ProteinGym's Tsuboyama domains

Scored from the existing seed-1, σ=1, unmasked ProteinGym run — no new inference. Domains are mapped to `hyperopt_splits.pkl` by PDB code, so training contamination is visible rather than assumed. "No split" means the domain appears in none of train, val or test_internal.

| Split | domains | pairs | cells | ρ (median) | IQR | p \< 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 23 | 87 | 25,542 | +0.270 | 0.186–0.368 | 21/23 |
| **no split (held out)** | 16 | 44 | 12,808 | +0.237 | 0.194–0.282 | 14/16 |
| test_internal | 2 | 22 | 5,830 | +0.207 | 0.138–0.275 | 2/2 |
| val | 6 | 11 | 3,494 | +0.148 | 0.114–0.187 | 5/6 |

Held-out domains reach 0.237 against 0.270 on training domains. That gap is small — much smaller than the contaminated estimators suggested, which makes sense: what those were partly measuring was the model's grasp of the assay response, and that is better on proteins it was fitted on. Individual held-out domains run from 0.473 (SAV1_MOUSE, z = 7.5) down to −0.024 (ODP2_GEOSE, the one clear failure), with 14 of 16 significant at p \< 0.05.

### The measurement artefact is not driving it

Re-running with the identifiability filters applied — per-protease 95% CI ≤ 0.5 and dG ≥ 0.5 on all three measurements, plus `min_additive_dG` ≥ 0, which is the training filter — *raises* the estimate: held-out 0.250 (10 domains), training 0.333. Cleaner data gives a stronger result, which is the direction consistent with real signal and inconsistent with the dynamic-range artefact generating it.

## 09Where the signal comes from

A correlation on held-out proteins shows generalization but not attribution. The cleanest possible control is the same model with the adapter contribution scaled to zero (`--lora_epsilon 0`, the `unmasked-esm-msr-small/sigma0.0` run). That is ESM3 itself, scored through the identical machinery: same backbone, same structure input, same two-pass wild-type/mutant scheme, same log-likelihood-ratio readout, same position pairs, same cells. Only the learned weights are gone. Its task correlation on HECD1 falls from 0.530 to 0.302, confirming the adapter really is switched off.

One convenience here: the calibration head applies a scale and a bias, which is a monotone transform — and this test is invariant to monotone transforms. So the σ = 0 run needs no recalibration to be comparable.

Adapter off versus adapter on, the same domains

Each line is one held-out Tsuboyama domain; left point is base ESM3, right point is the trained model

Fifteen of the sixteen held-out domains improve, median gain +0.145, Wilcoxon p = 2 × 10⁻⁴. Across all 47 Tsuboyama domains the gain holds in 44 (p = 3 × 10⁻¹²). Base ESM3 is not at zero — it reaches 0.09, the same level as ESM-IF1 — so the training sharpens and amplifies a signal the structure-aware backbone already carries rather than inventing one. The single regression, ODP2_GEOSE, is also the smallest matrix at 184 cells.

Widening the comparison: ProteinGym ships reference predictions from 95 models, including the current generation — ESM3, ESMC, xTrimoPGLM, Progen3, ProSST, VenusREM, SaProt, S3F, ESCOTT, PoET, SiteRM. None is trained on Tsuboyama ΔΔG. Running the identical test on all 95 separates two questions that conventional metrics merge.

All 95 ProteinGym reference models: how much they predict, and whether it is right

Signature SD on the horizontal axis, agreement with measurement on the vertical; 47 Tsuboyama DMS, 47,763 cells each

**Fifty-four of the ninety-five sit in the grey band on the left**, at signature SD ≈ 0.010 — the numerical floor. Their scoring function is additive across the two mutated positions, so their flip signature is structurally zero and their median ρ is −0.0001 with no value exceeding 0.010 in absolute terms. This includes ESM2 at every size, ESM1b, ESM1v, ESMC, xTrimoPGLM's masked variants, ProSST, ProtSSN, SaProt, GEMME, VESPA, ESCOTT, SiteRM, VenusREM, MIF and MIF-ST — *and ProteinGym's own ESM3 column*. These are not weak results; they are undefined ones. Of the 41 expressive models only two clear 0.05: ESM-IF1 at 0.087 and PoET at 0.065. ProteinMPNN is the extreme case of the distinction — the highest signature SD of any model at 0.225, predicting more identity-specific structure than anything else, with ρ = 0.024.

**ESM3 appears twice in that figure, and the gap between the two points is the readout.** ProteinGym's own ESM3 column is a site-independent masked-marginal likelihood; for a double mutant it is a sum of two single-position terms, so it is additive and its signature is structurally zero (ρ = −0.006, SD 0.0104). The same backbone scored through the MSR two-pass readout — full mutant sequence, structure attached, adapter at σ = 0 — reaches ρ = 0.094 with SD 0.051. Running ESM3 "in MSR mode" is therefore what makes the quantity expressible at all. The backbone was never the limiting factor; the readout was.

ESM-MSR is deliberately absent from that figure. Pooling it over all 47 domains would mix training proteins into the estimate, and the `unseen` label used earlier means only "absent from the three split lists" — not a sequence-identity guarantee. The set that *is* verified held out on a sequence-identity basis is the hyperopt `test_internal` split, and exactly two of its domains carry ProteinGym double-mutant matrices: CSN4_MOUSE (1UFM) and HECD1_HUMAN (3DKM), 22 pairs and 5,830 cells. Every model can be placed on the same axes there with no contamination at all.

Verified held-out proteins only: all 99 entries on identical cells

1UFM and 3DKM, the two sequence-identity-verified held-out domains with pair matrices; 22 pairs, 5,830 cells per model

ESM-MSR reaches ρ = 0.146 (z = 11.3), which is **3.1× the best reference model** — ESM-IF1 at 0.047 — and 7.2× its own untrained backbone in MSR mode (0.020, p = 0.13, not significant here). ProteinGym's site-independent ESM3 is at −0.012, and ESM-MSR's own additive WT pass at +0.010, both at the floor. Fifty-five of the 99 entries cannot express the quantity. On these two proteins the untrained MSR-mode readout is not significant, so the adapter training accounts for essentially all of the usable signal — a stronger attribution than the all-domain pooling suggested.

## 10Is it contact-mediated?

The test cannot distinguish a direct side-chain contact from coupling transmitted through the fold. What it can do is ask whether the signal behaves like a contact phenomenon. Grouping held-out and training position pairs by Cβ–Cβ distance:

Agreement falls with separation

Median ρ per distance bin, Tsuboyama position pairs

Unfiltered, ρ runs 0.250 / 0.202 / 0.120 / 0.105 across the four bins with Spearman(distance, ρ) = −0.27 over 163 pairs; with the identifiability filters applied the gradient steepens to −0.42 over 70 pairs. Base ESM3 on the identical pairs is both weaker and flatter — 0.082 / 0.087 / 0.042 / −0.003, Spearman −0.16 — so the training sharpens the distance dependence as well as the overall level. Consistent with a contact-mediated mechanism, though Tsuboyama's doubles were chosen as structural contacts, so the far bins are thin (3 pairs beyond 15 Å) and this is suggestive rather than decisive.

## 11Generalization beyond the assay

The same test on non-Tsuboyama ProteinGym DMS, which are fitness and binding readouts rather than ΔG, and which need deep position-pair matrices to be usable at all — only seven qualify.

| DMS | category | pairs | cells | ρ | z | p |
| --- | --- | --- | --- | --- | --- | --- |
| SPG1_STRSG_Wu_2016 | Binding | 6 | 2,061 | +0.073 | +3.2 | \<0.001 |
| GRB2_HUMAN_Faure_2021 | OrgFitness | 141 | 15,623 | +0.028 | +3.6 | \<0.001 |
| SPG1_STRSG_Olson_2014 | Binding | 1,485 | 535,820 | +0.026 | +18.0 | \<0.001 |
| YAP1_HUMAN_Araya_2012 | Binding | 3 | 140 | +0.023 | +0.2 | 0.84 |
| F7YBW8_MESOW_Aakre_2015 | OrgFitness | 3 | 339 | +0.017 | +0.3 | 0.77 |
| PABP_YEAST_Melamed_2013 | OrgFitness | 24 | 1,068 | +0.002 | −0.0 | 1.00 |
| CAPSD_AAV2S_Sinai_2021 | OrgFitness | 1 | 167 | −0.020 | −0.2 | 0.86 |

Three are significant, and the signal is about eightfold weaker than on ΔG data. The largest, SPG1_Olson, reaches z = 18 only because it carries 535,820 cells — the effect itself is 0.026. Read conservatively: **the capability survives the move to a different phenotype but is much reduced**, and the three nulls are all small-n rather than evidence of absence. Whether that reduction is the model's transfer failing or the fitness assays' own noise and compression is not resolved by this test.

## 12Why the MT pass beats the combined prediction

The MT pass alone scores higher than the released ensemble (0.158 vs 0.141), which is worth explaining because the obvious reading — that the WT pass carries a false non-additive signal — turns out to be wrong in a specific and more useful way.

The WT pass has a flip signature of **exactly zero**. It is a sum of per-position log-likelihood ratios on the wild-type background, so it is additive by construction and provably carries no identity-dependent information, true or false. It cannot contaminate with spurious epistasis because it contains no epistasis of any kind.

But an additive term is *not neutral* with respect to this statistic. Adding `0.5·(a_X + b_Y)` to the matrix leaves the column term `b_Y` harmless — constant within a column — while the row term `a_X` shifts each row by a different amount and therefore **reorders substitutions within every column**. Sweeping the blend weight shows the cost:

| blend | ρ | signature SD |
| --- | --- | --- |
| pure MT (w = 0) | +0.150 | 0.1006 |
| w = 0.25 | +0.144 | 0.0839 |
| w = 0.50 — the released `combined_pred` | +0.142 | 0.0605 |
| w = 0.75 | +0.101 | 0.0369 |
| pure WT (w = 1) | +0.000 | 0.0000 |

And the admixture *distorts* rather than merely dilutes: the combined signature correlates with the MT signature at only **ρ = 0.55**. If the WT pass were simply weakening a shared structure that would be near 1. Roughly half of the MT pass's identity-dependent structure is scrambled by averaging in an additive term.

This is a genuine trade-off rather than a defect. The half-and-half average is thermodynamically motivated — it is the mean of the two paths A→AB and B→AB and is exact for ΔΔG when epistasis is pairwise — so it earns its place for the calibrated ΔΔG output. It is simply the wrong place to read epistasis from. The blend sweep says the good region for the interaction channel is w ∈ \[0, 0.25\].

## 13What this suggests for training

Ordered by expected value. The first three follow directly from measurements in this report; the rest are reasoned rather than tested.

1 · Strongest recommendation

**Train with a within-column ranking loss on pair matrices.** The reason this metric is trustworthy is the reason the corresponding loss would be: a loss on within-column orderings at a position pair is invariant to the assay's monotone response and to the dynamic-range floor *by construction*, so it cannot be satisfied by learning saturation. The machinery already exists — `_compute_rank_loss` and ListMLE — and only the list construction changes: each column of each position-pair matrix becomes one list, with the measured ordering as its target. This optimizes the quantity we can actually verify, and it is indifferent to the sub-floor values that currently teach the model to predict an artefact.

2 · Measure it in validation

**Add a flip-signature metric and monitor it for the MT adapter.** `val_rho_combined_avg` cannot see this channel: conventional ΔΔΔG correlation returns +0.90 in simulation when neither side has any interaction, so it cannot distinguish MT configurations from each other. Compute the flip ρ on validation libraries that have complete pair matrices and use it for checkpoint selection, then hyperparameter-optimize the MT-specific settings against it — `lora_rank_mt`, `cond_weight`, `native_cond_weight`, `mt_single_anchor_weight`. These are currently being tuned against a metric that is blind to their purpose.

3 · Free, no retraining

**Expose the MT pass as the epistasis output, and keep `combined_pred` for ΔΔG.** Worth 0.141 → 0.158 immediately, and the blend weight could instead be tuned on the flip metric rather than fixed at 0.5. Also keep unmasked scoring: masked-marginal collapses the signature to exactly zero, so masking the scored position does not merely degrade the capability, it removes the model's ability to express it.

4 · Data

**Filter or down-weight training doubles by measured ΔG, not only by the additive prediction.** Excluding measured ΔG \< 0 at evaluation raised ESM-MSR 16% while moving competitors 2%, which says those cells are diluting the model's learned signal. 45% of sub-floor doubles are unidentifiable (CI > 0.5). But drop them carefully: the genuine-compensation class is 32.6% of the sub-floor set with a mean ΔΔΔG of +1.86, and that is real data. The clean resolution is to keep them under a *rank-only* loss, where their unreliable absolute value never enters.

5 · Where to spend capacity

**Weight toward the cells that carry identity information.** From the matched-background control, coupling is 1.54 kcal/mol SD at contact when both substitutions are individually worth ≥1 kcal/mol, against 0.30 for distant or mild pairs; and flip agreement declines with Cβ separation (ρ = −0.26). The information is concentrated in a small corner of the input space, and uniform weighting spends capacity where there is nothing to learn. A distance-gated MT contribution would also help the k ≥ 3 regime, where the model currently fails outright.

6 · The structure channel is the lever

**Revisit structure-encoder adaptation.** Every one of the 54 additive-readout models scores exactly zero; the only baselines above zero are structure-conditioned (ProteinMPNN 0.079, ESM-IF1 0.053). This capability lives in the structure channel, and the design doc notes that encoder adaptation became possible once the cache was stored unmasked.

Tested and did not work

Ensembling with ProteinMPNN. Its flip signature correlates with the MT pass's at only ρ = 0.149, so the two carry genuinely complementary content — but every linear blend is worse than the MT pass alone (0.148 at 25% MPNN, 0.142 at 50%, against 0.150). The reason is the same mechanism as the WT pass: mixing differently-scaled scores reorders within columns. If this complementarity is to be harvested it needs feature-level fusion or a learned gate, not a score-level average.

### Status of these recommendations (2026-10-04)

| # | Recommendation | State |
|---|---|---|
| 1 | within-column ranking loss | **Implemented and run** (`--lambda_rank_mt`, ListMLE over flip columns). Measured: it works on data it trains on and barely transfers; see §16.2 |
| 2 | flip metric in validation | **Done**, plus a per-pair version, `rho_epi_fast` / `rho_epi_full`, and delta-singles diagnostics; see §16.1 |
| 3 | expose the MT pass | not changed by this work |
| 4 | rank-only handling of unreliable doubles | **Implemented** as censoring, extended to both ends of the assay range; see §16.2 |
| 5, 6 | capacity weighting; structure-encoder adaptation | not attempted |

## 14What can be claimed

Supported

On the full held-out test docket, ESM-MSR predicts identity-dependent pairwise interaction epistasis at ρ = 0.141 ± 0.003 over three seeds (35 position pairs, z = 13.3), rising to 0.164 when doubles with measured ΔG below 0 are excluded, and outperforming every model tested on byte-identical rows. The statistic cannot be attributed to the assay's nonlinear response, to the single mutants' measurement error, to position-level coupling, or to any identity-independent bias.

Supported

Most published variant-effect predictors cannot express this quantity at all. Of 95 ProteinGym reference models, 54 use a readout that is additive across the two mutated positions, giving a structurally zero flip signature (signature SD ≈ 0.010, median ρ = −0.0001). Of the 41 that can express it, only ESM-IF1 (0.087) and PoET (0.065) exceed 0.05 when pooled over all 47 Tsuboyama domains, and on the two sequence-identity-verified held-out proteins ESM-MSR's 0.146 is 3.1× ESM-IF1's 0.047.

Supported

The scoring scheme, not only the trained weights, determines whether identity-dependent interaction is expressible. Three additive readouts from independent sources — ESM-MSR's WT pass, its masked-marginal rescoring, and ThermoMPNN-D's additive checkpoint — all land at the numerical floor (signature SD 0.0094–0.0103). ProteinGym's site-independent ESM3 scores −0.006 while the same backbone through the MSR two-pass readout reaches 0.094.

Supported

Among models evaluated on byte-identical held-out rows, ESM-MSR is ahead of all of them, by a margin many times the run-to-run seed spread: 1.9× Mutate Everything and ProteinMPNN, 3.3× ThermoMPNN-D, Rosetta and SPURS. Its own seed spread is the smallest in the table (±0.002). Rosetta predicts the most identity-specific structure of any entry (signature SD 0.120) and converts the least of it into agreement.

Supported

The capability is substantially acquired from the megascale supervision. With the adapter switched off the identical architecture and readout scores 0.09; paired domain by domain the trained model is higher in 44 of 47 Tsuboyama domains (median gain +0.12, Wilcoxon p = 3 × 10⁻¹²) and in 15 of 16 held-out domains (p = 2 × 10⁻⁴).

Supported

The residual baseline capability sits in the structure channel, not in sequence statistics: base ESM3 and ESM-IF1 both reach 0.09, ProteinMPNN 0.03, while every sequence-only language model and every alignment-based method scores ρ ≈ 0.00.

Supported

The agreement declines with Cβ separation between the two positions, consistent with a contact-mediated mechanism.

Supported

Reported values are lower bounds. Measurement noise in the double mutants attenuates the statistic and cannot inflate it; the simulation shows a further \~15% understatement even without noise.

Not supported

That the interaction is a *direct physical contact*. Coupling transmitted through the fold produces the same signature. The distance gradient is evidence toward contact, not proof of it.

Not supported

That the capability transfers strongly to other phenotypes. On fitness and binding assays it is ρ ≈ 0.03 — statistically present on the largest libraries, eightfold reduced, and untestable on the rest for want of pair depth. The paired baseline comparison is also equivocal there: the trained model improves on 4 of 7, and the 3 regressions are the 3 smallest matrices.

Not supported

That the model creates interaction knowledge from nothing. Base ESM3 is already at 0.09, so the honest framing is amplification and sharpening of a signal the structure-aware backbone carries, not de novo acquisition.

Not supported

Any statement about the *magnitude* of predicted interactions. This test reads orderings only; it is deliberately blind to scale. An earlier magnitude-based analysis suggested predicted interaction amplitude is \~2.5× too small, but that analysis used a contaminated estimator and should be redone before being relied on.

## 15Limitations

- **Matrices must be complete, and this is not optional.** Missing cells arise because the double failed to measure, so the missingness pattern depends on both substitution effects and is shared between the measured and predicted matrix. Centring an incomplete matrix admits spurious correlation up to +0.95 under a strict null. Every matrix here is trimmed to a fully complete sub-block first; any reimplementation must do the same.
- **Survivorship bias is not removed by confidence filtering.** Among sub-floor doubles, genuine compensation (32.6%) shows a mean ΔΔΔG of +1.86 — larger than the unidentifiable class. Filtering on CI removes fabricated values, not the selection effect.
- **Monotonicity is assumed.** The whole design rests on the assay response being order-preserving. A non-monotone readout would break it. For stability assays this is safe; for a readout with an optimum it would not be.
- **It needs deep pair matrices.** At least \~6 substitutions at each position with \~70% matrix completeness. Tsuboyama's designed doubles qualify; most fitness libraries do not, which is why only 7 of 149 non-Tsuboyama DMS could be scored at all.
- **Only pairs, only k = 2.** Nothing here tests higher-order mutants, where separate evidence shows the model fails outright.
- **Ordering discards magnitude.** A model could rank interactions perfectly and predict them at the wrong scale, and this test would be satisfied. Calibration needs its own measurement.
- **Distance mapping is unverified.** Sequence positions are mapped to structure by residue order, which is right for these single-domain constructs but worth confirming before publication.
- **"Unseen" is not a sequence-identity guarantee.** The 16-domain ProteinGym figure labelled no-split means only "absent from the three split lists". The verified held-out sets are the hyperopt test docket and, within ProteinGym, the two `test_internal` domains; quote those.
- **The test docket's doubles sit in 7 domains, not 36.** The other 29 test domains carry only single mutants. The 35 pair matrices are deep (median 324 variants) but the domain-level sample is small, so per-domain figures are indicative and the pooled statistic is the one to quote.
- **Models sharing a name may not share a method.** Our ProteinMPNN and ProteinGym's agree at only ρ ≈ 0.83 on identical variants and differ fourfold in signature SD, so numbers for nominally the same model are not transferable between the two figures. Any cross-paper comparison of this statistic needs the decoding scheme pinned down.
- **Signature SD at the floor is a statement about the readout, not the model.** A model placed in the grey band could in principle have interaction knowledge that its scoring function cannot emit. The honest phrasing is "this readout cannot express it", and for several of those models a non-additive readout may well be constructible — as the ESM3 comparison shows.

## 16Update: what changed in training and in the metrics since this note

Written 2026-10-04. **Nothing in this section is an improved score.** The changes below are implemented and unit-tested (87 tests, CPU); the first GPU runs of them were just started, and no result exists yet. Each statement says whether it is measured, implemented, or expected.

### 16.1 Metrics

| metric | what it is | relation to this note's statistic |
|---|---|---|
| `val_rho_flip_avg` (existing) | flip signature with the matrix indexed by the *scored position*, pooling columns across partner positions | **not** the statistic of §§05–06 |
| `val_rho_flip_pair_avg` (new) | one matrix per (scored position, partner position) pair, complete sub-block, double-centred, as in §03 | **the one to compare with the tables in §05–§06** |
| `val_rho_epi_fast`, `val_rho_epi_full` (new) | rank correlation of the two epistasis formulations: comb - WT, and comb_AB - comb_A - comb_B | conventional ΔΔΔG-type, so subject to Problem 1 of §02 |
| `val_auc_dead_*`, `val_auc_hyper_*` (new) | can the model separate assay-dead (`<-1`) and hyperstable (`>5`) variants from the rest | new; needs the out-of-range data in §16.2 |

**Measured: the pooled variant is partly earnable without the partner residue.** A predictor built from the measured data that knows the scored substitution and the partner *position* but nothing about the partner *residue* (the mean over a disjoint half of the partner residues, so it never sees its own cell) scores **0.137** on the pooled validation-style metric (three random splits: 0.1373, 0.1370, 0.1374) and **exactly 0** on the per-pair version. Trained models score about 0.20–0.21 on the pooled one. This is an upper bound on what is attainable without partner-residue information, since the predictor uses measured data from the same matrix, but it means a rise in `val_rho_flip_avg` can come from position-specific rather than residue-specific effects. The statistic in this note is the per-pair one and is unaffected. Source: `docs/anova_epistasis_report.md` §5.2, `analysis_notebooks/anova/flip_splithalf.py`.

**Not yet measured:** the per-pair value for any existing checkpoint. The numbers in §05–§06 were produced by `testdocket.py` / `testdocket_seeds.py`; `val_rho_flip_pair_avg` is a re-implementation of the same construction inside validation, and the two have not been cross-checked on a common checkpoint. Until that is done, treat agreement between them as expected, not demonstrated.

### 16.2 Training changes

1. **The within-column ranking loss recommended in §13.1 was implemented and run** (ListMLE over flip columns, `--lambda_rank_mt`). Measured on epoch-2 checkpoints with a forward-only capability test, flip loss against no flip loss: validation pooled flip score **0.214 vs 0.199** (paired +0.015; better on 6 of 16 libraries), but on the *training* libraries **0.359 vs 0.310** (paired +0.049; better on 19 of 21, p = 0.004). It works on data it trains on and barely transfers. Epoch-matched logged `val_rho_flip_avg` for the controls and flip arms overlap (about 0.17–0.23); epoch-to-epoch noise is about ±0.02–0.03.
2. **Censoring replaces the sub-floor handling of §07 and §13.4** (implemented, not yet run). The assay reports variants it cannot resolve as `<-1` (dead) or `>5` (hyperstable); these were dropped before. They are now kept as *censored* items: tied at the bottom (or top) of their rank list with a censored Plackett–Luce likelihood, and a one-sided hinge in regression that penalises only a prediction on the wrong side of the bound. `--censor_floor` applies the same treatment to numeric values at the floor. The WT-head ranking path is bit-identical when the flags are off (tested against the earlier commit). §07 showed that filtering on confidence leaves survivorship bias in place; censoring does not remove that bias either, it only stops the model being taught a value that was never measured. Data census: 5,114 dead and 4,312 hyperstable training singles, 6,796 dead conditional doubles; four libraries have a wild type that is itself out of range, so their `>5` variants are rank-only.
3. **A monotone saturating link** (`--link softclamp`, implemented, not yet run). Regression is done on the observed scale through one shared monotone function of the latent prediction; ranking is on the latent scale and is unchanged by it. This is the same reasoning as §03 applied to the regression head: the assay's response is monotone, so the model should not have to bend its latent scale to match it. It makes the separate scale/bias calibration head redundant (the link fixes the zero point, so `--shared_bias_init` is rejected with it).
4. **An interaction-only loss** (`--lambda_int_mt`, with `--flip_pair_groups` and `--flip_align_units`; implemented, not yet run): predictions and measurements are double-centred per position-pair matrix and compared. It is blind to row, column and pair-offset effects by construction, which are still learned through the ordinary regression. Measured yield on real batches: about 51 flip rows per micro-batch, 0.74 complete matrices and about 30 cells reaching the loss; 58% of flip rows land in a usable block. That is a thin signal per step, so a gain is not assured.

### 16.3 How much identity-dependent signal is there to learn

Measured, on 127,476 measured doubles (`docs/anova_epistasis_report.md`): identity-independent structure explains about **90%** of the out-of-sample variance of ΔΔΔG (assay saturation 60%, a per-pair offset 16%, row and column effects 13%). The leftover, 10.4%, is about the size of the measurement noise (replicate rows of the same mutant: SD 0.27 kcal/mol; trypsin versus chymotrypsin: 0.22–0.31), leaving roughly **0–5%, most likely about 2%**, for identity-specific interaction. That variance share is on the ΔΔΔG scale, whereas this note's statistic is rank-based within columns, so the two are not directly comparable; but both say the identity-specific part is small. It is consistent with §14's reading that the reported values are lower bounds, and it is a reason to expect modest returns from further training on this objective.

### 16.4 Runs in progress

The baseline guard (current code, none of the new flags, same seed as the reference run) is running first; then censoring alone, the link alone, and the link with censoring. The interaction-loss arms follow only if those do not hurt the guard metrics (`val_rho_wt_valid_avg`, `val_rmse_combined_avg`). The comparison metric is `val_rho_flip_pair_avg`, with at least two seeds per arm before a difference is believed. This section will be replaced by measured values when they exist. Plan and flags: `docs/supervisor_state.md`, `docs/epistasis_training_handoff.md` §2e–§2f.

Estimator and simulation battery: `ordering_test.py`. Core test docket: `testdocket.py` over `analysis_notebooks/predictions/hyperopt_splits-test`. All 95 reference models: `baseline_all.py` over `zero_shot/`. ProteinGym predictions from `pgym_results/unmasked-esm-msr-seed1/sigma1.0` (`lora_seed1.safetensors`, σ=1, unmasked); adapter-off predictions from `pgym_results/unmasked-esm-msr-small/sigma0.0`, which is the same pipeline at `--lora_epsilon 0`. Other baselines are ProteinGym's own reference set. Significance from 200 realignment permutations per DMS; paired comparisons by Wilcoxon signed-rank. The validation-split section reads `analysis_notebooks/predictions/hyperopt_splits-val` through the same `testdocket_seeds.py` with `PRED_DIR` pointed at it. No new inference was run for this report.