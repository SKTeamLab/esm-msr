# Findings: what the model learns about epistasis, and how we know

Measured with the **conditional-ordering test**. Full derivation, figures and baseline
sweeps: the methods report. This file is the compressed version and the reasoning behind
the code changes it motivated.

## 1. The measurement problem came first

Conventional ΔΔΔG correlation — `spearman_doubles_epi`, what the harness reported — cannot
distinguish a model that understands epistasis from one that has only learned the shape of
the assay. In simulation, with **zero** specific interaction on either side and only each
side's own saturating response:

| scenario (all nulls; honest answer is 0) | ΔΔΔG ρ | ordering test |
|---|---|---|
| no interaction either side, both saturate | **+0.903** | +0.000 |
| measurement interacts, model does not | +0.459 | +0.000 |
| both interact, but independently | +0.374 | −0.012 |
| hard dynamic-range floor | +0.268 | +0.003 |
| measurement cubically rescaled | **−0.251** | +0.000 |
| *shared interaction (true ρ = 0.898)* | +0.878 | +0.782 |

+0.903 under the null is indistinguishable from +0.878 with real shared interaction. Worst
null: 0.903 against 0.012, a factor of 75.

Three separate reasons ΔΔΔG fails:

1. **Nonspecific epistasis dominates.** 73% of ΔΔΔG variance on GRB2-SH3 abundance is a
   monotone function of the additive sum — it depends on *how big* the two effects are, never
   on *which residues* they are.
2. **The singles' measurement error is shared, not independent.** ΔΔΔG subtracts measured
   singles, and every double containing substitution X subtracts the *same* measurement of X.
   The error has the shape of a row effect plus a column effect and does not average away.
   Simulated variance inflation 1.7×.
3. **Position coupling is not identity coupling.** Buried pairs couple more on average; a
   model that has only learned that scores for it.

## 2. The test

If X and X′ do not interact with Y in an identity-dependent way, then **whichever of X and X′
is more destabilising is the same regardless of what sits at the partner position.** A
monotone assay response cannot change which of two numbers is larger, so that ordering holds
under the null whatever the assay did. An ordering flip is the one thing only
identity-dependent interaction can produce.

Statistic: rank within each column of the position-pair matrix, double-centre, correlate
measured against predicted. Under the null every column shares one ordering, the matrix is
constant along rows, and double-centring gives **exactly zero**.

Two numbers come out and must be kept apart:

* **signature SD** — how much identity-dependent structure a model predicts *at all*. An
  additive readout, or a fixed per-pair bias, gives exactly zero. Not a weak result; an
  undefined one.
* **ρ** — whether the structure it does predict is right.

What this controls for, as consequences of the construction rather than corrections:
nonspecific epistasis, the ΔG dynamic-range floor, shared model/assay saturation, each
mutation's own effect, position-pair coupling, and the singles' measurement error (never
read). Noise in the doubles attenuates only, so **every value is a lower bound**.

Two things it does not control for: a non-monotone assay response (would break it), and
indirect allosteric coupling (indistinguishable from contact).

**One implementation requirement that is not optional.** Matrices must be trimmed to a fully
complete sub-block before ranking. Cells go missing because the double failed to measure, so
missingness depends on both substitution effects and is shared between the measured and
predicted matrix; centring an incomplete matrix admits ρ up to **+0.95** with no interaction
present. Complete-block trimming returns exactly 0.0000 at every truncation level while
retaining real signal (0.799 vs 0.787).

## 3. Result: ESM-MSR does learn it, and leads the field

Full held-out test docket, 36 domains, 10,732 doubles in 35 near-complete pair matrices,
byte-identical rows, three training seeds where they exist.

| model | ρ (all cells) | ρ (measured ΔG ≥ 0) | signature SD |
|---|---|---|---|
| **ESM-MSR, MT pass only** | **+0.158 ± 0.007** | **+0.181 ± 0.010** | 0.0996 |
| **ESM-MSR, unmasked (released)** | **+0.141 ± 0.003** | **+0.164 ± 0.004** | 0.0567 |
| ESM-MSR, independence-masked | +0.130 ± 0.005 | +0.149 ± 0.009 | 0.0600 |
| Mutate Everything | +0.083 ± 0.011 | +0.085 ± 0.010 | 0.0996 |
| ProteinMPNN (unmasked, zero-shot) | +0.079 | +0.085 | 0.0921 |
| Rosetta `cartesian_ddg` | +0.044 ± 0.005 | +0.049 ± 0.003 | **0.1220** |
| ThermoMPNN-D (epistatic) | +0.044 ± 0.016 | +0.046 ± 0.019 | 0.0351 |
| SPURS | +0.044 ± 0.023 | +0.049 ± 0.019 | 0.0494 |
| ESM3 base, adapter off | +0.037 | +0.042 | 0.0505 |
| ESM-MSR masked-marginal *(control)* | **0.000** | 0.000 | **0.0000** |
| ESM-MSR WT pass only *(control)* | **0.000** | 0.000 | **0.0000** |
| ThermoMPNN-D additive *(control)* | −0.000 ± 0.001 | −0.003 ± 0.003 | 0.0016 |

Three additive controls from independent sources return **exactly zero** — the test behaving
as the construction requires, on real data, three times over.

On the two domains verified held out by sequence identity (`test_internal`: 1UFM, 3DKM),
ESM-MSR reaches **0.146**, which is **2.8× the best of 95 ProteinGym reference models**
(ESM-IF1, 0.053).

Other readings worth keeping:

* **54 of 95 ProteinGym reference models cannot express this quantity at all** (signature
  SD ≈ 0.010, median ρ = −0.0001). Their readout is additive across the two mutated
  positions. Includes ESM2 at every size, ESM1b, ESM1v, ESMC, xTrimoPGLM-MLM, ProSST,
  ProtSSN, SaProt, GEMME, VESPA, ESCOTT, SiteRM, VenusREM, MIF/MIF-ST — and ProteinGym's own
  ESM3 column.
* **The readout decides expressiveness, not the backbone.** ProteinGym's site-independent
  ESM3 scores −0.006; the same backbone through the MSR two-pass readout reaches 0.094.
* **Rosetta predicts the most identity-specific structure of any entry** (signature SD 0.120)
  and converts the least of it into agreement.
* **The capability lives in the structure channel.** Every sequence-only model is at zero; the
  only non-zero baselines are structure-conditioned (ProteinMPNN 0.079, ESM-IF1 0.053).
* **Declines with Cβ separation** (0.237 / 0.190 / 0.102 across 0–6 / 6–10 / 10–15 Å,
  Spearman −0.26), consistent with a contact-mediated mechanism. Base ESM3 is flatter
  (−0.16), so training sharpens the distance dependence as well as the level.
* **Models sharing a name may not share a method.** Our ProteinMPNN and ProteinGym's agree at
  only ρ ≈ 0.83 on identical variants and differ fourfold in signature SD.

## 4. The WT pass degrades the interaction channel

The MT pass alone beats the released ensemble. The WT pass has a flip signature of *exactly
zero* — it is a sum of per-position LLRs on the wild-type background, additive by
construction, carrying no identity-dependent information true or false. But an additive term
is not neutral for this statistic: adding `0.5·(a_X + b_Y)` leaves `b_Y` harmless (constant
within a column) while `a_X` shifts each row differently and **reorders substitutions within
every column**.

| blend | ρ | signature SD |
|---|---|---|
| pure MT | **+0.150** | 0.1006 |
| w = 0.25 | +0.144 | 0.0839 |
| w = 0.50 (released `combined_pred`) | +0.142 | 0.0605 |
| w = 0.75 | +0.101 | 0.0369 |
| pure WT | +0.000 | 0.0000 |

It **distorts rather than dilutes**: the combined signature correlates with the MT signature
at only **ρ = 0.55**. Roughly half the structure is scrambled by the admixture.

This is a trade-off, not a defect — the half-and-half average is the mean of the two
thermodynamic paths and is exact for ΔΔG under pairwise epistasis. It is simply the wrong
place to read epistasis from. Good region for the interaction channel is w ∈ [0, 0.25].

## 5. Sub-floor doubles: bad data and biased data are different things

The Tsuboyama release has **no quality flag** (only `fitting_error_t/c`, which are fit
residuals, and `Stabilizing_mut`, a classification). Of 47,967 doubles with both singles
measured, **61.8%** have an additive-predicted ΔG below the practical floor (+0.5), **47.6%**
below 0, **23.4%** below the stated −1 bound. All carry a numeric `ddG_ML` with nothing
marking them.

Within the sub-floor set (additive ΔG < 0, n = 22,840), the per-protease CI plus measured ΔG
splits it three ways:

| class | criterion | n | share | mean ΔΔΔG | median CI |
|---|---|---|---|---|---|
| Genuine compensation | CI ≤ 0.5, ΔG ≥ 0.5 | 7,456 | 32.6% | **+1.86** | 0.25 |
| Unidentifiable (bad data) | CI > 0.5 | 10,298 | 45.1% | +1.42 | 1.15 |
| Genuinely unfolded | CI ≤ 0.5, ΔG < 0.5 | 5,086 | 22.3% | +1.40 | 0.34 |

**All three classes show a large positive mean ΔΔΔG, and genuine compensation is the largest
of the three.** Filtering on confidence removes the fabricated values and leaves the
survivorship bias completely intact. Apparent stabilising epistasis at the floor is not
purely artefact — part of it is real compensation visible *because* it is real.

How the two modes affect the test: **pinning is harmless** (ρ = 0.0000 even with 74% of cells
given an unidentifiable value — a pinned value either preserves within-column order or
randomises it). **Truncation is what breaks it**, and complete-block trimming is the fix (§2).

Excluding measured ΔG < 0 at evaluation discards 16.8% of cells and raises ESM-MSR **16%**
(0.141 → 0.164) while moving competitors 2%. The unreliable cells were *diluting* the model's
measured accuracy, not inflating it.

## 6. What this changed in the code

| change | why |
|---|---|
| `--lambda_rank_mt` (new, default **1.0**) | Within-column rank loss on the MT pass. Invariant to the monotone assay response and the floor by construction, so it cannot be satisfied by learning saturation — which the regression term can and does. |
| `--lambda_reg_mt` default **0.0 → 1.0** | Only term giving the MT pass an absolute scale; `combined_pred` is uncalibrated without it. Matches the canonical command, which always passed 1.0. |
| `--flip_list_min` (new, default 4) | Below ~4 members a column's ordering is mostly noise. |
| `--subfloor_rank_only` (new, default on) | Sub-floor double-derived items are now **marked** `reg_ok=False` rather than dropped: kept in the rank losses, withheld from the regression. A third of them are real data (§5). |
| `flip_key` on `cond` / `native_cond` items | Groups items into flip columns. Within a column the conditional target `ddG(A|B)` differs from `ddG_AB` only by the constant `ddG_B`, so ordering by either is identical. |
| `val_rho_flip` (new metric) | `val_rho_combined_avg` is blind to this channel (§1), so MT hyperparameters were being tuned against a metric indifferent to their purpose. Double-centred, hence exclusive to identity-dependent interaction. |
| `epistasis_pred` (new inference column) | The MT-pass epistasis. `combined_pred` stays the calibrated ΔΔG output (§4). |

**Honest caveat on the loss.** Under strict additivity every column shares one ordering, so a
purely additive model already satisfies much of `--lambda_rank_mt`; what it cannot satisfy is
the per-column *deviation* from that consensus. The loss is artifact-immune and contains the
interaction term but is not exclusively about it. `val_rho_flip` double-centres the consensus
away and **is** exclusive. Train on the loss; judge on the metric.

## 7. Not implemented, and why

* **Contact/severity weighting.** Coupling SD is 1.54 kcal/mol at contact when both
  substitutions are individually worth ≥1 kcal/mol, against 0.30 for distant or mild pairs, so
  the learnable signal is concentrated. Needs per-item Cβ distance, which the cache does not
  store. Worth adding; would require a cache field.
* **`--min_measured_dG`.** Needs the measured ΔG of each double in the cache; only
  `ddG_A`/`ddG_B` are stored. `--min_additive_dG` plus `--subfloor_rank_only` covers most of
  the value.
* **Structure-encoder adaptation.** Left off by request. `ESM3.forward` never calls the
  encoder, so adapters there receive no gradient; enabling it means moving encoding into the
  forward pass, which the unmasked cache now makes possible. The evidence that it is worth
  doing is strong (§3, structure channel).
* **ProteinMPNN ensembling — tested, did not work.** Its signature correlates with the MT
  pass at only ρ = 0.149, so the content is genuinely complementary, but every linear blend is
  worse than the MT pass alone (0.148 at 25% MPNN, 0.142 at 50%, against 0.150). Same
  mechanism as §4. Would need feature-level fusion or a learned gate.
