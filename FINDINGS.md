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
| `--flip_group_units` (new, default on) | Orders MT work-unit rows by flip column. Measured 2.9× more items into the loss (11.8 vs 4.1 per step) and the loss fires on every step rather than 86% of them. |
| `--censor_floor` (new, default off; needs cache v6) | Censored Plackett–Luce for the flip loss: items measured at/below the floor are tied at the bottom — above-floor items must still beat them, but their order among themselves is free. Training on that order fitted noise (§5). Implemented, awaiting a GPU comparison (handoff §2d). |
| `--include_out_of_range` (new, default off; cache v7) + consolidated censoring (`esm_msr/censoring.py`) | Variants the assay reports only as `<-1` (dead) or `>5` (hyperstable) were being dropped by numeric parsing (about 3.7% and 8.3% of single-scan variants; 38% of doubles with additive prediction below -3). They are now kept as censored items: tied at the bottom / top in two-sided censored Plackett-Luce, and one-sided (hinge) in regression. Used by the WT head (singles) and the MT head (doubles). Subsumes the earlier `censored` flag; handoff §2e. |
| `val_rho_flip_pair` (new), `--link softclamp` (new), `--lambda_int_mt` with `--flip_pair_groups` / `--flip_align_units` (new) | From the ANOVA report: a per-pair flip metric that nothing ignoring the partner residue can score on; a monotone saturating link so the assay's saturation is not imitated by the adapters; and an interaction-only (double-centred) loss. All default off, handoff §2f. |
| `epistasis_pred` (new inference column) | The MT-pass epistasis. `combined_pred` stays the calibrated ΔΔG output (§4). |

**Honest caveat on the loss.** Under strict additivity every column shares one ordering, so a
purely additive model already satisfies much of `--lambda_rank_mt`; what it cannot satisfy is
the per-column *deviation* from that consensus. The loss is artifact-immune and contains the
interaction term but is not exclusively about it. `val_rho_flip` double-centres the consensus
away and **is** exclusive. Train on the loss; judge on the metric.

## 6b. Column-aware batch sampler (Resolved)

Implemented in `ProteinCyclingBatchSampler` (`esm_msr/data.py`). The sampler uses a dual-queue
bin-packing strategy: whole flip columns are packed intact into batch slots, and single mutants
are packed into remaining residual slots up to the token budget. This eliminates within-column
fragmentation across micro-batches, increasing the active items in the flip loss from ~12 to ~35-64
per step and making `--flip_list_min 4` consistently saturated.

## 6c. Empirical training results: Censored ListMLE and extended training

Systematic evaluation on `cache_v6` (anchor 1.0, seed 1 unless noted) across sequential runs:

| Run / Arm | Key Flags | Epochs | Peak `val_rho_flip_avg` | Peak `val_rho_epi_avg` | Best `val_rmse_combined_avg` | Key Finding |
|---|---|:---:|:---:|:---:|:---:|---|
| **A' (Uncensored Ref)** | `--lambda_rank_mt 1.0` | 3 | 0.213 (Ep 2) | 0.378 | 0.686 | Seed 1. Seed 2 (7 epochs): 0.173 at Ep 2, peak 0.219 (Ep 4), 0.202 at Ep 6 |
| **B' (No-Flip Control)** | `--lambda_rank_mt 0.0` | 4 | 0.228 (Ep 3) | 0.372 | 0.679 | Seed 1. 0.204 / 0.207 / 0.205 at Ep 0-2, then 0.228 at Ep 3 (the Ep 3 row is in the resumed run's `metrics.csv`; Ep 0-2 are in `metrics_backup_ep2.csv`) |
| **F0 (Censor Floor 0.0)** | `--censor_floor 0.0` | 8 | **0.227** (Ep 5) | **0.418** (Ep 2) | **0.671** (Ep 6) | 10.3% censored; highest direct epistasis & best RMSE |
| **F2 (Censor Floor 0.5)** | `--censor_floor 0.5` | 8 | **0.237** (Ep 5) | 0.396 (Ep 1) | 0.682 (Ep 4) | 24.5% censored; **all-time project record on flip metric** |

> **Metric rename.** `val_rho_epi_avg` in this table is the *old* single-head readout
> `comb - wt = 0.5*(mt - wt)` and is now logged as `val_rho_epi_fast_avg`. The new
> `val_rho_epi_full_avg` scores `comb_AB - comb_A - comb_B`. They differ by exactly
> `0.5*(delta_A + delta_B) - dW` (delta_X = mt_X - wt_X on singles, dW the WT head's
> non-additivity, ~constant), so `fast` is contaminated by any MT/WT disagreement on singles
> (which `--mt_single_anchor_*` controls) and `full` is not. Compare arms with different anchor
> settings on `full`. Neither separates identity-specific epistasis from nonspecific saturation
> (section 1); `val_rho_flip_avg` does. `esm_msr_testing.py` writes `*_DeltaSingles.csv` with the
> measured disagreement and a check of this identity.

> **Correction (earlier wording overclaimed).** This section previously concluded that the flip loss
> beats the released model and that "floor censoring is a decisive win". The data do not support
> either, because every comparison was confounded with training length and the control was never
> run to the same length. What the logs actually show, epoch-matched on `val_rho_flip_avg`:
>
> | epoch | B' no-flip (seed 1) | A' flip (seed 1) | A' flip (seed 2) |
> |---|---|---|---|
> | 0 | 0.204 | 0.174 | 0.142 |
> | 1 | 0.207 | 0.176 | 0.162 |
> | 2 | 0.205 | 0.213 | 0.173 |
> | 3 | 0.228 | n/a | 0.219 |
>
> 1. **The rank loss does not clearly help, and may slow early learning.** The no-flip control is
>    ahead at epochs 0-1 and level by epochs 2-3. Seed 2 of the flip arm swings by about 0.03 between
>    consecutive epochs (0.219 then 0.186), the same size as the gaps between arms, and there is one
>    seed for most arms. Nothing here is a resolved difference.
> 2. **Censoring is not shown to help.** F0 (0.227) and F2 (0.237) peak at Ep 5. The no-flip control
>    was never run past Ep 3 (0.228), so there is no matched baseline for those epochs. The F0 versus F2
>    ordering is within seed noise. The census figures (10.3% and 24.5% of flip items censored) are
>    descriptive only.
> 3. **"Beats the released model (0.198)" is not a like-for-like claim.** The 0.198 is the released
>    checkpoint on its own validation docket and these values are on this split's validation
>    libraries, which the handoff already says are not comparable.
> 4. **Training length is the one effect visible in the data**: every arm, including the no-flip
>    control, improves between Ep 0-2 and Ep 3-5.
>
> **Why the flip loss may not add much (diagnosis, not yet tested).**
> * *Gradient imbalance.* `norm_grad/lora_mt` is about 1-86 (typically about 30) with the rank loss
>   and about 0.3-0.5 without it, and `L_rank_mt` (about 19-35) dwarfs `L_reg_mt` (about 0.2-1.7). At
>   `--lambda_rank_mt 1.0` the rank loss dominates the MT adapter's gradient and is noisy, which would
>   explain an early deficit. The loss itself is working: it starts at chance level (about 35, the
>   ln(n!) for n about 14) and falls to about 19.
> * *Redundancy with the regression.* Conditional-target regression already carries the within-column
>   ordering, and `--subfloor_rank_only` already keeps floor-pinned items out of it, so the rank loss's
>   original advantage (immunity to the floor) is largely captured. On `cache_v6` (4,225 columns of at
>   least 6 members) a per-substitution consensus predictor reaches mean Spearman 0.78 with the
>   within-column order (about two-thirds of the rank variance; an upper bound, since the consensus
>   includes the column itself and pinned floor values agree). The loss is mostly rewarding the
>   additive ordering, and the interaction part is the smaller, noisier remainder.
>
> **To settle it:** run the no-flip control to 8 epochs, 2-3 seeds per arm, and sweep
> `--lambda_rank_mt` at 0.03 and 0.1 before concluding the loss or censoring has any effect. If it stays flat, target
> the interaction-specific part: rank the deviation from the consensus, or up-weight discordant pairs.

Key lessons that do hold:
1. **Training length matters**: every arm improves from Ep 0-2 to Ep 3-5.
2. **The loss is active and well-formed**, but its benefit over regression alone is unproven.


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
