# ProteinGym Activity: does a suggested mutation hurt activity?

Screening curves for the 43 ProteinGym assays in the **Activity** category (`coarse_selection_type == 'Activity'` in `ProteinGym/DMS_substitutions.csv`), single mutants only. The wild-type (WT) score of every assay is taken from the original measurements instead of being estimated from the score distribution, and a variant **hurts** when it lies **more than half an assay standard deviation below WT** (the assay SD is the standard deviation of all its single-mutant scores, in the assay's own units). The 18 enzyme assays are shown on their own and together with the 25 other function assays. ESM-MSR is FINAL_p0 seed 1 (the WT head on singles); the other models are drawn exactly as in `08_screening.py`.

```bash
python analysis/pg_activity/01_wt_reference.py          # WT and floor table from the raw DMS files (needs <root>/ProteinGym/substitutions_raw_DMS); writes data/wt_reference.csv
python analysis/pg_activity/02_activity_screening.py    # figures and tables in results/   (--B 200 bootstrap replicates; CPU, about 30 s)
```

## Summary

1. **A WT score exists for 36 of the 43 assays, including all 18 enzymes.** The curated `DMS_score` files carry no WT row, but the raw files do (11 assays have an explicit WT row, 17 have synonymous-codon rows), and for 7 enzymes and AICDA the original papers fix it. Seven non-enzyme assays have no WT anywhere (A0A1I9GEU1, A0A247D711, ADRB2, CCDB, PPARG, HRAS, SRC dasatinib) and keep the old plateau estimate, marked as such. The old estimate missed the true WT by more than one noise sd in 9 of the 35 assays that have one (`results/wt_vs_plateau`).
2. **The cutoff is one number with no units and needs no replicates**: half an assay SD. If the measurement noise is at most 10% of the assay variance (noise SD at most 0.32 assay SD), half an SD is at least 1.6 times the noise; with the best noise estimates available (the paper's, the synonymous-codon spread, else the spread of the functional plateau: the "ladder" below) it is a median 1.4 noise SD, and at least 2 noise SD for 33% of the assays, so it marks a drop that is probably real, not a certain one. Random picks hurt on average 50% of an enzyme assay's variants (46% over all assays).
3. **Choosing the top-ranked variants avoids about two-fifths of the hurting picks.** On the enzymes ESM-MSR's top-100 picks hurt 0.30 of the time, 40% below random; ThermoMPNN, SPURS, ProteinMPNN and ProSST are higher at K = 100 (paired over assays, 95% intervals exclude 0), MutateEverything is borderline, and ESM3-small-open is slightly lower. A cutoff of 5% or 10% of the variance as noise (0.45 or 0.63 assay SD) shifts the random level and every curve and leaves the order as it is.
4. **Floors are too scarce and too close to WT to define "hurts" as a share of the WT-to-floor distance** (checked, not used): a measured floor exists for 17 of the 43 assays with a measured WT (12 of 18 enzymes) and is more than three noise sd from WT in only 9; see *Floors*.

## What is in the folder

| Path | Content |
|---|---|
| `01_wt_reference.py` | reads each assay's raw DMS file, maps the raw phenotype column onto the curated score by a fit over the shared single mutants (identity, or a sign flip for CCDB, RPC1, S22A1), and extracts the explicit WT rows, the synonymous-codon rows and the stop-codon rows (count, median, spread); merges `data/wt_manual.csv` |
| `02_activity_screening.py` | the hurt rule, the curves, the landscape, WT-versus-plateau and floor-audit figures, the paired and sensitivity tables |
| `data/wt_reference.csv` | one row per assay: WT, tier, source, synonymous spread, stop-codon null and its spread, counts, and the raw file and column used |
| `data/wt_manual.csv` | the WT, null and noise values taken from the papers (OTC, PTEN, AICDA, CASP3, CASP7, SRC, RNC, UBE4B), each with its source and the evidence (what the paper's score is, how WT is defined, what was checked against the raw data) |
| `data/aicda_wt_by_position.csv` | AICDA: the WT-residue rows of MaveDB urn:mavedb:00000106-c-1 (11 positions), which ProteinGym dropped |
| `results/screening_{enzymes,all}{,_all_models}` | **the curves** (+ `.csv`): hurts = more than 0.5 assay SD below WT; `_all_models` adds ProSST, ProteinMPNN and ESM3-small-open |
| (`../results/16_screening_overview/screening_overview_{single,multi}*`) | the Activity curves (half-SD rule, all assays and enzymes) next to the Tsuboyama and Domainome curves, drawn by `../16_screening_overview.py` from the `.csv` files here |
| `results/paired_vs_esm_msr.csv` | model − ESM-MSR hurt rate at K = 10 and 100, paired over assays, 95% bootstrap interval (10000 replicates), for the half-SD rule, the 5% and 10% variance rules and the two other definitions |
| `results/assay_definitions.csv` | per assay: class, WT, the assay SD and the sd of each rule, floor and its evidence, window, the share that hurts under each definition, the old plateau estimate, the ProteinGym cutoff |
| `results/landscape` | score distribution of every assay with WT, the half-SD threshold (hurting region shaded), the floor, the ProteinGym cutoff and the old estimate |
| `results/wt_vs_plateau` | the old estimate minus the measured WT, in noise sd |
| `results/floor_audit` | which assays have a floor and how far it is from WT in noise units |
| `results/sensitivity.csv` | the curves at K = 1, 10, 100 under 17 variants of the definition (floor, ladder, subsets, the earlier plateau-based ones) |

## Definition

For one assay, with score x (the curated `DMS_score`, higher = more active) and wild-type score WT, a variant **hurts** when x < WT − 0.5 × SD, where SD is the standard deviation of all the assay's single-mutant scores (`assay_sd` in `results/assay_definitions.csv`). The cutoff has no units, so it means the same on a 0 to 1 scale and on an assay measured in the tens of thousands, and it needs no replicate noise. Read as noise: if the measurement noise is a share p of the assay variance, its SD is √p × SD, so half an assay SD is 0.5/√p noise SD: 1.6 at p = 10%, 2.2 at p = 5%.

**A screen of K variants** takes the K variants of the assay with the highest model score (the models predict stability, higher = more stabilizing, and are not trained for activity) and reports the fraction of them that hurt. Curve = the mean over the assays; the dashed line is the share of *all* variants that hurt (the expectation of random picks); the shaded band is the 95% bootstrap interval over assays (200 replicates while drafting), drawn for ESM-MSR only (the `.csv` files hold the interval of every model; the paired tables below compare ESM-MSR with each model).

**Other definitions** are kept for comparison (`sensitivity.csv`, `paired_vs_esm_msr.csv`):

| Name | Rule | Assays |
|---|---|---|
| var5, var10 | x < WT − 2 sd with sd² = 5% (10%) of the assay variance, i.e. a drop of more than 0.45 (0.63) assay SD: the noise is a share of the total variance | all 43 |
| ladder | x < WT − 2 sd, sd = the first available of: the noise stated in or derived from the paper (PTEN 0.51, CASP3 0.47, CASP7 0.45, SRC 0.27 from the paper's neutral band −0.46 to +0.63, RNC 0.05) > the spread (1.4826 × MAD) of the synonymous-codon rows when they are independent measurements (≥ 20 rows, ≥ half of the values distinct) > the spread of the functional plateau (the peak of the density of the scores above the ProteinGym cutoff, and 1.4826 × MAD of those scores around it) | all 43 |
| floor | (WT − x) / (WT − floor) > 5%: it lost more than 5% of the way from WT down to the floor | measured WT and measured floor: 17 (12 enzymes) |

### WT

| Tier | Assays | Source |
|---|---|---|
| A (11) | D7PM05, ENVZ, GFP_AEQVI, MET, PHOT, Q6WV12, Q8WTC7, RPC1 ×2, S22A1, VKOR1 | an explicit WT row in the raw file |
| B (17) | AMIE, ANCSZ, CAS9, HEM3, KCNE1, KCNH2, KCNJ2, LGK, NPC1 ×2, OXDA, PAI1, Q59976, Q837P4, Q837P5, RL40A, SC6A4 | median of the synonymous-codon rows (an amino-acid identity such as `A12A`) |
| C-paper (7) | CASP3, CASP7, OTC, PTEN, RNC, SRC (Ahler), UBE4B | the original paper (`data/wt_manual.csv`) |
| C-position (1) | AICDA | the WT residue of each position (`data/aicda_wt_by_position.csv`); positions 118 to 120 (57 of 209 variants), whose WT residue has fewer than 100 selected reads, are left out |
| C-open (7) | A0A1I9GEU1, A0A247D711, ADRB2, CCDB, PPARG, HRAS, SRC (dasatinib) | none found; the old plateau estimate is used, marked in red in the landscape |

The paper-derived values were found by one literature agent per assay and every claim was checked against the raw data (counts, the MaveDB table, the PNAS Dataset S1) before it was accepted. For most enzymes the paper does not print a WT number, because the score is *defined* relative to WT (a log-ratio to WT, or a regression coefficient relative to the WT residue); WT = 0 is then exact, but it is a statement about the parameterisation, not a measurement. Only UBE4B (+0.09, the no-mutation row of Dataset S1) and OTC (1, with the null at 0) are numbers on the paper's own scale.

## Results

Draft bootstrap (B = 200). Share of the K picks that hurt (x < WT − 0.5 assay SD), mean over assays (lower is better). Intervals are in `results/screening_*.csv`.

**Enzymes (18 assays)**

| Model | K=10 | K=100 |
|---|---|---|
| ESM-MSR | 0.33 | 0.30 |
| MutateEverything | 0.32 | 0.35 |
| SPURS | 0.37 | 0.36 |
| ThermoMPNN | 0.42 | 0.40 |
| ESM3-small-open | 0.32 | 0.27 |
| ProteinMPNN | 0.39 | 0.37 |
| ProSST | 0.44 | 0.38 |
| Rosetta | 0.54 | 0.45 |
| random picks | 0.50 | 0.50 |

**All 43 assays**

| Model | K=10 | K=100 |
|---|---|---|
| ESM-MSR | 0.32 | 0.29 |
| MutateEverything | 0.32 | 0.33 |
| SPURS | 0.36 | 0.34 |
| ThermoMPNN | 0.42 | 0.37 |
| ESM3-small-open | 0.25 | 0.26 |
| ProteinMPNN | 0.35 | 0.35 |
| ProSST | 0.42 | 0.33 |
| Rosetta | 0.51 | 0.41 |
| random picks | 0.46 | 0.46 |

**Paired over assays**: model minus ESM-MSR, positive = ESM-MSR hurts less; 95% interval over assays in brackets, \* = excludes 0.

*Enzymes*

| Model | K=10 | K=100 |
|---|---|---|
| MutateEverything | −0.01 [−0.09, +0.07] | +0.05 [−0.00, +0.09] |
| SPURS | +0.04 [−0.08, +0.14] | +0.07\* [+0.03, +0.11] |
| ThermoMPNN | +0.08 [−0.04, +0.21] | +0.11\* [+0.06, +0.15] |
| ESM3-small-open | −0.01 [−0.09, +0.07] | −0.03\* [−0.05, −0.00] |
| ProteinMPNN | +0.06 [−0.05, +0.17] | +0.07\* [+0.02, +0.12] |
| ProSST | +0.11 [−0.05, +0.24] | +0.09\* [+0.02, +0.15] |
| Rosetta | +0.21\* [+0.09, +0.33] | +0.16\* [+0.09, +0.21] |

*All assays*

| Model | K=10 | K=100 |
|---|---|---|
| MutateEverything | +0.00 [−0.06, +0.06] | +0.04\* [+0.01, +0.07] |
| SPURS | +0.03 [−0.03, +0.10] | +0.05\* [+0.02, +0.07] |
| ThermoMPNN | +0.10\* [+0.02, +0.17] | +0.07\* [+0.04, +0.11] |
| ESM3-small-open | −0.07\* [−0.14, −0.01] | −0.04\* [−0.06, −0.02] |
| ProteinMPNN | +0.03 [−0.05, +0.10] | +0.05\* [+0.02, +0.09] |
| ProSST | +0.10\* [+0.00, +0.19] | +0.04 [−0.00, +0.08] |
| Rosetta | +0.19\* [+0.12, +0.27] | +0.12\* [+0.08, +0.16] |

How to read it:

* **Against random.** Random picks hurt 0.50 of an enzyme assay's variants (0.46 over all assays). ESM-MSR's top picks hurt 0.33 at K = 10 and 0.30 at K = 100 on the enzymes, 33% and 41% below random (0.29 on all assays at K = 100, 36% below).
* **Models at K = 100.** ESM-MSR hurts less than ThermoMPNN, SPURS, ProteinMPNN and ProSST on the enzymes, and less than ThermoMPNN, SPURS, ProteinMPNN and MutateEverything on all assays (the paired intervals exclude 0); MutateEverything on the enzymes (+0.05) and ProSST on all assays (+0.04) just include 0. ESM3-small-open is lower than ESM-MSR by 0.03 on the enzymes (interval just excluding 0) and by 0.04 on all assays.
* **K = 10 and K = 1.** At K = 10 the paired intervals mostly include 0 on the enzymes; on all assays ThermoMPNN (+0.10) and ProSST (+0.10) hurt more than ESM-MSR and ESM3-small-open less (−0.07). One pick (K = 1) is one event per assay and is not interpreted.
* **The ESM3 column** is the published open model, not the unmasked σ = 0 read used elsewhere.
* **Rosetta** is the Cartesian ddG of `baselines/public/rosetta/raw_narval/proteingym_cartesian_preds_09_11.csv` (ProteinGym singles, already stabilizing-positive, used as is; mean ρ with the measured score +0.64 on the Stability assays, +0.31 on the Activity assays, positive for all 43). It scores 96.7% of the Activity singles (100% for the 25 non-enzyme assays; ANCSZ only 17%, CASP3, CAS9 and Q59976 94 to 97%); variants it does not score are skipped. Its top picks hurt *more* than random at small K (the curve lies above the dashed line up to K ≈ 20 on all assays), so the stabilizing extreme of Rosetta's ranking is not where activity survives. GeoDDG-3D is not drawn in these figures.
* **Other cutoffs** (5% and 10% of the variance: 0.45 and 0.63 assay SD) move the random level (0.48 and 0.42 over all assays) and every curve with it, and leave the order of the models as it is (`sensitivity.csv`, `paired_vs_esm_msr.csv`).

## The enzymes, one by one

Scales are as in the raw file after mapping to the curated score; the sd is the ladder's (the assay SD and the sd of the variance rules, `sd_var5` and `sd_var10`, are in `results/assay_definitions.csv`).

| Assay | Scale and what WT is | WT | ladder sd (source) |
|---|---|---|---|
| AICDA | enrichment f_sel / f_init within each position's own library (not normalised to WT) | per position: 18.4, 20.8, 11.7, 21.0, 8.3 (113–117), 0.86 (121), 17.7 (122), 1.5 (123) | 1.14 (plateau) |
| AMIE | normalised fitness, log-ratio-like | 0 (synonymous rows pinned at 0) | 0.13 (plateau) |
| ANCSZ | DMS_value | −0.02 (299 synonymous rows) | 0.046 (synonymous) |
| CAS9 | log2 fold change after positive selection | −0.18 (2777 synonymous rows) | 0.38 (synonymous) |
| CASP3 | PU-regression log-odds coefficient relative to the WT residue | 0 by construction | 0.47 (per-variant se, paper) |
| CASP7 | same as CASP3 | 0 by construction | 0.45 (paper) |
| ENVZ | mean_on (raw units) | 3.85 (explicit WT row) | 0.089 (plateau) |
| HEM3 | activity normalised to WT | 0.98 (283 synonymous rows) | 0.17 (synonymous) |
| LGK | selection log-ratio | 0 (synonymous rows pinned at 0) | 0.29 (plateau) |
| MET | IL-3 withdrawal score | 0 (explicit WT row) | 0.38 (synonymous) |
| OTC | growth score: arg3Δ = 0, WT yOTC = 1 | 1 by construction (no replicate sd reported) | 0.23 (plateau) |
| OXDA | activity fitness | −0.02 (331 synonymous rows) | 0.045 (synonymous) |
| PTEN | Enrich2 WT-normalised log fitness | 0 by construction (synonymous 95% range −1.1 to 0.9) | 0.51 (paper) |
| Q59976 | enrichment; the scale tops out at WT (only 5.6% of variants lie above it) | 0.004 (501 synonymous rows) | 0.28 (plateau) |
| RNC | log10 enrichment relative to synonymous WT alleles; saturates at about 25% of WT activity | 0 by construction | 0.05 (paper) |
| SRC (Ahler) | yeast-toxicity activity score, Enrich2 | 0 by construction | 0.27 (paper's neutral band) |
| UBE4B | log2 ratio | +0.09 (the 'NA-NA' no-mutation row of Dataset S1) | 0.67 (plateau; no noise reported) |
| VKOR1 | activity score | 0.91 (explicit WT row) | 0.20 (synonymous) |

Share of each assay's variants that hurt under the rule (0.5 SD), under the variance rules (5%, 10%) and, for comparison, under the ladder and the floor definition, with the floors:

| Assay | hurts: 0.5 SD | 5% | 10% | ladder | floor def. | floor (source) | window / sd |
|---|---|---|---|---|---|---|---|
| AICDA | 84% | 84% | 84% | 74% | 95% | 0.022 (paper) | 12.9 |
| AMIE | 53% | 55% | 48% | 47% | – | none | – |
| ANCSZ | 33% | 35% | 28% | 33% | 55% | −0.491 (stop codons, n=151) | 2.1 |
| CAS9 | 30% | 33% | 24% | 8% | 57% | −0.369 (stop codons, n=521) | 0.4 |
| CASP3 | 33% | 34% | 30% | 13% | 45% | −0.713 (stop codons, n=77) | 1.3 |
| CASP7 | 46% | 47% | 42% | 31% | 58% | −1.33 (stop codons, n=92) | 2.1 |
| ENVZ | 33% | 34% | 31% | 41% | 45% | 1.17 (stop codons, n=60) | 5.1 |
| HEM3 | 61% | 63% | 56% | 52% | 76% | 0.0018 (stop codons, n=288) | 5.7 |
| LGK | 75% | 76% | 70% | 52% | – | none | – |
| MET | 42% | 43% | 39% | 49% | – | none | – |
| OTC | 74% | 76% | 70% | 57% | 86% | 0 (paper) | 4.3 |
| OXDA | 68% | 69% | 63% | 47% | – | none | – |
| PTEN | 40% | 42% | 36% | 33% | 60% | −3.4 (paper) | 6.7 |
| Q59976 | 65% | 69% | 55% | 54% | – | none | – |
| RNC | 28% | 29% | 27% | 38% | 44% | −1.34 (stop codons, n=225) | 2.99 |
| SRC | 49% | 51% | 44% | 47% | 64% | −1.61 (stop codons, n=134) | 5.7 |
| UBE4B | 47% | 49% | 40% | 24% | 69% | −2.57 (paper) | 4.0 |
| VKOR1 | 35% | 36% | 32% | 28% | – | none | – |

The half-SD rule and the ladder agree within 0.1 for 10 of the 18 enzymes; they differ most where the ladder's sd is a large share of the variance (CAS9, CASP3: 37% and 48%, so the ladder is far more lenient) or a small one (ENVZ, RNC: 0.6%, where it is stricter than a fixed share of the variance would be).

## Floors

A floor is the score of variants that carry no activity. It is taken from the stop-codon variants of the raw file when there are at least 10 (median), else from a null stated by the paper (OTC: arg3Δ = 0; AICDA: stop codons 0.022; PTEN −3.4, inferred from the high-confidence loss-of-function variants; UBE4B −2.57, 33 stop codons of Dataset S1). Assays whose WT is only a plateau proxy are excluded. The floor was examined as a possible reference for a threshold ("hurts = lost more than 5% of the way from WT to the floor") and set aside.

**How many are there?** 19 of the 43 assays have one, 17 of them with a measured WT (`results/floor_audit`):

* enzymes, 12 of 18: ANCSZ, CAS9, CASP3, CASP7, ENVZ, HEM3, RNC, SRC (stop codons); AICDA, OTC, PTEN, UBE4B (paper). None for AMIE, LGK, MET, OXDA, Q59976, VKOR1: these are log-ratio or normalised scores with no null variant in the data and an open-ended low tail;
* other function and fluorescence assays, 5 of 25: RL40A, SC6A4, D7PM05, PHOT, Q6WV12 (plus two with a proxy WT, A0A1I9GEU1 and SRC dasatinib);
* a floor read off the data (at least 5% of the variants within one sd of the lowest score, i.e. a censored or saturated lower bound; `frac_within_sd_of_min`) would add OXDA (11%), KCNH2 (23%), Q8WTC7 (10%) and KCNE1 (5%); the other 20 assays without a measured floor show no pile-up.

**Are they well defined?** Take the window (WT − floor) in units of the noise at WT (the ladder sd, combined with the stop codons' own spread). Nine assays reach 3 or more: AICDA 13, PTEN 6.7, SRC 5.7, HEM3 5.7, ENVZ 5.1, OTC 4.3, UBE4B 4.0 (enzymes), D7PM05 9.6 and RL40A 7.8; RNC just misses (2.99); below it are ANCSZ 2.1, CASP7 2.1, PHOT 1.7, SC6A4 1.4, CASP3 1.3, Q6WV12 1.3 and CAS9 0.4 (its stop codons lie barely below the synonymous variants). The stop codons themselves spread widely (ANCSZ 0.22, CAS9 0.30, CASP7 0.45, RNC 0.44, SC6A4 1.5), so the floor is a fuzzy level, not a point.

**Is 5% of the way reasonable?** No: 5% of the window is below one noise sd in 14 of the 17 assays (the exceptions are ENVZ 1.5, RL40A 1.5, RNC 1.3; median 0.3). A variant that is truly neutral lies below WT − 5% of the window about as often as not, so the floor definition calls, on average, 63% of an enzyme assay's variants hurting (60% over the 17 assays), against 50% for the half-SD rule, and its curves rest on 12 enzymes (7 with a resolved floor). With a 10% threshold the random level is still 58%. The criterion would be sound only for an assay measured to better than 2.5% of its window, which is true of none of them.

## Sensitivity (K = 10, enzymes; `results/sensitivity.csv`)

| Definition | assays | ESM-MSR | MutateEverything | SPURS | ThermoMPNN | random |
|---|---|---|---|---|---|---|
| 0.5 assay SD (the rule) | 18 | 0.33 | 0.32 | 0.37 | 0.42 | 0.50 |
| variance, 5% (0.45 assay SD) | 18 | 0.34 | 0.35 | 0.39 | 0.43 | 0.51 |
| variance, 10% (0.63 assay SD) | 18 | 0.32 | 0.30 | 0.34 | 0.38 | 0.46 |
| ladder | 18 | 0.28 | 0.28 | 0.28 | 0.34 | 0.40 |
| floor, 5% | 12 | 0.44 | 0.47 | 0.54 | 0.57 | 0.63 |
| floor, 10% | 12 | 0.39 | 0.43 | 0.51 | 0.47 | 0.58 |
| floor, 5%, resolved floors only (window ≥ 3) | 7 | 0.57 | 0.54 | 0.56 | 0.63 | 0.71 |
| 0.5 assay SD, the 12 floor assays | 12 | 0.29 | 0.32 | 0.36 | 0.34 | 0.46 |
| variance 5%, the 12 floor assays | 12 | 0.30 | 0.34 | 0.38 | 0.35 | 0.48 |
| ladder, the 12 floor assays | 12 | 0.23 | 0.28 | 0.27 | 0.28 | 0.38 |
| variance 5%, the 7 resolved-floor assays | 7 | 0.41 | 0.40 | 0.44 | 0.43 | 0.57 |
| ladder, the 7 resolved-floor assays | 7 | 0.34 | 0.34 | 0.34 | 0.37 | 0.47 |

The definitions differ mostly in their baseline (the random level) and give the same picture: every model below random in every row, ESM-MSR or MutateEverything lowest, ThermoMPNN or SPURS highest. The earlier plateau-based definition (plateau WT and plateau sd) gave 0.21 / 0.22 / 0.29 / 0.32 for ESM-MSR / MutateEverything / SPURS / ThermoMPNN at K = 10 and 0.35 for random. The K = 1 and K = 100 values, the all-assay set and the variants with the plateau-proxy assays dropped are in the csv.

## Caveats

* **The half-SD rule.** A fixed fraction of the assay SD is one number for every assay: where the real noise is a small share of the variance (ENVZ, RNC, RL40A: under 1%) it is conservative, where it is larger (CASP3 48%, CAS9 37%) it flags real noise as loss. A variance dominated by a dead mode (HEM3, MET, OTC, CASP7) also inflates the SD; the plain SD is used, not a robust one.
* **Measurement scales.** The caspase scales are log-odds of passing a fluorescence sort with a per-variant se about as large as the WT-to-dead range; RNC saturates at about a quarter of WT activity (loss only); OTC and Q59976 are capped at WT. A "hurts" call is solid on these.
* **Inferred WT values.** The WT of PTEN, CASP3, CASP7, SRC and RNC (all 0) and OTC (1) is a property of how the paper defines the score, read from the paper and the authors' code, not a printed number; UBE4B's supporting statement "E = 1.06" was seen only as a search excerpt, the Dataset S1 row itself was read. Each is documented in `data/wt_manual.csv` with its confidence.
* **AICDA.** Its score is a within-position enrichment, so a global WT does not exist. Its hurts share (84%) is high because the 83% of variants that are dead sit far below the large WT of six of its eight usable positions; treat it as low confidence.
* **Seven assays use a plateau proxy for WT** (A0A1I9GEU1, A0A247D711, ADRB2, CCDB, PPARG, HRAS, SRC dasatinib). Dropping them raises the half-SD curves of all models by 0.01 to 0.02 at K = 10 (all assays) and does not change the order (`sensitivity.csv`).
* **ProteinGym cutoffs.** The binarization cutoff in `DMS_substitutions.csv` is the median of the scores for 24 of the 43 Activity assays and a manual value for 19; it is not the WT. For RPC1_LAMBD_Li_2019 it is listed on the raw (sign-flipped) scale (8.48 = minus the median of the curated scores) and is negated here (`cutoff_on_scale`).
* **Non-enzyme assays** differ in what activity means (binding, transport, channel current, fluorescence); the all-assay set mixes them, with the fluorescence readouts (7 sets) flagged in `assay_definitions.csv`.
* **Models.** None is trained for activity; what the curves measure is how often a stability-ranked pick destroys function, not whether it finds the active variants.
