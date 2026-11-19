# Epistasis in the DLG4 (PSD95-PDZ3) and GRB2 (GRB2-SH3) abundance and binding datasets

The four deep mutational scans of Faure et al. (Nature 2022; AbundancePCA and BindingPCA of PSD95-PDZ3 and GRB2-SH3 in the ddPCA libraries of single and double substitutions): how close are the two mutated residues
of the doubles, how much of the measured epistasis is signal, and how well can the cell-level epistasis statistic `cell_mag` be predicted at best.

Data: the library's `fitness` (the label `truth` of `baselines/public/_tasks/<set>.csv.gz`) and its measurement error `sigma` (DiMSum's count-based model on the merged triplicate). The abundance sets are no longer in the repository
(the strip of the duplicates of ProteinGym assays); their task tables are read from `<MSR_ROOT>/baselines/public/_tasks`. Epistasis of a double, as in the benchmark (`dddG_ML`): eps = f_AB - f_A - f_B on the fitness scale, for
the doubles whose two singles are measured; its error sigma_eps^2 = sigma_AB^2 + sigma_A^2 + sigma_B^2. Structures: the AlphaFold2 models of the ProteinGym assays (`ProteinGym/ProteinGym_AF2_structures`; the domain's residue p is
residue 310 + p of DLG4 and 158 + p of GRB2, found by aligning the sequence). All scripts run on the CPU in a minute or two.

```bash
python analysis/faure_epistasis/01_proximity.py     # proximity_distribution, proximity_summary.csv, pair_table.csv (+ the doubles tables the other scripts read)
python analysis/faure_epistasis/02_snr.py           # epistasis_snr, snr_summary.csv, snr_by_distance.csv
python analysis/faure_epistasis/03_cell_ceiling.py  # cell_mag_ceiling.csv, cell_mag_blocks.csv: blocks, noise ceilings, ESM-MSR (FINAL_p0)
python analysis/faure_epistasis/04_fig_ceiling.py   # cell_mag_ceiling
python analysis/faure_epistasis/05_cell_mag_significance.py   # cell_mag_significance.csv: bootstrap and permutation tests of ESM-MSR's cell_mag (about 3 min)
```

## What the library is (Faure et al.)

* **PDZ3 (DLG4)**: a shallow double-mutant library made by one round of NNK nicking mutagenesis on "an equimolar pool of nine single mutants and the wild-type" template, so a double is a background mutation plus one new substitution
  (median 4-5 doubles per single). In the data 99.8% of the doubles contain one of eight positions (1, 5, 10, 13, 14, 40, 49, 53), and for the 92-93% that contain exactly one of them, that position carries one fixed residue
  (G1, R5, I10, R13, V14, A40, L49, R53; the abundance library also has F5 and G5). The partners cover the whole domain.
* **SH3 (GRB2)**: error-prone PCR (binding) and two rounds of nicking mutagenesis (abundance) give singles and doubles over all positions; 1,540 of the 1,540 position pairs (binding) and 1,407 of 1,485 (abundance) have doubles.
* Biological triplicates; fitness errors from "a modified Poissonian (count-based) model" with replicate-specific terms, merged by weighted averaging; replicates correlate with Pearson R = 0.87-0.92; the three-state thermodynamic
  model (additive free-energy changes of folding and binding) fits the fitness with R^2 = 0.84-0.91 (bioRxiv version 10.1101/2021.09.14.460249, read through a summarising tool; the supplement with the per-library numbers was not read).

## 1. How close are the two residues of a double? (`proximity_distribution`)

Minimal heavy-atom distance between the two residues in the AlphaFold2 model; a contact is < 5 A. Doubles with both singles measured.

| dataset | doubles | position pairs | median distance (A) | contact < 5 A | < 8 A | < 12 A | > 16 A | median sequence separation |
|---|---|---|---|---|---|---|---|---|
| PDZ3 (DLG4) abundance | 5,010 | 593 | 12.8 | 8.2% | 20.5% | 44.7% | 30.4% | 34 |
| PDZ3 (DLG4) binding | 6,459 | 604 | 12.8 | 7.7% | 20.5% | 44.6% | 29.9% | 33 |
| SH3 (GRB2) abundance | 62,189 | 1,407 | 10.5 | 11.8% | 32.1% | 64.1% | 7.7% | 24 |
| SH3 (GRB2) binding | 32,258 | 1,540 | 10.3 | 16.4% | 35.2% | 62.5% | 12.6% | 17 |
| *baseline: all pairs of PDZ3 positions* | - | 3,486 | 12.3 | 12.1% | 24.9% | 48.0% | 27.9% | 25 |
| *baseline: PDZ3 pairs with a background position* | - | 636 | 12.1 | 12.3% | 25.6% | 49.5% | 27.4% | 27 |
| *baseline: all pairs of SH3 positions (binding)* | - | 1,540 | 10.2 | 17.4% | 36.2% | 63.4% | 12.3% | 17 |

The doubles are close to a random draw of position pairs: SH3 binding covers every pair once and has 16% contacts against 17% for all pairs; PDZ3 has fewer contacts than its pairs (8% against 12%). Only 8-16% of the doubles are in contact and 21-35% within 8 A; 30% of the PDZ3 doubles and 8-13% of the SH3 doubles are more than 16 A apart. The sequence separation is large
(median 33-34 residues in PDZ3, 17-24 in SH3). Doubles per position pair: PDZ3 8 and 11 (median; at most 26), SH3 binding 21 (at most 41), SH3 abundance 22 (1 to 333).

## 2. Signal and noise of the epistasis (`epistasis_snr`)

Two readings of the same doubles: the raw eps (the benchmark's `naive_rho` target) and the beyond-global residual r = eps - E[eps | x], x = f_A + f_B (the residual the pair, line and cell levels are computed on).
SNR = (variance - noise variance) / noise variance; the noise variance is the mean sigma_eps^2 of the reported errors, or a constant calibrated so that two replicates would correlate at R = 0.87 or 0.92 (the reported errors are
the more pessimistic: the mean reported variance of a merged variant implies a replicate correlation of 0.59-0.73, below the reported 0.87-0.92). "Above floor": additive expectation x > -0.5 (both substitutions tolerated;
32-54% of the doubles). Near the floor the reported errors exceed the actual scatter (a dead variant cannot move below the floor), so the signal of r vanishes in an all-doubles estimate.

| dataset | doubles | SD of eps | noise SD | eps variance a function of x | SNR of eps (reported error; R = 0.87 / 0.92) | above floor: doubles | SNR of the residual r (reported error; R = 0.87 / 0.92) | ceiling for r (Pearson; reported / R 0.87 / 0.92) | doubles with abs z > 3 |
|---|---|---|---|---|---|---|---|---|---|
| PDZ3 (DLG4) abundance | 5,010 | 0.40 | 0.20 | 78% | 3.1 (6.4 / 11.7) | 2,355 | 1.0 (0.5 / 1.6) | 0.70 / 0.57 / 0.78 | 6.9% (all doubles 5.6%) |
| PDZ3 (DLG4) binding | 6,459 | 0.40 | 0.19 | 84% | 3.4 (5.8 / 10.7) | 2,628 | 0.9 (-0.2 / 0.4) | 0.68 / 0.00 / 0.51 | 5.5% (all doubles 3.2%) |
| SH3 (GRB2) abundance | 62,189 | 0.34 | 0.16 | 73% | 3.7 (3.0 / 5.9) | 19,994 | 2.1 (0.4 / 1.4) | 0.82 / 0.54 / 0.77 | 16.0% (all doubles 8.9%) |
| SH3 (GRB2) binding | 32,258 | 0.21 | 0.14 | 56% | 1.1 (2.3 / 4.7) | 17,366 | -0.05 (0.04 / 0.8) | 0.00 / 0.20 / 0.66 | 6.7% (all doubles 4.8%) |

* **Most of the raw epistasis is the assay's saturation.** 56-84% of the variance of eps is a function of x alone; what is left is mostly noise (reported-error SNR of eps 1.1-3.7 over all doubles, 2-12 with replicate-calibrated noise).
* **Specific epistasis beyond the global curve has an SNR of about 1 or less.** Above the floor the residual is signal-dominated in SH3 abundance (SNR 2.1 with the reported errors, 0.4-1.4 with replicate-calibrated noise; signal SD 0.15-0.18 fitness units against a
  noise SD of 0.11), borderline in PDZ3 abundance and binding (about 1 with reported errors, -0.2 to 1.6 with calibrated noise) and absent on average in SH3 binding (the reported noise equals the residual variance; 0.04-0.8 with calibrated noise).
* **It is heavy-tailed, not Gaussian.** 5.5-16% of the doubles above the floor (3-9% of all doubles) have |r| > 3 sigma_eps (0.27% for noise alone), so a minority of doubles carries clear epistasis, and the typical double does not.
* **Proximity helps only a little.** The share of doubles with |z| > 3 and the signal SD are higher at contact than far away in PDZ3 abundance (10% against 5-7%) and SH3 binding (11% against 5%; signal SD 0.063 against 0 above the floor), but not in PDZ3 binding or SH3 abundance (17% in contact and beyond 12 A).
  Epistasis here is not confined to touching residues.

## 3. How well can `cell_mag` be predicted? (`cell_mag_ceiling`)

`cell_mag` (`esm_msr.epi_metrics`): the doubles of one position pair form a matrix; it is trimmed to a complete block (>= 3 x 3, pairs with at least 20 doubles), the beyond-global residuals are double-centred, and the Spearman rho of
predicted against measured double-centred residuals is averaged over blocks. Double-centring removes everything additive in the two residues, so the noise of the singles drops out and only the double's own error is left.

* **PDZ3 has no complete block**, so `cell_mag` is undefined on the DLG4 datasets: only 3 (abundance) and 11 (binding) position pairs have 20 doubles, all of them between two background sites, and their 20-26 cells are spread over 6-17 residues per side.
  Position-level statistics (`naive_rho`, `beyond_rho`) are the DLG4 epistasis metrics.
* **SH3 has 415 (abundance) and 895 (binding) blocks**, 3 x 3 to 6 x 6, 15,940 and 15,516 cells.
* **Noise ceiling** = the expected `cell_mag` of a perfect predictor of the true cell effects, with the true double-centred effects taken as the observed ones shrunk to the signal variance, measured = true + noise of the double
  (reported error, or the replicate-calibrated constant), simulated 30 times.

| dataset | blocks (cells) | stratum | ceiling, reported error | ceiling, replicate R = 0.87 | ceiling, R = 0.92 | ESM-MSR cell_mag |
|---|---|---|---|---|---|---|
| SH3 (GRB2) abundance | 415 (15,940) | all blocks | 0.33 | 0.51 | 0.73 | 0.034 |
| SH3 (GRB2) abundance | 129 (5,897) | blocks above floor | 0.63 | 0.43 | 0.70 | 0.073 |
| SH3 (GRB2) binding | 895 (15,516) | all blocks | 0.25 | 0.63 | 0.76 | 0.036 |
| SH3 (GRB2) binding | 558 (9,826) | blocks above floor | 0.46 | 0.51 | 0.70 | 0.061 |

**A realistic upper bound is about 0.5, within 0.3-0.7**: the three noise models give 0.33 / 0.51 / 0.73 (SH3 abundance) and 0.25 / 0.63 / 0.76 (SH3 binding) over all blocks and 0.43-0.70 / 0.46-0.70 over the blocks above the floor,
and the reported errors alone (the pessimistic model) give 0.25-0.33 over all blocks. ESM-MSR (FINAL_p0, comb(AB) - WT(A) - WT(B), seeds 1-3) scores 0.034 (all blocks; 0.073 above the floor) on SH3 abundance and
0.036 (0.061) on SH3 binding, about a tenth of the ceiling; its pooled beyond-global rho is -0.000 and 0.044 on SH3 and 0.034 and
0.063 on PDZ3, and its naive rho 0.34 / 0.24 (SH3) and 0.59 / 0.62 (PDZ3) against measured-eps ceilings of
0.73-0.89. The ceiling is the best any predictor could do; what a sequence and structure model can reach is lower by an amount these data cannot give (the benchmark's `cell_mag` on the Tsuboyama test set is 0.24-0.32 for the best clean models, `docs/BENCHMARKS_OCT2026.md`).

## 3b. Is ESM-MSR's `cell_mag` different from 0? (`05_cell_mag_significance.py`, `results/cell_mag_significance.csv`)

Per block, the Spearman rho of ESM-MSR (FINAL_p0, mean of seeds 1-3) with the measured double-centred residuals; `cell_mag` is their mean. Block bootstrap: blocks resampled (B = 1000 by default; the files were made at 100, so the smallest p is 0.01). Position bootstrap: the positions
are resampled and a block (i, j) weighted count(i) x count(j), because blocks that share a position are not independent. Permutation: the predicted residuals are permuted among the cells of each block, 200 times (smallest p 0.005).

| dataset | blocks | ESM-MSR cell_mag (seeds 1 / 2 / 3) | block bootstrap 95% interval, p | position bootstrap 95% interval, p | permutation null (mean, SD), p | blocks with rho > 0 |
|---|---|---|---|---|---|---|
| SH3 (GRB2) abundance, all blocks | 415 | 0.034 (0.045 / 0.032 / 0.026) | [+0.008, +0.054], p = 0.01 | [-0.006, +0.078], p = 0.08 | (+0.000, 0.014), p = 0.005 | 56% |
| SH3 (GRB2) abundance, blocks above floor | 129 | 0.073 (0.080 / 0.073 / 0.067) | [+0.030, +0.115], p = 0.01 | [-0.001, +0.154], p = 0.06 | (-0.001, 0.023), p = 0.005 | 68% |
| SH3 (GRB2) binding, all blocks | 895 | 0.036 (0.040 / 0.038 / 0.031) | [+0.018, +0.058], p = 0.01 | [-0.015, +0.081], p = 0.12 | (+0.000, 0.009), p = 0.005 | 54% |
| SH3 (GRB2) binding, blocks above floor | 558 | 0.061 (0.063 / 0.073 / 0.049) | [+0.037, +0.084], p = 0.01 | [+0.010, +0.112], p = 0.02 | (+0.000, 0.012), p = 0.005 | 59% |

All four values are above 0 against the permutation null and the block bootstrap (null draws centre on 0 with an SD of 0.01-0.02, and no null draw reaches the observed values). The more conservative position bootstrap leaves two of the four
unresolved: **the 0.073 of SH3 abundance above the floor has a position-bootstrap interval of [-0.001, +0.154] (p = 0.06)**, and the all-blocks values of SH3 abundance (0.034) and binding (0.036) have intervals that include 0 (p = 0.08, 0.12);
only 0.061 (SH3 binding, above floor) clears 0 under both bootstraps. The three seeds agree (0.067-0.080 for the 0.073). The effect is small: 55-68% of the blocks have rho > 0.

## Caveats

* The noise models bracket the answer: DiMSum's errors are pessimistic for the near-floor doubles (and imply replicate correlations below the reported 0.87-0.92); constant replicate-calibrated errors ignore that low-count variants are noisier.
  The per-library replicate correlations and the number of reads per variant (the Faure supplement) were not used.
* "Specific epistasis" is everything non-additive that is not a function of x = f_A + f_B: it includes the interplay of the two latent traits (folding and binding) that the three-state model predicts from the singles' energies, not only
  residue-specific interactions.
* Ceilings treat the shape of the true cell-effect distribution as that of the observed one (linear shrinkage) and the errors of different doubles as independent; the doubles of PDZ3 share one background single, whose error is common to them.
* Structures are AlphaFold2 models (domain pLDDT not inspected here), not the crystal structures with the ligand; distances are between the two mutated residues, not to the ligand.
