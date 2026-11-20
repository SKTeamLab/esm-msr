# Faure et al. (2022) & MegaScale Epistasis Geometry, SNR, and Noise-Ceiling Study

Self-contained mechanistic analysis of why higher-order epistasis metrics (`beyond_rho` vs double-centered `cell_mag`) behave differently across the four Faure et al. (2022) AbundancePCA / BindingPCA datasets (`DLG4-PDZ3`, `GRB2-SH3`) compared to Homology-Restricted MegaScale (`tsuboyama_capped_test`).

## Key Findings
1. **Combinatorial Block Geometry (`proximity_distribution`, `cell_mag_blocks.csv`)**:
   - `DLG4` (PDZ3) screens random NNK substitutions on **9 single-mutant backgrounds**, so every position pair has only 1 residue on one side and **0 complete `>= 3x3` blocks** (`cell_mag` is structurally undefined, while `beyond_rho` remains well-powered).
   - `GRB2` (SH3) error-prone PCR yields hundreds of `>= 3x3` blocks across random residue pairs (predominantly distant `> 12 Å` pairs with weak specific epistasis).
2. **Variance Decomposition & Noise Ceiling (`epistasis_snr`, `faure_vs_megascale_epistasis_snr`)**:
   - Raw `dddG` (`naive_rho`) on PCA assays is dominated (`50–70%` of variance) by global folding/assay saturation `E[eps | f_A + f_B]`.
   - After subtracting global saturation and double-centering within `3x3+` blocks, `GRB2` specific cell-level signal is small relative to DiMSum count noise (nominal Spearman ceiling `0.25–0.55`; `FINAL_p0` `cell_mag = 0.034–0.073`, statistically positive on binding under joint `[block/position × 3-seed]` bootstrap and cell-permutation tests).
   - By contrast, Homology-Restricted MegaScale test doubles target structural contacts at high cDNA proteolysis precision (Spearman ceiling `0.77–0.94`), enabling `FINAL_p0` to achieve **`cell_mag ≈ 0.30`**.

## Outputs (`analysis/results/99_misc/faure_epistasis/`)
- `proximity_summary.csv`, `pair_table.csv`, `proximity_distribution.{png,pdf}`
- `snr_summary.csv`, `snr_by_distance.csv`, `epistasis_snr.{png,pdf}`
- `cell_mag_ceiling.csv`, `cell_mag_blocks.csv`, `cell_mag_significance.csv`, `megascale_vs_faure_ceiling.csv`
- `faure_vs_megascale_epistasis_snr.{png,pdf}` & `cell_mag_ceiling.{png,pdf}`

## Usage
```bash
PYTHONPATH=src:analysis python -m studies.faure_epistasis.run --B 1000
```
