# Evaluation and Epistasis Metrics

This document specifies the $\Delta\Delta G$ stability and $\Delta\Delta\Delta G$ epistasis metrics implemented in `src/esm_msr/epi_metrics.py`, `src/esm_msr/val_stats.py`, and `inference_scripts/esm_msr_testing.py`, and reported in [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md).

---

## 1. Dual-Pass Predictions & Observation Scale

### 1.1 Wild-Type, Mutant-Context, and Ensemble Readouts

Sign convention follows Tsuboyama et al. (2023): $\Delta\Delta G = \Delta G_{\text{mut}} - \Delta G_{\text{wt}}$, where $\Delta G$ is unfolding free energy ($\Delta\Delta G > 0$ denotes stabilization). For a variant carrying $n_{\text{mut}}$ substitutions, ESM-MSR evaluates two calibrated log-likelihood-ratio passes with per-mutation bias scaling (`per_mutation_bias = True`):

- **Wild-type pass ($\text{WT}$)**: Evaluates each substitution in the wild-type sequence context. Because bias scales with $n_{\text{mut}}$, the unmasked WT pass is strictly additive on double mutants:
  $$\text{WT}(AB) = \text{WT}(A) + \text{WT}(B)$$
- **Mutant-context pass ($\widetilde{\text{MT}}$)**: Evaluates each mutated position in the context of its partner mutation(s), estimating $\Delta\Delta G(A \mid B) + \Delta\Delta G(B \mid A) = 2\Delta\Delta G_{AB} - (\Delta\Delta G_A + \Delta\Delta G_B)$.
- **Combined prediction ($\text{comb}$)**: Averages the two thermodynamic integration paths ($A \to AB$ and $B \to AB$):
  $$\text{comb}(AB) = \tfrac{1}{2}\bigl(\text{WT}(AB) + \widetilde{\text{MT}}(AB)\bigr)$$

### 1.2 Epistasis Heads and Controls (`src/esm_msr/epi_metrics.py`)

Predicted epistasis on a double mutant $AB$ is evaluated across four readouts relative to the WT single-mutant baseline $\text{WT}(A) + \text{WT}(B)$:

| Head | Latent $\Delta\Delta G_{AB}$ | Predicted epistasis $\Delta\Delta\Delta G'$ | Role |
| :--- | :--- | :--- | :--- |
| `comb` | $\tfrac{1}{2}(\text{WT}(AB) + \widetilde{\text{MT}}(AB))$ | $\text{comb}(AB) - \text{WT}(A) - \text{WT}(B)$ | **Ensemble epistasis** ($1\times$ interaction scale of the deployed combined predictor). |
| `epi` | $\widetilde{\text{MT}}(AB)$ | $\widetilde{\text{MT}}(AB) - \text{WT}(AB)$ | **Direct interaction head**: non-additive increment predicted by the MT pass relative to the additive WT pass on the double alone (without querying noisy single mutants from the MT head). |
| `ctx` | $\tfrac{1}{2}(\text{WT}(AB) + \widetilde{\text{WT}}(AB))$ (ranks) / $\widetilde{\text{WT}}(AB)$ (residuals) | $\widetilde{\text{WT}}(AB) - \text{WT}(AB)$ | **Context control**: WT adapter evaluated on the mutated sequence (`--val_cycle_passes`), isolating epistasis already present in the pretrained backbone plus single-mutant fine-tuning. |
| `add` | $\text{WT}(AB)$ | $0$ (latent); $h(c + \text{WT}(AB)) - h(c + \text{WT}(A)) - h(c + \text{WT}(B)) + c$ (observed) | **Saturation control**: strictly additive on the latent scale; nonzero on observed-scale `naive_rho` solely through assay floor curvature. |

For external and retrained baseline models evaluated in [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) (§2), predicted epistasis enters the suite as `epi`: each model's direct double-mutant prediction minus its corresponding single-mutant additive baseline (e.g., `ME2(AB) - ME1(A) - ME1(B)` for MutateEverything stage 2, `TMD(AB) - TM(A) - TM(B)` for ThermoMPNN-D, and `SPURS_multi(AB) - SPURS(A) - SPURS(B)` for SPURS multi).

### 1.3 Tobit Expected Observation Reading (`val_stats.tobit_link`)

Models trained with the noisy-floor Tobit likelihood (`--reg_nll tobit`) predict latent stability $z = \Delta G_{\text{wt}} + \Delta\Delta G_{\text{lat}}$ under the observation process $Y = \max(X, F)$, where $X \sim \mathcal{N}(z, \sigma_y^2)$ is the variant's true stability measurement and $F \sim \mathcal{N}(\mu_f, \sigma_f^2)$ is the assay unfolding floor. For observed-scale reporting, `tobit_link` evaluates the closed-form expected reading $E[\max(X, F)]$ (Clark, 1961):

$$h(z) = z\,\Phi(a) + \mu_f\,\Phi(-a) + \theta\,\varphi(a), \qquad \theta = \sqrt{\sigma_y^2 + \sigma_f^2}, \quad a = \frac{z - \mu_f}{\theta}$$

giving observed-scale $\Delta\Delta G_{\text{obs}} = h(\Delta G_{\text{wt}} + \Delta\Delta G_{\text{lat}}) - \Delta G_{\text{wt}}$ with zero free parameters beyond the likelihood constants $(\sigma_y, \mu_f, \sigma_f)$. Because $h$ is strictly monotone, within-library rank metrics and saturation-subtracted epistasis levels are invariant to $h$; only scale-dependent RMSE metrics and observed-scale `naive_rho` depend on it.

---

## 2. Stability ($\Delta\Delta G$) Validation Metrics (`src/esm_msr/val_stats.py`)

On the Tsuboyama `splits_oct06_capped` validation split (23 scaffolds; 9 libraries containing designed double-mutant matrices, 24 position-pair matrices, and 8,494 uncensored doubles with both constituent singles measured), validation computes:

- **Per-library metrics (`val_<metric>/<library>`)**: `rho_combined` (Spearman $\rho$ across all measured variants), `rho_singles_wt` (WT head Spearman $\rho$ on single mutants), `rho_doubles_combined` (combined Spearman $\rho$ on double mutants), `rmse_combined`, and `rmse_singles_wt` (kcal/mol).
- **In-distribution equal-library averages (`val_<metric>_avg`)**: Mean across in-distribution validation libraries of `rho_combined`, `rho_singles_wt`, `rho_doubles_combined`, `rmse_combined`, top-$k$ normalized discounted cumulative gain (`ndcg@k=96`), and stabilizing-variant NDCG (`ndcg>0`, for both combined and `_wt` heads), plus scaffold-class means `val_{nat,des,mutbg}_rho_combined_avg` (natural domains, de novo designs, and mutant-background libraries).
- **Out-of-distribution diagnostic summary (`val_<metric>_external`)**: Optional diagnostic loaders (`--external_val_sets`, defaulting to `ptmuld,s461,ssym`) are excluded from all in-distribution averages and summarized separately as `val_{rho_combined,rho_wt,rmse_combined,rmse_singles_wt}_external`.

---

## 3. Epistasis ($\Delta\Delta\Delta G$) Hierarchy & Saturation Removal

### 3.1 Variance Hierarchy of Double-Mutant Non-Additivity

For a double mutant substituting residue $a$ at position $i$ and residue $b$ at position $j$ ($i < j$) in a library with wild-type stability $c = \Delta G_{\text{wt}}$, measured epistasis is $\Delta\Delta\Delta G = \Delta\Delta G_{AB} - \Delta\Delta G_A - \Delta\Delta G_B$ and the additive stability expectation is $x = c + \Delta\Delta G_A + \Delta\Delta G_B$. Double mutants within a position pair $(i, j)$ form a matrix over substitutions $(a, b)$, decomposing $\Delta\Delta\Delta G$ into four nested levels:

| Level | Definition | In-sample validation $\Delta\Delta\Delta G$ variance share |
| :--- | :--- | :--- |
| **Global** | $E[\Delta\Delta\Delta G \mid x]$: assay floor/ceiling truncation and nonspecific global curvature as a function of additive stability $x$ alone | 56% |
| **Pair** | Position-pair matrix mean residual beyond the global curve (~70% between libraries) | 22% |
| **Line** | Row or column mean residual beyond the pair mean (average coupling of one substitution with the partner site) | 14% |
| **Cell** | Double-centered residual within a position-pair matrix (specific residue–residue interaction) | 6% |

### 3.2 Separating Chemical Interaction from Assay Floor Saturation

When $x = c + \Delta\Delta G_A + \Delta\Delta G_B$ falls below the unfolding floor of the proteolysis assay ($\Delta G \approx 0.0\text{--}0.5$ kcal/mol), $\Delta G_{AB}$ is pinned near the floor and measured $\Delta\Delta\Delta G \approx \text{floor} - x$ becomes strongly positive regardless of specific residue contacts. Because $h(c + a + b)$ is non-additive on the observed scale, saturation leaks into raw pair, line, and cell statistics if unadjusted. `src/esm_msr/epi_metrics.py` removes assay saturation using two complementary devices:

1. **Beyond-global residuals (magnitude metrics: `beyond_rho`, `beyond_lib`, `beyond_rho_pairavg`, `pair_rho`, `line_rho`, `cell_mag`, `class3_*`)**:
   - **Measured residual**: $r = \Delta\Delta\Delta G - \hat{E}[\Delta\Delta\Delta G \mid x]$, subtracting a 25-quantile-bin piecewise-linear smoother (`SMOOTH_BINS = 25`) fitted over measured additive expectation $x = c + \Delta\Delta G_A + \Delta\Delta G_B$.
   - **Predicted residual**: $r'_h = \Delta\Delta\Delta G'_h - \hat{E}[\Delta\Delta\Delta G'_h \mid x'_h]$, conditioning on the head's own latent additive expectation $x'_h = c + \text{WT}(A) + \text{WT}(B)$. Any predictor that is additive on the latent scale satisfies $\Delta\Delta\Delta G' = 0$ and scores identically $0$ at every residual level.
   - Pair offsets evaluate matrix means of $r$ and $r'_h$ (`MIN_PAIR_CELLS = 20`); line effects evaluate row and column means (`MIN_LINE = 3`) centered on their matrix mean; `cell_mag` trims each matrix to a complete block (`MIN_BLOCK = 3`), double-centers $r$ and $r'_h$, and averages per-block Spearman $\rho$.

2. **Within-column ranks and order-reversal contrasts (order metrics: `cell_rank`, `cell_flipacc`)**:
   - Within a single column of a position-pair matrix (one library, so fixed $c$ and monotone observation map $h$; one fixed partner substitution), any monotone nonlinearity preserves the rank order of $\Delta\Delta G_{AB}$ exactly.
   - **`cell_rank`**: On each complete $\ge 3 \times 3$ block, rank-transforms every column of measured $\Delta\Delta G_{AB}$ and of predicted latent $\Delta\Delta G_{AB}$, double-centers both rank matrices, and averages Spearman $\rho$ across both row/column orientations and across blocks. Any predictor whose within-column ordering is independent of the partner residue scores exactly $0$.
   - **`cell_flipacc`**: Identifies $2 \times 2$ sub-blocks $(i, i') \times (j, j')$ whose measured rank order reverses confidently between columns $j$ and $j'$ by more than `FLIP_DELTA = 0.6` kcal/mol on both sides ($\Delta\Delta G_{ij} - \Delta\Delta G_{i'j} > 0.6$ and $\Delta\Delta G_{ij'} - \Delta\Delta G_{i'j'} < -0.6$, or vice versa). Because no monotone assay transformation can invert a strict inequality, a confident reversal fixes the sign of the true latent interaction contrast. `cell_flipacc` reports the fraction of confident reversals matched in sign by the predicted latent second difference $L_{ij} - L_{i'j} - L_{ij'} + L_{i'j'}$ (ties count $0.5$, so an additive predictor scores $0.5$).

### 3.3 Metric Catalogue

| Category | Metric (`epi_<name>_<head>`) | Definition | Direction |
| :--- | :--- | :--- | :--- |
| Holistic observed | `naive_rho_{comb,epi,ctx,add}` | Pooled Spearman $\rho$ of observed-scale predicted epistasis vs measured $\Delta\Delta\Delta G$ across all doubles (includes floor saturation; `add` measures saturation alone) | Higher is better |
| Calibration | `global_err_{comb,epi}` | RMSE (kcal/mol) of predicted epistasis vs measured $\Delta\Delta\Delta G$ across all doubles | Lower is better |
| | `global_bias_{comb,epi}` | Mean signed error ($\Delta\Delta\Delta G' - \Delta\Delta\Delta G$, kcal/mol) across all doubles | Closer to $0$ |
| | `err_xgt1_comb`, `bias_xgt1_comb` | RMSE and mean signed error of `comb` epistasis restricted to doubles with $x > 1.0$ kcal/mol (well above the assay floor) | Lower / closer to $0$ |
| Beyond global | `beyond_rho_{comb,epi,ctx}` | Pooled Spearman $\rho$ of saturation-subtracted residuals $r'_h$ vs $r$ across all doubles | Higher is better |
| | `beyond_lib_{comb,epi,ctx}` | Pooled Spearman $\rho$ of library-centered residuals (removing inter-library mean offsets) | Higher is better |
| | `beyond_rho_pairavg_{comb,epi,ctx}` | Mean across position pairs of within-pair residual Spearman $\rho$ | Higher is better |
| Pair | `pair_rho_{comb,epi,ctx}` | Spearman $\rho$ across position-pair matrices of mean pair residuals ($\ge 20$ doubles per pair) | Higher is better |
| Line | `line_rho_{comb,epi,ctx}` | Pooled Spearman $\rho$ across matrix rows and columns ($\ge 3$ cells) of pair-centered line-mean residuals | Higher is better |
| Cell | `cell_mag_{comb,epi,ctx}` | Mean across complete $\ge 3\times 3$ blocks of Spearman $\rho$ on double-centered residual magnitudes | Higher is better |
| | `cell_rank_{comb,epi,ctx}` | Mean across complete $\ge 3\times 3$ blocks of Spearman $\rho$ on double-centered within-column ranks (saturation-free) | Higher is better |
| | `cell_flipacc_{comb,epi,ctx}` | Sign accuracy on confident $2\times 2$ measured order reversals ($|\Delta| > 0.6$ kcal/mol on both columns; $0.5$ = chance) | Higher is better |
| Classification | `class3_acc_{comb,epi,ctx}` | 3-class accuracy on residuals: negative ($r < -0.6$), neutral ($[-0.6, +0.6]$), positive ($r > +0.6$ kcal/mol) | Higher is better |
| | `class3_{pos,neg}_recall_{comb,epi,ctx}` | Recall on confident positive ($r > +0.6$) and negative ($r < -0.6$ kcal/mol) epistatic residuals | Higher is better |
| Summary | `epi_composite` | Arithmetic mean of `epi_cell_rank_comb` and `epi_beyond_lib_comb` | Higher is better |
| Population counts | `epi_n_{doubles,pairs,blocks,flips}` | Evaluated double mutants, position pairs, complete blocks, and confident reversals (`8,494 / 24 / 24 / 2,509` on capped validation; `12,455 / 43 / 43 / 4,236` on capped test) | — |

### 3.4 Simulation Specificity Check

Evaluating synthetic predictors with known ground-truth components on the empirical validation matrix design under a saturating noisy assay (mean across 16 simulation replicates) confirms that each metric isolates its intended hierarchy level:

| Simulated predictor components | `naive_rho` | `beyond_rho` | `pair_rho` | `line_rho` | `cell_mag` | `cell_rank` | `cell_flipacc` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Saturation only (additive latent + Tobit expected reading) | 0.58 | -0.04 | -0.05 | -0.03 | 0.01 | 0.00 | 0.55 |
| Latent curvature imitation + noise (no pair/line/cell terms) | 0.54 | 0.00 | 0.07 | -0.01 | 0.00 | 0.00 | 0.51 |
| + Pair offsets only | 0.72 | 0.63 | 0.84 | 0.11 | 0.02 | 0.00 | 0.54 |
| + Line effects only | 0.61 | 0.25 | 0.08 | 0.50 | 0.01 | 0.01 | 0.54 |
| + Cell interactions only | 0.60 | 0.18 | 0.10 | 0.13 | 0.37 | 0.29 | 0.89 |
| All levels + Tobit expected reading | 0.75 | 0.70 | 0.83 | 0.54 | 0.36 | 0.29 | 0.89 |
| All levels, unsaturated linear assay | 0.42 | 0.71 | 0.84 | 0.53 | 0.36 | 0.26 | 0.90 |

---

## 4. Inference Output Columns & Offline Recomputation

- **Inference (`inference_scripts/esm_msr_testing.py`)**: Writes per-variant calibrated predictions in `wt_lora_pred`, `mt_lora_pred`, `combined_pred` ($\tfrac{1}{2}(\text{WT} + \widetilde{\text{MT}})$), and `epi_pred` / `mt_lora_dddg_pred` ($\widetilde{\text{MT}}(AB) - \text{WT}(AB)$), alongside `combined_dddg_pred` ($0.5 \times (\widetilde{\text{MT}}(AB) - \text{WT}(AB))$).
- **Offline validation dump scoring (`scripts/epi_from_dump.py`)**: Validation checkpoints write `val_dump_<tag>.npz` containing per-item latent predictions, reverse-leg context passes, wild-type stabilities $\Delta G_{\text{wt}}$, and `link_summary`. Any saved validation dump can be rescored with optional position-pair bootstrap confidence intervals via:
  ```bash
  PYTHONPATH=src python scripts/epi_from_dump.py <path/to/val_dump.npz> [--boot 200] [--csv out.csv]
  ```
