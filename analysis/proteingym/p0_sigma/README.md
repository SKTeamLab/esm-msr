# p0 sigma on ProteinGym: per property and per protein

How the WT adapter of FINAL_p0 performs at LoRA scale sigma = 1 and 0.5 (FINAL_p0 seed 1, WT head) and at sigma = 0 (ESM3-small-open, unmasked: the released model with the adapter scaled to 0), measured
as per-assay Spearman rho over all variants of the assay (ProteinGym's convention; the additive WT-head prediction for multi-mutants). All 217 ProteinGym assays have predictions at all three sigmas. Four sigma reads come from a fallback file (`pgsigma.FALLBACK`; the `fallback` column of `results/per_assay_rho.csv` lists them): BRCA2 Erwood HEK293T at sigma 1 from the combined-head run of the same checkpoint (`final_pgym_comb/FINAL_p0/s1`, whose `wt_lora_pred` equals the WT-only run's exactly), and BRCA2, POLG_CXB3N and SCN5A at sigma 0 from an older ESM-MSR-small sigma-0 run (`pgym_results/unmasked-esm-msr-small/sigma0.0`), because the released ESM3-small-open sigma-0 read is missing for these three. That older read tracks the released one closely (over the 214 assays that have both, per-assay rho differs by 0.002 on average and 0.02 at most) but is not identical; rerunning the released sigma-0 read on the three assays would remove it (not run: needs the GPU).

```bash
python analysis/proteingym/p0_sigma/01_per_assay_rho.py          # results/per_assay_rho.csv: the 217 assays (flags train_overlap marks the 23 left out of every figure) + the three DMS-benchmark sets that are not in ProteinGym (DLG4 binding, GRB2 binding, MYO)
python analysis/proteingym/p0_sigma/02_fig_multi_property.py     # results/multi_property_index.csv and multi_property_rho_vs_sigma
python analysis/proteingym/p0_sigma/03_fig_property_summary.py   # results/property_delta_trajectories and property_delta_paired_boxes
```

| figure | content |
|---|---|
| `multi_property_rho_vs_sigma` | the earlier F12 (rho vs sigma, one panel per protein, one line per assay coloured by property, band = 95% bootstrap over variants) re-rendered with the p0 data at sigma 0, 0.5, 1. Proteins with assays of different properties: the 13 proteins of the earlier annotation (group A of `../sigma_tradeoff/scripts/common.py`) plus DLG4 and GRB2, whose ProteinGym assay is the abundance-type Faure set (its scores equal those of the abundance sets of the earlier DMS benchmark, to 4e-8) and whose binding sets come from the DMS benchmark: 15 proteins, 32 assays |
| `property_delta_trajectories` | one panel per property (assay category), x = sigma, y = delta rho versus sigma 0, one thin line per assay (assays independent), bold mean with its 95% bootstrap interval over assays |
| `property_delta_paired_boxes` | one box per property and sigma, in the style of the property-scaffold plots: IQR box, median, jittered dots, mean line with its value; sigma 0 (delta = 0 by definition, the anchor), 0.5 and 1 side by side; faint thin lines join the same assay |

**p-values (`results/property_delta_pvalues.csv`, printed under the σ ticks of both summary figures).** Is the mean Δρ versus σ = 0 different from 0? Two-sided bootstrap over assays (20000 resamples; p = 2 min(P(mean ≤ 0), P(mean ≥ 0)), the repo's convention in `msr_analysis/boot.py`), Holm-adjusted over the ten category × σ comparisons; the Wilcoxon signed-rank test over assays is the cross-check (it tests the typical assay rather than the mean). Assays are treated as independent, so several assays of one protein count separately.

| Category | assays | σ | mean Δρ | share improved | bootstrap p (Holm) | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|
| Stability | 43 | 0.5 | +0.142 | 100% | < 0.001 | < 0.001 |
| Stability | 43 | 1 | +0.186 | 100% | < 0.001 | < 0.001 |
| Expression | 18 | 0.5 | +0.049 | 94% | < 0.001 | 0.008 |
| Expression | 18 | 1 | +0.079 | 94% | 0.002 | 0.008 |
| Activity | 43 | 0.5 | +0.035 | 84% | < 0.001 | < 0.001 |
| Activity | 43 | 1 | +0.021 | 67% | 0.24 | 0.093 |
| Binding | 13 | 0.5 | +0.014 | 77% | 0.57 | 0.093 |
| Binding | 13 | 1 | -0.015 | 69% | 0.71 | 0.74 |
| OrganismalFitness | 77 | 0.5 | +0.033 | 92% | < 0.001 | < 0.001 |
| OrganismalFitness | 77 | 1 | +0.020 | 60% | 0.19 | 0.022 |

Stability, Expression (both σ) and the σ = 0.5 gains of Activity and Organismal fitness are significant under both tests. At σ = 1 Activity and Organismal fitness are borderline by the mean and significant or nearly so by the rank test (a typical assay gains, a few strongly harmed assays pull the mean down); Binding is not significant at either σ.

**Training overlap.** Every figure leaves out the 23 ProteinGym stability assays that overlap the ESM-MSR training data (exact or constituent overlap with the capped MegaScale inventory, `sameprot` in `../../reference/leak_audit_pg_per_dms.csv`, from `leak_audit/17_proteingym.py`): `pgsigma.train_overlap()` / `pgsigma.per_assay()`, flag `train_overlap` in `results/per_assay_rho.csv`. That leaves 194 assays (Stability 43, Expression 18, Activity 43, Binding 13, Organismal fitness 77); none of the 23 is in the multi-property index. The 4 assays that carry only a homology flag (H: AMFR 4G3O, RAD_ANTMA 2CJJ, VG08 2GP8; h: NUSA 1WCL) are kept.

Mean delta rho versus sigma 0 at sigma 0.5 / 1 (assays): Stability +0.143 / +0.186 (43), Expression +0.049 / +0.079 (18), Activity +0.035 / +0.021 (43), Binding +0.014 / −0.015 (13), Organismal fitness
+0.033 / +0.020 (77).

Drafts. One seed; the sigma = 0 reference is the released ESM3-small-open read, not a p0 checkpoint; the 0.25 and 0.75 scales of the earlier analysis are not computed for p0.
