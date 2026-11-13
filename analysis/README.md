# `analysis/`: Benchmark Scoring, Figures, and Leak Audit

Benchmark scoring, figure generation, and homology leak auditing for the repository. All figures and CSV tables in `results/` are computed by the top-level numbered scripts (`00_*.py`–`15_*.py`, `99_*.py`) from `msr_predictions/` and `baselines/{retrained,public}/`.

Run all scripts (CPU only, using `msr_venv`):
```bash
./run_all.sh
```

---

## Subfolders

| Subfolder | Contents |
|---|---|
| `msr_analysis/` | Shared Python package: `registry.py` (datasets, model metadata, colors), `data.py` (spine & prediction loaders, `sum_singles`, generalized epistasis computation, frame cache), `boot.py` (exact multiplicity Spearman bootstrap: protein, site, mutation), `calib.py` (one-donor OLS recalibration for RMSE), `link.py` (softplus floor link & post-link epistasis from `devel_link_params.json`), `screening.py` (hit-rate, top-`K` recovery, best-`ΔΔG` and rectified-gain-@-`K` curves), `style.py` (matplotlib formatting), `tables.py` (CSV summary exporters), `common.py` (shared helpers). |
| `scoring/` | Prediction collection and benchmark table assembly (`collect_predictions.py`, `benchmark_matrix.py`, `score_baseline_runs.py`, `score_me_proteingym.py`, `pgym_public_scores.py`, `pgym_rosetta_singles.py`, `resource_report.py`). |
| `leak_audit/` | Sequence (MMseqs2) and structure (Foldseek TM-align) homology and training-data overlap audit (`run_all.py`, steps `01_*.py`–`19_*.py`; see `leak_audit/README.md`). |
| `reference/` | Committed reference inputs: `benchmark_tables/*.csv` (parsed from `docs/BENCHMARKS_OCT2026.md`), `all_by_all_long.csv`, `devel_link_params.json` (checkpoint link parameters), `ctx_wt_rev_devel.csv.gz` (mutant-context WT-adapter reverse pass `~WT`), `leak_*.csv`, and `spurs_public_checkpoint.csv`. |
| `results/` | Generated figures (`.png` + `.pdf`) and backing tables (`.csv`), organized into per-script subdirectories (`00_core/` through `12_pair_heatmaps/`). |
| `cache/` | Pickled `Frame` objects (`frame_<dataset>.pkl`) and bootstrap replicate arrays (`boot_<unit>.pkl`); git-ignored and rebuilt automatically. |
| `notebooks/` | Archived exploratory notebooks (`*.ipynb`, `analysis_utils_msr.py`); not read by the analysis pipeline. |
| `tests/` | Unit tests verifying `msr_analysis.boot` against `scipy.stats.spearmanr`. |

---

## Generalized Epistasis Definition

Earlier scripts wrote ESM-MSR's predicted epistasis as `MT_AB - WT_AB` (or `comb_AB - WT_AB`). That shorthand has two issues:

1. **Masking generality**: `WT_A + WT_B = WT_AB` holds **only** when the wild-type pass is unmasked so single-site log-odds sum exactly on the double row. Under independent single-position masking (where only position `A` or position `B` is masked at a time on the wild-type sequence), `WT_AB` is either undefined or not equal to `WT_A + WT_B`. Every additive baseline and epistasis calculation in `analysis/` therefore looks up the component single-mutant rows explicitly via `msr_analysis.data.sum_singles`.
2. **2× magnitude**: Because the deployed ESM-MSR predictor on a double is the 50/50 ensemble `comb_AB = 1/2(MT_AB + WT_AB)`, the raw head difference `MT_AB - WT_AB = 2(comb_AB - WT_AB)` has twice the magnitude of the model's actual deviation from WT-single additivity.

All scripts and cached frames (`msr_analysis/data.py`) compute and expose the consistent 1× generalization for every MSR/ESM3 model family (`msr`, `msrdev`, `msrp0`, `msrrel`, `msrrel@0`, `esm3avg`):

```text
dddG_pred = 1/2(MT_AB + WT_AB) - WT_A - WT_B = comb_AB - WT_A - WT_B
```

stored as `:epi`, with corresponding additive expectation `:wt+ = WT_A + WT_B`. (For the mutant-context control `ctx ctrl`, `MT_AB` is replaced by the reverse pass `~WT_AB`: `1/2(~WT_AB + WT_AB) - WT_A - WT_B`.)

## Screening curves: rectified gain @ K and ProteinGym Activity (`08_screening.py`)

`screening_per_protein*` (variants screened per protein, K = 1 to 100): hit rate, P(at least one hit in the top K) and, as the third panel, **rectified gain @ K** = the sum of max(y, 0) over the K selected variants (labels clipped at zero, predictions never), averaged over proteins; dotted = oracle (K largest max(y, 0)), dashed = K random picks (K x mean max(y, 0)). It replaces "best measured value in top K" (still computed as `best` in the CSVs). ESM-MSR in these figures is **FINAL_p0, seed 1** (`msr_analysis.registry.P0 = final_bench/predictions`; the WT head on singles, the combined head on multi-mutants; `msrp0` in `data.py`).

`screening_per_protein_pg_activity` (+ `pg_activity_hit_definition.csv`): the same three panels for the 43 ProteinGym Activity assays, single mutants only (FINAL_p0 seed 1 WT head from `final_pgym_wt`, ME stage 1, ThermoMPNN, SPURS single). The assays have no common unit, so the hit is defined per assay:
* **WT-like level w** = the mode of a Gaussian KDE of the single-mutant scores above the assay's ProteinGym binarization cutoff (above the median where under 5% or over 95% of the singles pass the cutoff): the fully functional plateau. **Spread sd** = 1.4826 x MAD of the scores above that cutoff around w.
* **Label** y = (score - w) / sd, so zero is "as active as the WT-like plateau" and the unit is the assay's own noise scale; the rectified gain is the summed excess over the plateau in those units.
* **Hit** = y > 2, a variant clearly above the plateau (a gain of function; 4% of the singles on average, 0.3 to 11%, against 3% for Tsuboyama's > 0.5 kcal/mol). At 1 SD the hit rate is ~10% and mostly noise around the plateau.
* An assay needs at least 10 hits to be screened: 29 of 43 are kept (the 14 dropped are listed with their numbers in `pg_activity_hit_definition.csv`: saturated or bimodal assays such as GFP, the three Somermeyer fluorescence sets, CCDB, LGK, NPC1, OTC, MET, VKOR1, AICDA, KCNH2, two ENTFA and RPC1 low-expression).
* Assays are the resampling unit (95% bootstrap); P(recover >= 1) is over assays with a hit (all 29).
`screening_per_protein_pg_activity_all_models` (no markers) adds the zero-shot models with ProteinGym scores for every kept single: ProSST-2048, ProteinMPNN and ESM3-open, taken from the authors' published scores (`baselines/public/proteingym_published`; the ESM3 column is the published open model, not our unmasked sigma-0 read). Rosetta is omitted by choice, and GeoDDG has no ProteinGym scores.
Reading: stability models are close to random on Activity gain of function (hit rate 5 to 7% against 4% random; gain @ 100 57 for ESM-MSR, 42 to 50 for the baselines, 35 for random picks, oracle 350); all intervals overlap.
