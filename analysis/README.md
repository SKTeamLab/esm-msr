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
| `msr_analysis/` | Shared Python package: `registry.py` (datasets, model metadata, colors), `data.py` (spine & prediction loaders, `sum_singles`, generalized epistasis computation, frame cache), `boot.py` (exact multiplicity Spearman bootstrap: protein, site, mutation), `calib.py` (one-donor OLS recalibration for RMSE), `link.py` (softplus floor link & post-link epistasis from `devel_link_params.json`), `screening.py` (hit-rate, top-`K` recovery, best-`ΔΔG` curves), `style.py` (matplotlib formatting), `tables.py` (CSV summary exporters), `common.py` (shared helpers). |
| `scoring/` | Prediction collection and benchmark table assembly (`collect_predictions.py`, `benchmark_matrix.py`, `score_baseline_runs.py`, `score_me_proteingym.py`, `pgym_public_scores.py`, `pgym_rosetta_singles.py`, `resource_report.py`). |
| `leak_audit/` | Sequence (MMseqs2) and structure (Foldseek TM-align) homology and training-data overlap audit (`run_all.py`, steps `00_*.py`–`19_*.py`; see `leak_audit/README.md`). |
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
