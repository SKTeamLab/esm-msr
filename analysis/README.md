# `analysis/`: Benchmark Scoring, Figures, and Leak Audit

Benchmark scoring, figure generation, and homology leak auditing for the repository. All figures and CSV tables in `results/` are computed by the top-level numbered scripts (`00_*.py`–`16_*.py`, `99_*.py`) from `msr_predictions/`, `baselines/{retrained,public}/` and, for the FINAL_* ablations (`16_*`), the testing queue's `final_bench/` and `final_pgym_*/`.

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
* **Hit** = y > 0: the variant is more active than the WT-like level (the ProteinGym files carry no wild-type row, so the level is the estimated plateau; 29% of the singles on average, the sets sit near 30%). The first version of this figure used y > 2 (a clear gain of function, 4% of singles); it is a one-line switch (`TAU_DS['pg_activity']`).
* An assay needs at least 10 hits to be screened (`PG_MIN_HITS`): 42 of 43 are kept; CCDB (no spread above its cutoff, 0 hits) is the one dropped (`pg_activity_hit_definition.csv` lists every assay with its numbers).
* Assays are the resampling unit (95% bootstrap); P(recover >= 1) is over assays with a hit (all 42).
`screening_per_protein_pg_activity_all_models` (no markers) adds the zero-shot models with ProteinGym scores for every kept single: ProSST-2048, ProteinMPNN and ESM3-open, taken from the authors' published scores (`baselines/public/proteingym_published`; the ESM3 column is the published open model, not our unmasked sigma-0 read). Rosetta is omitted by choice, and GeoDDG has no ProteinGym scores.
`screening_per_protein_pg_activity_gainloss{,_all_models}` replace the hit-rate and P(recover) panels by **rectified gain**, **net change @ K** (sum of y over the K picks, not rectified: the expected aggregate change in activity; symlog axis) and **rectified loss @ K** (sum of max(-y, 0): the activity the screen gives up; lower is better; the oracle loss is 0 and not drawn). All rectified-gain (and loss) panels, here and for Tsuboyama and Domainome, are on a log axis.
Reading (K = 100): every model beats random picks on all three, but modestly. Net change: random -248, ESM-MSR -116, ProSST -120, SPURS -145, ME -148, ThermoMPNN -170, ProteinMPNN -171, ESM3-open -73 (the least negative; an oracle gets +256). Rectified gain: 29.6 random, 37 to 56 for the models, 271 oracle. Screening 100 variants per assay still loses far more activity than it gains (most variants of a functional protein are neutral to deleterious); the models mostly avoid loss rather than find gain. Hit rate at K = 10 is 32 to 52% against 29% random; intervals overlap among models.

## FINAL_* ablation table (`16_final_ablation_table.py`)

Scores the nine FINAL_* families (FINAL_p0 reference; masking: p1, p1_marg; losses: pw0, norank, rcthr0; architecture: noseqhwm, r2a4; data: mega). Each family differs from FINAL_p0 in one hparam group, listed in `msr_analysis/final_runs.py::FAMILIES`. Every family x seed x readout (wt, comb, mt, wt+ = sum of WT singles, epi = comb − WT_A − WT_B) is scored on every benchmark. Each checkpoint is tested under its own masking strategy, as the testing queue ran it.

**Inputs** (no GPU): `<MSR_ROOT>/final_bench/predictions/<folder>/final_checkpoints/<FAMILY>/s<seed>/best.ckpt_sigma1.0_*predictions.csv` and `<MSR_ROOT>/final_pgym_wt|final_pgym_comb/<FAMILY>/s<seed>/<DMS_id>.csv`. If you downloaded the predictions elsewhere, point `MSR_FINAL_BENCH`, `MSR_FINAL_PGYM_WT`, `MSR_FINAL_PGYM_COMB` and `MSR_FINAL_PGYM_XMASK` at those folders. Rows `FINAL_p0@independent` and `FINAL_p1@unmasked` are cross-masking runs, each checkpoint evaluated under the other family's masking. They fill in once the queue's lowest-priority `gamut_xmask` / `pgym_wt_xmask` tiers have run. Labels come from `baselines/public/_tasks` and ProteinGym categories from `baselines/pgym_per_dms/pgym_all_models_per_dms.csv`. A family or seed whose files are missing is left out, and `coverage.csv` lists what was found. A ProteinGym run counts only if it covers at least 212 of the 213 assays.

**Scoring** follows `01_core_rho` / `05_proteingym` / `06_epistasis`: measured rows; Tsuboyama = mean per-library ρ, Domainome = mean per-domain ρ; low-id drops the P/H MegaScale-homologous domains and assays; ΔΔΔG = pooled ρ over doubles with both singles measured, wild-type backbone. `primary` is the deployed readout: wt on singles-only columns and comb where multi-mutants are scored. FINAL_p0_mega trains on `mega_splits`, so its Tsuboyama cells are on its own split (marked †, `cells_long_mega_own_split.csv`).

**Outputs** in `results/16_ablation/`: `cells_long.csv`, `summary.csv` (seed mean / SD), `delta_s1.csv` (paired bootstrap of family − FINAL_p0 at seed 1, the only FINAL_p0 seed staged on 2026-11-13; protein level, site level for the DMS, assays within category for ProteinGym), `table_main_delta.{md,csv,tex}` (the main table: Δρ vs FINAL_p0 under the deployed readout on Tsuboyama test and doubles, the epistasis `beyond_rho` and `cell_mag` of `esm_msr.epi_metrics` on the 12,455 test doubles (comb(AB) − WT(A) − WT(B), paired bootstrap over the 13 test domains), external singles, PTMUL/PTMUL-D, Domainome low-id and ProteinGym avg; the single-protein DMS sets and FINAL_p0_mega, which belongs with the retrained-methods comparison, are left out of it; bold when |Δ| > 2 SE, with seed and sampling noise combined), `informativeness.csv` (family spread vs seed noise per column), `head_choice.csv` (comb − wt and whether the two readouts rank the families alike) and `layout_*.md` (candidate tables). `--render-only` redraws the layouts from the CSVs. `--B` sets the number of bootstrap draws (default 200 while drafting).
