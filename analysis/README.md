# `analysis/`: Benchmark Scoring, Figures, and Leak Audit

Benchmark scoring, figure generation, and homology leak auditing for the repository. All figures and CSV tables in `results/` are computed by the top-level numbered scripts (`00_*.py`–`17_*.py`, `98_*.py`, `99_*.py`; the DMS benchmark is DLG4 binding, GRB2 binding and MYO: the abundance, ESTA and GB1 sets of the earlier benchmark duplicate ProteinGym assays and were removed) from `msr_predictions/`, `baselines/{retrained,public}/` and, for the FINAL_* ablations (`98_final_ablation_table.py`), the testing queue's `final_bench/` and `final_pgym_*/`.

Run all scripts (CPU only, using `msr_venv`):
```bash
./run_all.sh
```

---

## Subfolders

| Subfolder | Contents |
|---|---|
| `msr_analysis/` | Shared Python package: `registry.py` (datasets, model metadata, colors), `data.py` (spine & prediction loaders, `sum_singles`, generalized epistasis computation, frame cache), `boot.py` (exact multiplicity Spearman bootstrap: protein, site, mutation), `calib.py` (one-donor OLS recalibration for RMSE), `link.py` (softplus floor link & post-link epistasis from `devel_link_params.json`), `screening.py` (hit-rate, top-`K` recovery, best-`ΔΔG` and rectified-gain-@-`K` curves), `style.py` (matplotlib formatting), `tables.py` (CSV summary exporters), `common.py` (shared helpers). |
| `scoring/` | Prediction collection and benchmark table assembly (`collect_predictions.py`, `benchmark_matrix.py`, `score_baseline_runs.py`, `score_me_proteingym.py`, `pgym_public_scores.py`, `pgym_rosetta_singles.py`, `resource_report.py`, `resource_summary.py`: inference cost per model and benchmark group from the fixed-slot resource-tracker runs that `collect_predictions.py run --groups resources` mirrors into `<MSR_ROOT>/resources_tracked/`; writes `results/inference_resources/`). |
| `proteingym/` | ProteinGym sigma analyses: `sigma_tradeoff/` (the Oct 27 baseline-model analysis: report, scripts, tables, figures) and `p0_sigma/` (FINAL_p0 at sigma 1 / 0.5 and the sigma = 0 reference: per-protein rho vs sigma and per-property delta-rho summaries). |
| `pg_activity/` | ProteinGym Activity assays: measured wild-type scores (`01_wt_reference.py`, `data/`), screening curves (hurts = more than half an assay SD below WT; the variance, floor and ladder rules are in the sensitivity tables), a floor audit and landscape figures (`02_activity_screening.py`, `results/`), and `README.md` with the definitions, per-enzyme evidence and results. |
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

## Screening curves: rectified gain @ K (`08_screening.py`)

`screening_per_protein*` (variants screened per protein, K = 1 to 100): hit rate, P(at least one hit in the top K) and, as the third panel, **rectified gain @ K** = the sum of max(y, 0) over the K selected variants (labels clipped at zero, predictions never), averaged over proteins; dotted = oracle (K largest max(y, 0)), dashed = K random picks (K x mean max(y, 0)). It replaces "best measured value in top K" (still computed as `best` in the CSVs). In every per-protein screening curve figure (`16_screening_overview.py`, `15_tsuboyama_single_screening.py` and `pg_activity/`) the shaded 95% bootstrap band is drawn for ESM-MSR only, so that it is not hidden under the other models' bands; the csv files hold every model's interval. ESM-MSR in these figures is **FINAL_p0, seed 1** (`msr_analysis.registry.P0 = final_bench/predictions`; the WT head on singles, the combined head on multi-mutants; `msrp0` in `data.py`).

**Overview figures (`16_screening_overview.py`, `results/16_screening_overview/`).** `08_screening.py` writes the per-protein curve tables (`results/08_screening/screening_per_protein*.csv`: default, `_singles` and `_composite` Tsuboyama / Domainome tables, four model families or `_all_models`); `16_screening_overview.py` draws the singles and composite ones. `screening_overview_{single,multi}*` is a 2 x 3 grid with one column per category and two vertically stacked panels each: Tsuboyama test set (hit rate; P(at least one hit in the top K)), Domainome (the same two), and ProteinGym Activity (single mutants; the hurt rate under the 5%-of-variance noise rule, top for all 43 assays, bottom for the 18 enzymes; lower is better). `single` = single mutants only, scored by the single-mutant models; `multi` = Tsuboyama singles + doubles, the singles scored by the single-mutant model and the doubles by the multi-mutant model of each family (Domainome is single-mutant only and identical in both; the Activity column is always single mutants). `screening_gain_per_variant_{single,multi}*` shows the rectified gain per variant (the rectified gain @ K divided by K) of Tsuboyama and Domainome on its own (the Domainome panel has no oracle line, which would hide the models). Each exists with markers (four families) or without (`_all_models`). All text is wrapped to its panel or the figure width and checked to lie inside the figure box (`msr_analysis.style.wrap_to_px`, `text_outside`).

**Gain-per-variant fit (`17_gain_per_variant_fit.py`, `results/17_gain_fit/`).** The rectified gain per variant of the Homology-Restricted MegaScale test set is a straight line in ln K: g(K) = a − b ln K (ESM-MSR: 0.84 − 0.120 ln K, rmse 0.004 kcal/mol over K = 1 to 100; `gain_fit_params.csv` has every model, the oracle, 95% protein-bootstrap intervals and the other shapes tried in `gain_fit_shapes.csv`); `gain_fit_{single,multi}` draw the fits and residuals.

**ProteinGym Activity** has its own folder, [`pg_activity/`](pg_activity/README.md): the wild-type score of each of the 43 Activity assays (read from the raw DMS files and the original papers), the screening curves against that measured WT (three definitions of "hurts", with a floor audit) for the 18 enzymes and for all assays, and the scripts, data and write-up behind them. It is not produced by `08_screening.py`.

## FINAL_* ablation table (`98_final_ablation_table.py`)

Scores the nine FINAL_* families (FINAL_p0 reference; masking: p1, p1_marg; losses: pw0, norank, rcthr0; architecture: noseqhwm, r2a4; data: mega). Each family differs from FINAL_p0 in one hparam group, listed in `msr_analysis/final_runs.py::FAMILIES`. Every family x seed x readout (wt, comb, mt, wt+ = sum of WT singles, epi = comb − WT_A − WT_B) is scored on every benchmark. Each checkpoint is tested under its own masking strategy, as the testing queue ran it.

**Inputs** (no GPU): `<MSR_ROOT>/final_bench/predictions/<folder>/final_checkpoints/<FAMILY>/s<seed>/best.ckpt_sigma1.0_*predictions.csv` and `<MSR_ROOT>/final_pgym_wt|final_pgym_comb/<FAMILY>/s<seed>/<DMS_id>.csv`. If you downloaded the predictions elsewhere, point `MSR_FINAL_BENCH`, `MSR_FINAL_PGYM_WT`, `MSR_FINAL_PGYM_COMB` and `MSR_FINAL_PGYM_XMASK` at those folders. Rows `FINAL_p0@independent` and `FINAL_p1@unmasked` are cross-masking runs, each checkpoint evaluated under the other family's masking. They fill in once the queue's lowest-priority `gamut_xmask` / `pgym_wt_xmask` tiers have run. Labels come from `baselines/public/_tasks` and ProteinGym categories from `baselines/pgym_per_dms/pgym_all_models_per_dms.csv`. A family or seed whose files are missing is left out, and `coverage.csv` lists what was found. A ProteinGym run counts only if it covers at least 212 of the 213 assays.

**Scoring** follows `01_core_rho` / `05_proteingym` / `06_epistasis`: measured rows; Tsuboyama = mean per-library ρ, Domainome = mean per-domain ρ; low-id drops the P/H MegaScale-homologous domains and assays; ΔΔΔG = pooled ρ over doubles with both singles measured, wild-type backbone. `primary` is the deployed readout: wt on singles-only columns and comb where multi-mutants are scored. FINAL_p0_mega trains on `mega_splits`, so its Tsuboyama cells are on its own split (marked †, `cells_long_mega_own_split.csv`).

**Outputs** in `results/98_ablation/`: `cells_long.csv`, `summary.csv` (seed mean / SD), `delta_s1.csv` (paired bootstrap of family − FINAL_p0 at seed 1; protein level, site level for the DMS, assays within category for ProteinGym), `table_main_delta.{md,csv,tex}` and `table_wt_delta.{md,csv,tex}` (the two ablation tables, below), `informativeness.csv` (family spread vs seed noise per column), `head_choice.csv` (comb − wt and whether the two readouts rank the families alike) and `layout_*.md` (candidate tables). `--render-only` redraws the layouts from the CSVs. `--B` sets the number of bootstrap draws (default 200 while drafting).

### The two ablation tables

Both tables list FINAL_p0 in absolute Spearman ρ (seed mean) and every other family as Δρ against it, grouped by the axis it changes. FINAL_p0_mega is left out of both: it belongs in the retrained-methods comparison. The single-protein DMS sets are left out too, since ProteinGym and Domainome cover out-of-distribution fitness. Both remain in `summary.csv` and `layout_D`.

- **Uncertainty:** a cell is bold when |Δ| > 2 SE. The SE combines seed noise and sampling noise.
  - Seed noise is each family's own seed SD. For a single-seed run it is the column's pooled SD.
  - Sampling noise comes from the paired seed-1 bootstrap in `delta_s1.csv`, over proteins, sites, assays or the 13 test domains.
- **‡:** the cell rests on fewer seeds than the rest of its row. Each cell's seeds are listed in the table's CSV.

**`table_main_delta`: deployed readout.** Each dataset is scored the way the model is used: the WT head on singles-only sets, and ½(MT + WT) where multi-mutants are scored.
- Columns: Tsuboyama test, Tsuboyama doubles, epistasis beyond ρ, epistasis cell magnitude, external singles (mean of 10), PTMUL / PTMUL-D, Domainome low-id and ProteinGym avg.
- The epistasis columns score comb(AB) − WT(A) − WT(B) against measured ΔΔΔG on the 12,455 Tsuboyama test doubles that have both singles measured, using `esm_msr.epi_metrics`, the code training validation uses.
  - *Beyond ρ* is the Spearman ρ above the global saturation curve.
  - *Cell magnitude* is computed on double-centred residuals within complete blocks of each position pair. Of the two it separates the families more cleanly; beyond ρ is noisy.
- Rows: every capped-split ablation, plus the two cross-masking rows once the queue's xmask tiers have run.

**`table_wt_delta`: WT head alone.** Every row is read from `wt_lora_pred`, which is what a WT-only deployment (`--skip_reverse`) outputs. On a multi-mutant row it is the WT head's own prediction of that row.
- Columns: Tsuboyama test, Tsuboyama singles, Tsuboyama doubles, external singles, PTMUL / PTMUL-D, Domainome low-id and ProteinGym avg.
- **Rows left out:** ablations that reach the MT head only: no pair loss (pair blocks backpropagate into the MT head only, per the `--block_pair_weight` help), MT LoRA rank 2, and p1 MT marginal (which differs from p1 only in the MT pass's masking).
  - Their WT heads still differ from FINAL_p0's, because training is joint. At matched seed the WT predictions on the Tsuboyama test singles correlate at r = 0.995–0.999, closer than FINAL_p0's own seeds (r ≈ 0.992).
  - So those rows would show seed noise, not an ablation effect. `MT_ONLY` in the script lists them.
- **Columns left out:** the two epistasis columns. The WT head is additive by construction, so a WT-only model has no epistasis to score.
- **Rows kept:** the ones whose change reaches the WT head: p1 masked, no rank loss, rank censor 0 and no sequence head. The cross-masking rows are kept too, since they change how the WT head is read at test time.

Both tables come from one function, `delta_table`, configured by `TABLES` in `98_final_ablation_table.py`. `--render-only` redraws them from the CSVs.
