# Homology & Holdout Filter Audit (`analysis/docs/HOMOLOGY_FILTER_AUDIT.md`)

## 1. Canonical Four-Part Holdout Rule (`msr_analysis/heldout.py`)

Every benchmark scored as a held-out evaluation set in `analysis/` is filtered by the **exact four-part sequence + 3D structure homology rule** that constructed the Homology-Restricted MegaScale split (`data/splits_oct06_capped.pkl`; see [`../../docs/SPLITTING_AND_HOMOLOGY_AUDIT.md`](../../docs/SPLITTING_AND_HOMOLOGY_AUDIT.md) §§1.2, 2.1).

A benchmark protein, domain, assay, or design scaffold is **excluded** (`homologous == True`) if **any** of the four criteria holds against any of the **172 capped training libraries** (validation libraries are not used):

| Criterion | Code | Tool & Normalisation | Exclusion Cut |
| :--- | :---: | :--- | :--- |
| **1. Sequence E-value** | `E` | MMseqs2 (`-s 7.5 -e 10.0`, rescaled to the 1,147-protein split search corpus) | $E \le 10^{-3}$ |
| **2. Whole-domain sequence identity** | `I` | Exact pairwise BLOSUM62 local alignment: `matches / min(L_query, L_target)` (shorter-sequence normalisation, *not* MMseqs2 aligned-column `fident`) | $\text{identity} > 0.40$ |
| **3. Structural E-value** | `S` | Foldseek (`--exhaustive-search 1` / `--max-seqs 3000` on MGnify, corpus-rescaled) | $E_{\text{struct}} \le 10^{-3}$ |
| **4. Structural fold similarity** | `T` | Foldseek TM-score normalised by the shorter chain (`max(qtmscore, ttmscore)`) | $\text{TM}_{\max} > 0.70$ |

### Builder & Loader Enforcement

- **Offline builder**: `analysis/audits/homology_and_splits/active_builders/tsuboyama_rule_flags.py` computes all four quantities and writes `analysis/reference/tsuboyama_rule_flags.csv`.
  - **Self-consistency check**: Running the 23 validation and 33 test libraries of `splits_oct06_capped` through the same code against the 172 training libraries flags **0/23** and **0/33** libraries ($\max \text{identity} = 0.395$, $\max \text{TM} = 0.696$).
  - The 12 classical thermostability sets and 3 functional DMS assays (`DLG4` binding, `GRB2` binding, `MYO`) were quarantined against *both* training and validation libraries when `splits_oct06_capped` was constructed (`0` boundary violations).
- **Runtime enforcement**: `msr_analysis/heldout.py` (`flags()`, `excluded(dataset)`, `keep_ids(dataset)`) is called directly inside `msr_analysis/data.py` (`load_frame`), `msr_analysis/final_runs.py` (`load_final_frame`), `msr_analysis/mgnify.py` (`load_mgnify`), and `msr_analysis/proteingym.py` (`pgym_reference`), so downstream pipelines cannot omit the filter.

### Canonical Holdout Counts Across the Four Unquarantined Benchmarks

| Benchmark (`heldout` key) | Total Items | Excluded (`homologous`) | Held-Out Items | Trigger Breakdown (`E` / `I` / `S` / `T`) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ProteinGym** (`proteingym`) | 217 assays | 41 | **176** (**173** scored) | 24 / 33 / 28 / 31 | Dropping the 4 `PG_DROP` $>1{,}800$-aa assays (`BRCA1`, `BRCA2`, `POLG_CXB3N`, `SCN5A`; 1 of which is among the 41 homologous assays and 3 are held-out) yields **213 scored $\to$ 173 canonical held-out assays**: `Activity=38` (**16** enzymes), `Binding=13`, `Expression=18`, `OrganismalFitness=67`, `Stability=37`. |
| **Domainome 1.0** (`domainome`) | 522 domains | 86 | **436** | 48 / 37 / 62 / 78 | **451,569** measured rows across 436 held-out domains (vs 540,265 rows across all 522 domains). |
| **MGnify Stability** (`mgnify`) | 180,367 scaffolds | 27,687 | **152,680** | 3,953 / 3,964 / 14,738 / 21,707 | **~17,000 scaffolds** are flagged by Foldseek 3D TM-score alone despite low sequence identity. |
| **Rocklin *de novo* designs** (`rocklin_denovo`) | 1,000 designs | 258 | **742** | 2 / 226 / 7 / 40 | 226 designs exceed 40% shorter-sequence identity to *de novo* $\alpha\alpha\beta\beta$ / bundle scaffolds in capped train. |

Supplementary breakdown figure: `results/00_supplementary_figs/holdout_filter.{png,pdf}` (`pipelines/p10_splits_and_leakage.py`).

---

## 2. Reconciling the Two Legitimate Homology Taxonomies (and Retiring Ad-Hoc Rule C)

Prior to the Phase 2 refactor, three distinct homology definitions coexisted across `analysis/`. The refactored codebase retains **only the two scientifically distinct rules**—with explicit naming and boundaries—and retires the third:

1. **Canonical Four-Part Split Holdout Rule (`msr_analysis/heldout.py`, `tsuboyama_rule_flags.csv`)**:
   - **Definition**: `E <= 1e-3 | id_shorter > 0.40 | struct_E <= 1e-3 | tm_max > 0.70` (no alignment-coverage floor; 3D Foldseek applied to every benchmark including MGnify and Rocklin *de novo*).
   - **Where used**: All headline and supplementary model evaluations (`p01`–`p09`, `p11`, `studies/faure_epistasis`, `studies/sigma_mechanisms`).
2. **Oct-2026 Literature Training-Exposure Audit (`audits/homology_and_splits/oct2026_leak_audit/`, `leak_audit_matrix.csv`)**:
   - **Definition**: Row-level exact/constituent mutation matching (`M`/`m`) plus coverage-gated protein/homolog tiers (`P`: $\ge 95\%$ id over $\ge 90\%$ cov; `H`: $E \le 10^{-3}$ or $\text{qTM} > 0.70$ over $\ge 50\%$ test coverage; `h`: $E \le 10^{-2}$ or $\text{id} \ge 30\%$ over $\ge 30\%$ coverage). Sequence-only on MGnify (`mgnify_homology_classes.csv.gz`).
   - **Where used**: Cross-model literature training-overlap superscripts in `docs/BENCHMARKS_OCT2026.md` (`p10_splits_and_leakage`, `00_for_reviewers/reference/`) and explicit sequence-homology stratum diagnostics (`all`, `heldout`, `clean`, `h`, `strong`) in `p04_mgnify_rocklin` (`05_mgnify/mgnify_homology`).
   - **Why the distinction matters on MGnify**: Cross-tabulating `mgnify_homology_classes.csv.gz` against `tsuboyama_rule_flags.csv` shows **22,216 MGnify scaffolds** (`19,352` sequence-`'none'` + `2,864` sequence-`'low'`) that share topology (`TM > 0.70` / `struct_E <= 1e-3`) with training scaffolds despite negligible sequence identity. `p04_mgnify_rocklin` therefore uses `heldout=True` (4-part rule) for all cross-model benchmarks (`05_mgnify/{mgnify_models_*, mgnify_by_mutation_count, rocklin_designs, new_data_epistasis}`) and reserves `heldout=False` strictly for the homology-stratification study (`05_mgnify/mgnify_homology`).
3. **Retired Ad-Hoc Rule C (`proteingym/p0_sigma/04_leak_per_assay.py`)**:
   - Legacy `04_leak_per_assay.py` defined a third, uncalibrated rule (`qtm > 0.7 | mmseqs_fident > 0.4` with aligned-column `fident` and no E-value threshold) and mixed **176** assays in Figure 3c (`esm-msr-small` fallback on 3 huge assays) against **173** assays in Figure 3b (`PG_DROP`).
   - **Resolution**: `04_leak_per_assay.py` is deleted; `msr_analysis/proteingym.py` standardizes every ProteinGym table and figure (including both Figure 3b and Figure 3c) onto the **173 canonical held-out assays** (`include_fallback_small=False` by default) and drops byte-duplicate `*_tsu.csv` and `*_nonleaked.csv` tables.

---

## 3. Pipeline-by-Pipeline Holdout Enforcement Summary

| Pipeline / Study | Benchmarks Filtered | Enforcement Mechanism |
| :--- | :--- | :--- |
| `p01_benchmarks_external` | Domainome (`436` held-out domains -> `04_domainome/`); `EXT12` (`02_thermostability/`) & `DMS3` (`99_misc/dms3/`) (split-quarantined) | `data.load_frame('domainome', heldout=True)` + `external.load_mutation_boot()` |
| `p02_tsuboyama` | `splits_oct06_capped` val (`23`) & test (`33`) (`01_megascale/`, 0/23, 0/33 homologous to train); `ESM3ΔG` excluded via `NO_ESM3DG` | `data.load_frame(ds)` |
| `p03_proteingym` | ProteinGym (`173` held-out assays -> `03_proteingym/` & `00_supplementary_figs/pg_ours_vs_published`) & Domainome (`436` domains) | `proteingym.pgym_reference(heldout=True, drop_large=True)` |
| `p04_mgnify_rocklin` | MGnify (`152,680` scaffolds) & Rocklin *de novo* (`742` designs) across `05_mgnify/` multi-model, mutation-count, Rocklin within-design, and beyond-epistasis tables; unfiltered strata explicitly labelled in `05_mgnify/mgnify_homology` | `mgnify.load_mgnify(heldout=True)` (fixes previously unfiltered `scoring/mgnify_{rdenovo_stats, epistasis_beyond, baselines_compare}.py`) |
| `p05_summary_main` | Figure 3a–c (`00_main_text_figs/figure3`), `06_summary/`, & `99_misc/esm_msr_vs_esm3dg/`: canonical 4-part filter across ProteinGym (`173`), Domainome (`436`), MGnify (`152,680`), and Rocklin (`742`) | Reads canonical `p01`/`p03`/`p04` tables + `proteingym.load_sigma_pgym(heldout=True)` |
| `p08_calibration_length` | `EXT12` (`02_thermostability/rmse/`), `TSU` & `436`-domain held-out Domainome (`08_sequence_length/`), and `DMS3` (`99_misc/dms3/`) | `data.load_frame(ds, heldout=True)` |
| `p09_screening` | MegaScale (`01_megascale/screening/`), external thermostability (`02_thermostability/screening/`), Domainome (`436` domains -> `04_domainome/screening/`), and ProteinGym Activity (`38` held-out assays, `16` held-out enzymes -> `03_proteingym/screening/`) | `data.load_frame('domainome', heldout=True)` + `heldout.keep_ids('proteingym')` |
| `p10_splits_and_leakage` | Visualizes the 4-part filter (`00_supplementary_figs/holdout_filter`), cross-set max identity/TM matrices (`00_supplementary_figs/split_construction/`), `splits_oct06_capped` vs `hyperopt` best partners (`00_for_reviewers/`), and **stratifies `mega_splits` test (`28` libraries) into `12` homology-clean vs `16` train/val-homologous libraries** (`mega_clean_vs_homologous_test`) | `audits/homology_and_splits/active_builders/*` + `reference/tsuboyama_rule_flags.csv` |
| `p11_ablations_and_heads` | All 11 `FINAL_*` families on held-out Domainome (`436`) and held-out ProteinGym (`173`) across `09_ablation/` and `99_misc/dual_view/` (`DMS3` isolated in `99_misc/dms3/`) | `final_runs.load_final_frame(..., heldout=True)` + `proteingym.load_final_pgym_seeds(..., heldout=True)` |
| `p12_manifest` | Records **both** canonical 436-domain `rho` and legacy 522-domain `rho_unfiltered` in `00_for_reviewers/reference/validation_vs_benchmarks.csv` so historical Oct-2026 table cells remain verifiable to $<0.002$ | `load_frame('domainome', heldout=True)` & `load_frame('domainome', heldout=False)` |
| `studies/sigma_mechanisms` | Upgraded from 214 unfiltered assays (which crashed on `NaN` joins after partial `common.py` edits) to **173 held-out assays** (`99_misc/sigma_mechanisms/`) | `proteingym.load_sigma_pgym(heldout=True, drop_large=True)` |
