# Specification & Audit: Training–Validation–Test Splitting and Cross-Set Homology Overlap Analysis

**Status**: CURRENT / REMEDIATED (November 2026)  
**Companion documents**: [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md), [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) (§6), [`known_issues.md`](known_issues.md) (§10), [`analysis_scripts/leak_audit/README.md`](../analysis_scripts/leak_audit/README.md).

This document specifies and audits:
1. **The training–validation–test splitting procedure** that produced `data/splits_oct06_capped.{csv,pkl}` and routes MegaScale data into ESM-MSR and retrained baselines (`MutateEverything`, `ThermoMPNN`/`ThermoMPNN-D`, and `SPURS`/`SPURS multi`).
2. **The cross-set homology overlap and leakage pipeline** (`preprocessing/split_overlap_report.py` and `analysis_scripts/leak_audit/run_all.py`), recording the code-level fixes applied in November 2026, the integration of 3D Foldseek structural homology on Domainome and ProteinGym, and the methodological assumptions that bind the reported numbers.

---

## 1. Training–Validation–Test Splitting Procedure (`splits_oct06_capped`)

### 1.1 Branch Unification (`devel-experimental` → `devel-revision`)

Previously, `devel-revision` carried the `data/splits_oct06_capped.{csv,pkl}` artifacts while retaining the legacy May 2026 `preprocessing/split_tsuboyama.py`. The four authoritative splitting commits from `devel-experimental` have now been cherry-picked onto `devel-revision` (`e6fe2bd`, `4f78186`, `6a6fd55`, `24cb698`), bringing:
- `preprocessing/homology.py` — three-channel MMseqs2 + Foldseek + BLOSUM62 local-identity homology graph builder, modularity/min-cut bridge pruner (`prune_bridges`), and 5-stratum greedy splitter (`generate_stratified_splits`).
- `preprocessing/split_tsuboyama.py` — updated wild-type sequence extraction (`ds.aa_seq_wt`) and external benchmark quarantine pipeline.
- `preprocessing/split_overlap_report.py` — standalone verification report over the split pair table.
- `data/splits_oct06_capped_library_assignment.csv` — full 412-scaffold split/exclusion assignment table (`172` `train`, `23` `val`, `33` `test`, `142` `excluded`).
- `src/esm_msr/preprocess_megascale.py` — `self.aa_seq_wt` wild-type sequence dictionary captured before `wt`-row filtering.

### 1.2 Splitting Algorithm Specification

`data/splits_oct06_capped.{csv,pkl}` is constructed across five stages:

1. **Dataset Preprocessing & Wild-Type Sequence Recovery**:
   - Loads `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` via `MegaScaleDatasetPreprocessor`, retaining true wild-type sequences from `ds.aa_seq_wt` (`mut_type == 'wt'` rows captured before mutant filtering).
   - Pools **412 MegaScale `code_wt` scaffolds** (370 after preprocessing filters + 42 pre-filtered scaffolds recorded in the assignment table) with **776 external quarantine proteins**: **771 chains** from 9 classical ΔΔG sets (`ssym`, `ptmul`, `s461`, `k2369`, `q3421`, `s571`, `s783`, `s2648`, `s8754`, prefixed `test_`) and **5 functional DMS targets** (`GRB2`, `DLG4`, `EstA`, `Myo`, `GB1`).
   - **Intentional quarantine scope**:
     - `S669`- and `K3822`-specific proteins outside their curated subsets `S461` and `K2369` were intentionally excluded from quarantine (and empirically 0 of those 35 proteins exceed the 40 % BLOSUM62 local-identity cap against kept scaffolds).
     - `PTMUL-D` is an exact subset of `PTMUL`, which is quarantined.
     - `ProteinGym` (217 DMS) and `Domainome` (522 human Pfam domains) are too large to quarantine without collapsing the MegaScale training pool and bridging unrelated clusters; their overlaps are therefore measured post-hoc by the leakage audit (including 3D Foldseek structural search).

2. **Three-Channel Homology Graph (`preprocessing/homology.py`)**:
   - **Sequence search**: `mmseqs easy-search -s 7.5 -e 10.0 --max-seqs 10000`.
   - **3D structure search**: `foldseek easy-search --exhaustive-search 1 -e 10.0` (`struct_tm_max = max(qtmscore, ttmscore)`).
   - **Dense pairwise local sequence identity**: All-vs-all Biopython `PairwiseAligner(mode='local', substitution_matrix=BLOSUM62, open_gap_score=-10, extend_gap_score=-1)`, dividing exact matches by `min(len(s1), len(s2))` (`408,330` unordered pairs).
   - **Composite edge rule (`homology_edge_mask`)**:
     $$\text{edge}(u, v) \iff \left(E_{\text{MMseqs2}} \le 10^{-3}\right) \lor \left(E_{\text{Foldseek}} \le 10^{-3}\right) \lor \left(\text{id}_{\text{BLOSUM62-local}} > 0.40\right) \lor \left(\text{TM}_{\max} > 0.70\right)$$

3. **External Quarantine & Bridge Pruning (`142` excluded scaffolds)**:
   - **132 scaffolds** have a direct homology edge (`560` edges total; `91` at identity > 0.90, `149` at TM > 0.90) to one of the 776 external/functional targets.
   - **8 scaffolds** sit in external-touching connected components via transitive homology.
   - **2 designed scaffolds** (`HEEH_KT_rd6_0872`, `HHH_rd4_0124`) are removed by `prune_bridges(max_component_size = 50)` using greedy modularity communities + minimum node cuts.

4. **Five-Stratum Greedy Assignment (`generate_stratified_splits`)**:
   - Evaluates `n_trials = 2,000` seeded permutations (`rng = 42`), pinning large components to `train` and assigning remaining components to `train` (`172`), `val` (`23`), or `test` (`33`) to match `test_frac = 0.18` and `val_frac = 0.12` across `natural_libs`, `designed_libs`, `libs_with_doubles`, `n_single`, and `n_double`.
   - Verified post-hoc (`09_homology_audit.py`): **0 of 25,027** default edges cross any `train`/`val`/`test` or `kept ↔ external` boundary.

5. **Non-Breaking Splitting Characteristics (Documented As-Is)**:
   - Biopython `_local_identity` selects `next(iter(aligner.align(s1, s2)))` over co-optimal BLOSUM62 tracebacks without coverage normalization (`≤ 14` percentage-point order sensitivity on 5 % of sampled pairs); all pairs in `splits_oct06_capped` remain strictly separated under the default rule (`0` cross-edges).
   - Quarantine outside external-touching components is 1-hop on direct external edges, leaving a documented `30–40 %` identity twilight zone (`h`: `15.6 %` of capped-test rows, `20.3 %` of capped-val rows; max direct identity to external `0.3953`, max TM `0.6998`).
   - Three scaffolds (`6NS8` in `val`; `1YEZ` and `2MKX` in `test`) consist solely of `.pdb_<mut>` stabilizing mutant-background libraries after preprocessing filters.

---

## 2. Cross-Set Homology & Leakage Pipeline: Remediations Applied (November 2026)

All reproducibility and robustness issues identified during the audit have been resolved in `analysis_scripts/leak_audit/` and propagated into `docs/benchmarks/*`, `analysis/reference/*`, `BENCHMARKS_OCT2026.md`, and `LEAKAGE_AUDIT_OCT2026.md`:

### 2.1 End-to-End Reproducibility (`analysis_scripts/leak_audit/run_all.py` + `data/`)
- Added upstream extraction scripts `00_build_inventories.py` (SPURS, EsmTherm, ThermoMPNN), `00b_extract_ifum_split.py` (IFUM `dataset_splits.pdf`), `00c_prepare_mgnify.py` (Zenodo MGnify FASTA + MMseqs2 search), centralized path/tool resolver `lib.py`, and single-command topological runner `run_all.py`.
- Tracked the 11 small plain-text reference files (`792 KB` total: `S2450.tsv`, `S2450.fasta`, `S9028.csv`, `DD_SM1242.csv`, `ifum_train_proteins.csv`, `esmtherm_split.csv`, `esmtherm_train_proteins.csv`, `tmpnnd_train_proteins.csv`, `spurs_single_train_proteins.csv`, `spurs_multi_train_proteins.csv`, `thermompnn_upstream_train_proteins.csv`) in `analysis_scripts/leak_audit/data/` with a `.gitignore` blocking all large/binary intermediates.
- Added SHA-256 FASTA content hashing (`mmseqs_hits.tsv.sha256`) in `05_proteins.py` so stale MMseqs2 caches automatically invalidate whenever `proteins.pkl` changes.

### 2.2 Bug Fixes & Resulting Quantitative Updates

1. **`09_homology_audit.py` External Node Mapping & Edge Mask (`split_homology_audit.json`)**:
   - **Fix**: Mapped all 771 `test_*` external nodes (`extids`) alongside the 5 functional DMS nodes to `'external'`, applied the composite default homology edge mask to `ex_pairs`, and split sensitivity cross-edge counts into `cross_internal` (`train/val/test`) and `cross_external` (`kept ↔ external`).
   - **Result**: Confirmed `default_cross_edges_real = 0` (`0` internal + `0` external), corrected `default_cross_edges_involving_excluded` from `366 → 880`, decomposed the 142 excluded scaffolds into `132` direct external-edge + `8` external-component + `2` bridge-pruned scaffolds, and updated threshold sensitivity (`identity > 0.30`: `1,220` = `206` internal + `1,014` external; `TM > 0.50`: `667` = `145` + `522`; `E ≤ 1e-2`: `10` = `4` + `6`).

2. **Eliminated E-Value Shadowing (`11_homology_classes.py`, `15_mgnify.py`)**:
   - **Mechanism**: Previously, a single training hit per query protein was selected by minimum `evalue` *before* checking alignment coverage (`cov_test ≥ 0.50` for `H`, `≥ 0.30` for `h`). A short, high-identity local motif hit (`cov_test < 0.30`) with a smaller E-value shadowed genuine full-domain homologs (`cov_test ≥ 0.50` or `≥ 0.30`) with slightly larger E-values, falsely downgrading the protein to `low`/`none`.
   - **Fix**: Partition all hits per `(inventory, query_pid)` by qualifying coverage tier first (`cov_test ≥ 0.50` for `H`, then `cov_test ≥ 0.30` for `h`), picking the best hit within the highest qualifying tier (`min(evalue, -fident, -cov_test)`).
   - **Result**: `esm_msr_capped` had `0` shadowed proteins on classical/Tsuboyama sets; `ESM3ΔG` upgraded human myoglobin (`MYO`: `– → h1`), `Ssym` (`h.21 → h.72`), `S783` (`h.22 → h.44`), `Q3421` (`h.24 → h.37`), and `Domainome` (`h.49 → h.51`).

3. **Normalised `.pdb_<mut>` Mutant-Background Libraries (`01`, `02`, `04`, `10`, `11`, `14`, `15`)**:
   - **Fix**: Replaced `\.pdb$` stripping with `norm_wt_name` (`<PDB>.pdb_<mut>` → `code = <PDB>_<mut>`, `code_wt = <PDB>`), preserving distinct mutant-background sequences per library code while linking every background library to its wild-type scaffold.
   - **Result**: Recovered `2MKX` (`1,778` test rows = `3.4 %`), `1YEZ` (`1,115` test rows), and `6NS8` (`1,317` val rows = `3.7 %`), raising `EsmTherm` same-protein (`P`) overlap to `90.6 %` (val) / `85.6 %` (test), `IFUM` to `71.5 %` (val) / `48.6 %` (test), and giving 100 % row conservation (`52,071 / 52,071` rows across 33 scaffolds) in `leak_audit_esm3dg_libraries.csv`.

4. **Direction-Invariant Shorter-Sequence Coverage (`05_proteins.py`, `06_families.py`)**:
   - **Mechanism**: Previously, `05_proteins.py` symmetrized directed MMseqs2 hits `(a → b)` into `(min(a,b), max(a,b))` and aggregated `qlen=('qlen','max'), tlen=('tlen','max')`. When MMseqs2 emitted a hit in only one direction (`48 aa → 56 aa`), `min(qlen, tlen)` was `48` (shorter length, `cov = 48/48 = 1.00`); when MMseqs2 emitted hits in *both* directions (`40.3 %` of pairs), `max` overwrote *both* `qlen` and `tlen` with `56` (longer length, `cov = 48/56 = 0.857 < 0.90`), silently switching the coverage denominator based on hit bidirectionality!
   - **Fix**: Looked up true `len_p` and `len_q` directly from `proteins.pkl`, defined `coverage` unconditionally over `min(len_p, len_q)`, and added an explicit `len_ratio = min(len_p, len_q) / max(len_p, len_q) ≥ 0.80` guard in `06_families.py` so single domains (`56 aa` GB1) never merge multidomain proteins (`448 aa` `SPG1_STRSG`).
   - **Result**: Identified a second 100 %-identical Domainome ↔ capped-train domain pair previously hidden by bidirectional length inflation: `Q9H161_PF00046_216` (56 aa, human ALX4 homeodomain) = MegaScale train library `2M0C` (48 aa, `48/56 = 0.857 ≥ 0.80`, `100.0 %` identity, **889 exact single-mutant rows**), shifting Domainome from `M.002 P.006` (`1,134` exact) to `M.004 P.004` (`2,023` exact + `2,288` `P`, total `4,311` rows unchanged).

5. **Construct- & Reverse-Aligned Literature Exact Matching (`12_code_exact.py`)**:
   - **Fix**: Replaced `S2648` proxy upper bounds on `DDGemb` with direct `S2450.tsv + S2450.fasta` alignment, added family/construct sequence alignment across alternate PDB accessions for `DDGemb`, `DDMut` (`S9028 + SM1242`), and `GeoDDG-3D` (`S8754`), added constituent (`m`) tracking on multi-mutant sets (`PTMUL`, `PTMUL-D`, `GB1`), and matched reverse mutant structures (`mt_pos_wt`, e.g. `Ssym` reverse halves and `S9028` reverse rows).
   - **Result**: Upgraded missed identical/constituent overlaps across `DDMut` (`Ssym` `29.7 % → 88.3 %`, `Q3421` `65.7 % → 83.5 %`, `PTMUL-D` `36.2 % → 71.1 %`, `S669` `0 → 10.0 %`, `S461` `0 → 14.5 %`), `GeoDDG-3D` (`S2648` `97.5 % → 100 %`, `Q3421` `68.7 % → 97.9 %`, `Ssym` `37.4 % → 50.0 %`, `PTMUL` `P.97 → m.73 P.24`, `S669` `0 → 1.2 %`), and `DDGemb` (`S783` `M.01 → 0` exact, `Q3421` `45.3 % → 63.4 %`, `PTMUL` `P.79 → m.48 P.32`).

6. **3D Foldseek Structure Homology on Domainome & ProteinGym (`05d_structure_homology.py`)**:
   - **Addition**: Exhaustive Foldseek 3D alignment of all 522 Domainome and 217 ProteinGym AF2/PDB structures against the 758 MegaScale/external benchmark structures (`leak_audit_structure_homology.csv`), integrated into `11_homology_classes.py`, `14_pg.py`, and `18_db.py` (`H`: `E_struct ≤ 1e-3` or `qTM > 0.70` with `qcov ≥ 0.50`; `h`: `E_struct ≤ 1e-2` with `qcov ≥ 0.30`).
   - **Result**: On **Domainome**, capped-train `H` domains rose from `40 → 60` (`+20` remote 3D structural homologs, `8.0 % → 10.9 %` of rows, `H.11`); on **ProteinGym**, capped-train `H` assays rose from `0 → 3` (`4,607` rows, `H.002`: `4G3O`, `2CJJ`, `2GP8`) and `h` assays from `1 → 2` (`6,305` rows, `h.003`: adding `RNC_ECOLI_Weeks_2023`).

7. **Notation & Retraining Alignment (`13_matrix.py`, `BENCHMARKS_OCT2026.md`, `LEAKAGE_AUDIT_OCT2026.md`)**:
   - Replaced the pretraining-corpus flag `C`/`ᶜ` with `–` across all scripts, tables, and documentation, and aligned all SPURS and ThermoMPNN-D documentation to their Oct06 capped-split retrained inventories.

---

## 3. Methodological Assumptions & Boundaries

1. **Upper-bound row accounting**: Every row of a trained library is treated as seen unless model-type-restricted (singles vs doubles), bounding private quality/censoring filters from above.
2. **Protein-level inference (`M?`)**: `EsmTherm` (`0.8 × P`) and `IFUM` (`1.0 × P`) publish domain splits rather than per-mutant split tables.
3. **Single-chain monomer representation**: All sequence and Foldseek searches operate on single chains (`chain = 'A'` for MegaScale, Domainome, and functional DMS targets), omitting inter-chain quaternary contacts.
4. **Row-pooled codes vs equal-assay averages**: Cell codes report fraction of pooled dataset rows (`leaked_rows / n`), whereas `ProteinGym`, `Domainome`, and `Tsuboyama` Spearman benchmarks weight each assay/domain/scaffold equally (`leak_audit_pg_per_dms.csv` and `leak_audit_structure_homology.csv` provide per-assay/per-domain resolution).
5. **Descriptive interpretation**: Leak codes classify training exposure rather than estimating causal Δρ inflation.
