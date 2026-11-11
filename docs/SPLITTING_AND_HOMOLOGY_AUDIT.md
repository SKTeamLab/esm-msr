# Training–Validation–Test Splitting and Cross-Set Homology Methodology

**Companion documents**: [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md) and [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) (§6).

This document specifies the procedures, tools, thresholds, and assumptions governing:
1. **The training–validation–test splitting pipeline** that produces `data/splits_oct06_capped.{csv,pkl}` and `data/splits_oct06_capped_library_assignment.csv` for ESM-MSR and all retrained baseline models (`MutateEverything`, `ThermoMPNN`/`ThermoMPNN-D`, and `SPURS`/`SPURS multi`).
2. **The cross-set sequence and 3D structural homology audit** (`docs/benchmarks/leak_audit.*`, `docs/benchmarks/split_homology_audit.json`) used to classify training–test overlap across all evaluated internal and third-party predictors.

---

## 1. Training–Validation–Test Splitting Procedure (`splits_oct06_capped`)

The split is implemented in `preprocessing/split_tsuboyama.py` and `preprocessing/homology.py` (with post-split verification in `preprocessing/split_overlap_report.py`) and proceeds in five stages:

### 1.1 Dataset Preprocessing & External Quarantine Pool

1. **Wild-type sequence recovery**: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` is loaded via `MegaScaleDatasetPreprocessor` (`src/esm_msr/preprocess_megascale.py`), capturing wild-type construct sequences (`mut_type == 'wt'`) prior to mutant filtering.
2. **Combined graph node set**: **412 MegaScale `code_wt` scaffolds** (370 scaffolds passing preprocessing filters plus 42 pre-filtered scaffolds recorded in `data/splits_oct06_capped_library_assignment.csv`) are pooled with **776 external quarantine proteins**:
   - **771 chains** from 9 classical in-vitro $\Delta\Delta G$ benchmark datasets (`ssym`, `ptmul`, `s461`, `k2369`, `q3421`, `s571`, `s783`, `s2648`, `s8754`), and
   - **5 functional deep mutational scanning (DMS) targets** (`GRB2`, `DLG4`, `EstA`, `Myo`, `GB1`).
3. **Quarantine scope**:
   - `S669`- and `K3822`-specific proteins outside their curated subsets `S461` and `K2369` were not added to the pre-split quarantine list; empirically, 0 of those 35 proteins exceed the 40% BLOSUM62 local-identity threshold against any kept MegaScale scaffold.
   - `PTMUL-D` is an exact subset of `PTMUL`, which is quarantined.
   - `ProteinGym` (217 DMS assays) and `Domainome` (522 domains; Beltran et al., *Nature* 2025, Zenodo record `14356805`, DOI `10.5281/zenodo.11043642`) span a broad fraction of natural protein families and cannot be quarantined a priori without collapsing the MegaScale training pool; their sequence and 3D structural overlaps with the training split are instead quantified post-hoc in [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md).

### 1.2 Three-Channel Sequence & Structural Homology Graph

Pairwise similarity is evaluated across the combined MegaScale and external protein pool using three complementary channels (`preprocessing/homology.py`):

1. **Profile/k-mer sequence search**: MMseqs2 (`mmseqs easy-search -s 7.5 -e 10.0 --max-seqs 10000`, v13-45111).
2. **3D structural alignment**: Exhaustive Foldseek search (`foldseek easy-search --exhaustive-search 1 -e 10.0`, commit `463739e`), recording $\text{TM}_{\max} = \max(\text{qtmscore}, \text{ttmscore})$.
3. **Dense pairwise local sequence alignment**: Exact all-vs-all Smith–Waterman alignment via Biopython `PairwiseAligner(mode='local', substitution_matrix=BLOSUM62, open_gap_score=-10, extend_gap_score=-1)`, defining shorter-chain local sequence identity as exact residue matches divided by $\min(|s_1|, |s_2|)$ across all 408,330 unordered scaffold pairs.

An undirected homology edge connects two proteins $(u, v)$ if **any** of the four criteria holds (`homology_edge_mask`):

$$\text{edge}(u, v) \iff \left(E_{\text{MMseqs2}} \le 10^{-3}\right) \lor \left(E_{\text{Foldseek}} \le 10^{-3}\right) \lor \left(\text{id}_{\text{BLOSUM62-local}} > 0.40\right) \lor \left(\text{TM}_{\max} > 0.70\right)$$

### 1.3 External Quarantine & Component Bridge Pruning

Excluding scaffolds connected to external benchmarks and pruning oversized graph bridges removes **142 MegaScale scaffolds**:
- **132 scaffolds** carry a direct homology edge to at least one of the 776 external benchmark targets (560 direct homology edges total, including 91 edges at sequence identity $> 0.90$ and 149 edges at $\text{TM}_{\max} > 0.90$).
- **8 scaffolds** belong to connected components containing an external-linked scaffold (transitive homology exclusion).
- **2 designed scaffolds** (`HEEH_KT_rd6_0872`, `HHH_rd4_0124`) are removed by `prune_bridges(max_component_size = 50)`, which partitions oversized connected components via greedy modularity communities and minimum node cuts so no single connected component dominates split assignment.

### 1.4 Five-Stratum Greedy Split Assignment

The remaining **228 kept scaffolds** are partitioned by connected component using `generate_stratified_splits` (`n_trials = 2,000` seeded permutations, `rng = 42`). Large components are pinned to `train`, and remaining components are assigned intact to **`train` (172 scaffolds)**, **`val` (23 scaffolds)**, or **`test` (33 scaffolds)** to match target proportions (`test_frac = 0.18`, `val_frac = 0.12`) across five strata simultaneously:
- Natural protein libraries (`natural_libs`)
- De novo designed libraries (`designed_libs`)
- Libraries containing double mutants (`libs_with_doubles`)
- Total single-mutant count (`n_single`)
- Total double-mutant count (`n_double`)

Three scaffolds (`6NS8` in `val`; `1YEZ` and `2MKX` in `test`) consist solely of stabilizing mutant-background libraries (`<PDB>.pdb_<mut>`) after preprocessing filters. Variant counts in `library_strata` are computed over uncensored numeric `ddG_ML` rows (`censor_range=False`, 542,229 post-filter variants), deterministically reproducing `data/splits_oct06_capped.pkl`, `data/splits_oct06_capped.csv`, and `data/splits_oct06_capped_library_assignment.csv` across `PYTHONHASHSEED` settings (`MMseqs2 13-45111`, `Foldseek 463739e`).

### 1.5 Split Verification & Threshold Sensitivity (`docs/benchmarks/split_homology_audit.json`)

Evaluating the 408,330-pair homology table across all 370 post-filter MegaScale scaffolds and 776 external quarantine proteins confirms strict boundary separation:

| Verification check | Result |
| :--- | :--- |
| Default homology rule enforcement | **25,027** default edges; **0** cross any kept split boundary (`0` internal `train`/`val`/`test`, `0` `kept` $\leftrightarrow$ `external`); all **880** cross-group edges involve the 142 excluded scaffolds |
| 3D structural channel contribution | **550** edges are Foldseek-only (no MMseqs2 hit and local sequence identity $\le 0.40$); **0** cross any kept boundary |
| Tightened local identity $> 0.35$ / $> 0.30$ / $> 0.25$ | **192** (`46` internal + `146` external) / **1,220** (`206` + `1,014`) / **6,453** (`560` + `5,893`) kept pairs |
| Tightened structural $\text{TM}_{\max} > 0.60$ / $> 0.50$ / $> 0.40$ | **91** (`21` internal + `70` external) / **667** (`145` + `522`) / **2,400** (`631` + `1,769`) kept pairs |
| Tightened sequence/structure $E \le 10^{-2}$ / $\le 10^{-1}$ | **10** (`4` internal + `6` external) / **113** (`37` + `76`) kept pairs |

Scaffolds sharing remote 30–40% sequence similarity without triggering the $E \le 10^{-3}$, $>40\%$ identity, or $>0.70$ TM-score exclusions constitute a documented twilight zone (`h`: 8,114 capped-test rows [15.6%] and 7,210 capped-val rows [20.3%]; maximum direct local sequence identity between any kept scaffold and an external target is `0.3953`, and maximum $\text{TM}_{\max}$ is `0.6998`).

---

## 2. Cross-Set Homology & Benchmark Overlap Procedure

To evaluate training–test overlap uniformly across both in-house/retrained models and external predictors trained on literature or metagenomic datasets, the benchmark leakage audit applies a four-step classification protocol (full results and model inventories in [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md)):

1. **Variant & Mutant-Background Normalization**:
   - Every benchmark variant across the 12 external $\Delta\Delta G$ datasets (25,632 rows), Tsuboyama capped validation and test sets (87,544 rows), 7 functional DMS assays (269,471 rows, including 149,360 GB1 rows), Domainome (534,385 rows across 522 domains), and ProteinGym (2,465,767 rows across 217 DMS assays) is normalized to `(protein_id, wild-type sequence, 1-based substitution set)` and verified against its wild-type sequence.
   - Stabilizing mutant-background MegaScale libraries (`<PDB>.pdb_<mut>`) preserve their background-substituted reference sequence under library code `<PDB>_<mut>` while mapping to their parent wild-type scaffold `<PDB>` (`code_wt`) for protein-level and domain-level overlap checks.

2. **Same-Protein Family Clustering & Construct Alignment**:
   - All benchmark and training-inventory sequences are searched all-vs-all with MMseqs2 (`easy-search -s 7.5 -e 10 --max-seqs 20000`, invalidated by SHA-256 FASTA content hash whenever sequence inventories change).
   - Alignment coverage is computed unconditionally relative to the shorter sequence, $\min(\text{alnlen}, \min(L_p, L_q)) / \min(L_p, L_q)$, paired with a length-ratio constraint $\min(L_p, L_q) / \max(L_p, L_q) \ge 0.80$. Same-protein families (`P`) are formed by single-linkage clustering at $\ge 95\%$ sequence identity and $\ge 90\%$ shorter-sequence coverage (with a secondary $\ge 90\%$ identity / $\ge 50\%$ coverage rule for classical literature constructs), allowing truncated domain constructs (e.g., 48-residue PDB constructs inside 56-residue Pfam domains) to cluster cleanly without bridging single domains to multidomain proteins.
   - Within each protein family, sequences are globally aligned (Biopython, BLOSUM62) to the family representative so exact (`M`) and constituent (`m`) mutation matches are detected across alternate PDB entries, purification tags, residue-numbering offsets, and reverse mutant structures (`mt_pos_wt`, such as reverse entries in `S9028` and `Ssym`).

3. **Coverage-Tiered Sequence & 3D Structural Homology Assignment**:
   - Candidate hits for each `(training_inventory, query_protein)` pair are partitioned by test-protein coverage tier prior to ranking by E-value: hits covering $\text{cov}_{\text{test}} \ge 0.50$ are evaluated first for strong homology (`H`), and hits covering $\text{cov}_{\text{test}} \ge 0.30$ are evaluated next for weak/ambiguous similarity (`h`). Evaluating coverage tiers before selecting the minimum E-value ensures short, high-identity local motif hits ($\text{cov}_{\text{test}} < 0.30$) never shadow full-domain homologs.
   - On **Domainome** and **ProteinGym**, MMseqs2 sequence homology is augmented with exhaustive Foldseek 3D structural alignment of all query structures (the 522 Domainome 1.0 domain PDBs from Beltran et al., 2025, Zenodo `14356805`, and the 217 ProteinGym v1.3 AlphaFold2 structures) against the 172 capped-train scaffold structures (`docs/benchmarks/leak_audit_structure_homology.csv`):
     - **Strong homolog (`H`)**: $E_{\text{MMseqs2}} \le 10^{-3}$ over $\text{cov}_{\text{test}} \ge 0.50$, or $E_{\text{Foldseek}} \le 10^{-3}$ / $\text{qTM} > 0.70$ over $\text{qcov} \ge 0.50$.
     - **Weak/twilight similarity (`h`)**: $E_{\text{MMseqs2}} \le 10^{-2}$ or sequence identity $\ge 30\%$ over $\text{cov}_{\text{test}} \ge 0.30$, or $E_{\text{Foldseek}} \le 10^{-2}$ over $\text{qcov} \ge 0.30$.

4. **Exact Metagenomic Hold-Out Verification (`ESM3ΔG`)**:
   - Every wild-type and mutant amino acid sequence across all 3.53 million benchmark rows is compared by exact SHA-1 hash join against the 960,216-sequence `MGnify Stability` training index (Zenodo `19411306`), followed by high-sensitivity MMseqs2 (`-s 7.5 -e 10 --max-seqs 500`) across all benchmark query proteins.

---

## 3. Methodological Assumptions & Boundaries

1. **Conservative upper-bound row accounting**: Unless a model's training stage restricts mutation order (e.g., single-mutant-only training), every valid row of a training library is treated as exposed during training. Private quality, censoring, or early-stopping filters can only remove training rows, so reported `M` and `m` fractions upper-bound exact exposure.
2. **Protein-level split disclosures (`M?`)**: Predictors that publish domain-level splits without per-variant split assignments (`EsmTherm` 80/10/10 per-row split; `IFUM` domain split) carry exact protein-level overlap (`P`) and inferred mutation-level expectations (`M? = 0.8 × P` for `EsmTherm`; `1.0 × P` for `IFUM`).
3. **Local alignment tie-breaking & 2-hop twilight zone**: Biopython's local aligner evaluates a single optimal BLOSUM62 traceback (`next(iter(aligner.align(s1, s2)))`) normalized by $\min(|s_1|, |s_2|)$, and external quarantine outside external-containing connected components is 1-hop on direct homology edges. Scaffolds sitting two hops from an external benchmark via an excluded intermediate remain strictly below the hard thresholds on all direct pairs ($\text{id} \le 0.3953$, $\text{TM}_{\max} \le 0.6998$).
4. **Single-chain monomer representation**: Sequence and Foldseek structural searches extract single chains (`chain = 'A'` for MegaScale, Domainome, and functional DMS targets), omitting inter-chain quaternary interfaces.
5. **Pooled row fractions vs equal-assay summary metrics**: Leak matrix codes report the fraction of pooled dataset rows in each overlap class (`affected_rows / n`), whereas `ProteinGym`, `Domainome`, and `Tsuboyama` headline Spearman figures weight each assay, domain, or scaffold equally. Per-assay and per-domain overlap tables (`docs/benchmarks/leak_audit_pg_per_dms.csv`, `docs/benchmarks/leak_audit_structure_homology.csv`) provide protein-level resolution.
6. **Descriptive scope**: Overlap classes document shared training exposure rather than estimating causal correlation inflation.

**Software versions**: MMseqs2 `13-45111`, Foldseek `463739e`, Biopython `1.85`, pandas `2.3.0`, NetworkX `3.3`, SQLite `3`.
