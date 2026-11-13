# Data-leakage audit of the October 2026 benchmark tables

**Scope.** Every model and dataset reported in [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md), on the Oct06 capped
split (`data/splits_oct06_capped.pkl`). This document gives:
- the method and notation;
- the training inventory of each model and how strong the evidence for it is;
- which overlaps are verified and which are inferred;
- the split-tooling validation;
- the findings for each model, the assumptions behind them, and how to rerun the audit
  (`python analysis/leak_audit/run_all.py`).

It is the companion to §6 of [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) and to
[`SPLITTING_AND_HOMOLOGY_AUDIT.md`](SPLITTING_AND_HOMOLOGY_AUDIT.md). It was revised on 2026-11-12; the changes are
listed in §11.

**Models audited.**
- **ESM-MSR:** released and baseline checkpoints, devel, every head and every σ.
- **ESM-MSR-small:** the March reference.
- **Third-party models trained on Megascale:** EsmTherm and IFUM.
- **Literature-ΔΔG models:** DDGemb, DDMut and GeoDDG-3D.
- **ESM3ΔG.**
- **Rosetta** `cartesian_ddg`.
- **Zero-shot models:** ProSST, ProteinMPNN, ESM3 small/medium/large and the σ = 0 control.

The baselines we retrained from scratch on the capped split are not audited separately: MutateEverything stages 1
and 2, ThermoMPNN / ThermoMPNN-D, and SPURS / SPURS multi (`BENCHMARKS_OCT2026.md` §1.6). They train on a subset
of ESM-MSR's capped-train rows: the 172 scaffolds, or 134 of them for ThermoMPNN-D. So by construction:
- their protein-level codes (`P`, `H`, `h`) equal ESM-MSR's, or are weaker;
- the singles-only models (ME stage 1, ThermoMPNN, SPURS) split ESM-MSR's `M` on multi-mutant rows into `M` + `m`.

**One-line result.**
- **Models trained on the capped split:** the split is sound. ESM-MSR, and the retrained baselines with it, have
  **zero** identical-mutation and **zero** same-protein overlap with:
  - all 12 external thermostability sets;
  - the 33 Tsuboyama capped-test scaffolds;
  - all 7 functional DMS assays.
- **The parts of the benchmark that were never quarantined:**
  - 23 of the 217 ProteinGym assays sit inside capped train, and 3 more are strong 3D homologs of it.
  - 2 of the 522 Domainome domains are capped-train libraries, and 60 more are strong 2D/3D homologs.
- **Megascale-trained third-party models:**
  - EsmTherm's own published split puts **51 %** of capped-test rows in its training set; this is verified row by
    row.
  - IFUM trained on the domains behind up to **46 %** of capped-test rows; this is a ceiling, since its rows are not
    public.
- **Literature-ΔΔG models** trained on 30–100 % of most classical external sets, at the mutation or
  constituent-mutation level.
- **ESM3ΔG** has **0** identical sequences anywhere in the benchmark.
- **Rosetta's** energy function was partly tuned on data that covers 12 % of capped-test rows (Rocklin-2017 designs)
  and up to 28 % of some external sets (Kellogg 2011). It is coded `V` (calibration), not `M`/`P`.

---

## 1. Method

The audit runs end to end via `analysis/leak_audit/run_all.py`. The 19 numbered steps `01`–`19` run in file-name
order; the number is the dependency order (§10).

1. **Normalise the test side (`04_testsets.py`).**
   - Every reported row is reduced to `(protein sequence, set of (WT, position, mutant))`, with positions validated in
     1-based sequence coordinates:
     - 0 mismatches on Tsuboyama val/test and ProteinGym;
     - 1,779 Domainome construct/numbering edge-case rows (0.33 %) dropped and recorded.
   - Rows audited: 3,382,799 in total.

     | dataset | rows |
     |---|---|
     | ProteinGym, all 217 DMS (including the 4 >1,800-residue proteins outside the 213-assay common subset) | 2,465,767 |
     | Domainome (522 domains) | 534,385 |
     | GB1 | 149,360 |
     | Tsuboyama capped val + test, keyed by exact library code (including `.pdb_<mut>` mutant-background libraries) | 87,544 |
     | 12 external sets | 25,632 |
     | the other 6 DMS assays | 120,111 |

2. **Build same-protein families and 2D/3D homology maps (`05`–`08`).**
   - **MMseqs2:** all test proteins and all inventory proteins are searched all-vs-all (`easy-search -s 7.5 -e 10
     --max-seqs 20000`).
     - Coverage is direction-invariant: `min(alnlen, min(len_p, len_q)) / min(len_p, len_q)`.
     - A length-ratio guard (`min_len / max_len ≥ 0.80`) stops a short domain from bridging two unrelated
       multidomain proteins.
     - Construct truncations still cluster at ≥ 95 % identity and ≥ 90 % shorter-sequence coverage. Example: the 48-aa
       `2M0C` inside the 56-aa Domainome `Q9H161_PF00046_216`.
   - **Family alignment:** each family member is globally aligned to its representative (Biopython, BLOSUM62).
     Mutations are mapped to the representative only where the aligned residue carries the same wild-type letter, so
     they match across alternate PDB entries, chain offsets and construct truncations.
   - **Foldseek 3D search (`07_structure_homology.py`):** exhaustive search against the 172 capped-train scaffold
     structures (and the literature-inventory PDB chains) for:
     - **Domainome:** 522 domain structures from Domainome 1.0 (Beltran et al., *Nature* 2025; Zenodo `14356805`);
     - **ProteinGym:** 217 AlphaFold2 structures.
3. **Reconstruct each model's training inventory (`01`, `02`, `03`, `05`, `10`–`16`).** Sources and evidence tiers
   are in §3.
4. **Classify every test row (`10`–`18`).** Exact (`M`) and constituent (`m`) matches use one of three methods:
   - **Megascale inventories:** family-canonical mutation keys (`10`).
   - **EsmTherm and IFUM:** exact joins on the full mutant sequence (`10`).
   - **Literature inventories:** construct-aligned sequence pairs, forward and reverse (`12`).

   Protein-level classes (`P` / `H` / `h`) come from one shared rule (`lib.classify_hits`). The rule takes every
   qualifying hit of a test protein, tests `cov_test ≥ 0.50` for `H` first, then `cov_test ≥ 0.30` for `h`. A short
   motif hit therefore cannot mask a full-domain homolog.

---

## 2. Notation

| code | meaning | test |
|---|---|---|
| `M` | identical variant in training (same protein family, identical mutation set, reverse structures included) | verified against the training rows |
| `m` | every mutation of the row is in training for the same protein, but not this combination | verified |
| `P` | same protein in training, no mutation match | ≥ 95 % identity over ≥ 90 % shorter-sequence coverage (`len_ratio ≥ 0.80`), or an MMseqs2 hit with ≥ 90 % identity over ≥ 50 % of the test protein |
| `H` | strong homolog in training | MMseqs2 E ≤ 1e-3 over ≥ 50 % of the test protein; or (Domainome, ProteinGym) Foldseek E ≤ 1e-3 or qTM > 0.70 over ≥ 50 % |
| `h` | weak or ambiguous homology | MMseqs2 E ≤ 1e-2 or ≥ 30 % identity over ≥ 30 % of the test protein; or Foldseek E ≤ 1e-2 over ≥ 30 % |
| `D` | the training set is documented as derived from this dataset | paper |
| `V` | the rows were used for validation, model selection or calibration (Rosetta) | split file, paper |
| `#` | not a test result: the dataset *is* the training set | file identity |
| `?` | ceiling, not verified: the training rows are not public (`M?`, `V?`) | inference from a public protein-level split |
| `–` | no supervised overlap: zero-shot / unsupervised models, or no class reaches one row | verified or documented |
| `n.a.` | outside the audit scope (legacy hyperopt-split columns) | – |

**Cell codes.** A cell lists every class present with its fraction of the dataset's reported rows, most severe
first, e.g. `M.59 m.05 P.32 H.02`.

**Superscripts in the numeric tables.** `BENCHMARKS_OCT2026.md` prints only the most severe class that reaches 0.5 %
of rows (`v / n ≥ 0.005`):
- the order is `M` / `M?` (ᴹ), `m` (ᵐ), `P` (ᴾ), `H` (ᴴ), `h` (ʰ), `V` / `V?` (ⱽ);
- `D` (ᴰ) is shown only when nothing else reaches the threshold;
- `#` always wins.

The full codes are in `leak_audit_matrix.csv` and in the companion matrices.

---

## 3. Inventories and the evidence behind them

**Evidence tiers:**
- **row-level:** the training rows (or their exact sequences) are public, and every count is a verified match.
- **protein-level:** only the protein or domain list is public, so mutation-level counts are ceilings (`?`).
- **calibration:** global parameters were tuned on the data (`V`).
- **documented:** described in the paper, no data.

| model row(s) | training inventory | size | source | evidence |
|---|---|---|---|---|
| ESM-MSR (released/baseline, devel; every head and σ) | every substitution of the 172 capped-train scaffolds; checkpoints selected on capped val (`V`) | 229,721 rows; 175 library codes, 3 of them `.pdb_<mut>` | `data/splits_oct06_capped.pkl` + Tsuboyama CSV | row-level |
| ESM-MSR-small (March reference) | the 120 train libraries of the legacy hyperopt split; selected on its 26 val libraries | — | `data/hyperopt_splits.pkl`, the `split_file` in the checkpoint's `hparams.yaml` (md5 match) | row-level |
| EsmTherm | `K50_dG_Dataset1_Dataset2`: 47 test-only domains; the other 414 split 80/10/10 per sequence (`build_dataset.py`, seed 42) | 527,785 sequences with split labels (incl. 198,783 multi-mutants and 64,784 indels) | S2 Spreadsheet `pcbi.1012248.s006.csv` (PLoS Comput Biol 2024), plus `wildtype_split.csv` (tracked as `esmtherm_split.csv`) | **row-level** (exact-sequence join) |
| IFUM | Megascale Dataset2, MMseqs2-30 % cluster split: 322 train domains, 86 in a merged val/test block; plus unlabelled DisProt (≤ 70 aa) | 322 train domains | `dataset_splits.pdf` → `ifum_train_proteins.csv` (`02_ifum_inventory.py`; the parse reproduces 322/86 exactly) | protein-level: `M?` = Dataset2 rows of train domains, `V?` = rows of val/test domains |
| DDGemb | `S2450` = S2648 minus 198 mutations > 25 % identical to S669, doubled with reverse variants | 2,450 forward + 2,450 reverse | `S2450.tsv` + `S2450.fasta` (ddgemb.biocomp.unibo.it) | row-level |
| DDMut (deployed) | single-mutant model: `S9028` = S2648 + FireProtDB (4,514 forward + 4,514 reverse); multi-mutant model: `SM1218` (training) + `SM444` (blind test), which the paper says the deployed model was trained on | 9,028 + 1,662 rows | `S9028.csv`, `DD_SM1218.csv`, `DD_SM444.csv` (biosig.lab.uq.edu.au/ddmut/data) | row-level |
| GeoDDG-3D | fine-tuned on `S8754`; encoder pretrained as GeoFitness on 74 DMS fitness sets (52 MaveDB + 22 DeepSequence; 70/10/20 per mutation within each protein) | 8,754 ΔΔG rows; 74 DMS proteins | GeoStab `data/ddG/S8754.csv`, `data/dms/dms.csv` (wild types: `geofitness_dms_wt.fasta`) | row-level (S8754); protein-level (GeoFitness) |
| ESM3ΔG | MGnify Stability fine-tune, **not** Megascale; the `*_augmented_*` checkpoints use a ~456k subset | 960,216 train sequences (60–80 aa) | Zenodo 19411306 `mgnify_training_index.csv` (Cho et al., bioRxiv 10.64898/2026.05.19.726285) | row-level (0 identical sequences) |
| Rosetta `cartesian_ddg` (2019, `beta_nov16_cart`) | no per-variant training; see the note below | 20 + 1 global parameters; Kellogg set: 1,210 mutations / 75 PDB | Rosetta docs `Updates-beta-nov16`; Park et al. 2016 JCTC SI; Kortemme-Lab `kellogg.csv` | calibration (`V`) |
| ProSST | AlphaFoldDB masked-LM pretraining; no ΔΔG fine-tune | — | ProSST paper | documented (`–`) |
| ProteinMPNN | PDB assemblies before 2021-08-02; sequence recovery | — | ProteinMPNN paper | documented (`–`) |
| ESM3 small/medium/large; σ = 0 control | UniRef/MGnify/JGI + PDB + AFDB pretraining; no ΔΔG fine-tune | — | ESM3 paper | documented (`–`) |

**What Rosetta's parameters were fit to:**
- **Reference energies.** The 20 `beta_nov16` reference energies were refit on an objective that includes:
  - the rank correlation on ProTherm ΔΔGs "following PMID 21287615" (Kellogg 2011);
  - the protease resistance of the Rocklin et al. 2017 designs.
- **Scale factor.** The REU → kcal/mol factor 1/2.94 is a least-squares fit, almost certainly on the Kellogg set
  (Park SI).

---

## 4. Concrete versus suspected overlap

The matrix (`leak_audit_matrix.csv`) holds the count for every cell. Each count falls into one of three evidence
classes.

**Verified row by row.** A training row and a test row are the same variant (or the same protein, for `P`), checked
by sequence. Codes without `?`:
- **ESM-MSR / ESM-MSR-small:** every cell (`M`, `m`, `P`, `H`, `h`).
- **EsmTherm:** `M` and `V` by exact join on the full mutant sequence.
  - capped test: `M.51` (26,789 rows);
  - capped val: `M.44`;
  - ProteinGym: `M.02`;
  - Domainome: `M.001`;
  - GB1: 42 rows.
- **DDGemb, DDMut, GeoDDG-3D (S8754):** every `M` / `m`, construct-aligned (forward and reverse).
- **ESM3ΔG:** 0 identical sequences.
- **All third-party `P`, `H`, `h` classes:** sequence (and, on Domainome / ProteinGym, structure) homology against
  the published training proteins.

**Inferred from a public protein-level split.** These are ceilings, marked `?`:
- **IFUM `M?`:** the test rows whose exact mutant sequence is a Dataset2 row of one of IFUM's 322 train domains.
  - Its domain-level split means every Dataset2 row of a train domain is in the training pool, so on Tsuboyama this
    is exact by construction: capped test `M?.46` (24,200 rows).
  - What is unverified is the row filters and whether the released checkpoint used this split.
  - The previous rule `M? = P` overstated GB1 (`M?1` → `M?<.001`) and Domainome (`M?.05` → `M?.004`).
- **IFUM `V?`:** rows of its 86-domain merged validation/test block, about half of which was used for model
  selection: capped test `V?.47`.

**Documented or calibration only (no mutation-level claim):**
- **DDGemb on S2648, DDMut on S2648 / S8754 (`D`):** the training sets are documented as derived from these sets; the
  verified `M` is also given.
- **GeoDDG-3D, GeoFitness:** DMS fitness pretraining is matched at protein level only. The `dms.csv` rows were not
  audited, so the overlap could be up to ~70 % `M` on the proteins it shares.
  - DLG4 abundance / binding: `P1`;
  - S461: `P.11`;
  - S669: `P.07`;
  - 3 Domainome domains;
  - 74 ProteinGym assays (GeoDDG-3D is not run on ProteinGym).
- **Rosetta (`V`):** global parameters were tuned on these rows (§7.6).
- **Zero-shot models (`–`):** pretraining corpora only. Two design-level caveats are in §7.7.

---

## 5. The datasets overlap each other

Measured on unique `(PDB code, sequence, mutation)` identifiers, before cross-PDB family alignment:

| set | s571 | s783 | s2648 | s8754 | s669 | s461 | ssym | q3421 | k3822 | k2369 | ptmul | ptmuld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **s571** (557) | 557 | 37 | 52 | 216 | 17 | 6 | 0 | 113 | 141 | 74 | 0 | 0 |
| **s783** (783) | 37 | 783 | 10 | 691 | 33 | 30 | 46 | 475 | 233 | 144 | 0 | 0 |
| **s2648** (2648) | 52 | 10 | 2648 | 2587 | 0 | 0 | 194 | 1549 | 979 | 566 | 0 | 0 |
| **s8754** (6411) | 216 | 691 | 2587 | 6411 | 0 | 0 | 256 | 2409 | 2682 | 1826 | 0 | 0 |
| **s669** (669) | 17 | 33 | 0 | 0 | 669 | 461 | 0 | 54 | 388 | 192 | 0 | 0 |
| **s461** (461) | 6 | 30 | 0 | 0 | 461 | 461 | 0 | 50 | 324 | 141 | 0 | 0 |
| **ssym** (684) | 0 | 46 | 194 | 256 | 0 | 0 | 684 | 205 | 0 | 0 | 0 | 0 |
| **q3421** (3421) | 113 | 475 | 1549 | 2409 | 54 | 50 | 205 | 3421 | 1974 | 1050 | 0 | 0 |
| **k3822** (3822) | 141 | 233 | 979 | 2682 | 388 | 324 | 0 | 1974 | 3822 | 2369 | 0 | 0 |
| **k2369** (2369) | 74 | 144 | 566 | 1826 | 192 | 141 | 0 | 1050 | 2369 | 2369 | 0 | 0 |
| **ptmul** (912) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 912 | 536 |
| **ptmuld** (536) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 536 | 536 |

Construct/family alignment and reverse structures (`12_literature_exact.py`, §7.4) raise the cross-PDB exact matches
further:

| training set | overlap with |
|---|---|
| `S8754` | 100 % of `S2648`, 97.9 % of `Q3421`, 91.2 % of `K2369`, 89.1 % of `K3822`, 50.0 % of `Ssym`, 8 of 669 rows of `S669` |
| `S9028` | 88.3 % of `Ssym`; 67 rows each of `S669` and `S461` |

On structure/variant keys, `K2369 ⊂ K3822`, `S461 ⊂ S669` and `PTMUL-D ⊂ PTMUL` hold exactly.

---

## 6. Does the split's homology tooling hold up?

The split was produced by `preprocessing/split_tsuboyama.py` and `preprocessing/homology.py`
(`data/splits_oct06_capped.pkl`, `data/splits_oct06_capped_library_assignment.csv`).

**The edge rule.** Two proteins are joined by an undirected homology edge when any of these holds: MMseqs2
E ≤ 1e-3, Foldseek E ≤ 1e-3, BLOSUM62 local identity > 0.40, or TM-score > 0.70 (normalised by the shorter chain).

`09_split_audit.py` re-evaluated the 408,330-pair table (`homology_pairs.csv`) over:
- all **370** Megascale libraries: 172 train, 23 val, 33 test and 142 excluded;
- all 776 external quarantine nodes: 771 structures from 9 classical ΔΔG sets plus 5 functional DMS proteins.

The results are in `docs/benchmarks/split_homology_audit.json`:

| test | result | verdict |
|---|---|---|
| rule is enforced | 25,027 default edges; **0** cross any kept boundary (0 internal `train/val/test`, 0 `kept ↔ external`); all **880** cross-group edges involve the 142 excluded scaffolds | pass |
| quarantine accounting | of the 142 excluded scaffolds: **132** have a direct homology edge to an external target (560 edges; 91 at identity > 0.90, 149 at TM > 0.90); **8** more belong to connected components that touch an external target transitively; **2** (`HEEH_KT_rd6_0872`, `HHH_rd4_0124`) were removed by `prune_bridges(max_component_size = 50)` | pass |
| Foldseek adds value | 550 structure-only edges (no MMseqs2 hit, identity ≤ 0.40); **0** cross a kept boundary | pass |
| threshold sensitivity | below | E-value + 40 % identity / 0.70 TM caps jointly guard the split |
| twilight zone (`h`) | 8,114 capped-test rows (15.6 %) and 7,210 val rows (20.3 %) have a 30–40 % identity match to a train scaffold that the rule did not exclude | flagged `h`, not `H` |
| non-transitive quarantine | 134 train, 7 val and 17 test scaffolds sit 2 hops from an external protein via an excluded bridge; every direct pair is below the hard caps (max identity 0.395, max TM 0.6998) | no strong pair missed |
| intentional quarantine scope | the split quarantined 9 classical ΔΔG sets (`Ssym`, `S461`, `S783`, `K2369`, `S8754`, `S2648`, `Q3421`, `PTMUL`, `S571`) and 5 functional DMS (`DLG4`, `GRB2`, `ESTA`, `MYO`, `GB1`). `S669`- and `K3822`-specific proteins outside `S461` / `K2369` were deliberately not quarantined; `PTMUL-D ⊂ PTMUL`. `ProteinGym` and `Domainome` are too large to quarantine without collapsing the training pool | quantified after the fact (§7.1) |

**Threshold sensitivity.** Each row tightens one threshold over the kept scaffolds and counts the edges that would
then cross a boundary (`cross_real` = internal + external):

| threshold | cross_real | internal | external |
|---|---|---|---|
| identity > 0.35 | 192 | 46 | 146 |
| identity > 0.30 | 1,220 | 206 | 1,014 |
| identity > 0.25 | 6,453 | 560 | 5,893 |
| TM > 0.60 | 91 | 21 | 70 |
| TM > 0.50 | 667 | 145 | 522 |
| TM > 0.40 | 2,400 | 631 | 1,769 |
| E ≤ 1e-2 | 10 | 4 | 6 |
| E ≤ 1e-1 | 113 | 37 | 76 |

---

## 7. What leaked, model by model

### 7.1 ESM-MSR (trained on the Oct06 capped split)

On all 12 external sets, both capped Tsuboyama splits and all 7 functional DMS assays, every ESM-MSR checkpoint has
**zero** `M`, `m`, `P` and `H` overlap. The only codes are:
- `h` (twilight-zone, 30–40 % identity) on Tsuboyama: val `h.20`, test `h.16`;
- `h ≤ .002` on S669, S461, K3822 and K2369;
- `V` on capped val, which was used for checkpoint selection.

The retrained baselines inherit these codes (scope note above).

**ProteinGym (all 217 DMS; 2D MMseqs2 + 3D Foldseek):**
- **23 Stability assays are Tsuboyama libraries in capped train.** Across all 217 DMS that is 45,623 exact + 2
  constituent rows out of 2,465,767: 1.85 % of rows, 10.6 % of assays (`M.02`).
- **3 more Tsuboyama Stability assays (4,607 rows, `H.002`) are strong 3D homologs** caught by Foldseek:

  | assay | capped-train homolog | qTM |
  |---|---|---|
  | `AMFR_HUMAN_Tsuboyama_2023_4G3O` | `2MYX` | 0.742 |
  | `RAD_ANTMA_Tsuboyama_2023_2CJJ` | `2L9R` | 0.733 |
  | `VG08_BPP22_Tsuboyama_2023_2GP8` | `HHH_rd1_0606` | 0.830 |

- **2 assays (6,305 rows, `h.003`) are weak homologs:** `RNC_ECOLI_Weeks_2023` ↔ `2L33` (Foldseek E 8.1e-5) and
  `THIO_ECOLI_Tsuboyama_2023_2TRX` ↔ `2MI6`.
- **The 4 >1,800-residue assays** (`BRCA1`, `BRCA2`, `POLG_CXB3N`, `SCN5A`) have no overlap of any kind.
- **Effect of removing them** (ESM-MSR devel, 3-seed mean, `esm_msr_pg_clean.json`):

  | assay set | DMS mean | Stability | PG average |
  |---|---|---|---|
  | all 217 | 0.557 | 0.880 | 0.515 |
  | without the 23 `M` assays (194) | 0.516 | 0.871 | 0.513 |
  | also without the 3 `H` assays (191) | 0.510 | – | 0.513 |

  On the 213-DMS common subset (213 → 190 → 187 DMS), DMS mean goes 0.561 → 0.520 → 0.514 and PG average
  0.516 → 0.514 → 0.514.

**Domainome 1.0 (522 domains; 2D MMseqs2 + 3D Foldseek):**
- **2 domains match capped-train libraries at 100 % identity** over the full Megascale construct:
  - `O95155_PF04564_1229` (UBE4B = `3L1X`): 1,134 of 1,206 rows exact;
  - `Q9H161_PF00046_216` (ALX4 homeodomain = `2M0C`): 889 of 1,038 rows exact.

  Together: **2,023 exact rows (0.38 %, `M.004`)** plus **2,288 same-protein rows (0.43 %, `P.004`)** across 4
  domains (`3L1X`, `2M0C`, `2K28`, `2L7M`).
- **60 strong-homolog domains** (58,422 rows, 10.9 %, `H.11`): 40 by sequence plus 20 more by 3D structure.
- **12 weak-homolog domains** (12,498 rows, `h.02`).

Cell code: `M.004 P.004 H.11 h.02` (ᴴ).

**Domainome low-identity subset (`domainome_low_id`, BENCHMARKS column `Domainome (low-id, 458)`).** These are the
458 domains (471,652 rows) whose structure-homology class against capped train is neither `P` nor `H`. Step 07 writes
the subset as its own test set, so every model is audited on it:
- ESM-MSR: `h.03` (the 12 weak-homolog domains);
- EsmTherm: `M.002 P.04 H.30`;
- IFUM: `M?.005 P.04 H.29`;
- DDGemb: `M<.001 P.03 H.19`;
- GeoDDG-3D: `M.001 P.06 H.31`;
- ESM3ΔG: `P.006 H.14 h.56`;
- zero-shot models: `–`.

### 7.2 ESM-MSR-small (March reference, legacy hyperopt split)

**Inventory.** The 120 hyperopt-train libraries, with checkpoints selected on the 26 hyperopt-val libraries. The
inventory was built before this revision, but its homology was never computed (§11), so all 12 external cells
printed `–`.

**Codes on the external sets.** There is no `M`, `m` or `P` overlap. Strong-homolog cells:

| set | code |
|---|---|
| S2648 | `H.02 h.003` |
| S8754 | `H.01 h.001` |
| Q3421 | `H.01 h.002` |
| PTMUL | `H.05 h.009` |
| PTMUL-D | `H.03 h.01` |
| K2369 | `H.004 h.004` |
| K3822 | `H.003 h.003` |
| S669 | `H.003` |
| S783 | `H.001` |

The first five reach 0.5 % and carry ᴴ.

**Capped Tsuboyama splits.** On the capped splits, which are not reported for this model, its train libraries cover
49 % of capped-test rows at protein level.

### 7.3 The third-party Megascale-trained models

| model | capped test | capped val | evidence |
|---|---|---|---|
| EsmTherm | `M.51 P.34 H.04 h.03 V.06` (26,789 identical variants; 3,155 rows only in its validation split) | `M.44 P.46 H.03 h.02 V.05` | row-level: exact join against its 527,785 published sequences |
| IFUM | `M?.46 P.02 H.07 h.06 V?.47` (24,200 rows; 24,608 in its val/test block) | `M?.70 P.01 h.07 V?.28` | protein-level split + public Dataset2 rows |

**Mutant-background libraries.** Mapping `.pdb_<mut>` names (`2MKX.pdb_L11A`, `6NS8.pdb_L8K`, `1YEZ.pdb_V46S`) to
their parent scaffolds brings in the `2MKX` test rows and `6NS8` validation rows that both models trained on.

**Elsewhere** (fractions of rows):

| dataset | EsmTherm | IFUM |
|---|---|---|
| ProteinGym | `M.02 P.02` | not run |
| Domainome | `M.001 P.05 H.37` | `M?.004 P.04 H.36` |
| GB1 | `M<.001 P1` | `M?<.001 P1` |
| K2369 | `M.03 P.36` | `M?.21 P.17` |

The external sets are mostly `P`, plus the `M`/`M?` rows that Megascale also measured.

### 7.4 The literature-ΔΔG models (`12_literature_exact.py`, construct- and reverse-verified)

| model | training set | verified exact (`M`) and constituent (`m`) overlap | notes |
|---|---|---|---|
| DDGemb | `S2450`, forward + reverse | `S2648` 2,429 (`M.92`, + `D`); `Q3421` `M.63`; `Ssym` 408 / 684 (`M.60`, the reverse copies match Ssym's reverse rows); `S8754` `M.44`; `K3822` `M.38`; `K2369` `M.33`; `S571` `M.08`; `PTMUL` `m.48`; `PTMUL-D` `m.54`; **`S783`, `S669`, `S461`: 0 exact** | S783's 10 S2648-shared rows were among the 198 rows removed for S669 homology |
| DDMut | `S9028` ∪ `SM1218` + `SM444` | `S2648` `M.98`; `Ssym` `M.88`; `Q3421` `M.84`; `K2369` `M.74`; `K3822` `M.72`; `S8754` `M.69`; `S783` `M.61`; `PTMUL` `M.80 m.11`; `PTMUL-D` `M.95`; `S571` `M.21`; **`S669` `M.10 P.19`; `S461` `M.15 P.24`** | the S669 / S461 `P` comes from the multi-mutant model's training proteins |
| GeoDDG-3D | `S8754` (+ GeoFitness proteins) | `S8754` `M1 #`; `S2648` `M1`; `Q3421` `M.98`; `K2369` `M.91`; `K3822` `M.89`; `S783` `M.88`; `Ssym` `M.50` (all forward halves); `S571` `M.41`; `PTMUL` `m.73`; `PTMUL-D` `m.78`; `S669` `M.01 P.07`; **`S461` `P.11`** (GeoFitness pretraining protein) | no external set is clean for GeoDDG-3D |

**DMS assays:**
- **GB1 (`1PGA`):**
  - EsmTherm `M<.001 P1` and IFUM `M?<.001 P1`, from the Megascale `1PGA` library;
  - GeoDDG-3D `M<.001 m.41 P.59`: 1,538 `1PGA` singles in S8754.
- **MYO (human myoglobin):**
  - GeoDDG-3D `M<.001 P1` (`3RGK` in S8754);
  - DDGemb `H1` (sperm-whale myoglobin);
  - ESM3ΔG `h1`.
- **DLG4 abundance / binding:** GeoDDG-3D `P1`: the PSD95 domain's sequence matches a GeoFitness pretraining wild
  type.
- **ESTA:** ESM3ΔG `h1`.
- **Everything else** is `–` for every model.

### 7.5 ESM3ΔG, verified

1. **Exact (`15_esm3dg_exact.py`).** A SHA-1 join of every benchmark row's wild-type *and* mutant sequence against
   all 960,216 training sequences returns **0 matches** in all 3,382,799 rows.
2. **Homology (`14_esm3dg_homology.py`).** High-sensitivity MMseqs2 (`s = 7.5, e = 10, max-seqs 500`) is scored with
   the shared rule.
   - **Capped test** (`leak_audit_esm3dg_libraries.csv`):
     - 8 / 33 scaffolds have a strong homolog: 21,185 rows (40.7 %), from `1GYZ`, `3DKM`, `2MKX`, `2D1U`, `2KRU`,
       `2K52`, `2L09`, `1ZHC`;
     - 21 / 33 have a weak match: 26,204 rows (50.3 %);
     - 4 / 33 have none: 4,682 rows.
   - **Under the shared same-protein rule (≥ 90 % identity over ≥ 50 %),** a few proteins move from `H` to `P`:
     capped val `P.19`, S571 `P.01`, Domainome `P.005`, ProteinGym `P.003`. None of these is an identical sequence.

### 7.6 Rosetta `cartesian_ddg` (2019, `beta_nov16_cart`)

**Our runs.** FastRelax in cartesian mode with coordinate constraints, then `cartesian_ddg -ddg:legacy true
-ddg:iterations 3 -ddg:bbnbrs 1 -fa_max_dis 9 -ddg:optimize_proline true`. Both steps use
`-corrections:beta_nov16_cart`. These settings match the example command in Park et al. 2016's SI, if its
`-beta_cart` meant the then-current `beta_nov16_cart` (inferred).

**Rosetta has no per-variant training. What touched experimental stability data:**
1. **ref2015 / opt-nov15 weights (not used here).** No ΔΔG data was used in the fit. The Park SI says of the
   Kellogg ΔΔG test: "This test is only used for validation".
2. **The 20 `beta_nov16` reference energies (used here).** They were refit on an objective that included:
   - the Spearman correlation on "ProTherm DDGs following PMID 21287615" (the Kellogg 2011 set: 1,210 mutations,
     75 PDB);
   - the protease resistance of the Rocklin et al. 2017 designs.

   Source: `docs.rosettacommons.org/docs/latest/Updates-beta-nov16`; the Park SI does not cover `beta_nov16`.
3. **The REU → kcal/mol factor 1/2.94.** A least-squares fit (Park SI, "Mutational ddG calculation", Fig. S1),
   almost certainly on the Kellogg set. It changes no rank metric.
4. **Frenz et al. 2020.** Their protocol choices (selected on a balanced 767-mutation ProTherm set) concern the
   refactored, non-legacy mode, so at most `optimize_proline` carries over.

**Coding.** About 20 amino-acid-type offsets and one scale factor cannot encode any specific variant or protein, so
this is calibration on a benchmark: `V`, not `M` or `P`. Rosetta's numbers on these rows are still not fully held
out.

**Overlap with the tuning data** (`16_rosetta.py`, `leak_audit_rosetta.csv`). Rows count when they are in the
Kellogg set by the same PDB + chain + residue + wild type + mutant, or by the same UniProt position via SIFTS:

| set | Q3421 | S783 | Ssym | K3822 | S2648 | PTMUL (constituent) | K2369 | S8754 | S571 | PTMUL-D | S461 | S669 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `V` | .28 | .22 | .19 | .18 | .16 | .15 | .13 | .12 | .04 | .04 | .009 | .006 |

- **Tsuboyama:** 7 of the 33 capped-test scaffolds and 3 val scaffolds are Rocklin-2017 designs
  (`{HHH,EHEE,HEEH,EEHEE}_rd{1-4}_*`), giving `V.12` (test, 6,304 rows) and `V.07` (val).
- **ProteinGym:** `V.002` at protein level.
- Rosetta is reported only on Tsuboyama capped and ProteinGym in the October tables. The external-set fractions
  apply to the hyperopt-era reference table (`BENCHMARKS_OCT2026.md` §1.5).

### 7.7 Zero-shot models

None of these models had ΔΔG or fitness supervision, so they are coded `–`. Two design-level caveats are not coded:
- **ProSST:** its structure-vocabulary size (K = 2048) was chosen in ablations that included ProteinGym zero-shot
  scores.
- **σ = 0 control:** it keeps the released checkpoint's calibration head (scale and bias, fit on the capped split).
  Spearman is unaffected within singles, but a per-mutation-count bias could matter on sets that mix singles and
  multi-mutants.

---

## 8. What changes in the reported conclusions

**Bold eligibility.** Cells with `M`, `M?`, `m`, `P`, `H`, `D` or `#` at ≥ 0.5 % of rows lose bold. `h`, `V`, `V?`
and `–` remain eligible.
- **§1.1:** ESM-MSR (devel/baseline) or ME stage 1 lead 11 of 12 sets; `DDGemb` leads S669 (0.570 against 0.560
  for devel).
- **§1.3 Domainome:** `ProteinMPNN` (0.548) takes sole bold, because every capped-train model carries ᴴ (`H.11`).
  On the low-identity column (`h.03` only for ESM-MSR, so eligible), the σ = 0 control leads at 0.545, ahead of
  ProteinMPNN (0.544) and the ESM-MSR (devel) WT head (0.542).
- **§1.4 and §3 ProteinGym:** `ProSST` takes bold (0.514 / 0.653 / 0.528), because every supervised Tsuboyama model
  carries ᴹ (`M.02`).

**Clean external-set means.** These are taken over the cells with no excluding flag.

| model | clean sets (of 12) | clean mean |
|---|---|---|
| ESM-MSR (devel) | 12 | 0.652 |
| ESM-MSR (baseline) | 12 | 0.646 |
| ME stage 1 | 12 | 0.615 |
| ME stage 2 | 12 | 0.605 |
| SPURS (retrained) | 12 | 0.588 |
| ThermoMPNN + D | 12 | 0.586 |
| ThermoMPNN | 12 | 0.585 |
| SPURS multi (retrained) | 12 | 0.556 |
| ProteinMPNN | 12 | 0.464 |
| ProSST | 12 | 0.423 |
| σ = 0 control | 12 | 0.378 |
| ESM3ΔG | 5 (S783, S669, S461, Ssym, Q3421) | – |
| DDGemb | 2 (S669, S461) | – |
| EsmTherm | 1 (Ssym) | – |
| IFUM | 1 (Ssym) | – |
| GeoDDG-3D | **0** (S461 now carries `P.11` from GeoFitness pretraining) | – |
| ESM-MSR-small (March reference) | 7 (no longer 12) | – |

**`BENCHMARKS_OCT2026.md` follows this revision.** Its leak-code matrices, superscripts, §2 row labels and §6
summary were regenerated from `leak_audit_matrix.csv` on 2026-11-12. The per-table CSVs in
`analysis/reference/benchmark_tables/` and the figures in `analysis/results/10_leakage_and_reference/` were
regenerated with them. No bolded number changed: every cell whose superscript changed was already excluded or not
leading.

---

## 9. Assumptions and inherent limits

1. **Library-level upper bounds.** Private row filters (censoring, early stopping) can only remove rows, so `M` / `m`
   counts for Megascale inventories err upward.
2. **Protein-only disclosures.**
   - IFUM's `M?` / `V?` are ceilings from its domain split.
   - GeoFitness is matched by wild-type sequence only (`P`).
3. **ESM3ΔG augmented checkpoints.** These are compared with the full 960,216-sequence index, which is a superset of
   their ~456k subset, so `H` / `h` are upper bounds. The 2,357 MGnify validation sequences are not searched; this
   is negligible.
4. **3D Foldseek scope.** Structural homology is computed for Domainome and ProteinGym against the capped-train and
   literature-inventory structures. Every other inventory and test set uses sequence homology only (MMseqs2,
   `s = 7.5`).
5. **Single-chain monomers.** Splitting and audit both extract single chains, ignoring oligomeric partners.
6. **Row fractions, not assay fractions.** Cell fractions divide leaked rows by total dataset rows. For example, the
   23 leaked ProteinGym DMS are 1.9 % of rows (`M.02`) but 10.6 % of assays.
7. **Descriptive scope.** Leak codes record training exposure, not causal score inflation.

---

## 10. Reproducibility

**Committed artifacts** (`docs/benchmarks/`, mirrored in `analysis/reference/` where analysis code reads them):

| file | contents |
|---|---|
| `leak_audit.sqlite` | every table below in one database (`inventories`, `cells`, `pg_per_dms`, `homology`, `structure_homology`, `code_exact`, `rosetta`, `esm3dg_*`, `geofitness_proteins`) |
| `leak_audit_matrix.csv` | every audited `(model, dataset)` cell: counts, `code`, `sup` |
| `leak_audit_inventories.csv` | inventory catalogue with provenance and evidence tier |
| `leak_audit_pg_per_dms.csv` | ESM-MSR capped-train inventory × 217 ProteinGym assays (`exact`, `const`, `sameprot`, `H`, `h`) |
| `leak_audit_homology.csv` | per inventory × dataset `P` / `H` / `h` row counts |
| `leak_audit_structure_homology.csv` | Foldseek 3D hits on Domainome and ProteinGym |
| `leak_audit_code_exact.csv` | verified exact / constituent counts for `S2450`, `S9028`, `SM1662`, the DDMut union, and `S8754` |
| `leak_audit_rosetta.csv` | rows in Rosetta's `beta_nov16` tuning data per dataset |
| `split_homology_audit.json` | split validation and threshold sensitivity (§6) |
| `leak_audit_esm3dg_libraries.csv`, `esm3dg_exact.csv`, `esm3dg_pg_exact_per_dms.csv` | ESM3ΔG per-scaffold homology and exact-sequence joins |

**Run:**

```bash
python analysis/leak_audit/run_all.py
```

`--from-step NN` resumes from a step. `THREADS` caps MMseqs2 and Foldseek. The full run takes about 4 minutes on
8 cores.

**Inputs:**
- **Tracked in `analysis/leak_audit/data/`:**
  - `S2450.*`, `S9028.csv`, `DD_SM1218.csv`, `DD_SM444.csv`;
  - `ifum_train_proteins.csv`, `esmtherm_split.csv`, `esmtherm_train_proteins.csv`, `kellogg2011_1210.csv`.
- **In `$LEAK_DATA_DIR/raw`, all public:**
  - `GS_S8754.csv`, `mgnify_index.csv`;
  - `esmtherm_pcbi.1012248.s006.csv`, `geofitness_dms_wt.fasta`;
  - `pdb_chain_uniprot.tsv.gz` (SIFTS);
  - the Domainome Supplementary Table 5.
- **Other directories:**
  - the Tsuboyama CSV under `$SOFTWARE_DIR/esm-msr/data/tsuboyama/`;
  - ProteinGym under `$LEAK_ROOT/ProteinGym`;
  - Domainome structures under `data/domainome1/`.

In a git worktree, untracked `data/` inputs are read from the main checkout (`lib.repo_data`). Step 07 stops with an
error if any Domainome or ProteinGym structure is missing.

| step | script | role |
|---|---|---|
| 01 | `01_esmtherm_inventory.py` | EsmTherm wild-type split → `esmtherm_train_proteins.csv` |
| 02 | `02_ifum_inventory.py` | parse IFUM `dataset_splits.pdf` → `ifum_train_proteins.csv` |
| 03 | `03_megascale.py` | Megascale library and variant tables (`.pdb_<mut>` normalised); capped-split scaffold sets |
| 04 | `04_testsets.py` | normalise every benchmark row to `(pid, seq, muts)` and validate coordinates |
| 05 | `05_inventory_proteins.py` | wild-type sequences and PDB residue maps for `S2450`, `S9028`, `SM1662`, `S8754` |
| 06 | `06_sequence_homology.py` | unified protein table + hash-cached MMseqs2 all-vs-all (`s = 7.5`) |
| 07 | `07_structure_homology.py` | exhaustive Foldseek of Domainome and ProteinGym structures; writes the `domainome_low_id` test set |
| 08 | `08_families.py` | ≥ 95 % identity / ≥ 90 % coverage families, aligned to representatives |
| 09 | `09_split_audit.py` | audit `homology_pairs.csv` over 370 libraries + 776 external nodes |
| 10 | `10_overlap_engine.py` | `M` / `m` / `P` for the ESM-MSR inventories; exact-sequence `M` / `V` for EsmTherm and IFUM |
| 11 | `11_homology_classes.py` | `P` / `H` / `h` per inventory × dataset (shared rule; GeoFitness proteins for GeoDDG-3D) |
| 12 | `12_literature_exact.py` | construct- and reverse-aligned `M` / `m` for DDGemb, DDMut, GeoDDG-3D |
| 13 | `13_mgnify_search.py` | MMseqs2 of benchmark proteins against the 960k MGnify training sequences |
| 14 | `14_esm3dg_homology.py` | ESM3ΔG `P` / `H` / `h` and the 33-scaffold table |
| 15 | `15_esm3dg_exact.py` | exact SHA-1 join of every WT and mutant sequence against the MGnify index |
| 16 | `16_rosetta.py` | rows in Rosetta's `beta_nov16` tuning data (Kellogg 2011, Rocklin 2017) |
| 17 | `17_proteingym.py` | per-DMS ProteinGym overlap and the ESM-MSR clean aggregates |
| 18 | `18_matrix.py` | compose the `(model, dataset)` leak codes |
| 19 | `19_export.py` | write `docs/benchmarks/` CSVs, the SQLite database and `analysis/reference/` mirrors |

**Tools:** MMseqs2 13-45111, Foldseek 463739e, Biopython 1.85, pandas 2.3.0, networkx 3.3, SQLite 3.

---

## 11. Revision log (2026-11-12)

**Inventories corrected:**
- **EsmTherm:** `M?` (estimated as 0.8 × `P`) is replaced by a verified exact-sequence join against its published
  S2 Spreadsheet. Its validation-only rows are `V`.
- **IFUM:** `M?` is now the exact-sequence ceiling from its domain split; it was `M? = P`, which overstated GB1 and
  Domainome. `V?` is added for its val/test block.
- **DDGemb:** reverse variants added; Ssym `M.30` → `M.60`. The S2648 count is the verified 2,429; it had been
  overwritten with the paper's 2,450.
- **DDMut:** the multi-mutant inventory is `SM1218 + SM444`, which the deployed model was trained on (previously
  `SM1242`). DDMut is evaluated row by row over the union with `S9028`: PTMUL `M.59 m.05` → `M.80 m.11`, S669 /
  S461 gain `P`.
- **GeoDDG-3D:** GeoFitness DMS pretraining proteins added at protein level. DLG4 `–` → `P1`, S461 `H.004` → `P.11`.
- **ESM-MSR-small:** its homology was never computed, so all its cells printed `–` (§7.2).

**Coding fixed:**
- `V` was set on capped val for every supervised model, including third-party models that never saw our validation
  split. Now only ESM-MSR carries it, plus each third-party model's own validation rows.
- ESM3ΔG `P` now uses the shared ≥ 90 % identity rule; it was exact sequence only.
- Hyperopt-split cells are `n.a.` instead of `–`.

**Added:**
- Rosetta, coded `V` (§7.6).
- The 458-domain low-identity Domainome subset is now derived by the pipeline (step 07), not added to the matrix by
  hand. With structure homology applied to it, the pipeline reproduces the hand-added codes except where this
  revision changed an inventory.

**Pipeline:**
- **Removed:** the retrained-baseline inventories, the upstream SPURS / ThermoMPNN inventories and the unused
  `esm3dg_released` inventory.
- **Restructured:** the steps are renumbered in run order, the homology rule is shared, and the matrix no longer
  hard-codes rho values.
- **Fixed:** step 07 crashed on `REPO_ROOT` and silently ran without Domainome structures inside a worktree.
- **Checked:** the mutation-mapping ±3-residue fallback was removed; no ESM-MSR count changed.
- **Corrected numbers:** 370 (not 412) libraries in the split audit; 3,382,799 (not 3.53 M) audited rows.
