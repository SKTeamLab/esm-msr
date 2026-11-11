# Data-leakage audit of the October 2026 benchmark tables

**Scope.** Every model and dataset reported in [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md), on the Oct06 capped
split (`data/splits_oct06_capped.pkl`). Companion to §6 of [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) and
[`SPLITTING_AND_HOMOLOGY_AUDIT.md`](SPLITTING_AND_HOMOLOGY_AUDIT.md); this document records the methodology,
reconstructed training inventories, split-tooling validation, model-by-model overlap findings, assumptions, and
single-command reproducibility recipe (`python analysis/leak_audit/run_all.py`).

**One-line result.** The capped split is sound for every model trained on it — **ESM-MSR**, **MutateEverything**
(stages 1 and 2), **ThermoMPNN / ThermoMPNN-D**, and **retrained SPURS / SPURS multi** have **zero** identical-mutation
and **zero** same-protein overlap with all 12 external thermostability sets, both Tsuboyama capped splits, and all 7
functional DMS assays — whereas the broader benchmark contains substantial third-party and unquarantined-suite overlap:
23 of the 217 ProteinGym assays (plus 3 strong 3D structural-homolog assays) and 2 of the 522 Domainome domains (plus
60 strong sequence/3D homolog domains) sit inside or homologous to the capped training split; the third-party
Megascale-trained models (`EsmTherm`, `IFUM`) trained on 21–27 of our 33 Tsuboyama test scaffolds (49–86 % of rows);
the literature-ΔΔG models (`DDGemb`, `DDMut`, `GeoDDG-3D`) trained on 30–100 % of most classical external sets at the
mutation or constituent level; and `ESM3ΔG` has **0** exact sequence matches anywhere in the benchmark (with 40.7 %
strong and 50.3 % weak sequence homologs on Tsuboyama test from its MGnify Stability fine-tune).

---

## 1. Method

The audit runs end-to-end via `analysis/leak_audit/run_all.py` (`00_build_inventories.py` → `18_db.py`, §9).

1. **Normalise the test side (`02_testsets.py`).** Every reported row is reduced to `(protein sequence, set of (WT, position, mutant))`.
   The audit evaluates all 2,465,767 ProteinGym rows across all 217 DMS assays (including the 4 >1,800-residue proteins excluded
   from the 213-assay common multi-model intersection), 534,385 Domainome
   rows (522 domains), 149,360 GB1 rows, 87,544 Tsuboyama capped val+test rows (each keyed by its exact library code,
   including `.pdb_<mut>` stabilizing-background libraries), 25,632 rows over the 12 external sets, and 269,471 rows
   over the seven DMS assays. Mutations are validated in 1-based sequence coordinates (`0` mismatches on Tsuboyama
   val/test and ProteinGym; `1,779` construct/numbering edge-case rows [0.33 %] dropped and recorded in Domainome).
2. **Build same-protein families and 2D/3D homology maps (`05b`, `05`, `05c`, `05d`, `06`).** All test proteins plus
   all inventory proteins are searched all-vs-all with MMseqs2 (`easy-search -s 7.5 -e 10 --max-seqs 20000`), computing
   direction-invariant shorter-sequence coverage `min(alnlen, min(len_p, len_q)) / min(len_p, len_q)` with explicit
   length-ratio guard (`min_len / max_len ≥ 0.80`) so short domains cannot bridge unrelated multidomain proteins while
   construct truncations (such as 48-aa `2M0C` inside 56-aa Domainome `Q9H161_PF00046_216`) cluster cleanly at ≥95 %
   identity and ≥90 % shorter-sequence coverage. Each family member is globally aligned (Biopython, BLOSUM62) to its
   representative so mutations can be matched across alternate PDB entries, chain offsets, and construct truncations.
   For **Domainome** (522 domain PDB structures from the Domainome 1.0 release; Beltran et al., *Nature* 2025, Zenodo record `14356805`, DOI `10.5281/zenodo.11043642`) and **ProteinGym** (217 AlphaFold2 structures), exhaustive Foldseek 3D structural search (`05d_structure_homology.py`) is
   additionally run against all 172 capped-train scaffold structures.
3. **Reconstruct each model's training inventory (`00`, `00b`, `00c`, `01`, `04`, `05c`).** For in-house and retrained
   models (`ESM-MSR`, `MutateEverything` st1/st2, `ThermoMPNN`, `ThermoMPNN-D`, `SPURS (retrained)`, `SPURS multi (retrained)`),
   inventories are constructed directly from `data/splits_oct06_capped.pkl` and the MegaScale table (172 scaffolds for
   all single/all-variant models; 134 measured-double or Rosetta-augmented scaffolds for `ThermoMPNN-D`). For third-party
   models (`EsmTherm`, `IFUM`, `DDGemb`, `DDMut`, `GeoDDG-3D`, `ESM3ΔG`), inventories are built from their published
   split/training files (§3). Zero-shot / unsupervised models without mutation-level ΔΔG supervision (`ProSST`,
   `ProteinMPNN`, `ESM3`, `Rosetta`) carry `–`.
4. **Classify every test row (`10`, `11`, `12`, `14`, `15`, `19`, `13`, `18`).** Sequence and structural homology hits
   are partitioned by query-coverage tier across all qualifying hits per query protein (`cov_test ≥ 0.50` tested first for
   `H`, then `cov_test ≥ 0.30` for `h`) so short local motif hits cannot mask full-domain homologs. Exact (`M`) and constituent (`m`) overlaps for
   literature-ΔΔG models (`DDGemb`, `DDMut`, `GeoDDG-3D`) are verified by construct/family sequence alignment including
   reverse mutant structures (`mt_pos_wt`).

---

## 2. Notation

| code | meaning | test |
|---|---|---|
| `M` | identical variant (same protein family, identical mutation set, including reverse structures) in training | verified against inventory |
| `m` | every mutation of the row appears in training in the same protein, but not this exact combination | verified |
| `P` | same protein in training, no mutation match | ≥95 % identity over ≥90 % shorter-seq coverage (`len_ratio ≥ 0.80`), or ≥90 %/≥50 % |
| `H` | strong homolog in training | MMseqs2 E ≤ 1e-3 over ≥50 % of test protein, or Foldseek E ≤ 1e-3 / qTM > 0.70 over ≥50 % (Domainome & PG) |
| `h` | weak/ambiguous homology | MMseqs2 E ≤ 1e-2 or ≥30 % identity over ≥30 % of test protein, or Foldseek E ≤ 1e-2 over ≥30 % (Domainome & PG) |
| `D` | documented training on the same source dataset, rows not public | paper/code |
| `–` | no supervised ΔΔG training overlap (includes physics-based and unsupervised pretraining-only models) | verified / documented |
| `U` | training data not disclosed | paper |
| `V` | dataset used for validation or model selection | split file |
| `#` | not a test result: the dataset is the training set | file identity |
| `?` | probable but unverified (e.g. `M?`, `P?`) | inference |

A cell lists every class present with its fraction of the dataset's reported rows, highest-severity first:
`M.59 m.05 P.32 H.02`. Numeric tables in `BENCHMARKS_OCT2026.md` print only the worst class reaching at least `0.5 %`
of rows (`v / n ≥ 0.005`) as a superscript; full codes appear in each companion matrix and in `leak_audit_matrix.csv`.

---

## 3. Inventories

| model row(s) | training inventory | n | source | evidence |
|---|---|---|---|---|
| ESM-MSR (released/baseline), ESM-MSR (devel), WT/MT heads, σ sweep | the 172 capped-train scaffolds, all substitutions | 229,721 rows / 172 scaffolds (175 library codes incl. 3 `.pdb_<mut>`) | `data/splits_oct06_capped.pkl` + MegaScale CSV | verified |
| MutateEverything stage 1 / stage 1+ | the 172 capped-train scaffolds, valid singles | 154,358 singles / 172 scaffolds | capped split; ME recipe | verified |
| MutateEverything stage 2 / stage 2+ | the same 172 scaffolds, singles + measured doubles | 207,546 rows / 172 scaffolds | capped split; ME recipe | verified |
| ThermoMPNN (additive) | the same 172 scaffolds, valid singles | 154,358 singles / 172 scaffolds | capped split; `train_single.yaml` | verified |
| ThermoMPNN-D | 134 of the 172 capped-train scaffolds (52 with measured doubles + 82 Rosetta-augmented natural scaffolds) | 53,188 doubles + 118,963 aug doubles / 134 scaffolds | capped split; `05c_tmpnnd_inventory.py` | verified |
| SPURS (retrained) | the 172 capped-train scaffolds, valid singles | 154,358 singles / 172 scaffolds | capped split (3 seeds) | verified |
| SPURS multi (retrained) | the same 172 scaffolds, singles (via init) + measured doubles | 207,546 rows / 172 scaffolds | capped split (3 seeds) | verified |
| EsmTherm | `K50_dG_Dataset1_Dataset2`; 414 wildtypes contribute train mutants under a per-row 80/10/10 split | 414 train domains | `esmtherm_train_proteins.csv` | protein-level (rows not public) |
| IFUM | Megascale Dataset2; 322 train libraries, 86 in unseparated val/test | 322 train libraries | `ifum_train_proteins.csv` (`00b_extract_ifum_split.py`) | protein-level |
| DDGemb | S2450 = S2648 − 18 sequences / 198 mutations >25 % identity to S669 | 2,450 rows / 115 UniProt | `S2450.tsv` + `S2450.fasta` | verified (sequence-aligned) |
| DDMut | S9028 = S2648 + FireProtDB (4,514 fwd + 4,514 rev); multi-mutant branch SM1242 | 9,028 single + 1,242 multi rows | `S9028.csv`, `DD_SM1242.csv` | verified (forward + reverse) |
| GeoDDG-3D | S8754 (its own training set) | 8,754 rows / 301 proteins | `GeoStab` `S8754.csv` | verified (construct-aligned) |
| ESM3ΔG | MGnify Stability fine-tune, **not** Megascale | 960,216 sequences | Zenodo 19411306 `mgnify_training_index.csv` | verified (0 exact WT or mutant sequences) |
| ProSST | AlphaFoldDB 18.8 M structures, masked-LM; no ΔΔG fine-tune (`–`) | — | ProSST paper | documented only |
| ProteinMPNN | PDB assemblies deposited before 2021-08-02, sequence recovery (`–`) | 25,361 clusters | ProteinMPNN paper | documented only |
| ESM3 small/medium/large, σ = 0 control | UniRef/MGnify/JGI + PDB (≤2020-05-01) + AFDB v4; no ΔΔG fine-tune (`–`) | 2.78 B proteins | ESM3 paper | documented only |
| Rosetta Cartesian ΔΔG | physics-based (`–`) | — | physics | n/a |

---

## 4. The datasets overlap each other

Measured on unique `(PDB code, sequence, mutation)` identifiers prior to cross-PDB family alignment:

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

When construct/family sequence alignment and reverse structures are also resolved (`12_code_exact.py`, §6.3), cross-PDB
exact matches rise even further (e.g. `S8754` covers `100 %` of `S2648`, `97.9 %` of `Q3421`, `91.2 %` of `K2369`,
`89.1 %` of `K3822`, `50.0 %` of `Ssym`, and `8/669` rows of `S669`; `S9028` covers `88.3 %` of `Ssym` and `67` rows
each of `S669` and `S461`). `K2369 ⊂ K3822`, `S461 ⊂ S669`, and `PTMUL-D ⊂ PTMUL` hold exactly on structure/variant keys.

---

## 5. Does the split's homology tooling hold up?

The split was produced by `preprocessing/split_tsuboyama.py` + `preprocessing/homology.py` (`data/splits_oct06_capped.pkl`
and `data/splits_oct06_capped_library_assignment.csv`). Its undirected homology edge rule is:
**MMseqs2 E ≤ 1e-3 or Foldseek E ≤ 1e-3 or BLOSUM62 local identity > 0.40 or TM-score > 0.70 (shorter-chain
normalised)**. Re-evaluating the 408,330-pair table (`homology_pairs.csv`) across all 412 MegaScale scaffolds and all
776 external quarantine nodes (`771` structures from 9 classical ΔΔG sets + `5` functional DMS proteins) via
`09_homology_audit.py` yields (`docs/benchmarks/split_homology_audit.json`):

| test | result | verdict |
|---|---|---|
| rule is enforced | 25,027 default edges; **0** cross any kept boundary (`0` internal `train/val/test`, `0` `kept ↔ external`); all **880** cross-group edges involve the 142 excluded scaffolds | pass |
| quarantine accounting | of 142 excluded scaffolds: **132** have a direct homology edge to an external target (560 homology edges; 91 at identity > 0.90, 149 at TM > 0.90), **8** more belong to external-touching connected components via transitive homology, and **2** (`HEEH_KT_rd6_0872`, `HHH_rd4_0124`) were removed by `prune_bridges(max_component_size = 50)` | pass |
| Foldseek adds value | 550 structure-only edges (no MMseqs2 hit, identity ≤ 0.40); **0** cross a kept boundary | pass |
| threshold sensitivity (`cross_real = internal + external`) | tightening one threshold over kept scaffolds: at identity > 0.35: **192** (`46` internal + `146` external); > 0.30: **1,220** (`206` + `1,014`); > 0.25: **6,453** (`560` + `5,893`); at TM > 0.60: **91** (`21` + `70`); > 0.50: **667** (`145` + `522`); > 0.40: **2,400** (`631` + `1,769`); at E ≤ 1e-2: **10** (`4` + `6`); ≤ 1e-1: **113** (`37` + `76`) | E-value + 40 % / 0.70 TM caps jointly guard the split |
| twilight zone (`h`) | 8,114 capped-test rows (15.6 %) and 7,210 val rows (20.3 %) have a 30–40 % identity match to a train scaffold that the rule did not exclude | flagged `h`, not `H` |
| non-transitive quarantine | 134 train, 7 val, and 17 test scaffolds sit 2 hops from an external protein via an excluded bridge; all direct pairs sit below the hard caps (max identity 0.395, max TM 0.6998) | no strong pair missed |
| intentional quarantine scope | the split quarantined the 9 classical ΔΔG sets (`Ssym`, `S461`, `S783`, `K2369`, `S8754`, `S2648`, `Q3421`, `PTMUL`, `S571`) and 5 functional DMS (`DLG4`, `GRB2`, `ESTA`, `MYO`, `GB1`); `S669`- and `K3822`-specific proteins outside `S461`/`K2369` were deliberately not quarantined, `PTMUL-D ⊂ PTMUL`, and `ProteinGym` and `Domainome` are too large to quarantine without collapsing the training pool | quantified post-hoc (§6.1, §6.5) |

---

## 6. What leaked, model by model

### 6.1 The models trained on the Oct06 capped split

**ESM-MSR** (baseline/released and devel, all heads and σ values), **MutateEverything** (stage 1 and stage 2),
**ThermoMPNN**, **ThermoMPNN-D**, and **retrained SPURS / SPURS multi** share **zero** identical-mutation (`M`),
**zero** constituent (`m`), **zero** same-protein (`P`), and **zero** strong-homolog (`H`) overlap with all 12 external
sets, both Tsuboyama capped splits, and all 7 functional DMS assays.

- **Identical protein sets across retrained models.** ESM-MSR, ME stage 1/2, ThermoMPNN, and retrained SPURS / SPURS
  multi all train on the exact same 172 scaffolds, so their `P`/`H`/`h` fractions are identical across all 25 benchmark
  columns, while `M`/`m` track singles-only (`ME st1`, `ThermoMPNN`, `SPURS`) vs singles+doubles (`ESM-MSR`, `ME st2`,
  `SPURS multi`). `ThermoMPNN-D` sees 134 of the 172 scaffolds (the 52 scaffolds with measured doubles plus 82 natural
  scaffolds with Rosetta single-mutant PDBs; the 36 `{EA,GG,XX}_run*` designs plus `2M7O` and `5GU9` have neither),
  lowering its weak-homology fractions on Tsuboyama (`h.07` val, `h.12` test vs `h.20` and `h.16`).
- **ProteinGym (all 217 DMS; 2D MMseqs2 + 3D Foldseek).**
  - **23 Stability DMS assays** (`45,623` exact + `2` constituent rows out of `2,465,767` across all 217 DMS; `1.85 %` of rows,
    `10.6 %` of assays) are Tsuboyama libraries in `capped train` (`M.02` for ESM-MSR / ME st2 / SPURS multi; `M.010 m.009`
    for ME st1 / ThermoMPNN / SPURS).
  - **3 additional Tsuboyama Stability DMS assays** (`4,607` rows, `H.002`) are strong 3D structural homologs of
    capped-train scaffolds caught by Foldseek (`AMFR_HUMAN_Tsuboyama_2023_4G3O` ↔ `2MYX` qTM `0.742`;
    `RAD_ANTMA_Tsuboyama_2023_2CJJ` ↔ `2L9R` qTM `0.733`; `VG08_BPP22_Tsuboyama_2023_2GP8` ↔ `HHH_rd1_0606` qTM `0.830`),
    and **2 DMS assays** (`6,305` rows, `h.003`) are weak homologs (`RNC_ECOLI_Weeks_2023` ↔ `2L33` Foldseek E `8.1e-5`;
    `THIO_ECOLI_Tsuboyama_2023_2TRX` ↔ `2MI6`). None of the 4 >1,800-residue assays (`BRCA1`, `BRCA2`, `POLG_CXB3N`, `SCN5A`)
    have any `M`, `P`, `H`, or `h` overlap with `capped train`.
  - Across all 217 assays (ESM-MSR devel 3-seed mean), removing the 23 `M` assays (194 DMS) moves DMS mean `0.557 → 0.516`,
    Stability `0.880 → 0.871`, and PG average `0.515 → 0.513`; also removing the 3 Foldseek `H` assays (191 DMS) puts
    DMS mean at `0.510` and PG average at `0.513` (on the 213-DMS common subset: `213 → 190 → 187` DMS, DMS mean `0.561 → 0.520 → 0.514`, PG average `0.516 → 0.514 → 0.514`).
- **Domainome 1.0 (522 domains, Beltran et al., 2025; 2D MMseqs2 + 3D Foldseek).**
  - **2 domains** match capped-train libraries at 100 % sequence identity over the full MegaScale construct:
    `O95155_PF04564_1229` (UBE4B = `3L1X`, `1,134 / 1,206` rows exact) and `Q9H161_PF00046_216` (ALX4 homeodomain =
    `2M0C`, `889 / 1,038` rows exact), totalling **2,023 exact rows (`0.38 %`, `M.004`)** plus **2,288 same-protein
    rows (`0.43 %`, `P.004`)** across 4 domains (`3L1X`, `2M0C`, `2K28`, `2L7M`).
  - Combining Foldseek 3D structural search with MMseqs2 sequence search identifies **60 strong-homolog domains
    (`58,422` rows, `10.9 %`, `H.11`; 40 by sequence plus 20 additional 3D structural homologs)** and **12 weak-homolog domains (`12,498` rows, `2.3 %`, `h.02`)**,
    giving cell code `M.004 P.004 H.11 h.02` (`ᴴ` at the 0.5 % numeric-table threshold).

### 6.2 The third-party Megascale-trained models (and historical upstream splits)

| model | trained on our 33 test scaffolds | exact | same protein | verdict |
|---|---|---|---|---|
| EsmTherm | 27 of 33 test scaffolds (incl. `2MKX` mutant-background) contribute train mutants (per-row 80/10/10) | rows not public (`M?.68` ≈ 0.8 × `P`) | 44,561 (`85.6 %`, `P.86`) | leaked at protein/mutant level |
| IFUM | 21 of 33 test scaffolds (incl. `2MKX`) in train | rows not public (`M?.49` = `P`) | 25,296 (`48.6 %`, `P.49`) | leaked at protein/mutant level |
| ESM3ΔG | **0 exact rows anywhere** | 0 | 0 | held-out; 8/33 strong (`40.7 %`, `21,185` rows) + 21/33 weak (`50.3 %`, `26,204` rows) homologs |
| *(historical)* upstream SPURS single / multi | 27 of 33 test scaffolds (ThermoMPNN 8:1:1 split + 25 % blacklist) | 22,588 singles (`43.4 %`) + 6,753 `m` / 6,401 doubles | 15,166 / 23,549 | why SPURS was retrained on Oct06 capped (§1.6 of `BENCHMARKS_OCT2026.md`) |

Mapping `.pdb_<mut>` mutant-background library names (`2MKX.pdb_L11A`, `6NS8.pdb_L8K`, `1YEZ.pdb_V46S`) to their parent
wild-type scaffolds includes the 1,778 `2MKX` test rows (`3.4 %`) and 1,317 `6NS8` validation rows (`3.7 %`) trained by
both `EsmTherm` and `IFUM`, yielding capped val/test `P` fractions of `90.6 % / 85.6 %` for `EsmTherm` and `71.5 % / 48.6 %` for `IFUM`.

### 6.3 The literature-ΔΔG models (`12_code_exact.py`, construct- & reverse-verified)

| model | training set | verified exact (`M`) and constituent (`m`) overlap | notes |
|---|---|---|---|
| DDGemb | `S2450` (`S2450.tsv` + `S2450.fasta`) | `S2648`: 2,449/2,648 (`92.5 %`); `Q3421`: 2,168 (`63.4 %`); `S8754`: 3,816 (`43.6 %`); `K3822`: 1,465 (`38.3 %`); `K2369`: 782 (`33.0 %`); `Ssym`: 203 (`29.7 %`); `S571`: 45 (`7.9 %`); `PTMUL`: `m.48` (435/914); `PTMUL-D`: `m.54` (289/536); **`S783`, `S669`, `S461`: 0 exact** | verified directly against `S2450.fasta` (`S783`'s 10 `S2648`-shared rows were among the 198 `S669`-homolog rows removed in `S2450`) |
| DDMut | `S9028` + `SM1242` | `S2648`: 2,599 (`98.2 %`); `Ssym`: 604/684 (`88.3 %`, fwd + rev); `Q3421`: 2,858 (`83.5 %`); `K2369`: 1,761 (`74.3 %`); `K3822`: 2,758 (`72.2 %`); `PTMUL-D`: 381 (`71.1 %`); `S8754`: 6,081 (`69.5 %`); `S783`: 477 (`60.9 %`); `PTMUL`: 538 exact + 44 `m` (`M.59 m.05`); `S571`: 119 (`20.8 %`); **`S669`: 67 (`10.0 %`); `S461`: 67 (`14.5 %`)** | `S9028` includes FireProtDB proteins overlapping `S669`/`S461` and reverse mutant structures overlapping `Ssym` |
| GeoDDG-3D | `S8754` | `S8754`: 8,754/8,754 (`100 %`, `#`); `S2648`: 2,648/2,648 (`100 %`); `Q3421`: 3,350 (`97.9 %`); `K2369`: 2,160 (`91.2 %`); `K3822`: 3,407 (`89.1 %`); `S783`: 692 (`88.4 %`); `Ssym`: 342 (`50.0 %`, all forward halves); `S571`: 234 (`41.0 %`); `PTMUL`: `m.73` (670/914); `PTMUL-D`: `m.78` (416/536); **`S669`: 8 (`1.2 %`); `S461`: 0** | construct-aligned matching recovers cross-PDB accessions; only `S461` has 0 exact mutations across all full-coverage models |

### 6.4 The DMS assays

All seven functional DMS assays (`DLG4` abundance/binding, `GRB2` abundance/binding, `MYO`, `ESTA`, `GB1`) have **zero**
overlap (`–`) for ESM-MSR, ME, ThermoMPNN(-D), retrained SPURS, and all zero-shot models:

- **GB1 (`1PGA`)**: `EsmTherm` (`P1 M?.80`) and `IFUM` (`P1 M?1`) trained on the MegaScale `1PGA` library; `GeoDDG-3D`
  (`M<.001 m.41 P.59`) trained on 1,538 `1PGA` singles in `S8754` (61 exact + 61,880 constituent rows). `ESM3ΔG` has zero
  GB1 hits (`–`).
- **MYO (human myoglobin)**: `GeoDDG-3D` (`M<.001 P1`) trained on human myoglobin (`3RGK` in `S8754`, 3 exact single
  mutants + 5,902 same-protein rows); `DDGemb` (`H1`) trained on sperm whale myoglobin in `S2450`; `ESM3ΔG` (`h1`) has a
  weak MGnify sequence homolog.
- **ESTA**: `ESM3ΔG` has weak MGnify similarity (`h1`); all other models are `–`.

### 6.5 ProteinGym, per assay

`docs/benchmarks/leak_audit_pg_per_dms.csv` records exact (`M`), constituent (`m`), same-protein (`P`), strong-homolog
(`H` / `strong_homolog` in SQLite, combining MMseqs2 + Foldseek 3D), and weak-homolog (`h` / `weak_homolog`) counts across
all 217 ProteinGym DMS assays:

| inventory | assays with `M`/`m` | assays with `H` (2D+3D) | assays with `h` (2D+3D) | rows (217 DMS) |
|---|---|---|---|---|
| ESM-MSR / ME st2 / SPURS multi (retrained) | 23 | 3 | 2 | 45,623 `M` + 2 `m` + 4,607 `H` + 6,305 `h` |
| ME st1 / ThermoMPNN / SPURS (retrained) | 23 | 3 | 2 | 24,252 `M` + 21,331 `m` + 42 `P` + 4,607 `H` + 6,305 `h` |
| ThermoMPNN-D (doubles, 134 scaffolds) | 11 | 3 | 1 | 21,371 `M` + 1,705 `m` + 22,549 `P` + 4,607 `H` + 4,277 `h` |
| EsmTherm | — (protein-level) | — | — | 100,873 `P` + 9,663 `H` + 1,813 `h` |
| IFUM | — (protein-level) | — | — | 95,854 `P` + 8,425 `H` + 2,860 `h` |
| ESM3ΔG | 0 | — | — | 0 `M`/`P`; 46,259 `H` + 810,928 `h` |

### 6.6 The ESM3ΔG hold-out, verified

1. **Exact (`19_esm3dg_exact.py`):** A SHA-1 join of every benchmark row's wild-type *and* mutant sequence against all
   960,216 sequences in `mgnify_training_index.csv` returns **0 matches** across all 3.53 M benchmark rows (`0/87,544`
   Tsuboyama capped val+test, `0/2,465,767` ProteinGym, `0/534,385` Domainome, `0/25,632` external, `0/269,471` DMS).
2. **Homology (`15_mgnify.py`):** High-sensitivity MMseqs2 (`s = 7.5, e = 10, max-seqs 500`, evaluated by coverage tier
   across all qualifying hits) over all 52,071 capped-test rows (`leak_audit_esm3dg_libraries.csv`) finds **8/33** test
   scaffolds with a strong homolog (`21,185` rows = `40.7 %`: `1GYZ`, `3DKM`, `2MKX`, `2D1U`, `2KRU`, `2K52`, `2L09`,
   `1ZHC`), **21/33** with a weak match (`26,204` rows = `50.3 %`), and **4/33** with none (`4,682` rows = `9.0 %`).

---

## 7. What changes in the reported conclusions

- **Bold eligibility (`v / n < 0.005` on `M`, `m`, `P`, `H`, `D`, `U`, `#`)**: `h`, `V`, and `–` (including zero-shot /
  unsupervised models) remain eligible. In §1.1, ESM-MSR (devel/baseline) or ME stage 1 lead 11 of 12 sets, and `DDGemb`
  leads `S669` (`0.570` vs `0.560` devel); in §1.3 `Domainome`, `ProteinMPNN` (`0.548`) takes sole bold because all
  capped-train models carry `ᴴ` (`H.11`); in §1.4 and §3 `ProteinGym`, `ProSST` (`0.514 / 0.653 / 0.528`) takes bold
  because all supervised Tsuboyama models carry `ᴹ` (`M.02`).
- **Clean external-set means (`BENCHMARKS_OCT2026.md` §6.5)**: ESM-MSR (devel `0.652`, baseline `0.646`), ME stage 1
  (`0.615`), ME stage 2 (`0.605`), retrained SPURS (`0.588`), ThermoMPNN+D (`0.586`), ThermoMPNN (`0.585`), retrained
  SPURS multi (`0.556`), ProteinMPNN (`0.464`), ProSST (`0.423`), and the σ = 0 control (`0.378`) retain all 12 sets;
  `DDGemb` retains 2 (`S669`, `S461`), `GeoDDG-3D` retains 1 (`S461`), `EsmTherm` retains 1 (`Ssym`), and `ESM3ΔG`
  retains 5 (`S783`, `S669`, `S461`, `Ssym`, `Q3421`).

---

## 8. Assumptions and inherent limits

1. **Library-level upper bounds on training rows:** Private row filters (censoring, early stopping) can only remove rows,
   so `M`/`m` counts err conservatively upward.
2. **Protein-only disclosures (`EsmTherm`, `IFUM`):** `M?` is inferred (`0.8 × P` for EsmTherm's 80/10/10 per-row split;
   `1.0 × P` for IFUM).
3. **ESM3ΔG augmented checkpoints:** Evaluated against the full 960,216-sequence MGnify index (`0` exact matches; upper
   bound on `H`/`h` since the `*_augmented_*` weights used a subset).
4. **3D Foldseek scope:** Exhaustive Foldseek structural homology (`05d_structure_homology.py`) is integrated into the
   splitting graph (412 scaffolds × 776 external nodes) and into **Domainome** and **ProteinGym** against the 172
   capped-train scaffolds; third-party literature inventories (`S2450`, `S9028`) and sequence-only MGnify (`960k`
   sequences without PDBs) use high-sensitivity MMseqs2 (`s = 7.5`).
5. **Single-chain monomer representation:** Both splitting and audit extract single chains (`chain = 'A'` for MegaScale,
   Domainome, and DMS targets), ignoring oligomeric partner chains.
6. **Row-level fractions vs per-assay/per-protein metrics:** Cell fractions divide leaked rows by total dataset rows
   (e.g. 23 leaked ProteinGym DMS = `1.9 %` of rows, `M.02`, vs `10.8 %` of unweighted DMS assays).
7. **Descriptive scope:** Leak codes record training exposure, not causal score inflation.

---

## 9. Reproducibility

**Committed artifacts** (`docs/benchmarks/`, mirrored where used by `analysis/reference/`):

| file | contents |
|---|---|
| `leak_audit.sqlite` | SQLite database: `inventories`, `cells` (709 cells), `pg_per_dms`, `homology`, `structure_homology`, `code_exact` |
| `leak_audit_matrix.csv` | all 709 `(model, dataset)` cells with counts, `code`, and `sup` |
| `leak_audit_inventories.csv` | inventory catalogue with provenance and evidence level |
| `leak_audit_pg_per_dms.csv` | every inventory × 217 ProteinGym assays (`exact`, `constituent`, `sameprot`, `H`, `h`) |
| `leak_audit_homology.csv` | per inventory × dataset `P`/`H`/`h` counts |
| `leak_audit_structure_homology.csv` | Foldseek 3D structural homology hits on Domainome and ProteinGym vs capped train |
| `leak_audit_code_exact.csv` | verified exact & constituent counts for `S2450`, `S9028`, `SM1242`, `S8754` |
| `split_homology_audit.json` | split-validation and threshold-sensitivity metrics (§5) |
| `leak_audit_esm3dg_libraries.csv` | per-scaffold ESM3ΔG homology over all 33 capped-test scaffolds (52,071 rows) |
| `esm3dg_exact.csv`, `esm3dg_pg_exact_per_dms.csv` | exact-sequence SHA-1 join results across all 3.51 M benchmark rows |

**Pipeline execution (`analysis/leak_audit/run_all.py`)**:

Small plain-text upstream reference files (`S2450.tsv`, `S2450.fasta`, `S9028.csv`, `DD_SM1242.csv`,
`ifum_train_proteins.csv`, `esmtherm_split.csv`, `esmtherm_train_proteins.csv`, `tmpnnd_train_proteins.csv`,
`spurs_*_train_proteins.csv`, `thermompnn_upstream_train_proteins.csv`; 792 KB total) are tracked in
`analysis/leak_audit/data/` with a `.gitignore` excluding large binary/archive intermediates. Run the entire
19-step DAG in topological order with:

```bash
python analysis/leak_audit/run_all.py
```

| step | script | role |
|---|---|---|
| 00 | `00_build_inventories.py` | extract upstream SPURS, EsmTherm, and ThermoMPNN tables when upstream clones are present |
| 00b | `00b_extract_ifum_split.py` | parse IFUM `dataset_splits.pdf` into `ifum_train_proteins.csv` |
| 01 | `01_megascale.py` | build MegaScale variant & scaffold tables (normalising `.pdb_<mut>` mutant backgrounds) |
| 02 | `02_testsets.py` | normalise all benchmark rows to `(pid, seq, muts)` and validate sequence coordinates |
| 04 | `04_overlap_repo_models.py` | write `capped_split_sets.pkl` and repo-model Tsuboyama sanity checks |
| 05b | `05b_inventory_proteins.py` | extract WT sequences for `S2450`, `S9028`, `SM1242`, `S8754` (`inventory_proteins.fasta`) |
| 05 | `05_proteins.py` | unified protein table + hash-invalidated MMseqs2 all-vs-all (`s = 7.5`) with direction-invariant coverage |
| 05c | `05c_tmpnnd_inventory.py` | compute the 134 capped-train scaffolds seen by `ThermoMPNN-D` |
| 05d | `05d_structure_homology.py` | exhaustive Foldseek 3D search of Domainome & ProteinGym PDBs against capped-train scaffolds |
| 00c | `00c_prepare_mgnify.py` | build `mgnify_train.fa` and run MMseqs2 (`s = 7.5`) of benchmark queries vs 960k MGnify sequences |
| 06 | `06_families.py` | cluster ≥95 % identity / ≥90 % shorter-seq coverage families (`len_ratio ≥ 0.80`) and align to reps |
| 07 | `07_relations.py` | summarise test-protein relations to capped train |
| 09 | `09_homology_audit.py` | audit `homology_pairs.csv` across all 412 scaffolds + 776 external nodes (`split_homology_audit.json`) |
| 10 | `10_overlap_engine.py` | per-inventory exact (`M`), constituent (`m`), and same-protein (`P`) counts |
| 11 | `11_homology_classes.py` | coverage-tiered sequence + 3D Foldseek `P`/`H`/`h` classification per inventory × dataset |
| 12 | `12_code_exact.py` | construct- & reverse-aligned exact (`M`) and constituent (`m`) overlaps for `S2450`, `S9028+SM1242`, `S8754` |
| 15 | `15_mgnify.py` | coverage-tiered MGnify homology (`esm3dg_mgnify`) + 33-scaffold `esm3dg_tsu_libraries.csv` |
| 19 | `19_esm3dg_exact.py` | exact SHA-1 join of all 3.51 M WT & mutant sequences against `mgnify_index.csv` |
| 14 | `14_pg.py` | per-DMS ProteinGym overlap & 2D+3D homology table + clean aggregate recomputation |
| 13 | `13_matrix.py` | compose 709 `(model, dataset)` leak codes (`–` for unsupervised models; 0.5 % superscript rule) |
| 18 | `18_db.py` | export `docs/benchmarks/leak_audit.sqlite` and sync CSVs to `docs/benchmarks/` and `analysis/reference/` |

**Tools.** MMseqs2 13-45111, Foldseek 463739e, Biopython 1.85, pandas 2.3.0, networkx 3.3, SQLite 3.
