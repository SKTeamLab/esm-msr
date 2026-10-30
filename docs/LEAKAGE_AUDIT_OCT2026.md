# Data-leakage audit of the October 2026 benchmark tables

**Scope.** Every model and dataset reported in [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md), on the Oct06 capped
split. Companion to §6 of that document, which carries the summary and the annotated tables; this document carries
the evidence, the inventories, the validation of the split's homology tooling, the assumptions and the
reproducibility recipe.

**One-line result.** The capped split is sound for the models it was built for — ESM-MSR, MutateEverything and
ThermoMPNN have zero identical-mutation and zero same-protein overlap with every reported test set — but the
benchmark as a whole is not clean. 23 of the 213 ProteinGym assays and 1 of the 522 Domainome domains are *inside*
the capped training split; the third-party "Megascale-trained" models trained on 21–27 of our 33 Tsuboyama test
libraries; the literature-ΔΔG models trained on 28–98 % of several external sets at the mutation level; and the
existing †/‡ flags were wrong in both directions (ESM3ΔG is not Megascale-trained; DDMut and GeoDDG are on S571,
Ssym and PTMUL, which the tables called clean).

---

## 1. Method

The audit is a four-step pipeline, run as `analysis_scripts/leak_audit/01`–`18` (see §9).

1. **Normalise the test side.** Every reported row is reduced to `(protein sequence, set of (WT, position, mutant))`.
   The reported tables contain 2,446,730 ProteinGym rows (213 assays), 534,385 Domainome rows (522 domains), 149,360
   GB1 rows, 87,544 Tsuboyama capped val+test rows, 25,632 rows over the 12 external sets, and 269,471 rows over the
   seven DMS assays. The mutations are taken in the numbering of the sequence each dataset actually scores
   (`seq_pos`/`mut_type`/`mutant`), and validated against that sequence: 0 rows in the Tsuboyama val/test tables,
   0 in ProteinGym, 1,779 in Domainome (0.33 %, construct/numbering edge cases, dropped and counted) fail the check.
2. **Build same-protein families.** All test proteins plus all inventory proteins (3,747 sequences) are searched
   all-vs-all with MMseqs2 (`easy-search -s 7.5 -e 10 --max-seqs 20000`) and clustered by single linkage at ≥95 %
   identity over ≥90 % coverage. Each member is aligned to its family representative (Biopython, global, BLOSUM62)
   and every mutation is re-expressed in representative coordinates, with a ±3-position neighbour search when the
   alignment is ambiguous at a repeat. This is what makes "identical mutation" meaningful across PDB entries,
   chain numbering and construct offsets (e.g. S2648 and Q3421 share the same PDB codes but not always the same
   construct).
3. **Reconstruct each model's training inventory.** Where the split is in the repo (ESM-MSR, ME, ThermoMPNN) the
   inventory is the train split itself. Where the training data ship with a public model or split file (SPURS,
   EsmTherm, IFUM, DDGemb, DDMut, GeoDDG, ESM3ΔG), the inventory is that file. Where only the corpus is documented
   (ProSST, ProteinMPNN, ESM3), the inventory is the corpus and the class is `C`. The provenance and the evidence
   level of each inventory is in §3.
4. **Classify every test row** against every inventory (classes in §2), aggregate per dataset, and compose the cell
   code. A row can be `M` for one model and `–` for another; the cells are per model.

The leakage classes are deliberately ordered by strength of evidence, not by presumed effect on the score. A
fully trained assay can score badly (the model may not generalise even to its training rows) and a same-protein
overlap may be worth nothing. The audit says what was trained, not what it bought.

## 2. Notation

| code | meaning | test |
|---|---|---|
| `M` | identical variant (same protein family, identical mutation set) in training | verified against the inventory file |
| `m` | every mutation of the row appears in training in the same protein, but not the exact combination | verified |
| `P` | same protein in training, no mutation match | ≥95 % identity over ≥90 % coverage (family), or ≥90 %/≥50 % |
| `H` | strong homolog in training | MMseqs2 E ≤ 1e-3 over ≥50 % of the test protein |
| `h` | weak/ambiguous homology | E ≤ 1e-2 or ≥30 % identity over ≥30 % of the test protein |
| `D` | documented training on the same source dataset, rows not public | paper/code |
| `C` | pretraining-corpus exposure only (WT sequence/structure, no mutation supervision) | paper |
| `U` | training data not disclosed | paper |
| `V` | dataset used for validation or model selection | split file |
| `#` | not a test result: the dataset is the training set | file identity |
| `?` | probable but unverified (e.g. `M?`, `P?`) | inference |

A cell lists every class present with its fraction of the dataset's reported rows, highest-severity first:
`M.59 P.35 H.02`. The numeric tables in `BENCHMARKS_OCT2026.md` carry only the worst class as a superscript
(suppressed below 0.5 % of rows there, retained in the companion matrix and in the database).

## 3. Inventories

| model row(s) | training inventory | n | source | evidence |
|---|---|---|---|---|
| ESM-MSR (released), ESM-MSR (devel), WT/MT heads, σ sweep, ESM-MSR-small | the capped-train libraries (172), all substitutions | 229,721 rows / 172 libraries | `data/splits_oct06_capped.pkl` + `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | verified |
| MutateEverything stage 1 / stage 1+ | the same libraries, singles (stage 1); additive readout uses the same model | 168 libraries with singles | capped split; ME recipe | verified |
| MutateEverything stage 2 / stage 2+ | the same libraries, singles then doubles (stage 2 is initialised from stage 1) | 168 libraries | capped split | verified |
| ThermoMPNN (additive) | the same libraries, singles | 168 libraries | capped split; `train_single.yaml` | verified |
| ThermoMPNN-D | the same libraries, doubles | 51 libraries with doubles | capped split; `train_epistatic.yaml` | verified |
| SPURS single | Megascale train+val libraries after the authors' >25 % identity blacklist | 212 train + 29 val libraries; 216,336 singles | `SPURS_mod/data`, ThermoMPNN 8:1:1 split, `mmseq_mut_search_0.25.m8` | verified |
| SPURS multi | the same libraries, doubles | 132,950 doubles | `SPURS_mod/data` | verified |
| EsmTherm | `K50_dG_Dataset1_Dataset2`; 414 wildtypes contribute train mutants under a per-row 80/10/10 split, only 47 are test-only | 414 train domains | `EsmTherm/datasets/split/wildtype_split.csv` | protein-level (rows not released) |
| IFUM | Megascale Dataset2; 322 train libraries, 86 in an unseparated val/test block | 322 train libraries | IFUM `dataset_splits.pdf` | protein-level |
| DDGemb | S2450 = S2648 − 18 sequences / 198 mutations >25 % identity to S669 | 2,450 mutations / 115 UniProt | `ddgemb.biocomp.unibo.it` S2450.tsv/.fasta | verified |
| DDMut | S9028 = S2648 + FireProtDB, forward+reverse; multi-mutant branch SM1242 | 9,028 rows / 171 PDB; 1,242 multi rows | `biosig.lab.uq.edu.au` S9028.csv, SM1242.csv | verified |
| GeoDDG-3D | S8754 (its own training set) | 8,754 rows / 301 proteins | `GeoStab` S8754.csv | verified (same file as the benchmark's S8754: 8,745/8,754 names identical) |
| ESM3ΔG | MGnify Stability fine-tune, **not** Megascale | 960,216 sequences | Zenodo 19411306 `mgnify_training_index.csv` | verified (0 exact WT or mutant sequences anywhere in the benchmark) |
| ProSST | AlphaFoldDB 18.8 M structures, masked-LM; no ΔΔG fine-tune | — | ProSST paper | documented only |
| ProteinMPNN | PDB assemblies deposited before 2021-08-02, sequence recovery | 25,361 clusters | ProteinMPNN paper | documented only |
| ESM3 small/medium/large, σ = 0 control | UniRef/MGnify/JGI + PDB (≤2020-05-01) + AFDB v4 | 2.78 B proteins | ESM3 paper | documented only |
| Rosetta Cartesian ΔΔG | none | — | physics | n/a |

Two corrections to the old table legend follow. **ESM3ΔG is not Megascale-trained**: its released `*_augmented_*`
weights are fine-tuned on the authors' own 960,216-sequence MGnify Stability set (five library versions K50dG /
DMSv4 / DMSv5 / DMSv7 of that same experiment), and the 28,172-mutant Megascale set is its held-out test. An exact
SHA-1 join of every benchmark wild-type and mutant sequence against that corpus returns 0 matches; 8 of the 33
capped test libraries have a strong homolog (40.7 % of rows) and 21 a weak one (50.3 %) — see §6.6. **S8754 is
GeoDDG's training set**, not an independent external set; its reported 0.841 on it is a training-set score.

## 4. The datasets overlap each other

Before any model, the test sets overlap. Measured on unique `(PDB code, sequence, mutation)` identifiers:

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

Consequences: a model trained on **S8754** is exact-leaked on S2648 (2,587/2,648), S783 (691/783), Q3421 (2,409/3,421),
K3822 (2,682/3,822), K2369 (1,826/2,369), S571 (216/557) and Ssym (256/684), and *not* on S669 or S461. A model
trained on **S2648** leaks S8754, Q3421, K3822, K2369, Ssym, S571 and S783 to the counts above, and neither S669
nor S461. S669 and S461 are the only two sets that are mutation-disjoint from the dominant ProTherm lineages, which
is why they carry the audit's cleanest external comparisons. K2369 ⊂ K3822, S461 ⊂ S669 and PTMUL-D ⊂ PTMUL exactly.

## 5. Does the split's homology tooling hold up?

The split was produced by `preprocessing/split_tsuboyama.py` + `preprocessing/homology.py`; its edge rule is
"MMseqs2 E ≤ 1e-3 **or** Foldseek E ≤ 1e-3 **or** identity > 0.40 **or** TM-score > 0.70 (shorter-chain
normalised)". We re-analysed the 408,330-pair table the split was built from (`homology_pairs.csv`) and the 370-library
assignment.

| test | result | verdict |
|---|---|---|
| rule is enforced | 25,027 edges; 366 cross-set edges, **all** involving the 142 excluded libraries; **0** cross a kept boundary | pass |
| the quarantine does work | all 142 excluded libraries have a direct pair to an external benchmark protein; 90 pairs at identity > 0.9, 117 at TM > 0.9 | pass |
| Foldseek adds value | 550 structure-only pairs (no MMseqs hit, identity ≤ 0.40); none crosses a kept boundary | pass |
| threshold sensitivity | at identity > 0.30, 218 train/val/test↔external pairs would newly cross; at >0.25, 610; at TM ≥ 0.5, 177; at E ≤ 1e-2, only 4 | the E-value is doing the work; the 40 % cap is the weaker half |
| the grey zone is real | 8,114 capped-test rows (15.6 %) and 7,210 val rows (20.3 %) have a 30–40 % identity match to a train library that the rule did not exclude | flagged `h`, not `H` |
| exclusion is not transitive | 134 train libraries share a component with an external protein through an excluded bridge; all 134 sit just below the thresholds (max identity 0.395, max TM 0.6998) | no strong pair missed |
| the rule's coverage | the quarantine is built from the 9 classical ΔΔG sets + 5 functional proteins only. ProteinGym and Domainome were deliberately not in it (Domainome would have bridged clusters) | **the main gap** |

The identity cap uses BLOSUM62 local matches / shorter length, not MMseqs2's alignment identity; the two disagree by
up to ~0.1 on short designs (one 0.411-MMseqs-identity pair is 0.326 by the split's measure and stays in train). This
is the source of the `h` grey zone, and it is why the audit reports those rows as ambiguous rather than clean.

## 6. What leaked, model by model

### 6.1 The models the split was built for

ESM-MSR (released and devel, all heads and σ values), MutateEverything stage 1/2, ThermoMPNN and ThermoMPNN-D
have **zero** identical-mutation and **zero** same-protein overlap with all 12 external sets, both Tsuboyama capped
splits, the 7 DMS assays and Domainome (outside the one domain below). The capped split does exactly what it was
designed to do. Their residual exposure is `h`: 15.6 % of test rows and 20.3 % of val rows are on proteins with a
30–40 % identity match to a train library, and the val columns are `V` (used for early stopping / checkpoint
selection).

**The retrained baselines share the inventory.** MutateEverything stage 1 (singles), stage 2 (singles then
doubles) and ThermoMPNN (singles; ThermoMPNN-D doubles only) were retrained on the same 172 libraries as ESM-MSR, so
their `P`/`H`/`h` columns are identical by construction and only `M`/`m` differ. The single weak-homology row in
S669 and S461 (`1IOJ` vs `HHH_rd2_0155`, 37 % identity over 61 % of the test protein, E = 5.5) is present for
ESM-MSR, ME stage 1/2 and ThermoMPNN and absent for ThermoMPNN-D, whose double-only train subset does not contain
that library. It is below the 0.5 % display threshold in the numeric tables. Retraining added no overlap: the
upstream models train on the whole Megascale pool (upstream ThermoMPNN alone covers 19 of the 33 capped test
libraries and 15,918/36,526 singles), strictly more than the capped split.

The one gap in the split's own scope is ProteinGym and Domainome, which it never guarded:

- **23 ProteinGym assays are 100 % in the capped train split.** They are Tsuboyama DMS that are also Megascale
  libraries in the train split; every row of each assay is an exact trained variant (45,623 rows, 1.9 % of
  ProteinGym, 10.8 % of its assays). All 23 are Stability assays; ESM-MSR scores 0.897 on them. Removing them moves
  the devel-seed DMS mean 0.561 → 0.520, the Stability category 0.880 → 0.871 and the PG average 0.516 → 0.514.
  MutateEverything (24,252 exact + 21,331 constituent) and ThermoMPNN (24,252 exact singles) are in the same position.
  The 23 assays are listed in §8.
- **One Domainome domain is a Megascale train library.** `O95155_PF04564_1229` (UBE4B) is the library `3L1X`:
  1,134 of its 1,206 rows are exact trained variants and 3,177 more rows (other domains) are same-protein. That is
  0.21 % of Domainome and 1 of 522 domains, so its effect on the per-domain mean is bounded (~0.001 ρ), but the
  domain's own score should not be reported as held out.
- **SPURS adds its own Domainome overlap:** 6,181 exact rows (1.2 %) because its train split contains several
  Megascale libraries that Domainome re-used.

**Ssym is not a Megascale set.** None of Ssym's 357 PDB chains or its 14 protein families is a Megascale library,
so SPURS, EsmTherm, IFUM and ESM3ΔG have no same-protein overlap with it. The best sequence match between an Ssym
protein and a SPURS train library is 70 % identity over 23 % of a lysozyme (E = 3); Foldseek of the 357 Ssym
structures against the 269 SPURS train structures returns 0 pairs at E ≤ 1e-3 or TM > 0.7, the strongest being
fold-level matches at TM 0.50–0.53 to small designs (lddt 0.35–0.43). SPURS's 0.784 on Ssym is therefore a clean
number, and it stands ahead of ESM-MSR's 0.773.

### 6.2 The third-party "Megascale-trained" models

| model | trained on our test libraries | exact | same protein | verdict |
|---|---|---|---|---|
| SPURS single | 24 of 33 test libraries | 21,282 rows (40.9 % of all rows; 58 % of singles) + 6,747 constituent | 18,453 | heavily leaked |
| SPURS multi | 9 of 13 double-bearing libraries | 6,400 rows (41.2 % of doubles) + 1,717 constituent | 21,477 | heavily leaked |
| EsmTherm | 27 of 33 test libraries contribute train mutants (per-row 80/10/10) | rows not public (`M?` ≈ 0.8 × same-protein) | 42,783 (82.2 %) | leaked at the protein level |
| IFUM | 21 of 33 test libraries in train | rows not public (`M?` = same-protein) | 23,518 (45.2 %) | leaked at the protein level |
| ESM3ΔG | **0 exact rows anywhere** | 0 | 0 | not trained on this data; 40.7 % strong + 50.3 % weak homologs |

SPURS's fit is the ThermoMPNN 8:1:1 split (239 train / 31 val), which is *not* the Oct06 split; its val is in the
fit. Its reported 0.866 on capped test (0.844 on val) is therefore not a held-out result, and the old bullet that
called it "not yet interpretable" now has a measured explanation. EsmTherm's split is per-row within domain, so 414
of its 461 domains contribute training mutants and the same domain appears in its train and test; the audit records
`P` (same protein) and `M?` (the expected 80 % exact subset) rather than claiming exact rows it cannot see. IFUM's
published split has only "Train" and "Val and test" sections; 21 of our test libraries are in its train.

ESM3ΔG is the correction in the other direction. Its released weights are fine-tuned on MGnify Stability, not
Megascale; its Megascale benchmark is the ThermoMPNN 28-protein/28,172-mutant test set. Against our 33 test
libraries, 0 are in its 960,216-sequence training index by exact sequence, 8 have a strong homolog (40.7 % of test
rows) and 21 a weak one (50.3 %). It is the cleanest third-party comparator on this split, with a homology caveat, and its ‡
flag should be removed.

### 6.3 The literature-ΔΔG models

| model | training set | exact overlap with the reported sets | notes |
|---|---|---|---|
| DDGemb | S2450 | S2648 2,450/2,648 (92.5 %); S8754 ≤2,587; Q3421 ≤1,549; K3822 ≤979; K2369 ≤566; Ssym 194; S571 52; S783 10 exact / 33 proteins; S669, S461 0; PTMUL/PTMUL-D 0 | upper bounds via S2648 (S2450 removes 198 mutations; the removed rows cannot all be located in the mapped copies) |
| DDMut | S9028 + SM1242 | S2648 2,347/2,644 (88.8 %); S8754 5,272/8,750 (60.3 %); Q3421 2,248 (65.7 %); K3822 2,195 (57.4 %); K2369 1,530 (64.6 %); S783 330 (42.1 %); Ssym 203 (29.7 %); S571 112 (19.6 %); PTMUL 538/914 (58.9 %); PTMUL-D 194/536 (36.2 %); S669, S461 0 exact but 11 %/16 % same-protein | the multi-mutant branch saw most of PTMUL; the server model was then further trained with blind tests |
| GeoDDG-3D | S8754 | S8754 8,745/8,754 (100 %, **the training set**); S2648 2,583 (97.7 %); S783 691 (88.3 %); K2369 1,803 (76.1 %); K3822 2,648 (69.3 %); Q3421 2,351 (68.7 %); S571 220 (38.5 %); Ssym 256 (37.4 %); S669, S461 0 exact; PTMUL ≈ 0 exact but 97 % same-protein | S783 is mutation-disjoint from S2648 but not from S8754; its 0.865 is a training-set score |

The "six sets with no documented overlap for any model" claim is withdrawn: only S669 and S461 have zero identical
mutations for every model, and even they are not same-protein-clean for DDMut (11 %/16 %) or GeoDDG (1 % + 3 %
homology). S571, Ssym and PTMUL are not clean for DDMut/GeoDDG; Q3421, K3822 and K2369 are not clean for DDGemb,
DDMut or GeoDDG; and the Megascale natural libraries put EsmTherm/IFUM/SPURS on many of the classical sets at the
same-protein level.

### 6.4 The DMS assays

The seven DMS assays (DLG4 abundance/binding, GRB2 abundance/binding, MYO, ESTA, GB1) are clean for ESM-MSR, ME and
ThermoMPNN. They are not clean for the Megascale and literature models:

- **GB1** is the Megascale library `1PGA`. EsmTherm and IFUM trained on it (all 149,360 rows same-protein, `P`, with
  an `M?` subset); GeoDDG's S8754 contains 1,538 GB1 single mutants (61 exact, 61,880 constituent rows); DDMut's
  S9028 contains 1,820 (69 exact, 103,334 constituent). SPURS removed 1PGA in its blacklist but a near-identical
  Megascale design remains (all 149,360 rows `H`/`P`). ESM3ΔG has no GB1 homolog. GB1 is the weakest column for every
  model, so the practical effect is small, but "seven DMS" is not seven clean assays.
- **DLG4/GRB2/MYO/ESTA** are not in any Megascale or literature training set we could reconstruct; the only flags
  are corpus exposure (`C`) for the zero-shot models and `h`/`H` traces from the MGnify fine-tune (ESTA is weak-
  homologous for ESM3ΔG).
- **Domainome** is clean for ESM-MSR/ME/ThermoMPNN except the UBE4B domain (§6.1), and same-protein/homolog-flagged
  for EsmTherm (5 %), IFUM (5 %), DDGemb (3 %), GeoDDG (5 %) and ESM3ΔG (19.8 % strong + 32.1 % weak homologs).

### 6.5 ProteinGym, per assay

`docs/benchmarks/leak_audit_pg_per_dms.csv` gives every inventory × every one of the 213 assays with exact and
constituent counts. The headline:

| inventory | assays with ≥50 % of rows trained | assays with any trained row | rows |
|---|---|---|---|
| ESM-MSR / ME stage 2 | 23 | 23 | 45,623 exact |
| ME stage 1 | 23 | 23 | 24,252 exact + 21,331 constituent |
| ThermoMPNN (singles) | 23 | 23 | 24,252 exact |
| SPURS single | 51 | 51 | 50,866 exact + 37,435 constituent |
| SPURS multi | 23 | 45 | 44,106 exact |
| EsmTherm | — (protein-level) | 97,045 same-protein rows | — |
| IFUM | — (protein-level) | 95,052 same-protein rows | — |
| ESM3ΔG | 0 | 0 exact | 43,119 strong + 724,254 weak homologs |

SPURS is more contaminated on ProteinGym than ESM-MSR: its train+val covers 51 of the 213 assays at ≥50 %, which is
why its 0.483 average is not comparable to ESM-MSR's 0.510.

### 6.6 The ESM3ΔG hold-out, verified

The 28,172-mutant held-out set is the ThermoMPNN 28-protein split, not ours, so it does not tell us whether our 33
capped test libraries were trained on. Two checks close that gap.

**Exact.** A SHA-1 join of every benchmark row's wild-type *and* mutant sequence against the 960,216-sequence MGnify
train split returns **0 matches**: 0/87,544 Tsuboyama capped val+test rows, 0/2,446,730 ProteinGym rows (including
all 117,811 Tsuboyama DMS rows), 0/534,385 Domainome rows, 0/25,632 external-set rows, 0/269,471 DMS-assay rows.
No exact protein or exact variant of ours is in ESM3ΔG's training data. The join is on sequences, so it also catches
a training mutant equal to one of our test mutants, which a mutation-name join would miss.

**Homology.** High-sensitivity MMseqs2 (s = 7.5, e = 10, max-seqs 500) against the same corpus: 8 of the 33 capped
test libraries have a strong homolog (41–68 % identity over 78–96 % of the test protein, E 1e-20 to 1e-6), covering
40.7 % of test rows; 21 libraries have a weak/ambiguous local match (50.3 % of rows) and 4 have none. The strong
hits are all natural PDB libraries (1GYZ, 3DKM, 2MKX, 2D1U, 2KRU, 2K52, 2L09, 1ZHC); the de novo designs are weak
or absent. Per-library table: `docs/benchmarks/leak_audit_esm3dg_libraries.csv`.

**Cost of the full comparison.** MMseqs2 is not expensive at this scale. Against the 960k-sequence corpus, a
33-query search takes 4.7–5.8 s including target indexing (1.3 GB peak RSS; 305 hits at s = 5.5/max-seqs 50, 1,283
at s = 7.5/max-seqs 500), and the full 2,583-protein test set takes **9.7 s** at s = 7.5 (1.5 GB RSS, 40,451 hits).
The audit's recipe is therefore: exact hash join first (instant, definitive), then MMseqs2 at s = 7.5 for every
protein the join does not resolve, then per-row coverage classification. No heuristic shortcut is needed.

## 7. What changes in the reported conclusions

- **Bold** now marks the best clean cell per column (no `M`, `m`, `P`, `H`, `D`, `U`, `V`, `#`; `h` and `C` remain
  eligible and are marked). New bold values include §1.1 s571 ESM-MSR (released) 0.456 (GeoDDG 0.526 is `M`); s783
  ESM-MSR (devel) 0.725; s2648 ME stage 1 0.701; s8754 ESM-MSR (devel) 0.625 (GeoDDG `#`, DDGemb/DDMut `M`); S669
  DDGemb 0.570 (GeoDDG/DDMut `P`); S461 ESM-MSR 0.756; §1.3 Domainome ProteinMPNN 0.548 (ESM-MSR `M`); §1.4 ProSST
  0.514/0.653/0.528 (all supervised rows `M`). §1.2's clean held-out group is unchanged.
- **The 12-set mean** remains in the table but is not a like-for-like ranking: GeoDDG's 0.767 and DDGemb's 0.697 are
  training-set-adjacent. Over cells with no flag, ESM-MSR (devel) 0.62, ME stage 1 0.56, ThermoMPNN 0.53 lead.
- **The "clean six" ranking is replaced** by a per-model clean-subset mean (Table §6.5): only ESM-MSR, ME,
  ThermoMPNN, ProSST, ProteinMPNN and the σ = 0 control are clean on all six; ESM-MSR (released) 0.621 leads them,
  with DDGemb 0.612 on the only two sets it has left.
- **The ProteinGym tie moves.** On all 213 assays ESM-MSR 0.510 and ProSST 0.514; on the 190 assays with no trained
  rows ESM-MSR is 0.520 and ProSST 0.528. The 0.22 Stability lead that ESM-MSR holds is entirely on the 23
  fully-trained assays; its four-category deficit is untouched.
- **SPURS is explained, not questioned.** Its 0.866 on capped test is 41 % exact rows and 35 % same-protein rows
  against ESM-MSR's 0; the "not yet interpretable" caveat can be retired.
- **ESM3ΔG's ‡ is withdrawn** and its S571/S783/Tsuboyama rows become genuine zero-shot comparisons with a homology
  caveat (40.7 % of Tsuboyama test rows have a strong homolog in its MGnify fine-tune).
- **S8754's GeoDDG cell is marked `#`** (training set) and the S783 GeoDDG cell is `M.88`; neither supports the old
  "GeoDDG leads the external sets" reading.
- **S571 is ΔTm, not ΔΔG** (its own literature, the GeoStab ΔTm set), which the old table did not say. It is kept
  for continuity but should not be averaged with ΔΔG sets without that caveat.

## 8. Ambiguities and unverifiable overlaps (conservative register)

Everything below is marked in the tables (`?`, `h`, `C`, `P`) and is counted as a caveat even when it may be
worthless.

1. **EsmTherm exact rows.** Only the wildtype split is public; `M? ≈ 0.8 × P` is an inference from the per-row
   80/10/10 split. The table flags `P` and `M?` separately so the reader can discount either.
2. **IFUM exact rows.** Same: domains are public, rows are not. `M? = P` assumes the model trained on all mutants of
   a train domain; the paper's 30–80 aa window is necessary but not verified per domain.
3. **ESM3ΔG augmented checkpoints.** The exact join returns 0 matches against the 960,216-sequence base index; the
   homology fractions use the high-sensitivity MMseqs2 run (s = 7.5, max-seqs 500). The augmented checkpoints we
   scored "used a more limited subset", so the fractions are an upper bound.
4. **S2450-inherited DDGemb overlaps.** S2450 = S2648 − 198 mutations homologous to S669; the removed rows cannot all
   be located in the mapped S2648 copy, so S8754/Q/K/Ssym/S571/S783 figures are upper bounds (`≤`).
5. **Identity vs E-value.** The split's identity is BLOSUM62 matches / shorter length; MMseqs2's is alignment
   identity. Pairs in the 0.30–0.40 band are `h`, not `H`, and one 0.411 pair is 0.326 by the split's measure. A
   stricter split would exclude 218–610 more pairs (§5).
6. **Structure homology outside the split.** Foldseek was validated on the split's pair table but not run on the
   960k MGnify corpus, the literature sets or Domainome; a structure-only homolog could be missed there. Sequence
   search is the audit's only homology channel for those inventories.
7. **Domainome's 1,779 unparsed rows** (0.33 %) are dropped; they are construct/numbering edge cases and cannot
   change a class.
8. **S8754 normalisation.** The training file and the benchmark's mapped copy agree on 8,745/8,754 names; the nine
   differences are treated as exact (same dataset).
9. **ProSST/ProteinMPNN/ESM3 corpora** are documented but not reconstructable here. `C` means "the corpus probably
   contains this protein", not "the model trained on this variant". It is a different kind of caveat from `M`.
10. **Validation (`V`).** ESM-MSR, ME, ThermoMPNN and SPURS used the capped val (or their own val) for checkpoint
    selection; the val column is marked. The test column is not.
11. **No effect size.** The audit does not estimate how much of a contaminated score is due to the leak. The
    memorised ProteinGym assays score 0.897, not 1.0, and GB1 is contaminated for four models yet is the weakest
    column for all of them.

## 9. Reproducibility

**Committed artifacts** (all under `docs/benchmarks/`):

| file | contents |
|---|---|
| `leak_audit.sqlite` | tables `inventories`, `cells` (676 model×dataset cells with counts and codes), `pg_per_dms`, `homology`, `code_exact` |
| `leak_audit_matrix.csv` | the same cells, flat |
| `leak_audit_inventories.csv` | inventory catalogue with provenance and evidence level |
| `leak_audit_pg_per_dms.csv` | every inventory × 213 ProteinGym assays, exact/constituent counts |
| `leak_audit_homology.csv` | per inventory × dataset `P`/`H`/`h` counts |
| `leak_audit_code_exact.csv` | exact counts for the code/position inventories (S2450, S9028, SM1242, S8754) |
| `split_homology_audit.json` | the split-validation numbers of §5 |
| `leak_audit_esm3dg_libraries.csv` | per-library ESM3ΔG homology for the 33 capped test libraries |
| `esm3dg_exact.csv`, `esm3dg_pg_exact_per_dms.csv` | exact-sequence join results for every benchmark row |

**Scripts** (`analysis_scripts/leak_audit/`, run in order; each is deterministic for the inputs below):

| order | script | what it does |
|---|---|---|
| 01 | `01_megascale.py` | compact Megascale library/variant tables from `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` |
| 02 | `02_testsets.py` | every test row → `(protein, mutation, label)`; validates positions against sequences |
| 05 | `05_proteins.py`, `05b_inventory_proteins.py` | protein corpus (test + inventory; PDB sequences extracted for S9028/SM1242) |
| 06 | `06_families.py` | same-protein families and representative-coordinate maps |
| 09 | `09_homology_audit.py` | split-rule audit, sensitivity, transitive and structure-only analysis |
| 10 | `10_overlap_engine.py` | per-inventory exact/constituent/same-protein/homolog counts |
| 11 | `11_homology_classes.py` | `P`/`H`/`h` at whole-protein level |
| 12 | `12_code_exact.py` | exact overlaps for DDGemb/DDMut/GeoDDG |
| 13–16 | `13_matrix.py`, `14_pg.py`, `15_mgnify.py`, `16_render.py` | cell codes, ProteinGym clean aggregates, ESM3ΔG corpus, table rendering |
| 19 | `19_esm3dg_exact.py` | exact SHA-1 join of every benchmark WT/mutant sequence against the MGnify train split |
| 18 | `18_db.py` | SQLite + CSVs |

**Inputs not in the repo** (public, with the URLs in `docs/LEAKAGE_AUDIT_OCT2026.md` §3): S2450.tsv/.fasta
(`ddgemb.biocomp.unibo.it/static/ddgemb/data/`), S9028.csv and SM1242.csv (`biosig.lab.uq.edu.au/ddmut/data/`),
S8754.csv (`github.com/Gonglab-THU/GeoStab`), the IFUM `dataset_splits.pdf`, the EsmTherm `wildtype_split.csv`
(local clone), the SPURS `mega_splits.pkl` + `mmseq_mut_search_0.25.m8` (local clone), the ThermoMPNN
`dataset_splits/` (local clone), the ProteinGym DMS files, the Domainome supplementary table (SPURS clone), and the
MGnify `mgnify_training_index.csv` (Zenodo 19411306, gated). The split's own pair table is the salvaged
`analysis/salvaged_2026-10-16/structure-homology-splits-e5ca04/.../homology_pairs.csv`.

**Tools.** MMseqs2 13-45111, Foldseek 463739e (static build), Biopython 1.85, pandas 2.3.0, networkx 3.3, SQLite.

## 10. Limitations

1. Training rows are treated as all rows of a trained library. Private filters (censored rows, ME's stage split,
   ThermoMPNN's early stopping, SPURS's m8 blacklist index alignment) can only remove rows, so the exact overlaps
   are upper bounds — the conservative direction.
2. A `P` classification is a sequence-identity judgement (≥95 %/≥90 % or ≥90 %/≥50 %). Constructs with missing tags
   or one-residue offsets are the same protein; a close paralogue above 90 % would also be, and is treated as `P`,
   i.e. conservative.
3. The megascale-trained inventories are reconstructed from the authors' public splits, not from their exact
   training runs. SPURS's m8 blacklist is applied by row id and the raw/filtered index alignment is uncertain; the
   audit uses the library-level retained set, which is the conservative choice.
4. No predictions were re-run and no scores were recomputed except the ProteinGym clean aggregates (from the local
   devel-seed per-DMS files), the GB1-excluded means and the clean six-set means. The numbers in the tables are the
   original ones.
5. The audit says nothing about whether a leak helps. That requires a retrain-without-the-overlap experiment, which
   was not run.
