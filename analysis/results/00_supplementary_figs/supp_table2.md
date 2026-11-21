# Supplementary Table 2 — Training-Set Provenance and Leakage Audit

## Supplementary Table 2a: Model Training-Set Inventories and Provenance

| Inventory | Description | Provenance | Evidence |
|---|---|---|---|
| `esm_msr_capped` | ESM-MSR released, baseline and devel checkpoints (every head, every sigma): all substitutions of the 172 capped-train scaffolds; checkpoints selected on capped val | data/splits_oct06_capped.pkl + Tsuboyama2023_Dataset2_Dataset3_20230416.csv | row-level verified |
| `corpus` | ProSST / ProteinMPNN / ESM3 zero-shot and the sigma=0 readout: unsupervised pretraining only | papers | documented (no supervised data) |
| `rosetta_beta_nov16` | Rosetta cartesian_ddg (2019, beta_nov16_cart): no per-variant training; 20 reference energies fit with Kellogg-2011 ProTherm ddG + Rocklin-2017 design stability in the objective; 1/2.94 scale fit on Kellogg | docs.rosettacommons.org Updates-beta-nov16; Park et al. 2016 JCTC SI; Kortemme-Lab/ddg kellogg.csv | calibration (V) |
| `geoddg_s8754` | GeoDDG-3D: fine-tuned on S8754 (its own test set S8754 is #) | GeoStab data/ddG/S8754.csv (GS_S8754.csv) | row-level verified |
| `geofitness_dms` | GeoDDG-3D encoder pretraining: GeoFitness, 74 DMS fitness sets (52 MaveDB + 22 DeepSequence) | GeoStab data/dms/dms.csv (wild types only: geofitness_dms_wt.fasta) | protein-level |
| `esm3dg_mgnify` | ESM3dG: MGnify Stability fine-tune, 960,216 train sequences (augmented checkpoints use a subset); not Megascale | Zenodo 19411306 mgnify_training_index.csv | row-level verified (0 exact sequences) |
| `esmtherm` | EsmTherm: K50_dG Dataset1+2; 47 test-only domains, the other 414 split 80/10/10 per sequence | PLoS Comput Biol 2024 S2 Spreadsheet pcbi.1012248.s006.csv (527,785 sequences with split) + datasets/split/wildtype_split.csv | row-level verified |
| `ifum` | IFUM: Megascale Dataset2, MMseqs2-30% cluster split; 322 train domains, 86 in a merged val/test block (no row lists) | IFUM dataset_splits.pdf (02_ifum_inventory.py) | protein-level (M?, V? are ceilings) |
| `ddgemb_s2450` | DDGemb: S2450 = S2648 minus 198 S669-homologous mutations, doubled with reverse variants | ddgemb.biocomp.unibo.it S2450.tsv + S2450.fasta | row-level verified |
| `ddmut` | DDMut as deployed (union of the two rows below); used for every DDMut cell | S9028.csv + DD_SM1218.csv + DD_SM444.csv | row-level verified |

## Supplementary Table 2b: Model × Benchmark Training-Overlap Matrix

| Model / Training Inventory | Ssym | S461 | S669 | K2369 | S783 | S2648 | Q3421 | S8754 | S571 | PTMUL | Mega Val | Mega Test | Domainome (unfilt.) | ProteinGym (unfilt.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR & capped retrains* | – | h.002 | h.001 | h<.001 | – | – | – | – | – | – | h.20 V | h.16 | M.004 P.004 H.11 h.02 | M.02 m<.001 H.002 h.003 |
| Zero-shot (ESM3 / ProteinMPNN / ProSST) | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Rosetta (beta_nov16_cart) | – | – | – | – | – | – | – | – | – | – | V.07 | V.12 | – | V.002 |
| GeoDDG-3D | M.50 P.50 | P.11 H.004 h.03 | M.01 P.07 H.03 h.03 | M.91 P.003 H.003 h.002 | M.88 h.07 | M1 | M.98 P.002 | M1 # | M.41 P.09 H.04 | m.73 P.24 H.004 | – | h.07 | M.001 P.05 H.28 h.08 | – |
| ESM3ΔG | h.72 | h.64 | h.55 | H.007 h.26 | h.44 | H.02 h.33 | H.004 h.37 | H.01 h.31 | P.01 H.005 h.33 | H.009 h.34 | P.19 H.30 h.24 | H.41 h.50 | P.005 H.22 h.51 | P.003 H.02 h.33 |
| ESMtherm | – | P.35 h.14 | P.24 H.003 h.09 | M.03 P.36 H.03 h.02 V.003 | M.004 P.19 h.04 | P.05 H.02 h.01 | M<.001 P.06 H.03 h.02 | M.01 P.21 H.02 h.008 V.002 | M.02 P.10 H.03 V.009 | P.13 H.06 h.01 V.001 | M.44 P.46 H.03 h.02 V.05 | M.51 P.34 H.04 h.03 V.06 | M.001 P.05 H.37 h.04 V<.001 | M.02 P.02 H.004 h<.001 V.003 |
| IFUM | – | P.21 h.14 | P.14 H.003 h.09 | M?.21 P.17 H.02 h.01 V?.003 | M?.03 P.006 H.03 h.11 | M?.006 P.05 H.02 h.006 | M?.01 P.05 H.02 h.01 | M?.12 P.10 H.02 h.003 V?.001 | M?.04 P.01 H.08 | M?.001 P.15 H.03 h.001 | M?.70 P.01 h.07 V?.28 | M?.46 P.02 H.07 h.06 V?.47 | M?.004 P.04 H.36 h.04 V?.001 | – |
| DDGemb | M.60 P.40 | – | – | M.33 P.17 H.03 h.003 | P.71 H.02 | M.92 P.008 H.02 h.001 D | M.63 P.21 H.02 | M.44 P.23 H.04 h.004 | M.08 P.21 H.03 h.01 | m.48 P.32 H.03 h.003 | – | h.02 | M<.001 P.03 H.17 h.07 | – |
| DDMut | M.88 P.12 | M.15 P.24 H.004 h.004 | M.10 P.19 H.02 h.004 | M.74 P.14 H.03 | M.61 P.35 H.005 | M.98 P.01 H.007 D | M.84 P.11 H.01 | M.69 P.21 H.03 h.002 D | M.21 P.26 H.05 | M.80 m.11 P.08 H.02 | – | – | – | – |

* `ESM-MSR & capped retrains` covers ESM-MSR, MutateEverything, ThermoMPNN, and SPURS retrained on the Homology-Restricted MegaScale split (`splits_oct06_capped.pkl`). Note that all held-out evaluations of Domainome (436/522 domains) and ProteinGym (173/213 scored assays) in the main text remove homologs to Train under the four-part rule; this table audits the full unfiltered benchmarks for completeness.
**Overlap tier codes** (fraction of benchmark rows affected, e.g. `M.91` = 91%, `M1` = 100%): `M` = exact mutation in training set (`#` = test set equals training set); `m` = constituent single mutation of a multi-mutant in training set; `P` = same protein (≥95% identity over ≥90% coverage); `H` = close homolog (MMseqs2 E ≤ 1e-3 or Foldseek qTM > 0.70 over ≥50% coverage); `h` = distant homolog (E ≤ 1e-2 or identity ≥ 30% over ≥30% coverage); `V` = validation/calibration overlap; `–` = no detected overlap.
