# Held-out filter audit of the analysis figures (Nov 18)

**Rule.** Every benchmark that is scored as held-out must be filtered the same way: a protein is dropped when it is homologous to the training set under the four-part rule that built the
Homology-Restricted MegaScale split. Any one of: (1) MMseqs2 E-value <= 1e-3, (2) identity > 0.40 (BLOSUM62 local alignment, matches / shorter length, every pair), (3) Foldseek E-value <= 1e-3,
(4) Foldseek TM-score > 0.70 (shorter-chain normalised). Training only (validation is not used). The four parts are identified in `docs/SPLITTING_AND_HOMOLOGY_AUDIT.md` (sections 1.2 and 2.1) and in
`results/00_supplementary_figs/split_construction/split_best_partner_capped` (panels = MMseqs2 E-value and identity, Foldseek E-value and TM-score) and
`results/00_supplementary_figs/holdout_filter` (how many proteins each part removes from each benchmark).

**Implementation.** `leak_audit/21_tsuboyama_rule.py` -> `reference/tsuboyama_rule_flags.csv` (the four values, four flags and the letters E, I, S, T per protein) -> `msr_analysis/heldout.py`
(`excluded(dataset)`), applied at the loaders (`data.load_frame('domainome')`, `final_runs.load_final_frame('domainome')`, `mgnify.load_mgnify`, `pgsigma.train_overlap`) so a script cannot forget it.
E-values depend on database size, so benchmark proteins are searched against the split's own 1,147-protein search corpus and only hits to training libraries are kept (reverse direction rescaled by
database size). **Check:** the validation and test libraries run through the same code against the training libraries are all unflagged (0/23, 0/33; max identity 0.395, max TM 0.696).

| benchmark | proteins | removed | held-out | by part (E / identity / Foldseek E / TM) |
|---|---|---|---|---|
| ProteinGym assays | 217 | 41 | 176 | 24 / 33 / 28 / 31 |
| Domainome domains | 522 | 86 | 436 | 48 / 37 / 62 / 78 |
| MGnify scaffolds | 180,367 | 27,687 | 152,680 | 3,953 / 3,964 / 14,738 / 21,707 |
| Rocklin designs | 1,000 | 258 | 742 | 2 / 226 / 7 / 40 |

The earlier flags were sequence-only for MGnify and Rocklin, used MMseqs2's default alignment-column identity (not the split's whole-protein identity), and the leak audit's P / H classes for
ProteinGym (Stability only) and Domainome. 17,000 MGnify scaffolds are removed by TM-score alone: structure matters.

## Script by script

| script / figures | scores | before | now |
|---|---|---|---|
| `05_proteingym.py` (pg_summary*, pg_per_dms, pg heat map) | ProteinGym, Domainome | base tables: Stability P/H only (26 assays) + Domainome P/H 64; parallel `_tsu` tables with the older rule | one filter, all categories: 173 of 213 assays, 436 domains; `_tsu` files kept as identical copies |
| `01_summary.py` (figure3, pg_categories*) | ProteinGym, Domainome, MGnify, Rocklin | `_tsu` tables (older rule); MGnify / Rocklin sequence-only | canonical flags for all four |
| `18b_mgnify_models.py`, `18c_mgnify_by_mutation_count.py` | MGnify, Rocklin | sequence-only flags | canonical flags (also through `load_mgnify`) |
| `proteingym/p0_sigma/*`, `combined_vs_wt/*`, `ablation_vs_p0/02,06` | ProteinGym | only the 23 same-protein Stability assays (`sameprot`) | canonical (`pgsigma.train_overlap`) |
| `04_dms_domainome.py`, `01_core_rho.py`, `msr_analysis/external.py` | Domainome | all 522 | held-out (loader; boot cache version 3) |
| `08_screening.py`, `16_screening_overview.py`, `17_gain_per_variant_fit.py` | Domainome | P/H 64 | canonical |
| `pg_activity/02_activity_screening.py` | ProteinGym Activity | all 43 | held-out Activity assays (5 removed) |
| `ablation_vs_p0/01_paired_vs_p0.py`, `98_final_ablation_table.py` | Domainome, ProteinGym | unfiltered base columns + "low-id" extra columns (P/H) | held-out everywhere; the low-id columns now equal the plain ones |
| `scoring/summarize_spurs_runs.py` | Domainome | P/H 64 | canonical |
| `proteingym/sigma_tradeoff/scripts/common.py` | ProteinGym (baseline sigma sweep) | all 214 | filter added in `ref_table`; **pipeline not rerun**: `REPORT.md` prose is for the 214-assay run, rerunning changes its numbers |
| `figure3` panel c, `p0_sigma/03_fig_property_summary` (excluded) | ProteinGym | held-out assays only (the earlier black version marked a different TM / identity leak test) | every assay drawn (176 of 217 held out), the ones excluded by the four-part rule grey (smaller markers; the included assays in a more saturated hue) and outside the box, median, mean, header counts (n held out of n drawn) and p-values |
| `18_mgnify.py` | MGnify, Rocklin | all scaffolds, stratified by homology class | **unfiltered by design** (`load_mgnify(heldout=False)`): it is the analysis of how homology changes the result |
| thermostability sets, functional DMS (GRB2, DLG4, EstA, Myo, GB1), Tsuboyama capped val / test | | quarantined / held out by the split against training and validation | unchanged |
| `tsuboyama/mega_splits/*` | mega_splits test | not homology held-out (documented) | unchanged |
| `scoring/*`, `leak_audit/04-19` | prediction collection, the original leak audit (P / H classes for `docs/BENCHMARKS_*`) | n/a | unchanged; `00_validate_against_tables.py` compares with BENCHMARKS tables that still use the older Domainome / ProteinGym columns, so its Domainome and ProteinGym cells will no longer match until those tables are regenerated |

## Reruns (Nov 18)

05, p0_sigma (01, 02, 03, 04, 05), 04, 08, pg_activity/02, 16, 17, 09, combined_vs_wt (all), ablation_vs_p0 (all), 98, 18b, 18c, 22 + 20b (split-set matrices), 01_summary, 99, 01_core_rho
(started last), tests (404 passed). Not rerun: `sigma_tradeoff` (above), `00_validate_against_tables.py`, the BENCHMARKS documents.
