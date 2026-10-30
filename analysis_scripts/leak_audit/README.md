# Leak-audit scripts (October 2026)

These scripts produced [`docs/LEAKAGE_AUDIT_OCT2026.md`](../../docs/LEAKAGE_AUDIT_OCT2026.md),
[`docs/BENCHMARKS_OCT2026.md`](../../docs/BENCHMARKS_OCT2026.md) §6, and the machine-readable tables under
[`docs/benchmarks/`](../../docs/benchmarks/): `leak_audit_matrix.csv`, `leak_audit_inventories.csv`,
`leak_audit_pg_per_dms.csv`, `leak_audit_homology.csv`, `leak_audit_code_exact.csv`, `split_homology_audit.json`,
`leak_audit.sqlite`.

## Layout and inputs

The scripts were run from the audit workspace (`LEAK_ROOT`, default `/home/sareeves/playground/esm-msr-devel`),
with inputs under `leak_audit/data/` and outputs under `leak_audit/out/`. `13_matrix.py` reads
`docs/benchmarks/all_by_all_long.csv` of the checkout it is in (override with `LONG_CSV`); `16_render.py` and `17_write_doc.py` still read the
devel-revision worktree at `repo/.claude/worktrees/devel-main-conservative-merge-5e554f` (for `docs/BENCHMARKS_OCT2026.md` and the repo `data/`).
They also read the public training files listed in the report's §3,
the local model clones (`~/software/{SPURS_mod,EsmTherm,ThermoMPNN,esm-msr}`), and the split's saved pair table
under `analysis/salvaged_2026-10-16/structure-homology-splits-e5ca04/`.

External inputs needed before a full re-run (all public; URLs in the report §3): `S2450.tsv/.fasta`,
`S9028.csv`, `SM1242.csv`, `S8754.csv`, the IFUM `dataset_splits.pdf`, the EsmTherm `wildtype_split.csv`,
the MGnify `mgnify_training_index.csv`, the ProteinGym DMS files, the Domainome supplementary table.

## Order

```
01_megascale.py            # Megascale library/variant tables
02_testsets.py             # test rows -> (protein, mutation) for every dataset
05_proteins.py             # protein corpus + MMseqs2 all-vs-all
05b_inventory_proteins.py  # PDB sequences for the code/position inventories
05c_tmpnnd_inventory.py    # the capped-train libraries ThermoMPNN-D can see (measured doubles + Rosetta-augmented singles)
06_families.py             # same-protein families + representative position maps
07_relations.py            # test-protein relation to the capped train set (early check)
09_homology_audit.py       # split rule, sensitivity, transitive/structure-only analysis
10_overlap_engine.py       # per-inventory exact/constituent/same-protein counts
11_homology_classes.py     # P/H/h classification
12_code_exact.py           # exact counts for DDGemb/DDMut/GeoDDG
13_matrix.py               # cell codes and fractions
14_pg.py                   # per-DMS ProteinGym leakage + clean aggregates
15_mgnify.py               # ESM3dG MGnify corpus homology (needs the mgnify .m8)
16_render.py               # annotated table fragments (reads the ORIGINAL BENCHMARKS doc)
17_write_doc.py            # splices the tables and §6 into docs/BENCHMARKS_OCT2026.md
18_db.py                   # SQLite + committed CSVs
19_esm3dg_exact.py         # exact WT/mutant SHA-1 join against the MGnify train split
```

`16_render.py` and `17_write_doc.py` must be run against the unmodified `BENCHMARKS_OCT2026.md`; check the file out
of `devel-revision` before re-running them. They regenerate whole tables and §6, so they were not rerun for the retrained-SPURS update
(2026-10-29): there the regenerated `leak_matrix.csv` was used to patch the doc's tables in place, which leaves later hand edits alone.

Every model retrained on the capped split (ESM-MSR, ME stage 1/2, ThermoMPNN, SPURS, SPURS multi) shares one protein set in `11_homology_classes.py`
(so `P`/`H`/`h` are identical for them); ThermoMPNN-D uses the subset from `05c_tmpnnd_inventory.py`. The cell codes are deterministic given the inputs; `13_matrix.py` also
needs `ifum_train_proteins.csv` and `esmtherm_train_proteins.csv` from the inventory data.

## ESM3dG training-corpus check (October 30 2026)

`19_esm3dg_exact.py` does the exact check: a SHA-1 join of every benchmark row's wild-type and mutant sequence
against the 960,216-sequence MGnify train split (`data/raw/mgnify_index.csv`, `split == 'train'`). It returns 0
matches for every dataset. `15_mgnify.py` then classifies homology from

    mmseqs easy-search data/raw/test_queries.fa data/raw/mgnify_train.fa data/raw/test_vs_mgnify_s75.m8 \
        /tmp/mm_tmp -s 7.5 -e 10 --max-seqs 500 --threads 8 \
        --format-output query,target,fident,alnlen,qlen,tlen,evalue,bits

which takes 9.7 s for all 2,583 test proteins (1.5 GB RSS) and 4.7-5.8 s for the 33 capped test libraries. The
older `test_vs_mgnify.m8` (s = 5.5, max-seqs 50) is used only if the s75 file is absent.
