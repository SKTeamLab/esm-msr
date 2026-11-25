"""Stage 12 — Benchmark table validation, 27-table figure coverage audit, and results manifest.

Consolidates:
  - ``00_parse_benchmark_tables.py``: optional ``--reparse`` extraction of ``docs/BENCHMARKS_OCT2026.md`` into
    ``reference/benchmark_tables/T*.csv`` and ``tables_index.csv``
  - ``00_validate_against_tables.py``: cell-by-cell validation of ``msr_analysis.data`` + ``DatasetBoot`` against
    ``reference/all_by_all_long.csv`` -> ``results/00_for_reviewers/reference/validation_vs_benchmarks.csv``
  - ``99_table_coverage.py``: verifies every benchmark table in ``tables_index.csv`` is represented by active,
    existing figures on disk (both ``.png`` and ``.pdf``) -> ``results/table_figure_map.csv`` + ``results/manifest_outputs.csv``

Bug fixes & scientific improvements:
  1. Updates ``TABLE_FIGURES`` across all 27 benchmark tables to map to active figures in the canonical dataset-first
     ``results/`` hierarchy (``01_megascale/epistasis/*``, ``00_main_text_figs/*``, ``00_supplementary_figs/*``,
     ``00_for_reviewers/*``, and ``99_misc/dms3/*``).
  2. Resolves the apparent ``+0.036`` discrepancy on ``domainome`` in ``validation_vs_benchmarks.csv`` by recording BOTH
     ``rho`` (the canonical 436-domain held-out Domainome used across all manuscript figures) and ``rho_unfiltered``
     (the pre-holdout 522-domain Domainome matching the legacy ``BENCHMARKS_OCT2026.md`` column to ``< 0.002``).
  3. Writes ``results/manifest_outputs.csv`` inventorying all generated figures and tables across ``results/``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from msr_analysis.boot import DatasetBoot
from msr_analysis.data import load_frame
from msr_analysis.paths import (
    MANIFEST_OUTPUTS_CSV,
    REFERENCE,
    REPO_ROOT,
    RESULTS,
    TABLE_FIGURE_MAP_CSV,
    VALIDATION_VS_BENCHMARKS_CSV,
    atomic_to_csv,
)
from msr_analysis.registry import ALL_DATASETS, PER_PROTEIN
from pipelines import base_parser

STAGE = 'manifest'
DESCRIPTION = 'Benchmark table validation, 27-table figure coverage verification, and results manifest.'

# Reference model labels in reference/all_by_all_long.csv -> (label, n_seeds)
REF_MODELS = {
    'msr:comb': ('ESM-MSR (released)', 3),
    'msr:wt': ('ESM-MSR (released) WT', 3),
    'msrdev:comb': ('ESM-MSR (devel)', 3),
    'msrdev:wt': ('ESM-MSR (devel) WT', 1),
    'msrrel:comb': ('ESM-MSR (released)', 1),
    'msrrel:wt': ('ESM-MSR (released) WT', 1),
    'msrrel@0:comb': ('ESM3-small-open (σ=0, MSR readout)', 1),
    'me1': ('ME stage1', 3),
    'me2': ('ME stage2', 3),
    'me1+': ('ME stage1+', 3),
    'me2+': ('ME stage2+', 3),
    'tmpnn': ('ThermoMPNN', 3),
    'tmpnnd': ('ThermoMPNN+D', 3),
    'spurs': ('SPURS (retrained)', 3),
    'spursm': ('SPURS multi (retrained)', 3),
    'esmtherm': ('ESMtherm', 1),
    'ddgemb': ('DDGemb', 1),
    'ifum_wt': ('IFUM (WT backbone)', 1),
    'ifum_esmfold': ('IFUM (native, ESMFold)', 1),
    'geoddg': ('GeoDDG-3D', 1),
    'esm3dg': ('ESM3dG', 1),
    'prosst': ('ProSST', 1),
    'pmpnn': ('ProteinMPNN', 1),
    'esm3s': ('ESM3-small zero-shot', 1),
    'esm3m': ('ESM3-medium zero-shot', 1),
    'esm3l': ('ESM3-large zero-shot', 1),
}
for _sig in ('0.25', '0.5', '0.75'):
    REF_MODELS[f'msrdev@{_sig}:comb'] = (f'ESM-MSR (devel) σ={_sig}', 3)
    REF_MODELS[f'msrrel@{_sig}:comb'] = (f'ESM-MSR (released) σ={_sig}', 1)

DS_REF = {
    'tsuboyama_capped_val': 'tsu_capped_val',
    'tsuboyama_capped_test': 'tsu_capped_test',
}

# Canonical active figure mapping for all 27 tables of BENCHMARKS_OCT2026.md
TABLE_FIGURES = {
    '1.1.1': ['02_thermostability/thermo_external_rho', '02_thermostability/thermo_overview', '06_summary/thermostability_sets', '00_main_text_figs/figure3'],
    '1.1.2': ['00_supplementary_figs/split_construction/split_max_identity_heatmap', '00_supplementary_figs/split_construction/split_max_tm_heatmap', '02_thermostability/thermo_external_rho', '02_thermostability/thermo_overview'],
    '1.2.1': ['01_megascale/megascale_overview', '01_megascale/megascale_per_library', '01_megascale/megascale_singles_doubles', '00_main_text_figs/figure3'],
    '1.2.2': ['00_supplementary_figs/split_construction/split_best_partner_capped', '00_supplementary_figs/split_construction/mega_clean_vs_homologous_test', '01_megascale/megascale_overview'],
    '1.3.1': ['04_domainome/domainome_vs_retrains', '04_domainome/domainome_sigma_vs_zero_shot', '99_misc/dms3/dms_vs_retrains', '99_misc/dms3/dms_sigma_vs_zero_shot', '06_summary/pg_categories', '00_main_text_figs/figure3'],
    '1.3.2': ['00_supplementary_figs/holdout_filter', '99_misc/dms3/dms_vs_retrains'],
    '1.3.3': ['04_domainome/domainome_sigma_sweep', '04_domainome/domainome_sigma_vs_zero_shot', '99_misc/dms3/dms_sigma_sweep', '99_misc/dms3/dms_sigma_vs_zero_shot'],
    '1.4.1': ['06_summary/pg_categories', '03_proteingym/pg_table_heatmap', '03_proteingym/pg_overview', '00_main_text_figs/figure3'],
    '1.4.2': ['00_supplementary_figs/holdout_filter', '03_proteingym/pg_overview'],
    '1.5.1': ['00_for_reviewers/reference/published_reference'],
    '1.7.1': ['05_mgnify/mgnify_overview', '05_mgnify/mgnify_per_scaffold', '05_mgnify/mgnify_scatter', '06_summary/pg_categories', '00_main_text_figs/figure3'],
    '1.7.2': ['05_mgnify/mgnify_homology', '00_supplementary_figs/holdout_filter'],
    '1.7.3': ['05_mgnify/mgnify_homology'],
    '1.7.4': ['05_mgnify/rocklin_overview', '06_summary/pg_categories', '00_main_text_figs/figure3'],
    '2.0.1': ['01_megascale/epistasis/epistasis_levels_bars_val', '01_megascale/epistasis/epistasis_naive_recomputed'],
    '2.0.2': ['01_megascale/epistasis/epistasis_levels_bars_test', '00_main_text_figs/epistasis_main_composite'],
    '2.1.1': ['01_megascale/epistasis/epistasis_levels_bars_val', '01_megascale/epistasis/epistasis_naive_recomputed'],
    '2.2.1': ['01_megascale/epistasis/epistasis_levels_bars_test', '00_main_text_figs/epistasis_main_composite', '01_megascale/epistasis/epistasis_scatter_dddG'],
    '2.3.1': ['01_megascale/epistasis/ddg_ranking', '01_megascale/epistasis/epistasis_scatter_ddG_doubles'],
    '2.5.1': ['00_for_reviewers/baseline_nobb_ablation', '09_ablation/ablation_architecture_suite'],
    '2.6.1': ['01_megascale/epistasis/lambda_scan', '99_misc/dms3/epistasis_lambda_scan'],
    '3.1': ['06_summary/pg_categories', '03_proteingym/pg_table_heatmap', '03_proteingym/pg_overview'],
    '3.2': ['03_proteingym/pg_per_dms_scatter', '03_proteingym/pg_property_sigma_summary'],
    '6.2.1': ['00_supplementary_figs/holdout_filter', '00_supplementary_figs/split_construction/split_max_identity_heatmap', '00_supplementary_figs/split_construction/split_max_tm_heatmap'],
    '6.3.1': ['00_supplementary_figs/holdout_filter', '00_supplementary_figs/split_construction/split_best_partner_capped'],
    '6.5.1': ['00_supplementary_figs/holdout_filter', '00_supplementary_figs/split_construction/mega_clean_vs_homologous_test'],
    '6.5.2': ['02_thermostability/thermo_external_rho', '02_thermostability/thermo_overview'],
}

FOLDER_OWNER = {
    '(root)': 'p12_manifest',
    '00_for_reviewers': 'p02_tsuboyama / p10_splits_and_leakage / p12_manifest',
    '00_main_text_figs': 'p05_summary_main / p06_epistasis_hierarchy / p09_screening',
    '00_supplementary_figs': 'p02 / p03 / p06 / p07 / p09 / p10',
    '01_megascale': 'p02_tsuboyama / p06_epistasis_hierarchy / p07_epistasis_mechanisms / p09_screening / p10_splits_and_leakage',
    '02_thermostability': 'p01_benchmarks_external / p08_calibration_length / p09_screening',
    '03_proteingym': 'p03_proteingym / p09_screening',
    '04_domainome': 'p01_benchmarks_external / p09_screening',
    '05_mgnify': 'p04_mgnify_rocklin',
    '06_summary': 'p05_summary_main',
    '07_screening_overview': 'p09_screening',
    '08_sequence_length': 'p08_calibration_length',
    '09_ablation': 'p11_ablations_and_heads',
    '10_inference_resources': 'scoring/resources',
    '99_misc': 'p01 / p03 / p05 / p06 / p07 / p08 / p11 / studies/*',
}


def reparse_benchmark_tables(md_path: Path) -> None:
    if not md_path.exists():
        raise FileNotFoundError(f'Benchmark markdown not found: {md_path}')
    lines = md_path.read_text(encoding='utf-8').splitlines()
    out = REFERENCE / 'benchmark_tables'
    out.mkdir(parents=True, exist_ok=True)
    sec, title, count = '0', '', {}
    i, index = 0, []
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^(#{2,4})\s+(\d+(?:\.\d+)?)?\.?\s*(.*)', ln)
        if m:
            if m.group(2):
                sec = m.group(2)
            title = m.group(3).strip()
        if ln.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|?$', lines[i + 1]):
            hdr = [c.strip() for c in ln.strip().strip('|').split('|')]
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].startswith('|'):
                r = [c.strip() for c in lines[j].strip().strip('|').split('|')]
                r = (r + [''] * len(hdr))[: len(hdr)]
                rows.append(r)
                j += 1
            count[sec] = count.get(sec, 0) + 1
            tid = f'{sec}.{count[sec]}'
            hdr = [h or f'col{k}' for k, h in enumerate(hdr)]
            atomic_to_csv(pd.DataFrame(rows, columns=hdr), out / f'T{tid}.csv', index=False)
            index.append(dict(table=tid, section=sec, heading=title, columns=' | '.join(hdr)[:160], rows=len(rows)))
            i = j
            continue
        i += 1
    atomic_to_csv(pd.DataFrame(index), out / 'tables_index.csv', index=False)
    print(f'[p12] Parsed {len(index)} tables from {md_path.name} -> {out}')


def run_validation_vs_benchmarks() -> pd.DataFrame:
    VALIDATION_VS_BENCHMARKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REFERENCE / 'all_by_all_long.csv')

    # Precompute unfiltered Domainome (522 domains) to compare directly against the pre-holdout BENCHMARKS table column
    fr_dom_unfilt = load_frame('domainome', heldout=False)
    boot_dom_unfilt = DatasetBoot(fr_dom_unfilt, list(REF_MODELS), per_protein=True)
    dom_unfilt_pt = {k: boot_dom_unfilt.point(k)[0] for k in boot_dom_unfilt.keys}

    rows = []
    for ds in ALL_DATASETS:
        fr = load_frame(ds)
        b = DatasetBoot(fr, list(REF_MODELS), per_protein=ds in PER_PROTEIN)
        for k in b.keys:
            rho, per_seed, n, ng = b.point(k)
            lab, ns = REF_MODELS[k]
            r = ref[(ref.model == lab) & (ref.dataset == DS_REF.get(ds, ds)) & (ref.n_seeds == ns)]
            doc = float(r['rho'].iloc[0]) if len(r) else np.nan
            rho_unfilt = dom_unfilt_pt.get(k, np.nan) if ds == 'domainome' else rho
            diff_vs_doc = (rho_unfilt - doc) if ds == 'domainome' else (rho - doc)
            note = f'heldout=True ({ng} domains); doc column is unfiltered 522 domains (see rho_unfiltered)' if ds == 'domainome' else ''
            rows.append(dict(
                dataset=ds,
                model=k,
                rho=rho,
                rho_unfiltered=rho_unfilt if ds == 'domainome' else np.nan,
                doc=doc,
                diff=rho - doc,
                diff_comparable=diff_vs_doc,
                n=n,
                n_groups=ng,
                seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                note=note,
            ))

    d = pd.DataFrame(rows)
    atomic_to_csv(d, VALIDATION_VS_BENCHMARKS_CSV, index=False, float_format='%.5f')
    ok = d.dropna(subset=['doc'])
    n_gt = int((ok['diff_comparable'].abs() > 0.005).sum())
    print(
        f"[p12] Validation vs BENCHMARKS: {len(ok)} documented cells; "
        f"|diff_comparable| median {ok['diff_comparable'].abs().median():.5f}, "
        f"max {ok['diff_comparable'].abs().max():.4f} (> 0.005 in {n_gt} cells)"
    )
    return d


def run_table_coverage_and_manifest() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    idx = pd.read_csv(REFERENCE / 'benchmark_tables' / 'tables_index.csv', dtype=str)
    rows, bad = [], []

    abl_suite = '09_ablation/ablation_architecture_suite'
    has_abl_suite = (RESULTS / f'{abl_suite}.png').exists()

    for r in idx.itertuples():
        figs = list(TABLE_FIGURES.get(r.table, []))
        if r.table == '2.5.1' and has_abl_suite and abl_suite not in figs:
            figs.append(abl_suite)
        if not figs:
            bad.append(f'table {r.table} ({r.heading}) has no mapped figure')
            rows.append(dict(table=r.table, heading=r.heading, status='MISSING', figures=''))
            continue
        miss = [
            f for f in figs
            if not (RESULTS / f'{f}.png').exists() or not (RESULTS / f'{f}.pdf').exists()
        ]
        if miss:
            bad.append(f'table {r.table}: missing .png/.pdf for {miss}')
        rows.append(dict(
            table=r.table,
            heading=r.heading,
            status='drawn' if not miss else 'MISSING',
            figures='; '.join(figs),
        ))

    map_df = pd.DataFrame(rows)
    atomic_to_csv(map_df, TABLE_FIGURE_MAP_CSV, index=False)

    # Inventory all generated artifacts under results/
    inv_rows = []
    exts = {'.png', '.pdf', '.csv', '.md', '.tex'}
    for p in sorted(RESULTS.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in exts or p.name.startswith('.'):
            continue
        rel = p.relative_to(RESULTS)
        top_folder = rel.parts[0] if len(rel.parts) > 1 else '(root)'
        st = p.stat()
        inv_rows.append(dict(
            rel_path=str(rel),
            top_folder=top_folder,
            kind=p.suffix.lstrip('.').lower(),
            size_bytes=int(st.st_size),
            owner=FOLDER_OWNER.get(top_folder, 'other'),
        ))
    inv_df = pd.DataFrame(inv_rows)
    atomic_to_csv(inv_df, MANIFEST_OUTPUTS_CSV, index=False)

    n_drawn = int((map_df['status'] == 'drawn').sum())
    print(f'[p12] Table coverage: {n_drawn}/{len(idx)} benchmark tables drawn; {len(inv_df)} output files cataloged in manifest_outputs.csv.')
    for b in bad:
        print(f'[p12] PROBLEM: {b}')
    return map_df, inv_df, bad


def add_args(ap) -> None:
    ap.add_argument(
        '--reparse',
        nargs='?',
        const=str(REPO_ROOT / 'docs' / 'BENCHMARKS_OCT2026.md'),
        default=None,
        help='Optional path to BENCHMARKS_OCT2026.md to re-extract reference/benchmark_tables/T*.csv.',
    )


def run(args=None) -> None:
    if args is None:
        ap = base_parser(STAGE, DESCRIPTION)
        add_args(ap)
        args = ap.parse_args()
    reparse_md = getattr(args, 'reparse', None) or os.environ.get('MSR_REPARSE_BENCHMARKS')
    if reparse_md:
        reparse_benchmark_tables(Path(reparse_md))
    run_validation_vs_benchmarks()
    _map_df, _inv_df, bad = run_table_coverage_and_manifest()
    if bad:
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = base_parser(STAGE, DESCRIPTION)
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == '__main__':
    main()
