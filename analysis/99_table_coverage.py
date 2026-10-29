#!/usr/bin/env python
"""Check that every table of BENCHMARKS_OCT2026.md appears in at least one figure of ``analysis/results``.

``TABLE_FIGURES`` is the map (table id as in ``reference/benchmark_tables/tables_index.csv`` -> figures that draw it).  The script
fails if a table of the index is missing from the map or a mapped figure file does not exist, and writes ``results/table_figure_map.csv``.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
T = '02_thermostability/'
TABLE_FIGURES = {
    '1.1.1': [T + 'thermo_all_models', T + 'thermo_vs_retrains', T + 'thermo_scatter_external'],
    '1.1.2': ['10_leakage_and_reference/leak_matrix_external', T + 'thermo_all_models'],
    '1.2.1': ['03_tsuboyama/tsuboyama_overview', '03_tsuboyama/tsuboyama_per_library', '03_tsuboyama/tsuboyama_singles_doubles'],
    '1.2.2': ['10_leakage_and_reference/leak_matrix_other', '03_tsuboyama/tsuboyama_overview'],
    '1.3.1': ['04_dms_domainome/dms_vs_retrains', '04_dms_domainome/dms_sigma_vs_zero_shot'],
    '1.3.2': ['10_leakage_and_reference/leak_matrix_other'],
    '1.3.3': ['04_dms_domainome/dms_sigma_sweep', '04_dms_domainome/dms_sigma_vs_zero_shot'],
    '1.4.1': ['05_proteingym/pg_categories', '05_proteingym/pg_table_heatmap'],
    '1.4.2': ['10_leakage_and_reference/leak_matrix_other'],
    '1.5.1': ['10_leakage_and_reference/published_reference'],
    '1.6.1': ['03_tsuboyama/tsuboyama_overview', '10_leakage_and_reference/spurs_published_vs_retrained'],
    '1.6.2': [T + 'thermo_vs_retrains', '10_leakage_and_reference/spurs_published_vs_retrained'],
    '1.6.3': ['04_dms_domainome/dms_vs_retrains', '10_leakage_and_reference/spurs_published_vs_retrained'],
    '2.0.1': ['06_epistasis/epistasis_authoritative_metrics'],
    '2.0.2': ['06_epistasis/epistasis_authoritative_metrics'],
    '2.1.1': ['06_epistasis/epistasis_hierarchy_val', '06_epistasis/epistasis_naive_recomputed'],
    '2.2.1': ['06_epistasis/epistasis_hierarchy_test', '06_epistasis/epistasis_naive_recomputed', '06_epistasis/epistasis_scatter_dddG'],
    '2.3.1': ['06_epistasis/ddg_ranking', '06_epistasis/epistasis_scatter_ddG_doubles'],
    '2.5.1': ['06_epistasis/nobb_ablation'],
    '2.6.1': ['06_epistasis/lambda_scan'],
    '3.1': ['05_proteingym/pg_categories', '05_proteingym/pg_table_heatmap'],
    '3.2': ['05_proteingym/pg_per_dms_scatter'],
    '6.2.1': ['10_leakage_and_reference/leak_audit_text_tables', '10_leakage_and_reference/leak_matrix_external'],
    '6.3.1': ['10_leakage_and_reference/leak_audit_text_tables'],
    '6.5.1': ['10_leakage_and_reference/leak_overlap_tsuboyama_test', '10_leakage_and_reference/leak_audit_text_tables'],
    '6.5.2': ['10_leakage_and_reference/clean_means'],
}
NOT_A_RESULT = {'7.1': 'file-location table (cluster paths); replaced by the "where things are" section of analysis/README.md'}


def main():
    idx = pd.read_csv(HERE / 'reference' / 'benchmark_tables' / 'tables_index.csv', dtype=str)
    rows, bad = [], []
    for r in idx.itertuples():
        if r.table in NOT_A_RESULT:
            rows.append(dict(table=r.table, heading=r.heading, status='not a result table', figures=NOT_A_RESULT[r.table]))
            continue
        figs = TABLE_FIGURES.get(r.table)
        if not figs:
            bad.append(f'table {r.table} ({r.heading}) has no figure')
            continue
        miss = [f for f in figs if not (RES / (f + '.png')).exists()]
        if miss:
            bad.append(f'table {r.table}: missing {miss}')
        rows.append(dict(table=r.table, heading=r.heading, status='drawn' if not miss else 'MISSING', figures='; '.join(figs)))
    pd.DataFrame(rows).to_csv(RES / 'table_figure_map.csv', index=False)
    print(f'{sum(r["status"] == "drawn" for r in rows)} of {len(idx)} tables drawn; {len(NOT_A_RESULT)} not a result table')
    for b in bad:
        print('PROBLEM:', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
