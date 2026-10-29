"""Score the output directory of a baseline run (inference_scripts/baselines/*_inference.py, tracked or not) with ESM-MSR's own metric code.

Reads tsuboyama_{val,test}.csv, external_<set>.csv, dms_<assay>.csv, domainome.csv and proteingym/summary.csv under --pred_dir, whatever exists,
and writes one summary CSV (one row per benchmark, ESM-MSR's update_stats columns). Every prediction column is stabilizing-positive, like ddG_ML.

  python analysis_scripts/score_baseline_runs.py --pred_dir outputs/tracked/spurs_retrain_s1 --label SPURS_s1 --out scored/ \\
      --col SPURS_single_additive --epi_col none                        # SPURS single (an additive model: predicted epistasis 0)
  python analysis_scripts/score_baseline_runs.py ... --col SPURS_multi --epi_col SPURS_multi_epi --pgym_col spearman_SPURS_multi_additive

Tsuboyama is scored on measured rows only (cens == 0): the inference scripts predict the bounded rows too and flag them in `cens`, and
including them inflates SPURS by about 0.005-0.007. Domainome is the mean over domains. ProteinGym is summarised as in benchmark_matrix.py:
PG avg = mean of the five category means, PG Stability = the Stability category, PG DMS mean = plain mean over the 213 common DMS
(the four in PG_DROP are left out); the per-DMS Spearman comes from proteingym/summary.csv (--pgym_col) and the categories from
--pgym_categories (outputs/pgym/pgym_all_models_per_dms.csv).

update_stats is taken verbatim from inference_scripts/esm_msr_testing.py (only the function definitions are executed, not the inference
stack), so the metric definitions cannot drift from the ESM-MSR rows.
"""
import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PG_DROP = {'BRCA1_HUMAN_Findlay_2018', 'SCN5A_HUMAN_Glazer_2019', 'POLG_CXB3N_Mattenberger_2021', 'BRCA2_HUMAN_Erwood_2022_HEK293T'}


def load_update_stats():
    import sys
    sys.path.insert(0, os.path.join(REPO, 'src'))
    from esm_msr import stats
    path = os.path.join(REPO, 'inference_scripts', 'esm_msr_testing.py')
    tree = ast.parse(open(path).read())
    keep = {'safe_spearman', 'safe_ndcg', 'update_stats'}
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep], type_ignores=[])
    ns = {'pd': pd, 'np': np, 'stats': stats}
    exec(compile(mod, path, 'exec'), ns)
    assert keep <= set(ns), f'esm_msr_testing.py no longer defines {sorted(keep - set(ns))}'
    probe = pd.DataFrame({'mut_type': ['A1G', 'A1G:C2D', 'E3K'], 'code': ['X'] * 3, 'ddG': [0.1, 0.5, -0.2], 'pred': [0.2, 0.4, -0.1],
                          'dddG_ML': [np.nan, 0.1, np.nan], 'pred_dddg': [np.nan, 0.2, np.nan]})
    ns['update_stats'](pd.DataFrame(), 'probe', probe, 'ddG', 'pred', 'dddG_ML', 'pred_dddg')      # a helper missing from `keep` fails here, not an hour in
    return ns['update_stats']


def score_tsuboyama(update_stats, path, col, epi_col):
    df = pd.read_csv(path, low_memory=False)
    if 'cens' in df.columns:
        df = df[df['cens'] == 0]
    is_double = df['mut_type'].str.contains(':')
    df['pred_dddg'] = np.where(is_double, df[epi_col] if epi_col else 0.0, np.nan)
    stats = pd.DataFrame()
    for code_wt, g in df.groupby('code_wt'):
        stats = update_stats(stats, code_wt, g, 'ddG_ML', col, 'dddG_ML', 'pred_dddg')
    return stats.mean(axis=0, numeric_only=True)


def pg_summary(summary_csv, col, categories_csv):
    s = pd.read_csv(summary_csv).set_index('DMS_id')[col]
    cat = pd.read_csv(categories_csv).set_index('DMS_id')['category']
    keep = cat.index[~cat.index.isin(PG_DROP)]
    s = s.reindex(keep)
    if s.isna().any():
        raise SystemExit(f'{int(s.isna().sum())} of the {len(keep)} ProteinGym DMS have no {col} in {summary_csv}')
    g = s.groupby(cat.reindex(keep)).mean()
    out = {'PG avg': g.mean(), 'PG Stability': g['Stability'], 'PG DMS mean': s.mean(), 'n_dms': len(keep)}
    out.update({f'PG category {c}': v for c, v in g.items()})                 # the five category means behind PG avg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out', required=True, help='directory for <label>_summary.csv')
    ap.add_argument('--label', required=True)
    ap.add_argument('--col', required=True, help='prediction column, e.g. SPURS_single_additive, SPURS_multi, pred_combined')
    ap.add_argument('--epi_col', default='none', help="predicted-epistasis column for Tsuboyama doubles, or 'none' (additive model: 0)")
    ap.add_argument('--pgym_col', default=None, help='per-DMS Spearman column of proteingym/summary.csv (default: spearman_<col>)')
    ap.add_argument('--pgym_categories', default=os.path.expanduser('~/esm-msr-devel/outputs/pgym/pgym_all_models_per_dms.csv'))
    a = ap.parse_args()
    epi_col = None if a.epi_col == 'none' else a.epi_col
    update_stats = load_update_stats()
    os.makedirs(a.out, exist_ok=True)
    rows = {}
    for split in ('val', 'test'):
        p = os.path.join(a.pred_dir, f'tsuboyama_{split}.csv')
        if os.path.exists(p):
            rows[f'tsuboyama_{split}_all'] = score_tsuboyama(update_stats, p, a.col, epi_col)
    for kind in ('external', 'dms'):
        for p in sorted(glob.glob(os.path.join(a.pred_dir, f'{kind}_*.csv'))):
            name = os.path.basename(p)[len(kind) + 1:-4]
            df = pd.read_csv(p, low_memory=False)
            st = update_stats(pd.DataFrame(), name, df, 'ddG', a.col)
            rows[f'{kind}_{name}'] = st.iloc[0]
    p = os.path.join(a.pred_dir, 'domainome.csv')
    if os.path.exists(p):
        df = pd.read_csv(p, low_memory=False)
        st = pd.DataFrame()
        for code, g in df.groupby('code'):
            st = update_stats(st, code, g, 'ddG', a.col)
        rows['domainome_mean_over_domains'] = st.mean(axis=0, numeric_only=True)
    out = pd.DataFrame(rows).T
    pg = os.path.join(a.pred_dir, 'proteingym', 'summary.csv')
    if os.path.exists(pg):
        for k, v in pg_summary(pg, a.pgym_col or f'spearman_{a.col}', a.pgym_categories).items():
            out.loc[f'proteingym_{k}', 'spearman_all'] = v
    out.insert(0, 'label', a.label)
    path = os.path.join(a.out, f'{a.label}_summary.csv')
    out.to_csv(path, float_format='%.6f')
    print(f'wrote {path}')
    print(out[[c for c in ('spearman_all', 'spearman_singles', 'spearman_doubles', 'n_singles', 'n_doubles') if c in out.columns]].to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
