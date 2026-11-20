"""Unified prediction & ProteinGym scoring utilities for baseline and multi-seed runs.

Imports core rank/calibration/NDCG metrics directly from ``esm_msr.stats`` (no ``ast.parse`` / ``exec`` on
``inference_scripts/esm_msr_testing.py``) and path/cohort definitions from ``msr_analysis``.

Subcommands:
  python analysis/scoring/score_runs.py baseline --pred_dir ... --out ... --label ... --col ...
  python analysis/scoring/score_runs.py me-pgym --stage {stage1,stage2} [--seeds 1 2 3] [--out ...]
  python analysis/scoring/score_runs.py pgym-public [--out pgym_public_per_dms.csv]
  python analysis/scoring/score_runs.py pgym-rosetta --preds proteingym_cartesian_preds_09_11.csv [--out ...]
  python analysis/scoring/score_runs.py seeded-summary --scored ... --tag ... --predictions ... --source ... --out ...
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / 'src'
_ANALYSIS_DIR = _REPO_ROOT / 'analysis'
for _p in (str(_SRC_DIR), str(_ANALYSIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm_msr.stats import (  # noqa: E402
    compute_ndcg_flexible,
    safe_ndcg_k96,
    safe_ndcg_t0,
    safe_rmse,
    safe_spearman as esm_safe_spearman,
)
from msr_analysis import heldout  # noqa: E402
from msr_analysis.paths import (  # noqa: E402
    MSR_ROOT,
    PGYM_INPUTS_DIR,
    PGYM_PER_DMS_DIR,
    PGYM_PUBLIC_ZIP,
    PGYM_REF_CSV,
    atomic_to_csv,
)
from msr_analysis.proteingym import CATS as _PGYM_CATS, PG_DROP as _PGYM_DROP  # noqa: E402

PG_DROP: set[str] = set(_PGYM_DROP)
CATS: list[str] = list(_PGYM_CATS)


def rho(pred, y) -> float:
    """Spearman correlation over rows with finite predictions and labels."""
    pred_arr = np.asarray(pred, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(pred_arr) & np.isfinite(y_arr)
    if int(ok.sum()) < 2:
        return float('nan')
    return float(esm_safe_spearman(pred_arr[ok], y_arr[ok]))


def safe_df_spearman(df: pd.DataFrame, col1: str, col2: str) -> float:
    """DataFrame wrapper around ``esm_msr.stats.safe_spearman`` after dropping NaN pairs."""
    valid_df = df[[col1, col2]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    return float(esm_safe_spearman(valid_df[col1].to_numpy(dtype=float), valid_df[col2].to_numpy(dtype=float)))


def safe_df_rmse(df: pd.DataFrame, pred_col: str, true_col: str) -> float:
    """DataFrame wrapper around ``esm_msr.stats.safe_rmse``."""
    valid_df = df[[pred_col, true_col]].dropna()
    if valid_df.empty:
        return float('nan')
    return float(safe_rmse(valid_df[pred_col].to_numpy(dtype=float), valid_df[true_col].to_numpy(dtype=float)))


def safe_df_ndcg(df: pd.DataFrame, pred_col: str, true_col: str, *, top_n: int | None = None, threshold: float | None = None) -> float:
    """DataFrame wrapper around ``esm_msr.stats.{safe_ndcg_k96,safe_ndcg_t0,compute_ndcg_flexible}``."""
    valid_df = df[[pred_col, true_col]].dropna()
    if len(valid_df) < 2:
        return float('nan')
    preds = valid_df[pred_col].to_numpy(dtype=float)
    truths = valid_df[true_col].to_numpy(dtype=float)
    if top_n == 96 and threshold is None:
        val = safe_ndcg_k96(preds, truths)
        return float(val) if np.isfinite(val) else float('nan')
    if threshold == 0 and top_n is None:
        val = safe_ndcg_t0(preds, truths)
        return float(val) if np.isfinite(val) else float('nan')
    val, _, _, _ = compute_ndcg_flexible(preds.reshape(1, -1), truths.reshape(1, -1), top_n=top_n, threshold=threshold)
    return float(val) if np.isfinite(val) else float('nan')


def update_stats_df(
    stats_df: pd.DataFrame,
    row_name: str,
    res_df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    epi_true_col: str | None = 'dddG_ML',
    epi_pred_col: str | None = None,
    time_val: float | None = None,
) -> pd.DataFrame:
    """Populate ESM-MSR validation summary columns for ``row_name`` using ``esm_msr.stats`` primitives."""
    if pred_col not in res_df.columns:
        print(f"Warning: Skipping stats update for '{row_name}'; column '{pred_col}' not found in predictions.")
        return stats_df

    stats_df.at[row_name, 'spearman_all'] = safe_df_spearman(res_df, true_col, pred_col)
    stats_df.at[row_name, 'rmse_all'] = safe_df_rmse(res_df, pred_col, true_col)

    if 'mut_type' in res_df.columns:
        mut_str = res_df['mut_type'].astype(str)
        is_double = mut_str.str.contains(':', regex=False)
        is_single = ~is_double

        stats_df.at[row_name, 'spearman_singles'] = safe_df_spearman(res_df[is_single], true_col, pred_col)
        stats_df.at[row_name, 'n_singles'] = int(is_single.sum())

        stats_df.at[row_name, 'spearman_doubles'] = safe_df_spearman(res_df[is_double], true_col, pred_col)
        stats_df.at[row_name, 'n_doubles'] = int(is_double.sum())
    else:
        stats_df.at[row_name, 'spearman_singles'] = float('nan')
        stats_df.at[row_name, 'n_singles'] = 0
        stats_df.at[row_name, 'spearman_doubles'] = float('nan')
        stats_df.at[row_name, 'n_doubles'] = 0

    if epi_pred_col and epi_true_col and epi_true_col in res_df.columns and epi_pred_col in res_df.columns:
        stats_df.at[row_name, 'spearman_doubles_epi'] = safe_df_spearman(res_df, epi_true_col, epi_pred_col)
    else:
        stats_df.at[row_name, 'spearman_doubles_epi'] = float('nan')

    stats_df.at[row_name, 'ndcg@96'] = safe_df_ndcg(res_df, pred_col, true_col, top_n=96)
    stats_df.at[row_name, 'ndcg>0'] = safe_df_ndcg(res_df, pred_col, true_col, threshold=0)

    if time_val is not None:
        stats_df.at[row_name, 'time'] = time_val

    return stats_df


def load_update_stats() -> Callable[..., pd.DataFrame]:
    """Return ``update_stats_df`` after running a quick self-check probe."""
    probe = pd.DataFrame({
        'mut_type': ['A1G', 'A1G:C2D', 'E3K'],
        'code': ['X'] * 3,
        'ddG': [0.1, 0.5, -0.2],
        'pred': [0.2, 0.4, -0.1],
        'dddG_ML': [np.nan, 0.1, np.nan],
        'pred_dddg': [np.nan, 0.2, np.nan],
    })
    update_stats_df(pd.DataFrame(), 'probe', probe, 'ddG', 'pred', 'dddG_ML', 'pred_dddg')
    return update_stats_df


def score_tsuboyama(
    update_stats_fn: Callable[..., pd.DataFrame] | None,
    path: str | Path,
    col: str,
    epi_col: str | None,
) -> pd.Series:
    """Score a Tsuboyama prediction CSV on measured rows (``cens == 0``), averaged across ``code_wt``."""
    fn = update_stats_fn or update_stats_df
    df = pd.read_csv(path, low_memory=False)
    if 'cens' in df.columns:
        df = df[df['cens'] == 0].copy()
    else:
        df = df.copy()
    is_double = df['mut_type'].astype(str).str.contains(':', regex=False)
    df['pred_dddg'] = np.where(is_double, df[epi_col] if epi_col else 0.0, np.nan)
    st = pd.DataFrame()
    for code_wt, g in df.groupby('code_wt'):
        st = fn(st, code_wt, g, 'ddG_ML', col, 'dddG_ML', 'pred_dddg')
    return st.mean(axis=0, numeric_only=True)


def pg_summary(
    summary_csv: str | Path,
    col: str,
    categories_csv: str | Path | None = None,
) -> dict[str, float | int]:
    """Summarise per-DMS Spearman into PG avg, PG Stability, PG DMS mean, and category means excluding ``PG_DROP``."""
    cat_path = Path(categories_csv) if categories_csv is not None else (PGYM_PER_DMS_DIR / 'pgym_all_models_per_dms.csv')
    s = pd.read_csv(summary_csv).set_index('DMS_id')[col]
    cat = pd.read_csv(cat_path).set_index('DMS_id')['category']
    keep = cat.index[~cat.index.isin(PG_DROP)]
    s = s.reindex(keep)
    if s.isna().any():
        raise SystemExit(f'{int(s.isna().sum())} of the {len(keep)} ProteinGym DMS have no {col} in {summary_csv}')
    g = s.groupby(cat.reindex(keep)).mean()
    out: dict[str, float | int] = {
        'PG avg': float(g.mean()),
        'PG Stability': float(g['Stability']) if 'Stability' in g.index else float('nan'),
        'PG DMS mean': float(s.mean()),
        'n_dms': int(len(keep)),
    }
    out.update({f'PG category {c}': float(v) for c, v in g.items()})
    return out


def score_baseline_dir(
    pred_dir: str | Path,
    out_dir: str | Path,
    label: str,
    col: str,
    epi_col: str | None = None,
    pgym_col: str | None = None,
    pgym_categories: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Score every benchmark CSV under ``pred_dir`` and atomically write ``<out_dir>/<label>_summary.csv``."""
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    update_fn = load_update_stats()
    rows: dict[str, pd.Series] = {}
    for split in ('val', 'test'):
        p = pred_dir / f'tsuboyama_{split}.csv'
        if p.exists():
            rows[f'tsuboyama_{split}_all'] = score_tsuboyama(update_fn, p, col, epi_col)
    for kind in ('external', 'dms'):
        for p_str in sorted(glob.glob(str(pred_dir / f'{kind}_*.csv'))):
            p = Path(p_str)
            name = p.stem[len(kind) + 1:]
            df = pd.read_csv(p, low_memory=False)
            st = update_fn(pd.DataFrame(), name, df, 'ddG', col)
            rows[f'{kind}_{name}'] = st.iloc[0]
    dom = pred_dir / 'domainome.csv'
    if dom.exists():
        df = pd.read_csv(dom, low_memory=False)
        st = pd.DataFrame()
        for code, g in df.groupby('code'):
            st = update_fn(st, code, g, 'ddG', col)
        rows['domainome_mean_over_domains'] = st.mean(axis=0, numeric_only=True)
    out = pd.DataFrame(rows).T
    pg = pred_dir / 'proteingym' / 'summary.csv'
    if pg.exists():
        for k, v in pg_summary(pg, pgym_col or f'spearman_{col}', pgym_categories).items():
            out.loc[f'proteingym_{k}', 'spearman_all'] = v
    out.insert(0, 'label', label)
    out_path = out_dir / f'{label}_summary.csv'
    atomic_to_csv(out, out_path, index=True, float_format='%.6f')
    return out_path, out


# ------------------------------------------------------------------------------------------------ MutateEverything ProteinGym
def score_me_assay(res: pd.DataFrame, labels: pd.DataFrame | None) -> float:
    """Spearman rho of one MutateEverything ProteinGym assay (upstream ``ddg_pred`` negated, or labelled ``pred_combined``)."""
    if 'pred_combined' in res.columns:
        return rho(res['pred_combined'], res['DMS_score'])
    assert labels is not None, 'labels DataFrame required for unlabelled upstream ME result files'
    j = labels[['mut_type_renumbered', 'DMS_score']].merge(
        res[['mut_info', 'ddg_pred']], left_on='mut_type_renumbered', right_on='mut_info', how='left',
    )
    return rho(-j['ddg_pred'].to_numpy(dtype=float), j['DMS_score'].to_numpy(dtype=float))


def summarise_me_pgym(per_dms: pd.Series, cat: pd.Series) -> dict[str, float | int]:
    """PG avg, Stability, DMS mean, and category means over kept assays (excluding ``PG_DROP``)."""
    keep = [d for d in cat.index if d not in PG_DROP]
    s = per_dms.reindex(keep)
    g = s.groupby(cat.reindex(keep)).mean()
    out: dict[str, float | int] = {c: float(g[c]) for c in CATS if c in g.index}
    out.update({'PG avg': float(g.mean()), 'DMS mean': float(s.mean()), 'n': int(s.notna().sum())})
    return out


def run_me_pgym(
    stage: str,
    seeds: Sequence[int] = (1, 2, 3),
    root: str | Path | None = None,
    out: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score MutateEverything ProteinGym predictions across seeds and print summary."""
    base = Path(root).expanduser() if root is not None else MSR_ROOT
    cat = pd.read_csv(base / 'baselines/pgym_per_dms/pgym_all_models_per_dms.csv').set_index('DMS_id')
    per_dict: dict[int, pd.Series] = {}
    for k in seeds:
        r: dict[str, float] = {}
        for f in sorted(glob.glob(str(base / f'baselines/retrained/mutate_everything/predictions/{stage}/seed{k}/proteingym/*.csv'))):
            name = os.path.basename(f)[:-4]
            if name == 'summary':
                continue
            labels = pd.read_csv(base / f'pgym_inputs/{name}.csv', usecols=['mut_type_renumbered', 'DMS_score'])
            r[name] = score_me_assay(pd.read_csv(f, low_memory=False), labels)
        per_dict[k] = pd.Series(r, dtype=float)
    per = pd.DataFrame(per_dict)
    rows = {f'seed{k}': summarise_me_pgym(per[k], cat['category']) for k in seeds}
    rows['mean'] = summarise_me_pgym(per.mean(axis=1), cat['category'])
    tab = pd.DataFrame(rows).T
    print(tab.round(4).to_string())
    sd = tab.loc[[f'seed{k}' for k in seeds], ['PG avg', 'Stability', 'DMS mean']].std(ddof=1)
    print('seed sd (ddof 1): ' + '  '.join(f'{c} {v:.4f}' for c, v in sd.items()))
    if out:
        atomic_to_csv(per, out, index=True)
    if stage == 'stage1' and 'MutateEverything single' in cat.columns:
        keep = [d for d in cat.index if d not in PG_DROP]
        diff = (per.mean(axis=1).reindex(keep) - cat.loc[keep, 'MutateEverything single']).abs().max()
        print(f'max |recomputed - pgym_all_models_per_dms "MutateEverything single"| over the 213 DMS: {diff:.1e}')
    return per, tab


# ------------------------------------------------------------------------------------------------ ProteinGym published & Rosetta singles
def score_pgym_public(
    out_csv: str | Path = 'pgym_public_per_dms.csv',
    models: Sequence[str] = ('ProSST-2048', 'ProteinMPNN'),
    zip_path: str | Path | None = None,
    inputs_dir: str | Path | None = None,
    categories_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Per-DMS Spearman of ProteinGym's published zero-shot scores (mapped AF2 subset vs full assay)."""
    zp = Path(zip_path) if zip_path is not None else PGYM_PUBLIC_ZIP
    inp = Path(inputs_dir) if inputs_dir is not None else PGYM_INPUTS_DIR
    cat_path = Path(categories_csv) if categories_csv is not None else (PGYM_PER_DMS_DIR / 'pgym_all_models_per_dms.csv')
    model_cols = list(models)
    cat = pd.read_csv(cat_path).set_index('DMS_id')['category']
    rows = []
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
        for dms in cat.index:
            arc_name = f'{dms}.csv'
            if arc_name not in names:
                continue
            with z.open(arc_name) as f:
                pub = pd.read_csv(io.TextIOWrapper(f), usecols=['mutant', 'DMS_score'] + model_cols)
            mine = pd.read_csv(inp / f'{dms}.csv', usecols=['mutant'])
            sub = pub[pub['mutant'].isin(set(mine['mutant']))]
            r: dict[str, object] = dict(DMS_id=dms, category=cat[dms], n_public=len(pub), n_mapped=len(sub))
            for m in model_cols:
                r[f'{m} rho_mapped'] = rho(sub[m], sub['DMS_score']) if len(sub) > 2 else float('nan')
                r[f'{m} rho_full'] = rho(pub[m], pub['DMS_score']) if len(pub) > 2 else float('nan')
                r[f'{m} n_scored'] = int(pub[m].notna().sum())
            rows.append(r)
    out_df = pd.DataFrame(rows)
    atomic_to_csv(out_df, out_csv, index=False)
    return out_df


def score_pgym_rosetta_singles(
    preds_csv: str | Path,
    out_csv: str | Path = 'pgym_rosetta_singles_per_dms.csv',
    ref_csv: str | Path | None = None,
    inputs_dir: str | Path | None = None,
    categories_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Per-DMS Spearman of Rosetta Cartesian ddG on ProteinGym single substitutions (verifies stabilizing-positive sign)."""
    rp = Path(ref_csv) if ref_csv is not None else PGYM_REF_CSV
    inp = Path(inputs_dir) if inputs_dir is not None else PGYM_INPUTS_DIR
    cat_path = Path(categories_csv) if categories_csv is not None else (PGYM_PER_DMS_DIR / 'pgym_all_models_per_dms.csv')
    raw = pd.read_csv(preds_csv)
    key_parts = raw['protein'].astype(str).str.extract(r'^P(?P<i>\d+)_(?P<pm>.+)$')
    raw = raw.assign(i=key_parts['i'].astype(int), pm=key_parts['pm'], cart=raw['cartesian_ddg_dir'].astype(float))
    ref = pd.read_csv(rp)
    cat = pd.read_csv(cat_path).set_index('DMS_id')['category']
    rows = []
    for i, dms in enumerate(ref['DMS_id']):
        if dms not in cat.index:
            continue
        sub = raw[raw['i'] == i]
        mine = pd.read_csv(inp / f'{dms}.csv', usecols=['mutant', 'DMS_score'])
        mine = mine[~mine['mutant'].astype(str).str.contains(':', regex=False)].copy()
        mine['pm'] = mine['mutant'].astype(str).str[1:]
        j = mine.merge(sub[['pm', 'cart']].drop_duplicates('pm'), on='pm', how='left')
        rows.append(dict(
            DMS_id=dms,
            category=cat[dms],
            n_singles=len(mine),
            n_scored=int(j['cart'].notna().sum()),
            rho=rho(j['cart'], j['DMS_score']) if int(j['cart'].notna().sum()) > 2 else float('nan'),
        ))
    out_df = pd.DataFrame(rows)
    stab = out_df[out_df['category'] == 'Stability']['rho'].mean()
    if not (np.isfinite(stab) and stab > 0.2):
        raise SystemExit('sign check failed: cartesian_ddg_dir no longer correlates positively with stability')
    atomic_to_csv(out_df, out_csv, index=False)
    return out_df


# ------------------------------------------------------------------------------------------------ Multi-seed run summary
SEEDED_SETS = ('s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld')
SEEDED_DMS = (
    'DLG4_HUMAN_Faure_2021_binding_domain',
    'GRB2_HUMAN_Faure_2021_binding_domain',
    'MYO_HUMAN_Kung_2025_display',
)
SEEDED_KEYS = (
    [('tsu_capped_val', 'tsuboyama_val_all'), ('tsu_capped_test', 'tsuboyama_test_all')]
    + [(s, f'external_{s}') for s in SEEDED_SETS]
    + [(d, f'dms_{d}') for d in SEEDED_DMS]
    + [
        ('domainome', 'domainome_mean_over_domains'),
        ('PG avg', 'proteingym_PG avg'),
        ('PG Stability', 'proteingym_PG Stability'),
        ('PG DMS mean', 'proteingym_PG DMS mean'),
    ]
)


def summarize_seeded_runs(
    scored_dir: str | Path,
    tag: str,
    predictions_dir: str | Path,
    source: str,
    out_csv: str | Path,
    models: Sequence[tuple[str, str, str]] = (
        ('SPURS (retrained)', 'SPURS', 'SPURS_single_additive'),
        ('SPURS multi (retrained)', 'SPURSm', 'SPURS_multi'),
    ),
    seeds: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Aggregate per-seed ``score_baseline_dir`` summaries into mean and sample SD (``ddof=1``) plus held-out Domainome."""
    scored_dir = Path(scored_dir)
    predictions_dir = Path(predictions_dir)
    dom_excl = heldout.excluded('domainome')
    rows = []
    for model_name, prefix, col in models:
        per_seed = [pd.read_csv(scored_dir / f'{prefix}_{tag}_s{k}_summary.csv', index_col=0) for k in seeds]
        for ds_name, key in SEEDED_KEYS:
            vals = np.array([float(s.loc[key, 'spearman_all']) for s in per_seed], dtype=float)
            sd = float(vals.std(ddof=1)) if len(vals) >= 2 else float('nan')
            rows.append((model_name, ds_name, float(vals.mean()), sd, len(seeds), source))
        dom_vals = []
        for k in seeds:
            d = pd.read_csv(predictions_dir / f'seed{k}' / 'domainome.csv', usecols=['code', 'ddG', col], low_memory=False).dropna()
            per_dom = [rho(g[col], g['ddG']) for c, g in d.groupby('code') if str(c) not in dom_excl and len(g) > 2]
            dom_vals.append(float(np.nanmean(per_dom)))
        v = np.asarray(dom_vals, dtype=float)
        sd = float(v.std(ddof=1)) if len(v) >= 2 else float('nan')
        rows.append((model_name, 'domainome_low_id', float(v.mean()), sd, len(seeds), source))
    out_df = pd.DataFrame(rows, columns=['model', 'dataset', 'rho', 'sd', 'n_seeds', 'source'])
    atomic_to_csv(out_df, out_csv, index=False)
    return out_df


# ------------------------------------------------------------------------------------------------ CLI entrypoints
def cli_baseline(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Score a baseline run directory with ESM-MSR metrics.')
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out', required=True, help='directory for <label>_summary.csv')
    ap.add_argument('--label', required=True)
    ap.add_argument('--col', required=True, help='prediction column, e.g. SPURS_single_additive, SPURS_multi, pred_combined')
    ap.add_argument('--epi_col', default='none', help="predicted-epistasis column for Tsuboyama doubles, or 'none' (additive: 0)")
    ap.add_argument('--pgym_col', default=None, help='per-DMS Spearman column of proteingym/summary.csv (default: spearman_<col>)')
    ap.add_argument('--pgym_categories', default=str(PGYM_PER_DMS_DIR / 'pgym_all_models_per_dms.csv'))
    a = ap.parse_args(argv)
    epi = None if a.epi_col == 'none' else a.epi_col
    path, out = score_baseline_dir(a.pred_dir, a.out, a.label, a.col, epi_col=epi, pgym_col=a.pgym_col, pgym_categories=a.pgym_categories)
    print(f'wrote {path}')
    cols = [c for c in ('spearman_all', 'spearman_singles', 'spearman_doubles', 'n_singles', 'n_doubles') if c in out.columns]
    print(out[cols].to_string(float_format=lambda x: f'{x:.4f}'))


def cli_me_pgym(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Score MutateEverything ProteinGym predictions.')
    ap.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--root', default=str(MSR_ROOT))
    ap.add_argument('--out', help='write the per-DMS rho of every seed here')
    a = ap.parse_args(argv)
    run_me_pgym(a.stage, seeds=a.seeds, root=a.root, out=a.out)


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in ('-h', '--help'):
        ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
        ap.add_argument('cmd', choices=['baseline', 'me-pgym', 'pgym-public', 'pgym-rosetta', 'seeded-summary'])
        ap.parse_args(args_list)
        return
    cmd, rest = args_list[0], args_list[1:]
    if cmd == 'baseline':
        cli_baseline(rest)
    elif cmd == 'me-pgym':
        cli_me_pgym(rest)
    elif cmd == 'pgym-public':
        ap = argparse.ArgumentParser()
        ap.add_argument('--out', default='pgym_public_per_dms.csv')
        ap.add_argument('--models', default='ProSST-2048,ProteinMPNN')
        a = ap.parse_args(rest)
        df = score_pgym_public(a.out, models=[m.strip() for m in a.models.split(',') if m.strip()])
        print(f'{len(df)} DMS -> {a.out}')
    elif cmd == 'pgym-rosetta':
        ap = argparse.ArgumentParser()
        ap.add_argument('--preds', required=True)
        ap.add_argument('--out', default='pgym_rosetta_singles_per_dms.csv')
        a = ap.parse_args(rest)
        df = score_pgym_rosetta_singles(a.preds, out_csv=a.out)
        print(f'{len(df)} DMS -> {a.out}')
    elif cmd == 'seeded-summary':
        ap = argparse.ArgumentParser()
        ap.add_argument('--scored', required=True)
        ap.add_argument('--tag', required=True)
        ap.add_argument('--predictions', required=True)
        ap.add_argument('--source', required=True)
        ap.add_argument('--out', required=True)
        a = ap.parse_args(rest)
        df = summarize_seeded_runs(a.scored, a.tag, a.predictions, a.source, a.out)
        print(f'wrote {a.out}: {len(df)} rows')
    else:
        raise SystemExit(f'unknown subcommand: {cmd}')


if __name__ == '__main__':
    main()
