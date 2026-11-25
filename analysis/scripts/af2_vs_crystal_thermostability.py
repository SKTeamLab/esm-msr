#!/usr/bin/env python3
"""AF2 (ColabFold) vs. crystal structure analysis on the 12 external thermostability benchmarks.

Retrieves and analyzes the matched ESM-MSR-small (March reference checkpoint) inference runs
completed on the local cluster (`subhak-gpu.uwaterloo.ca`, job 380998, 2026-10-08) across all 12
external thermostability datasets (25,632 variants; 805 benchmark groups sharing 762 unique WT
sequences folded with ColabFold AF2 monomer-ptm, 5 models, 3 recycles, Amber GPU relax).

Follows current `analysis/AGENTS.md` and `msr_analysis` conventions:
  - Canonical deployed readout: WT-view (`wt_lora_pred`) on the 10 single-mutant sets and
    Dual-view (`combined_pred`) on the 2 multi-mutant sets (`PTMUL`, `PTMUL-D`).
  - Also reports the original Oct 8 Dual-view (`combined_pred`) readout across all 12 sets for
    direct continuity with `esm-msr-small_af2_vs_crystal.csv` and `docs/BENCHMARKS_OCT2026.md §4`.
  - Joint cross-dataset paired protein-cluster bootstrap (using canonical `Frame.spine['cluster']`
    from `msr_analysis.data.load_frame`, so shared proteins across overlapping subsets/supersets
    and all 15 Ssym WT families are resampled jointly) and paired mutation-level bootstrap
    (`B = 10,000`, finite-sample `1/(B+1)` two-sided p-values, Holm adjustment across 12 sets).
  - Ssym direct (WT crystal) vs. reverse (mutant crystal) & 342-pair antisymmetry decomposition.
  - Publication-styled figures via `msr_analysis.style` (`#ffffff`, `dpi=220`, Type 42 PDF fonts,
    hollow/filled marker encoding) and `.csv` / `.md` / `.tex` booktabs tables via `msr_analysis.tables`.

Usage:
    PYTHONPATH=src:analysis python analysis/scripts/af2_vs_crystal_thermostability.py [--B 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import _Ranks, ci, global_cluster_index  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.paths import MSR, atomic_to_csv, atomic_write_text, results_dir  # noqa: E402
from msr_analysis.registry import EXT12, LABEL  # noqa: E402
from msr_analysis.stats import boot_pvalue, fmt_p, holm  # noqa: E402
from msr_analysis.tables import format_stat_table_md  # noqa: E402

AF2_DIR = MSR / 'esm-msr-small_march_af2'
CRYSTAL_DIR = MSR / 'esm-msr-small_march_crystal'

SINGLE_SETS = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369']
MULTI_SETS = ['ptmul', 'ptmuld']
MAIN_TEXT_8 = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754', 'ptmul']

# Alphabetical dataset order matching SI Table S4 (`table_S4_thermostability_all.tex`)
TABLE_ORDER = ['k2369', 'k3822', 'ptmul', 'ptmuld', 'q3421', 's461', 's571', 's669', 's783', 's2648', 's8754', 'ssym']

DATASET_NOTES = {
    'k2369': ('Main-text, single', r'$\subset$ K3822; 91% in S8754'),
    'k3822': ('Additional, single', r'$\supset$ K2369; 89% in S8754'),
    'ptmul': ('Main-text, multi', r'$\supset$ PTMUL-D'),
    'ptmuld': ('Additional, multi', r'doubles $\subset$ PTMUL'),
    'q3421': ('Main-text, single', '98% in S8754'),
    's461': ('Main-text, single', r'$\subset$ S669'),
    's571': ('Additional, single', r'$\Delta T_m$ ($^\circ$C)'),
    's669': ('Additional, single', r'$\supset$ S461'),
    's783': ('Main-text, single', '90% in S8754'),
    's2648': ('Main-text, single', r'$\subset$ S8754'),
    's8754': ('Main-text, single', r'$\supset$ S2648, Ssym-dir'),
    'ssym': ('Main-text, single', '342 dir + 342 rev (15 WTs)'),
}

DATASET_NOTES_MD = {
    'k2369': 'Main-text, single; ⊂ K3822',
    'k3822': 'Additional, single; ⊃ K2369',
    'ptmul': 'Main-text, multi; ⊃ PTMUL-D',
    'ptmuld': 'Additional, multi; ⊂ PTMUL',
    'q3421': 'Main-text, single; 98% in S8754',
    's461': 'Main-text, single; ⊂ S669',
    's571': 'Additional, single; ΔTm (°C)',
    's669': 'Additional, single; ⊃ S461',
    's783': 'Main-text, single; 90% in S8754',
    's2648': 'Main-text, single; ⊂ S8754',
    's8754': 'Main-text, single; ⊃ S2648, Ssym-dir',
    'ssym': 'Main-text, single; 342 dir + 342 rev',
}

# Colors matching `msr_analysis.models` (`ESM-MSR` = #000000; slate blue `#2c5f8a` from `ABLATION_STYLES`/`VARIANT_COLOR`)
COLOR_CRYSTAL = '#000000'
COLOR_AF2 = '#2c5f8a'
COLOR_DELTA_POS = '#1b8a4b'
COLOR_DELTA_NEG = '#c0392b'
COLOR_DELTA_NEUT = '#2c5f8a'


def _load_paired_datasets() -> dict[str, dict]:
    """Load matched crystal and AF2 predictions and assign canonical WT protein clusters."""
    data: dict[str, dict] = {}
    for ds in EXT12:
        fr = load_frame(ds)
        dc = pd.read_csv(CRYSTAL_DIR / f'{ds}.csv')
        da = pd.read_csv(AF2_DIR / f'{ds}.csv')

        if len(dc) != len(da) or len(dc) != len(fr.spine):
            raise ValueError(f'Row count mismatch on {ds}: spine={len(fr.spine)}, cry={len(dc)}, af2={len(da)}')
        if not (dc['id'].astype(str).to_numpy() == da['id'].astype(str).to_numpy()).all():
            raise ValueError(f'ID order mismatch between crystal and AF2 on {ds}')
        if not (dc['index'].to_numpy() == da['index'].to_numpy()).all():
            raise ValueError(f'Row index order mismatch between crystal and AF2 on {ds}')

        y = dc['ddG'].to_numpy(dtype=np.float64)
        if not np.allclose(y, da['ddG'].to_numpy(dtype=np.float64), equal_nan=False):
            raise ValueError(f'Ground-truth ddG mismatch between crystal and AF2 on {ds}')

        # Canonical WT protein clustering matching `msr_analysis.data._wt_cluster`:
        # `wt_code` on Ssym (15 WT families), 4-char PDB `code` on all other external sets.
        if 'wt_code' in dc.columns:
            cluster = dc['wt_code'].fillna(dc['code']).astype(str).to_numpy()
        else:
            cluster = dc['code'].astype(str).to_numpy()
        spine_n_cl = int(fr.spine['cluster'].nunique())
        if len(np.unique(cluster)) != spine_n_cl:
            raise ValueError(f'Cluster count mismatch on {ds}: {len(np.unique(cluster))} vs spine {spine_n_cl}')

        is_multi = ds in MULTI_SETS
        dep_col = 'combined_pred' if is_multi else 'wt_lora_pred'
        data[ds] = dict(
            ds=ds,
            is_multi=is_multi,
            dep_head='Dual-view' if is_multi else 'WT-view',
            dep_col=dep_col,
            y=y,
            cluster=cluster,
            group_cc=(dc['code'].astype(str) + dc['chain'].astype(str)).to_numpy(),
            cry_dep=dc[dep_col].to_numpy(dtype=np.float64),
            af2_dep=da[dep_col].to_numpy(dtype=np.float64),
            cry_wt=dc['wt_lora_pred'].to_numpy(dtype=np.float64),
            af2_wt=da['wt_lora_pred'].to_numpy(dtype=np.float64),
            cry_comb=dc['combined_pred'].to_numpy(dtype=np.float64),
            af2_comb=da['combined_pred'].to_numpy(dtype=np.float64),
            df_cry=dc,
            df_af2=da,
        )
    return data


def _ranks_2d(rk: _Ranks, W_ord: np.ndarray) -> np.ndarray:
    """Vectorized exact mid-ranks for `(n, C)` integer weight columns on a single pooled group."""
    before = np.cumsum(W_ord, axis=0) - W_ord
    if len(rk.tie_start) == len(rk.order):
        return before + (W_ord + 1.0) * 0.5
    tie_T = np.add.reduceat(W_ord, rk.tie_start, axis=0)
    mid = before[rk.tie_start] + (tie_T + 1.0) * 0.5
    return mid[rk.tie_id]


def _weighted_spearman_multi(ry_rk: _Ranks, rx_rks: list[_Ranks], W: np.ndarray) -> list[np.ndarray]:
    """Compute exact weighted Spearman rho for multiple `rx_rks` sharing `(ry_rk, W)` in one Y pass."""
    Wy_ord = W[ry_rk.order]
    ry_ord = _ranks_2d(ry_rk, Wy_ord)
    sw = Wy_ord.sum(axis=0)
    wry_ord = Wy_ord * ry_ord
    with np.errstate(invalid='ignore', divide='ignore'):
        my = wry_ord.sum(axis=0) / sw
        vy = (wry_ord * ry_ord).sum(axis=0) / sw - my * my
    wry = wry_ord[ry_rk.inv]
    bady = (sw < 3.0) | (vy <= 0.0) | ~np.isfinite(vy)

    outs = []
    for rx_rk in rx_rks:
        Wx_ord = W[rx_rk.order]
        rx_ord = _ranks_2d(rx_rk, Wx_ord)
        wrx_ord = Wx_ord * rx_ord
        with np.errstate(invalid='ignore', divide='ignore'):
            mx = wrx_ord.sum(axis=0) / sw
            vx = (wrx_ord * rx_ord).sum(axis=0) / sw - mx * mx
            cxy = (rx_ord * wry[rx_rk.order]).sum(axis=0) / sw - mx * my
            r = cxy / np.sqrt(vx * vy)
        r[bady | (vx <= 0.0) | ~np.isfinite(r)] = np.nan
        outs.append(r)
    return outs


def _weighted_spearman_batch(ry_rk: _Ranks, rx_rk: _Ranks, W: np.ndarray) -> np.ndarray:
    """Compute exact weighted Spearman rho across `C` columns of integer row multiplicities `W (n, C)`."""
    return _weighted_spearman_multi(ry_rk, [rx_rk], W)[0]


def _weighted_rmse_batch(err2: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute weighted RMSE across `C` columns of `W (n, C)` given squared errors `err2 (n,)`."""
    sw = W.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mse = (err2 @ W) / sw
        out = np.sqrt(mse)
    out[sw <= 0] = np.nan
    return out


def _weighted_pearson_batch(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute weighted Pearson r across `C` columns of `W (n, C)`."""
    sw = W.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx = (x @ W) / sw
        my = (y @ W) / sw
        vx = ((x * x) @ W) / sw - mx * mx
        vy = ((y * y) @ W) / sw - my * my
        cxy = ((x * y) @ W) / sw - mx * my
        r = cxy / np.sqrt(vx * vy)
    r[(sw < 3.0) | (vx <= 0.0) | (vy <= 0.0) | ~np.isfinite(r)] = np.nan
    return r


def run_paired_bootstraps(data: dict[str, dict], B: int = 10000, seed: int = 42) -> dict:
    """Joint paired protein-cluster and mutation-level bootstraps across all 12 datasets.

    Shared protein clusters across overlapping datasets (`S461 ⊂ S669`, `K2369 ⊂ K3822`,
    `S2648 ⊂ S8754`, `PTMUL-D ⊂ PTMUL`, and all 15 `Ssym` WT families) are resampled jointly
    via `global_cluster_index`, so both per-dataset and cross-dataset macro-average CIs/p-values
    respect full protein-level dependence.
    """
    prep = {}
    for ds, d in data.items():
        g0 = np.zeros(len(d['y']), dtype=np.int64)
        is_multi = d['is_multi']
        rx_c_dep = _Ranks(d['cry_dep'], g0)
        rx_a_dep = _Ranks(d['af2_dep'], g0)
        rx_c_comb = rx_c_dep if is_multi else _Ranks(d['cry_comb'], g0)
        rx_a_comb = rx_a_dep if is_multi else _Ranks(d['af2_comb'], g0)
        prep[ds] = dict(
            is_multi=is_multi,
            ry=_Ranks(d['y'], g0),
            rx_cry_dep=rx_c_dep,
            rx_af2_dep=rx_a_dep,
            rx_cry_comb=rx_c_comb,
            rx_af2_comb=rx_a_comb,
            err2_cry_dep=(d['y'] - d['cry_dep']) ** 2,
            err2_af2_dep=(d['y'] - d['af2_dep']) ** 2,
            err2_cry_comb=(d['y'] - d['cry_comb']) ** 2,
            err2_af2_comb=(d['y'] - d['af2_comb']) ** 2,
        )

    gidx, G_global = global_cluster_index({ds: data[ds]['cluster'] for ds in EXT12})

    reps = {
        kind: {
            ds: {
                m: np.empty(B, dtype=np.float64)
                for m in (
                    'rho_cry_dep', 'rho_af2_dep', 'rho_cry_comb', 'rho_af2_comb',
                    'rmse_cry_dep', 'rmse_af2_dep', 'rmse_cry_comb', 'rmse_af2_comb',
                )
            }
            for ds in EXT12
        }
        for kind in ('protein', 'mutation')
    }

    chunk = 128
    # 1. Joint protein-cluster bootstrap across all 12 datasets
    rng_p = np.random.default_rng(seed)
    p_unif = np.full(G_global, 1.0 / G_global, dtype=np.float64)
    for c0 in range(0, B, chunk):
        c1 = min(B, c0 + chunk)
        C = c1 - c0
        counts_global = rng_p.multinomial(G_global, p_unif, size=C).T.astype(np.float64)

        for ds in EXT12:
            W = counts_global[gidx[ds]]  # (n_ds, C)
            p = prep[ds]
            r = reps['protein'][ds]
            if p['is_multi']:
                rc, ra = _weighted_spearman_multi(p['ry'], [p['rx_cry_dep'], p['rx_af2_dep']], W)
                r['rho_cry_dep'][c0:c1] = r['rho_cry_comb'][c0:c1] = rc
                r['rho_af2_dep'][c0:c1] = r['rho_af2_comb'][c0:c1] = ra
                ec = _weighted_rmse_batch(p['err2_cry_dep'], W)
                ea = _weighted_rmse_batch(p['err2_af2_dep'], W)
                r['rmse_cry_dep'][c0:c1] = r['rmse_cry_comb'][c0:c1] = ec
                r['rmse_af2_dep'][c0:c1] = r['rmse_af2_comb'][c0:c1] = ea
            else:
                rc_d, ra_d, rc_c, ra_c = _weighted_spearman_multi(
                    p['ry'], [p['rx_cry_dep'], p['rx_af2_dep'], p['rx_cry_comb'], p['rx_af2_comb']], W
                )
                r['rho_cry_dep'][c0:c1] = rc_d
                r['rho_af2_dep'][c0:c1] = ra_d
                r['rho_cry_comb'][c0:c1] = rc_c
                r['rho_af2_comb'][c0:c1] = ra_c
                r['rmse_cry_dep'][c0:c1] = _weighted_rmse_batch(p['err2_cry_dep'], W)
                r['rmse_af2_dep'][c0:c1] = _weighted_rmse_batch(p['err2_af2_dep'], W)
                r['rmse_cry_comb'][c0:c1] = _weighted_rmse_batch(p['err2_cry_comb'], W)
                r['rmse_af2_comb'][c0:c1] = _weighted_rmse_batch(p['err2_af2_comb'], W)
    print('  [boot] protein-cluster bootstrap done', flush=True)

    # 2. Paired mutation-level bootstrap per dataset
    rng_m = np.random.default_rng(seed + 1000)
    for ds in EXT12:
        n = len(data[ds]['y'])
        p_n = np.full(n, 1.0 / n, dtype=np.float64)
        p = prep[ds]
        r = reps['mutation'][ds]
        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            C = c1 - c0
            W = rng_m.multinomial(n, p_n, size=C).T.astype(np.float64)
            if p['is_multi']:
                rc, ra = _weighted_spearman_multi(p['ry'], [p['rx_cry_dep'], p['rx_af2_dep']], W)
                r['rho_cry_dep'][c0:c1] = r['rho_cry_comb'][c0:c1] = rc
                r['rho_af2_dep'][c0:c1] = r['rho_af2_comb'][c0:c1] = ra
                ec = _weighted_rmse_batch(p['err2_cry_dep'], W)
                ea = _weighted_rmse_batch(p['err2_af2_dep'], W)
                r['rmse_cry_dep'][c0:c1] = r['rmse_cry_comb'][c0:c1] = ec
                r['rmse_af2_dep'][c0:c1] = r['rmse_af2_comb'][c0:c1] = ea
            else:
                rc_d, ra_d, rc_c, ra_c = _weighted_spearman_multi(
                    p['ry'], [p['rx_cry_dep'], p['rx_af2_dep'], p['rx_cry_comb'], p['rx_af2_comb']], W
                )
                r['rho_cry_dep'][c0:c1] = rc_d
                r['rho_af2_dep'][c0:c1] = ra_d
                r['rho_cry_comb'][c0:c1] = rc_c
                r['rho_af2_comb'][c0:c1] = ra_c
                r['rmse_cry_dep'][c0:c1] = _weighted_rmse_batch(p['err2_cry_dep'], W)
                r['rmse_af2_dep'][c0:c1] = _weighted_rmse_batch(p['err2_af2_dep'], W)
                r['rmse_cry_comb'][c0:c1] = _weighted_rmse_batch(p['err2_cry_comb'], W)
                r['rmse_af2_comb'][c0:c1] = _weighted_rmse_batch(p['err2_af2_comb'], W)
    print('  [boot] mutation-level bootstrap done', flush=True)

    return reps


def run_ssym_analysis(data: dict[str, dict], B: int = 10000, seed: int = 77) -> pd.DataFrame:
    """Compute Ssym direct, reverse, pooled, and 342-pair antisymmetry metrics with paired CIs."""
    dc = data['ssym']['df_cry'].copy()
    da = data['ssym']['df_af2'].copy()
    clusters = data['ssym']['cluster']
    dc['cluster'] = clusters
    da['cluster'] = clusters

    is_dir = (dc['code'] == dc['wt_code']).to_numpy()
    di_c = dc[is_dir].sort_values('index').reset_index(drop=True)
    rv_c = dc[~is_dir].sort_values('index').reset_index(drop=True)
    di_a = da[is_dir].sort_values('index').reset_index(drop=True)
    rv_a = da[~is_dir].sort_values('index').reset_index(drop=True)

    # Verify exact 1-to-1 direct-reverse pairing (342 pairs across 15 WT protein clusters)
    assert len(di_c) == 342 and len(rv_c) == 342
    assert (di_c['wt_code'].to_numpy() == rv_c['wt_code'].to_numpy()).all()
    assert (di_c['wild_type'].to_numpy() == rv_c['mutation'].to_numpy()).all()
    assert (di_c['mutation'].to_numpy() == rv_c['wild_type'].to_numpy()).all()
    assert np.allclose(di_c['ddG'].to_numpy(), -rv_c['ddG'].to_numpy())
    assert (di_c['id'].to_numpy() == di_a['id'].to_numpy()).all()
    assert (rv_c['id'].to_numpy() == rv_a['id'].to_numpy()).all()

    # Map 15 WT clusters
    wt_codes, cl_idx = np.unique(di_c['wt_code'].astype(str).to_numpy(), return_inverse=True)
    G = len(wt_codes)
    assert G == 15

    rng = np.random.default_rng(seed)
    W_pair = np.empty((342, B), dtype=np.float64)
    for j in range(B):
        c_mult = np.bincount(rng.integers(0, G, size=G), minlength=G)
        W_pair[:, j] = c_mult[cl_idx]
    W_pool = np.vstack([W_pair, W_pair])  # exact paired resamples across direct + reverse

    y_dir = di_c['ddG'].to_numpy(float)
    y_rev = rv_c['ddG'].to_numpy(float)
    y_all = np.concatenate([y_dir, y_rev])

    g0_p = np.zeros(342, dtype=np.int64)
    g0_a = np.zeros(684, dtype=np.int64)
    ry_dir = _Ranks(y_dir, g0_p)
    ry_rev = _Ranks(y_rev, g0_p)
    ry_all = _Ranks(y_all, g0_a)

    rows = []
    for head_label, col in [('WT-view', 'wt_lora_pred'), ('Dual-view', 'combined_pred')]:
        xc_d = di_c[col].to_numpy(float)
        xa_d = di_a[col].to_numpy(float)
        xc_r = rv_c[col].to_numpy(float)
        xa_r = rv_a[col].to_numpy(float)
        xc_all = np.concatenate([xc_d, xc_r])
        xa_all = np.concatenate([xa_d, xa_r])

        for subset, yv, ry_rk, xc, xa, W in [
            ('Direct (WT struct)', y_dir, ry_dir, xc_d, xa_d, W_pair),
            ('Reverse (Mut struct)', y_rev, ry_rev, xc_r, xa_r, W_pair),
            ('Pooled (Dir + Rev)', y_all, ry_all, xc_all, xa_all, W_pool),
        ]:
            rc_pt = float(spearmanr(yv, xc)[0])
            ra_pt = float(spearmanr(yv, xa)[0])
            pc_pt = float(pearsonr(yv, xc)[0])
            pa_pt = float(pearsonr(yv, xa)[0])
            rmse_c = float(np.sqrt(np.mean((yv - xc) ** 2)))
            rmse_a = float(np.sqrt(np.mean((yv - xa) ** 2)))
            pr_rho = float(spearmanr(xc, xa)[0])

            g0 = np.zeros(len(yv), dtype=np.int64)
            rc_rep = _weighted_spearman_batch(ry_rk, _Ranks(xc, g0), W)
            ra_rep = _weighted_spearman_batch(ry_rk, _Ranks(xa, g0), W)
            d_rep = ra_rep - rc_rep
            rc_lo, rc_hi = ci(rc_rep)
            ra_lo, ra_hi = ci(ra_rep)
            d_lo, d_hi = ci(d_rep)
            p_prot = boot_pvalue(d_rep)
            rows.append(dict(
                head=head_label,
                subset=subset,
                metric='Spearman_rho',
                n=len(yv),
                n_clusters=G,
                crystal=rc_pt, crystal_lo=rc_lo, crystal_hi=rc_hi,
                af2=ra_pt, af2_lo=ra_lo, af2_hi=ra_hi,
                delta=ra_pt - rc_pt, delta_lo=d_lo, delta_hi=d_hi,
                p_prot=p_prot,
                crystal_r=pc_pt, af2_r=pa_pt,
                crystal_rmse=rmse_c, af2_rmse=rmse_a,
                pred_rho=pr_rho,
            ))

        # Antisymmetry Pearson r(dir, rev) and bias delta = mean(dir + rev)/2
        anti_c_pt = float(pearsonr(xc_d, xc_r)[0])
        anti_a_pt = float(pearsonr(xa_d, xa_r)[0])
        anti_c_rep = _weighted_pearson_batch(xc_d, xc_r, W_pair)
        anti_a_rep = _weighted_pearson_batch(xa_d, xa_r, W_pair)
        anti_d_rep = anti_a_rep - anti_c_rep
        rows.append(dict(
            head=head_label,
            subset='Antisymmetry r(dir, rev)',
            metric='Pearson_r_antisym',
            n=342,
            n_clusters=G,
            crystal=anti_c_pt, crystal_lo=ci(anti_c_rep)[0], crystal_hi=ci(anti_c_rep)[1],
            af2=anti_a_pt, af2_lo=ci(anti_a_rep)[0], af2_hi=ci(anti_a_rep)[1],
            delta=anti_a_pt - anti_c_pt, delta_lo=ci(anti_d_rep)[0], delta_hi=ci(anti_d_rep)[1],
            p_prot=boot_pvalue(anti_d_rep),
            crystal_r=np.nan, af2_r=np.nan,
            crystal_rmse=np.nan, af2_rmse=np.nan,
            pred_rho=np.nan,
        ))

        bias_c_vec = 0.5 * (xc_d + xc_r)
        bias_a_vec = 0.5 * (xa_d + xa_r)
        bias_c_pt = float(np.mean(bias_c_vec))
        bias_a_pt = float(np.mean(bias_a_vec))
        sw = W_pair.sum(axis=0)
        bias_c_rep = (bias_c_vec[:, None] * W_pair).sum(axis=0) / sw
        bias_a_rep = (bias_a_vec[:, None] * W_pair).sum(axis=0) / sw
        bias_d_rep = bias_a_rep - bias_c_rep
        rows.append(dict(
            head=head_label,
            subset='Bias <delta> (kcal/mol)',
            metric='Antisym_bias',
            n=342,
            n_clusters=G,
            crystal=bias_c_pt, crystal_lo=ci(bias_c_rep)[0], crystal_hi=ci(bias_c_rep)[1],
            af2=bias_a_pt, af2_lo=ci(bias_a_rep)[0], af2_hi=ci(bias_a_rep)[1],
            delta=bias_a_pt - bias_c_pt, delta_lo=ci(bias_d_rep)[0], delta_hi=ci(bias_d_rep)[1],
            p_prot=boot_pvalue(bias_d_rep),
            crystal_r=np.nan, af2_r=np.nan,
            crystal_rmse=np.nan, af2_rmse=np.nan,
            pred_rho=np.nan,
        ))

    return pd.DataFrame(rows)


def build_summary_tables(data: dict[str, dict], reps: dict, B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble full per-dataset + macro-average summary tables for deployed and Dual-view heads."""
    orig = pd.read_csv(AF2_DIR / 'esm-msr-small_af2_vs_crystal.csv').set_index('set')

    def _assemble(mode: str) -> pd.DataFrame:
        rows = []
        raw_p_prot = []
        raw_p_mut = []
        for ds in TABLE_ORDER:
            d = data[ds]
            y = d['y']
            xc = d['cry_dep'] if mode == 'deployed' else d['cry_comb']
            xa = d['af2_dep'] if mode == 'deployed' else d['af2_comb']
            head_name = d['dep_head'] if mode == 'deployed' else 'Dual-view'
            suf = 'dep' if mode == 'deployed' else 'comb'

            rho_c = float(spearmanr(y, xc)[0])
            rho_a = float(spearmanr(y, xa)[0])
            d_rho = rho_a - rho_c
            r_c = float(pearsonr(y, xc)[0])
            r_a = float(pearsonr(y, xa)[0])
            rmse_c = float(np.sqrt(np.mean((y - xc) ** 2)))
            rmse_a = float(np.sqrt(np.mean((y - xa) ** 2)))
            pred_rho = float(spearmanr(xc, xa)[0])
            pred_r = float(pearsonr(xc, xa)[0])
            pred_mad = float(np.mean(np.abs(xa - xc)))

            rp = reps['protein'][ds]
            rm = reps['mutation'][ds]
            dp_rep = rp[f'rho_af2_{suf}'] - rp[f'rho_cry_{suf}']
            dm_rep = rm[f'rho_af2_{suf}'] - rm[f'rho_cry_{suf}']
            drm_p_rep = rp[f'rmse_af2_{suf}'] - rp[f'rmse_cry_{suf}']
            drm_m_rep = rm[f'rmse_af2_{suf}'] - rm[f'rmse_cry_{suf}']

            p_prot = boot_pvalue(dp_rep)
            p_mut = boot_pvalue(dm_rep)
            raw_p_prot.append(p_prot)
            raw_p_mut.append(p_mut)

            rows.append(dict(
                dataset=ds,
                label=LABEL[ds],
                is_summary=False,
                is_multi=d['is_multi'],
                head=head_name,
                n=len(y),
                n_clusters=int(len(np.unique(d['cluster']))),
                n_groups_cc=int(len(np.unique(d['group_cc']))),
                crystal_rho=rho_c,
                crystal_prot_lo=ci(rp[f'rho_cry_{suf}'])[0],
                crystal_prot_hi=ci(rp[f'rho_cry_{suf}'])[1],
                crystal_mut_lo=ci(rm[f'rho_cry_{suf}'])[0],
                crystal_mut_hi=ci(rm[f'rho_cry_{suf}'])[1],
                af2_rho=rho_a,
                af2_prot_lo=ci(rp[f'rho_af2_{suf}'])[0],
                af2_prot_hi=ci(rp[f'rho_af2_{suf}'])[1],
                af2_mut_lo=ci(rm[f'rho_af2_{suf}'])[0],
                af2_mut_hi=ci(rm[f'rho_af2_{suf}'])[1],
                delta_rho=d_rho,
                delta_prot_lo=ci(dp_rep)[0],
                delta_prot_hi=ci(dp_rep)[1],
                delta_mut_lo=ci(dm_rep)[0],
                delta_mut_hi=ci(dm_rep)[1],
                p_prot=p_prot,
                p_mut=p_mut,
                crystal_r=r_c,
                af2_r=r_a,
                delta_r=r_a - r_c,
                crystal_rmse=rmse_c,
                crystal_rmse_prot_lo=ci(rp[f'rmse_cry_{suf}'])[0],
                crystal_rmse_prot_hi=ci(rp[f'rmse_cry_{suf}'])[1],
                crystal_rmse_mut_lo=ci(rm[f'rmse_cry_{suf}'])[0],
                crystal_rmse_mut_hi=ci(rm[f'rmse_cry_{suf}'])[1],
                af2_rmse=rmse_a,
                af2_rmse_prot_lo=ci(rp[f'rmse_af2_{suf}'])[0],
                af2_rmse_prot_hi=ci(rp[f'rmse_af2_{suf}'])[1],
                af2_rmse_mut_lo=ci(rm[f'rmse_af2_{suf}'])[0],
                af2_rmse_mut_hi=ci(rm[f'rmse_af2_{suf}'])[1],
                delta_rmse=rmse_a - rmse_c,
                delta_rmse_prot_lo=ci(drm_p_rep)[0],
                delta_rmse_prot_hi=ci(drm_p_rep)[1],
                p_rmse_prot=boot_pvalue(drm_p_rep),
                p_rmse_mut=boot_pvalue(drm_m_rep),
                pred_rho=pred_rho,
                pred_r=pred_r,
                pred_mad=pred_mad,
                oct8_cluster_delta=float(orig.loc[ds, 'delta']),
                oct8_cluster_p=float(orig.loc[ds, 'p_two_sided']),
                notes_md=DATASET_NOTES_MD[ds],
                notes_tex=f"{DATASET_NOTES[ds][0]}; {DATASET_NOTES[ds][1]}",
            ))

        adj_prot = holm(raw_p_prot)
        adj_mut = holm(raw_p_mut)
        for row, ap, am in zip(rows, adj_prot, adj_mut):
            row['p_prot_holm'] = float(ap)
            row['p_mut_holm'] = float(am)

        # Macro-average summary rows (computed jointly over aligned protein-cluster & mutation draws)
        by_ds = {r['dataset']: r for r in rows}
        suf = 'dep' if mode == 'deployed' else 'comb'
        cohorts = [
            ('mean_10_singles', 'Mean (10 single-mutant sets)', SINGLE_SETS, False, 'WT-view' if mode == 'deployed' else 'Dual-view', 'Unweighted macro-mean'),
            ('mean_8_main', 'Mean (8 main-text sets)', MAIN_TEXT_8, False, 'Deployed' if mode == 'deployed' else 'Dual-view', 'Fig. 3a cohort (excl. S571, S669, K3822, PTMUL-D)'),
            ('mean_2_multi', 'Mean (2 multi-mutant sets)', MULTI_SETS, True, 'Dual-view', 'PTMUL & PTMUL-D'),
            ('mean_all_12', 'Mean (all 12 sets)', EXT12, False, 'Deployed' if mode == 'deployed' else 'Dual-view', 'All 12 external benchmarks'),
        ]
        for key, lbl, dss, is_m, hd, note in cohorts:
            ddg_dss = [d for d in dss if d != 's571']  # exclude ΔTm (°C) from kcal/mol RMSE average
            c_rho_pt = float(np.mean([by_ds[d]['crystal_rho'] for d in dss]))
            a_rho_pt = float(np.mean([by_ds[d]['af2_rho'] for d in dss]))
            c_r_pt = float(np.mean([by_ds[d]['crystal_r'] for d in dss]))
            a_r_pt = float(np.mean([by_ds[d]['af2_r'] for d in dss]))
            c_rmse_pt = float(np.mean([by_ds[d]['crystal_rmse'] for d in ddg_dss]))
            a_rmse_pt = float(np.mean([by_ds[d]['af2_rmse'] for d in ddg_dss]))
            pred_rho_pt = float(np.mean([by_ds[d]['pred_rho'] for d in dss]))
            pred_r_pt = float(np.mean([by_ds[d]['pred_r'] for d in dss]))
            pred_mad_pt = float(np.mean([by_ds[d]['pred_mad'] for d in dss]))

            c_p_rep = np.nanmean([reps['protein'][d][f'rho_cry_{suf}'] for d in dss], axis=0)
            a_p_rep = np.nanmean([reps['protein'][d][f'rho_af2_{suf}'] for d in dss], axis=0)
            c_m_rep = np.nanmean([reps['mutation'][d][f'rho_cry_{suf}'] for d in dss], axis=0)
            a_m_rep = np.nanmean([reps['mutation'][d][f'rho_af2_{suf}'] for d in dss], axis=0)
            dp_rep = a_p_rep - c_p_rep
            dm_rep = a_m_rep - c_m_rep

            rc_p_rep = np.nanmean([reps['protein'][d][f'rmse_cry_{suf}'] for d in ddg_dss], axis=0)
            ra_p_rep = np.nanmean([reps['protein'][d][f'rmse_af2_{suf}'] for d in ddg_dss], axis=0)
            rc_m_rep = np.nanmean([reps['mutation'][d][f'rmse_cry_{suf}'] for d in ddg_dss], axis=0)
            ra_m_rep = np.nanmean([reps['mutation'][d][f'rmse_af2_{suf}'] for d in ddg_dss], axis=0)
            drm_p_rep = ra_p_rep - rc_p_rep
            drm_m_rep = ra_m_rep - rc_m_rep

            n_tot = int(sum(by_ds[d]['n'] for d in dss))
            n_cl_u = int(len(set().union(*(set(data[d]['cluster']) for d in dss))))
            n_cc_u = int(len(set().union(*(set(data[d]['group_cc']) for d in dss))))

            rows.append(dict(
                dataset=key,
                label=lbl,
                is_summary=True,
                is_multi=is_m,
                head=hd,
                n=n_tot,
                n_clusters=n_cl_u,
                n_groups_cc=n_cc_u,
                crystal_rho=c_rho_pt,
                crystal_prot_lo=ci(c_p_rep)[0],
                crystal_prot_hi=ci(c_p_rep)[1],
                crystal_mut_lo=ci(c_m_rep)[0],
                crystal_mut_hi=ci(c_m_rep)[1],
                af2_rho=a_rho_pt,
                af2_prot_lo=ci(a_p_rep)[0],
                af2_prot_hi=ci(a_p_rep)[1],
                af2_mut_lo=ci(a_m_rep)[0],
                af2_mut_hi=ci(a_m_rep)[1],
                delta_rho=a_rho_pt - c_rho_pt,
                delta_prot_lo=ci(dp_rep)[0],
                delta_prot_hi=ci(dp_rep)[1],
                delta_mut_lo=ci(dm_rep)[0],
                delta_mut_hi=ci(dm_rep)[1],
                p_prot=boot_pvalue(dp_rep),
                p_mut=boot_pvalue(dm_rep),
                p_prot_holm=np.nan,
                p_mut_holm=np.nan,
                crystal_r=c_r_pt,
                af2_r=a_r_pt,
                delta_r=a_r_pt - c_r_pt,
                crystal_rmse=c_rmse_pt,
                crystal_rmse_prot_lo=ci(rc_p_rep)[0],
                crystal_rmse_prot_hi=ci(rc_p_rep)[1],
                crystal_rmse_mut_lo=ci(rc_m_rep)[0],
                crystal_rmse_mut_hi=ci(rc_m_rep)[1],
                af2_rmse=a_rmse_pt,
                af2_rmse_prot_lo=ci(ra_p_rep)[0],
                af2_rmse_prot_hi=ci(ra_p_rep)[1],
                af2_rmse_mut_lo=ci(ra_m_rep)[0],
                af2_rmse_mut_hi=ci(ra_m_rep)[1],
                delta_rmse=a_rmse_pt - c_rmse_pt,
                delta_rmse_prot_lo=ci(drm_p_rep)[0],
                delta_rmse_prot_hi=ci(drm_p_rep)[1],
                p_rmse_prot=boot_pvalue(drm_p_rep),
                p_rmse_mut=boot_pvalue(drm_m_rep),
                pred_rho=pred_rho_pt,
                pred_r=pred_r_pt,
                pred_mad=pred_mad_pt,
                oct8_cluster_delta=float(np.mean([orig.loc[d, 'delta'] for d in dss])),
                oct8_cluster_p=np.nan,
                notes_md=note + (' (RMSE excl. S571)' if 's571' in dss else ''),
                notes_tex=note + (' (RMSE excl.\\ S571)' if 's571' in dss else ''),
            ))
        return pd.DataFrame(rows)

    return _assemble('deployed'), _assemble('dualview')


def _fmt_signed(v: float, nd: int = 3) -> str:
    if not np.isfinite(v):
        return '—'
    s = f'{v:+.{nd}f}'
    return s.replace('-', '−')


def _fmt_p_clean(p: float, B: int) -> str:
    if not np.isfinite(p):
        return '—'
    if p <= 1.05 / (B + 1):
        return f'<{1.0 / B:.0e}'.replace('1e-0', '10^-').replace('1e-', '10^-')
    if p < 0.001:
        return f'{p:.4f}'
    return f'{p:.3f}'


def _fmt_p_tex(p: float, B: int) -> str:
    if not np.isfinite(p):
        return '---'
    if p <= 1.05 / (B + 1):
        exp = int(round(np.log10(B)))
        return rf'$<10^{{-{exp}}}$'
    if p < 0.001:
        return f'{p:.4f}'
    return f'{p:.3f}'


def write_comparative_tables(
    df_dep: pd.DataFrame,
    df_dual: pd.DataFrame,
    df_ssym: pd.DataFrame,
    out_dir: Path,
    B: int,
) -> None:
    """Write machine-readable CSV, GitHub Markdown, and copy-pastable SI booktabs LaTeX tables."""
    atomic_to_csv(df_dep, out_dir / 'af2_vs_crystal_summary.csv', index=False, float_format='%.5f')
    atomic_to_csv(df_dual, out_dir / 'af2_vs_crystal_dualview_summary.csv', index=False, float_format='%.5f')
    atomic_to_csv(df_ssym, out_dir / 'ssym_direct_reverse_antisymmetry.csv', index=False, float_format='%.5f')

    # 1. Markdown comparative table (deployed readout)
    md_rows = []
    for r in df_dep.itertuples():
        ci_p = f'[{_fmt_signed(r.delta_prot_lo)}, {_fmt_signed(r.delta_prot_hi)}]'
        p_h = _fmt_p_clean(r.p_prot_holm, B) if not r.is_summary else '—'
        p_pr = _fmt_p_clean(r.p_prot, B)
        p_mu = _fmt_p_clean(r.p_mut, B)
        rmse_unit = '°C' if r.dataset == 's571' else ''
        md_rows.append({
            'Dataset': f'**{r.label}**' if r.is_summary else r.label,
            'n (clust.)': f'{r.n:,} ({r.n_clusters})',
            'Head': r.head,
            'Crystal ρ': f'{r.crystal_rho:.3f}',
            'AF2 ρ': f'{r.af2_rho:.3f}',
            'Δρ (AF2 − Cryst.) [95% prot. CI]': f'{_fmt_signed(r.delta_rho)} {ci_p}',
            'p_prot (Holm)': f'{p_pr} ({p_h})' if not r.is_summary else p_pr,
            'p_mut': p_mu,
            'Crystal / AF2 r': f'{r.crystal_r:.3f} / {r.af2_r:.3f}',
            'Crystal / AF2 RMSE': f'{r.crystal_rmse:.3f} / {r.af2_rmse:.3f}{rmse_unit}',
            'ρ(AF2, Cryst.)': f'{r.pred_rho:.3f}',
            'Notes': r.notes_md,
        })
    md_text = (
        '# AlphaFold2 vs. Crystal Structures on External Thermostability Benchmarks (ESM-MSR-small)\n\n'
        f'Matched evaluation of ColabFold AF2 structures (762 unique WT sequences, 5 models, 3 recycles, Amber relax) '
        f'against experimental crystal structures across all 12 external thermostability sets (25,632 rows; '
        f'canonical deployed head: WT-view on single-mutant sets, Dual-view on PTMUL / PTMUL-D; paired joint '
        f'protein-cluster and mutation-level bootstraps, $B = {B:,}$, two-sided $p$-values).\n\n'
        + format_stat_table_md(pd.DataFrame(md_rows))
    )
    atomic_write_text(out_dir / 'af2_vs_crystal_summary.md', md_text)

    # 2. LaTeX booktabs SI table (matching `table_S4_thermostability_all.tex` conventions)
    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\caption{\textbf{AlphaFold2 vs.\ Crystal Structures Across All 12 External Thermostability Benchmarks (ESM-MSR-small)}. '
        r'Comparison of ColabFold AlphaFold2 monomer-pTM structures (762 unique wild-type sequences; 5 models, 3 recycles, Amber GPU relaxation) '
        r'against experimental crystal structures on the 12 external thermostability datasets in alphabetical order '
        r'(subset/superset pairs annotated: K2369 $\subset$ K3822, PTMUL-D $\subset$ PTMUL, S461 $\subset$ S669, S2648 $\subset$ S8754, and Ssym-dir $\subset$ S8754). '
        r'Predictions use the canonical deployed head per dataset composition (WT-view on single-mutant datasets; Dual-view on multi-mutant PTMUL and PTMUL-D). '
        r'Columns report variant count $n$ (unique protein clusters), pooled Spearman $\rho$ for crystal and AF2 structures, '
        r'paired difference $\Delta\rho = \rho_{\mathrm{AF2}} - \rho_{\mathrm{crystal}}$ with 95\% protein-cluster bootstrap CI ($B=' + f'{B:,}' + r'$), '
        r'two-sided paired bootstrap $p$-values under protein-cluster ($p_{\mathrm{prot}}$, with Holm-adjusted $p_{\mathrm{Holm}}$ across the 12 sets) '
        r'and mutation ($p_{\mathrm{mut}}$) resampling, Pearson $r$, raw RMSE (kcal/mol; $^\circ$C for S571, excluded from RMSE macro-means), '
        r'and prediction-vs-prediction concordance $\rho(\hat{y}_{\mathrm{AF2}}, \hat{y}_{\mathrm{cryst}})$.}\label{tab:si_af2_vs_crystal}',
        r'\begin{tabular}{lrcccccccccc}',
        r'\toprule',
        r'Dataset & $n$ (cl.) & Head & Cryst.\ $\rho$ & AF2 $\rho$ & $\Delta\rho$ [95\% prot.\ CI] & $p_{\mathrm{prot}}$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{mut}}$ & $r$ (Cr/AF2) & RMSE (Cr/AF2) & $\rho_{\mathrm{pred}}$ \\',
        r'\midrule',
    ]
    for idx_r, r in enumerate(df_dep.itertuples()):
        if idx_r == len(TABLE_ORDER):
            tex_lines.append(r'\midrule')
        ds_lbl = r.label.replace('&', r'\&')
        if r.is_summary:
            ds_lbl = rf'\textbf{{{ds_lbl}}}'
        d_s = f'{r.delta_rho:+.3f}'.replace('-', '$-$')
        lo_s = f'{r.delta_prot_lo:+.3f}'.replace('-', '$-$')
        hi_s = f'{r.delta_prot_hi:+.3f}'.replace('-', '$-$')
        ci_tex = f'{d_s} [{lo_s}, {hi_s}]'
        if not r.is_summary and np.isfinite(r.p_prot) and r.p_prot < 0.05:
            ci_tex = rf'\textbf{{{ci_tex}}}'
        rmse_tag = r'$^\circ$C' if r.dataset == 's571' else ''
        tex_lines.append(
            f'{ds_lbl} & {r.n:,} ({r.n_clusters}) & {r.head} & '
            f'{r.crystal_rho:.3f} & {r.af2_rho:.3f} & {ci_tex} & '
            f'{_fmt_p_tex(r.p_prot, B)} & {_fmt_p_tex(r.p_prot_holm, B)} & {_fmt_p_tex(r.p_mut, B)} & '
            f'{r.crystal_r:.3f}/{r.af2_r:.3f} & {r.crystal_rmse:.2f}/{r.af2_rmse:.2f}{rmse_tag} & {r.pred_rho:.3f} \\\\'
        )
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]
    atomic_write_text(out_dir / 'af2_vs_crystal_summary.tex', '\n'.join(tex_lines))

    # 3. Head comparison table (Canonical Deployed vs Oct 8 Cluster Dual-view Combined)
    comp_rows = []
    d_by = df_dep.set_index('dataset')
    c_by = df_dual.set_index('dataset')
    for ds in TABLE_ORDER + ['mean_10_singles', 'mean_8_main', 'mean_2_multi', 'mean_all_12']:
        rd = d_by.loc[ds]
        rc = c_by.loc[ds]
        comp_rows.append({
            'Dataset': f'**{rd["label"]}**' if rd['is_summary'] else rd['label'],
            'Deployed Cryst / AF2 ρ': f'{rd["crystal_rho"]:.3f} / {rd["af2_rho"]:.3f}',
            'Deployed Δρ [95% prot. CI]': f'{_fmt_signed(rd["delta_rho"])} [{_fmt_signed(rd["delta_prot_lo"])}, {_fmt_signed(rd["delta_prot_hi"])}]',
            'Deployed p_prot': _fmt_p_clean(rd['p_prot'], B),
            'Dual-view Cryst / AF2 ρ': f'{rc["crystal_rho"]:.3f} / {rc["af2_rho"]:.3f}',
            'Dual-view Δρ [95% prot. CI]': f'{_fmt_signed(rc["delta_rho"])} [{_fmt_signed(rc["delta_prot_lo"])}, {_fmt_signed(rc["delta_prot_hi"])}]',
            'Dual-view p_prot': _fmt_p_clean(rc['p_prot'], B),
            'Oct 8 CSV Δρ (p)': f'{_fmt_signed(rc["oct8_cluster_delta"])} ({rc["oct8_cluster_p"]:.3f})' if np.isfinite(rc['oct8_cluster_p']) else _fmt_signed(rc['oct8_cluster_delta']),
        })
    comp_md = (
        '# Head Sensitivity: Canonical Deployed (WT-view Singles / Dual-view Multi) vs. Oct 8 Dual-view Combined\n\n'
        + format_stat_table_md(pd.DataFrame(comp_rows))
    )
    atomic_write_text(out_dir / 'af2_vs_crystal_head_comparison.md', comp_md)

    # 4. Ssym direct / reverse / antisymmetry markdown table
    ssym_md_rows = []
    for r in df_ssym.itertuples():
        ssym_md_rows.append({
            'Head': r.head,
            'Ssym Component': r.subset,
            'n (WT clust.)': f'{int(r.n)} ({int(r.n_clusters)})',
            'Crystal [95% CI]': f'{r.crystal:.3f} [{r.crystal_lo:.3f}, {r.crystal_hi:.3f}]',
            'AF2 [95% CI]': f'{r.af2:.3f} [{r.af2_lo:.3f}, {r.af2_hi:.3f}]',
            'Δ (AF2 − Cryst.) [95% CI]': f'{_fmt_signed(r.delta)} [{_fmt_signed(r.delta_lo)}, {_fmt_signed(r.delta_hi)}]',
            'p_prot': _fmt_p_clean(r.p_prot, B),
            'RMSE Cr / AF2': f'{r.crystal_rmse:.3f} / {r.af2_rmse:.3f}' if np.isfinite(r.crystal_rmse) else '—',
        })
    atomic_write_text(
        out_dir / 'ssym_direct_reverse_antisymmetry.md',
        '# Ssym Direct vs. Reverse and Antisymmetry Breakdown (Crystal vs. AlphaFold2)\n\n'
        + format_stat_table_md(pd.DataFrame(ssym_md_rows)),
    )


def plot_main_figure(df_dep: pd.DataFrame, out_dir: Path, B: int) -> None:
    """Two-panel publication figure: (a) paired Spearman rho; (b) paired delta-rho with protein & mutation CIs."""
    style.setup()

    # Order: 10 single-mutant sets sorted descending by Crystal rho, then 2 multi-mutant sets, then 2 macro-means
    singles_df = df_dep[(~df_dep['is_summary']) & (~df_dep['is_multi'])].sort_values('crystal_rho', ascending=False)
    multi_df = df_dep[(~df_dep['is_summary']) & (df_dep['is_multi'])].sort_values('crystal_rho', ascending=False)
    sum_df = df_dep[df_dep['dataset'].isin(['mean_10_singles', 'mean_2_multi'])].set_index('dataset').loc[
        ['mean_10_singles', 'mean_2_multi']
    ].reset_index()

    ordered = pd.concat([singles_df, multi_df, sum_df], ignore_index=True)
    n_s = len(singles_df)
    n_m = len(multi_df)

    # X positions with clean visual gaps separating singles | multi | means
    xs = []
    x = 0.0
    for i, r in enumerate(ordered.itertuples()):
        if i == n_s:
            x += 0.55
        elif i == n_s + n_m:
            x += 0.55
        xs.append(x)
        x += 1.0
    xs = np.array(xs, dtype=float)
    div1 = 0.5 * (xs[n_s - 1] + xs[n_s])
    div2 = 0.5 * (xs[n_s + n_m - 1] + xs[n_s + n_m])

    xlabels = []
    for r in ordered.itertuples():
        if r.dataset == 'mean_10_singles':
            xlabels.append('Mean\n(10 singles)')
        elif r.dataset == 'mean_2_multi':
            xlabels.append('Mean\n(2 multi)')
        elif r.dataset == 's571':
            xlabels.append('S571\n(ΔTm)')
        else:
            xlabels.append(r.label)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.6, 7.6), sharex=True,
        gridspec_kw={'height_ratios': [1.32, 1.0], 'hspace': 0.14},
    )

    # Alternating vertical grey bands & section dividers on both panels
    for ax in (ax1, ax2):
        for i, xi in enumerate(xs):
            if i % 2 == 0:
                ax.axvspan(xi - 0.5, xi + 0.5, color=style.BAND, zorder=0, lw=0)
        # Subtle warm tint behind summary means
        ax.axvspan(div2, xs[-1] + 0.55, color='#eef3f8', alpha=0.55, zorder=0, lw=0)
        ax.axvline(div1, color='#999999', lw=0.95, ls='--', zorder=1)
        ax.axvline(div2, color='#666666', lw=1.05, ls='--', zorder=1)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)

    # ---------------- Panel a: Crystal vs AF2 Spearman rho ----------------
    dx = 0.16
    for xi, r in zip(xs, ordered.itertuples()):
        is_hollow = not bool(r.is_multi)
        # Crystal (left)
        xc_pos = xi - dx
        ax1.plot([xc_pos, xc_pos], [r.crystal_prot_lo, r.crystal_prot_hi],
                 color=COLOR_CRYSTAL, lw=3.0, alpha=0.35, solid_capstyle='butt', zorder=2)
        ax1.errorbar(
            xc_pos, r.crystal_rho,
            yerr=[[max(0.0, r.crystal_rho - r.crystal_mut_lo)], [max(0.0, r.crystal_mut_hi - r.crystal_rho)]],
            fmt='none', ecolor=COLOR_CRYSTAL, elinewidth=1.0, capsize=2.0, capthick=0.9, alpha=0.9, zorder=3,
        )
        kw_c = dict(
            marker='D', ms=7.2 if not is_hollow else 7.6, color=COLOR_CRYSTAL, mec=COLOR_CRYSTAL,
            mfc='white' if is_hollow else COLOR_CRYSTAL, mew=2.0 if is_hollow else 1.1, ls='none', zorder=5,
        )
        ax1.plot(xc_pos, r.crystal_rho, **kw_c)

        # AF2 (right)
        xa_pos = xi + dx
        ax1.plot([xa_pos, xa_pos], [r.af2_prot_lo, r.af2_prot_hi],
                 color=COLOR_AF2, lw=3.0, alpha=0.35, solid_capstyle='butt', zorder=2)
        ax1.errorbar(
            xa_pos, r.af2_rho,
            yerr=[[max(0.0, r.af2_rho - r.af2_mut_lo)], [max(0.0, r.af2_mut_hi - r.af2_rho)]],
            fmt='none', ecolor=COLOR_AF2, elinewidth=1.0, capsize=2.0, capthick=0.9, alpha=0.9, zorder=3,
        )
        kw_a = dict(
            marker='o', ms=7.0 if not is_hollow else 7.5, color=COLOR_AF2, mec=COLOR_AF2,
            mfc='white' if is_hollow else COLOR_AF2, mew=2.0 if is_hollow else 1.1, ls='none', zorder=5,
        )
        ax1.plot(xa_pos, r.af2_rho, **kw_a)

    ax1.set_ylabel(r'Spearman $\rho$ (pooled)')
    ax1.set_ylim(0.24, 0.90)
    ax1.text(-0.048, 1.04, 'a', transform=ax1.transAxes, fontsize=12.5, fontweight='bold', va='bottom', ha='left')

    # Group span headers above Panel a (matching `style.bar_panel` headers)
    tr1 = ax1.get_xaxis_transform()
    spans = [
        (xs[0] - 0.42, xs[n_s - 1] + 0.42, 'Single-mutant benchmarks (WT-view, hollow)'),
        (xs[n_s] - 0.42, xs[n_s + n_m - 1] + 0.42, 'Multi-mutant\n(Dual-view, filled)'),
        (xs[n_s + n_m] - 0.42, xs[-1] + 0.42, 'Macro-means'),
    ]
    for x0, x1, title in spans:
        ax1.plot([x0, x1], [1.015, 1.015], transform=tr1, clip_on=False, color='#444444', lw=1.1)
        ax1.text((x0 + x1) / 2, 1.028, title, transform=tr1, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')

    # ---------------- Panel b: Paired Delta-rho (AF2 - Crystal) ----------------
    ax2.axhline(0.0, color='#333333', lw=1.0, ls='--', zorder=1.5)
    for xi, r in zip(xs, ordered.itertuples()):
        is_hollow = not bool(r.is_multi)
        sig_prot = (not r.is_summary) and (r.p_prot < 0.05)
        sig_holm = (not r.is_summary) and np.isfinite(r.p_prot_holm) and (r.p_prot_holm < 0.05)
        col = COLOR_DELTA_NEG if (sig_prot and r.delta_rho < 0) else (COLOR_DELTA_POS if (sig_prot and r.delta_rho > 0) else COLOR_DELTA_NEUT)

        # Thick protein-cluster paired 95% CI bar
        ax2.plot([xi, xi], [r.delta_prot_lo, r.delta_prot_hi],
                 color=col, lw=3.8, alpha=0.35, solid_capstyle='butt', zorder=2.5)
        # Thin mutation-level paired 95% CI whisker
        ax2.errorbar(
            xi, r.delta_rho,
            yerr=[[max(0.0, r.delta_rho - r.delta_mut_lo)], [max(0.0, r.delta_mut_hi - r.delta_rho)]],
            fmt='none', ecolor=col, elinewidth=1.1, capsize=2.4, capthick=1.0, alpha=0.95, zorder=3.5,
        )
        ax2.plot(
            xi, r.delta_rho,
            marker='D' if r.is_summary else 'o',
            ms=7.0 if not is_hollow else 7.4,
            color=col, mec=col,
            mfc='white' if is_hollow else col,
            mew=2.0 if is_hollow else 1.2,
            ls='none', zorder=5,
        )
        if sig_prot:
            ax2.plot(xi, r.delta_rho, marker='o', ms=11.5, mfc='none', mec=style.RING, mew=1.1, ls='none', zorder=6)

        # Crisp value annotation placed cleanly clear of protein CI bounds
        txt = _fmt_signed(r.delta_rho, 3)
        if sig_holm:
            txt += '**'
        elif sig_prot:
            txt += '*'
        if r.delta_rho >= 0:
            y_pos = max(r.delta_prot_hi, r.delta_mut_hi, r.delta_rho) + 0.0065
            va = 'bottom'
        else:
            y_pos = min(r.delta_prot_lo, r.delta_mut_lo, r.delta_rho) - 0.0065
            va = 'top'
        ax2.text(
            xi, y_pos, txt, ha='center', va=va, fontsize=7.6,
            fontweight='bold' if (r.is_summary or sig_prot) else 'normal',
            color=col if sig_prot else '#222222', zorder=8,
            bbox=dict(fc='white', ec='none', alpha=0.78, pad=0.5),
        )

    ax2.set_ylabel(r'Paired $\Delta\rho$ (AF2 $-$ Crystal)')
    ax2.set_ylim(-0.108, 0.088)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(xlabels, fontsize=8.8)
    for tick, r in zip(ax2.get_xticklabels(), ordered.itertuples()):
        if r.is_summary:
            tick.set_fontweight('bold')
    ax2.text(-0.048, 1.03, 'b', transform=ax2.transAxes, fontsize=12.5, fontweight='bold', va='bottom', ha='left')

    # Shared publication legend at top (matching `thermostability_sets.png`)
    handles = [
        Line2D([0], [0], marker='D', ms=7.2, color=COLOR_CRYSTAL, mec=COLOR_CRYSTAL, mfc=COLOR_CRYSTAL, ls='none', label='Crystal structure (ESM-MSR-small)'),
        Line2D([0], [0], marker='o', ms=7.0, color=COLOR_AF2, mec=COLOR_AF2, mfc=COLOR_AF2, ls='none', label='AlphaFold2 structure (ColabFold)'),
        Line2D([0], [0], color='#555555', lw=3.6, alpha=0.38, label='thick bar: 95% protein-cluster CI'),
        Line2D([0], [0], color='#555555', lw=1.1, label='thin whisker: 95% mutation-level CI'),
        Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec=style.RING, mew=1.1, ls='none', label=r'$p_{\mathrm{prot}} < 0.05$ (* unadj., ** Holm)'),
        *style.version_handles(size=6.5),
    ]
    fig.legend(
        handles=handles, loc='upper center', bbox_to_anchor=(0.51, 0.965), ncol=4, fontsize=8.2,
        frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.2, handletextpad=0.4,
    )
    fig.suptitle(
        f'AlphaFold2 vs. crystal structures on external thermostability benchmarks (ESM-MSR-small; B = {B:,} paired bootstraps)',
        x=0.015, ha='left', y=0.995, fontsize=10.8, fontweight='bold',
    )
    r_s10 = df_dep[df_dep['dataset'] == 'mean_10_singles'].iloc[0]
    r_m2 = df_dep[df_dep['dataset'] == 'mean_2_multi'].iloc[0]
    min_holm = float(df_dep.loc[~df_dep['is_summary'], 'p_prot_holm'].min())
    n_holm_sig = int((df_dep.loc[~df_dep['is_summary'], 'p_prot_holm'] < 0.05).sum())
    holm_str = (
        f'no single dataset survives Holm adjustment across all 12 sets (min p_Holm = {_fmt_p_clean(min_holm, B)})'
        if n_holm_sig == 0
        else f'{n_holm_sig}/12 datasets survive Holm adjustment (min p_Holm = {_fmt_p_clean(min_holm, B)})'
    )
    note = (
        '762 unique WT sequences folded with ColabFold AF2 (5 models, 3 recycles, Amber relax) vs. experimental crystal structures across 12 external benchmarks (25,632 variants; 0 indexing mismatches).\n'
        'Canonical deployed head per dataset: WT-view (hollow) on single-mutant sets, Dual-view (filled) on PTMUL / PTMUL-D. Protein-cluster bootstrap resamples canonical protein clusters jointly across\n'
        f'overlapping datasets (S461 ⊂ S669, K2369 ⊂ K3822, S2648 ⊂ S8754, PTMUL-D ⊂ PTMUL; 15 WT families in Ssym). Macro-means: 10 singles Δρ = {_fmt_signed(r_s10.delta_rho)} [{_fmt_signed(r_s10.delta_prot_lo)}, {_fmt_signed(r_s10.delta_prot_hi)}], '
        f'2 multi Δρ = {_fmt_signed(r_m2.delta_rho)} [{_fmt_signed(r_m2.delta_prot_lo)}, {_fmt_signed(r_m2.delta_prot_hi)}]; {holm_str}.'
    )
    fig.text(0.015, 0.008, note, fontsize=7.6, va='bottom', ha='left', color=style.INK_NOTE)
    fig.subplots_adjust(left=0.068, right=0.985, top=0.835, bottom=0.115)
    style.save(fig, out_dir / 'af2_vs_crystal_thermostability')


def plot_diagnostics_figure(df_dep: pd.DataFrame, df_ssym: pd.DataFrame, out_dir: Path, B: int) -> None:
    """Three-panel diagnostic figure: (a) raw RMSE; (b) AF2-Crystal prediction concordance; (c) Ssym dir/rev & antisymmetry."""
    style.setup()

    singles_ddg = df_dep[(~df_dep['is_summary']) & (~df_dep['is_multi']) & (df_dep['dataset'] != 's571')].sort_values('crystal_rho', ascending=False)
    multi_df = df_dep[(~df_dep['is_summary']) & (df_dep['is_multi'])].sort_values('crystal_rho', ascending=False)
    ddg_ordered = pd.concat([singles_ddg, multi_df], ignore_index=True)

    all_singles = df_dep[(~df_dep['is_summary']) & (~df_dep['is_multi'])].sort_values('crystal_rho', ascending=False)
    all_ordered = pd.concat([all_singles, multi_df], ignore_index=True)

    fig = plt.figure(figsize=(14.2, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1.0], hspace=0.38, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---------------- Panel a: Raw RMSE (kcal/mol) across the 11 ΔΔG sets ----------------
    xs_a = np.arange(len(ddg_ordered), dtype=float)
    n_s_ddg = len(singles_ddg)
    xs_a[n_s_ddg:] += 0.5
    div_a = 0.5 * (xs_a[n_s_ddg - 1] + xs_a[n_s_ddg])

    for i, xi in enumerate(xs_a):
        if i % 2 == 0:
            ax_a.axvspan(xi - 0.5, xi + 0.5, color=style.BAND, zorder=0, lw=0)
    ax_a.axvline(div_a, color='#888888', lw=0.95, ls='--', zorder=1)
    ax_a.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)

    dx = 0.16
    for xi, r in zip(xs_a, ddg_ordered.itertuples()):
        is_hollow = not bool(r.is_multi)
        # Crystal RMSE
        xc = xi - dx
        ax_a.plot([xc, xc], [r.crystal_rmse_prot_lo, r.crystal_rmse_prot_hi],
                  color=COLOR_CRYSTAL, lw=3.0, alpha=0.35, solid_capstyle='butt', zorder=2)
        ax_a.errorbar(
            xc, r.crystal_rmse,
            yerr=[[max(0.0, r.crystal_rmse - r.crystal_rmse_mut_lo)], [max(0.0, r.crystal_rmse_mut_hi - r.crystal_rmse)]],
            fmt='none', ecolor=COLOR_CRYSTAL, elinewidth=1.0, capsize=2.0, zorder=3,
        )
        ax_a.plot(xc, r.crystal_rmse, marker='D', ms=7.2, color=COLOR_CRYSTAL, mec=COLOR_CRYSTAL,
                  mfc='white' if is_hollow else COLOR_CRYSTAL, mew=2.0 if is_hollow else 1.1, ls='none', zorder=5)

        # AF2 RMSE
        xa = xi + dx
        ax_a.plot([xa, xa], [r.af2_rmse_prot_lo, r.af2_rmse_prot_hi],
                  color=COLOR_AF2, lw=3.0, alpha=0.35, solid_capstyle='butt', zorder=2)
        ax_a.errorbar(
            xa, r.af2_rmse,
            yerr=[[max(0.0, r.af2_rmse - r.af2_rmse_mut_lo)], [max(0.0, r.af2_rmse_mut_hi - r.af2_rmse)]],
            fmt='none', ecolor=COLOR_AF2, elinewidth=1.0, capsize=2.0, zorder=3,
        )
        ax_a.plot(xa, r.af2_rmse, marker='o', ms=7.0, color=COLOR_AF2, mec=COLOR_AF2,
                  mfc='white' if is_hollow else COLOR_AF2, mew=2.0 if is_hollow else 1.1, ls='none', zorder=5)

        # Compact ΔRMSE annotation above pair
        y_top = max(r.crystal_rmse_prot_hi, r.af2_rmse_prot_hi) + 0.055
        ax_a.text(xi, y_top, _fmt_signed(r.delta_rmse, 2), ha='center', va='bottom', fontsize=7.5, color='#333333')

    ax_a.set_xticks(xs_a)
    ax_a.set_xticklabels([r.label for r in ddg_ordered.itertuples()], fontsize=8.8)
    ax_a.set_xlim(xs_a[0] - 0.6, xs_a[-1] + 0.6)
    ax_a.set_ylim(0.70, 2.72)
    ax_a.set_ylabel('Raw RMSE (kcal/mol; lower is better)')
    ax_a.set_title('Raw RMSE: Crystal vs. AlphaFold2 across the 11 ΔΔG benchmarks (annotated with ΔRMSE = AF2 − Crystal)', loc='left', fontsize=9.8)
    ax_a.text(-0.045, 1.04, 'a', transform=ax_a.transAxes, fontsize=12.5, fontweight='bold', va='bottom', ha='left')
    hs_a = [
        Line2D([0], [0], marker='D', ms=6.8, color=COLOR_CRYSTAL, mec=COLOR_CRYSTAL, mfc=COLOR_CRYSTAL, ls='none', label='Crystal'),
        Line2D([0], [0], marker='o', ms=6.8, color=COLOR_AF2, mec=COLOR_AF2, mfc=COLOR_AF2, ls='none', label='AlphaFold2'),
        Line2D([0], [0], color='#555555', lw=3.4, alpha=0.38, label='95% protein CI'),
        Line2D([0], [0], color='#555555', lw=1.0, label='95% mutation CI'),
    ]
    ax_a.legend(handles=hs_a, loc='upper left', ncol=4, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc')

    # ---------------- Panel b: Prediction concordance rho(y_AF2, y_Crystal) ----------------
    xs_b = np.arange(len(all_ordered), dtype=float)
    n_s_all = len(all_singles)
    xs_b[n_s_all:] += 0.45
    div_b = 0.5 * (xs_b[n_s_all - 1] + xs_b[n_s_all])

    ax_b.axvline(div_b, color='#888888', lw=0.9, ls='--', zorder=1)
    ax_b.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    for xi, r in zip(xs_b, all_ordered.itertuples()):
        hatched = not bool(r.is_multi)
        ax_b.bar(
            xi, r.pred_rho, width=0.62,
            color=COLOR_AF2, alpha=0.62 if hatched else 0.92,
            edgecolor=COLOR_AF2 if hatched else 'black',
            hatch='////' if hatched else None, lw=0.85, zorder=3,
        )
        ax_b.text(xi, r.pred_rho + 0.0028, f'{r.pred_rho:.3f}', ha='center', va='bottom', fontsize=6.9, rotation=90, color='#222222')

    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels([r.label for r in all_ordered.itertuples()], rotation=35, ha='right', fontsize=8.2)
    ax_b.set_xlim(xs_b[0] - 0.65, xs_b[-1] + 0.65)
    ax_b.set_ylim(0.90, 1.008)
    ax_b.set_ylabel(r'Prediction concordance $\rho(\hat{y}_{\mathrm{AF2}}, \hat{y}_{\mathrm{crystal}})$')
    ax_b.set_title('Per-variant prediction agreement (AF2 vs. Crystal)', loc='left', fontsize=9.8)
    ax_b.text(-0.095, 1.04, 'b', transform=ax_b.transAxes, fontsize=12.5, fontweight='bold', va='bottom', ha='left')
    ax_b.legend(
        handles=[
            Patch(fc=COLOR_AF2, ec=COLOR_AF2, alpha=0.62, hatch='////', label='WT-view (singles)'),
            Patch(fc=COLOR_AF2, ec='black', alpha=0.92, label='Dual-view (multi)'),
        ],
        loc='lower left', fontsize=7.8, frameon=True, facecolor='white', edgecolor='#cccccc',
    )

    # ---------------- Panel c: Ssym direct vs reverse & antisymmetry ----------------
    # Show WT-view (deployed on Ssym) Crystal vs AF2 across Direct, Reverse, Pooled, and |Antisymmetry r|
    ss_wt = df_ssym[df_ssym['head'] == 'WT-view'].set_index('subset')
    ss_dv = df_ssym[df_ssym['head'] == 'Dual-view'].set_index('subset')
    cats = [
        ('Direct (WT struct)', 'Direct\n(WT struct)'),
        ('Reverse (Mut struct)', 'Reverse\n(Mut struct)'),
        ('Pooled (Dir + Rev)', 'Pooled\n(n=684)'),
        ('Antisymmetry r(dir, rev)', '|Antisym. r|\n(342 pairs)'),
    ]
    xc_idx = np.arange(len(cats), dtype=float)
    w = 0.32
    ax_c.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    for i, (key, _lbl) in enumerate(cats):
        # Use WT-view for Direct/Reverse/Pooled and |r| for Antisymmetry (with Dual-view diamond markers overlaid!)
        r_wt = ss_wt.loc[key]
        r_dv = ss_dv.loc[key]
        flip = -1.0 if key == 'Antisymmetry r(dir, rev)' else 1.0
        vc = flip * r_wt['crystal']
        va = flip * r_wt['af2']
        vc_lo, vc_hi = sorted([flip * r_wt['crystal_lo'], flip * r_wt['crystal_hi']])
        va_lo, va_hi = sorted([flip * r_wt['af2_lo'], flip * r_wt['af2_hi']])

        ax_c.bar(i - w / 2, vc, width=w, color=COLOR_CRYSTAL, alpha=0.82, edgecolor='black', lw=0.8, zorder=3,
                 label='Crystal (WT-view)' if i == 0 else None)
        ax_c.bar(i + w / 2, va, width=w, color=COLOR_AF2, alpha=0.85, edgecolor='black', lw=0.8, zorder=3,
                 label='AF2 (WT-view)' if i == 0 else None)
        ax_c.errorbar(i - w / 2, vc, yerr=[[max(0.0, vc - vc_lo)], [max(0.0, vc_hi - vc)]],
                      fmt='none', ecolor=style.RING, elinewidth=1.2, capsize=2.5, zorder=5)
        ax_c.errorbar(i + w / 2, va, yerr=[[max(0.0, va - va_lo)], [max(0.0, va_hi - va)]],
                      fmt='none', ecolor=style.RING, elinewidth=1.2, capsize=2.5, zorder=5)

        # Dual-view markers overlaid for reference
        dv_c = flip * r_dv['crystal']
        dv_a = flip * r_dv['af2']
        ax_c.plot(i - w / 2, dv_c, marker='D', ms=5.0, mfc='white', mec='black', mew=1.2, ls='none', zorder=6,
                  label='Dual-view head' if i == 0 else None)
        ax_c.plot(i + w / 2, dv_a, marker='D', ms=5.0, mfc='white', mec=COLOR_AF2, mew=1.2, ls='none', zorder=6)

        # Value labels clear of both CI whiskers and Dual-view markers
        ax_c.text(i - w / 2, max(vc_hi, dv_c) + 0.018, f'{vc:.2f}', ha='center', va='bottom', fontsize=7.3, color='#111111')
        ax_c.text(i + w / 2, max(va_hi, dv_a) + 0.018, f'{va:.2f}', ha='center', va='bottom', fontsize=7.3, color=COLOR_AF2, fontweight='bold')

    ax_c.set_xticks(xc_idx)
    ax_c.set_xticklabels([lbl for _, lbl in cats], fontsize=8.4)
    ax_c.set_ylim(0.35, 1.04)
    ax_c.set_ylabel(r'Spearman $\rho$ / $|r_{\mathrm{antisym}}|$')
    ax_c.set_title('Ssym breakdown: why AF2 drops most on reverse mutations', loc='left', fontsize=9.8)
    ax_c.text(-0.12, 1.04, 'c', transform=ax_c.transAxes, fontsize=12.5, fontweight='bold', va='bottom', ha='left')
    ax_c.legend(loc='lower right', fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc')

    fig.suptitle(
        f'AF2 vs. crystal structural robustness diagnostics: RMSE, prediction concordance, and Ssym decomposition (B = {B:,})',
        x=0.015, ha='left', y=0.985, fontsize=10.6, fontweight='bold',
    )
    r571 = df_dep[df_dep['dataset'] == 's571'].iloc[0]
    pr_lo = float(df_dep.loc[~df_dep['is_summary'], 'pred_rho'].min())
    pr_hi = float(df_dep.loc[~df_dep['is_summary'], 'pred_rho'].max())
    d_rev_wt = float(ss_wt.loc['Reverse (Mut struct)', 'delta'])
    d_rev_dv = float(ss_dv.loc['Reverse (Mut struct)', 'delta'])
    d_dir_wt = float(ss_wt.loc['Direct (WT struct)', 'delta'])
    d_dir_dv = float(ss_dv.loc['Direct (WT struct)', 'delta'])
    diag_note = (
        f'Panel a excludes S571 (ΔTm in °C; Crystal RMSE = {r571.crystal_rmse:.2f} °C vs. AF2 = {r571.af2_rmse:.2f} °C). '
        f'Panel b shows Spearman concordance between Crystal and AF2 predictions on identical variants (range {pr_lo:.3f}–{pr_hi:.3f}).\n'
        f'Panel c decomposes Ssym (684 variants across 15 WT protein families; whiskers = 95% WT-family bootstrap CI): AF2 loses more accuracy on reverse mutations '
        f'(solved mutant crystal backbones replaced by AF2\n'
        f'single-mutant predictions, Δρ = {_fmt_signed(d_rev_wt)} WT-view / {_fmt_signed(d_rev_dv)} Dual-view) '
        f'than direct mutations (Δρ = {_fmt_signed(d_dir_wt)} / {_fmt_signed(d_dir_dv)}); white diamonds mark the Dual-view head.'
    )
    fig.text(0.015, 0.008, diag_note, fontsize=7.5, va='bottom', ha='left', color=style.INK_NOTE)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.905, bottom=0.125)
    style.save(fig, out_dir / 'af2_vs_crystal_diagnostics')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--B', type=int, default=10000, help='Bootstrap replicates (default 10000)')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed')
    ap.add_argument('--plot-only', action='store_true', help='Regenerate figures from existing summary CSVs')
    args = ap.parse_args()

    out_dir = results_dir('thermostability') / 'af2_vs_crystal'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        df_dep = pd.read_csv(out_dir / 'af2_vs_crystal_summary.csv')
        df_ssym = pd.read_csv(out_dir / 'ssym_direct_reverse_antisymmetry.csv')
        plot_main_figure(df_dep, out_dir, B=args.B)
        plot_diagnostics_figure(df_dep, df_ssym, out_dir, B=args.B)
        print(f'[af2_vs_crystal] Regenerated figures in {out_dir}', flush=True)
        return

    print(f'[af2_vs_crystal] Loading 12 external datasets from {AF2_DIR} and {CRYSTAL_DIR}...', flush=True)
    data = _load_paired_datasets()

    print(f'[af2_vs_crystal] Running joint protein-cluster and mutation-level paired bootstraps (B={args.B:,})...', flush=True)
    reps = run_paired_bootstraps(data, B=args.B, seed=args.seed)

    print('[af2_vs_crystal] Building summary tables & Ssym direct/reverse decomposition...', flush=True)
    df_dep, df_dual = build_summary_tables(data, reps, B=args.B)
    df_ssym = run_ssym_analysis(data, B=args.B, seed=args.seed + 35)

    # Verify exact reproduction of Oct 8 Dual-view Combined point estimates
    orig = pd.read_csv(AF2_DIR / 'esm-msr-small_af2_vs_crystal.csv').set_index('set')
    dual_ds = df_dual[~df_dual['is_summary']].set_index('dataset')
    max_diff = max(
        abs(float(dual_ds.loc[ds, 'delta_rho']) - float(orig.loc[ds, 'delta']))
        for ds in EXT12
    )
    print(f'[af2_vs_crystal] Verification vs Oct 8 cluster CSV (Dual-view delta_rho): max abs diff = {max_diff:.2e}', flush=True)
    assert max_diff < 1e-12, f'Mismatch vs Oct 8 CSV: {max_diff}'

    write_comparative_tables(df_dep, df_dual, df_ssym, out_dir, B=args.B)
    plot_main_figure(df_dep, out_dir, B=args.B)
    plot_diagnostics_figure(df_dep, df_ssym, out_dir, B=args.B)

    print(f'[af2_vs_crystal] Done! Outputs written to {out_dir}', flush=True)
    print('\n=== Summary (Canonical Deployed Head) ===')
    cols = ['label', 'n', 'n_clusters', 'head', 'crystal_rho', 'af2_rho', 'delta_rho', 'delta_prot_lo', 'delta_prot_hi', 'p_prot', 'p_prot_holm', 'p_mut', 'crystal_rmse', 'af2_rmse', 'pred_rho']
    print(df_dep[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== Summary (Dual-view / Oct 8 Head, Macro-means) ===')
    print(df_dual[df_dual['is_summary']][cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== Ssym Direct vs Reverse & Antisymmetry ===')
    print(df_ssym[['head', 'subset', 'crystal', 'af2', 'delta', 'delta_lo', 'delta_hi', 'p_prot']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
