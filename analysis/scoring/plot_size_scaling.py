"""Plot controlled protein-size scaling benchmark results (RTX A4500 20 GB + 30/40 GB VRAM extrapolation).

Reads <in_dir>/<model>/results.csv produced by benchmark_size_scaling.py (RTX A4500 20 GB fixed slot)
plus optional <vram_ext_dir>/<model>/results.csv (e.g. RTX 5090 32 GB B=1 VRAM measurements beyond 20 GB)
and generates:
  - analysis/results/10_inference_resources/03_size_scaling_vram_and_screening_a4500.{png,pdf}
  - analysis/results/10_inference_resources/size_scaling_a4500_all.csv
  - analysis/results/10_inference_resources/size_scaling_a4500_summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import nnls

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'analysis'))

from msr_analysis.paths import INFERENCE_RESOURCES_DIR, MSR_ROOT  # noqa: E402
from msr_analysis import style  # noqa: E402
from msr_analysis.models import BY_NAME  # noqa: E402

style.setup()

# All ESM-MSR variants use black / neutral greys (no blue)
C_ESM_BLACK = '#111111'
C_ESM_DARK = '#4d4d4d'
C_ESM_LIGHT = '#8c8c8c'

SPECS = [
    dict(key='esm_msr_wt', label='ESM-MSR (WT-view)', color=C_ESM_BLACK, marker='D', hollow=True, ls='--', lw=2.2),
    dict(key='esm_msr_dual', label='ESM-MSR (Dual-view)', color=C_ESM_BLACK, marker='D', hollow=False, ls='-', lw=2.5),
    dict(key='esm_msr_indep', label='ESM-MSR (Indep. masking)', color=C_ESM_DARK, marker='s', hollow=False, ls='-', lw=2.1),
    dict(key='esm_msr_marg', label='ESM-MSR (Masked marginal)', color=C_ESM_LIGHT, marker='^', hollow=False, ls='-', lw=2.1),
    dict(key='spurs_single', label='SPURS (single)', color=BY_NAME['SPURS'].color, marker='s', hollow=True, ls='--', lw=1.9),
    dict(key='spurs_multi', label='SPURS-multi', color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0),
    dict(key='thermompnn_single', label='ThermoMPNN (SSM single)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=True, ls='--', lw=1.9),
    dict(key='thermompnn_d', label='ThermoMPNN-D (SSM double)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=False, ls='-', lw=2.0),
    dict(key='proteinmpnn', label='ProteinMPNN (zero-shot)', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0),
    dict(key='me_st1', label='MutateEverything (Stage 1)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=True, ls='--', lw=1.9),
    dict(key='me_st2', label='MutateEverything (Stage 2)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0),
    dict(key='prosst', label='ProSST-2048 (zero-shot)', color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0),
]
SPEC_BY_KEY = {s['key']: s for s in SPECS}

# Deduplicated model families for Panel A (models with identical B=1 VRAM merged into one curve)
VRAM_FAMILIES = [
    dict(key='esm_msr_wt', label='ESM-MSR (all variants)', short='ESM-MSR (all)', color=C_ESM_BLACK, marker='D', ls='-', lw=2.5, z=6),
    dict(key='me_st2', label='MutateEverything (St1 / St2)', short='MutateEverything', color=BY_NAME['MutateEverything'].color, marker='h', ls='-', lw=2.1, z=5),
    dict(key='prosst', label='ProSST-2048', short='ProSST-2048', color=BY_NAME['ProSST'].color, marker='X', ls=':', lw=2.1, z=4),
    dict(key='spurs_single', label='SPURS (single / multi)', short='SPURS', color=BY_NAME['SPURS'].color, marker='s', ls='-', lw=2.1, z=4),
    dict(key='thermompnn_d', label='ThermoMPNN / -D', short='ThermoMPNN / -D', color=BY_NAME['ThermoMPNN'].color, marker='p', ls='-', lw=2.1, z=4),
    dict(key='proteinmpnn', label='ProteinMPNN', short='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', ls='-.', lw=2.1, z=4),
]

# Deduplicated single-mutant specs for Panel B (identical 1-mut curves merged: indep+marg, me_st1+me_st2)
SINGLE_SPECS = [
    dict(key='esm_msr_dual', label='ESM-MSR (Dual-view)', color=C_ESM_BLACK, marker='D', hollow=False, ls='-', lw=2.5, z=6),
    dict(key='esm_msr_indep', label='ESM-MSR (Indep. / Marginal)', color=C_ESM_DARK, marker='s', hollow=False, ls='-', lw=2.1, z=6),
    dict(key='esm_msr_wt', label='ESM-MSR (WT-view)', color=C_ESM_BLACK, marker='D', hollow=True, ls='--', lw=2.2, z=6),
    dict(key='me_st2', label='MutateEverything (St1 / St2)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0, z=4),
    dict(key='prosst', label='ProSST-2048', color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0, z=4),
    dict(key='proteinmpnn', label='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0, z=4),
    dict(key='spurs_multi', label='SPURS-multi', color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0, z=4),
    dict(key='spurs_single', label='SPURS (single)', color=BY_NAME['SPURS'].color, marker='s', hollow=True, ls='--', lw=1.9, z=4),
    dict(key='thermompnn_single', label='ThermoMPNN (SSM single)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=True, ls='--', lw=1.9, z=4),
]

# Double-mutant specs for Panel C (all 7 epistatic models are distinct)
DOUBLE_SPECS = [
    dict(key='esm_msr_dual', label='ESM-MSR (Dual-view)', color=C_ESM_BLACK, marker='D', hollow=False, ls='-', lw=2.5, z=6),
    dict(key='esm_msr_indep', label='ESM-MSR (Indep. masking)', color=C_ESM_DARK, marker='s', hollow=False, ls='-', lw=2.1, z=6),
    dict(key='esm_msr_marg', label='ESM-MSR (Masked marginal)', color=C_ESM_LIGHT, marker='^', hollow=False, ls='-', lw=2.1, z=6),
    dict(key='proteinmpnn', label='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0, z=4),
    dict(key='spurs_multi', label='SPURS-multi', color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0, z=4),
    dict(key='me_st2', label='MutateEverything (St2)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0, z=4),
    dict(key='thermompnn_d', label='ThermoMPNN-D (SSM double)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=False, ls='-', lw=2.0, z=4),
]


def _plot_curve(ax, x, y, s, zorder=4, alpha=0.95, label=None):
    hollow = s.get('hollow', False)
    ax.plot(
        x, y,
        color=s['color'], ls=s['ls'], lw=s['lw'], alpha=alpha,
        marker=s['marker'], ms=5.8 if 'esm_msr' in s['key'] else 5.2,
        mfc='white' if hollow else s['color'],
        mec=s['color'], mew=1.5 if hollow else 0.7,
        label=s['label'] if label is None else label,
        zorder=zorder,
    )


def _fmt_dur(sec: float) -> str:
    if not np.isfinite(sec):
        return '—'
    if sec < 1:
        return f'{sec * 1e3:.0f} ms'
    if sec < 60:
        return f'{sec:.1f} s'
    if sec < 3600:
        return f'{sec / 60:.1f} m'
    if sec < 86400:
        return f'{sec / 3600:.1f} h'
    if sec < 86400 * 365:
        return f'{sec / 86400:.1f} d'
    return f'{sec / (86400 * 365):.1f} yr'


def _fmt_L(val: float) -> str:
    if not np.isfinite(val):
        return '—'
    if val >= 10000:
        return f'{val / 1000:.1f}k'
    if val >= 1000:
        return f'{val / 1000:.2f}k'
    return f'{int(round(val))}'


def load_all(in_dir: Path) -> pd.DataFrame:
    dfs = []
    for s in SPECS:
        p = in_dir / s['key'] / 'results.csv'
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            jl = in_dir / s['key'] / 'results.jsonl'
            if jl.exists() and jl.stat().st_size > 0:
                dfs.append(pd.read_json(jl, lines=True))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_vram_ext(ext_dir: Path | None) -> pd.DataFrame:
    if ext_dir is None or not ext_dir.exists():
        return pd.DataFrame()
    dfs = []
    for p in sorted(ext_dir.glob('*/results.csv')):
        d = pd.read_csv(p)
        if not d.empty:
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fit_vram_model(key: str, L_obs: np.ndarray, V_obs: np.ndarray, oom_row: pd.Series | None) -> tuple[float, float, float]:
    """Fit physics-grounded non-negative quadratic V(L) = a + b*L + c*L^2 to measured B=1 PyTorch alloc VRAM."""
    if key == 'prosst':
        # ProSST-2048 is O(L^2) in self-attention above L=4096: at L=4096 alloc=5.69 GB; at L=8192 A4500 OOM
        # occurred with 17.65 GiB allocated + 3.00 GiB requested (= 20.65 GiB exact lower bound for 12*L^2*4B).
        L_hi = np.array([4096.0, 8192.0])
        V_hi = np.array([float(V_obs[-1]), 20.65])
        c = (V_hi[1] - V_hi[0]) / (L_hi[1] ** 2 - L_hi[0] ** 2)
        a = V_hi[0] - c * L_hi[0] ** 2
        return float(max(0.47, a)), 0.0, float(c)
    X = np.column_stack([np.ones_like(L_obs), L_obs, L_obs ** 2])
    w = np.sqrt(L_obs / L_obs.max())
    coef, _ = nnls(X * w[:, None], V_obs * w)
    return float(coef[0]), float(coef[1]), float(coef[2])


def solve_L_for_vram(a: float, b: float, c: float, target_gb: float) -> float:
    if target_gb <= a:
        return np.nan
    if c > 1e-14:
        return float((-b + np.sqrt(b * b + 4.0 * c * (target_gb - a))) / (2.0 * c))
    if b > 1e-14:
        return float((target_gb - a) / b)
    return np.nan


def compute_vram_fits(df: pd.DataFrame, df_ext: pd.DataFrame) -> list[dict]:
    """Combine A4500 B=1 measurements with clean >20 GB 32GB-card B=1 measurements and fit 20/30/40 GB limits."""
    out = []
    for fam in VRAM_FAMILIES:
        key = fam['key']
        sub = df[df['model'] == key].sort_values('seq_len')
        if sub.empty:
            continue
        ok = sub[sub['status'] == 'ok'].copy()
        oom_rows = sub[sub['status'] == 'oom']
        oom_row = oom_rows.iloc[0] if len(oom_rows) else None

        L_a4500 = ok['seq_len'].to_numpy(dtype=float)
        V_a4500 = ok['vram_min_alloc_gb'].to_numpy(dtype=float)

        # Append any clean >20 GB B=1 measurement from the 32 GB card (e.g. ESM-MSR L=2048 = 23.62 GB)
        L_ext, V_ext = np.array([]), np.array([])
        if not df_ext.empty:
            esub = df_ext[(df_ext['model'] == key) & (df_ext['status'] == 'ok')].sort_values('seq_len')
            esub = esub[esub['seq_len'] > (L_a4500.max() if len(L_a4500) else 0)]
            if len(esub):
                L_ext = esub['seq_len'].to_numpy(dtype=float)
                V_ext = esub['vram_min_alloc_gb'].to_numpy(dtype=float)

        L_all = np.concatenate([L_a4500, L_ext])
        V_all = np.concatenate([V_a4500, V_ext])
        a, b, c = fit_vram_model(key, L_all, V_all, oom_row)

        out.append(dict(
            fam=fam,
            L_a4500=L_a4500, V_a4500=V_a4500,
            L_ext=L_ext, V_ext=V_ext,
            L_all=L_all, V_all=V_all,
            oom_a4500=float(oom_row['seq_len']) if oom_row is not None else np.nan,
            a=a, b=b, c=c,
            L_20=solve_L_for_vram(a, b, c, 20.0),
            L_30=solve_L_for_vram(a, b, c, 30.0),
            L_40=solve_L_for_vram(a, b, c, 40.0),
        ))
    return out


def plot_scaling(df: pd.DataFrame, df_ext: pd.DataFrame, out_dir: Path) -> tuple[Path, list[dict]]:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.5))
    ax_vram, ax_1m, ax_2m = axes

    fits = compute_vram_fits(df, df_ext)

    # ---- Panel A: Minimum VRAM (B=1) vs Protein Length L (measured + extrapolated to >= 40 GB) ----
    max_L_vram = 65536.0
    xt_vram = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for f in fits:
        fam = f['fam']
        z = fam['z']
        # Measured solid curve (A4500 + clean 32 GB card points)
        ax_vram.plot(
            f['L_all'], f['V_all'],
            color=fam['color'], ls=fam['ls'], lw=fam['lw'], alpha=0.95,
            marker=fam['marker'], ms=5.8 if fam['key'] == 'esm_msr_wt' else 5.2,
            mfc=fam['color'], mec=fam['color'], mew=0.8,
            label=fam['label'], zorder=z,
        )
        # Highlight any >20 GB 32GB-card measured point with a crisp outer ring
        if len(f['L_ext']):
            ax_vram.scatter(
                f['L_ext'], f['V_ext'],
                s=115, facecolors='none', edgecolors=fam['color'], linewidths=1.4, zorder=z + 2,
            )
        # Smooth physics-based extrapolation from last measured point up to 44 GB
        L_start = float(f['L_all'][-1])
        V_start = float(f['V_all'][-1])
        L_end = solve_L_for_vram(f['a'], f['b'], f['c'], 44.0)
        if np.isfinite(L_end) and L_end > L_start:
            L_grid = np.geomspace(L_start, min(L_end, max_L_vram * 1.15), 80)
            V_grid = f['a'] + f['b'] * L_grid + f['c'] * (L_grid ** 2)
            # Anchor start seamlessly to last observed point
            V_grid = V_grid + (V_start - V_grid[0]) * np.exp(-3.0 * (L_grid - L_start) / max(1.0, L_end - L_start))
            ax_vram.plot(L_grid, V_grid, color=fam['color'], ls='--', lw=1.35, alpha=0.65, zorder=z - 1)
        # Point markers at 30 GB (usable 32 GB card) and 40 GB (A100) crossings
        for target_v, L_cross in [(30.0, f['L_30']), (40.0, f['L_40'])]:
            if np.isfinite(L_cross) and L_cross <= max_L_vram * 1.15:
                ax_vram.scatter(
                    [L_cross], [target_v], s=32, marker='o',
                    facecolors='white', edgecolors=fam['color'], linewidths=1.4, zorder=z + 2,
                )

    # Horizontal GPU memory capacity tiers (clean white backing so curves never collide with text)
    lbl_bg = dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.9)
    ax_vram.axhline(20.0, color=style.RING, ls='--', lw=1.15, zorder=3)
    ax_vram.text(28, 20.55, '20 GB (RTX A4500)', fontsize=7.2, color=style.RING, fontweight='bold', ha='left', va='bottom', bbox=lbl_bg, zorder=9)
    ax_vram.axhline(30.0, color='#555555', ls='--', lw=1.05, zorder=3)
    ax_vram.text(28, 29.45, '~30 GB (usable on 32 GB GPU)', fontsize=7.1, color='#444444', fontweight='bold', ha='left', va='top', bbox=lbl_bg, zorder=9)
    ax_vram.axhline(40.0, color='#555555', ls=':', lw=1.05, zorder=3)
    ax_vram.text(65536 * 1.18, 40.55, '40 GB (A100)', fontsize=7.2, color='#444444', fontweight='bold', ha='right', va='bottom', bbox=lbl_bg, zorder=9)

    ax_vram.axvspan(2048, max_L_vram * 1.30, color='#f2f2f2', zorder=0)
    ax_vram.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    ax_vram.text(2048 * 1.15, 13.5, 'tiled\nextensions', fontsize=6.9, color='#777777', va='center', ha='left')

    # Compact upper-left inset table of estimated L_max at 20 / ~30 / 40 GB
    tbl_lines = ['Est. max L (B=1)  20G   30G   40G']
    tbl_lines.append('─' * 33)
    for f in sorted(fits, key=lambda x: x['L_30']):
        tbl_lines.append(
            f"{f['fam']['short']:<16s} {_fmt_L(f['L_20']):>5s} {_fmt_L(f['L_30']):>5s} {_fmt_L(f['L_40']):>5s}"
        )
    ax_vram.text(
        0.025, 0.975, '\n'.join(tbl_lines),
        transform=ax_vram.transAxes, va='top', ha='left',
        fontsize=6.5, family='monospace', color='#222222',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#cccccc', alpha=0.93, lw=0.8),
        zorder=10,
    )

    ax_vram.set_xscale('log')
    ax_vram.set_xlim(24, max_L_vram * 1.30)
    ax_vram.set_ylim(0, 45.5)
    ax_vram.set_xticks(xt_vram)
    ax_vram.set_xticklabels([f'{v // 1024}k' if v >= 8192 else str(v) for v in xt_vram], fontsize=7.6, rotation=32, ha='right')
    ax_vram.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_vram.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_vram.set_ylabel('Minimum peak allocated VRAM at $B=1$ (GB)')
    ax_vram.set_title('A. Min VRAM ($B=1$) & 30–40 GB extrapolation', loc='left')
    ax_vram.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # ---- Panel B: Full single-mutant screen time (19L variants, RTX A4500 20 GB) ----
    max_L_scr = 32768.0
    xt_scr = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for s in SINGLE_SPECS:
        sub = df[(df['model'] == s['key']) & (df['status'] == 'ok')].sort_values('seq_len')
        ok = sub[sub['single_screen_s'].notna() & (sub['single_screen_s'] > 0)]
        if ok.empty:
            continue
        _plot_curve(ax_1m, ok['seq_len'], ok['single_screen_s'], s, zorder=s['z'])
    ax_1m.axvspan(2048, max_L_scr * 1.30, color='#f2f2f2', zorder=0)
    ax_1m.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    for yref, txt in [(0.1, '0.1 s'), (1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h'), (86400.0, '1 d')]:
        ax_1m.axhline(yref, color='#888888', ls=':', lw=0.75, zorder=1)
        ax_1m.text(max_L_scr * 1.18, yref * 1.15, txt, fontsize=7.2, color='#666666', ha='right')
    ax_1m.set_xscale('log')
    ax_1m.set_yscale('log')
    ax_1m.set_xlim(24, max_L_scr * 1.30)
    ax_1m.set_ylim(5e-3, 1.5e5)
    ax_1m.set_xticks(xt_scr)
    ax_1m.set_xticklabels([f'{v // 1024}k' if v >= 8192 else str(v) for v in xt_scr], fontsize=7.8, rotation=30, ha='right')
    ax_1m.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_1m.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_1m.set_ylabel('Full single-mutant screen ($19L$ variants) time (s)')
    ax_1m.set_title('B. Full single-mutant screen (RTX A4500 20 GB)', loc='left')
    ax_1m.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # ---- Panel C: Full double-mutant screen time (361 * L*(L-1)/2, RTX A4500 20 GB, 24 h cap) ----
    CAP_24H = 86400.0
    for s in DOUBLE_SPECS:
        sub = df[(df['model'] == s['key']) & (df['status'] == 'ok')].sort_values('seq_len')
        ok = sub[sub['double_screen_s'].notna() & (sub['double_screen_s'] > 0)]
        if ok.empty:
            continue
        above = ok[ok['double_screen_s'] >= CAP_24H]
        if len(above):
            cutoff_L = float(above['seq_len'].iloc[0])
            ok_plot = ok[ok['seq_len'] <= cutoff_L]
            _plot_curve(ax_2m, ok_plot['seq_len'], ok_plot['double_screen_s'], s, zorder=s['z'])
            ax_2m.scatter([cutoff_L], [float(above['double_screen_s'].iloc[0])],
                          marker='^', s=62, facecolors='none', edgecolors=style.RING, linewidths=1.5, zorder=s['z'] + 2)
        else:
            _plot_curve(ax_2m, ok['seq_len'], ok['double_screen_s'], s, zorder=s['z'])
    # Reference dotted line for additive-only ESM-MSR WT-view (1 WT pass; no epistasis)
    wt_ok = df[(df['model'] == 'esm_msr_wt') & (df['status'] == 'ok')].sort_values('seq_len')
    if len(wt_ok):
        ax_2m.plot(
            wt_ok['seq_len'], wt_ok['single_screen_s'],
            color=C_ESM_BLACK, ls=':', lw=1.6, alpha=0.65,
            label='ESM-MSR WT-view (additive only)', zorder=3,
        )
    ax_2m.axvspan(2048, max_L_scr * 1.30, color='#f2f2f2', zorder=0)
    ax_2m.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    ax_2m.axhline(CAP_24H, color=style.RING, ls='--', lw=1.3, zorder=7)
    ax_2m.axhspan(CAP_24H, 5e6, color=style.RING, alpha=0.07, zorder=0)
    ax_2m.text(28, CAP_24H * 1.35, '24 h cap (larger proteins impractical)', fontsize=7.5,
               color=style.RING, fontweight='bold', va='bottom', ha='left')
    for yref, txt in [(1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h'), (86400.0, '24 h'), (86400.0 * 30, '1 mo')]:
        if yref != CAP_24H:
            ax_2m.axhline(yref, color='#888888', ls=':', lw=0.75, zorder=1)
        ax_2m.text(max_L_scr * 1.18, yref * 1.18, txt, fontsize=7.2, color=style.RING if yref == CAP_24H else '#666666', ha='right')
    ax_2m.set_xscale('log')
    ax_2m.set_yscale('log')
    ax_2m.set_xlim(24, max_L_scr * 1.30)
    ax_2m.set_ylim(2e-2, 5e6)
    ax_2m.set_xticks(xt_scr)
    ax_2m.set_xticklabels([f'{v // 1024}k' if v >= 8192 else str(v) for v in xt_scr], fontsize=7.8, rotation=30, ha='right')
    ax_2m.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_2m.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_2m.set_ylabel('Full double-mutant screen ($361\\binom{L}{2}$ variants) time (s)')
    ax_2m.set_title('C. Full double-mutant screen (RTX A4500, 24 h cap)', loc='left')
    ax_2m.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # Clean, unified top legend with non-redundant entries + visual key for extrapolation
    legend_handles = [
        Line2D([0], [0], color=C_ESM_BLACK, marker='D', ms=5.8, mfc=C_ESM_BLACK, mec=C_ESM_BLACK, ls='-', lw=2.5, label='ESM-MSR (Dual-view; Panel A: all variants)'),
        Line2D([0], [0], color=C_ESM_DARK, marker='s', ms=5.6, mfc=C_ESM_DARK, mec=C_ESM_DARK, ls='-', lw=2.1, label='ESM-MSR (Indep. masking; B: +Marginal)'),
        Line2D([0], [0], color=C_ESM_LIGHT, marker='^', ms=5.6, mfc=C_ESM_LIGHT, mec=C_ESM_LIGHT, ls='-', lw=2.1, label='ESM-MSR (Masked marginal, C)'),
        Line2D([0], [0], color=C_ESM_BLACK, marker='D', ms=5.6, mfc='white', mec=C_ESM_BLACK, mew=1.5, ls='--', lw=2.2, label='ESM-MSR (WT-view; C: dotted additive)'),
        Line2D([0], [0], color=BY_NAME['SPURS'].color, marker='s', ms=5.2, mfc=BY_NAME['SPURS'].color, mec=BY_NAME['SPURS'].color, ls='-', lw=2.0, label='SPURS-multi (hollow: single)'),
        Line2D([0], [0], color=BY_NAME['ThermoMPNN'].color, marker='p', ms=5.4, mfc=BY_NAME['ThermoMPNN'].color, mec=BY_NAME['ThermoMPNN'].color, ls='-', lw=2.0, label='ThermoMPNN-D (hollow: single SSM)'),
        Line2D([0], [0], color=BY_NAME['ProteinMPNN'].color, marker='d', ms=5.2, mfc=BY_NAME['ProteinMPNN'].color, mec=BY_NAME['ProteinMPNN'].color, ls='-.', lw=2.0, label='ProteinMPNN (zero-shot)'),
        Line2D([0], [0], color=BY_NAME['MutateEverything'].color, marker='h', ms=5.4, mfc=BY_NAME['MutateEverything'].color, mec=BY_NAME['MutateEverything'].color, ls='-', lw=2.0, label='MutateEverything (St2; A–B: St1/St2)'),
        Line2D([0], [0], color=BY_NAME['ProSST'].color, marker='X', ms=5.4, mfc=BY_NAME['ProSST'].color, mec=BY_NAME['ProSST'].color, ls=':', lw=2.0, label='ProSST-2048 (zero-shot)'),
        Line2D([0], [0], color='#555555', ls='--', lw=1.35, marker='o', ms=4.8, mfc='white', mec='#555555', mew=1.3, label='Panel A: quadratic VRAM fit (30 / 40 GB)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=7.9, frameon=False, columnspacing=1.2)

    stem = out_dir / '03_size_scaling_vram_and_screening_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.825,
        title='Protein-Length Scaling of Minimum VRAM (20–40 GB) and Full Mutational Screening Cost (1× RTX A4500 20 GB)',
        note=(
            'Panel A shows process-allocated peak VRAM at B = 1 across exact power-of-two lengths (real ProteinGym AF2 structures & MSAs up to L = 2,048; '
            'tiled extensions above 2,048). Points ≤ 20 GB were measured on RTX A4500 (20 GB); circled ESM-MSR L = 2,048 (23.6 GB) was measured at B = 1 on a 32 GB RTX 5090. '
            'Dashed Panel A curves fit V(L) = a + bL + cL² to estimate max length at ~30 GB (usable on a 32 GB GPU leaving system headroom) and 40 GB (A100). '
            'Panels B–C show strictly apples-to-apples RTX A4500 optimal-batch screening times; identical curves are merged per panel (ESM-MSR Indep./Marginal share 2L single-mutant passes).'
        ),
    )
    return stem, fits


def write_summary_md(df: pd.DataFrame, fits: list[dict], out_dir: Path) -> None:
    lines = [
        '# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB + 30/40 GB VRAM Extrapolation)',
        '',
        '## Estimated Maximum Protein Length ($B = 1$ PyTorch-Allocated VRAM)',
        '',
        '| Model Family | Measured Max `L` | VRAM @ Max `L` | A4500 OOM `L` | Est. `L_max` @ 20 GB | Est. `L_max` @ ~30 GB (32 GB usable) | Est. `L_max` @ 40 GB |',
        '|---|---|---|---|---|---|---|',
    ]
    for f in sorted(fits, key=lambda x: x['L_30']):
        oom_s = f"{int(f['oom_a4500']):,}" if np.isfinite(f['oom_a4500']) else '—'
        lines.append(
            f"| {f['fam']['label']} | {int(f['L_all'][-1]):,} | {f['V_all'][-1]:.2f} GB | {oom_s} | "
            f"{int(round(f['L_20'])):,} | {int(round(f['L_30'])):,} | {int(round(f['L_40'])):,} |"
        )
    lines += [
        '',
        '## Full Mutational Screening Cost (RTX A4500 20 GB Fixed Slot)',
        '',
        '| Model | Max `L` ok (A4500) | First OOM `L` | Weights GB | VRAM @ `L=128` | VRAM @ `L=512` | VRAM @ `L=1024` | 1-mut @ `L=256` | 1-mut @ `L=1024` | 2-mut @ `L=256` | 2-mut @ `L=1024` |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for s in SPECS:
        sub = df[df['model'] == s['key']].sort_values('seq_len')
        if sub.empty:
            continue
        ok = sub[sub['status'] == 'ok']
        oom = sub[sub['status'] == 'oom']
        max_ok = int(ok['seq_len'].max()) if len(ok) else 0
        oom_l = str(int(oom['seq_len'].iloc[0])) if len(oom) else f'>{max_ok}'
        w_gb = f'{ok["vram_weights_gb"].median():.2f}' if len(ok) else '—'

        def _at(target_bin, col, is_time=False):
            m = ok[ok['bin'] == target_bin]
            if m.empty or col not in m.columns:
                return '—'
            if col == 'double_screen_s' and 'double_screen_method' in m.columns and str(m['double_screen_method'].iloc[0]) == 'impractical_gt_24h':
                return 'impractical (>24 h)'
            if pd.isna(m[col].iloc[0]):
                return '—'
            v = float(m[col].iloc[0])
            if is_time:
                tag = _fmt_dur(v)
                if col == 'double_screen_s' and v >= 86400.0:
                    tag += ' (cap)'
                return tag
            return f'{v:.2f} GB'

        lines.append(
            f"| {s['label']} | {max_ok:,} | {oom_l} | {w_gb} | "
            f"{_at(128, 'vram_min_alloc_gb')} | {_at(512, 'vram_min_alloc_gb')} | {_at(1024, 'vram_min_alloc_gb')} | "
            f"{_at(256, 'single_screen_s', True)} | {_at(1024, 'single_screen_s', True)} | "
            f"{_at(256, 'double_screen_s', True)} | {_at(1024, 'double_screen_s', True)} |"
        )
    (out_dir / 'size_scaling_a4500_summary.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--in_dir', default=str(MSR_ROOT / 'size_scaling_a4500'))
    ap.add_argument('--vram_ext_dir', default=str(MSR_ROOT / 'size_scaling_5090_vram'))
    ap.add_argument('--out_dir', default=str(INFERENCE_RESOURCES_DIR))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    ext_dir = Path(args.vram_ext_dir) if args.vram_ext_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(in_dir)
    if df.empty:
        raise SystemExit(f'No size-scaling results found in {in_dir}')
    df_ext = load_vram_ext(ext_dir)
    df.to_csv(out_dir / 'size_scaling_a4500_all.csv', index=False)
    stem, fits = plot_scaling(df, df_ext, out_dir)
    write_summary_md(df, fits, out_dir)
    print(f'Wrote: {stem}.png')


if __name__ == '__main__':
    main()
