"""Plot controlled protein-size scaling benchmark results (RTX A4500 20 GB + 95%-card step/half-step + 40 GB quadratic fit).

Reads <in_dir>/<model>/results.csv produced by benchmark_size_scaling.py (RTX A4500 20 GB fixed slot + CPU-only WT-view)
plus <vram_ext_dir>/<model>/results.csv (95%-allowance single next-step / half-step B=1 VRAM point on RTX 5090 32 GB)
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
C_ESM_MID = '#6e6e6e'
C_ESM_LIGHT = '#8c8c8c'

SPECS = [
    dict(key='esm_msr_wt', label='ESM-MSR (WT-view)', color=C_ESM_BLACK, marker='D', hollow=True, ls='--', lw=2.2),
    dict(key='esm_msr_wt_cpu', label='ESM-MSR (WT-view, 6-core CPU)', color=C_ESM_MID, marker='o', hollow=True, ls=':', lw=2.1),
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

# Deduplicated model families for Panel A (models with identical B=1 VRAM merged into one curve)
VRAM_FAMILIES = [
    dict(key='esm_msr_wt', label='ESM-MSR (all variants)', short='ESM-MSR (all)', color=C_ESM_BLACK, marker='D', ls='-', lw=2.4, z=6),
    dict(key='me_st2', label='MutateEverything (St1 / St2)', short='MutateEverything', color=BY_NAME['MutateEverything'].color, marker='h', ls='-', lw=2.1, z=5),
    dict(key='prosst', label='ProSST-2048', short='ProSST-2048', color=BY_NAME['ProSST'].color, marker='X', ls=':', lw=2.1, z=4),
    dict(key='spurs_single', label='SPURS (single / multi)', short='SPURS', color=BY_NAME['SPURS'].color, marker='s', ls='-', lw=2.1, z=4),
    dict(key='proteinmpnn', label='ProteinMPNN', short='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', ls='-.', lw=2.1, z=4),
    dict(key='thermompnn_d', label='ThermoMPNN / -D', short='ThermoMPNN / -D', color=BY_NAME['ThermoMPNN'].color, marker='p', ls='-', lw=2.1, z=4),
]

# Deduplicated single-mutant specs for Panel B (SPURS single only; CPU WT-view added; indep+marg & ME St1+St2 merged)
SINGLE_SPECS = [
    dict(key='esm_msr_dual', label='ESM-MSR (Dual-view)', color=C_ESM_BLACK, marker='D', hollow=False, ls='-', lw=2.5, z=6),
    dict(key='esm_msr_indep', label='ESM-MSR (Indep. / Marginal)', color=C_ESM_DARK, marker='s', hollow=False, ls='-', lw=2.1, z=6),
    dict(key='esm_msr_wt_cpu', label='ESM-MSR (WT-view, 6-CPU)', color=C_ESM_MID, marker='o', hollow=True, ls=':', lw=2.1, z=6),
    dict(key='esm_msr_wt', label='ESM-MSR (WT-view, GPU)', color=C_ESM_BLACK, marker='D', hollow=True, ls='--', lw=2.2, z=6),
    dict(key='me_st2', label='MutateEverything (St1 / St2)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0, z=4),
    dict(key='prosst', label='ProSST-2048', color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0, z=4),
    dict(key='proteinmpnn', label='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0, z=4),
    dict(key='spurs_single', label='SPURS (single)', color=BY_NAME['SPURS'].color, marker='s', hollow=True, ls='--', lw=2.0, z=4),
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

VRAM_TIERS = [16.0, 24.0, 32.0, 40.0]


def _plot_curve(ax, x, y, s, zorder=4, alpha=0.95, label=None):
    hollow = s.get('hollow', False)
    ax.plot(
        x, y,
        color=s['color'], ls=s['ls'], lw=s['lw'], alpha=alpha,
        marker=s['marker'], ms=5.7 if 'esm_msr' in s['key'] else 5.2,
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


def fit_final3_quadratic(L_obs: np.ndarray, V_obs: np.ndarray) -> tuple[float, float, float]:
    """Fit non-negative quadratic V(L) = a + b*L + c*L^2 to the final 3 empirical B=1 measurements."""
    L3 = L_obs[-3:]
    V3 = V_obs[-3:]
    X = np.column_stack([np.ones_like(L3), L3, L3 ** 2])
    coef, _ = nnls(X, V3)
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
    """Combine A4500 power-of-two points with the single 95%-cap next/half-step point and fit the final 3 points."""
    out = []
    for fam in VRAM_FAMILIES:
        key = fam['key']
        sub = df[df['model'] == key].sort_values('seq_len')
        if sub.empty:
            continue
        ok = sub[sub['status'] == 'ok'].copy()

        L_a4500 = ok['seq_len'].to_numpy(dtype=float)
        V_a4500 = ok['vram_min_alloc_gb'].to_numpy(dtype=float)

        L_ext, V_ext = np.array([]), np.array([])
        if not df_ext.empty:
            esub = df_ext[(df_ext['model'] == key) & (df_ext['status'] == 'ok')].sort_values('seq_len')
            esub = esub[esub['seq_len'] > (L_a4500.max() if len(L_a4500) else 0)].head(1)
            if len(esub):
                L_ext = esub['seq_len'].to_numpy(dtype=float)
                V_ext = esub['vram_min_alloc_gb'].to_numpy(dtype=float)

        L_all = np.concatenate([L_a4500, L_ext])
        V_all = np.concatenate([V_a4500, V_ext])
        a, b, c = fit_final3_quadratic(L_all, V_all)
        tier_L = {int(t): solve_L_for_vram(a, b, c, t) for t in VRAM_TIERS}

        out.append(dict(
            fam=fam,
            L_all=L_all, V_all=V_all,
            a=a, b=b, c=c,
            tier_L=tier_L,
        ))
    return out


def plot_scaling(df: pd.DataFrame, df_ext: pd.DataFrame, out_dir: Path) -> tuple[Path, list[dict]]:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.5))
    ax_vram, ax_1m, ax_2m = axes

    fits = compute_vram_fits(df, df_ext)

    # ---- Panel A: Minimum VRAM (B=1) vs Protein Length L (clean solid curves + final-3 quadratic fit to 40 GB) ----
    max_L_vram = 65536.0
    xt_vram = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for f in fits:
        fam = f['fam']
        z = fam['z']
        ax_vram.plot(
            f['L_all'], f['V_all'],
            color=fam['color'], ls=fam['ls'], lw=fam['lw'], alpha=0.95,
            marker=fam['marker'], ms=5.6 if fam['key'] == 'esm_msr_wt' else 5.0,
            mfc=fam['color'], mec=fam['color'], mew=0.7,
            label=fam['label'], zorder=z,
        )
        # Clean dashed quadratic continuation from last measured point up to 44 GB (no extra circles)
        L_start = float(f['L_all'][-1])
        V_start = float(f['V_all'][-1])
        L_end = solve_L_for_vram(f['a'], f['b'], f['c'], 44.0)
        if np.isfinite(L_end) and L_end > L_start:
            L_grid = np.geomspace(L_start, min(L_end, max_L_vram * 1.15), 80)
            V_grid = f['a'] + f['b'] * L_grid + f['c'] * (L_grid ** 2)
            V_grid = V_grid + (V_start - V_grid[0]) * np.exp(-3.0 * (L_grid - L_start) / max(1.0, L_end - L_start))
            ax_vram.plot(L_grid, V_grid, color=fam['color'], ls='--', lw=1.4, alpha=0.62, zorder=z - 1)

    # Horizontal common GPU VRAM card-size reference lines: 16, 24, 32, 40 GB
    lbl_bg = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8)
    for tv, col, lstyle, x_pos, ha_s in [
        (16.0, '#777777', ':',  28,           'left'),
        (24.0, '#666666', '--', 28,           'left'),
        (32.0, '#444444', '--', 65536 * 1.18, 'right'),
        (40.0, '#222222', '-',  65536 * 1.18, 'right'),
    ]:
        ax_vram.axhline(tv, color=col, ls=lstyle, lw=1.0, alpha=0.82, zorder=2)
        ax_vram.text(
            x_pos, tv + 0.45, f'{int(tv)} GB', fontsize=7.3, color=col,
            fontweight='bold', ha=ha_s, va='bottom', bbox=lbl_bg, zorder=9,
        )

    ax_vram.axvspan(2048, max_L_vram * 1.30, color='#f2f2f2', zorder=0)
    ax_vram.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    ax_vram.text(2048 * 1.15, 11.2, 'tiled\nextensions', fontsize=6.9, color='#777777', va='center', ha='left')

    # Compact upper-left inset table of estimated L_max at 16 / 24 / 32 / 40 GB
    tbl_lines = ['Est. max L         16G   24G   32G   40G']
    tbl_lines.append('─' * 40)
    for f in sorted(fits, key=lambda x: x['tier_L'][32]):
        tl = f['tier_L']
        tbl_lines.append(
            f"{f['fam']['short']:<17s}{_fmt_L(tl[16]):>5s} {_fmt_L(tl[24]):>5s} {_fmt_L(tl[32]):>5s} {_fmt_L(tl[40]):>5s}"
        )
    ax_vram.text(
        0.022, 0.978, '\n'.join(tbl_lines),
        transform=ax_vram.transAxes, va='top', ha='left',
        fontsize=6.2, family='monospace', color='#222222',
        bbox=dict(boxstyle='round,pad=0.32', facecolor='white', edgecolor='#cccccc', alpha=0.94, lw=0.8),
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
    ax_vram.set_title('A. Min VRAM ($B=1$) & quadratic fit to 40 GB', loc='left')
    ax_vram.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # ---- Panel B: Full single-mutant screen time (19L variants: RTX A4500 20 GB + 6-core CPU ESM-MSR WT) ----
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
    ax_1m.set_title('B. Full single-mutant screen (RTX A4500 & 6-CPU)', loc='left')
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
        else:
            _plot_curve(ax_2m, ok['seq_len'], ok['double_screen_s'], s, zorder=s['z'])
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
    ax_2m.axhspan(CAP_24H, 8e6, color=style.RING, alpha=0.07, zorder=0)
    ax_2m.text(
        0.03, 0.96, '24 h cap (larger proteins impractical)',
        transform=ax_2m.transAxes, fontsize=7.6, color=style.RING, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#fdf2f2', edgecolor=style.RING, alpha=0.92, lw=0.6),
        zorder=10,
    )
    for yref, txt in [(1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h'), (86400.0, '24 h'), (86400.0 * 30, '1 mo')]:
        if yref != CAP_24H:
            ax_2m.axhline(yref, color='#888888', ls=':', lw=0.75, zorder=1)
        ax_2m.text(max_L_scr * 1.18, yref * 1.18, txt, fontsize=7.2, color=style.RING if yref == CAP_24H else '#666666', ha='right')
    ax_2m.set_xscale('log')
    ax_2m.set_yscale('log')
    ax_2m.set_xlim(24, max_L_scr * 1.30)
    ax_2m.set_ylim(2e-2, 8e6)
    ax_2m.set_xticks(xt_scr)
    ax_2m.set_xticklabels([f'{v // 1024}k' if v >= 8192 else str(v) for v in xt_scr], fontsize=7.8, rotation=30, ha='right')
    ax_2m.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_2m.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_2m.set_ylabel('Full double-mutant screen ($361\\binom{L}{2}$ variants) time (s)')
    ax_2m.set_title('C. Full double-mutant screen (RTX A4500, 24 h cap)', loc='left')
    ax_2m.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # Clean, unified top legend
    legend_handles = [
        Line2D([0], [0], color=C_ESM_BLACK, marker='D', ms=5.6, mfc=C_ESM_BLACK, mec=C_ESM_BLACK, ls='-', lw=2.4, label='ESM-MSR (Dual-view; A: all variants)'),
        Line2D([0], [0], color=C_ESM_DARK, marker='s', ms=5.5, mfc=C_ESM_DARK, mec=C_ESM_DARK, ls='-', lw=2.1, label='ESM-MSR (Indep. masking; B: +Marginal)'),
        Line2D([0], [0], color=C_ESM_LIGHT, marker='^', ms=5.5, mfc=C_ESM_LIGHT, mec=C_ESM_LIGHT, ls='-', lw=2.1, label='ESM-MSR (Masked marginal, C)'),
        Line2D([0], [0], color=C_ESM_BLACK, marker='D', ms=5.5, mfc='white', mec=C_ESM_BLACK, mew=1.5, ls='--', lw=2.2, label='ESM-MSR (WT-view, GPU; C: dotted additive)'),
        Line2D([0], [0], color=C_ESM_MID, marker='o', ms=5.5, mfc='white', mec=C_ESM_MID, mew=1.5, ls=':', lw=2.1, label='ESM-MSR (WT-view, 6-core CPU, B)'),
        Line2D([0], [0], color=BY_NAME['SPURS'].color, marker='s', ms=5.2, mfc=BY_NAME['SPURS'].color, mec=BY_NAME['SPURS'].color, ls='-', lw=2.0, label='SPURS-multi (hollow: single, B)'),
        Line2D([0], [0], color=BY_NAME['ThermoMPNN'].color, marker='p', ms=5.4, mfc=BY_NAME['ThermoMPNN'].color, mec=BY_NAME['ThermoMPNN'].color, ls='-', lw=2.0, label='ThermoMPNN-D (hollow: single SSM)'),
        Line2D([0], [0], color=BY_NAME['ProteinMPNN'].color, marker='d', ms=5.2, mfc=BY_NAME['ProteinMPNN'].color, mec=BY_NAME['ProteinMPNN'].color, ls='-.', lw=2.0, label='ProteinMPNN (zero-shot)'),
        Line2D([0], [0], color=BY_NAME['MutateEverything'].color, marker='h', ms=5.4, mfc=BY_NAME['MutateEverything'].color, mec=BY_NAME['MutateEverything'].color, ls='-', lw=2.0, label='MutateEverything (St2; A–B: St1/St2)'),
        Line2D([0], [0], color=BY_NAME['ProSST'].color, marker='X', ms=5.4, mfc=BY_NAME['ProSST'].color, mec=BY_NAME['ProSST'].color, ls=':', lw=2.0, label='ProSST-2048 (zero-shot)'),
        Line2D([0], [0], color='#555555', ls='--', lw=1.4, label='Panel A: final-3-point quadratic fit to 40 GB'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=7.7, frameon=False, columnspacing=1.1)

    stem = out_dir / '03_size_scaling_vram_and_screening_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.82,
        title='Protein-Length Scaling of Minimum VRAM (16–40 GB) and Full Mutational Screening Cost (1× RTX A4500 20 GB)',
        note=(
            'Panel A shows process-allocated peak VRAM at B = 1 across exact power-of-two lengths (real ProteinGym AF2 structures & MSAs up to L = 2,048; '
            'tiled extensions above 2,048), extended by one 95%-card-cap (zero-overflow) step on a 32 GB GPU: 2×L_ok if it fits (ESM-MSR 2,048; ProSST 8,192) '
            'or 1.5×L_ok (MutateEverything 1,536; SPURS 12,288; ProteinMPNN & ThermoMPNN/-D 24,576). Dashed curves fit V(L) = a + bL + cL² to each model’s final 3 '
            'empirical points to estimate capacity at 16 / 24 / 32 / 40 GB. Panel B shows RTX A4500 optimal-batch single-mutant times plus 6-core CPU-only ESM-MSR WT-view '
            '(60 GB RAM, reaching L = 2,048 in 54 s); Panel C shows RTX A4500 double-mutant times with a 24 h practicality cap.'
        ),
    )
    return stem, fits


def write_summary_md(df: pd.DataFrame, fits: list[dict], out_dir: Path) -> None:
    lines = [
        '# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB + 95%-Cap Half-Step Probes & 40 GB Quadratic Fit)',
        '',
        '## Estimated Maximum Protein Length ($B = 1$ PyTorch-Allocated VRAM, Final-3-Point Quadratic Fit)',
        '',
        '| Model Family | Final 3 Empirical Points `(L, GB)` | Est. `L_max` @ 16 GB | Est. `L_max` @ 24 GB | Est. `L_max` @ 32 GB | Est. `L_max` @ 40 GB |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for f in sorted(fits, key=lambda x: x['tier_L'][32]):
        tl = f['tier_L']
        pts = ', '.join(f'({int(l):,}, {v:.2f})' for l, v in zip(f['L_all'][-3:], f['V_all'][-3:]))
        lines.append(
            f"| {f['fam']['label']} | {pts} | "
            f"{int(round(tl[16])):,} | {int(round(tl[24])):,} | {int(round(tl[32])):,} | {int(round(tl[40])):,} |"
        )
    lines += [
        '',
        '## Full Mutational Screening Cost (RTX A4500 20 GB Fixed Slot + 6-Core CPU ESM-MSR WT)',
        '',
        '| Model | Max `L` ok | First OOM `L` | Weights GB | VRAM/RAM @ `L=128` | VRAM/RAM @ `L=512` | VRAM/RAM @ `L=1024` | 1-mut @ `L=256` | 1-mut @ `L=1024` | 2-mut @ `L=256` | 2-mut @ `L=1024` |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for s in SPECS:
        sub = df[df['model'] == s['key']].sort_values('seq_len')
        if sub.empty:
            continue
        ok = sub[sub['status'] == 'ok']
        oom = sub[sub['status'] == 'oom']
        max_ok = int(ok['seq_len'].max()) if len(ok) else 0
        oom_l = str(int(oom['seq_len'].iloc[0])) if len(oom) else ('4096 (60G RAM)' if s['key'] == 'esm_msr_wt_cpu' else f'>{max_ok}')
        w_gb = f'{ok["vram_weights_gb"].median():.2f}' if ('vram_weights_gb' in ok.columns and ok['vram_weights_gb'].notna().any()) else '—'
        mem_col = 'ram_rss_peak_gb' if s['key'] == 'esm_msr_wt_cpu' else 'vram_min_alloc_gb'

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
            f"{_at(128, mem_col)} | {_at(512, mem_col)} | {_at(1024, mem_col)} | "
            f"{_at(256, 'single_screen_s', True)} | {_at(1024, 'single_screen_s', True)} | "
            f"{_at(256, 'double_screen_s', True)} | {_at(1024, 'double_screen_s', True)} |"
        )
    (out_dir / 'size_scaling_a4500_summary.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--in_dir', default=str(MSR_ROOT / 'size_scaling_a4500'))
    ap.add_argument('--vram_ext_dir', default=str(MSR_ROOT / 'size_scaling_halfstep_95'))
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
