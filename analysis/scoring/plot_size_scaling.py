"""Plot controlled protein-size scaling benchmark results (RTX A4500 20 GB, fixed slot).

Reads <in_dir>/<model>/results.csv produced by benchmark_size_scaling.py and generates:
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
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'analysis'))

from msr_analysis.paths import INFERENCE_RESOURCES_DIR, MSR_ROOT  # noqa: E402
from msr_analysis import style  # noqa: E402
from msr_analysis.models import BY_NAME  # noqa: E402

style.setup()

SPECS = [
    dict(key='esm_msr_wt', label='ESM-MSR (WT-view)', color=BY_NAME['ESM-MSR'].color, marker='D', hollow=True, ls='--', lw=2.3),
    dict(key='esm_msr_dual', label='ESM-MSR (Dual-view)', color=BY_NAME['ESM-MSR'].color, marker='D', hollow=False, ls='-', lw=2.6),
    dict(key='esm_msr_indep', label='ESM-MSR (Indep. masking)', color='#2c5f8a', marker='s', hollow=False, ls='-', lw=2.0),
    dict(key='esm_msr_marg', label='ESM-MSR (Masked marginal)', color='#5b8db8', marker='^', hollow=False, ls='-', lw=2.0),
    dict(key='spurs_single', label='SPURS (single)', color=BY_NAME['SPURS'].color, marker='s', hollow=True, ls='--', lw=1.9),
    dict(key='spurs_multi', label='SPURS-multi', color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0),
    dict(key='thermompnn_single', label='ThermoMPNN (SSM single)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=True, ls='--', lw=1.9),
    dict(key='thermompnn_d', label='ThermoMPNN-D (SSM double)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=False, ls='-', lw=2.0),
    dict(key='proteinmpnn', label='ProteinMPNN (zero-shot)', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0),
    dict(key='me_st1', label='MutateEverything (Stage 1)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=True, ls='--', lw=1.9),
    dict(key='me_st2', label='MutateEverything (Stage 2)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0),
    dict(key='prosst', label='ProSST-2048 (zero-shot)', color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0),
]


def _plot_curve(ax, x, y, s, zorder=4, alpha=0.95, label=None):
    ax.plot(
        x, y,
        color=s['color'], ls=s['ls'], lw=s['lw'], alpha=alpha,
        marker=s['marker'], ms=6.0 if 'esm_msr' in s['key'] else 5.2,
        mfc='white' if s['hollow'] else s['color'],
        mec=s['color'], mew=1.5 if s['hollow'] else 0.7,
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


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2))
    ax_vram, ax_1m, ax_2m = axes

    max_L = max(float(df['seq_len'].max()), 4096.0)
    xt = [v for v in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] if v <= max_L * 1.15]

    # ---- Panel A: Minimum VRAM (B=1) vs Protein Length L ----
    for s in SPECS:
        sub = df[df['model'] == s['key']].sort_values('seq_len')
        if sub.empty:
            continue
        ok = sub[sub['status'] == 'ok']
        oom = sub[sub['status'] == 'oom']
        z = 6 if 'esm_msr' in s['key'] else 4
        if len(ok):
            _plot_curve(ax_vram, ok['seq_len'], ok['vram_min_alloc_gb'], s, zorder=z)
        if len(oom):
            ox = float(oom['seq_len'].iloc[0])
            ax_vram.scatter([ox], [19.7], marker='x', s=65, color=s['color'], linewidths=2.0, zorder=8)
            if len(ok):
                ax_vram.plot([float(ok['seq_len'].iloc[-1]), ox], [float(ok['vram_min_alloc_gb'].iloc[-1]), 19.7],
                             color=s['color'], ls=':', lw=1.0, alpha=0.65, zorder=z - 1)
    ax_vram.axhline(20.0, color=style.RING, ls='--', lw=1.2, zorder=7)
    ax_vram.text(34, 20.35, '20 GB RTX A4500 limit (× = B=1 OOM)', fontsize=7.6, color=style.RING, fontweight='bold')
    ax_vram.axvspan(2048, max_L * 1.25, color='#f2f2f2', zorder=0)
    ax_vram.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    ax_vram.text(2048 * 1.08, 1.0, 'tiled\nextensions', fontsize=7.0, color='#777777', va='bottom')
    ax_vram.set_xscale('log')
    ax_vram.set_ylim(0, 22.0)
    ax_vram.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_vram.set_ylabel('Minimum peak allocated VRAM at $B=1$ (GB)')
    ax_vram.set_title('A. Minimum VRAM requirement ($B=1$) vs $L$', loc='left')
    ax_vram.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # ---- Panel B: Full Single-Mutant Screen (19L variants) Cost vs L ----
    for s in SPECS:
        sub = df[(df['model'] == s['key']) & (df['status'] == 'ok')].sort_values('seq_len')
        if sub.empty or 'single_screen_s' not in sub.columns:
            continue
        ok = sub[sub['single_screen_s'].notna() & (sub['single_screen_s'] > 0)]
        if ok.empty:
            continue
        z = 6 if 'esm_msr' in s['key'] else 4
        _plot_curve(ax_1m, ok['seq_len'], ok['single_screen_s'], s, zorder=z)
    ax_1m.axvspan(2048, max_L * 1.25, color='#f2f2f2', zorder=0)
    ax_1m.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    for yref, txt in [(0.1, '0.1 s'), (1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h'), (86400.0, '1 d')]:
        ax_1m.axhline(yref, color='#888888', ls=':', lw=0.75, zorder=1)
        ax_1m.text(max_L * 1.15, yref * 1.12, txt, fontsize=7.2, color='#666666', ha='right')
    ax_1m.set_xscale('log')
    ax_1m.set_yscale('log')
    ax_1m.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_1m.set_ylabel('Full single-mutant screen (19$L$ variants) time (s)')
    ax_1m.set_title('B. Full single-mutant screening cost (19$L$)', loc='left')
    ax_1m.grid(True, color=style.GRID, lw=0.6, zorder=0)

    # ---- Panel C: Full Double-Mutant Screen (361 * L(L-1)/2 variants) Cost vs L ----
    CAP_24H = 86400.0
    for s in SPECS:
        sub = df[(df['model'] == s['key']) & (df['status'] == 'ok')].sort_values('seq_len')
        if sub.empty or 'double_screen_s' not in sub.columns:
            continue
        ok = sub[sub['double_screen_s'].notna() & (sub['double_screen_s'] > 0)]
        if ok.empty:
            continue
        z = 6 if 'esm_msr' in s['key'] else 4
        # Split at 24 h practicality cap: plot all points up through the first bin reaching >= 24 h
        above = ok[ok['double_screen_s'] >= CAP_24H]
        if len(above):
            cutoff_L = float(above['seq_len'].iloc[0])
            ok_plot = ok[ok['seq_len'] <= cutoff_L]
            _plot_curve(ax_2m, ok_plot['seq_len'], ok_plot['double_screen_s'], s, zorder=z)
            # Mark the cap-crossing point with an upward triangle / cap indicator
            ax_2m.scatter([cutoff_L], [float(above['double_screen_s'].iloc[0])],
                          marker='^', s=60, facecolors='none', edgecolors=style.RING, linewidths=1.5, zorder=z + 2)
        else:
            _plot_curve(ax_2m, ok['seq_len'], ok['double_screen_s'], s, zorder=z)
    # Reference dashed line for additive-only ESM-MSR WT-view (1 WT pass; no epistasis)
    wt_ok = df[(df['model'] == 'esm_msr_wt') & (df['status'] == 'ok')].sort_values('seq_len')
    if len(wt_ok):
        ax_2m.plot(
            wt_ok['seq_len'], wt_ok['single_screen_s'],
            color=BY_NAME['ESM-MSR'].color, ls=':', lw=1.6, alpha=0.65,
            label='ESM-MSR WT-view (additive only, no epistasis)', zorder=3,
        )
    ax_2m.axvspan(2048, max_L * 1.25, color='#f2f2f2', zorder=0)
    ax_2m.axvline(2048, color='#aaaaaa', ls=':', lw=0.9, zorder=1)
    # Prominent 24-hour practicality cap band
    ax_2m.axhline(CAP_24H, color=style.RING, ls='--', lw=1.3, zorder=7)
    ax_2m.axhspan(CAP_24H, 5e6, color=style.RING, alpha=0.07, zorder=0)
    ax_2m.text(28, CAP_24H * 1.35, '24 h cap (larger proteins impractical)', fontsize=7.6,
               color=style.RING, fontweight='bold', va='bottom', ha='left')
    for yref, txt in [(1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h'), (86400.0, '24 h'), (86400.0 * 30, '1 mo')]:
        if yref != CAP_24H:
            ax_2m.axhline(yref, color='#888888', ls=':', lw=0.75, zorder=1)
        ax_2m.text(max_L * 1.15, yref * 1.18, txt, fontsize=7.2, color=style.RING if yref == CAP_24H else '#666666', ha='right')
    ax_2m.set_xscale('log')
    ax_2m.set_yscale('log')
    ax_2m.set_ylim(2e-2, 5e6)
    ax_2m.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_2m.set_ylabel('Full double-mutant screen ($361\\binom{L}{2}$ variants) time (s)')
    ax_2m.set_title('C. Full double-mutant screening cost (24 h cap)', loc='left')
    ax_2m.grid(True, color=style.GRID, lw=0.6, zorder=0)

    for ax in axes:
        ax.set_xlim(24, max_L * 1.30)
        ax.set_xticks(xt)
        ax.set_xticklabels([f'{v // 1024}k' if v >= 8192 else str(v) for v in xt], fontsize=8.0, rotation=30, ha='right')
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Unified legend cleanly placed above the 3 panels (below title)
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=8.2, frameon=False)

    stem = out_dir / '03_size_scaling_vram_and_screening_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.80,
        title='Controlled Protein-Length Scaling of Minimum VRAM and Full Mutational Screening Cost (1× RTX A4500 20 GB)',
        note=(
            'Measured on 1× NVIDIA RTX A4500 (20 GB, fixed slot). One protein per exact power-of-two size bin (L = 32, 64, 128, ..., 65,536): '
            'real ProteinGym AF2 structures & MSAs up to L = 2,048 (truncated where needed to exact powers of two), and tiled extensions beyond '
            'L = 2,048 until each model hits OOM at B = 1. Panels B–C use each model’s optimal fitting batch size B*(L) (ThermoMPNN / ThermoMPNN-D '
            'use upstream v2_ssm.py cached-backbone SSM paths; ProteinMPNN uses score_unmasked_fast deduplication: L rows for singles, L + 19L(L−1) for doubles).'
        ),
    )
    return stem


def write_summary_md(df: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        '# Controlled Size-Scaling Benchmark Summary (RTX A4500 20 GB)',
        '',
        '| Model | Max L ok | First OOM L | Weights GB | VRAM @ L=128 | VRAM @ L=512 | VRAM @ L=1024 | 1-mut @ L=256 | 1-mut @ L=1024 | 2-mut @ L=256 | 2-mut @ L=1024 |',
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
    ap.add_argument('--out_dir', default=str(INFERENCE_RESOURCES_DIR))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_all(in_dir)
    if df.empty:
        raise SystemExit(f'No results found under {in_dir}')
    df.to_csv(out_dir / 'size_scaling_a4500_all.csv', index=False)
    write_summary_md(df, out_dir)
    stem = plot_scaling(df, out_dir)
    print('Wrote:', stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
