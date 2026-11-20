"""Publication figures for ``studies/sigma_mechanisms/`` on ``FINAL_p0`` (173 held-out ProteinGym assays)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from msr_analysis import models as M
from msr_analysis import proteingym as PG
from msr_analysis import style
from .metrics import AAS, AA_CLASS


def draw_sigma_mechanisms_composite(
    out_dir: Path,
    site_df: pd.DataFrame,
    aa_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    n_heldout: int,
    B: int,
) -> None:
    """Render the 4-panel Mechanistic ``σ`` Trade-Off Composite (`sigma_mechanisms_composite.{png,pdf}`)."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.4))
    axa, axb = axes[0, 0], axes[0, 1]
    axc, axd = axes[1, 0], axes[1, 1]
    cats = list(PG.PG_CATEGORIES)
    x = np.arange(len(cats))
    axa.axhline(0, color='#666666', lw=0.9, ls='--', zorder=1)
    axa.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    for i in range(len(cats)):
        if i % 2 == 0:
            axa.axvspan(i - 0.5, i + 0.5, color=style.BAND, lw=0, zorder=0)

    bar_specs = [
        ('between', 0.5, -0.27, 0.17, '#7fb3d5', 'Between-site (σ = 0.5)'),
        ('between', 1.0, -0.09, 0.17, '#1f5f8b', 'Between-site (σ = 1.0)'),
        ('within', 0.5, +0.09, 0.17, '#f5b041', 'Within-site (σ = 0.5)'),
        ('within', 1.0, +0.27, 0.17, '#d35400', 'Within-site (σ = 1.0)'),
    ]
    for comp, sg, off, bw, col_hex, lbl in bar_specs:
        sub = site_df[(site_df['component'] == comp) & (site_df['sigma'] == sg)].set_index('category').loc[cats]
        axa.bar(x + off, sub['delta'], bw, color=col_hex, edgecolor='black', lw=0.6, label=lbl, zorder=3)
        axa.errorbar(
            x + off, sub['delta'],
            yerr=[sub['delta'] - sub['lo'], sub['hi'] - sub['delta']],
            fmt='none', ecolor='#222222', elinewidth=1.0, capsize=2.0, zorder=4,
        )
    n_by_cat = site_df[(site_df['component'] == 'between') & (site_df['sigma'] == 1.0)].set_index('category')['n_assays']
    axa.set_xticks(x)
    axa.set_xticklabels([f'{PG.PG_CATEGORY_SHORT[c]}\n({int(n_by_cat[c])})' for c in cats], fontsize=8.5)
    axa.set_xlim(-0.5, len(cats) - 0.5)
    axa.set_ylabel('Δ Spearman ρ vs σ = 0')
    axa.set_title('a   Site sensitivity vs within-site substitution ranking', loc='left', fontsize=10.0, fontweight='bold')
    axa.legend(ncol=2, fontsize=7.8, loc='upper right', frameon=True, facecolor='white', edgecolor='#dddddd')

    # ---- Panel b: 20-AA predicted percentile rank shift vs empirical assay tolerance ------------
    cls_color = {
        'hydrophobic': '#1f5f8b',
        'polar': '#1baf7a',
        'charged': '#d35400',
        'Pro': '#6c3483',
        'Gly': '#7f8c8d',
    }
    cls_label = {
        'hydrophobic': 'Hydrophobic (A,V,I,L,M,F,W,Y,C)',
        'polar': 'Polar (S,T,N,Q,H)',
        'charged': 'Charged (D,E,K,R)',
        'Pro': 'Proline (P)',
        'Gly': 'Glycine (G)',
    }
    axb.axhline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    axb.axvline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    axb.grid(axis='both', color=style.GRID, lw=0.6, zorder=0)
    for cls_key in ('hydrophobic', 'polar', 'charged', 'Pro', 'Gly'):
        sub = aa_df[aa_df['aa_class'] == cls_key]
        # Connect functional tolerance (hollow) -> stability/expression tolerance (filled) horizontally at model shift y
        for r in sub.itertuples():
            axb.plot(
                [r.meas_tol_functional * 100, r.meas_tol_stability_expr * 100],
                [r.shift_10_all * 100, r.shift_10_all * 100],
                color=cls_color[cls_key], alpha=0.35, lw=1.2, zorder=2,
            )
            axb.plot(r.meas_tol_functional * 100, r.shift_10_all * 100, 'o', ms=5.5, mfc='white', mec=cls_color[cls_key], mew=1.3, zorder=3)
            axb.plot(r.meas_tol_stability_expr * 100, r.shift_10_all * 100, 'o', ms=6.2, color=cls_color[cls_key], zorder=4)
            x_txt = max(r.meas_tol_functional, r.meas_tol_stability_expr) * 100 + 0.55
            axb.text(x_txt, r.shift_10_all * 100, r.aa, fontsize=8.0, fontweight='bold', color=cls_color[cls_key], va='center', zorder=5)

    handles_b = [Line2D([0], [0], marker='o', color='none', mfc=cls_color[k], mec=cls_color[k], ms=6, label=cls_label[k]) for k in cls_label]
    handles_b += [
        Line2D([0], [0], marker='o', color='none', mfc='#444444', mec='#444444', ms=6, label='Filled: Stability/Expression data'),
        Line2D([0], [0], marker='o', color='none', mfc='white', mec='#444444', mew=1.3, ms=5.5, label='Hollow: Activity/Binding/Fitness data'),
    ]
    axb.legend(handles=handles_b, fontsize=7.3, loc='upper left', frameon=True, facecolor='white', edgecolor='#dddddd')
    axb.set_xlabel('Measured tolerance: mean fitness percentile − 50% (pct pts)')
    axb.set_ylabel('Model rank shift (σ = 1 − σ = 0, pct pts)')
    axb.set_title('b   20-amino-acid rank shift vs empirical assay preference', loc='left', fontsize=10.0, fontweight='bold')

    # ---- Panel c: Top-10% forward selection vs Bottom-10% damage filtering decoupling -----------
    axc.axhline(0, color='#666666', lw=0.9, ls='--', zorder=1)
    axc.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    for i in range(len(cats)):
        if i % 2 == 0:
            axc.axvspan(i - 0.5, i + 0.5, color=style.BAND, lw=0, zorder=0)

    tail_specs = [
        ('bot10_filtering_gain', 'bot10_lo', 'bot10_hi', 0.5, -0.27, 0.17, '#76d7c4', 'Bottom-10% damage filtering (σ = 0.5)'),
        ('bot10_filtering_gain', 'bot10_lo', 'bot10_hi', 1.0, -0.09, 0.17, '#117a65', 'Bottom-10% damage filtering (σ = 1.0)'),
        ('top10_selection_gain', 'top10_lo', 'top10_hi', 0.5, +0.09, 0.17, '#af7ac5', 'Top-10% forward selection (σ = 0.5)'),
        ('top10_selection_gain', 'top10_lo', 'top10_hi', 1.0, +0.27, 0.17, '#6c3483', 'Top-10% forward selection (σ = 1.0)'),
    ]
    for col_v, col_lo, col_hi, sg, off, bw, col_hex, lbl in tail_specs:
        sub = tail_df[tail_df['sigma'] == sg].set_index('category').loc[cats]
        axc.bar(x + off, sub[col_v] * 100, bw, color=col_hex, edgecolor='black', lw=0.6, label=lbl, zorder=3)
        axc.errorbar(
            x + off, sub[col_v] * 100,
            yerr=[(sub[col_v] - sub[col_lo]) * 100, (sub[col_hi] - sub[col_v]) * 100],
            fmt='none', ecolor='#222222', elinewidth=1.0, capsize=2.0, zorder=4,
        )
    n_t_cat = tail_df[tail_df['sigma'] == 1.0].set_index('category')['n_assays']
    axc.set_xticks(x)
    axc.set_xticklabels([f'{PG.PG_CATEGORY_SHORT[c]}\n({int(n_t_cat[c])})' for c in cats], fontsize=8.5)
    axc.set_xlim(-0.5, len(cats) - 0.5)
    axc.set_ylabel('True fitness percentile gain vs σ = 0 (pct pts, + = better)')
    axc.set_title('c   Damage-filtering vs top-10% forward selection decoupling', loc='left', fontsize=10.0, fontweight='bold')
    axc.legend(ncol=2, fontsize=7.5, loc='upper right', frameon=True, facecolor='white', edgecolor='#dddddd')

    # ---- Panel d: Multi-property Pareto trajectories across σ ∈ {0, 0.5, 1} ---------------------
    axd.axhline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    axd.axvline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    axd.grid(axis='both', color=style.GRID, lw=0.6, zorder=0)

    for r in pareto_df.itertuples():
        dx = np.array([0.0, r.abund_05 - r.abund_0, r.abund_1 - r.abund_0])
        dy = np.array([0.0, r.func_05 - r.func_0, r.func_1 - r.func_0])
        axd.plot(dx, dy, '-', color='#85929e', alpha=0.48, lw=1.1, zorder=2)
        axd.plot(dx[1], dy[1], 'o', ms=4.5, color='#2a78d6', alpha=0.75, zorder=3)
        axd.plot(dx[2], dy[2], 's', ms=4.5, color='#eb6834', alpha=0.75, zorder=3)
        if abs(dy[2]) > 0.035 or dx[2] > 0.14:
            axd.annotate(r.short, (dx[2], dy[2]), xytext=(4, 2), textcoords='offset points', fontsize=7.3, color='#333333')

    # Mean Pareto trajectory across all held-out multi-property pairs
    mx = np.array([0.0, float((pareto_df['abund_05'] - pareto_df['abund_0']).mean()), float((pareto_df['abund_1'] - pareto_df['abund_0']).mean())])
    my = np.array([0.0, float((pareto_df['func_05'] - pareto_df['func_0']).mean()), float((pareto_df['func_1'] - pareto_df['func_0']).mean())])
    axd.plot(mx, my, '-', color='#000000', lw=2.6, zorder=5)
    axd.plot(0.0, 0.0, '^', ms=7.5, color='#7f8c8d', mec='black', zorder=6, label='σ = 0 (ESM3-small-open)')
    axd.plot(mx[1], my[1], 'D', ms=7.8, color='#2a78d6', mec='black', zorder=6, label=f'σ = 0.5 (mean: Δabund {mx[1]:+.3f}, Δfunc {my[1]:+.3f})')
    axd.plot(mx[2], my[2], 'D', ms=7.8, color='#000000', mec='black', zorder=6, label=f'σ = 1.0 (mean: Δabund {mx[2]:+.3f}, Δfunc {my[2]:+.3f})')
    share_05 = (mx[1] / mx[2] * 100) if mx[2] > 0 else 0.0
    axd.text(
        0.03, 0.96,
        f'n = {len(pareto_df)} held-out protein pairs\nσ = 0.5 retains {share_05:.0f}% of abundance Δρ\nwith Δfunc = {my[1]:+.3f} (vs {my[2]:+.3f} at σ = 1)',
        transform=axd.transAxes, va='top', ha='left', fontsize=8.0,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.92), zorder=7,
    )
    axd.set_xlabel('Δρ on paired abundance / expression assay (vs σ = 0)')
    axd.set_ylabel('Δρ on paired activity / binding assay (vs σ = 0)')
    axd.set_title('d   Stability–function Pareto trajectories across σ ∈ {0, 0.5, 1}', loc='left', fontsize=10.0, fontweight='bold')
    axd.legend(loc='lower left', fontsize=7.6, frameon=True, facecolor='white', edgecolor='#dddddd')

    style.finish_figure(
        fig,
        out_dir / 'sigma_mechanisms_composite',
        title=f'Mechanisms of thermodynamic LoRA scaling σ on ProteinGym ({n_heldout} held-out assays; FINAL_p0 seeds 1–3; B = {B:,})',
        note=(
            f'All panels evaluate FINAL_p0 (seeds 1–3) vs σ = 0 on the canonical {n_heldout} held-out ProteinGym assays (joint [assay × triplicate-seed] 95% CIs, B = {B:,}). '
            'Stability fine-tuning boosts between-site sensitivity ranking (panel a) and bottom-10% damage filtering (panel c) across all functional categories, '
            f'while promoting hydrophobic core substitutions (panel b); σ = 0.5 retains {share_05:.0f}% of abundance gain with higher functional Δρ ({my[1]:+.3f} vs {my[2]:+.3f} at σ = 1; panel d).'
        ),
        note_fontsize=7.5,
    )


def draw_aa_category_heatmaps(out_dir: Path, aa_df: pd.DataFrame, site_df: pd.DataFrame) -> None:
    """Render companion 5-category × 20-AA predicted shift vs empirical tolerance heatmaps (`aa_category_heatmaps.{png,pdf}`)."""
    cats = list(PG.PG_CATEGORIES)
    n_by_cat = site_df[(site_df['component'] == 'between') & (site_df['sigma'] == 1.0)].set_index('category')['n_assays']
    H_shift = np.array([[float(aa_df.loc[aa_df['aa'] == a, f'shift_10_{c}'].iloc[0]) for a in AAS] for c in cats])
    H_meas = np.array([[float(aa_df.loc[aa_df['aa'] == a, f'meas_tol_{c}'].iloc[0]) for a in AAS] for c in cats])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.2), sharey=True)
    for ax, H, ttl, vm in [
        (ax1, H_shift, 'a   Model rank shift: pct(σ = 1) − pct(σ = 0) (pct pts)', 0.16),
        (ax2, H_meas, 'b   Measured assay tolerance: mean fitness pct − 50% (pct pts)', 0.13),
    ]:
        ax.imshow(H, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='auto')
        ax.set_xticks(range(len(AAS)))
        ax.set_xticklabels(AAS, fontsize=8.5, fontweight='semibold')
        ax.set_xlabel('Substituted-to amino acid')
        ax.set_title(ttl, loc='left', fontsize=9.8, fontweight='bold')
        for i in range(len(cats)):
            for j in range(len(AAS)):
                v_pct = int(round(H[i, j] * 100))
                txt_v = '0' if v_pct == 0 else f'{v_pct:+d}'
                ax.text(
                    j, i, txt_v, ha='center', va='center', fontsize=6.8,
                    color='white' if abs(H[i, j]) > 0.62 * vm else '#111111',
                )
    ax1.set_yticks(range(len(cats)))
    ax1.set_yticklabels([f'{PG.PG_CATEGORY_SHORT[c]} ({int(n_by_cat[c])})' for c in cats], fontsize=8.8)

    style.finish_figure(
        fig,
        out_dir / 'aa_category_heatmaps',
        title='Amino-acid rank shifts induced by stability LoRA (σ = 1 vs 0) vs empirical category tolerances (FINAL_p0 seeds 1–3)',
        note='Single-mutant rows of held-out ProteinGym assays with ≥ 100 singles. Values shown in percentile points (× 100).',
        top=0.88,
    )
