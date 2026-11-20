"""Stage 10 — Split construction, homology holdout filters, reviewer reference tables, and mega_splits provenance.

Consolidates:
  - ``10_leakage_and_reference.py`` (reference items): SPURS public vs retrained inflation, published-paper hyperopt
    reference (table 1.5), and third-party ``nobb`` baseline ablation reference (table 2.5)
    -> ``00_for_reviewers/reference/{published_reference.*, spurs_published_vs_retrained.*}`` & ``00_for_reviewers/baseline_nobb_ablation.*``
  - ``20_split_best_partner.py``: best cross-set sequence & structural partner scatter for ``capped`` and ``hyperopt`` splits
    -> ``00_supplementary_figs/split_construction/split_best_partner_capped.*`` & ``00_for_reviewers/split_best_partner_hyperopt.*``
  - ``20b_split_overlap_heatmaps.py``: lower-triangular max identity & max TM-score heatmaps across all 16 benchmark sets
  - ``21b_holdout_filter_figure.py``: 4-criterion homology filter breakdown across ProteinGym, Domainome, MGnify, Rocklin
  - ``tsuboyama/mega_splits/02_split_provenance.py``: ``mega_splits`` provenance & homology audit against ``splits_oct06_capped``
    -> ``01_megascale/mega_splits/provenance/*`` & ``00_supplementary_figs/split_construction/mega_clean_vs_homologous_test.*``
"""
from __future__ import annotations

import hashlib
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from msr_analysis import style
from msr_analysis.common import load_rho_df, primary_ci
from msr_analysis.data import Frame, load_frame
from msr_analysis.heldout import flags
from msr_analysis.paths import (
    DATA_DIR,
    REFERENCE,
    SALVAGED_HOMOLOGY_CSV,
    TASKS,
    atomic_to_csv,
    results_dir,
)
from msr_analysis.registry import EXT12, LABEL, MODELS
from msr_analysis.stats import boot_domain_seeds, boot_pvalue, make_domain_seed_draws
from msr_analysis.tables import numeric, num, read_table
from . import base_parser, resolve_b

STAGE = 'splits_and_leakage'
DESCRIPTION = 'Split construction, 4-part homology holdout filter, reviewer reference comparisons, and mega_splits clean-vs-homologous audit.'

INTERNAL = ('train', 'val', 'test')
FUNCTIONAL = {'GRB2', 'DLG4', 'EstA', 'Myo', 'GB1'}
E_CUT, ID_CUT, TM_CUT = 1e-3, 0.40, 0.70
NO_HIT = -1.0


def resolve_homology_pairs_csv() -> Path | None:
    if SALVAGED_HOMOLOGY_CSV.exists():
        return SALVAGED_HOMOLOGY_CSV
    alt = SALVAGED_HOMOLOGY_CSV.parent.parent / 'homology' / 'homology_pairs.csv'
    if alt.exists():
        return alt
    return None


def _strip_pdb(x: str) -> str:
    s = str(x)
    return s[:-4] if s.endswith('.pdb') else s


def _panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04, ha: str = 'left') -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha=ha)


def membership_from_split(split_path: Path, pair_ids: set[str]) -> dict[str, str]:
    with open(split_path, 'rb') as fh:
        splits = pickle.load(fh)
    mem: dict[str, str] = {}
    for s in INTERNAL:
        for c in splits.get(s, []):
            mem[_strip_pdb(c)] = s
    for pid in pair_ids:
        if pid.startswith('test_'):
            mem[pid] = 'external'
        elif pid in FUNCTIONAL:
            mem[pid] = 'functional'
    return mem


# ----------------------------------------------------------------------------------------------
# Part 1: Reviewer reference comparisons (00_for_reviewers/reference & 00_for_reviewers; zero MODELS mutation!)
# ----------------------------------------------------------------------------------------------
def run_reviewer_reference(B: int) -> None:
    out = results_dir('reviewer_reference')
    rev_out = results_dir('for_reviewers')

    # 1. SPURS public vs retrained -> 00_for_reviewers/reference/spurs_published_vs_retrained.{csv,pdf,png}
    ref_sp = pd.read_csv(REFERENCE / 'spurs_public_checkpoint.csv')
    pub_sp = ref_sp[ref_sp.model == 'SPURS (public checkpoint)'].set_index('dataset').rho
    rho_df = load_rho_df()
    P_ci = primary_ci(rho_df)
    ds_list = EXT12 + [
        'tsuboyama_capped_val', 'tsuboyama_capped_test',
        'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
        'MYO_HUMAN_Kung_2025_display', 'domainome',
    ]
    ren = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}
    fig, ax = plt.subplots(figsize=(12, 3.6))
    sp_rows = []
    for i, d_name in enumerate(ds_list):
        v = P_ci.get((d_name, 'spurs'))
        p = pub_sp.get(ren.get(d_name, d_name), np.nan)
        if v is None or not np.isfinite(p):
            continue
        ax.plot([i, i], [v[0], p], color='#bbbbbb', lw=2, zorder=1)
        ax.errorbar([i], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt='s', color='#d35400', capsize=2, zorder=3, ms=6)
        ax.plot([i], [p], marker='s', mfc='white', mec='#d35400', ms=7, ls='none', zorder=4)
        sp_rows.append(dict(dataset=d_name, retrained=v[0], published=p, inflation=p - v[0]))
    ax.plot([], [], 's', color='#d35400', label='retrained on Homology-Restricted MegaScale (3 seeds)')
    ax.plot([], [], 's', mfc='white', mec='#d35400', label='public checkpoint (trained on 24 of the 33 test libraries)')
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels([LABEL[d_name] for d_name in ds_list], rotation=35, ha='right')
    ax.set_ylabel('Spearman ρ')
    ax.set_ylim(0.20, 0.92)
    ax.legend(fontsize=8, loc='lower left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.92)
    ax.set_title('SPURS: what training overlap adds (public checkpoint against retrained SPURS)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, out / 'spurs_published_vs_retrained')
    atomic_to_csv(pd.DataFrame(sp_rows), out / 'spurs_published_vs_retrained.csv', index=False, float_format='%.4f')

    # 2. Published-paper reference (table 1.5.1) -> 00_for_reviewers/reference/published_reference.{pdf,png}
    d_pr = read_table('1.5.1')
    ref_models = list(d_pr.columns[1:])
    groups = [r for r in d_pr.iloc[:, 0]]
    table_pr = {}
    for _, r in d_pr.iterrows():
        for m in ref_models:
            v, _s = num(r[m])
            if np.isfinite(v):
                table_pr[(r.iloc[0], m)] = v
    pal = ['#000000', '#34495e', '#1abc9c', '#999999', '#f1948a', '#e74c3c', '#922b21', '#2ecc71', '#abebc6', '#3498db', '#1f5f8b']
    mk = ['D', 'd', '+', '^', 'o', 'o', 'o', 'h', 'h', 'p', 'P']
    fig, ax = plt.subplots(figsize=(14, 4.6))
    style.band_axis(ax, len(groups))
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    offs_pr = np.linspace(-0.36, 0.36, len(ref_models))
    leg_hs = []
    for j, (m, c, k) in enumerate(zip(ref_models, pal, mk)):
        hl = (m == 'ESM-MSR (published)')
        hollow_m = (m == 'ME additive (hyperopt)')
        sz = 6.0 * (1.35 if hl else 1.0)
        xs_m = [i + offs_pr[j] for i, g in enumerate(groups) if np.isfinite(table_pr.get((g, m), np.nan))]
        ys_m = [table_pr[(g, m)] for g in groups if np.isfinite(table_pr.get((g, m), np.nan))]
        kw_m = dict(
            marker=k, ms=sz * (style.HOLLOW_SIZE if hollow_m else 1.0), color=c, mec=c,
            mfc='none' if hollow_m else c, mew=style.HOLLOW_MEW if hollow_m else 1.2, ls='none',
        )
        if xs_m:
            ax.plot(xs_m, ys_m, zorder=5 if hl else 3, **kw_m)
        kw_leg = dict(kw_m, ms=7.0 * (style.HOLLOW_SIZE if hollow_m else 1.0))
        leg_hs.append(plt.Line2D([0], [0], label=m, **kw_leg))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace(' (per-domain avg.)', '') for g in groups], rotation=30, ha='right')
    ax.set_ylabel('Spearman ρ (published-paper evaluation)')
    ax.set_ylim(0.0, 1.03)
    ax.legend(handles=leg_hs, loc='upper right', ncol=4, fontsize=7.3, frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.94)
    ax.set_title('Table 1.5: published-paper evaluation (hyperopt split, earlier models; compare within this table only)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, out / 'published_reference')

    # 3. Third-party mutant-background ablation reference (table 2.5.1) -> 00_for_reviewers/baseline_nobb_ablation.{pdf,png}
    d_nb = read_table('2.5.1')
    V_nb, _ = numeric(d_nb)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x_nb = np.arange(V_nb.shape[1])
    w_nb = 0.8 / max(len(V_nb), 1)
    nb_cols = {
        'MutateEverything stage 2': MODELS['me2']['color'],
        'MutateEverything stage 1': MODELS['me1']['color'],
        'ThermoMPNN': MODELS['tmpnn']['color'],
    }
    for i, (name, row) in enumerate(V_nb.iterrows()):
        ax.bar(x_nb + (i - 1) * w_nb, row.values, w_nb, color=nb_cols.get(name, '#999999'), edgecolor='black', lw=0.6, label=name)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x_nb)
    ax.set_xticklabels(V_nb.columns, rotation=20, ha='right')
    ax.set_ylabel('Δ (without − with\nmutant-background libraries)', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title('Table 2.5: third-party baseline mutant-background ablation (document reference)', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, rev_out / 'baseline_nobb_ablation')


# ----------------------------------------------------------------------------------------------
# Part 2: 20_split_best_partner & 20b_split_overlap_heatmaps & 21b_holdout_filter_figure
# ----------------------------------------------------------------------------------------------
def _groups_of_split(pairs: pd.DataFrame, mem: dict[str, str]) -> dict[str, pd.DataFrame]:
    p = pairs.copy()
    p['set_a'], p['set_b'] = p['a'].map(mem), p['b'].map(mem)
    p = p.dropna(subset=['set_a', 'set_b'])
    both = pd.concat([
        p.rename(columns={'a': 'lib', 'set_a': 'lib_set', 'b': 'partner', 'set_b': 'ref_set'}),
        p.rename(columns={'b': 'lib', 'set_b': 'lib_set', 'a': 'partner', 'set_a': 'ref_set'}),
    ])
    both = both[both['lib_set'].isin(INTERNAL) & (both['lib_set'] != both['ref_set'])]
    out = {}
    for label, ls, rs in (
        ('val → train', ['val'], ['train']),
        ('test → train', ['test'], ['train']),
        ('train / val → external', ['train', 'val'], ['external', 'functional']),
    ):
        g = both[both['lib_set'].isin(ls) & both['ref_set'].isin(rs)]
        agg = g.groupby('lib').agg(
            seq_evalue=('seq_evalue', 'min'),
            identity=('identity', 'max'),
            struct_evalue=('struct_evalue', 'min'),
            tm=('struct_tm_max', 'max'),
        )
        libs = sorted(k for k, v in mem.items() if v in ls and not k.startswith('test_'))
        out[label] = agg.reindex(libs)
    return out


def _draw_best_partner(groups: dict[str, pd.DataFrame]) -> plt.Figure:
    colors = {'val → train': '#2a78d6', 'test → train': '#eb6834', 'train / val → external': '#1baf7a'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for label, g in groups.items():
        c = colors[label]
        ys = (-np.log10(g['seq_evalue'].clip(lower=1e-12))).fillna(NO_HIT)
        a.scatter(g['identity'], ys, s=22, c=c, alpha=0.75, edgecolors='white', linewidths=0.6, label=f'{label} (n = {len(g)})')
        h = g.dropna(subset=['struct_evalue'])
        b.scatter(h['tm'], -np.log10(h['struct_evalue'].clip(lower=1e-12)), s=22, c=c, alpha=0.75, edgecolors='white', linewidths=0.6)
    for ax in (a, b):
        ax.axhline(-np.log10(E_CUT), color='#52514e', lw=1, ls='--')
        ax.grid(alpha=0.2)
        ax.axhline(NO_HIT, color='#bbbbbb', lw=0.6, zorder=0)
    a.axvline(ID_CUT, color='#52514e', lw=1, ls='--')
    b.axvline(TM_CUT, color='#52514e', lw=1, ls='--')
    a.text(0.01, -np.log10(E_CUT) + 0.12, 'E-value ≤ 10⁻³', fontsize=8, color='#52514e', ha='left', transform=a.get_yaxis_transform())
    b.text(0.01, -np.log10(E_CUT) + 0.12, 'E-value ≤ 10⁻³', fontsize=8, color='#52514e', ha='left', transform=b.get_yaxis_transform())
    a.text(ID_CUT + 0.003, 0.99, 'identity > 0.40', fontsize=8, color='#52514e', rotation=90, va='top', transform=a.get_xaxis_transform())
    b.text(TM_CUT + 0.003, 0.99, 'TM-score > 0.70', fontsize=8, color='#52514e', rotation=90, va='top', transform=b.get_xaxis_transform())
    a.set_xlabel('Sequence identity of best partner (BLOSUM62 local, shorter chain)')
    b.set_xlabel('TM-score of best partner (shorter-chain normalised)')
    a.set_ylabel('−log10 E-value of best partner')
    a.text(0.0, 1.03, 'MMseqs2 E-value and identity (criteria 1 and 2)', transform=a.transAxes, fontsize=9.5, fontweight='bold', color='#444444')
    b.text(0.0, 1.03, 'Foldseek E-value and TM-score (criteria 3 and 4)', transform=b.transAxes, fontsize=9.5, fontweight='bold', color='#444444')
    _panel_label(a, 'a', x=-0.10, y=1.07)
    _panel_label(b, 'b', x=-0.04, y=1.07)
    top = int(np.ceil(max(a.get_ylim()[1], b.get_ylim()[1], 4.4) - 0.3))
    a.set_ylim(NO_HIT - 0.3, top + 0.3)
    ticks = list(range(0, top + 1))
    a.set_yticks([-1] + ticks)
    a.set_yticklabels(['no hit'] + [str(t) for t in ticks])
    a.set_xlim(a.get_xlim()[0], max(a.get_xlim()[1], ID_CUT + 0.02))
    b.set_xlim(b.get_xlim()[0], max(b.get_xlim()[1], TM_CUT + 0.02))
    a.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


def run_20_split_construction() -> None:
    out_sc = results_dir('split_construction')
    out_rev = results_dir('for_reviewers')

    # 20b: Set-overlap heatmaps
    set_lbl = {
        'train': 'Train', 'val': 'Validation', 'test': 'Test', 'ssym': 'Ssym', 'ptmul': 'PTMUL', 's461': 'S461',
        'k2369': 'K2369', 'q3421': 'Q3421', 's571': 'S571', 's783': 'S783', 's2648': 'S2648', 's8754': 'S8754',
        'ProteinGym': 'ProteinGym', 'Domainome': 'Domainome', 'MGnify': 'MGnify', 'Rocklin designs': 'Rocklin designs',
    }
    cuts = {'identity': 0.40, 'tm': 0.70}
    stems = {'identity': 'split_max_identity_heatmap', 'tm': 'split_max_tm_heatmap'}
    titles = {
        'identity': 'Max sequence identity (MMseqs2, normalised by the shorter sequence)',
        'tm': 'Max TM-score (Foldseek, shorter-chain normalised)',
    }
    for kind in ('identity', 'tm'):
        m = pd.read_csv(REFERENCE / f'split_set_overlap_{kind}.csv', index_col=0)
        sets = list(m.index)
        n = len(sets)
        fig, ax = plt.subplots(figsize=(10.4, 9.0))
        v = m.to_numpy(float).copy()
        v[np.triu_indices(n)] = np.nan
        v_mask = np.ma.masked_invalid(v)
        cmap = plt.get_cmap('Reds').copy()
        im = ax.imshow(v_mask, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        for i in range(n):
            for j in range(i):
                x = m.iloc[i, j]
                if np.isfinite(x):
                    ax.text(j, i, f'{x:.2f}', ha='center', va='center', fontsize=7.6, color='white' if x > 0.55 else '#222222')
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#e4e4e4', ec='none'))
                    ax.text(j, i, '–', ha='center', va='center', fontsize=7.4, color='#777777')
        for e in np.cumsum([3, 9, 4])[:-1] - 0.5:
            ax.plot([-0.5, e], [e, e], color='black', lw=1.1)
            ax.plot([e, e], [e, n - 0.5], color='black', lw=1.1)
        ax.set_xlim(-0.5, n - 1.5)
        ax.set_ylim(n - 0.5, 0.5)
        ax.set_xticks(range(n - 1))
        ax.set_yticks(range(1, n))
        ax.set_xticklabels([set_lbl[s] for s in sets[:-1]], rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
        ax.set_yticklabels([set_lbl[s] for s in sets[1:]], fontsize=9)
        for sp_v in ax.spines.values():
            sp_v.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(titles[kind], fontsize=9.5)
        cb.ax.axhline(cuts[kind], color='black', lw=1.2)
        tks = sorted(set(np.round(np.arange(0, 1.01, 0.2), 2)) | {cuts[kind]})
        cb.set_ticks(tks)
        cb.set_ticklabels([f'{t:.2f}'.rstrip('0').rstrip('.') if t != cuts[kind] else f'{t:.2f} (rule)' for t in tks])
        fig.tight_layout()
        style.save(fig, out_sc / stems[kind])

    # 21b: Holdout filter figure
    out_supp = results_dir('supplementary_figs')
    f_flags = flags()
    sets_hf = [
        ('proteingym', 'ProteinGym\nassays'),
        ('domainome', 'Domainome\ndomains'),
        ('mgnify', 'MGnify\nscaffolds'),
        ('rocklin_denovo', 'Rocklin\ndesigns'),
    ]
    crit = [
        ('f_seq_e', 'MMseqs2 E-value ≤ 1e-3', '#2a78d6'),
        ('f_identity', 'Identity > 0.40', '#7fb3e8'),
        ('f_struct_e', 'Foldseek E-value ≤ 1e-3', '#d35400'),
        ('f_tm', 'Foldseek TM-score > 0.70', '#f0a36b'),
    ]
    hf_rows = []
    for ds_n, _ in sets_hf:
        g = f_flags[f_flags.dataset == ds_n]
        n_tot = len(g)
        r = dict(dataset=ds_n, n=n_tot, flagged=int(g.homologous.sum()), kept=n_tot - int(g.homologous.sum()))
        for c, _, _ in crit:
            r[c] = int(g[c].sum())
            r[c + '_only'] = int((g[c] & (g[[k for k, _, _ in crit]].sum(axis=1) == 1)).sum())
        hf_rows.append(r)
    T_hf = pd.DataFrame(hf_rows)
    atomic_to_csv(T_hf, out_supp / 'holdout_filter_counts.csv', index=False)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    x_hf = np.arange(len(sets_hf))
    w_hf = 0.15
    cols_hf = [('homologous', 'Removed (any criterion)', '#333333')] + crit
    for j, (c, lab, col) in enumerate(cols_hf):
        v = [100 * (T_hf.loc[i, 'flagged'] if c == 'homologous' else T_hf.loc[i, c]) / T_hf.loc[i, 'n'] for i in range(len(sets_hf))]
        ax_a.bar(x_hf + (j - 2) * w_hf, v, w_hf, color=col, edgecolor='black', lw=0.6, label=lab, zorder=3)
        for xi, vi in zip(x_hf + (j - 2) * w_hf, v):
            ax_a.text(xi, vi + 0.4, f'{vi:.0f}' if vi >= 1 else f'{vi:.1f}', ha='center', va='bottom', fontsize=7.6, zorder=4)
    ax_a.set_xticks(x_hf)
    ax_a.set_xticklabels([f'{lab}\n(n = {int(T_hf.loc[i, "n"]):,})' for i, (_, lab) in enumerate(sets_hf)], fontsize=9.5)
    ax_a.set_ylabel('% of the benchmark homologous to the training set')
    ax_a.grid(axis='y', color='#dddddd', lw=0.6, zorder=0)
    ax_a.set_ylim(0, 1.3 * ax_a.get_ylim()[1])
    ax_a.legend(fontsize=8.2, frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left')

    bottom = np.zeros(len(sets_hf))
    for c, lab, col in crit:
        v = np.array([T_hf.loc[i, c + '_only'] for i in range(len(sets_hf))], float)
        share = v / T_hf['flagged'].to_numpy() * 100
        ax_b.bar(x_hf, share, 0.6, bottom=bottom, color=col, edgecolor='black', lw=0.6, label=lab, zorder=3)
        for xi, bi, si in zip(x_hf, bottom, share):
            if si >= 6:
                ax_b.text(xi, bi + si / 2, f'{si:.0f}%', ha='center', va='center', fontsize=8, zorder=4)
        bottom += share
    ax_b.bar(x_hf, 100 - bottom, 0.6, bottom=bottom, color='#bbbbbb', edgecolor='black', lw=0.6, label='two or more criteria', zorder=3)
    for xi, bi, si in zip(x_hf, bottom, 100 - bottom):
        if si >= 6:
            ax_b.text(xi, bi + si / 2, f'{si:.0f}%', ha='center', va='center', fontsize=8, zorder=4)
    ax_b.set_xticks(x_hf)
    ax_b.set_xticklabels([lab for _, lab in sets_hf], fontsize=9.5)
    ax_b.set_ylabel('% of the removed proteins')
    ax_b.set_ylim(0, 100)
    ax_b.legend(fontsize=8.2, frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3)
    _panel_label(ax_a, 'a', x=-0.10, y=1.04)
    _panel_label(ax_b, 'b', x=-0.12, y=1.04)
    ax_a.set_title('Caught by each criterion', fontsize=10.5, loc='left')
    ax_b.set_title('Caught by one criterion only', fontsize=10.5, loc='left')
    fig.tight_layout()
    style.save(fig, out_supp / 'holdout_filter')

    # 20: Best cross-set partner scatter (requires homology_pairs.csv)
    pairs_csv = resolve_homology_pairs_csv()
    if pairs_csv is None:
        print('[p10] Skipping split_best_partner: homology_pairs.csv not found')
        return
    pairs = pd.read_csv(pairs_csv)
    pair_ids = set(pairs['a']) | set(pairs['b'])
    for sp_name, pkl_name, out_dir, stem in (
        ('capped', 'splits_oct06_capped.pkl', out_sc, 'split_best_partner_capped'),
        ('hyperopt', 'hyperopt_splits.pkl', out_rev, 'split_best_partner_hyperopt'),
    ):
        pkl_path = DATA_DIR / pkl_name
        if not pkl_path.exists():
            continue
        mem = membership_from_split(pkl_path, pair_ids)
        groups = _groups_of_split(pairs, mem)
        fig = _draw_best_partner(groups)
        style.save(fig, out_dir / stem)
        rows = pd.concat({k: v.reset_index() for k, v in groups.items()}, names=['comparison']).reset_index(level=0)
        atomic_to_csv(rows, out_dir / f'{stem}.csv', index=False, float_format='%.5g')


# ----------------------------------------------------------------------------------------------
# Part 3: mega_splits provenance + New High-Impact Composite Figure #4 (mega_clean_vs_homologous_test)
# ----------------------------------------------------------------------------------------------
def _domain_kind(c: str) -> str:
    if re.match(r'^[0-9][A-Za-z0-9]{3}$', c):
        return 'natural'
    if 'TrROS_Hall' in c:
        return 'de novo (trRosetta hallucination)'
    if re.match(r'^(v2_)?(HHH|HEEH|EHEE|EEHEE|EHE|HEE)', c):
        return 'de novo (Rocklin design)'
    return 'other (redesign of a natural domain)' if c.startswith('v2_') else 'other'


def run_mega_clean_vs_homologous(B: int) -> None:
    out_sc = results_dir('split_construction')
    out_prov = results_dir('mega_provenance')

    mega_pkl = DATA_DIR / 'mega_splits.pkl'
    cap_pkl = DATA_DIR / 'splits_oct06_capped.pkl'
    if not mega_pkl.exists() or not cap_pkl.exists():
        print('[p10] Skipping mega_clean_vs_homologous: split pickle missing')
        return

    with open(mega_pkl, 'rb') as fh:
        sp_mega = pickle.load(fh)
    with open(cap_pkl, 'rb') as fh:
        sp_cap = pickle.load(fh)
    parts_mega = {k: {_strip_pdb(x) for x in sp_mega[k]} for k in INTERNAL}
    parts_cap = {k: {_strip_pdb(x) for x in sp_cap[k]} for k in INTERNAL}

    # Provenance CSVs: md5 & test composition
    atomic_to_csv(
        pd.DataFrame([
            dict(file='repo/data/mega_splits.pkl', md5=hashlib.md5(mega_pkl.read_bytes()).hexdigest()),
            dict(file='repo/data/splits_oct06_capped.pkl', md5=hashlib.md5(cap_pkl.read_bytes()).hexdigest()),
        ]),
        out_prov / 'split_file_md5.csv', index=False,
    )
    comp_rows = []
    for nm, ps, ds_tag in (('mega_splits', parts_mega, 'mega'), ('capped', parts_cap, 'capped')):
        task_f = TASKS / f'tsuboyama_{ds_tag}_test.csv.gz'
        if not task_f.exists():
            continue
        t = pd.read_csv(task_f, low_memory=False)
        t['n'] = t.mut_type.astype(str).str.count(':') + 1
        k_cnt = pd.Series([_domain_kind(c) for c in t.code_wt.unique()]).value_counts()
        comp_rows.append(dict(
            split=nm, train_libs=len(ps['train']), val_libs=len(ps['val']),
            test_libs=t.code_wt.nunique(), test_backbones=t.code.nunique(),
            rows=len(t), measured=int((t.cens == 0).sum()),
            singles=int((t.n == 1).sum()), doubles=int((t.n == 2).sum()),
            **{f'test_{a}': int(b) for a, b in k_cnt.items()},
        ))
    if comp_rows:
        atomic_to_csv(pd.DataFrame(comp_rows), out_prov / 'test_composition.csv', index=False)

    pairs_csv = resolve_homology_pairs_csv()
    if pairs_csv is None:
        print('[p10] Skipping mega_clean_vs_homologous figure: homology_pairs.csv not found')
        return
    P_hom = pd.read_csv(pairs_csv)
    fr_raw = load_frame('tsuboyama_mega_test')
    fr = Frame(fr_raw.name, fr_raw.spine, dict(fr_raw.preds), dict(fr_raw.notes))
    sp = fr.spine
    test_libs = sorted(sp.loc[sp['use'], 'group'].astype(str).unique())

    # Identify best train partner in mega_splits['train'] for each of the 28 mega_splits test domains
    train_set = parts_mega['train']
    both = pd.concat([
        P_hom[P_hom['a'].isin(test_libs) & P_hom['b'].isin(train_set)].rename(columns={'a': 'lib', 'b': 'partner'}),
        P_hom[P_hom['b'].isin(test_libs) & P_hom['a'].isin(train_set)].rename(columns={'b': 'lib', 'a': 'partner'}),
    ], ignore_index=True)

    dom_meta = []
    for lib in test_libs:
        sub = both[both['lib'] == lib]
        seq_e = float(sub['seq_evalue'].min()) if len(sub) and sub['seq_evalue'].notna().any() else np.nan
        ident = float(sub['identity'].max()) if len(sub) and sub['identity'].notna().any() else 0.0
        str_e = float(sub['struct_evalue'].min()) if len(sub) and sub['struct_evalue'].notna().any() else np.nan
        tm = float(sub['struct_tm_max'].max()) if len(sub) and sub['struct_tm_max'].notna().any() else 0.0
        is_hom = bool(
            (np.isfinite(seq_e) and seq_e <= E_CUT)
            or (np.isfinite(str_e) and str_e <= E_CUT)
            or (ident > ID_CUT)
            or (tm > TM_CUT)
        )
        dom_meta.append(dict(
            library=lib, stratum='homologous' if is_hom else 'clean',
            kind=_domain_kind(lib), in_capped_train=(lib in parts_cap['train']),
            best_train_identity=ident, best_train_tm=tm,
            min_seq_evalue=seq_e, min_struct_evalue=str_e,
        ))
    meta_df = pd.DataFrame(dom_meta)
    atomic_to_csv(meta_df, out_prov / 'mega_test_domain_homology.csv', index=False, float_format='%.5g')

    clean_libs = sorted(meta_df.loc[meta_df.stratum == 'clean', 'library'])
    hom_libs = sorted(meta_df.loc[meta_df.stratum == 'homologous', 'library'])

    # Build composite keys on mega_splits test frame
    n1 = (sp['n_mut'] == 1).to_numpy()
    model_specs = [
        ('ESM-MSR', 'msrp0:wt', 'msrp0:comb', 'msrp0:epi', MODELS['msrp0:comb']['color'], False),
        ('MutateEverything', 'me1', 'me2', 'me2:epi', MODELS['me2']['color'], False),
        ('ThermoMPNN (retrain)', 'tmpnn', 'tmpnnd', 'tmpnnd:epi', MODELS['tmpnnd']['color'], False),
        ('ThermoMPNN (published)', 'tmpnnpub', 'tmpnndpub', 'tmpnndpub:epi', MODELS['tmpnnpub']['color'], True),
        ('SPURS (frozen adapter)', 'spurs', 'spursm', 'spursm:epi', MODELS['spurs']['color'], False),
        ('SPURS (trained adapter)', 'spursu', 'spursum', 'spursum:epi', MODELS['spursu']['color'], False),
        ('SPURS (published)', 'spurspub', 'spurspubm', 'spurspubm:epi', MODELS['spurspub']['color'], True),
    ]
    active_models = []
    for label, ks, kd, ke, col, hatched in model_specs:
        if ks in fr.preds and kd in fr.preds:
            ck = f'{ks}|{kd}'
            fr.preds[ck] = np.where(n1[None, :], fr.preds[ks], fr.preds[kd])
            active_models.append((label, ck, ke if ke in fr.preds else None, col, hatched))

    # Per-domain Spearman on all variants & epistasis dddG on doubles
    y_all = sp['y'].to_numpy()
    y_epi = sp['dddG_ML'].to_numpy()
    use_all = sp['use'].to_numpy() & np.isfinite(y_all)
    use_epi = sp['use'].to_numpy() & (sp['n_mut'].to_numpy() == 2) & np.isfinite(y_epi)
    grp = sp['group'].astype(str).to_numpy()

    def eval_stratum(libs_sub: list[str], key: str, mask: np.ndarray, y_vec: np.ndarray, seed_rng: int) -> tuple[float, float, float, np.ndarray, int]:
        arr = np.atleast_2d(fr.preds[key])
        n_s = arr.shape[0]
        V = np.full((len(libs_sub), n_s), np.nan, dtype=float)
        for idx_l, lib in enumerate(libs_sub):
            m = mask & (grp == lib) & np.isfinite(arr).all(axis=0)
            if m.sum() >= 10:
                rhos = [float(spearmanr(arr[s, m], y_vec[m])[0]) for s in range(n_s)]
                if all(np.isfinite(rhos)):
                    V[idx_l, :] = rhos
        n_d = int(np.isfinite(V).any(axis=1).sum())
        counts, seed_w = make_domain_seed_draws(len(libs_sub), B=B, seed=seed_rng, seed_counts=(2, 3))
        pt, reps = boot_domain_seeds(V, B=B, counts=counts, seed_weights=seed_w)
        lo, hi = np.nanquantile(reps, [0.025, 0.975])
        return pt, float(lo), float(hi), reps, n_d

    stat_rows = []
    reps_cache = {}
    for task_name, mask, y_vec, key_idx in (
        ('all_variants', use_all, y_all, 1),
        ('doubles_epistasis', use_epi, y_epi, 2),
    ):
        for strat_name, libs_s in (('all_28', test_libs), ('homologous_16', hom_libs), ('clean_12', clean_libs)):
            for m_info in active_models:
                m_label, ck, ke, _col, _ht = m_info
                key = ck if key_idx == 1 else ke
                if key is None:
                    continue
                pt, lo, hi, reps, n_d = eval_stratum(libs_s, key, mask, y_vec, seed_rng=19)
                reps_cache[(task_name, strat_name, m_label)] = reps
                stat_rows.append(dict(task=task_name, stratum=strat_name, model=m_label, key=key,
                                      rho=pt, lo=lo, hi=hi, n_domains=n_d))

    # Paired inflation delta (homologous - clean) & paired gap vs ESM-MSR on clean_12
    stat_df = pd.DataFrame(stat_rows)
    for idx_r, r in stat_df.iterrows():
        ref_r = reps_cache.get((r['task'], r['stratum'], 'ESM-MSR'))
        cur_r = reps_cache.get((r['task'], r['stratum'], r['model']))
        if ref_r is not None and cur_r is not None and r['model'] != 'ESM-MSR':
            stat_df.loc[idx_r, 'p_vs_esm_msr'] = boot_pvalue(ref_r - cur_r, null=0.0)
        else:
            stat_df.loc[idx_r, 'p_vs_esm_msr'] = np.nan

    atomic_to_csv(stat_df, out_sc / 'mega_clean_vs_homologous_test.csv', index=False, float_format='%.4f')

    # Build 3-panel composite figure: mega_clean_vs_homologous_test.{png,pdf}
    fig = plt.figure(figsize=(16.4, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1.25])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel a: 28 mega_splits test domains best train partner (Identity vs TM-score)
    for strat, col, mk, lbl in (
        ('clean', '#1baf7a', 'o', f'Homology-clean test domains (n = {len(clean_libs)})'),
        ('homologous', '#d62728', 's', f'Train-homologous test domains (n = {len(hom_libs)})'),
    ):
        sub = meta_df[meta_df.stratum == strat]
        ax_a.scatter(sub['best_train_identity'], sub['best_train_tm'], s=44, c=col, marker=mk,
                     alpha=0.88, edgecolors='black', linewidths=0.6, label=lbl, zorder=4)
    ax_a.axvline(ID_CUT, color='#52514e', ls='--', lw=1.1, zorder=2)
    ax_a.axhline(TM_CUT, color='#52514e', ls='--', lw=1.1, zorder=2)
    ax_a.axvspan(ID_CUT, 1.02, color='#fdecea', alpha=0.45, zorder=0)
    ax_a.axhspan(TM_CUT, 1.02, color='#fdecea', alpha=0.45, zorder=0)
    ax_a.text(ID_CUT + 0.015, 0.35, 'id > 0.40', fontsize=7.8, color='#52514e', rotation=90, va='center')
    ax_a.text(0.03, TM_CUT + 0.02, 'TM > 0.70', fontsize=7.8, color='#52514e')
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel('Max sequence identity to mega_splits train')
    ax_a.set_ylabel('Max Foldseek TM-score to mega_splits train')
    ax_a.set_title(f'mega_splits test set: {len(hom_libs)}/{len(test_libs)} domains violate holdout rule', loc='left', fontsize=9.4, fontweight='bold')
    ax_a.legend(loc='lower right', fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95)
    ax_a.grid(color=style.GRID, lw=0.6, zorder=0)
    _panel_label(ax_a, 'a', x=-0.12, y=1.04)

    # Panel b: Per-domain Spearman on all variants (Homologous 16 vs Clean 12)
    models_b = [m[0] for m in active_models]
    cols_b = {m[0]: m[3] for m in active_models}
    yb = np.arange(len(models_b))
    h_bar = 0.34
    sub_all = stat_df[stat_df.task == 'all_variants']
    for j, (st_key, st_lbl, alpha_v, hatch_v) in enumerate((
        ('homologous_16', f'Train-homologous ({len(hom_libs)} domains)', 0.48, '////'),
        ('clean_12', f'Homology-clean ({len(clean_libs)} domains)', 0.94, None),
    )):
        s_df = sub_all[sub_all.stratum == st_key].set_index('model').reindex(models_b)
        offs = yb + (j - 0.5) * h_bar
        vals = s_df['rho'].to_numpy(float)
        los = s_df['lo'].to_numpy(float)
        his = s_df['hi'].to_numpy(float)
        for i_m, m_lbl in enumerate(models_b):
            c_m = cols_b[m_lbl]
            ax_b.barh(offs[i_m], vals[i_m], h_bar, color=c_m, alpha=alpha_v,
                      edgecolor=c_m if hatch_v else 'black', hatch=hatch_v, lw=0.7, zorder=3)
            ax_b.errorbar(vals[i_m], offs[i_m], xerr=[[max(vals[i_m] - los[i_m], 0)], [max(his[i_m] - vals[i_m], 0)]],
                          fmt='none', ecolor='#222222', elinewidth=1.0, capsize=2.0, zorder=5)
            ax_b.text(max(his[i_m], vals[i_m]) + 0.008, offs[i_m], f'{vals[i_m]:.3f}', va='center', fontsize=7.1,
                      fontweight='bold' if (st_key == 'clean_12' and m_lbl == 'ESM-MSR') else 'normal')
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(models_b, fontsize=8.2)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.45, 0.96)
    ax_b.set_xlabel('Mean per-domain Spearman ρ (all variants)')
    ax_b.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    ax_b.set_title('All-variant ΔΔG ranking: homologous vs clean test domains', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_b, 'b', x=-0.06, y=1.04)

    # Panel c: Epistasis dddG Spearman on doubles (Homologous vs Clean)
    sub_epi = stat_df[stat_df.task == 'doubles_epistasis']
    models_c = [m[0] for m in active_models if m[2] is not None]
    yc = np.arange(len(models_c))
    for j, (st_key, alpha_v, hatch_v) in enumerate((
        ('homologous_16', 0.48, '////'),
        ('clean_12', 0.94, None),
    )):
        s_df = sub_epi[sub_epi.stratum == st_key].set_index('model').reindex(models_c)
        offs = yc + (j - 0.5) * h_bar
        vals = s_df['rho'].to_numpy(float)
        los = s_df['lo'].to_numpy(float)
        his = s_df['hi'].to_numpy(float)
        for i_m, m_lbl in enumerate(models_c):
            c_m = cols_b[m_lbl]
            ax_c.barh(offs[i_m], vals[i_m], h_bar, color=c_m, alpha=alpha_v,
                      edgecolor=c_m if hatch_v else 'black', hatch=hatch_v, lw=0.7, zorder=3)
            ax_c.errorbar(vals[i_m], offs[i_m], xerr=[[max(vals[i_m] - los[i_m], 0)], [max(his[i_m] - vals[i_m], 0)]],
                          fmt='none', ecolor='#222222', elinewidth=1.0, capsize=2.0, zorder=5)
            ax_c.text(max(his[i_m], vals[i_m], 0.0) + 0.010, offs[i_m], f'{vals[i_m]:.3f}', va='center', fontsize=7.1,
                      fontweight='bold' if (st_key == 'clean_12' and m_lbl == 'ESM-MSR') else 'normal')
    ax_c.axvline(0.0, color='#666666', lw=0.8, ls=':')
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([] if models_c == models_b else models_c, fontsize=8.2)
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.10, 0.72)
    ax_c.set_xlabel('Mean per-domain Spearman ρ (double-mutant ΔΔΔG)')
    ax_c.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    n_hom_epi = int(sub_epi[sub_epi.stratum == 'homologous_16']['n_domains'].max())
    n_cln_epi = int(sub_epi[sub_epi.stratum == 'clean_12']['n_domains'].max())
    ax_c.set_title(f'Epistasis ΔΔΔG ranking ({n_hom_epi} homologous vs {n_cln_epi} clean libraries)', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_c, 'c', x=-0.04, y=1.04)

    fig.legend(
        handles=[
            Patch(fc='#888888', ec='#666666', alpha=0.48, hatch='////', label=f'Train-homologous test domains ({len(hom_libs)} all-variant / {n_hom_epi} epistasis)'),
            Patch(fc='#888888', ec='black', alpha=0.94, label=f'Homology-clean test domains ({len(clean_libs)} all-variant / {n_cln_epi} epistasis)'),
        ],
        loc='upper center', bbox_to_anchor=(0.66, 0.995), ncol=2, fontsize=8.2,
        frameon=True, facecolor='white', edgecolor='#cccccc',
    )
    fig.text(
        0.01, 0.01,
        f'All retrained models in panels b–c were trained on mega_splits train (3 seeds each; published checkpoints carry overlap with test).  '
        f'Whiskers: 95% joint domain × seed bootstrap (B = {B:,}).',
        fontsize=7.8, color='#444444', ha='left', va='bottom',
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.93))
    style.save(fig, out_sc / 'mega_clean_vs_homologous_test')


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------
def run(args=None) -> int:
    if args is None:
        ap = base_parser(STAGE, DESCRIPTION)
        args = ap.parse_args()
    B = resolve_b(args)
    print(f'[p10:{STAGE}] Running B={B} ...', flush=True)
    run_reviewer_reference(B=B)
    run_20_split_construction()
    run_mega_clean_vs_homologous(B=B)
    print(f'[p10:{STAGE}] Completed.', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    return run(ap.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
