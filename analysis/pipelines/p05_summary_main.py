#!/usr/bin/env python3
"""Pipeline p05: Main-text Figure 3 composite, thermostability & ProteinGym summaries, and ESM-MSR vs ESM3dG.

Consolidates:
  - 01_summary.py                  -> results/06_summary/{pg_categories, pg_categories_per_scaffold, thermostability_sets, thermostability_sets.csv}
                                      and results/00_main_text_figs/figure3.{png,pdf}
  - 01b_thermostability_pvalues.py -> results/06_summary/{thermostability_sets_pvalues.{csv,md}, thermostability_sets_best.csv, thermostability_sets_msr_heads.csv}
  - 22_esm_msr_vs_esm3dg.py        -> results/99_misc/esm_msr_vs_esm3dg/esm_msr_vs_esm3dg.{csv,md,png,pdf}

Key methodological fixes vs legacy scripts:
  - Calls msr_analysis.proteingym.{load_sigma_pgym, sigma_delta_pvalues, draw_property_sigma_panel} directly (zero importlib).
  - Never mutates registry.MODELS in place; uses local label overrides.
  - Fixes 22_esm_msr_vs_esm3dg.py KeyError ('pg:esm3dg' -> 'ESM3ΔG' via load_all_models_pgym) and resamples ESM-MSR seeds jointly with assays.
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import out_dir
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import EXT_ORDER, fresh_frames, model_key_for_dataset, seed_frame
from msr_analysis.heldout import flags as heldout_flags
from msr_analysis.paths import REFERENCE, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.proteingym import (
    PG_CATEGORIES,
    boot_pgym_categories,
    draw_property_sigma_panel,
    load_all_models_pgym,
    load_sigma_pgym,
    sigma_delta_pvalues,
)
from msr_analysis.registry import DMS3, LABEL, MODELS
from msr_analysis.stats import boot_pvalue, fmt_p, holm

from . import base_parser, resolve_b

STAGE = 'summary_main'
DESCRIPTION = 'Main-text Figure 3 composite, thermostability & ProteinGym category summaries, ESM-MSR vs ESM3dG'

MODEL_ORDER = list(M.ordered(M.UNIFIED + ['ESM-MSR σ=0.5']))
HOLLOW_DOMAINOME = [m for m in MODEL_ORDER if M.BY_NAME[m].two_versions]
INFERRED = {'MGnify': 'MGnify Inferred\nDDGs', 'Rocklin designs': 'Rocklin Designs\nInferred DDGs'}
INFERRED_MODEL = {
    'ESM-MSR': 'ESM-MSR', 'ESM-MSR σ=0.5': 'ESM-MSR σ=0.5',
    'SPURS': 'SPURS', 'ThermoMPNN': 'ThermoMPNN',
    'ProteinMPNN': 'ProteinMPNN', 'ProSST': 'ProSST', 'ESM3-small-open': 'ESM3-small-open',
    'GeoDDG-3D': 'GeoDDG-3D', 'ESM3dG': 'ESM3ΔG', 'MutateEverything': 'MutateEverything',
}
MIN_COVERAGE = 0.8
EXCLUDE_THERMO = {'s571', 'ptmuld'}
THERMO_NOTE = (
    'Seeds 1-3 averaged for the retrained models (seeds resampled with the draws); 95% mutation-level bootstrap. '
    'Singles models on the single-mutant sets, models trained with doubles on PTMUL (ThermoMPNN additive; Rosetta Cartesian ddG throughout). '
    'Hollow markers: singles models (ESM-MSR WT-view, MutateEverything Stage 1, ThermoMPNN, SPURS single); '
    'filled: models trained with doubles (ESM-MSR Dual-view, MutateEverything Stage 2, SPURS-multi) and the models with one version. '
    'SPURS is the retrain with the trained adapter. Sets ordered by the performance of ESM-MSR (mean of seeds 1-3), highest first. '
    'Red ring = documented training overlap (GeoDDG-3D, ESM3ΔG; leak audit; '
    'labels show most severe overlap tier & share: M = exact mutations, m = constituent singles, P = same protein, H = close homolog, train src = training data source).'
)


# ----------------------------------------------------------------------------------------------
# 1. Summary figures & Figure 3 (01_summary.py)
# ----------------------------------------------------------------------------------------------
def _inferred_models(pooled: bool = False) -> dict:
    f = results_dir('mgnify', create=False) / ('mgnify_models_pooled.csv' if pooled else 'mgnify_models_per_scaffold.csv')
    if not f.exists():
        return {}
    out: dict = {}
    for r in pd.read_csv(f).itertuples():
        if r.model in INFERRED_MODEL and r.coverage >= MIN_COVERAGE:
            out.setdefault(r.dataset, (int(r.n_set), {}))[1][INFERRED_MODEL[r.model]] = (
                float(r.rho), float(r.lo), float(r.hi),
                int(r.n_variants if pooled else r.n_scaffolds),
                float(r.coverage), bool(getattr(r, 'overlap', False)),
            )
    return out


def _read_pg_summary_df(B: int) -> tuple[pd.DataFrame, int, int, str]:
    pg_dir = results_dir('proteingym', create=False)
    sum_p = pg_dir / 'pg_summary.csv' if (pg_dir / 'pg_summary.csv').exists() else pg_dir / 'pg_summary_tsu.csv'
    sig_p = pg_dir / 'pg_sigma_summary.csv' if (pg_dir / 'pg_sigma_summary.csv').exists() else pg_dir / 'pg_sigma_summary_tsu.csv'
    if sum_p.exists() and sig_p.exists():
        R = pd.concat([pd.read_csv(sum_p), pd.read_csv(sig_p)], ignore_index=True)
        R = R[R['model'] != 'ESM-MSR σ=1']
    else:
        mean_df, seeds_map = load_all_models_pgym(heldout=True)
        R, _ = boot_pgym_categories(seeds_map, mean_df['category'], B=B)
        dom_fr = load_frame('domainome')
        dom_s, _ = seed_frame('domainome', dom_fr, models=MODEL_ORDER)
        pt_d, rp_d = bootstrap_rho({'domainome': dom_s}, list(dom_s.preds), kind='protein', B=B, seed=30, per_protein=True)
        dom_rows = [
            dict(model=m, column='Domainome', value=pt_d['domainome'][m][0], lo=ci(rp_d['domainome'][m])[0], hi=ci(rp_d['domainome'][m])[1], n_dms=pt_d['domainome'][m][3])
            for m in pt_d['domainome']
        ]
        R = pd.concat([R, pd.DataFrame(dom_rows)], ignore_index=True)

    note_file = pg_dir / 'pg_sigma_note.txt'
    if note_file.exists():
        toks = note_file.read_text().split()[::2][:2]
        k05, n_mm = int(toks[0]), int(toks[1])
    else:
        k05, n_mm = 30, 30

    tsu_note_f = pg_dir / 'pg_tsu_note.txt'
    tsu_note = tsu_note_f.read_text().strip() if tsu_note_f.exists() else '40 homologous ProteinGym assays and 86 homologous Domainome domains'
    return R, k05, n_mm, tsu_note


def _pg_prepare(B: int, pooled: bool = True, sigma_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict:
    R, k05, n_mm, tsu_note = _read_pg_summary_df(B=B)
    label_overrides = {
        'Rosetta': 'Rosetta (singles)',
        'ESM-MSR': 'ESM-MSR σ = 1',
        'ESM-MSR σ=0.5': 'ESM-MSR σ = 0.5' + ('' if k05 == n_mm else f' (combined on {k05} of {n_mm})'),
        'ESM3-small-open': 'ESM3-small-open (σ = 0)',
    }
    fl = heldout_flags()
    n_fl = {d: (int(g['homologous'].sum()), len(g)) for d, g in fl.groupby('dataset')}
    n_pg = int(R[(R['model'] == 'ESM-MSR') & (R['column'] == 'PG avg')]['n_dms'].iloc[0])
    n_dom = int(R[(R['model'] == 'ESM-MSR') & (R['column'] == 'Domainome')]['n_dms'].iloc[0])
    note = (
        'Held-out = not a homolog of the training set under the rule that built the Homology-Restricted MegaScale split '
        '(any of: MMseqs2 E-value ≤ 1e-3, identity > 0.40 over the shorter sequence, Foldseek E-value ≤ 1e-3, Foldseek TM-score > 0.70, '
        'against the capped training libraries; the same four-part filter for every set); left out: '
        + tsu_note
        + f' ({n_pg} ProteinGym assays and {n_dom} domains remain); MGnify {n_fl["mgnify"][0]:,} of {n_fl["mgnify"][1]:,} scaffolds '
        f'and Rocklin designs {n_fl["rocklin_denovo"][0]} of {n_fl["rocklin_denovo"][1]:,}.\n'
        'Seeds 1-3 averaged for the retrained models. Singles versions on singles assays / Domainome; doubles versions on PGym multi-mutants '
        '(MutateEverything Stage 1 / Stage 2; ThermoMPNN additive / -D on doubles; SPURS single / SPURS-multi). Rosetta: PGym singles only (no Domainome).'
    )

    table = {(r.column, r.model): r.value for r in R.itertuples()}
    cis = {(r.column, r.model): (r.lo, r.hi) for r in R.itertuples()}
    cols = ['PG avg', 'Domainome']
    labels = [f'ProteinGym avg.\n({n_pg} held-out assays)', f'Domainome avg.\n({n_dom} held-out domains)']
    inf = _inferred_models(pooled)
    extra = [(INFERRED[d], inf[d][0], inf[d][1]) for d in INFERRED if d in inf]
    groups = cols + [f'x{i}' for i in range(len(extra))]
    labels += [
        (f'{n.split(chr(10))[0].split()[0]} Pooled\n({round(k / 1000):,}k variants)' if pooled else f'{n.split(chr(10))[0].split()[0]} Avg.\n({k:,} held-out WTs)')
        for n, k, _ in extra
    ]
    rings = set()
    for i, (_, _, vals) in enumerate(extra):
        for m, v in vals.items():
            table[(f'x{i}', m)], cis[(f'x{i}', m)] = v[0], (v[1], v[2])
            if v[5]:
                rings.add((f'x{i}', m))
    if ('PG avg', 'ESM3ΔG') in table:
        rings.add(('PG avg', 'ESM3ΔG'))
    hollow = {('Domainome', m) for m in HOLLOW_DOMAINOME}

    if sigma_cache is not None:
        t_all, Rp = sigma_cache
    else:
        t_held, per_seed_held = load_sigma_pgym(heldout=True, drop_large=True, include_fallback_small=False)
        b_sig = max(B, 10000) if B > 100 else B
        Rp = sigma_delta_pvalues(t_held, per_seed_dep=per_seed_held, B=b_sig, seed=3)
        t_all, _ = load_sigma_pgym(heldout=False, drop_large=True, include_fallback_small=False)

    mm_note = (
        f'ESM-MSR σ = 1, σ = 0.5 and ESM3-small-open (σ = 0, the same architecture with the adapters off) are read as deployed: '
        f'the combined prediction on the {n_mm} ProteinGym assays with multi-mutants, the WT head on the others '
        '(Domainome has no multi-mutants, so its models with two versions are the hollow singles versions). '
        'Right panel: the same readout per ProteinGym assay (one dot per assay; grey thin x = assays homologous to the training set, '
        'drawn but left out of the box, median, mean, header counts, and p-values; box = IQR, thick line = median with its value, thin line = mean; '
        'p under the ticks: two-sided bootstrap over assays & seeds that mean Δρ = 0, Holm-adjusted over the 10 category × σ comparisons).'
    )
    note += '\n' + mm_note
    if extra:
        part = [f'{m} on {n.replace(chr(10), " ")} ({v[3]:,} of {k:,} {"variants" if pooled else "scaffolds"})' for n, k, vals in extra for m, v in vals.items() if v[4] < 0.999]
        how = (
            'one Spearman rho over all measured variants of the set (seeds 1-3 averaged), whiskers resample scaffolds & seeds; '
            'ESM-MSR is read on the observed Tobit-linked scale h(dG_wt + latent) − dG_wt, the other models raw'
            if pooled else
            'mean over WTs of per-WT Spearman rho (≥ 10 measured variants), seeds 1-3 averaged, whiskers resample WTs & seeds'
        )
        note += (
            '\nMGnify and Rocklin Designs Inferred DDGs: ' + how + '. Multi-mutant versions are read (ESM-MSR combined head, SPURS-multi, '
            'ThermoMPNN-D, MutateEverything stage 2; ProteinMPNN and ProSST direct); ESM3-small-open is the WT head; '
            'ESM3ΔG was fine-tuned on MGnify stability data (red rings: training overlap).'
            + (' Models scored on part of a set only: ' + '; '.join(part) + '.' if part else '')
        )
    return dict(
        pooled=pooled, groups=groups, cols=cols, labels=labels, table=table, cis=cis,
        hollow=hollow, rings=rings, extra=extra, t_all=t_all, Rp=Rp, note=note,
        label_overrides=label_overrides,
    )


def _overlap_short_labels() -> dict[tuple[str, str], str]:
    """Return ``(dataset, model_display) -> '<tier> <pct>'`` for the most severe overlap tier in leak_audit_matrix.csv."""
    df = pd.read_csv(REFERENCE / 'leak_audit_matrix.csv')
    model_map = {'GeoDDG-3D': 'GeoDDG-3D', 'ESM3dG': 'ESM3ΔG'}
    sup_to_col = {
        '#': ('M', 'exact'),
        'ᴹ': ('M', 'exact'),
        'ᵐ': ('m', 'constituent'),
        'ᴾ': ('P', 'P'),
        'ᴴ': ('H', 'H'),
        'ʰ': ('h', 'h'),
    }
    out: dict[tuple[str, str], str] = {}
    for raw_m, disp_m in model_map.items():
        for r in df[df['model'] == raw_m].itertuples():
            sup = str(getattr(r, 'sup', '') or '').strip()
            n_val = float(r.n) if np.isfinite(r.n) and r.n > 0 else np.nan
            if sup not in sup_to_col or not np.isfinite(n_val):
                continue
            code, col = sup_to_col[sup]
            pct = 100.0 * float(getattr(r, col)) / n_val
            pct_str = f'{pct:.0f}%' if pct >= 0.95 else f'{pct:.1f}%'
            out[(str(r.dataset), disp_m)] = f'{code} {pct_str}'
    # Map panel b group names for ESM3ΔG (x0 = MGnify Pooled, x1 = Rocklin Pooled)
    out[('x0', 'ESM3ΔG')] = 'train src'
    out[('x1', 'ESM3ΔG')] = 'train src'
    return out


def _draw_pg_left(ax, pg: dict, ylim=(0.04, 1.0)) -> None:
    groups = pg['groups']
    style_ov = {k: {'label': v} for k, v in pg['label_overrides'].items()}
    style.dot_panel(
        ax, groups, MODEL_ORDER, pg['table'], pg['cis'],
        highlight=['ESM-MSR', 'ESM-MSR σ=0.5'], hollow=pg['hollow'],
        ylabel=r'Spearman $\rho$' if pg['pooled'] else r'Spearman $\rho$ (mean over assays / domains / WTs)',
        xlabels=pg['labels'], rot=0, size=6.2, spread=0.84, ylim=ylim, style_overrides=style_ov,
    )
    if len(groups) > len(pg['cols']):
        ax.axvline(len(pg['cols']) - 0.5, color='#999999', lw=0.95, ls='--', zorder=2)
    offs_all = np.linspace(-0.42, 0.42, len(MODEL_ORDER))
    ov_labels = _overlap_short_labels()
    for (g, m) in sorted(pg.get('rings', ()), key=lambda k: MODEL_ORDER.index(k[1])):
        x_pt = groups.index(g) + offs_all[MODEL_ORDER.index(m)]
        y_pt = pg['table'][(g, m)]
        ax.plot(x_pt, y_pt, marker='o', ms=11.5, mfc='none', mec=style.RING, mew=1.1, ls='none', zorder=6)
        lbl = ov_labels.get((g, m))
        if lbl:
            lo_ci, hi_ci = pg['cis'].get((g, m), (y_pt, y_pt))
            if g == 'x0':
                y_txt, va_txt = (hi_ci if np.isfinite(hi_ci) else y_pt) + 0.030, 'bottom'
            else:
                y_txt, va_txt = (lo_ci if np.isfinite(lo_ci) else y_pt) - 0.030, 'top'
            ax.text(
                x_pt, y_txt, lbl, ha='center', va=va_txt,
                fontsize=6.6, fontweight='semibold', color=style.RING, zorder=8,
                bbox=dict(fc='white', ec='none', alpha=0.72, pad=0.4),
            )
    hs = [
        Line2D([0], [0], label=pg['label_overrides'].get(m, MODELS[m]['label']), **style.marker_kwargs(m, size=6.8))
        for m in MODEL_ORDER
    ]
    if pg.get('rings'):
        hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec=style.RING, mew=1.1, ls='none', label='training overlap (ESM3ΔG)'))
    hs += style.version_handles()
    ax.legend(
        handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=7.6,
        frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.0, handletextpad=0.3, borderpad=0.4, labelspacing=0.35,
    )


def _thermo_data() -> dict:
    S = pd.read_csv(results_dir('thermostability', create=False) / 'thermostability_rho.csv')
    S = S[~S['dataset'].isin(EXCLUDE_THERMO)].copy()
    sets_all = sorted(S['dataset'].unique())
    seed1 = S[S['model'] == 'ESM-MSR'].set_index('dataset')['rho']
    f_ab = results_dir('ablation', create=False) / 'cells_long.csv'
    mean_ab = {}
    if f_ab.exists():
        c = pd.read_csv(f_ab)
        c = c[(c['family'] == 'FINAL_p0') & (c['head'] == 'primary') & c['dataset'].isin(sets_all)]
        if c['seed'].nunique() >= 2:
            mean_ab = c.groupby('dataset')['rho'].mean().to_dict()
    val = {d: float(mean_ab.get(d, seed1[d])) for d in sets_all}
    sets = sorted(sets_all, key=lambda d: -val[d])
    models = [m for m in EXT_ORDER if (S['model'] == m).any()]
    S = S[S['model'].isin(models)].copy()
    S['version'] = [
        'singles' if M.BY_NAME[r.model].two_versions and r.key == M.BY_NAME[r.model].single
        else ('doubles' if M.BY_NAME[r.model].two_versions else '')
        for r in S.itertuples()
    ]
    ymin = min(0.10, float(np.floor((np.nanmin(S['lo']) - 0.03) * 20) / 20))
    return dict(S=S, sets=sets, val=val, models=models, ymin=ymin)


def _draw_thermo(ax, th: dict, ylim=None) -> list:
    S, sets, models = th['S'], th['sets'], th['models']
    table = {(r.dataset, r.model): r.rho for r in S.itertuples()}
    cis = {(r.dataset, r.model): (r.lo, r.hi) for r in S.itertuples()}
    hollow = {(r.dataset, r.model) for r in S.itertuples() if r.version == 'singles'}
    spread = 0.78
    label_ov = {'Rosetta': 'Rosetta (singles)'}
    style.dot_panel(
        ax, sets, models, table, cis,
        highlight=['ESM-MSR'], hollow=hollow,
        ylabel=r'Spearman $\rho$ (pooled)',
        xlabels=[LABEL.get(d, d) for d in sets],
        rot=0, ylim=ylim or (th['ymin'], 0.98), spread=spread, size=6.0,
    )
    offs = np.linspace(-spread / 2, spread / 2, len(models))
    ov_labels = _overlap_short_labels()
    for i, d in enumerate(sets):
        for j, m in enumerate(models):
            r = S[(S['dataset'] == d) & (S['model'] == m)]
            if len(r) and r.iloc[0]['leak_flag']:
                x_pt = i + offs[j]
                y_pt = float(r.iloc[0]['rho'])
                ax.plot(x_pt, y_pt, marker='o', ms=10.5, mfc='none', mec=style.RING, mew=1.1, ls='none', zorder=6)
                lbl = ov_labels.get((d, m))
                if lbl:
                    lo_ci, hi_ci = float(r.iloc[0]['lo']), float(r.iloc[0]['hi'])
                    # Place GeoDDG-3D labels above (except S461 where it sits near neighbours -> below), ESM3ΔG labels below
                    place_above = (m == 'GeoDDG-3D' and d != 's461')
                    pad_y = 0.036 if (m == 'ESM3ΔG' and d == 's2648') else 0.026
                    y_txt = (hi_ci + pad_y) if place_above else (lo_ci - pad_y)
                    va_txt = 'bottom' if place_above else 'top'
                    ax.text(
                        x_pt, y_txt, lbl, ha='center', va=va_txt,
                        fontsize=6.6, fontweight='semibold', color=style.RING, zorder=8,
                        bbox=dict(fc='white', ec='none', alpha=0.72, pad=0.4),
                    )
    hs = [Line2D([0], [0], label=label_ov.get(m, MODELS[m]['label']), **style.marker_kwargs(m, size=6.8)) for m in models]
    flagged = [m for m in models if S[S['model'] == m]['leak_flag'].any()]
    hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec=style.RING, mew=1.1, ls='none', label=f'training overlap ({", ".join(flagged)})'))
    return hs + style.version_handles()


def run_summary_and_figure3(B: int) -> pd.DataFrame:
    out_sum = out_dir('summary')
    out_main = out_dir('main_text_figs')

    t_held, per_seed_held = load_sigma_pgym(heldout=True, drop_large=True, include_fallback_small=False)
    b_sig = max(B, 10000) if B > 100 else B
    Rp_shared = sigma_delta_pvalues(t_held, per_seed_dep=per_seed_held, B=b_sig, seed=3)
    t_all_shared, _ = load_sigma_pgym(heldout=False, drop_large=True, include_fallback_small=False)
    sig_cache = (t_all_shared, Rp_shared)

    # pg_categories (pooled and per_scaffold)
    pg_main = None
    for pooled in (True, False):
        pg = _pg_prepare(B=B, pooled=pooled, sigma_cache=sig_cache)
        if pooled:
            pg_main = pg
        fig = plt.figure(figsize=(15.6, 6.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.43, 0.57], wspace=0.15)
        ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        _draw_pg_left(ax, pg)
        draw_property_sigma_panel(ax2, pg['t_all'], pg['Rp'], mark_leaks=True, stats_without_leaks=True, compact=True)
        kwt = dict(fontsize=7.4, color='#444444')
        tt = fig.text(0.012, 0.008, style.wrap_to_px(fig, pg['note'], 0.975 * fig.bbox.width, **kwt), va='bottom', ha='left', **kwt)
        foot = tt.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
        fig.subplots_adjust(left=0.05, right=0.995, top=0.89, bottom=foot + 0.115)
        for a_ax, letter in ((ax, 'b'), (ax2, 'c')):
            pos = a_ax.get_position()
            fig.text(pos.x0 - (0.047 if letter == 'b' else 0.052), 0.985, letter, ha='left', va='top', fontsize=14, fontweight='bold')
        style.save(fig, out_sum / ('pg_categories' if pooled else 'pg_categories_per_scaffold'))

    # thermostability_sets + CSV
    th = _thermo_data()
    fig_t, ax_t = plt.subplots(figsize=(11.2, 5.8))
    hs_t = _draw_thermo(ax_t, th)
    fig_t.legend(handles=hs_t, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3)
    fig_t.suptitle('External thermostability sets: Spearman ρ across models (seeds 1-3 averaged; whiskers: 95% mutation-level bootstrap)', x=0.012, ha='left', y=0.985, fontsize=10.5, fontweight='bold')
    kw = dict(fontsize=8.0, color='#444444')
    t_txt = fig_t.text(0.012, 0.01, style.wrap_to_px(fig_t, THERMO_NOTE, 0.96 * fig_t.bbox.width, **kw), va='bottom', ha='left', **kw)
    foot_t = t_txt.get_window_extent(fig_t.canvas.get_renderer()).height / fig_t.bbox.height
    fig_t.tight_layout(rect=(0.0, foot_t + 0.02, 1.0, 0.86))
    style.save(fig_t, out_sum / 'thermostability_sets')

    order_map = {d: i + 1 for i, d in enumerate(th['sets'])}
    D_th = th['S'].assign(order=th['S']['dataset'].map(order_map), msr_p0_seed_mean=th['S']['dataset'].map(th['val'])).sort_values(['order', 'model'])
    atomic_to_csv(D_th, out_sum / 'thermostability_sets.csv', index=False, float_format='%.4f')

    # Main-text Figure 3 (a, b, c composite)
    assert pg_main is not None
    fig3 = plt.figure(figsize=(15.6, 11.6))
    gs3 = fig3.add_gridspec(2, 2, height_ratios=[0.45, 0.55], width_ratios=[0.43, 0.57], hspace=0.42, wspace=0.15)
    axa, axb, axc = fig3.add_subplot(gs3[0, :]), fig3.add_subplot(gs3[1, 0]), fig3.add_subplot(gs3[1, 1])
    hs3 = _draw_thermo(axa, th, ylim=(th['ymin'], 1.2))
    axa.set_yticks([t for t in np.arange(np.ceil(th['ymin'] * 5) / 5, 1.0001, 0.2)])
    axa.legend(handles=hs3, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3)
    _draw_pg_left(axb, pg_main)
    draw_property_sigma_panel(axc, pg_main['t_all'], pg_main['Rp'], mark_leaks=True, stats_without_leaks=True, compact=True)
    note3 = 'a. ' + THERMO_NOTE + '\nb, c. ' + pg_main['note']
    kwt3 = dict(fontsize=7.4, color='#444444')
    tt3 = fig3.text(0.012, 0.006, style.wrap_to_px(fig3, note3, 0.975 * fig3.bbox.width, **kwt3), va='bottom', ha='left', **kwt3)
    foot3 = tt3.get_window_extent(fig3.canvas.get_renderer()).height / fig3.bbox.height
    fig3.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=foot3 + 0.075)
    fig3.canvas.draw()
    pa, pb, pc = (a_x.get_position() for a_x in (axa, axb, axc))
    fig3.text(pa.x0 - 0.04, 0.99, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    ytop = pb.y1 + 0.062
    fig3.text(pb.x0 - 0.047, ytop, 'b', ha='left', va='top', fontsize=14, fontweight='bold')
    fig3.text(pc.x0 - 0.052, ytop, 'c', ha='left', va='top', fontsize=14, fontweight='bold')
    style.save(fig3, out_main / 'figure3')
    return D_th


# ----------------------------------------------------------------------------------------------
# 2. Thermostability paired p-values & PTMUL/PTMUL-D Dual-view vs WT-head test (01b)
# ----------------------------------------------------------------------------------------------
_PVAL_FR: dict = {}
_HEAD_FR: dict = {}
CHUNK = 5000


def _pval_worker(job):
    ds, kind, B_chunk, seed = job
    fr = _PVAL_FR[ds]
    pt, rp = bootstrap_rho({ds: fr}, list(fr.preds), kind=kind, B=B_chunk, seed=seed, per_protein=False)
    return ds, kind, dict(rp[ds]), {m: (v[0], v[2]) for m, v in pt[ds].items()}


def _head_worker(job):
    ds, kind, B_chunk, seed = job
    fr = _HEAD_FR[ds]
    pt, rp = bootstrap_rho({ds: fr}, list(fr.preds), kind=kind, B=B_chunk, seed=seed, per_protein=False)
    return ds, kind, dict(rp[ds]), {m: v[0] for m, v in pt[ds].items()}


def run_thermostability_pvalues(D_th: pd.DataFrame, B: int, workers: int = 4) -> None:
    out = out_dir('summary')
    sets = list(D_th.sort_values('order').drop_duplicates('dataset')['dataset'])
    leak = {(r.dataset, r.model): bool(r.leak_flag) for r in D_th.itertuples()}

    _PVAL_FR.clear()
    for ds, fr in fresh_frames(D_th[D_th['dataset'].isin(sets)]).items():
        _PVAL_FR[ds] = seed_frame(ds, fr)[0]

    jobs = [
        (ds, kind, min(CHUNK, B - i * CHUNK), 1000 * (1 + int(kind == 'protein')) + i)
        for ds in sets for kind in ('mutation', 'protein') for i in range(int(np.ceil(B / CHUNK)))
    ]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_pval_worker, jobs, chunksize=1)
    reps_acc: dict = {}
    pts: dict = {}
    for ds, kind, rp, pt in res:
        for m, v in rp.items():
            reps_acc.setdefault((ds, kind, m), []).append(v)
        pts[ds] = pt
    reps = {k: np.concatenate(v) for k, v in reps_acc.items()}

    rows, best = [], []
    for ds in sets:
        others = [m for m in EXT_ORDER if m != 'ESM-MSR' and m in pts[ds]]
        clean = [m for m in others if not leak.get((ds, m), False)]
        msr = pts[ds]['ESM-MSR'][0]
        for kind in ('mutation', 'protein'):
            e = reps[(ds, kind, 'ESM-MSR')]
            ps = {}
            for m in others:
                d = e - reps[(ds, kind, m)]
                lo, hi = np.nanquantile(d, [0.025, 0.975])
                ps[m] = boot_pvalue(d, null=0.0, alternative='two-sided')
                rows.append(dict(
                    dataset=ds, unit=kind, model=m,
                    excluded_training_overlap=leak.get((ds, m), False),
                    rho_msr=msr, rho_model=pts[ds][m][0], n_model=pts[ds][m][1],
                    delta=msr - pts[ds][m][0], lo=float(lo), hi=float(hi), p=ps[m], B=len(d),
                ))
            adj = dict(zip(clean, holm([ps[m] for m in clean])))
            for r in rows:
                if r['dataset'] == ds and r['unit'] == kind:
                    r['p_holm'] = adj.get(r['model'], np.nan)
            mx = np.max(np.vstack([reps[(ds, kind, m)] for m in clean]), axis=0)
            d_best = e - mx
            top = max(clean, key=lambda m: pts[ds][m][0])
            lo_b, hi_b = np.nanquantile(d_best, [0.025, 0.975])
            best.append(dict(
                dataset=ds, unit=kind, rho_msr=msr, best_model=top, rho_best=pts[ds][top][0],
                delta_point=msr - pts[ds][top][0], delta_vs_max_mean=float(np.nanmean(d_best)),
                lo=float(lo_b), hi=float(hi_b),
                p_vs_best=boot_pvalue(d_best, null=0.0, alternative='two-sided'),
                p_vs_runner_up=ps[top], msr_is_top=bool(msr > pts[ds][top][0]), B=len(d_best),
            ))

    P_df, Bt_df = pd.DataFrame(rows), pd.DataFrame(best)
    Bt_df['p_vs_best_holm_sets'] = np.nan
    for kind in ('mutation', 'protein'):
        m_k = Bt_df['unit'] == kind
        Bt_df.loc[m_k, 'p_vs_best_holm_sets'] = holm(Bt_df.loc[m_k, 'p_vs_best'].to_numpy())
    atomic_to_csv(P_df, out / 'thermostability_sets_pvalues.csv', index=False, float_format='%.6g')
    atomic_to_csv(Bt_df, out / 'thermostability_sets_best.csv', index=False, float_format='%.6g')

    lines = [
        f'**ESM-MSR against the best other model without training overlap** (paired bootstrap, seeds 1-3 averaged, B = {B:,})',
        '',
        '| set | ESM-MSR ρ | best other (ρ) | Δρ | p, mutation resampling | p, protein resampling | p vs best, Holm over sets (mutation / protein) |',
        '|---|---|---|---|---|---|---|',
    ]
    for ds in sets:
        m_r = Bt_df[(Bt_df['dataset'] == ds) & (Bt_df['unit'] == 'mutation')].iloc[0]
        p_r = Bt_df[(Bt_df['dataset'] == ds) & (Bt_df['unit'] == 'protein')].iloc[0]
        lines.append(
            f'| {LABEL.get(ds, ds)} | {m_r.rho_msr:.3f} | {m_r.best_model} ({m_r.rho_best:.3f}) | {m_r.delta_point:+.3f} | '
            f'{fmt_p(m_r.p_vs_best, B=B, prefix="")} | {fmt_p(p_r.p_vs_best, B=B, prefix="")} | '
            f'{fmt_p(m_r.p_vs_best_holm_sets, B=B, prefix="")} / {fmt_p(p_r.p_vs_best_holm_sets, B=B, prefix="")} |'
        )
    lines += [
        '',
        '**Pairwise: ESM-MSR against every model** (Δρ = ESM-MSR − model; p two-sided; Holm over models without overlap of the set)',
        '',
        '| set | model | ρ model | Δρ | 95% interval (mutation) | p (mutation) | Holm | p (protein) | Holm |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for ds in sets:
        for m in [m for m in EXT_ORDER if m != 'ESM-MSR']:
            a_ = P_df[(P_df['dataset'] == ds) & (P_df['model'] == m) & (P_df['unit'] == 'mutation')]
            b_ = P_df[(P_df['dataset'] == ds) & (P_df['model'] == m) & (P_df['unit'] == 'protein')]
            if not len(a_):
                continue
            a_row, b_row = a_.iloc[0], b_.iloc[0]
            lines.append(
                f'| {LABEL.get(ds, ds)} | {m}{" *" if a_row.excluded_training_overlap else ""} | {a_row.rho_model:.3f} | '
                f'{a_row.delta:+.3f} | [{a_row.lo:+.3f}, {a_row.hi:+.3f}] | {fmt_p(a_row.p, B=B, prefix="")} | '
                f'{"–" if a_row.excluded_training_overlap else fmt_p(a_row.p_holm, B=B, prefix="")} | {fmt_p(b_row.p, B=B, prefix="")} | '
                f'{"–" if a_row.excluded_training_overlap else fmt_p(b_row.p_holm, B=B, prefix="")} |'
            )
    atomic_write_text(out / 'thermostability_sets_pvalues.md', '\n'.join(lines) + '\n')

    # PTMUL / PTMUL-D Dual-view vs WT head paired test
    S_all = pd.read_csv(results_dir('thermostability', create=False) / 'thermostability_rho.csv')
    _HEAD_FR.clear()
    for ds in ('ptmul', 'ptmuld'):
        fr = fresh_frames(S_all[S_all['dataset'] == ds])[ds]
        _HEAD_FR[ds] = Frame(ds, fr.spine, {'comb': fr.preds['msrp0:comb'], 'wt': fr.preds['msrp0:wt']})
    h_jobs = [
        (ds, kind, min(CHUNK, B - i * CHUNK), 7300 + i)
        for ds in ('ptmul', 'ptmuld') for kind in ('mutation', 'protein') for i in range(int(np.ceil(B / CHUNK)))
    ]
    with mp.get_context('fork').Pool(workers) as pool:
        h_res = pool.map(_head_worker, h_jobs, chunksize=1)
    h_reps: dict = {}
    h_pts: dict = {}
    for ds, kind, rp, pt in h_res:
        for m, v in rp.items():
            h_reps.setdefault((ds, kind, m), []).append(v)
        h_pts[ds] = pt
    h_rows = []
    for ds in ('ptmul', 'ptmuld'):
        for kind in ('mutation', 'protein'):
            d = np.concatenate(h_reps[(ds, kind, 'comb')]) - np.concatenate(h_reps[(ds, kind, 'wt')])
            d = d[np.isfinite(d)]
            lo, hi = np.quantile(d, [0.025, 0.975])
            pt = h_pts[ds]
            h_rows.append(dict(
                dataset=ds, seeds='seeds 1-3 (mean)', unit=kind,
                rho_combined=pt['comb'], rho_wt=pt['wt'], delta=pt['comb'] - pt['wt'],
                lo=float(lo), hi=float(hi),
                p_one_sided=boot_pvalue(d, null=0.0, alternative='greater'),
                p_two_sided=boot_pvalue(d, null=0.0, alternative='two-sided'),
                B=len(d),
            ))
    atomic_to_csv(pd.DataFrame(h_rows), out / 'thermostability_sets_msr_heads.csv', index=False, float_format='%.6g')


# ----------------------------------------------------------------------------------------------
# 3. ESM-MSR vs ESM3ΔG cross-benchmark table & figure (22_esm_msr_vs_esm3dg.py, bug-fixed)
# ----------------------------------------------------------------------------------------------
def run_esm_msr_vs_esm3dg(B: int) -> pd.DataFrame:
    out = out_dir('esm_msr_vs_esm3dg')
    thermo_sets = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754', 's571', 'ptmul', 'ptmuld']
    leak_df = pd.read_csv(REFERENCE / 'leak_audit_matrix.csv')
    leak = dict(zip(leak_df.loc[leak_df['model'] == 'ESM3dG', 'dataset'], leak_df.loc[leak_df['model'] == 'ESM3dG', 'code'].fillna('')))

    def _paired_fr(ds_n: str, fr: Frame):
        msr_key = model_key_for_dataset('ESM-MSR', fr)
        if msr_key is None or 'esm3dg' not in fr.preds:
            return None
        a, b = fr.preds[msr_key], fr.preds['esm3dg']
        ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        sp = fr.spine.copy()
        sp['use'] = sp['use'].to_numpy() & ok
        return Frame(ds_n, sp, {'ESM-MSR': a, 'ESM3dG': b})

    def _run_group(items: dict, kind: str, per_protein: bool, seed: int) -> list[dict]:
        frames = {k: v[0] for k, v in items.items()}
        pt, reps = bootstrap_rho(frames, ['ESM-MSR', 'ESM3dG'], kind=kind, B=B, seed=seed, per_protein=per_protein)
        g_rows = []
        for k, (_fr, metric, group, lk) in items.items():
            (rm, _ps, n, ng), (re, _ps2, _, _) = pt[k]['ESM-MSR'], pt[k]['ESM3dG']
            d = reps[k]['ESM-MSR'] - reps[k]['ESM3dG']
            g_rows.append(dict(
                group=group, benchmark=k, metric=metric, n=int(ng) if per_protein else int(n),
                msr=rm, msr_lo=ci(reps[k]['ESM-MSR'])[0], msr_hi=ci(reps[k]['ESM-MSR'])[1],
                esm3dg=re, e_lo=ci(reps[k]['ESM3dG'])[0], e_hi=ci(reps[k]['ESM3dG'])[1],
                delta=rm - re, d_lo=float(np.nanquantile(d, 0.025)), d_hi=float(np.nanquantile(d, 0.975)),
                p=boot_pvalue(d, null=0.0, alternative='two-sided'), paired=True, leak=lk,
            ))
        return g_rows

    rows = []
    th_items = {LABEL.get(ds, ds): (pfr, 'pooled rho', 'Thermostability sets', leak.get(ds, '')) for ds in thermo_sets for pfr in [_paired_fr(ds, load_frame(ds))] if pfr is not None}
    rows += _run_group(th_items, 'mutation', False, 11)
    dms_items = {LABEL.get(ds, ds): (pfr, 'pooled rho', 'DMS sets', leak.get(ds, '')) for ds in DMS3 for pfr in [_paired_fr(ds, load_frame(ds))] if pfr is not None}
    rows += _run_group(dms_items, 'mutation', False, 12)
    dom_pfr = _paired_fr('domainome', load_frame('domainome'))
    if dom_pfr is not None:
        rows += _run_group({'Domainome (held-out domains)': (dom_pfr, 'mean per-domain rho', 'Domainome', leak.get('domainome', ''))}, 'protein', True, 13)

    # ProteinGym held-out assays: joint [assay × seed] paired bootstrap via load_all_models_pgym (fixes T['pg:esm3dg'] KeyError!)
    mean_pg, seeds_pg = load_all_models_pgym(heldout=True)
    ok_pg = np.isfinite(mean_pg['ESM-MSR'].to_numpy()) & np.isfinite(mean_pg['ESM3ΔG'].to_numpy())
    mean_pg = mean_pg.loc[ok_pg]
    sub_seeds = {m: seeds_pg[m].loc[mean_pg.index] for m in ('ESM-MSR', 'ESM3ΔG')}
    sum_pg, reps_pg = boot_pgym_categories(sub_seeds, mean_pg['category'], B=B, seed=22)
    pt_lk = {(r.model, r.column): r for r in sum_pg.itertuples()}
    for c in ['PG avg', *PG_CATEGORIES]:
        rm_r, re_r = pt_lk[('ESM-MSR', c)], pt_lk[('ESM3ΔG', c)]
        d_pg = reps_pg['ESM-MSR'][c] - reps_pg['ESM3ΔG'][c]
        bench_lbl = 'PG average' if c == 'PG avg' else c
        rows.append(dict(
            group='ProteinGym (held-out assays)', benchmark=bench_lbl, metric='mean per-assay rho',
            n=int(rm_r.n_dms),
            msr=rm_r.value, msr_lo=rm_r.lo, msr_hi=rm_r.hi,
            esm3dg=re_r.value, e_lo=re_r.lo, e_hi=re_r.hi,
            delta=rm_r.value - re_r.value, d_lo=float(np.nanquantile(d_pg, 0.025)), d_hi=float(np.nanquantile(d_pg, 0.975)),
            p=boot_pvalue(d_pg, null=0.0, alternative='two-sided'), paired=True,
            leak=leak.get({'PG avg': 'PG avg', 'Stability': 'PG Stability'}.get(c, ''), ''),
        ))

    # MGnify & Rocklin rows from 05_mgnify tables
    for fname, metric in (('mgnify_models_pooled.csv', 'pooled rho'), ('mgnify_models_per_scaffold.csv', 'mean per-scaffold rho')):
        fp = results_dir('mgnify', create=False) / fname
        if not fp.exists():
            continue
        d_mg = pd.read_csv(fp)
        d_mg = d_mg[~d_mg['model'].str.contains('same ')]
        for ds in ('MGnify', 'Rocklin designs'):
            a, e = d_mg[(d_mg['dataset'] == ds) & (d_mg['model'] == 'ESM-MSR')], d_mg[(d_mg['dataset'] == ds) & (d_mg['model'] == 'ESM3dG')]
            if a.empty or e.empty:
                continue
            a_r, e_r = a.iloc[0], e.iloc[0]
            n_val = int(a_r['n_variants'] if 'n_variants' in a_r and pd.notna(a_r.get('n_variants')) else a_r['n_scaffolds'])
            rows.append(dict(
                group='MGnify / Rocklin (not held out for ESM3ΔG)', benchmark=ds, metric=metric, n=n_val,
                msr=a_r['rho'], msr_lo=a_r['lo'], msr_hi=a_r['hi'],
                esm3dg=e_r['rho'], e_lo=e_r['lo'], e_hi=e_r['hi'],
                delta=a_r['rho'] - e_r['rho'], d_lo=np.nan, d_hi=np.nan, p=np.nan, paired=False, leak='same source',
            ))

    T = pd.DataFrame(rows)
    atomic_to_csv(T, out / 'esm_msr_vs_esm3dg.csv', index=False, float_format='%.4f')

    headers = ['Benchmark', 'Metric', 'n', 'ESM-MSR\nρ [95% CI]', 'ESM3ΔG\nρ [95% CI]', 'Δρ = ESM-MSR − ESM3ΔG\n[95% CI, paired]', 'ESM3ΔG overlap\n(leak audit)']
    fmt_ci = lambda v, lo, hi: f'{v:.3f}  [{lo:.3f}, {hi:.3f}]' if np.isfinite(lo) else f'{v:.3f}'
    cells = [
        [
            r.benchmark, r.metric, f'{r.n:,}', fmt_ci(r.msr, r.msr_lo, r.msr_hi), fmt_ci(r.esm3dg, r.e_lo, r.e_hi),
            (f'{r.delta:+.3f}  [{r.d_lo:+.3f}, {r.d_hi:+.3f}]' if r.paired else f'{r.delta:+.3f}  (separate intervals)'),
            r.leak or '–',
        ]
        for r in T.itertuples()
    ]
    note = (
        f'ESM-MSR: deployed readout (WT-view on singles, Dual-view where multi-mutants are scored), seeds 1-3 averaged; ESM3ΔG: one run, direct prediction. '
        f'Δρ intervals: paired bootstrap (B = {B:,}, seeds resampled; mutation level for thermostability & DMS, domain level for Domainome, '
        'category-stratified assays × seeds for ProteinGym; MGnify & Rocklin: separate scaffold bootstraps). '
        'ESM3ΔG is fine-tuned on MGnify stability data (960,216 sequences), so MGnify & Rocklin are not held out for it. '
        'Green / red: Δρ interval excludes 0 (ESM-MSR higher / lower).'
    )

    md_lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in headers) + ' |', '|' + '---|' * len(headers)]
    last = None
    for r, row in zip(T.itertuples(), cells):
        if r.group != last:
            md_lines.append(f'| **{r.group}** |' + ' |' * (len(headers) - 1))
            last = r.group
        md_lines.append('| ' + ' | '.join(row) + ' |')
    atomic_write_text(out / 'esm_msr_vs_esm3dg.md', '\n'.join(md_lines) + '\n\n' + note + '\n')

    widths = [4.3, 2.0, 0.9, 2.3, 2.3, 3.0, 1.9]
    n_groups = T['group'].nunique()
    nrows = len(cells) + n_groups
    fig = plt.figure(figsize=(sum(widths) + 0.4, 0.34 * (nrows + 1.2) + 1.5))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 1.25 / fig.get_figheight(), sum(widths) / fig.get_figwidth(), 1 - 1.4 / fig.get_figheight()])
    ax.axis('off')
    body, kinds, last = [], [], None
    for r, row in zip(T.itertuples(), cells):
        if r.group != last:
            body.append([r.group] + [''] * (len(headers) - 1))
            kinds.append(('group', None))
            last = r.group
        body.append(row)
        kinds.append(('row', r))
    t = ax.table(cellText=body, colLabels=headers, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(widths) for w in widths], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#d0d0d0')
        c.set_linewidth(0.5)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
            continue
        kind, r_obj = kinds[i - 1]
        if kind == 'group':
            c.set_facecolor('#f0f0f0')
            c.set_edgecolor('#f0f0f0')
            if j == 0:
                c.set_text_props(ha='left', fontweight='bold')
                c._loc = 'left'
                c.PAD = 0.02
            continue
        c.set_facecolor('#ffffff')
        if j == 0:
            c.set_text_props(ha='left')
            c._loc = 'left'
            c.PAD = 0.04
        if j == 5 and r_obj.paired and np.isfinite(r_obj.d_lo) and (r_obj.d_lo > 0 or r_obj.d_hi < 0):
            c.set_facecolor('#d6ecd9' if r_obj.delta > 0 else '#f6d5d0')
    kw = dict(fontsize=8.2, color='#444444')
    fig.text(0.2 / fig.get_figwidth(), 1.1 / fig.get_figheight(), style.wrap_to_px(fig, note, 0.97 * fig.bbox.width, **kw), va='top', ha='left', **kw)
    style.save(fig, out / 'esm_msr_vs_esm3dg')
    return T


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION)
    ap.add_argument('--workers', type=int, default=4, help='Worker processes for paired bootstrap p-values')
    ap.add_argument('--only', choices=('all', 'figure3'), default='all', help='Optional substage selector')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    D_th = run_summary_and_figure3(B=B)
    if args.only == 'figure3':
        return 0
    run_thermostability_pvalues(D_th, B=B, workers=args.workers)
    run_esm_msr_vs_esm3dg(B=B)
    return 0


if __name__ == '__main__':
    sys.exit(main())
