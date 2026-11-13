#!/usr/bin/env python
"""Thermostability: the 10 external sets, unified main models (including Rosetta), seed 1, mutation-level bootstrap.

Datasets (10 sets; S669, K3822, and mean-of-12 aggregate excluded):
- 8 single-mutant sets: S571, S783, S2648, S8754, S461 (sliced from S669 onto S461 labels),
  Ssym, Q3421, K2369 (sliced from K3822 onto K2369 labels).
- 2 multi-mutant sets: PTMUL and PTMUL-D (doubles only, sliced from PTMUL onto PTMUL-D labels).

Models (9 unified models, seed 1 only):
- Singles sets use singles models (ESM-MSR p0 WT head, MutateEverything st1, ThermoMPNN, SPURS,
  ESM3-small-open WT head) plus ProteinMPNN, ProSST, GeoDDG-3D, and Rosetta Cartesian ddG.
- Multi-mutant sets use multi models (ESM-MSR p0 combined, MutateEverything st2, ThermoMPNN additive on PTMUL /
  ThermoMPNN-D on PTMUL-D, SPURS multi, ESM3-small-open combined, ProteinMPNN, ProSST, GeoDDG-3D,
  and Rosetta Cartesian ddG).

Outputs:
- ``thermo_external_rho.{png,pdf}`` and ``thermostability_rho.csv``: Spearman's rho across models (vertical dataset bands).
- ``thermo_overview.{png,pdf}``: 2x5 horizontal forest overview across all models in the external frames.
- ``thermo_interval_types.{png,pdf}``: mutation-level vs. protein-level 95% bootstrap interval half-widths.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import leak_table, out_dir  # noqa: E402
from msr_analysis.data import Frame, load_frame  # noqa: E402
from msr_analysis.external_s1 import (  # noqa: E402
    EXT_ORDER,
    load_s1_mutation_boot,
    model_key_for_dataset,
    register_ext_styles,
)
from msr_analysis.registry import CACHE, LABEL, MODELS  # noqa: E402

OUT = out_dir('02_thermostability')
EXT10 = ['s571', 's783', 's2648', 's8754', 's461', 'ssym', 'q3421', 'k2369', 'ptmul', 'ptmuld']
THERMO_OVERVIEW_KEYS = [
    'msrp0:comb', 'msrp0:wt',
    'msrdev:comb', 'msrdev:wt',
    'msr:comb', 'msr:wt',
    'msrrel:comb', 'msrrel:wt',
    'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm',
    'esm3dg', 'rosetta', 'ifum_esmfold', 'ifum_wt', 'esmtherm', 'ddgemb', 'ddmut', 'geoddg',
    'prosst', 'pmpnn', 'msrrel@0:comb', 'esm3s', 'esm3m', 'esm3l',
]
DB_TRAINED = ['ddmut', 'ddgemb', 'geoddg']
THEMES_NOTE = (
    'Seed 1 only; 95% mutation-level bootstrap. Singles models on the 8 single-mutant sets; '
    'multi models on PTMUL/PTMUL-D (ThermoMPNN additive on PTMUL, -D on PTMUL-D; Rosetta Cartesian ddG throughout). '
    'Red ring = GeoDDG-3D documented training overlap.'
)


def _ensure_rosetta_frames(refresh=False):
    need_boot = refresh
    if not refresh:
        for d in EXT10:
            if 'rosetta' not in load_frame(d).preds:
                need_boot = True
                break
    if need_boot:
        # Refresh non-parent sets and child sets (which recursively refresh s669, k3822, ptmul).
        for d in ['s571', 's783', 's2648', 's8754', 'ssym', 'q3421', 's461', 'k2369', 'ptmuld']:
            load_frame(d, refresh=True)
    return need_boot


def main(refresh=False):
    register_ext_styles()
    need_boot = _ensure_rosetta_frames(refresh=refresh)
    boot = load_s1_mutation_boot(refresh=need_boot)
    P, reps, keys_used = boot['point'], boot['reps'], boot['keys']
    leak = leak_table()

    active_models = [m for m in EXT_ORDER if any((d, m) in P for d in EXT10)]

    rows = []
    for m in active_models:
        for d in EXT10:
            if (d, m) not in P:
                continue
            raw_k = keys_used[(d, m)]
            v = P[(d, m)]
            rows.append(dict(
                model=m, key=raw_k, dataset=d,
                rho=v[0], lo=v[1], hi=v[2],
                leak_flag=bool(leak.get((raw_k, d), False)),
            ))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'thermostability_rho.csv', index=False, float_format='%.4f')

    # ---- figure 1: standalone single-panel dot plot across EXT10 (vertical panels per dataset) ----------
    table = {(d, m): P[(d, m)][0] for d in EXT10 for m in active_models if (d, m) in P}
    cis = {(d, m): (P[(d, m)][1], P[(d, m)][2]) for d in EXT10 for m in active_models if (d, m) in P}
    min_lo = min(v[0] for v in cis.values() if np.isfinite(v[0]))
    ymin = min(0.10, float(np.floor((min_lo - 0.03) * 20) / 20))

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    spread = 0.78
    style.dot_panel(
        ax, EXT10, active_models, table, cis,
        highlight=['ESM-MSR'], ylabel='Spearman ρ',
        xlabels=[LABEL.get(d, d) for d in EXT10],
        ylim=(ymin, 0.98), spread=spread, size=6.0,
    )
    offs = np.linspace(-spread / 2, spread / 2, len(active_models))
    for i, d in enumerate(EXT10):
        for j, m in enumerate(active_models):
            if (d, m) in P and leak.get((keys_used[(d, m)], d), False):
                v = P[(d, m)][0]
                ax.plot(i + offs[j], v, marker='o', ms=10.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', zorder=6)

    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in active_models]
    hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', label='training overlap (GeoDDG-3D)'))
    fig.legend(
        handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.942), ncol=5, fontsize=8.0,
        frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3,
    )
    fig.suptitle(
        'External thermostability sets: Spearman ρ across all models (seed 1; whiskers: 95% mutation-level bootstrap)',
        x=0.012, ha='left', y=0.985, fontsize=10.5, fontweight='bold',
    )
    fig.text(0.012, 0.01, THEMES_NOTE, fontsize=8.0, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.84))
    style.save(fig, OUT / 'thermo_external_rho')

    # ---- figure 2: mutation-level vs protein-level bootstrap intervals (ESM-MSR p0, seed 1) -------------
    p0_frames = {}
    for d in EXT10:
        fr = load_frame(d)
        k = model_key_for_dataset('ESM-MSR', fr)
        p0_frames[d] = Frame(d, fr.spine, {'ESM-MSR': fr.preds[k][:1]})
    _, prot_reps = bootstrap_rho(p0_frames, ['ESM-MSR'], kind='protein', B=1000, seed=21, per_protein=False)
    wp, wm = [], []
    for d in EXT10:
        pl, ph = ci(prot_reps[d]['ESM-MSR'])
        ml, mh = P[(d, 'ESM-MSR')][1], P[(d, 'ESM-MSR')][2]
        wp.append((ph - pl) / 2.0)
        wm.append((mh - ml) / 2.0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(EXT10))
    ax.bar(x - 0.2, wm, 0.4, color='#9ecae1', label='resampling mutations')
    ax.bar(x + 0.2, wp, 0.4, color='#08519c', label='resampling proteins')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in EXT10], rotation=30, ha='right')
    ax.set_ylabel('95% interval half-width of ρ')
    ax.legend()
    ax.set_title('Intervals for ESM-MSR (p0, seed 1; WT head on singles sets / combined on PTMUL(-D))', loc='left')
    fig.tight_layout()
    style.save(fig, OUT / 'thermo_interval_types')

    # ---- figure 3: tsuboyama_overview-style 2x5 horizontal forest plot across EXT10 --------------------
    thermo_overview(leak, refresh=need_boot)


def _load_thermo_overview_boot(refresh=False):
    cache_path = CACHE / 'thermo_overview_s1_boot.pkl'
    if cache_path.exists() and not refresh:
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and set(cached.get('keys', [])) == set(THERMO_OVERVIEW_KEYS):
            return cached

    s1_frames = {}
    meta = {}
    for d in EXT10:
        fr = load_frame(d)
        sp = fr.spine
        u = sp['use'].to_numpy()
        meta[d] = (int(u.sum()), int(sp.loc[u, 'cluster'].nunique()))
        s1_frames[d] = Frame(d, sp, {k: fr.preds[k][:1] for k in THERMO_OVERVIEW_KEYS if k in fr.preds})

    pt_p, rep_p = bootstrap_rho(s1_frames, THERMO_OVERVIEW_KEYS, kind='protein', B=1000, seed=11, per_protein=False)
    pt_m, rep_m = bootstrap_rho(s1_frames, THERMO_OVERVIEW_KEYS, kind='mutation', B=1000, seed=11, per_protein=False)

    P_prot, P_mut = {}, {}
    for d in EXT10:
        for k in THERMO_OVERVIEW_KEYS:
            if k in pt_p[d]:
                lo_p, hi_p = ci(rep_p[d][k])
                P_prot[(d, k)] = (float(pt_p[d][k][0]), float(lo_p), float(hi_p))
            if k in pt_m[d]:
                lo_m, hi_m = ci(rep_m[d][k])
                P_mut[(d, k)] = (float(pt_m[d][k][0]), float(lo_m), float(hi_m))

    payload = dict(keys=list(THERMO_OVERVIEW_KEYS), meta=meta, P_prot=P_prot, P_mut=P_mut)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload


def thermo_overview(leak, refresh=False):
    ov = _load_thermo_overview_boot(refresh=refresh)
    P_prot, P_mut, meta = ov['P_prot'], ov['P_mut'], ov['meta']

    label_overrides = {
        'msrdev:comb': 'ESM-MSR (devel) combined',
        'msrdev:wt':   'ESM-MSR (devel) WT head',
        'geoddg':      'GeoDDG-3D*',
        'ddgemb':      'DDGemb*',
        'ddmut':       'DDMut*',
    }
    saved_labels = {k: MODELS[k]['label'] for k in label_overrides}
    for k, lab in label_overrides.items():
        MODELS[k]['label'] = lab

    try:
        mean_rho = {k: float(np.mean([P_prot[(d, k)][0] for d in EXT10])) for k in THERMO_OVERVIEW_KEYS}
        db_keys = sorted([k for k in THERMO_OVERVIEW_KEYS if k in DB_TRAINED], key=lambda k: mean_rho[k])
        held_keys = sorted([k for k in THERMO_OVERVIEW_KEYS if k not in DB_TRAINED], key=lambda k: mean_rho[k])
        keys = db_keys + held_keys

        rows = [EXT10[:5], EXT10[5:]]
        fig, axes = plt.subplots(2, 5, figsize=(17.8, 2 * (0.28 * len(keys) + 1.2)), sharey=True)
        for r_idx, row_ds in enumerate(rows):
            for c_idx, ds in enumerate(row_ds):
                ax = axes[r_idx, c_idx]
                vals = [P_prot[(ds, k)] for k in keys if (ds, k) in P_prot]
                mut_vals = [P_mut[(ds, k)] for k in keys if (ds, k) in P_mut]
                min_lo = min(min(lo for _v, lo, _hi in vals), min(lo for _v, lo, _hi in mut_vals))
                max_hi = max(max(hi for _v, _lo, hi in vals), max(hi for _v, _lo, hi in mut_vals))
                x_lo = float(np.floor((min_lo - 0.03) * 20) / 20)
                span = max(0.30, max_hi - x_lo)
                x_hi = max_hi + 0.24 * span

                for i, k in enumerate(keys):
                    v, lo, hi = P_prot[(ds, k)]
                    m = P_mut.get((ds, k))
                    kw = style.marker_kwargs(k, size=5.6)
                    ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.3, alpha=0.45, solid_capstyle='butt')
                    if m:
                        ax.plot([m[1], m[2]], [i, i], color=kw['color'], lw=0.8)
                    ax.plot(v, i, **kw)
                    if leak.get((k, ds), False):
                        ax.plot(v, i, marker='o', ms=9.5, mfc='none', mec='#c0392b', mew=0.95, ls='none')
                    r_edge = max(hi, m[2] if m else hi)
                    ax.text(r_edge + 0.022 * span, i, f'{v:.3f}', va='center', fontsize=6.9, color='#444')

                ax.axhspan(-0.5, len(db_keys) - 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
                n_use, n_prot = meta[ds]
                ax.set_title(f'{LABEL[ds]} (n = {n_use:,}, {n_prot} prot.)', loc='left', fontsize=9.3)
                ax.grid(axis='x', color=style.GRID, lw=0.6)
                ax.set_xlim(x_lo, x_hi)
                if r_idx == 1:
                    ax.set_xlabel('Spearman ρ', fontsize=9.0)

            ax_l = axes[r_idx, 0]
            ax_l.set_yticks(range(len(keys)))
            ax_l.set_yticklabels([MODELS[k]['label'] for k in keys], fontsize=8.0)
            ax_l.set_ylim(-0.6, len(keys) - 0.4)
            for t, k in zip(ax_l.get_yticklabels(), keys):
                t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555'}[MODELS[k]['group']])
                if MODELS[k]['group'] == 'msr':
                    t.set_fontweight('bold')

        fig.suptitle(
            'External thermostability sets (seed 1): thick bar = resampling proteins; thin line = resampling mutations; red ring = documented training overlap\n'
            '* trained on ProThermDB / ThermoMutDB / S2648 / S8754: not held out (grey rows, at the bottom)',
            fontsize=9.2, y=1.01,
        )
        fig.tight_layout()
        style.save(fig, OUT / 'thermo_overview')
    finally:
        for k, lab in saved_labels.items():
            MODELS[k]['label'] = lab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='Recompute external frames and seed-1 mutation bootstrap cache')
    args = ap.parse_args()
    main(refresh=args.refresh)
