#!/usr/bin/env python
"""Summary figures: one panel that puts the benchmarks with many assays side by side.

``pg_categories``   two panels.  Left: the ProteinGym average (mean of the five category means: Activity 43, Binding 13, Expression 18, Org. fitness 73, Stability 40/66), the 458 low-homology Domainome domains, then MGnify and
                    Rocklin Designs Inferred DDGs (n WTs): mean Spearman rho per column.  Right: the ProteinGym assays by property, ESM-MSR only: delta rho versus sigma = 0 at sigma 0.5 and 1 (paired box / dot plot,
                    ``proteingym/p0_sigma/03_fig_property_summary.py``).  Left of every cluster, side by side: ESM-MSR p0 at LoRA scale sigma = 1 (deployed readout: combined on multi-mutant assays, WT head elsewhere), sigma = 0.5 (WT head; combined not run)
                    and ESM3-small-open (sigma = 0, WT head); then the other models (seed 1; each with its multi-mutant version on the multi-mutant assays).  ProteinGym and Domainome come from ``results/05_proteingym/pg_summary.csv`` and ``pg_sigma_summary.csv``
                    (``05_proteingym.py``); the two inferred-ddG columns are ESM-MSR p0 at sigma = 1 alone (the only model and scale run on them), the mean per-WT rho of the combined prediction over the WTs with at
                    least 10 measured variants, three seeds, from ``results/18_mgnify/mgnify_summary.csv`` (``18_mgnify.py``; MGnify without the scaffolds that have a strong homolog or the same protein in the capped training set).
``thermostability_sets``   the external thermostability sets (S571 and PTMUL-D left out) for the nine unified models, seed 1, 95% mutation-level bootstrap, as ``02_thermostability/thermo_external_rho``,
                    with the sets ordered left to right by the performance of ESM-MSR p0 (seed mean of FINAL_p0 s1-s3 from ``results/98_ablation/cells_long.csv``, the seed-1 value of
                    ``results/02_thermostability/thermostability_rho.csv`` when that file is missing), highest first.  Hollow markers (thick edge): singles models (ESM-MSR WT head, MutateEverything st1, ThermoMPNN,
                    SPURS single); filled: models trained with doubles (ESM-MSR combined, MutateEverything st2, ThermoMPNN-D, SPURS multi) and the models with one version.  Writes ``thermostability_sets.csv``.
                    The p-values of the figure are in ``01b_thermostability_pvalues.py``.

Needs ``05_proteingym.py`` and ``18_mgnify.py`` first (without the MGnify table the two inferred-ddG columns are left out) and ``02_thermostability.py`` for the second figure.

    python analysis/01_summary.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.external_s1 import EXT_ORDER, register_ext_styles  # noqa: E402
from msr_analysis.registry import LABEL, MODELS, RESULTS  # noqa: E402

OUT = out_dir('01_summary')
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']
SIGMA_ORDER = ['ESM-MSR σ=1', 'ESM-MSR σ=0.5', 'ESM3-small-open', 'MutateEverything', 'ThermoMPNN', 'SPURS', 'ProteinMPNN', 'ProSST', 'GeoDDG-3D', 'Rosetta']
PG_LABEL = {
    'PG avg': 'ProteinGym\naverage* (5 categories)',
    'Activity': 'ProteinGym\nActivity (43)',
    'Binding': 'ProteinGym\nBinding (13)',
    'Expression': 'ProteinGym\nExpression (18)',
    'OrganismalFitness': 'ProteinGym\nOrg. fitness (73)',
    'Stability': 'ProteinGym\nStability* (40/66)',
    'Domainome': 'Domainome*\n(458/522)',
}
PG_COLS = CATS + ['PG avg', 'DMS mean']


def inferred_column(dataset, stratum):
    """(value, lo, hi, n WTs) of ESM-MSR p0 on an inferred-ddG set of ``18_mgnify.py`` (MGnify scaffolds, Rocklin designs): the mean per-WT rho of the combined prediction over the WTs with at least 10
    measured variants, seeds 1-3, scaffolds / designs resampled; None when the set was not scored."""
    f = RESULTS / '18_mgnify' / 'mgnify_summary.csv'
    if not f.exists():
        return None
    S = pd.read_csv(f)
    t = S[(S.dataset == dataset) & (S.stratum == stratum) & (S['subset'] == 'overall') & (S.unit == 'per scaffold') & (S['head'] == 'comb') & (S.form == 'latent')]
    if not len(t):
        return None
    r = t.iloc[0]
    return float(r.rho), float(r.lo), float(r.hi), int(r.n_scaffolds)


def _property_module():
    """``proteingym/p0_sigma/03_fig_property_summary.py`` (the per-property paired box / dot plot of rho against sigma) as a module, for its ``draw_paired``."""
    d = Path(__file__).resolve().parent / 'proteingym' / 'p0_sigma'
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('fig_property_summary', d / '03_fig_property_summary.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pg_categories():
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    k05, n_mm = (int(x) for x in (RESULTS / '05_proteingym' / 'pg_sigma_note.txt').read_text().split()[::2][:2])      # "k of n": multi-mutant assays with a sigma-0.5 combined prediction
    MODELS['ESM-MSR σ=1'] = dict(MODELS['ESM-MSR'], label='ESM-MSR σ = 1')
    MODELS['ESM-MSR σ=0.5'] = dict(MODELS['ESM-MSR'], label='ESM-MSR σ = 0.5' + ('' if k05 == n_mm else ' (combined pending)'), color='#4d4d4d')
    MODELS['ESM3-small-open'] = dict(MODELS['ESM3-small-open'], label='ESM3-small-open (σ = 0)')
    R = pd.concat([pd.read_csv(RESULTS / '05_proteingym' / 'pg_summary.csv'), pd.read_csv(RESULTS / '05_proteingym' / 'pg_sigma_summary.csv')], ignore_index=True)
    note = (RESULTS / '05_proteingym' / 'pg_categories_note.txt').read_text().strip()
    table = {(r.column, r.model): r.value for r in R.itertuples()}
    cis = {(r.column, r.model): (r.lo, r.hi) for r in R.itertuples()}
    cols = ['PG avg', 'Domainome']
    labels = [PG_LABEL[c] for c in cols]
    extra = [(name, inferred_column(ds, st)) for name, ds, st in (('MGnify Inferred\nDDGs', 'MGnify', 'lowid'), ('Rocklin Designs\nInferred DDGs', 'Rocklin designs', 'all'))]
    extra = [(n, v) for n, v in extra if v]
    groups = cols + [f'x{i}' for i in range(len(extra))]
    labels += [f'{n}\n({v[3]:,} WTs)' for n, v in extra]

    fps = _property_module()
    t = fps.data()                                                      # ProteinGym assays without training-data overlap: per-assay rho of the deployed ESM-MSR readout at sigma 0, 0.5, 1
    Rp = fps.pvalues(t, 'property_delta_pvalues.csv')

    fig = plt.figure(figsize=(15.6, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.43, 0.57], wspace=0.15)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    style.dot_panel(ax, groups, SIGMA_ORDER, table, cis, highlight=['ESM-MSR σ=1', 'ESM-MSR σ=0.5'], ylabel=r'Spearman $\rho$ (mean over assays / domains / WTs)', xlabels=labels, rot=0, size=6.2, spread=0.84, ylim=(0.12, 1.0))
    for x in range(len(cols) - 1, len(groups) - 1):              # dashed rules after ProteinGym, after Domainome, between the two inferred-ddG sets
        ax.axvline(x + 0.5, color='#999999', lw=0.95, ls='--', zorder=2)
    kw = style.marker_kwargs('ESM-MSR σ=1', size=6.2 * 1.35)
    for i, (n, v) in enumerate(extra):
        x = len(cols) + i
        ax.errorbar([x], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=4)
        ax.plot([x], [v[0]], zorder=5, **kw)
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in SIGMA_ORDER]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.0, handletextpad=0.3, borderpad=0.4, labelspacing=0.35)
    fps.draw_paired(ax2, t, Rp, compact=True)
    fig.suptitle('ESM-MSR against other models, and the effect of the LoRA scale σ on ProteinGym', x=0.012, ha='left', y=0.985, fontsize=10.5, fontweight='bold')

    mm_note = (f'ESM-MSR σ = 1, σ = 0.5 and ESM3-small-open (σ = 0, the same architecture with the adapters off) are read as deployed: the combined prediction on the {n_mm} ProteinGym assays with multi-mutants, the WT head on the others '
               f'(Domainome has no multi-mutants). The σ = 0.5 combined prediction exists for {k05} of the {n_mm} multi-mutant assays' + (' (the others read the WT head until the queued runs finish)' if k05 < n_mm else '')
               + '; the σ = 0 combined prediction is the March ESM-MSR-small checkpoint\'s σ = 0 run (adapters off, unmasked; its calibration heads), the WT head that of the released σ = 0 run. '
               'Right panel: the same readout per ProteinGym assay (23 assays with training-set overlap left out; one dot per assay, faint lines join an assay; box = IQR, thin line = median, dark line = mean with its value; p under the ticks: '
               'two-sided bootstrap over assays that the mean Δρ is 0, Holm-adjusted over the ten category × σ comparisons).')
    note += '\n' + mm_note
    if extra:
        note += ('\nMGnify and Rocklin Designs Inferred DDGs: ESM-MSR p0 at σ = 1 only (the other models and scales were not run on them); combined prediction, mean over WTs of the per-WT rho of the WT\'s measured variants, WTs with at least 10 '
                 'variants (n WTs), mean of seeds 1-3, whiskers resample WTs and seeds. MGnify without the scaffolds that have a strong homolog (H) or the same protein in the capped training set (MMseqs2).')
    kwt = dict(fontsize=7.4, color='#444444')
    tt = fig.text(0.012, 0.008, style.wrap_to_px(fig, note, 0.975 * fig.bbox.width, **kwt), va='bottom', ha='left', **kwt)
    foot = tt.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.subplots_adjust(left=0.05, right=0.995, top=0.835, bottom=foot + 0.115)
    for a, txt in ((ax, 'Spearman ρ of ESM-MSR and the other models'), (ax2, 'ESM-MSR: Δρ versus σ = 0 per assay, by property')):
        pos = a.get_position()
        fig.text(pos.x0, 0.895, txt, ha='left', va='bottom', fontsize=10, fontweight='bold')
    fig.canvas.draw()
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in pg_categories', bad)
    style.save(fig, OUT / 'pg_categories')
    print('wrote', OUT / 'pg_categories', '|', [(n.replace(chr(10), ' '), v) for n, v in extra])


EXCLUDE_THERMO = {'s571', 'ptmuld'}
# (singles model, model trained with doubles) of the models that have both; the others (ESM3-small-open, ProteinMPNN, ProSST, GeoDDG-3D, Rosetta) have one version
VERSIONS = {'ESM-MSR': ('msrp0:wt', 'msrp0:comb'), 'MutateEverything': ('me1', 'me2'), 'ThermoMPNN': ('tmpnn', 'tmpnnd'), 'SPURS': ('spurs', 'spursm')}
HOLLOW_MEW = 2.0            # edge width of the hollow (singles) markers; the default is 1.2
HOLLOW_SIZE = 1.2           # their size relative to a filled marker, so that the hole stays visible
THERMO_NOTE = ('Seed 1 only; 95% mutation-level bootstrap. Singles models on the single-mutant sets, models trained with doubles on PTMUL (ThermoMPNN additive; Rosetta Cartesian ddG throughout). '
               'Hollow markers: singles models (ESM-MSR WT head, MutateEverything st1, ThermoMPNN, SPURS single); filled: models trained with doubles (ESM-MSR combined, MutateEverything st2, SPURS multi) and the models '
               'with one version. SPURS is the retrain with the trained adapter. Sets ordered by the performance of ESM-MSR p0 (mean of seeds 1-3), highest first. Red ring = GeoDDG-3D documented training overlap.')


def msr_order(sets, S):
    """``sets`` sorted by the ESM-MSR p0 rho, highest first: the seed mean of FINAL_p0 (``98_ablation/cells_long.csv``, deployed readout) or, without it, the seed-1 value of ``S``."""
    seed1 = S[S.model == 'ESM-MSR'].set_index('dataset')['rho']
    f = RESULTS / '98_ablation' / 'cells_long.csv'
    mean = {}
    if f.exists():
        c = pd.read_csv(f)
        c = c[(c.family == 'FINAL_p0') & (c['head'] == 'primary') & c.dataset.isin(sets)]
        mean = c.groupby('dataset').rho.mean().to_dict() if c.seed.nunique() >= 2 else {}
    val = {d: mean.get(d, seed1[d]) for d in sets}
    return sorted(sets, key=lambda d: -val[d]), val


def thermostability_sets():
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    S = pd.read_csv(RESULTS / '02_thermostability' / 'thermostability_rho.csv')
    S = S[~S.dataset.isin(EXCLUDE_THERMO)]
    sets, val = msr_order(sorted(S.dataset.unique()), S)
    models = [m for m in EXT_ORDER if (S.model == m).any()]
    kinds = {m: dict(zip(VERSIONS[m], ('singles', 'doubles'))) for m in VERSIONS}
    S = S.assign(version=[kinds.get(r.model, {}).get(r.key, '') for r in S.itertuples()])
    ymin = min(0.10, float(np.floor((np.nanmin(S.lo) - 0.03) * 20) / 20))

    def kwargs(m, version, size):
        kw = style.marker_kwargs(m, size=size)
        if version == 'singles':
            kw.update(mfc='none', mew=HOLLOW_MEW, ms=kw['ms'] * HOLLOW_SIZE)
        else:
            kw['mfc'] = kw['color']
        return kw

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    spread = 0.78
    style.dot_panel(ax, sets, models, {}, None, ylabel=r'Spearman $\rho$ (pooled)', xlabels=[LABEL.get(d, d) for d in sets], rot=0, ylim=(ymin, 0.98), spread=spread, size=6.0)
    offs = np.linspace(-spread / 2, spread / 2, len(models))
    for i, d in enumerate(sets):
        for j, m in enumerate(models):
            x = i + offs[j]
            hl = m == 'ESM-MSR'
            r = S[(S.dataset == d) & (S.model == m)].iloc[0]
            kw = kwargs(m, r.version, 6.0 * (1.35 if hl else 1.0))
            ax.errorbar([x], [r.rho], yerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=(5 if hl else 3) - 1)
            ax.plot(x, r.rho, zorder=5 if hl else 3, **kw)
            if r.leak_flag:
                ax.plot(x, r.rho, marker='o', ms=10.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', zorder=6)
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in models]
    hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', label='training overlap (GeoDDG-3D)'))
    grey = '#555555'
    hs += [Line2D([0], [0], marker='o', ms=6.8, mfc=grey, mec=grey, ls='none', label='filled: trained with doubles'),
           Line2D([0], [0], marker='o', ms=6.8 * HOLLOW_SIZE, mfc='none', mec=grey, mew=HOLLOW_MEW, ls='none', label='hollow: singles model')]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=4, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3)
    fig.suptitle('External thermostability sets: Spearman ρ across models (seed 1; whiskers: 95% mutation-level bootstrap)', x=0.012, ha='left', y=0.985, fontsize=10.5, fontweight='bold')
    kw = dict(fontsize=8.0, color='#444444')
    t = fig.text(0.012, 0.01, style.wrap_to_px(fig, THERMO_NOTE, 0.96 * fig.bbox.width, **kw), va='bottom', ha='left', **kw)
    foot = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0.0, foot + 0.02, 1.0, 0.86))
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in thermostability_sets', bad)
    style.save(fig, OUT / 'thermostability_sets')
    order = {d: i + 1 for i, d in enumerate(sets)}
    D = S.assign(order=S.dataset.map(order), msr_p0_seed_mean=S.dataset.map(val)).sort_values(['order', 'model'])
    D.to_csv(OUT / 'thermostability_sets.csv', index=False, float_format='%.4f')
    print('thermostability_sets, left to right:', ', '.join(f'{LABEL.get(d, d)} {val[d]:.3f}' for d in sets))


def main():
    pg_categories()
    thermostability_sets()


if __name__ == '__main__':
    main()
