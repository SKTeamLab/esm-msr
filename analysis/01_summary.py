#!/usr/bin/env python
"""Summary figures: one panel that puts the benchmarks with many assays side by side.

``pg_categories``   two panels.  Left: the ProteinGym average (mean of the five category means: Activity 43, Binding 13, Expression 18, Org. fitness 73, Stability 40/66), the 458 low-homology Domainome domains, then MGnify and
                    Rocklin Designs Inferred DDGs (n WTs): mean Spearman rho per column.  Right: the ProteinGym assays by property, ESM-MSR only: delta rho versus sigma = 0 at sigma 0.5 and 1 (paired box / dot plot,
                    ``proteingym/p0_sigma/03_fig_property_summary.py``).  Left of every cluster, side by side: ESM-MSR p0 at LoRA scale sigma = 1 (deployed readout: combined on multi-mutant assays, WT head elsewhere), sigma = 0.5 (combined on the multi-mutant assays that have it, WT head otherwise)
                    and ESM3-small-open (sigma = 0, WT head); then the other models (seed 1; each with its multi-mutant version on the multi-mutant assays).  ProteinGym and Domainome come from ``results/05_proteingym/pg_summary.csv`` and ``pg_sigma_summary.csv``
                    (``05_proteingym.py``); the two inferred-ddG columns are the mean per-WT rho (WTs with at least 10 measured variants; ``results/18_mgnify/mgnify_models_per_scaffold.csv`` from ``18b_mgnify_models.py``) of ESM-MSR p0 at sigma = 1,
                    SPURS (-multi), ThermoMPNN (-D), ProteinMPNN, ProSST and, on MGnify, ESM3-small-open (WT head), each on the scaffolds it has scored completely (models with under 80% of a set's scaffolds are left out; MGnify without the
                    scaffolds that have a strong homolog or the same protein in the capped training set).
``thermostability_sets``   the external thermostability sets (S571 and PTMUL-D left out) for the nine unified models, seed 1, 95% mutation-level bootstrap, as ``02_thermostability/thermo_external_rho``,
                    with the sets ordered left to right by the performance of ESM-MSR p0 (seed mean of FINAL_p0 s1-s3 from ``results/98_ablation/cells_long.csv``, the seed-1 value of
                    ``results/02_thermostability/thermostability_rho.csv`` when that file is missing), highest first.  Hollow markers (thick edge): singles models (ESM-MSR WT head, MutateEverything st1, ThermoMPNN,
                    SPURS single); filled: models trained with doubles (ESM-MSR combined, MutateEverything st2, ThermoMPNN-D, SPURS multi) and the models with one version.  Writes ``thermostability_sets.csv``.
                    The p-values of the figure are in ``01b_thermostability_pvalues.py``.

Needs ``05_proteingym.py``, ``18_mgnify.py`` and ``18b_mgnify_models.py`` first (without the MGnify table the two inferred-ddG columns are left out) and ``02_thermostability.py`` for the second figure.

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
from msr_analysis.registry import LABEL, MODELS, REFERENCE, RESULTS  # noqa: E402

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
HOLLOW_DOMAINOME = ['ESM-MSR σ=1', 'ESM-MSR σ=0.5', 'MutateEverything', 'ThermoMPNN', 'SPURS']      # models with a singles and a doubles version: Domainome is read with the singles one


INFERRED = {'MGnify': 'MGnify Inferred\nDDGs', 'Rocklin designs': 'Rocklin Designs\nInferred DDGs'}
INFERRED_MODEL = {'ESM-MSR': 'ESM-MSR σ=1', 'SPURS': 'SPURS', 'ThermoMPNN': 'ThermoMPNN', 'ProteinMPNN': 'ProteinMPNN', 'ProSST': 'ProSST', 'ESM3-small-open': 'ESM3-small-open'}
MIN_COVERAGE = 0.8                                  # a model enters an inferred-ddG set when it has scored at least this share of the set's scaffolds completely


def inferred_models(pooled=False):
    """{dataset: (n WTs, or n variants when pooled, {model key of SIGMA_ORDER: (rho, lo, hi, n scaffolds or variants, coverage)})} from ``18b_mgnify_models.py``'s table; {} when it was not run."""
    f = RESULTS / '18_mgnify' / ('mgnify_models_pooled.csv' if pooled else 'mgnify_models_per_scaffold.csv')
    if not f.exists():
        return {}
    out = {}
    for r in pd.read_csv(f).itertuples():
        if r.model in INFERRED_MODEL and r.coverage >= MIN_COVERAGE:
            out.setdefault(r.dataset, (int(r.n_set), {}))[1][INFERRED_MODEL[r.model]] = (float(r.rho), float(r.lo), float(r.hi), int(r.n_variants if pooled else r.n_scaffolds), float(r.coverage))
    return out


def _property_module():
    """``proteingym/p0_sigma/03_fig_property_summary.py`` (the per-property paired box / dot plot of rho against sigma) as a module, for its ``draw_paired``."""
    d = Path(__file__).resolve().parent / 'proteingym' / 'p0_sigma'
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('fig_property_summary', d / '03_fig_property_summary.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pg_prepare(pooled=True):
    """Everything the left (dots) and right (paired boxes) ProteinGym panels need: the table / intervals per (group, model), labels, the hollow (singles-model) points, the property data and p-values, the footnote."""
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    k05, n_mm = (int(x) for x in (RESULTS / '05_proteingym' / 'pg_sigma_note.txt').read_text().split()[::2][:2])      # "k of n": multi-mutant assays with a sigma-0.5 combined prediction
    MODELS['ESM-MSR σ=1'] = dict(MODELS['ESM-MSR'], label='ESM-MSR σ = 1')
    MODELS['ESM-MSR σ=0.5'] = dict(MODELS['ESM-MSR'], label='ESM-MSR σ = 0.5' + ('' if k05 == n_mm else f' (combined on {k05} of {n_mm})'), color='#4d4d4d')
    MODELS['ESM3-small-open'] = dict(MODELS['ESM3-small-open'], label='ESM3-small-open (σ = 0)')
    R = pd.concat([pd.read_csv(RESULTS / '05_proteingym' / 'pg_summary_tsu.csv'), pd.read_csv(RESULTS / '05_proteingym' / 'pg_sigma_summary_tsu.csv')], ignore_index=True)
    fl = pd.read_csv(REFERENCE / 'tsuboyama_rule_flags.csv')
    n_fl = {d: (int(g.homologous.sum()), len(g)) for d, g in fl.groupby('dataset')}
    tsu_pg = set(fl.loc[(fl['dataset'] == 'proteingym') & fl.homologous, 'item_id'])
    n_pg = int(R[(R.model == 'ESM-MSR σ=1') & (R.column == 'PG avg')].n_dms.iloc[0])
    n_dom = int(R[(R.model == 'ESM-MSR σ=1') & (R.column == 'Domainome')].n_dms.iloc[0])
    note = ('Held-out = not a homolog of the training set under the rule that built the Homology-Restricted MegaScale split (MMseqs2 or Foldseek E ≤ 1e-3, local sequence identity > 0.40, or TM-score > 0.70 to a capped-train scaffold); left out: '
            + (RESULTS / '05_proteingym' / 'pg_tsu_note.txt').read_text().strip() + f' ({n_pg} ProteinGym assays and {n_dom} domains remain); MGnify {n_fl["mgnify"][0]:,} of {n_fl["mgnify"][1]:,} scaffolds and Rocklin designs {n_fl["rocklin_denovo"][0]} of {n_fl["rocklin_denovo"][1]:,} '
            '(sequence evidence only: no structure search was run on them).\n'
            'Seed 1 only. Singles models on singles assays / Domainome; multi on PGym multi-mutants (ME st1/st2; ThermoMPNN additive / -D on doubles; SPURS single/multi). Rosetta: PGym singles only (no Domainome).')
    table = {(r.column, r.model): r.value for r in R.itertuples()}
    cis = {(r.column, r.model): (r.lo, r.hi) for r in R.itertuples()}
    cols = ['PG avg', 'Domainome']
    labels = [f'ProteinGym avg.\n({n_pg} held-out assays)', f'Domainome avg.\n({n_dom} held-out domains)']
    inf = inferred_models(pooled)
    extra = [(INFERRED[d], inf[d][0], inf[d][1]) for d in INFERRED if d in inf]
    groups = cols + [f'x{i}' for i in range(len(extra))]
    labels += [f'{n.split(chr(10))[0].split()[0]} ' + ('Pooled' if pooled else 'Avg.') + f'\n({round(k / 1000):,}k variants)' if pooled else f'{n.split(chr(10))[0].split()[0]} Avg.\n({k:,} held-out WTs)' for n, k, _ in extra]
    for i, (_, _, vals) in enumerate(extra):
        for m, v in vals.items():
            table[(f'x{i}', m)], cis[(f'x{i}', m)] = v[0], (v[1], v[2])
    hollow = {('Domainome', m) for m in HOLLOW_DOMAINOME}          # Domainome holds singles only: the models with two versions are drawn as their singles version, hollow, as in the thermostability sets

    fps = _property_module()
    t = fps.data()                                                      # ProteinGym assays without training-data overlap: per-assay rho of the deployed ESM-MSR readout at sigma 0, 0.5, 1
    t = t[~t.assay.isin(tsu_pg)].reset_index(drop=True)                 # and without the homologs of the rule above
    Rp = fps.pvalues(t, 'property_delta_pvalues_tsu.csv')

    mm_note = (f'ESM-MSR σ = 1, σ = 0.5 and ESM3-small-open (σ = 0, the same architecture with the adapters off) are read as deployed: the combined prediction on the {n_mm} ProteinGym assays with multi-mutants, the WT head on the others '
               f'(Domainome has no multi-mutants, so its models with two versions are the hollow singles versions). The σ = 0.5 combined prediction exists for {k05} of the {n_mm} multi-mutant assays' + (' (the others read the WT head until the queued runs finish)' if k05 < n_mm else '')
               + '; the σ = 0 combined prediction is the March ESM-MSR-small checkpoint\'s σ = 0 run (adapters off, unmasked; its calibration heads), the WT head that of the released σ = 0 run. '
               'Right panel: the same readout per ProteinGym assay (assays homologous to the training set left out, as on the left; one dot per assay, faint lines join an assay; box = IQR, thick line = median with its value, thin line = mean; p under the ticks: '
               'two-sided bootstrap over assays that the mean Δρ is 0, Holm-adjusted over the ten category × σ comparisons).')
    note += '\n' + mm_note
    if extra:
        part = [f'{m} on {n.replace(chr(10), " ")} ({v[3]:,} of {k:,} {"variants" if pooled else "scaffolds"})' for n, k, vals in extra for m, v in vals.items() if v[4] < 0.999]
        how = ('one Spearman rho over all measured variants of the set (seeds 1-3 averaged where a model has them), whiskers resample scaffolds; ESM-MSR is read on the observed scale, h(dG_wt + latent) - dG_wt (the Tobit link, the checkpoints\' constants), '
               'the other models raw' if pooled else
               'mean over WTs of the per-WT rho of the WT\'s measured variants, WTs with at least 10 variants (a within-WT rank is unaffected by the Tobit link), seeds 1-3 averaged where a model has them, whiskers resample WTs')
        note += ('\nMGnify and Rocklin Designs Inferred DDGs: ' + how + '. Multi-mutant versions are read (ESM-MSR combined head, SPURS-multi, ThermoMPNN-D; ProteinMPNN and ProSST score every variant directly); ESM3-small-open is the WT head '
                 '(its combined prediction was not run on these sets). '
                 + ('Models scored on part of a set only: ' + '; '.join(part) + '. ' if part else '') + 'IFUM, ESMtherm and DDGemb are left to the supplement (they do not beat the other transfer models and saw part of the test sets in training); GeoDDG-3D, ESM3dG and Rosetta are not in these two columns yet (running or not collected).')
    return dict(pooled=pooled, groups=groups, cols=cols, labels=labels, table=table, cis=cis, hollow=hollow, extra=extra, fps=fps, t=t, Rp=Rp, note=note)


def style_kwargs(m, singles, size):
    """Marker of model ``m``: hollow (thick edge, larger) for a singles model, filled otherwise: the same in the thermostability sets and the ProteinGym / Domainome / MGnify panel."""
    kw = style.marker_kwargs(m, size=size)
    if singles:
        kw.update(mfc='none', mew=HOLLOW_MEW, ms=kw['ms'] * HOLLOW_SIZE)
    else:
        kw['mfc'] = kw['color']
    return kw


def version_handles():
    grey = '#555555'
    return [Line2D([0], [0], marker='o', ms=6.8, mfc=grey, mec=grey, ls='none', label='filled: trained with doubles'),
            Line2D([0], [0], marker='o', ms=6.8 * HOLLOW_SIZE, mfc='none', mec=grey, mew=HOLLOW_MEW, ls='none', label='hollow: singles model')]


def draw_pg_left(ax, pg, ylim=(0.04, 1.0)):
    """Left panel: one cluster per column (ProteinGym average, Domainome, MGnify, Rocklin), one dot per model; the singles versions on Domainome are hollow."""
    groups, hollow = pg['groups'], pg['hollow']
    table = {k: v for k, v in pg['table'].items() if k not in hollow}
    style.dot_panel(ax, groups, SIGMA_ORDER, table, {k: v for k, v in pg['cis'].items() if k not in hollow}, highlight=['ESM-MSR σ=1', 'ESM-MSR σ=0.5'],
                    ylabel=r'Spearman $\rho$' if pg['pooled'] else r'Spearman $\rho$ (mean over assays / domains / WTs)', xlabels=pg['labels'], rot=0, size=6.2, spread=0.84, ylim=ylim)
    offs = np.linspace(-0.42, 0.42, len(SIGMA_ORDER))
    for (g, m) in sorted(hollow, key=lambda k: SIGMA_ORDER.index(k[1])):
        if (g, m) not in pg['table']:
            continue
        x, v, (lo, hi) = groups.index(g) + offs[SIGMA_ORDER.index(m)], pg['table'][(g, m)], pg['cis'][(g, m)]
        hl = m in ('ESM-MSR σ=1', 'ESM-MSR σ=0.5')
        kw = style_kwargs(m, True, 6.2 * (1.35 if hl else 1.0))
        ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=(5 if hl else 3) - 1)
        ax.plot([x], [v], zorder=5 if hl else 3, **kw)
    if len(groups) > len(pg['cols']):                                    # one dashed rule, where the figure turns from the ProteinGym / Domainome assay averages to the stability sets (MGnify, Rocklin)
        ax.axvline(len(pg['cols']) - 0.5, color='#999999', lw=0.95, ls='--', zorder=2)
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in SIGMA_ORDER] + version_handles()
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.0, handletextpad=0.3, borderpad=0.4, labelspacing=0.35)


def pg_categories(pooled=True):
    """The ProteinGym summary figure (panels b and c alone); ``pooled`` (the default, file ``pg_categories``): the MGnify and Rocklin columns are one Spearman rho over all variants (ESM-MSR Tobit-linked); otherwise
    (``pg_categories_per_scaffold``) the mean per-scaffold rho."""
    pg = pg_prepare(pooled)
    fig = plt.figure(figsize=(15.6, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.43, 0.57], wspace=0.15)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    draw_pg_left(ax, pg)
    pg['fps'].draw_paired(ax2, pg['t'], pg['Rp'], compact=True)
    kwt = dict(fontsize=7.4, color='#444444')
    tt = fig.text(0.012, 0.008, style.wrap_to_px(fig, pg['note'], 0.975 * fig.bbox.width, **kwt), va='bottom', ha='left', **kwt)
    foot = tt.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.subplots_adjust(left=0.05, right=0.995, top=0.89, bottom=foot + 0.115)
    for a, letter in ((ax, 'b'), (ax2, 'c')):                    # panel letters, Nature style: bold lowercase at the top left (a is the thermostability figure)
        pos = a.get_position()
        fig.text(pos.x0 - (0.047 if letter == 'b' else 0.052), 0.985, letter, ha='left', va='top', fontsize=14, fontweight='bold')
    fig.canvas.draw()
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in pg_categories', bad)
    stem = 'pg_categories' if pooled else 'pg_categories_per_scaffold'
    style.save(fig, OUT / stem)
    print('wrote', OUT / stem, '|', [(n.replace(chr(10), ' '), k, sorted(vals)) for n, k, vals in pg['extra']])


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


def thermo_data():
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    S = pd.read_csv(RESULTS / '02_thermostability' / 'thermostability_rho.csv')
    S = S[~S.dataset.isin(EXCLUDE_THERMO)]
    sets, val = msr_order(sorted(S.dataset.unique()), S)
    models = [m for m in EXT_ORDER if (S.model == m).any()]
    kinds = {m: dict(zip(VERSIONS[m], ('singles', 'doubles'))) for m in VERSIONS}
    S = S.assign(version=[kinds.get(r.model, {}).get(r.key, '') for r in S.itertuples()])
    return dict(S=S, sets=sets, val=val, models=models, ymin=min(0.10, float(np.floor((np.nanmin(S.lo) - 0.03) * 20) / 20)))


def draw_thermo(ax, th, ylim=None):
    """The thermostability sets (one cluster per set, ordered by ESM-MSR p0; one dot per model; hollow = singles model) on ``ax``; returns the legend handles."""
    S, sets, models = th['S'], th['sets'], th['models']
    spread = 0.78
    style.dot_panel(ax, sets, models, {}, None, ylabel=r'Spearman $\rho$ (pooled)', xlabels=[LABEL.get(d, d) for d in sets], rot=0, ylim=ylim or (th['ymin'], 0.98), spread=spread, size=6.0)
    offs = np.linspace(-spread / 2, spread / 2, len(models))
    for i, d in enumerate(sets):
        for j, m in enumerate(models):
            x = i + offs[j]
            hl = m == 'ESM-MSR'
            r = S[(S.dataset == d) & (S.model == m)].iloc[0]
            kw = style_kwargs(m, r.version == 'singles', 6.0 * (1.35 if hl else 1.0))
            ax.errorbar([x], [r.rho], yerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=(5 if hl else 3) - 1)
            ax.plot(x, r.rho, zorder=5 if hl else 3, **kw)
            if r.leak_flag:
                ax.plot(x, r.rho, marker='o', ms=10.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', zorder=6)
    hs = [Line2D([0], [0], label=MODELS[m]['label'], **style.marker_kwargs(m, size=6.8)) for m in models]
    hs.append(Line2D([0], [0], marker='o', ms=8.5, mfc='none', mec='#c0392b', mew=1.1, ls='none', label='training overlap (GeoDDG-3D)'))
    return hs + version_handles()


def write_thermo_table(th):
    order = {d: i + 1 for i, d in enumerate(th['sets'])}
    D = th['S'].assign(order=th['S'].dataset.map(order), msr_p0_seed_mean=th['S'].dataset.map(th['val'])).sort_values(['order', 'model'])
    D.to_csv(OUT / 'thermostability_sets.csv', index=False, float_format='%.4f')
    print('thermostability_sets, left to right:', ', '.join(f'{LABEL.get(d, d)} {th["val"][d]:.3f}' for d in th['sets']))


def thermostability_sets():
    th = thermo_data()
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    hs = draw_thermo(ax, th)
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
    write_thermo_table(th)


def summary_abc(pooled=True):
    """Main-text figure: a = the thermostability sets (whole top), b = ProteinGym / Domainome / MGnify / Rocklin, c = ESM-MSR against the LoRA scale per ProteinGym property; no titles, every note below.
    Written to ``results/00_main_text_figs/summary_abc.{png,pdf}``."""
    th, pg = thermo_data(), pg_prepare(pooled)
    W, H = 15.6, 11.6
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 0.55], width_ratios=[0.43, 0.57], hspace=0.42, wspace=0.15)
    axa, axb, axc = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    hs = draw_thermo(axa, th, ylim=(th['ymin'], 1.2))
    axa.set_yticks([t for t in np.arange(np.ceil(th['ymin'] * 5) / 5, 1.0001, 0.2)])
    axa.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=8.0, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.1, handletextpad=0.3)
    draw_pg_left(axb, pg)
    pg['fps'].draw_paired(axc, pg['t'], pg['Rp'], compact=True)
    note = 'a. ' + THERMO_NOTE + '\nb, c. ' + pg['note']
    kwt = dict(fontsize=7.4, color='#444444')
    tt = fig.text(0.012, 0.006, style.wrap_to_px(fig, note, 0.975 * fig.bbox.width, **kwt), va='bottom', ha='left', **kwt)
    foot = tt.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=foot + 0.075)
    fig.canvas.draw()
    pa, pb, pc = (a.get_position() for a in (axa, axb, axc))
    fig.text(pa.x0 - 0.04, 0.99, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    ytop = pb.y1 + 0.062                                                 # b and c: just above the category headers of c
    fig.text(pb.x0 - 0.047, ytop, 'b', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pc.x0 - 0.052, ytop, 'c', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.canvas.draw()
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in summary_abc', bad)
    out = out_dir('00_main_text_figs')
    plt.rcParams['savefig.bbox'] = 'standard'
    fig.savefig(out / 'summary_abc.png', dpi=300)
    fig.savefig(out / 'summary_abc.pdf')
    print('wrote', out / 'summary_abc.png', fig.get_size_inches())


def main():
    pg_categories()
    pg_categories(pooled=False)
    thermostability_sets()
    summary_abc()


if __name__ == '__main__':
    main()
