#!/usr/bin/env python
"""Training overlap, clean-cell means and the published-paper reference: BENCHMARKS tables 1.1-1.4 (leak codes), 1.5, 6.2, 6.3, 6.5.

* ``leak_matrix_external`` / ``leak_matrix_other``  the leak-code tables (1.1, 1.2, 1.3, 1.4) as colour maps: the worst class a model has with a
  dataset (classes of table 6.2: M identical mutation, m constituent mutations, P same protein, H strong homolog, h weak similarity, D documented
  training on the source set, C pretraining only, # the cell is a training-set score), thresholded at 0.5% of rows like the document.
* ``leak_overlap_tsuboyama_test``  table 6.5 'what the audit finds' for the capped test libraries: the fraction of rows each model has seen.
* ``clean_means``  table 6.5 'clean external-set means': the mean over the 12 sets with and without the cells that carry overlap.
* ``spurs_published_vs_retrained``  the overlap made visible: SPURS' public checkpoint (leaky; values frozen from the earlier tables, no longer in the document) against the retrained one.
* ``published_reference``  table 1.5, the evaluation of the published paper (hyperopt split, earlier models; compare within it only).
* ``leak_audit_text_tables``  tables 6.2, 6.3 and 6.5 (findings) set as figures.
"""
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import load_rho_df, out_dir, primary_ci  # noqa: E402
from msr_analysis.registry import EXT12, MODELS, REFERENCE  # noqa: E402
from msr_analysis.tables import clean_label, num, read_table  # noqa: E402

OUT = out_dir('10_leakage_and_reference')
SEV = {'#': 7, 'M': 6, 'm': 5, 'D': 5, 'U': 5, 'P': 4, 'H': 3, 'h': 2, 'C': 1}
COLORS = ['#ffffff', '#cfe3f1', '#e8e8e8', '#ffe699', '#f4b183', '#ee7a4d', '#d62728', '#7a0019']
NAMES = ['none ≥ 0.5% of rows', 'C  pretraining only', 'h  weak similarity', 'P  same protein', 'H  strong homolog', 'm / D / U  constituent or documented', 'M  identical mutation', '#  training-set score']
_tok = re.compile(r'([#MmPHhDCU])\??\s*(<?\.?\d*\.?\d*)')


def worst(code):
    s = str(code)
    if s.strip() in ('', '–', '-', 'n.a.'):
        return 0
    best = 0
    for cls, frac in _tok.findall(s):
        f = frac.replace('<', '')
        try:
            v = float(f) if f else 1.0
        except ValueError:
            v = 1.0
        if cls == 'C' or cls == '#' or v >= 0.005:
            best = max(best, SEV.get(cls, 0))
    return best


def leak_map(tid, ax, short):
    d = read_table(tid)
    rows = [clean_label(x) for x in d.iloc[:, 0]]
    keep = [c for c in d.columns[1:] if (d[c].str.strip() != '–').any()]
    d = d[[d.columns[0]] + keep]
    cols = [c.replace(' (leak code)', '') for c in keep]
    Z = np.array([[worst(x) for x in d[c]] for c in keep]).T
    ax.imshow(Z, cmap=ListedColormap(COLORS), vmin=0, vmax=7, aspect='auto')
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            t = str(d.iloc[i, j + 1]).replace(' V', '').replace('V', '')
            if t.strip() not in ('', '–'):
                ax.text(j, i, textwrap.fill(t, 9), ha='center', va='center', fontsize=4.8 if short else 5.4, color='white' if Z[i, j] >= 6 else 'black')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def leak_maps():
    fig, ax = plt.subplots(figsize=(13, 8.6))
    leak_map('1.1.2', ax, True)
    ax.set_title('Training overlap of each model with the 12 external sets (table 1.1, leak codes; fraction of rows in the cell)', loc='left')
    ax.legend(handles=[Patch(fc=c, ec='#999', label=n) for c, n in zip(COLORS, NAMES)], loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=8)
    fig.tight_layout()
    style.save(fig, OUT / 'leak_matrix_external')
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.2), gridspec_kw=dict(width_ratios=[2.2, 8, 3.2]))
    for ax, tid, t in zip(axes, ('1.2.2', '1.3.2', '1.4.2'), ('Tsuboyama (1.2)', 'DMS and Domainome (1.3)', 'ProteinGym (1.4)')):
        leak_map(tid, ax, tid != '1.2.2')
        ax.set_title(t, loc='left')
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    style.save(fig, OUT / 'leak_matrix_other')


def overlap_bars():
    d = pd.concat([pd.read_csv(REFERENCE / 'leak_audit_matrix.csv'), pd.read_csv(REFERENCE / 'leak_public_spurs.csv').assign(model=lambda x: x.model + ' (public checkpoint)')])
    t = d[d.dataset == 'tsu_capped_test'].copy()
    for c in ('exact', 'constituent', 'P', 'H', 'h'):
        t[c] = t[c].fillna(0) / t['n']
    t['seen'] = t['exact'] + t['constituent']
    keep = ['SPURS (public checkpoint)', 'SPURS multi (public checkpoint)', 'ESMtherm', 'IFUM (WT backbone)', 'ESM3dG', 'DDGemb', 'GeoDDG-3D', 'ESM-MSR (released)', 'ME stage1', 'ME stage2', 'TMPNN', 'TMPNN+D',
            'SPURS (retrained)', 'SPURS multi (retrained)', 'ProSST', 'ProteinMPNN']
    t = t[t.model.isin(keep)].drop_duplicates('model').set_index('model').reindex(keep)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for c, col, lab in (('exact', '#d62728', 'identical mutation (M)'), ('constituent', '#ee7a4d', 'constituent mutations (m)'), ('P', '#ffc000', 'same protein, other mutations (P)'),
                        ('H', '#9ecae1', 'strong homolog (H)'), ('h', '#d9d9d9', 'weak similarity (h)')):
        v = t[c].fillna(0).to_numpy()
        ax.barh(y, v, left=left, color=col, label=lab, height=.7)
        left += v
    ax.set_yticks(y); ax.set_yticklabels([m.replace('ESM-MSR (released)', 'ESM-MSR (baseline, devel)').replace(' (retrained)', '').replace('ESM3dG', 'ESM3ΔG*').replace('ESMtherm', 'ESMtherm*').replace('IFUM (WT backbone)', 'IFUM, WT backbone*') for m in t.index]); ax.invert_yaxis()
    ax.set_xlabel('fraction of the capped Tsuboyama test rows'); ax.set_xlim(0, 1.2)
    ax.legend(fontsize=7.5, loc='lower right'); ax.set_title('What each model has seen of the capped test libraries (leak audit); * trained on Megascale data', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, OUT / 'leak_overlap_tsuboyama_test')


def clean_means():
    d = read_table('6.5.2')
    rows = []
    for _, r in d.iterrows():
        m = re.match(r'\s*\**\s*([\d.]+)\s*\**\s*\((\d+)\)', r['clean mean (n of 12)'])
        rows.append(dict(model=clean_label(r['model']), clean=float(m.group(1)), n_clean=int(m.group(2)), all12=num(r['all-12 mean'])[0], excluded=r['cells excluded']))
    R = pd.DataFrame(rows).sort_values('clean')
    R.to_csv(OUT / 'clean_means.csv', index=False, float_format='%.3f')
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(R))
    ax.hlines(y, R.all12, R.clean, color='#bbbbbb', lw=2)
    ax.scatter(R.all12, y, s=40, facecolor='white', edgecolor='k', zorder=3, label='mean over all 12 sets')
    ax.scatter(R.clean, y, s=40, color=['#000000' if 'ESM-MSR' in m else '#d35400' for m in R.model], zorder=4, label='mean over the cells without overlap')
    for i, r in enumerate(R.itertuples()):
        ax.text(max(r.clean, r.all12) + .008, i, f'{r.clean:.3f}  ({r.n_clean} of 12 sets)', va='center', fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(R.model); ax.set_xlim(0.3, 0.95); ax.set_xlabel('mean Spearman ρ over the 12 external sets')
    ax.legend(loc='lower right', fontsize=8); ax.set_title('Clean-cell means (table 6.5): models with overlap lose cells, not necessarily score', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, OUT / 'clean_means')


def spurs_pub_vs_retrained():
    ref = pd.read_csv(REFERENCE / 'spurs_public_checkpoint.csv')
    pub = ref[ref.model == 'SPURS (public checkpoint)'].set_index('dataset').rho
    df = load_rho_df()
    P = primary_ci(df)
    ds = EXT12 + ['tsuboyama_capped_val', 'tsuboyama_capped_test', 'DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain',
                  'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain', 'domainome']
    ren = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}
    from msr_analysis.registry import LABEL
    fig, ax = plt.subplots(figsize=(12, 3.6))
    rows = []
    for i, d in enumerate(ds):
        v = P.get((d, 'spurs'))
        p = pub.get(ren.get(d, d), np.nan)
        if v is None or not np.isfinite(p):
            continue
        ax.plot([i, i], [v[0], p], color='#bbbbbb', lw=2, zorder=1)
        ax.errorbar([i], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt='s', color='#d35400', capsize=2, zorder=3, ms=6)
        ax.plot([i], [p], marker='s', mfc='white', mec='#d35400', ms=7, ls='none', zorder=4)
        rows.append(dict(dataset=d, retrained=v[0], published=p, inflation=p - v[0]))
    ax.plot([], [], 's', color='#d35400', label='retrained on the capped split (3 seeds; the SPURS of the other figures)'); ax.plot([], [], 's', mfc='white', mec='#d35400', label='public checkpoint (trained on 24 of the 33 test libraries)')
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([LABEL[d] for d in ds], rotation=35, ha='right'); ax.set_ylabel('Spearman ρ'); ax.legend(fontsize=8, loc='lower left')
    ax.set_title('SPURS: what training overlap adds (public checkpoint, frozen from the earlier BENCHMARKS tables, against the retrained SPURS)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, OUT / 'spurs_published_vs_retrained')
    pd.DataFrame(rows).to_csv(OUT / 'spurs_published_vs_retrained.csv', index=False, float_format='%.4f')


def published_reference():
    d = read_table('1.5.1')
    models = list(d.columns[1:])
    groups = [r for r in d.iloc[:, 0]]
    table = {}
    for _, r in d.iterrows():
        for m in models:
            v, _s = num(r[m])
            if np.isfinite(v):
                table[(r.iloc[0], m)] = v
    pal = ['#000000', '#34495e', '#1abc9c', '#999999', '#f1948a', '#e74c3c', '#922b21', '#2ecc71', '#abebc6', '#3498db', '#1f5f8b']
    mk = ['D', 'd', '+', '^', 'o', 'o', 'o', 'h', 'h', 'p', 'P']
    for m, c, k in zip(models, pal, mk):
        MODELS[f'ref:{m}'] = dict(label=m, color=c, marker=k, fill=m not in ('ME additive (hyperopt)',), group='ref')
    fig, ax = plt.subplots(figsize=(14, 4.6))
    style.dot_panel(ax, groups, [f'ref:{m}' for m in models], {(g, f'ref:{m}'): v for (g, m), v in table.items()}, None, highlight=['ref:ESM-MSR (published)'],
                    ylabel='Spearman ρ (published-paper evaluation)', xlabels=[g.replace(' (per-domain avg.)', '') for g in groups], size=6)
    style.model_legend(ax, [f'ref:{m}' for m in models], loc='upper right', ncol=4, fontsize=7.3)
    ax.set_title('Table 1.5: the evaluation of the published paper (hyperopt split, earlier models; compare within this table only)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, OUT / 'published_reference')


def draw_text_table(ax, d, widths, fs=7.2, title=''):
    """rows of wrapped text on a plain axes (matplotlib's table cannot size wrapped rows)"""
    ax.axis('off')
    ax.set_xlim(0, 1)
    xs = np.r_[0, np.cumsum(widths) / np.sum(widths)][:-1] + 0.004
    lines_all = [[textwrap.wrap(str(x).replace('`', '').replace('**', ''), max(int(w * 1.05), 6)) or [''] for x, w in zip(r, widths)] for r in d.values.tolist()]
    head = [textwrap.wrap(c, max(int(w * 1.05), 6)) for c, w in zip(d.columns, widths)]
    lh = fs * 1.55
    heights = [max(len(c) for c in head)] + [max(len(c) for c in r) for r in lines_all]
    total = sum(h * lh + 5 for h in heights)
    ax.set_ylim(total, 0)
    y = 0
    for j, (r, h) in enumerate(zip([head] + lines_all, heights)):
        if j == 0:
            ax.add_patch(plt.Rectangle((0, y), 1, h * lh + 5, color='#eeeeee', lw=0))
        for x, cell in zip(xs, r):
            for i, ln in enumerate(cell):
                ax.text(x, y + 3 + i * lh, ln, fontsize=fs, va='top', ha='left', fontweight='bold' if j == 0 else 'normal')
        y += h * lh + 5
        ax.plot([0, 1], [y, y], color='#dddddd', lw=.5)
    ax.set_title(title, loc='left', fontsize=10)
    return total


def text_tables():
    specs = [('6.2.1', 'Table 6.2: notation of the leak codes', [6, 46, 60, 8]), ('6.3.1', 'Table 6.3: training inventories', [20, 56, 38, 22]),
             ('6.5.1', 'Table 6.5: what the audit finds', [55, 40, 40])]
    heights = []
    for tid, _, w in specs:
        d = read_table(tid)
        heights.append(sum(max(len(textwrap.wrap(str(x), max(int(ww * 1.05), 6))) or 1 for x, ww in zip(r, w)) for r in d.values.tolist()) + 3)
    fig, axes = plt.subplots(3, 1, figsize=(13, 0.19 * sum(heights) + 1.5), gridspec_kw=dict(height_ratios=heights))
    for ax, (tid, title, w) in zip(axes, specs):
        draw_text_table(ax, read_table(tid), w, title=title)
    fig.tight_layout()
    style.save(fig, OUT / 'leak_audit_text_tables')


if __name__ == '__main__':
    leak_maps()
    overlap_bars()
    clean_means()
    spurs_pub_vs_retrained()
    published_reference()
    text_tables()
    print(pd.read_csv(OUT / 'spurs_published_vs_retrained.csv').describe().loc[['mean', 'min', 'max']].round(3))
