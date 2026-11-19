"""ESM-MSR against ESM3dG on every benchmark where both have predictions (a curiosity table, not a paper figure).

ESM-MSR is read as deployed (WT-view on singles-only sets, the Dual-view (combined head) where multi-mutants are scored; seeds 1-3 averaged) and ESM3dG is its single run (the direct prediction of every variant).
Metrics are those of Figure 3 and its source scripts: pooled Spearman rho on the thermostability and DMS sets (mutation-level bootstrap), mean per-domain rho on Domainome (the held-out domains), the mean per-assay rho of ProteinGym by category (held-out assays; assays resampled within category; no Stability category) and the pooled / per-scaffold rho of
MGnify and Rocklin (``18b_mgnify_models.py``).  Both models are scored on the same rows, and delta = ESM-MSR - ESM3dG has a paired bootstrap interval (shared draws, seeds resampled) everywhere except MGnify and
Rocklin, whose two intervals come from separate scaffold bootstraps.

ESM3dG is not scored on the Megascale (Tsuboyama) libraries, the Homology-Restricted MegaScale val / test sets and ProteinGym's 64 Tsuboyama Stability assays: its training data overlap them (so it has no ProteinGym average).
It is fine-tuned on MGnify stability data (960,216 training sequences; ``reference/leak_audit_inventories.csv``): MGnify and Rocklin are not held out for it, and the homology classes of the
leak audit (``reference/leak_audit_matrix.csv``: P same protein, H strong homolog, h weak homolog; the fraction of the set's rows) show how much of every other set has a homolog in its training data.

    python analysis/22_esm_msr_vs_esm3dg.py [--B 1000]
Writes ``results/22_esm_msr_vs_esm3dg/esm_msr_vs_esm3dg.{csv,md,png,pdf}``.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import out_dir
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import model_key_for_dataset
from msr_analysis.heldout import excluded
from msr_analysis.registry import CACHE, DMS3, LABEL, REFERENCE, RESULTS, TSU

OUT = out_dir('22_esm_msr_vs_esm3dg')
THERMO = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754', 's571', 'ptmul', 'ptmuld']
PG_CATS = ['Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness']
LEAK_NAME = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}


def p_two(d):
    d = d[np.isfinite(d)]
    return float(max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / (len(d) + 1)))


def paired_frame(ds, fr, mask=None, msr_key=None):
    """A Frame whose preds are ``ESM-MSR`` (all seeds) and ``ESM3dG`` on the rows where both are finite (and ``mask``)."""
    msr_key = msr_key or model_key_for_dataset('ESM-MSR', fr)
    if msr_key is None or 'esm3dg' not in fr.preds:
        return None
    a, b = fr.preds[msr_key], fr.preds['esm3dg']
    ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    sp = fr.spine.copy()
    sp['use'] = sp['use'].to_numpy() & ok & (True if mask is None else mask)
    return Frame(ds, sp, {'ESM-MSR': a, 'ESM3dG': b}), msr_key


def run_frames(items, kind, per_protein, B, seed):
    """items: {row label: (Frame, metric text)} -> rows of the table."""
    rows = []
    frames = {k: v[0] for k, v in items.items()}
    pt, reps = bootstrap_rho(frames, ['ESM-MSR', 'ESM3dG'], kind=kind, B=B, seed=seed, per_protein=per_protein)
    for k, (_, metric, group, leak) in items.items():
        (rm, _, n, ng), (re, _, _, _) = pt[k]['ESM-MSR'], pt[k]['ESM3dG']
        d = reps[k]['ESM-MSR'] - reps[k]['ESM3dG']
        rows.append(dict(group=group, benchmark=k, metric=metric, n=int(ng) if per_protein else int(n), msr=rm, msr_lo=ci(reps[k]['ESM-MSR'])[0], msr_hi=ci(reps[k]['ESM-MSR'])[1],
                         esm3dg=re, e_lo=ci(reps[k]['ESM3dG'])[0], e_hi=ci(reps[k]['ESM3dG'])[1], delta=rm - re, d_lo=np.nanquantile(d, 0.025), d_hi=np.nanquantile(d, 0.975), p=p_two(d), paired=True, leak=leak))
    return rows


def leak_codes():
    m = pd.read_csv(REFERENCE / 'leak_audit_matrix.csv')
    m = m[m.model == 'ESM3dG']
    return dict(zip(m.dataset, m.code.fillna('')))


def pg_rows(B, leak):
    T = pd.read_csv(CACHE / 'pg_all_models.csv', index_col=0)
    cat = pd.read_csv(RESULTS / '05_proteingym' / 'pg_all_models_per_dms.csv', index_col='DMS_id')['category']
    keep = [a for a in T.index if a in cat.index and a not in excluded('proteingym')]
    T = T.loc[keep]
    cat = cat.loc[keep]
    a, b = T['ESM-MSR'].to_numpy(float), T['pg:esm3dg'].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    rng = np.random.default_rng(22)
    cats = [c for c in PG_CATS if c != 'Stability']                       # ESM3ΔG is not scored on the Stability category: 64 of its 66 assays are Megascale libraries, which its training data overlap
    idx = {c: np.flatnonzero((cat == c).to_numpy() & ok) for c in cats}
    draws = {c: rng.choice(v, (B, len(v))) for c, v in idx.items()}
    def stats(x):
        pt = {c: float(np.mean(x[idx[c]])) for c in cats}
        rp = {c: x[draws[c]].mean(axis=1) for c in cats}
        pt['PG average'] = float(np.mean([pt[c] for c in cats]))
        rp['PG average'] = np.mean([rp[c] for c in cats], axis=0)
        return pt, rp
    pm, rm = stats(a)
    pe, re = stats(b)
    rows = []
    for c in ['PG average'] + cats:
        d = rm[c] - re[c]
        n = int(ok.sum()) if c == 'PG average' else len(idx[c])
        rows.append(dict(group='ProteinGym (held-out assays, no Stability)', benchmark='PG average (4 categories)' if c == 'PG average' else c, metric='mean per-assay rho', n=n, msr=pm[c], msr_lo=np.quantile(rm[c], .025), msr_hi=np.quantile(rm[c], .975),
                         esm3dg=pe[c], e_lo=np.quantile(re[c], .025), e_hi=np.quantile(re[c], .975), delta=pm[c] - pe[c], d_lo=np.quantile(d, .025), d_hi=np.quantile(d, .975), p=p_two(d), paired=True,
                         leak=leak.get({'PG average': 'PG avg', 'Stability': 'PG Stability'}.get(c, ''), '')))
    return rows


def mgnify_rows():
    rows = []
    for f, metric in (('mgnify_models_pooled.csv', 'pooled rho'), ('mgnify_models_per_scaffold.csv', 'mean per-scaffold rho')):
        d = pd.read_csv(RESULTS / '18_mgnify' / f)
        d = d[~d.model.str.contains('same ')]
        for ds in ('MGnify', 'Rocklin designs'):
            a, e = d[(d.dataset == ds) & (d.model == 'ESM-MSR')], d[(d.dataset == ds) & (d.model == 'ESM3dG')]
            if a.empty or e.empty:
                continue
            a, e = a.iloc[0], e.iloc[0]
            n = int(a.n_variants if 'n_variants' in a and pd.notna(a.get('n_variants')) else a.n_scaffolds)
            rows.append(dict(group='MGnify / Rocklin (not held out)', benchmark=ds, metric=metric, n=n, msr=a.rho, msr_lo=a.lo, msr_hi=a.hi, esm3dg=e.rho, e_lo=e.lo, e_hi=e.hi,
                             delta=a.rho - e.rho, d_lo=np.nan, d_hi=np.nan, p=np.nan, paired=False, leak='same source'))
    return rows


def make_table(B):
    leak = leak_codes()
    rows = []
    items = {}
    for ds in THERMO:
        fr = load_frame(ds)
        r = paired_frame(ds, fr)
        if r:
            items[LABEL.get(ds, ds)] = (r[0], 'pooled rho', 'Thermostability sets', leak.get(ds, ''))
    rows += run_frames(items, 'mutation', False, B, 11)
    items = {}
    for ds in DMS3:
        fr = load_frame(ds)
        r = paired_frame(ds, fr)
        if r:
            items[LABEL.get(ds, ds)] = (r[0], 'pooled rho', 'DMS sets', leak.get(ds, ''))
    rows += run_frames(items, 'mutation', False, B, 12)
    items = {}
    fr = load_frame('domainome')
    r = paired_frame('domainome', fr)
    if r:
        items['Domainome (held-out domains)'] = (r[0], 'mean per-domain rho', 'Domainome', leak.get('domainome', ''))
    for ds in ():                                                                  # ESM3ΔG is not scored on the Megascale (Tsuboyama) sets: its training data overlap them
        fr = load_frame(ds)
        nm = fr.spine['n_mut'].to_numpy()
        for tag, mask, key in (('', None, None), (', singles', nm == 1, 'msrp0:wt'), (', doubles', nm == 2, 'msrp0:comb')):
            if tag and ds != TSU[1]:
                continue
            r = paired_frame(ds, fr, mask, key)
            if r:
                items[LABEL.get(ds, ds) + tag] = (r[0], 'mean per-library rho', 'Domainome', leak.get(LEAK_NAME.get(ds, ds), ''))
    rows += run_frames(items, 'protein', True, B, 13)
    rows += pg_rows(B, leak)
    rows += mgnify_rows()
    return pd.DataFrame(rows)


def fmt(v, lo=None, hi=None):
    s = f'{v:.3f}'
    return s if lo is None or not np.isfinite(lo) else f'{s}  [{lo:.3f}, {hi:.3f}]'


def table_cells(T):
    out = []
    for r in T.itertuples():
        dcell = f'{r.delta:+.3f}  [{r.d_lo:+.3f}, {r.d_hi:+.3f}]' if r.paired else f'{r.delta:+.3f}  (separate intervals)'
        out.append([r.benchmark, r.metric, f'{r.n:,}', fmt(r.msr, r.msr_lo, r.msr_hi), fmt(r.esm3dg, r.e_lo, r.e_hi), dcell, r.leak or '–'])
    return out


HEAD = ['Benchmark', 'Metric', 'n', 'ESM-MSR\nρ [95% CI]', 'ESM3ΔG\nρ [95% CI]', 'Δρ = ESM-MSR − ESM3ΔG\n[95% CI, paired]', 'ESM3ΔG overlap\n(leak audit)']


def write_md(T):
    cells = table_cells(T)
    lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in HEAD) + ' |', '|' + '---|' * len(HEAD)]
    last = None
    for r, row in zip(T.itertuples(), cells):
        if r.group != last:
            lines.append(f'| **{r.group}** |' + ' |' * (len(HEAD) - 1))
            last = r.group
        lines.append('| ' + ' | '.join(row) + ' |')
    (OUT / 'esm_msr_vs_esm3dg.md').write_text('\n'.join(lines) + '\n\n' + NOTE + '\n')


NOTE = ('ESM-MSR: deployed readout (WT-view on singles, Dual-view where multi-mutants are scored), seeds 1-3 averaged; ESM3ΔG: one run, direct prediction. Both on the same rows. Δρ intervals: paired bootstrap '
        '(draws shared by both models, seeds resampled; mutation level for the thermostability and DMS sets, domain level for Domainome, assays within category for ProteinGym (no Stability: Megascale overlap); MGnify and Rocklin: '
        'separate scaffold bootstraps). ESM3ΔG is not scored on Megascale (Tsuboyama) data, which its training data overlap. It is fine-tuned on MGnify stability data (960,216 sequences): MGnify and Rocklin are not held out for it. Overlap = the leak audit\'s share of the set\'s rows '
        'whose protein has a homolog in that training data (P same protein, H strong homolog, h weak homolog), before the four-part held-out filter. Green / red: Δρ interval excludes 0 (ESM-MSR higher / lower).')


def render(T):
    cells = table_cells(T)
    W = [3.0, 2.0, 0.9, 2.3, 2.3, 3.0, 1.9]
    n_groups = T.group.nunique()
    nrows = len(cells) + n_groups
    fig = plt.figure(figsize=(sum(W) + 0.4, 0.34 * (nrows + 1.2) + 1.5))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 1.25 / fig.get_figheight(), sum(W) / fig.get_figwidth(), 1 - 1.4 / fig.get_figheight()])
    ax.axis('off')
    body, kinds, last = [], [], None
    for r, row in zip(T.itertuples(), cells):
        if r.group != last:
            body.append([r.group] + [''] * (len(HEAD) - 1)); kinds.append(('group', None)); last = r.group
        body.append(row); kinds.append(('row', r))
    t = ax.table(cellText=body, colLabels=HEAD, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(W) for w in W], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#d0d0d0')
        c.set_linewidth(0.5)
        if i == 0:
            c.set_facecolor('#e6e6e6'); c.set_text_props(fontweight='bold'); continue
        kind, r = kinds[i - 1]
        if kind == 'group':
            c.set_facecolor('#f0f0f0'); c.set_edgecolor('#f0f0f0')
            if j == 0:
                c.set_text_props(ha='left', fontweight='bold'); c._loc = 'left'; c.PAD = 0.02
            continue
        c.set_facecolor('#ffffff')
        if j == 0:
            c.set_text_props(ha='left'); c._loc = 'left'; c.PAD = 0.04
        if j == 5 and r.paired and np.isfinite(r.d_lo) and (r.d_lo > 0 or r.d_hi < 0):
            c.set_facecolor('#d6ecd9' if r.delta > 0 else '#f6d5d0')
    from msr_analysis import style
    kw = dict(fontsize=8.2, color='#444444')
    fig.text(0.2 / fig.get_figwidth(), 1.1 / fig.get_figheight(), style.wrap_to_px(fig, NOTE, 0.97 * fig.bbox.width, **kw), va='top', ha='left', **kw)
    fig.savefig(OUT / 'esm_msr_vs_esm3dg.png', dpi=200)
    fig.savefig(OUT / 'esm_msr_vs_esm3dg.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000)
    a = ap.parse_args()
    T = make_table(a.B)
    T.to_csv(OUT / 'esm_msr_vs_esm3dg.csv', index=False, float_format='%.4f')
    write_md(T)
    render(T)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 100):
        print(T[['group', 'benchmark', 'metric', 'n', 'msr', 'esm3dg', 'delta', 'd_lo', 'd_hi']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
