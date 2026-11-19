#!/usr/bin/env python
"""ESM-MSR against each other transfer method on ranking all Tsuboyama capped-test variants: two tables.

Every model is read as deployed: its singles version on the singles and its version trained with doubles on the doubles (ESM-MSR WT-view / Dual-view, FINAL_p0 s1-s3; MutateEverything Stage 1 / Stage 2; ThermoMPNN /
ThermoMPNN-D; SPURS single / SPURS-multi, the retrain with the trained adapter).  delta = rho(ESM-MSR) - rho(other); p one-sided = P(delta <= 0) (the claim "ESM-MSR ranks better"), p two-sided =
max(2 min(P(delta <= 0), P(delta >= 0)), 1 / (B + 1)); Holm over the three comparisons.

``domain mean``   the mean over the 33 test domains of the per-domain Spearman rho (the 'overall' dash of figure 2), seeds averaged; paired bootstrap that resamples **domains** (``--B-dom`` draws).
``pooled``        one Spearman rho over all test variants, seeds averaged; ESM-MSR is read with the Tobit link (``msr_analysis/tobit.py``), the others raw.  Two resampling units: **mutation** (the rows of every
                  domain are resampled within the domain, so the domain composition stays fixed) and **domain** (whole domains), each ``--B-pool`` draws shared by the models.  The raw (unlinked) ESM-MSR is listed too.

Mutation-level intervals say how much the pooled rho would move if different mutations of these same domains had been measured; they ignore that the test domains are themselves a sample, so they are narrower than
the domain-level ones, which are the ones that support a claim about new domains.

Output ``results/03_tsuboyama/overall_vs_transfer_{domain_mean,pooled}.{csv,md,png,pdf}``.

    python analysis/11j_overall_vs_transfer.py [--B-dom 100000] [--B-pool 1000]
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from msr_analysis import models as M
from msr_analysis.common import out_dir
from msr_analysis.tobit import linked

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m11i', HERE / '11i_doubles_vs_singles_pvalues.py')
m11i = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11i)
pub, holm, fmt_p = m11i.pub, m11i.holm, m11i.fmt_p

OUT = out_dir('03_tsuboyama')
OTHERS = [(M.BY_NAME[n].label_all.replace('\n', ' '), M.BY_NAME[n].single, M.BY_NAME[n].multi) for n in ('MutateEverything', 'ThermoMPNN', 'SPURS')]      # the roster's transfer models, each scored by the version that fits the variant


def p_tails(r):
    r = r[np.isfinite(r)]
    one = max((r <= 0).mean(), 1.0 / (len(r) + 1))
    two = max(2 * min((r <= 0).mean(), (r >= 0).mean()), 1.0 / (len(r) + 1))
    return one, two, *np.percentile(r, [2.5, 97.5])


def load():
    fr, _ = pub.load()
    sp = fr.spine
    n1 = (sp['n_mut'] == 1).to_numpy()[None, :]
    dep = {'ESM-MSR (Tobit-linked)': None, 'ESM-MSR (raw)': np.where(n1, fr.preds['msrp0:wt'], fr.preds['msrp0:comb'])}          # WT-view on singles, Dual-view on doubles, FINAL_p0 seeds 1-3
    dep['ESM-MSR (Tobit-linked)'] = linked(dep['ESM-MSR (raw)'], sp['dG_wt'].to_numpy(float))
    for name, a, b in OTHERS:
        dep[name] = np.where(n1, fr.preds[a], fr.preds[b])
    return fr, dep


def domain_mean(fr, dep, B):
    sp = fr.spine
    use, g = sp['use'].to_numpy(), sp['group'].to_numpy()
    libs = sorted(np.unique(g[use]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=B).astype(float)
    for k, v in dep.items():
        fr.preds['dep:' + k] = v
    V = {k: pub.per_library(fr, 'dep:' + k, use, pub.rho, 3) for k in dep}
    rows = []
    for name, _, _ in OTHERS:
        both = V['ESM-MSR (raw)'].index.intersection(V[name].index)
        d = pub._diff_df(V['ESM-MSR (raw)'], V[name], both)                       # the per-domain rho cannot change with the link: raw = linked
        one, two, lo, hi = p_tails(pub.boot_reps(d, libs, counts))
        dom = d.mean(axis=1)
        rows.append(dict(model=name, n_domains=len(both), rho_msr=V['ESM-MSR (raw)'].loc[both].mean(axis=1).mean(), rho_other=V[name].loc[both].mean(axis=1).mean(), delta=float(dom.mean()),
                         lo=lo, hi=hi, domains_msr_better=int((dom > 0).sum()), p_one=one, p=two, B=B))
    res = pd.DataFrame(rows)
    res['p_one_holm'], res['p_holm'] = holm(res.p_one), holm(res.p)
    return res


def pooled(fr, dep, B):
    sp = fr.spine
    use, g, y = sp['use'].to_numpy(), sp['group'].to_numpy(), sp['y'].to_numpy()
    idx = np.flatnonzero(use)
    gi = g[idx]
    libs = sorted(np.unique(gi))
    by = [idx[gi == lib] for lib in libs]
    sizes = np.array([len(b) for b in by])
    order = np.concatenate(by)
    dom_of = np.repeat(np.arange(len(libs)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    names = list(dep)
    P = {k: np.atleast_2d(v) for k, v in dep.items()}

    def rho_of(rows):
        ry = rankdata(y[rows])
        return {k: np.mean([np.corrcoef(rankdata(p[rows]), ry)[0, 1] for p in P[k]]) for k in names}

    pt = rho_of(order)
    rng = np.random.default_rng(9)
    reps = {'mutation': [], 'domain': []}
    for _ in range(B):
        r = starts[dom_of] + (rng.random(len(order)) * sizes[dom_of]).astype(int)           # mutation level: rows resampled inside every domain
        reps['mutation'].append(rho_of(order[r]))
        di = rng.integers(0, len(libs), len(libs))                                           # domain level: whole domains resampled
        reps['domain'].append(rho_of(np.concatenate([by[i] for i in di])))
    rows = []
    for a in ('ESM-MSR (Tobit-linked)', 'ESM-MSR (raw)'):
        for name, _, _ in OTHERS:
            row = dict(esm=a, model=name, rho_msr=pt[a], rho_other=pt[name], delta=pt[a] - pt[name], B=B)
            for unit in ('mutation', 'domain'):
                d = np.array([r[a] - r[name] for r in reps[unit]])
                one, two, lo, hi = p_tails(d)
                row.update({f'p_one_{unit}': one, f'p_{unit}': two, f'lo_{unit}': lo, f'hi_{unit}': hi})
            rows.append(row)
    res = pd.DataFrame(rows)
    for a, sub in res.groupby('esm'):
        for unit in ('mutation', 'domain'):
            res.loc[sub.index, f'p_one_holm_{unit}'] = holm(sub[f'p_one_{unit}'])
            res.loc[sub.index, f'p_holm_{unit}'] = holm(sub[f'p_{unit}'])
    return res


def render(rows, head, widths, stem, note):
    fig = plt.figure(figsize=(sum(widths) + 0.4, 0.42 * (len(rows) + 1.6) + 1.0))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 0.8 / fig.get_figheight(), sum(widths) / fig.get_figwidth(), 1 - 0.88 / fig.get_figheight()])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=head, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(widths) for w in widths], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#cccccc')
        c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
        elif j == 0:
            c.set_text_props(ha='left', fontweight='bold')
            c._loc = 'left'
    fig.text(0.2 / fig.get_figwidth(), 0.68 / fig.get_figheight(), note, fontsize=8.5, color='#444444', va='top', ha='left')
    fig.savefig(OUT / f'{stem}.png', dpi=200)
    fig.savefig(OUT / f'{stem}.pdf')
    plt.close(fig)


def md(rows, head, note, stem):
    lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    (OUT / f'{stem}.md').write_text('\n'.join(lines) + '\n\n' + note.replace('\n', ' ') + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B-dom', type=int, default=100000)
    ap.add_argument('--B-pool', type=int, default=1000)
    a = ap.parse_args()
    fr, dep = load()
    d = domain_mean(fr, dep, a.B_dom)
    d.to_csv(OUT / 'overall_vs_transfer_domain_mean.csv', index=False, float_format='%.5g')
    pl = pooled(fr, dep, a.B_pool)
    pl.to_csv(OUT / 'overall_vs_transfer_pooled.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(d.round(5).to_string(index=False))
        print(pl.round(5).to_string(index=False))

    fl = lambda p, B: fmt_p(p, 1 / (B + 1))
    head1 = ['ESM-MSR against', 'ρ\nESM-MSR', 'ρ\nother', 'Δρ [95% CI]', 'Domains where\nESM-MSR is higher', 'p\none-sided', 'p one-sided\n(Holm)', 'p\ntwo-sided', 'p two-sided\n(Holm)']
    rows1 = [[r.model, f'{r.rho_msr:.3f}', f'{r.rho_other:.3f}', f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]', f'{r.domains_msr_better} / {r.n_domains}', fl(r.p_one, r.B), fl(r.p_one_holm, r.B), fl(r.p, r.B), fl(r.p_holm, r.B)] for r in d.itertuples()]
    note1 = (f'Tsuboyama capped test set, all variants (singles and doubles), each model read as deployed (singles version on singles, doubles-trained version on doubles). ρ: mean over test domains of the per-domain Spearman ρ, seeds averaged;\n'
             f'Δρ = ESM-MSR − other. Paired bootstrap over domains ({int(d.B.iloc[0]):,} draws shared by all models). One-sided p = P(Δρ ≤ 0); two-sided p = twice the smaller tail. Holm over the three comparisons.')
    render(rows1, head1, [2.2, 0.9, 0.9, 2.6, 1.7, 1.0, 1.3, 1.0, 1.3], 'overall_vs_transfer_domain_mean', note1)
    md(rows1, head1, note1, 'overall_vs_transfer_domain_mean')

    L = pl[pl.esm == 'ESM-MSR (Tobit-linked)']
    head2 = ['ESM-MSR\n(Tobit-linked) against', 'ρ\nESM-MSR', 'ρ\nother', 'Δρ [95% CI]\nmutation resampling', 'p one-sided\n(Holm)', 'p two-sided\n(Holm)', 'Δρ [95% CI]\ndomain resampling', 'p one-sided\n(Holm)', 'p two-sided\n(Holm)']
    rows2 = [[r.model, f'{r.rho_msr:.3f}', f'{r.rho_other:.3f}', f'{r.delta:+.3f}  [{r.lo_mutation:+.3f}, {r.hi_mutation:+.3f}]', fl(r.p_one_holm_mutation, r.B), fl(r.p_holm_mutation, r.B),
              f'{r.delta:+.3f}  [{r.lo_domain:+.3f}, {r.hi_domain:+.3f}]', fl(r.p_one_holm_domain, r.B), fl(r.p_holm_domain, r.B)] for r in L.itertuples()]
    note2 = (f'Tsuboyama capped test set, all variants, each model read as deployed. ρ: one Spearman ρ over all test variants, seeds averaged; ESM-MSR with the post hoc Tobit link h(dG_wt + latent) − dG_wt, the other models raw. '
             f'Δρ = ESM-MSR − other.\nBootstrap of {int(L.B.iloc[0]):,} draws shared by the models: mutation resampling resamples the rows inside every domain (domain composition fixed); domain resampling resamples whole domains. '
             'One-sided p = P(Δρ ≤ 0); Holm over the three comparisons.')
    render(rows2, head2, [2.4, 0.9, 0.9, 2.5, 1.2, 1.2, 2.5, 1.2, 1.2], 'overall_vs_transfer_pooled', note2)
    md(rows2, head2, note2, 'overall_vs_transfer_pooled')
    print('wrote', OUT / 'overall_vs_transfer_pooled.png')


if __name__ == '__main__':
    main()
