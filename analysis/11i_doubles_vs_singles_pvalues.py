#!/usr/bin/env python
"""Do the models trained with doubles lose performance on single mutants against their singles-only versions?  Tsuboyama capped test set, single-mutant subset.

For every model that has both versions, the statistic is the mean over test domains (domains with at least ``MIN_N`` single mutants) of the per-domain Spearman rho between prediction and ddG on the singles,
averaged over the seeds the model has.  ``delta`` = rho(trained with doubles) - rho(singles version); the intervals and p-values come from one bootstrap that resamples **domains** (shared by every model,
``--B`` draws; the models' seeds are resampled too, as in ``11_publication_figures``): p = max(2 min(P(delta <= 0), P(delta >= 0)), 1 / (B + 1)).  ``p one-sided`` is P(delta >= 0): the evidence that the
doubles-trained version is worse.  Holm adjustment over the models of the table that have a test.  A model whose two versions give the same single-mutant
predictions up to float noise (ThermoMPNN-D scores singles with the ThermoMPNN network: max difference under 0.5% of the prediction SD) is listed without a test.

Versions (singles -> trained with doubles, ``msr_analysis.models``): ESM-MSR WT-view -> Dual-view (the combined head; FINAL_p0 s1-s3); MutateEverything Stage 1 -> Stage 2; ThermoMPNN -> ThermoMPNN-D; SPURS single (retrain with the trained adapter) ->
SPURS-multi.  (No other ESM-MSR version and no frozen-adapter SPURS.)

Needs ``11_publication_figures.py``'s inputs (the Tsuboyama capped-test frame).  Output ``results/03_tsuboyama/doubles_vs_singles_on_singles.{csv,md,png,pdf}``.

    python analysis/11i_doubles_vs_singles_pvalues.py [--B 100000]
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msr_analysis import models as M
from msr_analysis.common import out_dir

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pub', HERE / '11_publication_figures.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

OUT = out_dir('03_tsuboyama')
# (label, singles key, key trained with doubles, headline, singles version, doubles version): the models that have both, in the roster order
MODELS = [(n, M.BY_NAME[n].single, M.BY_NAME[n].multi, True, M.version_label(n, 0), M.version_label(n, 1)) for n in M.TRANSFER]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1.0, (len(p) - np.arange(len(p))) * p[o]))
    out = np.empty(len(p))
    out[o] = adj
    return out


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def compute(B):
    fr, _ = pub.load()
    sp = fr.spine
    use = sp['use'].to_numpy()
    single = use & (sp['n_mut'].to_numpy() == 1)
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=B).astype(float)
    rows = []
    for name, ks, kd, head, ls, ld in MODELS:
        vs, vd = pub.per_library(fr, ks, single, pub.rho, pub.MIN_N), pub.per_library(fr, kd, single, pub.rho, pub.MIN_N)
        both = vs.index.intersection(vd.index)
        d = pub._diff_df(vd, vs, both)
        reps = pub.boot_reps(d, libs, counts)
        reps = reps[np.isfinite(reps)]
        dom = d.mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        ds_, dd_ = fr.preds[ks][:, single], fr.preds[kd][:, single]
        same = bool(np.nanmax(np.abs(ds_ - dd_)) < 5e-3 * np.nanstd(ds_))                   # the two versions score singles with the same network (float noise only): nothing to test
        rows.append(dict(model=name, same_predictions=same, headline=head, singles_version=ls, doubles_version=ld, n_domains=len(both), n_seeds=vs.shape[1],
                         rho_singles=float(vs.loc[both].mean(axis=1).mean()), rho_doubles=float(vd.loc[both].mean(axis=1).mean()), delta=float(dom.mean()), lo=lo, hi=hi,
                         domains_worse=int((dom < 0).sum()), p=float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1.0 / (len(reps) + 1))), p_loss_one_sided=float((reps >= 0).mean())))
    res = pd.DataFrame(rows)
    h = (res.headline & ~res.same_predictions).to_numpy()
    res['p_holm'] = np.nan
    res.loc[h, 'p_holm'] = holm(res.loc[h, 'p'])
    res['B'] = B
    res.to_csv(OUT / 'doubles_vs_singles_on_singles.csv', index=False, float_format='%.5g')
    return res


def fmt_p(p, floor=1e-5):
    return f'< {floor:.0e}'.replace('e-0', 'e-') if p <= 1.5 * floor else f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'


def table_rows(res):
    out = []
    for r in res.itertuples():
        verdict = 'lower' if r.delta < 0 else 'higher'
        sig = stars(r.p_holm) if r.headline else stars(r.p) + ' (not in family)'
        if r.same_predictions:
            out.append([r.model, f'{r.singles_version} → {r.doubles_version}', f'{r.rho_singles:.3f}', f'{r.rho_doubles:.3f}', 'identical predictions', '–', '–', '–', 'same network on singles, not tested'])
            continue
        out.append([r.model, f'{r.singles_version} → {r.doubles_version}', f'{r.rho_singles:.3f}', f'{r.rho_doubles:.3f}', f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]',
                    f'{r.domains_worse} / {r.n_domains}', fmt_p(r.p, 1 / (r.B + 1)), fmt_p(r.p_holm, 1 / (r.B + 1)) if r.headline else '–', f'{verdict}, {sig}'])
    return out


HEAD = ['Model', 'Singles version →\ntrained with doubles', 'ρ\nsingles', 'ρ trained\nwith doubles', 'Δρ [95% CI]', 'Domains\nlower', 'p', 'p\n(Holm)', 'Doubles-trained\nversion']


def write_md(res):
    rows = table_rows(res)
    lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in HEAD) + ' |', '|' + '---|' * len(HEAD)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    B = int(res.B.iloc[0])
    note = (f'\nTsuboyama capped test set, single mutants only. ρ = mean over test domains (≥ {pub.MIN_N} singles) of the per-domain Spearman ρ, seeds averaged; Δρ = doubles-trained − singles. '
            f'Paired bootstrap over domains ({B:,} draws, shared by all models); p two-sided; Holm over the tested models (ThermoMPNN is not tested: ThermoMPNN-D gives the same single-mutant predictions as ThermoMPNN).\n')
    (OUT / 'doubles_vs_singles_on_singles.md').write_text('\n'.join(lines) + '\n' + note)


def render(res):
    rows = table_rows(res)
    W = [2.5, 3.0, 0.9, 1.2, 2.8, 1.0, 0.95, 0.95, 3.1]                    # column widths in inches
    fig = plt.figure(figsize=(sum(W) + 0.4, 0.42 * (len(rows) + 1.6) + 0.85))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 0.62 / fig.get_figheight(), sum(W) / fig.get_figwidth(), 1 - 0.7 / fig.get_figheight()])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=HEAD, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(W) for w in W], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#cccccc')
        c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
            continue
        r = res.iloc[i - 1]
        c.set_facecolor('#ffffff' if r.headline else '#f6f6f6')
        if j == 0:
            c.set_text_props(ha='left', fontweight='bold' if r.headline else 'normal')
            c._loc = 'left'
            c.PAD = 0.04
        if r.same_predictions:
            continue
        p = r.p_holm if r.headline else r.p
        if j == 8 and p < 0.05:
            c.set_facecolor('#f6d5d0' if r.delta < 0 else '#d6ecd9')
    fig.text(0.2 / fig.get_figwidth(), 0.5 / fig.get_figheight(), f'Tsuboyama capped test set, single mutants. ρ: mean over test domains (≥ {pub.MIN_N} singles) of the per-domain Spearman ρ, seeds averaged; Δρ = trained with doubles − singles version. '
             f'p: two-sided, paired bootstrap over\ndomains ({int(res.B.iloc[0]):,} draws shared by all models); Holm over the tested models (ThermoMPNN-D scores singles with the ThermoMPNN network, so it is not tested).\n'
             'Red: significantly lower; green: significantly higher (p < 0.05).', fontsize=8.5, color='#444444', va='top', ha='left')
    fig.savefig(OUT / 'doubles_vs_singles_on_singles.png', dpi=200)
    fig.savefig(OUT / 'doubles_vs_singles_on_singles.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=100000)
    ap.add_argument('--figures-only', action='store_true')
    a = ap.parse_args()
    res = pd.read_csv(OUT / 'doubles_vs_singles_on_singles.csv') if a.figures_only else compute(a.B)
    res['headline'] = res['headline'].astype(bool)
    res['same_predictions'] = res['same_predictions'].astype(bool)
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(res.drop(columns=['singles_version', 'doubles_version']).round(4).to_string(index=False))
    write_md(res)
    render(res)
    print('wrote', OUT / 'doubles_vs_singles_on_singles.png')


if __name__ == '__main__':
    main()
