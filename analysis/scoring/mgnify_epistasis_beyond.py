"""Beyond-global ("beyond rho", saturation-subtracted) epistasis of ESM-MSR on all the new data (MGnify + the Rocklin de novo designs) with a
mutation-level bootstrap test of "epistasis prediction > 0".

Predicted epistasis = comb(AB) - WT(A) - WT(B) (latent, no link; the 'epi' head of ``esm_msr.epi_metrics`` fed comb, as the Megascale tables,
analysis/98_final_ablation_table.py).  Rows enter as in the tables: measured doubles whose two singles are both measured, all three rows uncensored
(cens == 0), wild-type backbones only.  Beyond rho = Spearman of r' against r, r = dddG - E[dddG | x] (x = dG_wt + ddG_A + ddG_B) and
r' = dddG' - E[dddG' | x'] (x' = dG_wt + the WT head's additive prediction); both E[. | .] are 25-bin smoothers refitted on the table, so what is
scored is the epistasis above the global saturation curve (library / pair / line / cell levels all included).  Naive rho (the raw dddG) is
reported next to it.  The pair, line and cell levels need position-pair matrices (>= 20 doubles per pair) that these scaffolds do not have.

Groups: MGnify (mgnify == True), Rocklin designs (rocklin_denovo), Together (one table, one smoother).  The point value is the mean over the three
seeds.  Tests (one sided, H0: rho <= 0; p = (1 + #draws <= 0) / (B + 1) on the seed-mean statistic, the same draw for every seed):
  mutation level   resample the doubles (a double's two singles stay with it; the smoothers are refitted on every draw) -- the requested test
  scaffold level   resample whole scaffolds -- the companion that respects the correlation of doubles that share a scaffold and singles

  python analysis/scoring/mgnify_epistasis_beyond.py final_mgnify/FINAL_p0 [--B 1000] [--B_scaf 200] [--jobs 5] [--out DIR]
Writes summary.csv, boot_draws.csv.gz, residuals_together.csv.gz and beyond_rho_epistasis.{png,pdf} into the output directory.
"""
import argparse
import copy
import os
import re
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
from esm_msr import epi_metrics as E  # noqa: E402
sys.path.insert(0, HERE)
from mgnify_labels import apply_v2_labels  # noqa: E402

SEEDS = (1, 2, 3)
GROUPS = ('MGnify', 'Rocklin designs', 'Together')
COLS = ['uid', 'source', 'code_wt', 'mut_type', 'n_mut', 'ddG_ML', 'dddG_ML', 'dG_wt', 'cens', 'backbone_mutation', 'wt_lora_pred', 'combined_pred']
_MUT = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')
TABLES = {}      # (group, seed) -> E.Table, shared with the forked workers
META = {}


def parse(m, code):
    out = []
    for part in str(m).split(':'):
        g = _MUT.match(part.strip())
        if not g:
            return None
        out.append((code, int(g.group(2)), g.group(3)))
    return tuple(out)


def load(root):
    per = []
    for s in SEEDS:
        fr = []
        for sub in ('all', 'rdenovo'):
            d = pd.read_csv(os.path.join(root, f's{s}', sub, 'predictions.csv.gz'), usecols=lambda c: c in COLS + ['in_range_1_5', 'qc_pass_both', 'wt_CI', 'dG_ML', 'ddG', 'ddG_additive_ML', 'has_all_singles'], low_memory=False)
            d = apply_v2_labels(d)
            d['source'] = d['source'] if 'source' in d else 'mgnify'          # the v1 run has no source column (every row is MGnify)
            fr.append(d)
        d = pd.concat(fr, ignore_index=True).sort_values('uid', ignore_index=True)
        per.append(d)
    assert all((per[0]['uid'].to_numpy() == x['uid'].to_numpy()).all() for x in per)
    return per


def build_tables(per):
    b = per[0]
    bb = b['backbone_mutation']
    print('backbone_mutation:', bb.value_counts(dropna=False).to_dict())
    wtbb = ~bb.map(lambda v: str(v).lower() in ('true', '1', '1.0')).to_numpy()
    use = (np.isfinite(b['ddG_ML'].to_numpy(float)) & (b['cens'].to_numpy() == 0) & wtbb)
    is_mg = (b['source'] == 'mgnify').to_numpy()
    for g, gm in (('MGnify', is_mg), ('Rocklin designs', ~is_mg), ('Together', np.ones(len(b), bool))):
        idx = np.flatnonzero(use & gm & b['n_mut'].isin([1, 2]).to_numpy())
        keys = [parse(m, c) for m, c in zip(b['mut_type'].to_numpy()[idx], b['code_wt'].astype(str).to_numpy()[idx])]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        base = dict(mut_key=keys, subset_type=st, ddG=b['ddG_ML'].to_numpy(float)[idx], dddG=b['dddG_ML'].to_numpy(float)[idx], dG_wt=b['dG_wt'].to_numpy(float)[idx])
        for j, s in enumerate(SEEDS):
            d = per[j]
            wt, comb = d['wt_lora_pred'].to_numpy(float)[idx], d['combined_pred'].to_numpy(float)[idx]
            TABLES[(g, s)] = E.Table(dict(base, wt=wt, mt=comb, comb=comb), None, heads=('epi', 'add'))
        t = TABLES[(g, 1)]
        META[g] = dict(n_doubles=int(t.n), n_scaffolds=int(len(np.unique(t.lib))))
        print(g, META[g], flush=True)


def stats(t):
    return E._rho(*E.beyond_global(t, 'epi')[::-1]), E.naive_rho(t, 'epi')


def sub(t, idx):
    r = copy.copy(t)
    n0 = t.n
    for name, v in vars(t).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    r._mats = None
    return r


def one_draw(task):
    g, kind, i = task
    rng = np.random.default_rng([7, GROUPS.index(g), kind == 'scaffold', i])
    t0 = TABLES[(g, 1)]
    if kind == 'mutation':
        idx = rng.integers(0, t0.n, t0.n)
    else:
        libs = np.unique(t0.lib)
        order = np.argsort(t0.lib, kind='stable')
        bounds = np.searchsorted(t0.lib[order], np.r_[libs, libs.max() + 1])
        pick = rng.integers(0, len(libs), len(libs))
        idx = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
    v = np.array([stats(sub(TABLES[(g, s)], idx)) for s in SEEDS])      # seeds x (beyond, naive)
    return (g, kind, i, *v.mean(axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--B_scaf', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'new_data_epistasis')
    os.makedirs(out, exist_ok=True)
    build_tables(load(a.root))
    pts = {g: np.array([stats(TABLES[(g, s)]) for s in SEEDS]) for g in GROUPS}
    for g in GROUPS:
        print(g, 'beyond/naive per seed', np.round(pts[g], 4).tolist(), flush=True)
    tasks = [(g, 'mutation', i) for g in GROUPS for i in range(a.B)] + [(g, 'scaffold', i) for g in GROUPS for i in range(a.B_scaf)]
    with get_context('fork').Pool(a.jobs) as p:
        res = p.map(one_draw, tasks, chunksize=4)
    bd = pd.DataFrame(res, columns=['group', 'kind', 'draw', 'beyond_rho', 'naive_rho'])
    bd.to_csv(os.path.join(out, 'boot_draws.csv.gz'), index=False, float_format='%.5f')
    rows = []
    for g in GROUPS:
        for mi, metric in enumerate(('beyond_rho', 'naive_rho')):
            r = dict(group=g, metric=metric, **META[g], seed_values=' '.join(f'{x:.4f}' for x in pts[g][:, mi]), value=float(pts[g][:, mi].mean()),
                     sd_seeds=float(pts[g][:, mi].std()))
            for kind, tag in (('mutation', 'mut'), ('scaffold', 'scaf')):
                v = bd[(bd.group == g) & (bd.kind == kind)][metric].to_numpy()
                r.update({f'lo_{tag}': float(np.percentile(v, 2.5)), f'hi_{tag}': float(np.percentile(v, 97.5)), f'B_{tag}': len(v),
                          f'p_gt0_{tag}': float((1 + (v <= 0).sum()) / (len(v) + 1)), f'frac_le0_{tag}': float((v <= 0).mean())})
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, 'summary.csv'), index=False, float_format='%.5g')
    pd.set_option('display.width', 250)
    print(S.drop(columns=['seed_values']).to_string(index=False))
    # residuals of the Together table (seed-mean r'), for the scatter
    t = TABLES[('Together', 1)]
    rm = E.beyond_global(t, 'epi')[0]
    rp = np.mean([E.beyond_global(TABLES[('Together', s)], 'epi')[1] for s in SEEDS], axis=0)
    pd.DataFrame(dict(r_measured=rm, r_pred_seedmean=rp, x=t.x, dddG=t.dddG)).to_csv(os.path.join(out, 'residuals_together.csv.gz'), index=False, float_format='%.4f')
    plot(out, S, bd, rm, rp)


def plot(out, S, bd, rm, rp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
    COL = {'MGnify': '#1baf7a', 'Rocklin designs': '#eb6834', 'Together': '#2a78d6'}
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw=dict(width_ratios=[1.05, 1.0, 1.05]))
    # A: predicted vs measured beyond-global epistasis (Together)
    ok = np.isfinite(rm) & np.isfinite(rp)
    lim = lambda v: np.percentile(v[ok], [0.5, 99.5])
    xl, yl = lim(rm), lim(rp)
    hb = ax[0].hexbin(rm[ok], rp[ok], gridsize=55, extent=(*xl, *yl), bins='log', cmap='Blues', mincnt=1, linewidths=0)
    ax[0].axhline(0, color=GRID, lw=0.8, zorder=0); ax[0].axvline(0, color=GRID, lw=0.8, zorder=0)
    sT = S[(S.group == 'Together') & (S.metric == 'beyond_rho')].iloc[0]
    ax[0].set_xlabel("measured epistasis beyond the global curve, r (kcal/mol)")
    ax[0].set_ylabel("predicted, r′ (seed mean; kcal/mol)")
    ax[0].set_title(f"A  {int(sT.n_doubles):,} doubles, MGnify + Rocklin designs", loc='left', fontsize=9.5)
    ax[0].text(0.03, 0.96, f"beyond ρ = {sT.value:.3f}", transform=ax[0].transAxes, va='top', fontsize=9.5)
    cb = fig.colorbar(hb, ax=ax[0], fraction=0.045, pad=0.02); cb.set_label('doubles per cell (log)', color=MUTED); cb.outline.set_visible(False)
    # B: point values with both intervals
    ypos = {(g, m): k + (0.17 if m == 'naive_rho' else -0.17) for k, g in enumerate(GROUPS[::-1]) for m in ('beyond_rho', 'naive_rho')}
    for _, r in S.iterrows():
        y = ypos[(r.group, r.metric)]
        ax[1].plot([r.lo_scaf, r.hi_scaf], [y, y], color=COL[r.group], lw=1.0, alpha=0.55, solid_capstyle='butt')
        ax[1].plot([r.lo_mut, r.hi_mut], [y, y], color=COL[r.group], lw=3.2, solid_capstyle='butt')
        ax[1].plot([r.value], [y], 'o' if r.metric == 'beyond_rho' else 's', ms=7, mfc=COL[r.group] if r.metric == 'beyond_rho' else 'white', mec=COL[r.group], mew=1.6, zorder=3)
        ax[1].text(max(r.hi_scaf, r.hi_mut) + 0.006, y, f"{r.value:.3f}", va='center', fontsize=8.5, color=INK)
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_yticks(range(len(GROUPS))); ax[1].set_yticklabels([f"{g}\n(n = {META_N(S, g):,})" for g in GROUPS[::-1]])
    ax[1].set_xlabel('Spearman ρ against measured epistasis')
    ax[1].set_title('B  beyond ρ (●) and naive ρ (□)', loc='left', fontsize=9.5)
    ax[1].text(0.0, -0.22, 'thick: 95% mutation-level bootstrap; thin: 95% scaffold-level', transform=ax[1].transAxes, fontsize=8, color=MUTED)
    ax[1].grid(axis='x', color=GRID, lw=0.6); ax[1].set_axisbelow(True)
    # C: bootstrap distributions of beyond rho
    for g in GROUPS:
        v = bd[(bd.group == g) & (bd.kind == 'mutation')]['beyond_rho'].to_numpy()
        r = S[(S.group == g) & (S.metric == 'beyond_rho')].iloc[0]
        p = r.p_gt0_mut
        lab = f"{g}: p < {1 / (len(v) + 1):.3g}" if (v <= 0).sum() == 0 else f"{g}: p = {p:.3g}"
        ax[2].hist(v, bins=40, histtype='stepfilled' if g == 'Together' else 'step', color=COL[g], alpha=0.35 if g == 'Together' else 1.0, lw=1.6, label=lab)
    ax[2].axvline(0, color=INK, lw=1.0)
    ax[2].set_xlabel('beyond ρ of the resample (seed mean)'); ax[2].set_ylabel('bootstrap draws')
    ax[2].set_title('C  mutation-level bootstrap, H0: beyond ρ ≤ 0', loc='left', fontsize=9.5)
    ax[2].legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'beyond_rho_epistasis.{ext}'), dpi=200, bbox_inches='tight', facecolor='white')
    print('figure written')


def META_N(S, g):
    return int(S[S.group == g].n_doubles.iloc[0])


if __name__ == '__main__':
    main()
