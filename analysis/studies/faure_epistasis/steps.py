"""Modular steps for the Faure et al. (2022) & MegaScale epistasis geometry, SNR, and ``cell_mag`` ceiling study.

Refactored onto ``msr_analysis.{paths,structures,stats,style,epistasis,final_runs}`` and ``esm_msr.epi_metrics``:
  - Auto-builds ``cache/faure_doubles_<ds>.pkl`` atomically via ``paths.atomic_pickle_dump`` when missing.
  - Resamples triplicate ``FINAL_p0`` seeds jointly with blocks and positions at ``B >= 1000``.
  - Adds a paired Homology-Restricted MegaScale (`tsuboyama_capped_test`) vs Faure (GRB2/DLG4) epistasis SNR and
    ``cell_mag`` noise-ceiling comparison table & composite figure.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from esm_msr import epi_metrics as E
from msr_analysis import epistasis as EPI
from msr_analysis import stats as ST
from msr_analysis import style
from msr_analysis.final_runs import load_final_frame
from msr_analysis.paths import CACHE, PGYM_AF2_DIR, TASKS, atomic_pickle_dump, atomic_to_csv
from msr_analysis.structures import align_domain_residues, cb_distance_matrix, heavy_atom_distance_matrix
from msr_analysis.tobit import TOBIT

SETS = [
    'DLG4_HUMAN_Faure_2021_abundance_domain',
    'DLG4_HUMAN_Faure_2021_binding_domain',
    'GRB2_HUMAN_Faure_2021_abundance_domain',
    'GRB2_HUMAN_Faure_2021_binding_domain',
]
LABEL = {
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'PDZ3 (DLG4) abundance',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'PDZ3 (DLG4) binding',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'SH3 (GRB2) abundance',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'SH3 (GRB2) binding',
}
SHORT = {
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
}
COLOR = {
    'DLG4_HUMAN_Faure_2021_abundance_domain': '#2a78d6',
    'DLG4_HUMAN_Faure_2021_binding_domain': '#1baf7a',
    'GRB2_HUMAN_Faure_2021_abundance_domain': '#6250d6',
    'GRB2_HUMAN_Faure_2021_binding_domain': '#eb6834',
}
STRUCT_PDB = {
    'DLG4': PGYM_AF2_DIR / 'DLG4_HUMAN.pdb',
    'GRB2': PGYM_AF2_DIR / 'GRB2_HUMAN.pdb',
}

FLOOR_X = -0.5
BACKGROUND_MIN_DOUBLES = 300
BINS = [0, 4, 5, 8, 12, 16, 99]
BIN_LABEL = ['< 4', '4–5', '5–8', '8–12', '12–16', '> 16']
DBINS = [(0, 5, 'contact\n< 5 Å'), (5, 8, '5–8 Å'), (8, 12, '8–12 Å'), (12, 99, '> 12 Å')]
R_REP = (0.87, 0.92)


def load_faure_set(ds: str) -> tuple[dict[str, tuple[float, float]], pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(TASKS / f'{ds}.csv.gz', low_memory=False).drop_duplicates('id')
    s = d[d['NMUTS'] == 1]
    single = {str(m): (float(f), float(e)) for m, f, e in zip(s['mutation1'], s['truth'], s['sigma'])}
    t = d[d['NMUTS'] == 2].copy()
    t['fA'] = [single.get(str(m), (np.nan, np.nan))[0] for m in t['mutation1']]
    t['sA'] = [single.get(str(m), (np.nan, np.nan))[1] for m in t['mutation1']]
    t['fB'] = [single.get(str(m), (np.nan, np.nan))[0] for m in t['mutation2']]
    t['sB'] = [single.get(str(m), (np.nan, np.nan))[1] for m in t['mutation2']]
    out = pd.DataFrame(dict(
        id=t['id'].astype(str).to_numpy(),
        mut1=t['mutation1'].astype(str).to_numpy(),
        mut2=t['mutation2'].astype(str).to_numpy(),
        pos1=t['pos1'].astype(int).to_numpy(),
        pos2=t['pos2'].astype(int).to_numpy(),
        aa1=t['to1'].astype(str).to_numpy(),
        aa2=t['to2'].astype(str).to_numpy(),
        wt1=t['fr1'].astype(str).to_numpy(),
        wt2=t['fr2'].astype(str).to_numpy(),
        f=t['truth'].to_numpy(float),
        s=t['sigma'].to_numpy(float),
        fA=t['fA'].to_numpy(float),
        sA=t['sA'].to_numpy(float),
        fB=t['fB'].to_numpy(float),
        sB=t['sB'].to_numpy(float),
    ))
    out['n_all_doubles'] = len(out)
    out = out[np.isfinite(out[['f', 's', 'fA', 'sA', 'fB', 'sB']]).all(axis=1)].reset_index(drop=True)
    out['x'] = out['fA'] + out['fB']
    out['eps'] = out['f'] - out['x']
    out['sig_eps'] = np.sqrt(out['s'] ** 2 + out['sA'] ** 2 + out['sB'] ** 2)
    return single, out, d


def get_or_build_doubles_with_distances(ds: str, refresh: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    prot = ds.split('_')[0]
    single, dbl, raw = load_faure_set(ds)
    dom_res, _ = align_domain_residues(STRUCT_PDB[prot], str(raw['ref_seq'].iloc[0]))
    L = max(dom_res)
    cb = cb_distance_matrix(dom_res, one_based_size=L)
    mn = heavy_atom_distance_matrix(dom_res, one_based_size=L)

    pkl_path = CACHE / f'faure_doubles_{ds}.pkl'
    if pkl_path.exists() and not refresh:
        with open(pkl_path, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, pd.DataFrame) and {'dmin', 'dcb', 'sep'} <= set(cached.columns) and len(cached) == len(dbl):
            return single, cached, raw, cb, mn

    dbl = dbl.copy()
    dbl['dmin'] = mn[dbl['pos1'].to_numpy(), dbl['pos2'].to_numpy()]
    dbl['dcb'] = cb[dbl['pos1'].to_numpy(), dbl['pos2'].to_numpy()]
    dbl['sep'] = (dbl['pos2'] - dbl['pos1']).abs()
    atomic_pickle_dump(dbl, pkl_path)
    return single, dbl, raw, cb, mn


# ----------------------------------------------------------------------------------------------
# Step 1: Structural proximity & pair-sampling geometry
# ----------------------------------------------------------------------------------------------
def run_proximity(out_dir: Path, refresh: bool = False) -> dict:
    rows, pairs, store = [], [], {}

    def _summary_row(name, dist_min, dist_cb, sep, extra=None):
        r = dict(
            name=name, n=len(dist_min),
            min_p10=float(np.percentile(dist_min, 10)), min_p25=float(np.percentile(dist_min, 25)),
            min_median=float(np.median(dist_min)), min_p75=float(np.percentile(dist_min, 75)),
            min_p90=float(np.percentile(dist_min, 90)), cb_median=float(np.median(dist_cb)),
            sep_median=float(np.median(sep)),
            contact_lt5=float((dist_min < 5).mean()), lt8=float((dist_min < 8).mean()), lt12=float((dist_min < 12).mean()),
        )
        for lo, hi, lab in zip(BINS[:-1], BINS[1:], BIN_LABEL):
            r[f'bin_{lab}'] = float(((dist_min >= lo) & (dist_min < hi)).mean())
        r.update(extra or {})
        return r

    for ds in SETS:
        single, dbl, _raw, cb, mn = get_or_build_doubles_with_distances(ds, refresh=refresh)
        pc = dbl.groupby(['pos1', 'pos2']).size().rename('n_doubles').reset_index()
        pc['dmin'] = mn[pc['pos1'].to_numpy(), pc['pos2'].to_numpy()]
        pc['dcb'] = cb[pc['pos1'].to_numpy(), pc['pos2'].to_numpy()]
        pc.insert(0, 'set', ds)
        pairs.append(pc)
        sp = sorted({int(m[1:-1]) for m in single})
        allp = [(i, j) for a, i in enumerate(sp) for j in sp[a + 1:]]
        ai, aj = np.array([p[0] for p in allp]), np.array([p[1] for p in allp])
        rows.append(_summary_row(f'{ds}|doubles', dbl['dmin'].to_numpy(), dbl['dcb'].to_numpy(), dbl['sep'].to_numpy(), dict(set=ds, kind='doubles', n_pairs=len(pc), doubles_per_pair_median=float(pc['n_doubles'].median()))))
        rows.append(_summary_row(f'{ds}|pairs', pc['dmin'].to_numpy(), pc['dcb'].to_numpy(), (pc['pos2'] - pc['pos1']).to_numpy(), dict(set=ds, kind='observed position pairs', n_pairs=len(pc))))
        rows.append(_summary_row(f'{ds}|all pairs', mn[ai, aj], cb[ai, aj], aj - ai, dict(set=ds, kind='all pairs of measured positions', n_pairs=len(allp))))
        store[ds] = dict(dbl=dbl, pc=pc, allmin=mn[ai, aj], allsep=aj - ai)
        if ds.startswith('DLG4'):
            cnt = pd.concat([dbl['pos1'], dbl['pos2']]).value_counts()
            bg = sorted(cnt[cnt >= BACKGROUND_MIN_DOUBLES].index)
            bp = [(i, j) for (i, j) in allp if i in bg or j in bg]
            bi, bj = np.array([p[0] for p in bp]), np.array([p[1] for p in bp])
            rows.append(_summary_row(f'{ds}|background pairs', mn[bi, bj], cb[bi, bj], bj - bi, dict(set=ds, kind=f'pairs with background site {bg}', n_pairs=len(bp))))
            store[ds]['bgmin'], store[ds]['bg'] = mn[bi, bj], bg

    S = pd.DataFrame(rows)
    atomic_to_csv(S, out_dir / 'proximity_summary.csv', index=False, float_format='%.4g')
    atomic_to_csv(pd.concat(pairs, ignore_index=True), out_dir / 'pair_table.csv', index=False, float_format='%.4g')

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    ax = axes[0, 0]
    for ds in SETS:
        v = np.sort(store[ds]['dbl']['dmin'].to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLOR[ds], lw=1.8, label=LABEL[ds])
    for arr_v, ls_v, col_v, lbl_v in [
        (store[SETS[0]]['allmin'], '--', '#888888', 'All pairs of PDZ3 positions'),
        (store[SETS[0]]['bgmin'], ':', '#444444', 'PDZ3 pairs with background site'),
        (store[SETS[2]]['allmin'], '--', '#b5b5b5', 'All pairs of SH3 positions'),
    ]:
        vs = np.sort(arr_v)
        ax.plot(vs, np.arange(1, len(vs) + 1) / len(vs), color=col_v, ls=ls_v, lw=1.2, label=lbl_v)
    ax.axvline(5, color='#888888', lw=0.9, ls=':')
    ax.set_xlim(0, 30)
    ax.grid(axis='both', color=style.GRID, lw=0.6)
    ax.set_xlabel('minimal heavy-atom distance (Å)')
    ax.set_ylabel('cumulative share of doubles')
    ax.set_title('a   Distance of mutated residue pairs', loc='left')
    ax.legend(fontsize=7.3, loc='lower right')

    ax = axes[0, 1]
    w = 0.20
    for k, ds in enumerate(SETS):
        r = S[S['name'] == f'{ds}|doubles'].iloc[0]
        ax.bar(np.arange(len(BIN_LABEL)) + (k - 1.5) * w, [r[f'bin_{l}'] for l in BIN_LABEL], width=w, color=COLOR[ds], label=SHORT[ds])
    ax.set_xticks(range(len(BIN_LABEL)))
    ax.set_xticklabels(BIN_LABEL)
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_xlabel('minimal heavy-atom distance (Å)')
    ax.set_ylabel('share of doubles')
    ax.set_title('b   Doubles by distance bin', loc='left')
    ax.legend(fontsize=7.6)

    ax = axes[1, 0]
    for ds in SETS:
        vs = np.sort(store[ds]['dbl']['sep'].to_numpy())
        ax.plot(vs, np.arange(1, len(vs) + 1) / len(vs), color=COLOR[ds], lw=1.8, label=SHORT[ds])
    ax.set_xscale('log')
    ax.grid(axis='both', color=style.GRID, lw=0.6)
    ax.set_xlabel('sequence separation |i − j| (residues)')
    ax.set_ylabel('cumulative share of doubles')
    ax.set_title('c   Sequence separation', loc='left')

    ax = axes[1, 1]
    for ds in SETS:
        c = store[ds]['pc']['n_doubles']
        bins_g = np.unique(np.round(np.geomspace(1, c.max() + 1, 28)).astype(int))
        ax.hist(c, bins=bins_g, histtype='step', color=COLOR[ds], lw=1.8, label=f'{SHORT[ds]} (median {int(c.median())})')
    ax.set_xscale('log')
    ax.grid(axis='both', color=style.GRID, lw=0.6)
    ax.set_xlabel('doubles per position pair')
    ax.set_ylabel('position pairs')
    ax.set_title('d   Doubles per position pair', loc='left')
    ax.legend(fontsize=7.6)

    n_counts = [len(store[d]['dbl']) for d in SETS]
    n_bg = len(store[SETS[0]]['bg'])
    style.finish_figure(
        fig,
        out_dir / 'proximity_distribution',
        title='Faure et al. (2022) double-mutant structural proximity and sampling design',
        note=(
            f'Doubles with both singles measured (PDZ3 abundance {n_counts[0]:,}, binding {n_counts[1]:,}; '
            f'SH3 abundance {n_counts[2]:,}, binding {n_counts[3]:,}). AlphaFold2 heavy-atom distances (< 5 Å = direct contact). '
            f'PDZ3 (DLG4) samples random substitutions on {n_bg} single-mutant backgrounds (1 residue on one side of each pair -> zero ≥3×3 blocks).'
        ),
    )
    return store


# ----------------------------------------------------------------------------------------------
# Step 2: Epistasis variance decomposition & SNR by distance
# ----------------------------------------------------------------------------------------------
def run_snr(out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    rows, byd, store = [], [], {}

    def _vstats(v, noise):
        sig = float(v.var() - noise)
        v_tot = float(v.var())
        return dict(
            var=v_tot, noise=noise, signal=sig, snr=sig / noise,
            reliability=sig / v_tot, ceiling=float(np.sqrt(max(sig, 0.0) / v_tot)),
        )

    for ds in SETS:
        _, d0, raw, _, _ = get_or_build_doubles_with_distances(ds, refresh=False)
        d0 = d0.copy()
        d0['g'] = EPI.smooth_on(d0['x'].to_numpy(), d0['eps'].to_numpy())
        d0['r'] = d0['eps'] - d0['g']
        d0['z'] = d0['r'] / d0['sig_eps']
        allv = raw[raw['NMUTS'] <= 2]
        vt = float(allv['truth'].var() - (allv['sigma'] ** 2).mean())
        r_rep = float(vt / (vt + 3.0 * (allv['sigma'] ** 2).mean()))
        store[ds] = d0

        for stratum, m in (('all', np.ones(len(d0), dtype=bool)), ('above floor', (d0['x'] > FLOOR_X).to_numpy())):
            d = d0[m]
            noise = float((d['sig_eps'] ** 2).mean())
            a, b = _vstats(d['eps'], noise), _vstats(d['r'], noise)
            rec = dict(
                set=ds, stratum=stratum, n=len(d),
                eps_sd=float(d['eps'].std()), noise_sd=float(np.sqrt(noise)),
                global_share=float(1.0 - d['r'].var() / d['eps'].var()),
                implied_replicate_r=r_rep, fitness_sd=float(allv['truth'].std()),
                median_sigma=float(allv['sigma'].median()),
                z2=float((d['z'].abs() > 2).mean()), z3=float((d['z'].abs() > 3).mean()), z_sd=float(d['z'].std()),
            )
            rec.update({f'raw_{k}': v for k, v in a.items()})
            rec.update({f'resid_{k}': v for k, v in b.items()})
            for Rr in R_REP:
                n_cal = vt * (1.0 - Rr) / Rr
                rec[f'raw_snr_R{Rr}'] = float((d['eps'].var() - n_cal) / n_cal)
                rec[f'resid_snr_R{Rr}'] = float((d['r'].var() - n_cal) / n_cal)
                rec[f'resid_ceiling_R{Rr}'] = float(np.sqrt(max(float(d['r'].var()) - n_cal, 0.0) / float(d['r'].var())))
            rows.append(rec)
            for lo, hi, lab in DBINS:
                sub = d[(d['dmin'] >= lo) & (d['dmin'] < hi)]
                if len(sub) < 30:
                    continue
                nz = float((sub['sig_eps'] ** 2).mean())
                byd.append(dict(
                    set=ds, stratum=stratum, bin=lab.replace('\n', ' '), n=len(sub),
                    resid_sd=float(sub['r'].std()), noise_sd=float(np.sqrt(nz)),
                    signal_sd=float(np.sqrt(max(float(sub['r'].var()) - nz, 0.0))),
                    snr=float((sub['r'].var() - nz) / nz),
                    z2=float((sub['z'].abs() > 2).mean()), z3=float((sub['z'].abs() > 3).mean()),
                ))

    S, Bd = pd.DataFrame(rows), pd.DataFrame(byd)
    atomic_to_csv(S, out_dir / 'snr_summary.csv', index=False, float_format='%.4g')
    atomic_to_csv(Bd, out_dir / 'snr_by_distance.csv', index=False, float_format='%.4g')

    Bda = Bd[Bd['stratum'] == 'above floor']
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    ax = axes[0, 0]
    x0 = np.arange(len(SETS))
    for k, (stratum, off) in enumerate((('all', -0.2), ('above floor', 0.2))):
        T = S[S['stratum'] == stratum].set_index('set').loc[SETS]
        tot = T['raw_var'].to_numpy()
        glob = (T['global_share'] * T['raw_var']).to_numpy()
        spec = T['resid_signal'].clip(lower=0).to_numpy()
        nz = np.minimum(T['raw_noise'], T['resid_var']).to_numpy()
        xs = x0 + off
        ax.bar(xs, glob / tot, width=0.36, color='#9aa5b1', label='Global saturation f(f_A + f_B)' if k == 0 else None)
        ax.bar(xs, spec / tot, width=0.36, bottom=glob / tot, color='#eb6834', label='Specific epistasis (signal)' if k == 0 else None)
        ax.bar(xs, nz / tot, width=0.36, bottom=(glob + spec) / tot, color='#444444', label='Measurement noise' if k == 0 else None)
        for xi, g, s_, n_ in zip(xs, glob / tot, spec / tot, nz / tot):
            if g > 0.04:
                ax.text(xi, g / 2, f'{g:.0%}', ha='center', va='center', fontsize=7.4, color='white', fontweight='bold')
            if s_ > 0.04:
                ax.text(xi, g + s_ / 2, f'{s_:.0%}', ha='center', va='center', fontsize=7.4, fontweight='bold')
            if n_ > 0.04:
                ax.text(xi, g + s_ + n_ / 2, f'{n_:.0%}', ha='center', va='center', fontsize=7.4, color='white', fontweight='bold')
        for xi in xs:
            ax.text(xi, 1.015, 'all' if stratum == 'all' else 'above\nfloor', ha='center', va='bottom', fontsize=6.6, color='#555555')
    ax.set_xticks(x0)
    ax.set_xticklabels([SHORT[d] for d in SETS])
    ax.set_ylim(0, 1.38)
    ax.set_ylabel('share of raw epistasis variance Var(ε)')
    ax.set_title('a   Variance decomposition of measured epistasis ε', loc='left')
    ax.legend(fontsize=7.1, loc='upper right', ncol=1, frameon=True, facecolor='white', edgecolor='#dddddd', bbox_to_anchor=(0.99, 0.995))

    ax = axes[0, 1]
    for ds in SETS:
        t = Bda[Bda['set'] == ds]
        ax.plot(range(len(t)), t['z3'] * 100, 'o-', color=COLOR[ds], label=SHORT[ds], ms=5)
    ax.axhline(2 * (1 - norm.cdf(3)) * 100, color='#444444', ls='--', lw=1.0)
    ax.set_xticks(range(len(DBINS)))
    ax.set_xticklabels([b[2] for b in DBINS])
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_ylabel('% of doubles with |r| / σ_ε > 3')
    ax.set_title('b   Detectable specific epistasis by distance (above floor)', loc='left')
    ax.legend(fontsize=7.6, loc='upper right', frameon=True, facecolor='white', edgecolor='#dddddd')

    ax = axes[1, 0]
    for ds in SETS:
        t = Bda[Bda['set'] == ds]
        ax.plot(range(len(t)), t['signal_sd'], 'o-', color=COLOR[ds], label=SHORT[ds], ms=5)
        ax.plot(range(len(t)), t['noise_sd'], ':', color=COLOR[ds], lw=1.4)
    ax.set_xticks(range(len(DBINS)))
    ax.set_xticklabels([b[2] for b in DBINS])
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_ylabel('SD of beyond-global residual r (fitness units)')
    ax.set_title('c   Signal SD (solid) vs noise SD (dotted), above floor', loc='left')
    ax.legend(fontsize=7.6)

    ax = axes[1, 1]
    d_sh3 = store[SETS[3]]
    d_sh3 = d_sh3[d_sh3['x'] > FLOOR_X]
    grid = np.linspace(-6, 6, 61)
    for lo, hi, lab, col in ((0, 5, 'Contact (< 5 Å)', '#d35400'), (12, 99, 'Distant (> 12 Å)', '#2a78d6')):
        m = (d_sh3['dmin'] >= lo) & (d_sh3['dmin'] < hi)
        h, edges = np.histogram(d_sh3['z'][m].clip(-6, 6), bins=grid, density=True)
        ax.plot((edges[:-1] + edges[1:]) / 2, h, color=col, lw=1.6, label=f'{lab} (n = {int(m.sum()):,})')
    ax.plot(grid, norm.pdf(grid), color='#444444', ls='--', lw=1.2, label='Gaussian noise N(0, 1)')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.0)
    ax.grid(axis='both', color=style.GRID, lw=0.6)
    ax.set_xlabel('z = r / σ_ε')
    ax.set_ylabel('probability density')
    ax.set_title('d   Standardized residual z on SH3 (GRB2) binding', loc='left')
    ax.legend(fontsize=7.6)

    style.finish_figure(
        fig,
        out_dir / 'epistasis_snr',
        title='Signal-to-noise decomposition of measured epistasis across Faure et al. (2022) PCA assays',
        note=(
            'ε = f_AB − f_A − f_B; noise variance = mean(σ_AB² + σ_A² + σ_B²) from DiMSum merged-triplicate errors. '
            'Global saturation = 25-quantile-bin conditional expectation E[ε | f_A + f_B]; specific epistasis = residual r = ε − E[ε | x].'
        ),
    )
    return S


# ----------------------------------------------------------------------------------------------
# Step 3 & 4: `cell_mag` noise ceiling, joint [block/pos × 3-seed] significance & MegaScale comparison
# ----------------------------------------------------------------------------------------------
def _dc(M: np.ndarray) -> np.ndarray:
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def _dc_var(S2: np.ndarray) -> np.ndarray:
    R, C = S2.shape
    Pr, Pc = np.eye(R) - 1.0 / R, np.eye(C) - 1.0 / C
    return (Pr ** 2) @ S2 @ (Pc ** 2).T


def _fast_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def blocks_of(d: pd.DataFrame, group_cols: list[str] = ['pos1', 'pos2']) -> list[tuple]:
    out = []
    for key, g in d.groupby(group_cols):
        if len(g) < E.MIN_PAIR_CELLS:
            continue
        ra, ca = sorted(g['aa1'].unique()), sorted(g['aa2'].unique())
        ri = {a: i for i, a in enumerate(ra)}
        ci_m = {b: j for j, b in enumerate(ca)}
        M_mat = np.full((len(ra), len(ca)), np.nan)
        I_mat = np.full((len(ra), len(ca)), -1, dtype=int)
        for ix, a, b, r in zip(g.index, g['aa1'], g['aa2'], g['r']):
            M_mat[ri[a], ci_m[b]] = r
            I_mat[ri[a], ci_m[b]] = ix
        blk = E._complete_block(M_mat)
        if blk is None:
            continue
        sub = I_mat[np.ix_(*blk)]
        out.append((sub, float(d['x'].to_numpy()[sub.ravel()].mean()), key))
    return out


def simulate_block_ceiling(r_arr: np.ndarray, blocks: list, s2_per_row: np.ndarray, rng: np.random.Generator, nsim: int = 30) -> tuple[float, float, float, float]:
    C_mats = [_dc(r_arr[b[0]]) for b in blocks]
    S2_mats = [s2_per_row[b[0]] for b in blocks]
    V_mats = [_dc_var(s2) for s2 in S2_mats]
    c2 = float(np.mean(np.concatenate([c.ravel() ** 2 for c in C_mats])))
    nv = float(np.mean(np.concatenate([v.ravel() for v in V_mats])))
    sig = max(c2 - nv, 0.0)
    kappa = np.sqrt(sig / c2) if c2 > 0 else 0.0
    sims = []
    for _ in range(nsim):
        rh = []
        for c, s2 in zip(C_mats, S2_mats):
            T = kappa * c
            meas = T + _dc(rng.normal(0, np.sqrt(s2)))
            rh.append(_fast_rho(T.ravel(), meas.ravel()) if sig > 0 else 0.0)
        sims.append(float(np.nanmean(rh)))
    pearson_ceil = float(np.sqrt(sig / (sig + nv))) if (sig + nv) > 0 and sig > 0 else 0.0
    return float(np.mean(sims)), pearson_ceil, float(np.sqrt(sig)), float(np.sqrt(nv))


def _load_faure_p0_residuals(ds: str, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fr = load_final_frame(ds, families=['FINAL_p0'])
    pos_map = {k: i for i, k in enumerate(fr.spine['id'].astype(str))}
    ix = np.array([pos_map.get(i, -1) for i in d['id']])
    ok = ix >= 0
    res, epis = [], []
    for s in range(fr.preds['FINAL_p0:epi'].shape[0]):
        epi = fr.preds['FINAL_p0:epi'][s][np.where(ok, ix, 0)]
        add = fr.preds['FINAL_p0:wt+'][s][np.where(ok, ix, 0)]
        full = np.full(len(d), np.nan)
        good = ok & np.isfinite(epi) & np.isfinite(add)
        full[good] = epi[good] - EPI.smooth_on(add[good], epi[good])
        res.append(full)
        epis.append(np.where(good, epi, np.nan))
    return np.asarray(res), np.asarray(epis)


def _compute_megascale_comparison(rng: np.random.Generator, B: int) -> list[dict]:
    """Compute identical epistasis SNR, noise ceiling, and FINAL_p0 ``cell_mag`` on Homology-Restricted MegaScale test doubles."""
    raw = pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', low_memory=False)
    fr = load_final_frame('tsuboyama_capped_test', families=['FINAL_p0'])
    sp = fr.spine
    # Measured WT-backbone singles & doubles
    raw_wt = raw[(raw['cens'] == 0) & (raw['mut_structure'] == '-')].copy()
    raw_wt['sigma_ci'] = (raw_wt['deltaG_95CI'].astype(float) / 3.92).fillna(TOBIT['sigma_y'])
    s_df = raw_wt[raw_wt['mut_type'].astype(str).str.count(':') == 0]
    s_lookup = {
        (str(c), str(m)): (float(ddg), float(sg))
        for c, m, ddg, sg in zip(s_df['code'], s_df['mut_type'], s_df['ddG_ML'], s_df['sigma_ci'])
    }

    d_df = raw_wt[raw_wt['mut_type'].astype(str).str.count(':') == 1].copy()
    parts = d_df['mut_type'].astype(str).str.split(':', expand=True)
    d_df['m1'], d_df['m2'] = parts[0], parts[1]
    d_df['fA'] = [s_lookup.get((str(c), str(m)), (np.nan, np.nan))[0] for c, m in zip(d_df['code'], d_df['m1'])]
    d_df['sA'] = [s_lookup.get((str(c), str(m)), (np.nan, np.nan))[1] for c, m in zip(d_df['code'], d_df['m1'])]
    d_df['fB'] = [s_lookup.get((str(c), str(m)), (np.nan, np.nan))[0] for c, m in zip(d_df['code'], d_df['m2'])]
    d_df['sB'] = [s_lookup.get((str(c), str(m)), (np.nan, np.nan))[1] for c, m in zip(d_df['code'], d_df['m2'])]
    d_df = d_df[np.isfinite(d_df[['ddG_ML', 'fA', 'fB', 'dG_ML']]).all(axis=1)].reset_index(drop=True)

    d_df['pos1'] = d_df['m1'].str[1:-1].astype(int)
    d_df['pos2'] = d_df['m2'].str[1:-1].astype(int)
    d_df['aa1'] = d_df['m1'].str[-1]
    d_df['aa2'] = d_df['m2'].str[-1]
    # Add dG_wt so x = dG_wt + ddG_A + ddG_B matches epi_metrics folding-stability floor (> 0 kcal/mol = folded)
    dG_wt = (d_df['dG_ML'] - d_df['ddG_ML']).to_numpy(float)
    d_df['x'] = dG_wt + d_df['fA'].to_numpy(float) + d_df['fB'].to_numpy(float)
    d_df['eps'] = d_df['dddG_ML'].to_numpy(float)
    d_df['r'] = d_df['eps'] - EPI.smooth_on(d_df['x'].to_numpy(), d_df['eps'].to_numpy())

    pos_sp = {str(k): i for i, k in enumerate(sp['id'].astype(str))}
    ix = np.array([pos_sp.get(str(i), -1) for i in d_df['id']])
    ok = ix >= 0
    P_seeds, E_seeds = [], []
    for sd in range(fr.preds['FINAL_p0:epi'].shape[0]):
        epi = fr.preds['FINAL_p0:epi'][sd][np.where(ok, ix, 0)]
        wt_p = fr.preds['FINAL_p0:wt+'][sd][np.where(ok, ix, 0)]
        x_pred = dG_wt + wt_p
        good = ok & np.isfinite(epi) & np.isfinite(x_pred)
        full = np.full(len(d_df), np.nan)
        full[good] = epi[good] - EPI.smooth_on(x_pred[good], epi[good])
        P_seeds.append(full)
        E_seeds.append(np.where(good, epi, np.nan))
    P_seeds = np.asarray(P_seeds)

    # Direct empirical split-protease reliability: Spearman(dc(r_trypsin), dc(r_chymotrypsin)) within each >=3x3 block,
    # stepped up via Spearman-Brown sqrt(2 * r_tc / (1 + r_tc)) for the two-protease ML composite.
    d_df['rt'] = d_df['deltaG_t'].to_numpy(float) - EPI.smooth_on(d_df['x'].to_numpy(), np.nan_to_num(d_df['deltaG_t'].to_numpy(float), nan=0.0))
    d_df['rc'] = d_df['deltaG_c'].to_numpy(float) - EPI.smooth_on(d_df['x'].to_numpy(), np.nan_to_num(d_df['deltaG_c'].to_numpy(float), nan=0.0))

    def _protease_split_ceiling(blocks_sel):
        rhos_tc, pears_tc = [], []
        for b in blocks_sel:
            sub = b[0]
            mt = d_df['deltaG_t'].to_numpy(float)[sub]
            mc = d_df['deltaG_c'].to_numpy(float)[sub]
            if np.isfinite(mt).all() and np.isfinite(mc).all():
                bt, bc = _dc(mt).ravel(), _dc(mc).ravel()
                rhos_tc.append(_fast_rho(bt, bc))
                den = float(np.sqrt((bt ** 2).sum() * (bc ** 2).sum()))
                if den > 0:
                    pears_tc.append(float((bt * bc).sum() / den))
        r_sp = max(float(np.nanmean(rhos_tc)), 0.0) if rhos_tc else 0.0
        r_pe = max(float(np.nanmean(pears_tc)), 0.0) if pears_tc else 0.0
        return float(np.sqrt(2.0 * r_sp / (1.0 + r_sp))), float(np.sqrt(2.0 * r_pe / (1.0 + r_pe)))

    bl_all = blocks_of(d_df, group_cols=['code', 'pos1', 'pos2'])
    bl_abv = [b for b in bl_all if b[1] > 0.0]
    out_rows = []
    for stratum, sel in (('all blocks', bl_all), ('blocks above floor', bl_abv)):
        per_seed_blk = np.array([
            [_fast_rho(_dc(p[b[0]]).ravel(), _dc(d_df['r'].to_numpy()[b[0]]).ravel()) for b in sel]
            for p in P_seeds
        ])
        pt_cm, reps_cm = ST.boot_domain_seeds(per_seed_blk.T, B=B, seed=21)
        lo_cm, hi_cm = np.nanquantile(reps_cm, [0.025, 0.975])
        cs_ci, pc_ci, ssd_ci, nsd_ci = simulate_block_ceiling(d_df['r'].to_numpy(), sel, d_df['sigma_ci'].to_numpy(float) ** 2, rng)
        cs_tc, pc_tc = _protease_split_ceiling(sel)
        for n_name, cs, pc, ssd, nsd in (
            ('cDNA 95%CI posterior SD', cs_ci, pc_ci, ssd_ci, nsd_ci),
            ('Trypsin–chymotrypsin split ceiling', cs_tc, pc_tc, ssd_ci, np.nan),
        ):
            out_rows.append(dict(
                dataset='tsuboyama_capped_test',
                label='MegaScale test (13 domains)',
                stratum=stratum,
                noise_model=n_name,
                doubles=len(d_df),
                blocks=len(sel),
                cells=int(sum(b[0].size for b in sel)),
                ceiling_spearman=cs,
                ceiling_pearson=pc,
                signal_sd=ssd,
                noise_sd=nsd,
                model_cell_mag=pt_cm,
                model_cell_mag_lo=float(lo_cm),
                model_cell_mag_hi=float(hi_cm),
                fraction_of_ceiling=(pt_cm / cs) if cs > 0 else np.nan,
            ))
    return out_rows


def run_ceiling_and_significance(out_dir: Path, B: int = 1000, nperm: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    c_rows, b_rows, sig_rows = [], [], []

    for ds in SETS:
        _, d, raw = load_faure_set(ds)
        d['r'] = d['eps'] - EPI.smooth_on(d['x'].to_numpy(), d['eps'].to_numpy())
        allv = raw[raw['NMUTS'] <= 2]
        var_f = float(allv['truth'].var() - (allv['sigma'] ** 2).mean())
        bl = blocks_of(d, group_cols=['pos1', 'pos2'])
        for b in bl:
            b_rows.append(dict(set=ds, pos1=b[2][0], pos2=b[2][1], rows=b[0].shape[0], cols=b[0].shape[1], mean_x=b[1]))

        P, epis = _load_faure_p0_residuals(ds, d)
        base = dict(
            set=ds, doubles=len(d),
            pairs_ge20=int((d.groupby(['pos1', 'pos2']).size() >= E.MIN_PAIR_CELLS).sum()),
            model_beyond_rho=float(np.mean([spearmanr(p[np.isfinite(p)], d['r'].to_numpy()[np.isfinite(p)])[0] for p in P])),
            model_naive_rho=float(np.mean([spearmanr(e[np.isfinite(e)], d['eps'].to_numpy()[np.isfinite(e)])[0] for e in epis])),
        )
        if not bl:
            c_rows.append(dict(base, stratum='no complete block', blocks=0, cells=0))
            continue

        for stratum, sel in (('all blocks', bl), ('blocks above floor', [b for b in bl if b[1] > FLOOR_X])):
            if len(sel) < 5:
                continue
            keep = [b for b in sel if all(np.isfinite(p[b[0]]).all() for p in P)]
            idx = [b[0] for b in keep]
            meas = [_dc(d['r'].to_numpy()[i]).ravel() for i in idx]
            per_seed = np.array([[_fast_rho(_dc(p[i]).ravel(), m) for i, m in zip(idx, meas)] for p in P])  # (3, n_blocks)
            block_means = per_seed.mean(axis=0)
            mc = float(block_means.mean())

            for n_mod in ('nominal', f'R={R_REP[0]}', f'R={R_REP[1]}'):
                if n_mod == 'nominal':
                    s2_vec = d['s'].to_numpy(float) ** 2
                else:
                    R_val = float(n_mod.split('=')[1])
                    s2_vec = np.full(len(d), var_f * (1.0 - R_val) / (3.0 * R_val))
                cs, pc, ssd, nsd = simulate_block_ceiling(d['r'].to_numpy(), keep, s2_vec, rng)
                c_rows.append(dict(
                    base, stratum=stratum, noise_model=n_mod, blocks=len(keep),
                    cells=int(sum(i.size for i in idx)),
                    ceiling_spearman=cs, ceiling_pearson=pc, signal_sd=ssd, noise_sd=nsd, model_cell_mag=mc,
                ))

            # Joint [block × 3-seed] bootstrap AND joint [position × 3-seed] bootstrap + permutation null
            rng_sig = np.random.default_rng(11)
            _, bb = ST.boot_domain_seeds(per_seed.T, B=B, seed=11)

            pos_u = sorted({int(x) for b in keep for x in b[2]})
            pidx = {p_v: k for k, p_v in enumerate(pos_u)}
            b1 = np.array([pidx[b[2][0]] for b in keep])
            b2 = np.array([pidx[b[2][1]] for b in keep])
            sw_pos = rng_sig.multinomial(3, [1 / 3] * 3, size=B).astype(float) / 3.0
            pb = []
            for bi in range(B):
                cnt = np.bincount(rng_sig.integers(0, len(pos_u), len(pos_u)), minlength=len(pos_u))
                w = cnt[b1] * cnt[b2]
                blk_b = sw_pos[bi] @ per_seed
                pb.append(float((w * blk_b).sum() / w.sum()) if w.sum() > 0 else np.nan)
            pb = np.asarray(pb, dtype=float)

            null = []
            for _ in range(nperm):
                vals = []
                for i, m in zip(idx, meas):
                    perm = rng_sig.permutation(i.size).reshape(i.shape)
                    vals.append(np.mean([_fast_rho(_dc(p[i].ravel()[perm.ravel()].reshape(i.shape)).ravel(), m) for p in P]))
                null.append(float(np.mean(vals)))
            null_arr = np.asarray(null, dtype=float)

            sig_rows.append(dict(
                set=ds, stratum=stratum, blocks=len(keep), cells=int(sum(i.size for i in idx)),
                cell_mag=mc, seed_values='/'.join(f'{x:.3f}' for x in per_seed.mean(axis=1)),
                block_boot_lo=float(np.nanquantile(bb, 0.025)), block_boot_hi=float(np.nanquantile(bb, 0.975)),
                block_boot_p=ST.boot_pvalue(bb, null=0.0, alternative='two-sided'),
                pos_boot_lo=float(np.nanquantile(pb, 0.025)), pos_boot_hi=float(np.nanquantile(pb, 0.975)),
                pos_boot_p=ST.boot_pvalue(pb, null=0.0, alternative='two-sided'),
                null_mean=float(null_arr.mean()), null_sd=float(null_arr.std(ddof=1)),
                perm_p=float(max((np.sum(null_arr >= mc) + 1) / (nperm + 1), 1.0 / (nperm + 1))),
                share_blocks_positive=float((block_means > 0).mean()),
                B=B,
            ))

    C_df = pd.DataFrame(c_rows)
    Bk_df = pd.DataFrame(b_rows)
    Sig_df = pd.DataFrame(sig_rows)
    atomic_to_csv(C_df, out_dir / 'cell_mag_ceiling.csv', index=False, float_format='%.4g')
    atomic_to_csv(Bk_df, out_dir / 'cell_mag_blocks.csv', index=False, float_format='%.4g')
    atomic_to_csv(Sig_df, out_dir / 'cell_mag_significance.csv', index=False, float_format='%.4g')

    # Paired MegaScale vs Faure ceiling comparison table & 3-panel composite figure
    mega_rows = _compute_megascale_comparison(rng, B=B)
    faure_cmp = []
    for r in C_df[C_df['noise_model'].notna()].itertuples():
        hit_s = Sig_df[(Sig_df['set'] == r.set) & (Sig_df['stratum'] == r.stratum)].iloc[0]
        faure_cmp.append(dict(
            dataset=r.set, label=SHORT[r.set], stratum=r.stratum, noise_model=r.noise_model,
            doubles=int(r.doubles), blocks=int(r.blocks), cells=int(r.cells),
            ceiling_spearman=float(r.ceiling_spearman), ceiling_pearson=float(r.ceiling_pearson),
            signal_sd=float(r.signal_sd), noise_sd=float(r.noise_sd),
            model_cell_mag=float(r.model_cell_mag),
            model_cell_mag_lo=float(hit_s['pos_boot_lo']), model_cell_mag_hi=float(hit_s['pos_boot_hi']),
            fraction_of_ceiling=(float(r.model_cell_mag) / float(r.ceiling_spearman)) if r.ceiling_spearman > 0 else np.nan,
        ))
    cmp_df = pd.DataFrame(faure_cmp + mega_rows)
    atomic_to_csv(cmp_df, out_dir / 'megascale_vs_faure_ceiling.csv', index=False, float_format='%.4g')

    # Figure: 3-panel Faure + MegaScale block availability, noise ceiling, and ESM-MSR performance
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.5), gridspec_kw=dict(width_ratios=[0.95, 1.35, 1.15]))

    # Panel a: Complete >=3x3 blocks available
    ax = axes[0]
    x_bars = np.arange(len(SETS) + 1)
    mega_all = [r for r in mega_rows if r['stratum'] == 'all blocks'][0]
    nb = [int((Bk_df['set'] == d).sum()) for d in SETS] + [int(mega_all['blocks'])]
    ncells = [int(Bk_df[Bk_df['set'] == d].eval('rows * cols').sum()) if (Bk_df['set'] == d).any() else 0 for d in SETS] + [int(mega_all['cells'])]
    bar_cols = [COLOR[d] for d in SETS] + ['#111111']
    ax.bar(x_bars, nb, color=bar_cols, width=0.62)
    for xi, (n_b, n_c) in enumerate(zip(nb, ncells)):
        ax.text(xi, n_b + max(nb) * 0.025, f'{n_b} blocks\n({n_c:,} cells)' if n_b else '0 blocks\n(1D NNK)', ha='center', va='bottom', fontsize=7.6)
    ax.set_xticks(x_bars)
    ax.set_xticklabels([SHORT[d] for d in SETS] + ['MegaScale\ntest (13 dom.)'], rotation=22, ha='right', fontsize=8.3)
    ax.set_ylim(0, max(nb) * 1.26)
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_ylabel('complete ≥ 3 × 3 blocks')
    ax.set_title('a   Combinatorial block availability', loc='left')

    # Panel b: Faure SH3 (GRB2) noise ceiling vs ESM-MSR
    ax = axes[1]
    ax.axhline(0, color='#777777', lw=0.9, ls='--')
    mcol = {'nominal': '#444444', 'R=0.87': '#eb6834', 'R=0.92': '#1baf7a'}
    mlab = {'nominal': 'DiMSum reported error', 'R=0.87': 'Replicate R = 0.87', 'R=0.92': 'Replicate R = 0.92'}
    pos, ticks, labs = 0.0, [], []
    for ds in SETS[2:]:
        for stratum in ('all blocks', 'blocks above floor'):
            t = cmp_df[(cmp_df['dataset'] == ds) & (cmp_df['stratum'] == stratum)]
            v = {r.noise_model: float(r.ceiling_spearman) for r in t.itertuples()}
            ax.plot([pos, pos], [min(v.values()), max(v.values())], color='#cccccc', lw=5.5, solid_capstyle='round', zorder=1)
            for nm_k, val_c in v.items():
                ax.plot(pos, val_c, 'o', color=mcol[nm_k], ms=6.8, zorder=3, label=mlab[nm_k] if pos == 0.0 else None)
            r0 = t.iloc[0]
            mv = float(r0['model_cell_mag'])
            ax.errorbar(pos, mv, yerr=[[mv - r0['model_cell_mag_lo']], [r0['model_cell_mag_hi'] - mv]], fmt='D', color=COLOR[ds], ms=6.2, capsize=2.5, zorder=4, label='ESM-MSR (FINAL_p0)' if pos == 0.0 else None)
            ax.text(pos + 0.16, mv, f'{mv:.3f}', va='center', fontsize=8.0, color=COLOR[ds], fontweight='bold')
            ticks.append(pos)
            labs.append(f'{SHORT[ds]}\n{stratum}\n({int(r0["blocks"])} blks)')
            pos += 1.0
        pos += 0.4
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=7.9)
    ax.set_ylim(-0.10, 1.02)
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_ylabel('cell_mag (mean block Spearman ρ)')
    ax.set_title('b   SH3 (GRB2) noise ceiling vs ESM-MSR', loc='left')
    ax.legend(fontsize=7.5, loc='upper left', frameon=True, facecolor='white', edgecolor='#dddddd')

    # Panel c: MegaScale test doubles noise ceiling vs ESM-MSR (explaining 0.28 vs 0.06)
    ax = axes[2]
    ax.axhline(0, color='#777777', lw=0.9, ls='--')
    mcol_m = {'cDNA 95%CI posterior SD': '#2c5f8a', 'Trypsin–chymotrypsin split ceiling': '#8d6e63'}
    ticks_m, labs_m = [], []
    for pi, stratum in enumerate(('all blocks', 'blocks above floor')):
        tm = cmp_df[(cmp_df['dataset'] == 'tsuboyama_capped_test') & (cmp_df['stratum'] == stratum)]
        vm = {r.noise_model: float(r.ceiling_spearman) for r in tm.itertuples()}
        ax.plot([pi, pi], [min(vm.values()), max(vm.values())], color='#cccccc', lw=5.5, solid_capstyle='round', zorder=1)
        for nm_k, val_c in vm.items():
            ax.plot(pi, val_c, 's', color=mcol_m[nm_k], ms=6.8, zorder=3, label=nm_k if pi == 0 else None)
        r0 = tm.iloc[0]
        mv = float(r0['model_cell_mag'])
        ax.errorbar(pi, mv, yerr=[[mv - r0['model_cell_mag_lo']], [r0['model_cell_mag_hi'] - mv]], fmt='D', color='#000000', ms=6.8, capsize=2.8, zorder=4, label='ESM-MSR (FINAL_p0)' if pi == 0 else None)
        ax.text(pi + 0.14, mv, f'{mv:.3f}', va='center', fontsize=8.3, color='#000000', fontweight='bold')
        ticks_m.append(pi)
        labs_m.append(f'MegaScale test\n{stratum}\n({int(r0["blocks"])} blks)')
    ax.set_xticks(ticks_m)
    ax.set_xticklabels(labs_m, fontsize=8.2)
    ax.set_xlim(-0.55, 1.65)
    ax.set_ylim(-0.10, 1.02)
    ax.grid(axis='y', color=style.GRID, lw=0.6)
    ax.set_ylabel('cell_mag (mean block Spearman ρ)')
    ax.set_title('c   MegaScale test ceiling vs ESM-MSR', loc='left')
    ax.legend(fontsize=7.5, loc='center right', frameon=True, facecolor='white', edgecolor='#dddddd')

    _, dlg_d, _ = load_faure_set(SETS[0])
    n_bg_dlg = int((pd.concat([dlg_d['pos1'], dlg_d['pos2']]).value_counts() >= BACKGROUND_MIN_DOUBLES).sum())
    mg_sub = cmp_df[cmp_df['dataset'] == 'tsuboyama_capped_test']
    mg_c_lo, mg_c_hi = float(mg_sub['ceiling_spearman'].min()), float(mg_sub['ceiling_spearman'].max())
    mg_m_lo, mg_m_hi = float(mg_sub['model_cell_mag'].min()), float(mg_sub['model_cell_mag'].max())
    sh_nom = cmp_df[(cmp_df['dataset'].isin(SETS[2:])) & (cmp_df['noise_model'] == 'nominal')]
    sh_c_lo, sh_c_hi = float(sh_nom['ceiling_spearman'].min()), float(sh_nom['ceiling_spearman'].max())
    sh_m_lo, sh_m_hi = float(sh_nom['model_cell_mag'].min()), float(sh_nom['model_cell_mag'].max())
    style.finish_figure(
        fig,
        out_dir / 'faure_vs_megascale_epistasis_snr',
        title=f'Why cell_mag differs across assays: block design and measurement ceiling on Faure PCA vs MegaScale (B = {B:,})',
        note=(
            f'PDZ3 (DLG4) uses {n_bg_dlg} single-mutant backgrounds (≥{BACKGROUND_MIN_DOUBLES} doubles), yielding 0 ≥3×3 blocks. '
            f'SH3 (GRB2) random mutagenesis yields sparse noisy blocks (DiMSum error ceiling {sh_c_lo:.2f}–{sh_c_hi:.2f}; '
            f'ESM-MSR {sh_m_lo:.3f}–{sh_m_hi:.3f}, position-bootstrap 95% CIs shown, B = {B:,}). '
            f'Homology-Restricted MegaScale test doubles target contact/buried pairs at high proteolysis precision '
            f'(ceiling {mg_c_lo:.2f}–{mg_c_hi:.2f}; ESM-MSR cell_mag {mg_m_lo:.3f}–{mg_m_hi:.3f}).'
        ),
        note_fontsize=7.5,
        top=0.88,
    )
    # Also save cell_mag_ceiling alias for complete coverage
    style.save(fig, out_dir / 'cell_mag_ceiling', close=True)
    return C_df, Sig_df, cmp_df
