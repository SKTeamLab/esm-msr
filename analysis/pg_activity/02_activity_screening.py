#!/usr/bin/env python
"""ProteinGym Activity: does a suggested mutation hurt activity?  Screening against the measured wild type, with a drop of half the assay's standard deviation counted as real.

The 43 Activity-category assays (single mutants only), with the wild-type score WT of every assay taken from ``data/wt_reference.csv`` (built by ``01_wt_reference.py``) instead of an estimated
plateau: an explicit WT row, the median of the synonymous-codon rows, or the value fixed by the paper's scoring; position by position for AICDA (whose score is an enrichment inside each position's
own library; positions whose WT residue has < 100 selected reads are left out).  Seven assays have no WT anywhere (A0A1I9GEU1, A0A247D711, ADRB2, CCDB, PPARG, HRAS, SRC dasatinib): they keep the old
estimate (the peak of the functional plateau), marked "plateau proxy".  For one assay, with score x (curated ``DMS_score``, higher = more active), a variant **hurts** when

    x < WT - 0.5 x (standard deviation of all single-mutant scores of the assay)

(the assay SD, in the assay's own units: the rule has no units and needs no replicate information).  If the measurement noise is at most 10% of the assay variance, half an assay SD is at least 1.6
times the noise.  The curves show this one definition, for the 18 enzymes and for all 43 assays.  Four other definitions are kept for comparison in ``sensitivity.csv`` and ``paired_vs_esm_msr.csv``
(``DEFS_ALL``): ``var5`` / ``var10`` (x < WT - 2 sd, sd^2 = 5% / 10% of the assay variance, i.e. 0.45 / 0.63 assay SD), ``floor`` (the variant lost more than 5% of the way from WT down to the floor,
the score of null variants: >= 10 stop codons or a paper-defined null; only assays with a measured WT and floor) and ``ladder`` (sd = the noise stated in the paper > the spread of the synonymous-codon
rows when they are independent measurements > the spread of the functional plateau).

A screen of K variants takes the K with the highest model score (stabilizing-positive: the models are not retrained for activity) and reports, averaged over assays, the fraction that hurt; the dashed
line is the share of all variants that hurt (random picks).  Intervals are 95% bootstrap over assays; the figures draw the band of ESM-MSR only (the csv files hold all).

Outputs (``results/``): ``assay_definitions.csv`` (per assay: WT, floor and its evidence, the sd's, the share that hurts under each definition), ``landscape``, ``wt_vs_plateau``, ``floor_audit``,
``screening_{enzymes,all}{,_all_models}`` (curves + csv), ``paired_vs_esm_msr.csv``, ``sensitivity.csv``.  ESM-MSR is FINAL_p0 (WT head), and ME / ThermoMPNN / SPURS the retrains: seeds 1-3 of each (the screening statistics per seed, then the seed mean; the bootstrap resamples assays and seeds).

    python analysis/pg_activity/02_activity_screening.py [--B 200]
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402
from msr_analysis.registry import PUBLIC, RETRAINED, ROOT  # noqa: E402

scr = importlib.import_module('08_screening')          # per_protein() and the model families / colours shared with the other screening figures
DATA, OUT = HERE / 'data', HERE / 'results'
OUT.mkdir(exist_ok=True)

PG_DIR = ROOT / 'final_pgym_wt' / 'FINAL_p0' / 's1'      # FINAL_p0 WT head, one file per assay (seed 1: the assay list and labels; the predictions of seeds 1-3 are read below)
SEEDS = (1, 2, 3)
ESM3_S0 = ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only'          # ESM3-small-open: our sigma = 0 read of the ESM-MSR checkpoint
ESM3_S0_FALLBACK = ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'                                       # the March run at sigma 0, for the assays the first folder lacks
ROSETTA_PG = PUBLIC / 'rosetta' / 'raw_narval' / 'proteingym_cartesian_preds_09_11.csv'     # Rosetta Cartesian ddG, ProteinGym singles (keys 'P<row of DMS_substitutions.csv>_<position><residue>')
PG_FAM = scr.FAM
PG_FAM_ALL = [f for f in scr.FAM_ALL if f not in ('geoddg', 'esm3dg')]        # no GeoDDG-3D or ESM3ΔG scores for the Activity assays
TAU = 2.0              # hurts (var5, var10, ladder definitions): more than TAU noise sd below WT
DROP_SD = 0.5          # hurts (the definition of the figures): more than this fraction of the assay's standard deviation below WT
F_LOSS = 0.05          # hurts (floor definition): more than this share of the way from WT to the floor lost
F_VAR = 0.05           # hurts (variance definition): noise variance = this share of the variance of all scores of the assay
MIN_STOP = 10          # stop-codon rows needed for a measured floor
MIN_WINDOW = 3.0       # a floor is "resolved" when (WT - floor) / sqrt(sd_WT^2 + sd_floor^2) is at least this
MIN_WT_READS = 100     # AICDA: a position's WT is used only if its WT residue has at least this many selected reads
MIN_SYN = 20           # synonymous rows needed for a noise estimate
KS = [1, 3, 5, 10, 20, 50, 100]
AICDA = 'AICDA_HUMAN_Gajula_2014_3cycles'

# catalytic enzymes among the Activity assays (readout = catalytic activity of the protein itself, from `selection_assay`); not: receptors, channels, transporters, binders, inhibitors, the GFPs
PG_ENZYME = {AICDA, 'AMIE_PSEAE_Wrenbeck_2017', 'ANCSZ_Hobbs_2022', 'CAS9_STRP1_Spencer_2017_positive', 'CASP3_HUMAN_Roychowdhury_2020', 'CASP7_HUMAN_Roychowdhury_2020',
             'ENVZ_ECOLI_Ghose_2023', 'HEM3_HUMAN_Loggerenberg_2023', 'LGK_LIPST_Klesmith_2015', 'MET_HUMAN_Estevam_2023', 'OTC_HUMAN_Lo_2023', 'OXDA_RHOTO_Vanella_2023_activity',
             'PTEN_HUMAN_Mighell_2018', 'Q59976_STRSQ_Romero_2015', 'RNC_ECOLI_Weeks_2023', 'SRC_HUMAN_Ahler_2019', 'UBE4B_MOUSE_Starita_2013', 'VKOR1_HUMAN_Chiasson_2020_activity'}
# fluorescence readouts: the four fluorescent-protein brightness sets, CreiLOV (PHOT_CHLRE_Chen_2023: a LOV-based fluorescent protein, not the phototropin kinase) and the two CI-repressor sets read through GFP
PG_FLUOR = {'D7PM05_CLYGR_Somermeyer_2022', 'GFP_AEQVI_Sarkisyan_2016', 'Q6WV12_9MAXI_Somermeyer_2022', 'Q8WTC7_9CNID_Somermeyer_2022', 'RPC1_LAMBD_Li_2019_high-expression',
            'RPC1_LAMBD_Li_2019_low-expression', 'PHOT_CHLRE_Chen_2023'}

MATH = ('per assay (single mutants; x = score, higher = more active; WT = measured wild-type score on the assay\'s own scale, AICDA: that of each position):\n'
        '  hurts: x < WT - 0.5 x SD, where SD is the standard deviation of all single-mutant scores of the assay (no replicate information used)\n'
        'curve: mean over assays of (variants that hurt among the K with the highest model score) / K; dashed: share of all variants of the assay that hurt (random picks); band: 95% bootstrap over assays, ESM-MSR only; lower is better')


def cutoff_on_scale(cut, score):
    """The ProteinGym binarization cutoff on the curated score scale.  For the sign-flipped assays (``raw_DMS_directionality`` = -1) the table lists the cutoff of RPC1_LAMBD_Li_2019 on the raw
    scale (low-expression: 8.48 = minus the median of the curated scores, while every curated score is below -4.5): a cutoff outside the range of the scores whose negative lies inside is negated."""
    return -cut if (cut < score.min() or cut > score.max()) and score.min() <= -cut <= score.max() else cut


def plateau(score, cut):
    """WT-like level of an assay estimated from the scores alone (the estimate this analysis replaces) and the spread of that plateau: the mode of a Gaussian KDE of the scores above the
    assay's binarization cutoff ``cut`` (the median where fewer than 5% or more than 95% of the singles are above it) and 1.4826 x MAD of those scores around the mode."""
    from scipy.stats import gaussian_kde
    frac = (score > cut).mean()
    if frac < 0.05 or frac > 0.95:
        cut = np.median(score)
    up = score[score > cut]
    xs = np.linspace(up.min(), up.max(), 400)
    w = float(xs[np.argmax(gaussian_kde(up)(xs))])
    return w, float(1.4826 * np.median(np.abs(up - w)))


def save_checked(fig, stem):
    """Save the figure after checking that no title, label, note or legend leaves the figure box."""
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, OUT / stem)


def short_names(ids):
    """Short, unique display names: GENE_ORGANISM and, where two assays share them, the gene and the rest of the id (author, year, variant)."""
    base = {d: f"{d.split('_')[0]}_{d.split('_')[1][:6]}" for d in ids}
    dup = pd.Series(base).value_counts()
    return {d: (b if dup[b] == 1 else f"{b.split('_')[0]} {d.split('_', 2)[2]}") for d, b in base.items()}


def noise(t, plateau_sd):
    """Measurement noise sd of an assay and where it comes from."""
    if np.isfinite(t.manual_noise):
        return float(t.manual_noise), 'paper'
    if t.n_syn >= MIN_SYN and t.syn_distinct >= 0.5 * t.n_syn and t.syn_sd > 0:
        return float(t.syn_sd), 'synonymous rows'
    return plateau_sd, 'plateau'


def load_assays():
    """All held-out Activity assays (of the 43; ``msr_analysis.heldout``) as one Frame (``spine``: one row per single mutant with its labels under every definition; ``preds``: the models' scores) and the per-assay table."""
    ref = pd.read_csv(ROOT / 'ProteinGym' / 'DMS_substitutions.csv')
    row_of = {d: i for i, d in enumerate(ref.DMS_id)}                   # the Rosetta file is keyed by the row of this table
    ref = ref[ref.coarse_selection_type == 'Activity'].set_index('DMS_id')
    from msr_analysis.heldout import excluded                           # leave out the assays homologous to the training set (the split's four-part rule)
    ref = ref[~ref.index.isin(excluded('proteingym'))]
    ros = pd.read_csv(ROSETTA_PG).rename(columns={'Unnamed: 0': 'key', 'cartesian_ddg_dir': 'cart'})
    ros['i'] = ros.key.str.extract(r'^P(\d+)_')[0].astype(int)
    ros['pm'] = ros.key.str.split('_', n=1).str[1]                       # '<position><substituted residue>', no wild-type letter
    ros = {i: g.drop_duplicates('pm').set_index('pm')['cart'] for i, g in ros.groupby('i')}
    wtref = pd.read_csv(DATA / 'wt_reference.csv').set_index('DMS_id')
    aicda = pd.read_csv(DATA / 'aicda_wt_by_position.csv').set_index('pos')
    srcs = {'msrp0:wt': (lambda k: ROOT / 'final_pgym_wt' / 'FINAL_p0' / f's{k}', 'wt_lora_pred'),            # the retrained models: one directory per seed
            'me1': (lambda k: RETRAINED / 'mutate_everything' / 'predictions' / 'stage1' / f'seed{k}' / 'proteingym', 'pred_combined'),
            'tmpnn': (lambda k: RETRAINED / 'thermompnn' / 'predictions' / 'single_additive' / f'seed{k}' / 'proteingym', 'pred_single_additive'),
            'spurs': (lambda k: RETRAINED / 'spurs' / 'predictions' / f'seed{k}' / 'proteingym', 'SPURS_single_additive'),
            'pmpnn': (lambda k: PUBLIC / 'proteinmpnn' / 'proteingym' / {1: 'fast', 2: 'fast_seed1', 3: 'fast_seed2'}[k], 'mpnn_score')}      # our ProteinMPNN: three runs, one per-variant file per assay
    pub_cols = {'prosst': 'ProSST-2048', 'pmpnn_pub': 'ProteinMPNN', 'esm3_pub': 'ESM3'}   # the scores ProteinGym ships (baselines/public/proteingym_published): ProSST (no run of ours on ProteinGym yet: harness job queued), and ProteinMPNN / ESM3 kept only for the comparison with ours (``23_pg_ours_vs_published.py``)
    rows, preds, info = [], {**{k: {sd: [] for sd in SEEDS} for k in srcs}, **{k: [] for k in list(pub_cols) + ['rosetta', 'msrrel@0:wt']}}, []
    for dms, r in ref.iterrows():
        base = pd.read_csv(PG_DIR / f'{dms}.csv', usecols=['mutant', 'DMS_score'])
        base = base[~base['mutant'].str.contains(':')].reset_index(drop=True)
        x = base['DMS_score'].to_numpy()
        assert np.isfinite(x).all(), dms
        cut = cutoff_on_scale(r.DMS_binarization_cutoff, x)
        w_pl, sd_pl = plateau(x, cut)
        t = wtref.loc[dms]
        use = np.ones(len(x), bool)
        if t.tier == 'C-position':
            pos = base['mutant'].str.extract(r'(\d+)')[0].astype(int)
            wt = pos.map(aicda['wt_score']).to_numpy(float)
            use = (pos.map(aicda['wt_count_selected']) >= MIN_WT_READS).to_numpy()
            kind, wt_scalar = 'paper (per position)', np.nan
        elif t.tier == 'C-open':
            wt, kind, wt_scalar = np.full(len(x), w_pl), 'plateau proxy', w_pl
        else:
            wt = np.full(len(x), t.wt_value)
            kind, wt_scalar = {'A': 'explicit WT row', 'B': 'synonymous rows', 'C-paper': 'paper'}[t.tier], float(t.wt_value)
        sd, sd_src = noise(t, sd_pl)
        var_all = float(x[use].var())
        sd_v5, sd_v10 = np.sqrt(0.05 * var_all), np.sqrt(0.10 * var_all)
        # floor: the score of null variants (stop codons in the raw file, else a null stated by the paper)
        if t.n_null >= MIN_STOP:
            fl, fl_src, n_fl, fl_sd = float(t.null_stop), 'stop codons', int(t.n_null), float(t.null_sd)
        elif np.isfinite(t.manual_null_value):
            fl, fl_src, n_fl, fl_sd = float(t.manual_null_value), 'paper', 0, np.nan
        else:
            fl, fl_src, n_fl, fl_sd = np.nan, '', 0, np.nan
        span = wt - fl
        floor_ok = bool(np.isfinite(fl) and (span[use] > 0).all() and kind != 'plateau proxy')       # a proxy WT cannot anchor a span
        r_floor = np.where(floor_ok, span, np.nan)
        r_floor = (wt - x) / r_floor
        span_med = float(np.median(span[use])) if np.isfinite(fl) else np.nan
        window = span_med / np.hypot(sd, 0 if np.isnan(fl_sd) else fl_sd) if np.isfinite(fl) else np.nan
        z, z_v5, z_v10, z_pl, z_old = (x - wt) / sd, (x - wt) / sd_v5, (x - wt) / sd_v10, (x - wt) / sd_pl, (x - w_pl) / sd_pl
        z_sd = (x - wt) / np.sqrt(var_all)                                                              # the drop in units of the assay's standard deviation
        cls = 'enzyme' if dms in PG_ENZYME else ('fluorescence' if dms in PG_FLUOR else 'other')
        frac = lambda m: float(m[use].mean()) if m is not None else np.nan                             # noqa: E731
        info.append(dict(DMS_id=dms, cls=cls, n_singles=len(x), n_used=int(use.sum()), tier=t.tier, wt_kind=kind, wt=wt_scalar, sd=sd, sd_source=sd_src,
                         var_all=var_all, assay_sd=np.sqrt(var_all), sd_var5=sd_v5, sd_var10=sd_v10, sd_share_of_variance=sd ** 2 / var_all,
                         floor=fl, floor_source=fl_src, n_floor=n_fl, floor_sd=fl_sd, floor_ok=floor_ok, span=span_med, window_over_noise=window,
                         floor_resolved=bool(floor_ok and window >= MIN_WINDOW), frac_within_sd_of_min=float((x < x.min() + sd).mean()), loss5_over_sd=F_LOSS * span_med / sd if np.isfinite(span_med) else np.nan,
                         plateau_w=w_pl, plateau_sd=sd_pl, plateau_minus_wt_in_sd=(w_pl - wt_scalar) / sd if (np.isfinite(wt_scalar) and kind != 'plateau proxy') else np.nan,
                         pg_cutoff=cut, pg_cutoff_method=r.DMS_binarization_method,
                         hurt_floor5=frac(r_floor > F_LOSS) if floor_ok else np.nan, hurt_floor10=frac(r_floor > 0.10) if floor_ok else np.nan,
                         hurt_sd=frac(z_sd < -DROP_SD), hurt_var5=frac(z_v5 < -TAU), hurt_var10=frac(z_v10 < -TAU), hurt_ladder=frac(z < -TAU),
                         hurt_plateau_sd=frac(z_pl < -TAU), hurt_old_proxy=float((z_old < -TAU).mean())))
        for k, (dirs, col) in srcs.items():
            for sd in SEEDS:
                s = pd.read_csv(dirs(sd) / f'{dms}.csv', usecols=['mutant', col]).drop_duplicates('mutant').set_index('mutant')[col]
                preds[k][sd].append(s.reindex(base['mutant']).to_numpy(dtype=float))
        preds['rosetta'].append(ros.get(row_of[dms], pd.Series(dtype=float)).reindex(base['mutant'].str[1:]).to_numpy(dtype=float))   # Rosetta Cartesian ddG, already stabilizing-positive
        f0 = ESM3_S0 / f'{dms}.csv' if (ESM3_S0 / f'{dms}.csv').exists() else ESM3_S0_FALLBACK / f'{dms}.csv'      # ESM3-small-open = our sigma = 0 read (adapter off, WT head, one deterministic run)
        s0 = pd.read_csv(f0, usecols=['mutant', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant')['wt_lora_pred']
        preds['msrrel@0:wt'].append(s0.reindex(base['mutant']).to_numpy(dtype=float))
        pub = pd.read_csv(PUBLIC / 'proteingym_published' / f'{dms}.csv', usecols=['mutant'] + list(pub_cols.values())).drop_duplicates('mutant').set_index('mutant')
        for k, col in pub_cols.items():                                  # variants a model has no score for stay NaN: that model skips them
            preds[k].append(pub[col].reindex(base['mutant']).to_numpy(dtype=float))
        rows.append(pd.DataFrame({'id': dms + '_' + base['mutant'], 'y': z, 'score': x, 'wt': wt, 'z': z, 'z_sd': z_sd, 'z_var5': z_v5, 'z_var10': z_v10, 'z_plateau_sd': z_pl, 'z_old_proxy': z_old,
                                  'r_floor': r_floor, 'use_wt': use, 'use_floor': use & floor_ok, 'group': dms, 'mut': base['mutant'], 'n_mut': 1, 'use': True}))
    spine = pd.concat(rows, ignore_index=True)
    info = pd.DataFrame(info).set_index('DMS_id')
    arr = {k: (np.stack([np.concatenate(v[sd]) for sd in SEEDS]) if k in srcs else np.concatenate(v)[None, :]) for k, v in preds.items()}      # (seeds, variants); one row for the models with a single run
    cov = {k: round(float(np.isfinite(v).all(axis=0).mean()), 3) for k, v in arr.items()}
    print('coverage of the singles by each model:', cov)
    return Frame('pg_activity', spine, arr), info


def label_frame(fr, groups, label, sign, use_col='use_wt'):
    """The Frame for ``per_protein``: the assays in ``groups``, label y = sign * ``label`` (a hit is y above the definition's threshold), rows without a usable WT / floor switched off."""
    sp = fr.spine
    sel = sp['group'].isin(groups).to_numpy()
    s = sp[sel].reset_index(drop=True)
    s = s.assign(y=sign * s[label], use=s[use_col])
    return Frame('pg_activity', s, {k: v[:, sel] for k, v in fr.preds.items()})


# definitions of "hurts": key -> (panel title, label column, sign of the label, threshold on the label, usable-rows column, assay filter).  DEFS are drawn side by side; DEFS_ALL also feed the
# paired comparison (and, with variants, the sensitivity table)
everything = lambda i, d: True                                                                          # noqa: E731
DEFS = {
    'sd': ('hurts if x < WT − 0.5 assay SD', 'z_sd', -1, DROP_SD, 'use_wt', everything),
}
DEFS_ALL = {
    **DEFS,
    'var5': ('noise = 5% of the assay variance: hurts if x < WT − 2 sd', 'z_var5', -1, TAU, 'use_wt', everything),
    'var10': ('noise = 10% of the assay variance: hurts if x < WT − 2 sd', 'z_var10', -1, TAU, 'use_wt', everything),
    'floor': ('lost > 5% of the way from WT to the floor', 'r_floor', +1, F_LOSS, 'use_floor', lambda i, d: bool(i.loc[d, 'floor_ok'])),
    'ladder': ('x < WT − 2 sd, sd from paper / replicates / plateau', 'z', -1, TAU, 'use_wt', everything),
}


def curves(fr, info, group_set, fams, B, ks=KS):
    """{definition: (curve table, number of assays)} for the assays of ``group_set``."""
    out = {}
    for key, (_, lab, sign, tau, use_col, ok) in DEFS.items():
        g = [d for d in info.index if d in group_set and ok(info, d)]
        out[key] = (scr.per_protein('pg', ks, tau, B, mode='singles', fams=fams, fr=label_frame(fr, g, lab, sign, use_col)) if g else pd.DataFrame(), len(g))
    return out


def fig_curves(res, set_title, stem, fams, markers):
    fig, axes = plt.subplots(1, len(DEFS), figsize=(max(9.8, 5.9 * len(DEFS)), 5.7), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (key, (title, *_)) in zip(axes, DEFS.items()):
        d, n = res[key]
        if len(d) == 0:
            ax.axis('off')
            continue
        for f in fams:
            x = d[(d.family == f) & (d.metric == 'hit_rate')].sort_values('K')
            if x.empty:
                continue
            if markers:
                k_ = scr.FAM_STYLE[f]
                col, mk = style.style_of(k_)['color'], style.style_of(k_)['marker']
                ax.plot(x.K, x.value, color=col, lw=2.2 if f == 'msr' else 1.3, marker=mk, ms=4.5, label=scr.FAM_LABEL_SINGLES[f], zorder=4 if f == 'msr' else 3)
                if f == 'msr':                          # the CI band is drawn for ESM-MSR only
                    ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=.18, lw=0, zorder=1)
            else:
                col, ls, lw, zo = scr.FAM_COLOR_ALL[f]
                ax.plot(x.K, x.value, color=col, ls=ls, lw=lw, label=scr.FAM_LABEL_SINGLES[f], zorder=zo)
                if f == 'msr':
                    ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=.15, lw=0, zorder=1)
        ref_fam = 'msr'
        rr = d[(d.family == ref_fam) & (d.metric == 'rand_hit')].sort_values('K')
        ax.plot(rr.K, rr.value, color='#c0392b', ls=(0, (3, 2)), lw=1.4, zorder=3)
        ax.set_xscale('log'); ax.set_xlabel('variants screened per assay (K)'); ax.set_ylabel('fraction of the K picks that hurt (lower is better)' if ax is axes[0] else '')
        ax.set_xticks(sorted(d.K.unique())); ax.set_xticklabels(sorted(d.K.unique()))
        ax.set_ylim(0, None)
        ax.set_title(f'{title}\n({n} assays)', loc='left', fontsize=9.2)
    handles = []
    for f in fams:
        if f in next(iter(res.values()))[0].family.values:
            if markers:
                k_ = scr.FAM_STYLE[f]
                handles.append(Line2D([0], [0], color=style.style_of(k_)['color'], marker=style.style_of(k_)['marker'], lw=2.0, ms=5, label=scr.FAM_LABEL_SINGLES[f]))
            else:
                col, ls, lw, _ = scr.FAM_COLOR_ALL[f]
                handles.append(Line2D([0], [0], color=col, ls=ls, lw=lw + 0.4, label=scr.FAM_LABEL_SINGLES[f]))
    handles.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='random picks'))
    fig.suptitle(set_title, x=0.005, ha='left', fontsize=11, fontweight='bold', y=0.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.958 if len(DEFS) > 1 else 0.972), ncol=len(handles) if len(handles) <= 5 else 4, fontsize=8.6, handlelength=3.4, frameon=True, facecolor='white', edgecolor='#cccccc')
    kw = dict(fontsize=7.2, family='monospace', color='#333333')
    t = fig.text(0.008, 0.012, style.wrap_to_px(fig, MATH, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)      # the footnote wrapped to the figure width
    foot = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, foot + 0.02, 1, 0.875 if len(DEFS) > 1 else 0.93))
    save_checked(fig, stem)


def fig_landscape(fr, info):
    """Score distribution of every assay: WT (green; AICDA: one tick per position), WT - 0.5 assay SD (red, hurting region shaded), the floor (black dash-dot), ProteinGym cutoff (blue dashed), old plateau estimate (grey dotted)."""
    sp = fr.spine
    order = sorted(info.index, key=lambda d: ({'enzyme': 0, 'other': 1, 'fluorescence': 2}[info.loc[d, 'cls']], d))
    nc = 7
    nr = int(np.ceil(len(order) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.35 * nr))
    aicda = pd.read_csv(DATA / 'aicda_wt_by_position.csv')
    names = short_names(order)
    pct = lambda v: '–' if not np.isfinite(v) else f'{100 * v:.0f}'                                       # noqa: E731
    for ax, d in zip(axes.ravel(), order):
        r = info.loc[d]
        s = sp[sp['group'] == d]
        x = s['score'].to_numpy()
        h, edges = np.histogram(x, bins=45)
        ax.bar(edges[:-1], h, width=np.diff(edges), align='edge', color='#a9a9a9', lw=0, zorder=2)
        if d == AICDA:
            ok = aicda['wt_count_selected'] >= MIN_WT_READS
            ax.plot(aicda.loc[ok, 'wt_score'], np.full(int(ok.sum()), h.max() * 0.04), marker='v', ls='', color='#1e8449', ms=4.5, zorder=5)
            ax.plot(aicda.loc[~ok, 'wt_score'], np.full(int((~ok).sum()), h.max() * 0.04), marker='v', ls='', mfc='none', mec='#1e8449', ms=4.5, zorder=5)
        else:
            lo = r.wt - DROP_SD * r.assay_sd
            ax.axvspan(min(x.min(), lo) - 1e-9 * abs(x.max()), lo, color='#f5b7b1', alpha=0.7, lw=0, zorder=1)
            ax.axvline(r.wt, color='#1e8449', lw=1.2, zorder=3, ls='-' if r.wt_kind != 'plateau proxy' else (0, (1, 1)))
            ax.axvline(lo, color='#c0392b', lw=1.0, zorder=3)
            if np.isfinite(r.floor):
                ax.axvline(r.floor, color='black', lw=1.0, ls='-.', zorder=4)
        ax.axvline(r.pg_cutoff, color='#1f4e79', lw=1.0, ls='--', zorder=4)
        if r.wt_kind != 'plateau proxy':
            ax.axvline(r.plateau_w, color='#777777', lw=1.0, ls=':', zorder=4)
        ax.set_yticks([])
        ax.tick_params(labelsize=6, length=2)
        col = {'enzyme': '#1f4e79', 'other': '#000000', 'fluorescence': '#7f7f7f'}[r.cls]
        ax.set_title(names[d] + {'enzyme': ' (enzyme)', 'other': '', 'fluorescence': ' (fluor.)'}[r.cls], fontsize=6.8, loc='left', color=col, fontweight='bold', pad=17)
        ax.text(0.0, 1.085, f'N={int(r.n_used)}   hurts (WT − 0.5 SD): {pct(r.hurt_sd)}%', transform=ax.transAxes, ha='left', va='bottom', fontsize=5.5, color='#333333')
        note = {'plateau proxy': 'WT: plateau proxy (no measured WT)', 'paper (per position)': 'WT: paper, per position'}.get(r.wt_kind, f'WT: {r.wt_kind}') + f'; floor: {r.floor_source or "none"}'
        ax.text(0.0, 1.015, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=5.4, color='#c0392b' if r.wt_kind == 'plateau proxy' else '#555555', style='italic')
    for ax in axes.ravel()[len(order):]:
        ax.axis('off')
    fig.suptitle('ProteinGym Activity: single-mutant scores (grey); measured WT (green; dotted = no WT available: plateau estimate; AICDA: WT of each position, hollow = < 100 reads, left out); floor (black dash-dot);\n'
                 'WT − 0.5 × the assay standard deviation (red, hurting region shaded); ProteinGym binarization cutoff (blue dashed); old plateau estimate (grey dotted).  Blue titles: enzymes; grey: fluorescence.  Numbers: % of variants that hurt',
                 fontsize=8.4, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_checked(fig, 'landscape')


def fig_wt_vs_plateau(info):
    """How far the plateau estimate was from the measured WT, in units of the assay's sd."""
    d = info[np.isfinite(info['plateau_minus_wt_in_sd'])].copy()
    d = d.sort_values(['cls', 'plateau_minus_wt_in_sd'], key=lambda c: c.map({'enzyme': 0, 'other': 1, 'fluorescence': 2}) if c.name == 'cls' else c)
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(d) + 1.4))
    cols = {'enzyme': '#1f4e79', 'other': '#000000', 'fluorescence': '#7f7f7f'}
    y = np.arange(len(d))[::-1]
    ax.barh(y, d['plateau_minus_wt_in_sd'], color=[cols[c] for c in d['cls']], height=0.7)
    ax.axvline(0, color='#333333', lw=0.8)
    for v in (-1, 1):
        ax.axvline(v, color='#999999', lw=0.6, ls=':')
    ax.set_yticks(y)
    names = short_names(list(info.index))
    ax.set_yticklabels([f"{names[i]}  (WT tier {t})" for i, t in zip(d.index, d['tier'])], fontsize=6.8)
    for tl, c in zip(ax.get_yticklabels(), d['cls']):
        tl.set_color(cols[c])
    ax.set_xlabel('(plateau estimate − measured WT) / sd\nblue: enzymes, black: other function, grey: fluorescence readouts', fontsize=8)
    fig.tight_layout()
    ax.set_title(style.wrap_to_px(fig, 'How far the old WT estimate (plateau peak) was from the measured wild type', 0.97 * ax.get_window_extent(fig.canvas.get_renderer()).width, fontsize=9.5, fontweight='bold'),
                 loc='left', fontsize=9.5)
    fig.tight_layout()
    save_checked(fig, 'wt_vs_plateau')


def fig_floor_audit(info):
    """Which assays have a floor, and how it compares with the noise: left, the window WT - floor in units of the noise sd at WT (and of the floor's own spread); right, where 5% of that window
    lies in the same units (the threshold of the floor definition) against the 2 sd of the other definitions."""
    names = short_names(list(info.index))
    order = sorted(info.index, key=lambda d: ({'enzyme': 0, 'other': 1, 'fluorescence': 2}[info.loc[d, 'cls']], -(info.loc[d, 'window_over_noise'] if np.isfinite(info.loc[d, 'window_over_noise']) else -1e9), d))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.215 * len(order) + 1.9), sharey=True)
    y = np.arange(len(order))[::-1]
    cols = {'enzyme': '#1f4e79', 'other': '#000000', 'fluorescence': '#7f7f7f'}
    for yy, d in zip(y, order):
        r = info.loc[d]
        if not np.isfinite(r.window_over_noise) or r.span <= 0:
            for ax in axes:
                ax.text(0.015, yy, 'no measured floor', transform=ax.get_yaxis_transform(), fontsize=6.3, color='#999999', va='center', style='italic')
            continue
        mk = dict(marker='o' if r.floor_source == 'stop codons' else 's', ms=5.5, color=cols[r.cls], mfc=cols[r.cls] if r.floor_resolved else 'white', mew=1.2, ls='')
        if r.wt_kind == 'plateau proxy':
            mk.update(marker='x', mfc='none')
        axes[0].plot(r.window_over_noise, yy, **mk)
        axes[1].plot(r.loss5_over_sd, yy, **mk)
    axes[0].set_xscale('log'); axes[1].set_xscale('log'); axes[0].set_ylim(-1, len(order))
    axes[0].axvline(MIN_WINDOW, color='#c0392b', lw=0.9, ls='--'); axes[0].axvline(1, color='#999999', lw=0.7, ls=':')
    axes[1].axvline(TAU, color='#c0392b', lw=0.9, ls='--'); axes[1].axvline(1, color='#999999', lw=0.7, ls=':')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[d] for d in order], fontsize=6.6)
    for tl, d in zip(axes[0].get_yticklabels(), order):
        tl.set_color(cols[info.loc[d, 'cls']])
    axes[0].set_xlabel('window (WT − floor) / noise sd  (red: 3)', fontsize=8)
    axes[1].set_xlabel('5% of the window / noise sd  (red: 2 sd, the other definitions)', fontsize=8)
    axes[0].set_title('how far the floor is from WT, in noise units', loc='left', fontsize=9)
    axes[1].set_title('where the 5%-of-the-way threshold lies, in noise units', loc='left', fontsize=9)
    fig.suptitle(style.wrap_to_px(fig, 'Floors of the ProteinGym Activity assays   (circle: floor from stop codons, square: from the paper; filled: window ≥ 3; hollow: < 3; cross: floor but plateau-proxy WT; '
                                  'noise sd = the ladder sd, plus the stop codons\' own spread)', 0.96 * fig.bbox.width, fontsize=8), fontsize=8, x=0.005, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_checked(fig, 'floor_audit')


def paired(fr, info, B=10000, ks=(10, 100)):
    """ESM-MSR against every other model, paired over assays: the mean over assays of (hurt rate of the model - hurt rate of ESM-MSR) among the K picks, positive = ESM-MSR hurts less, with a 95%
    bootstrap interval over assays, for every definition in ``DEFS_ALL`` and for the enzymes and all assays."""
    rows = []
    for setname, gs in (('enzymes', PG_ENZYME), ('all', set(info.index))):
        for key, (_, lab, sign, tau, use_col, ok) in DEFS_ALL.items():
            g = [d for d in info.index if d in gs and ok(info, d)]
            f = label_frame(fr, g, lab, sign, use_col)
            sp = f.spine
            for K in ks:
                tabs = {}
                for fam in PG_FAM_ALL:
                    k = scr.fam_key(fam, f, mode='singles')
                    if k is None:
                        continue
                    m = sp['use'].to_numpy() & np.isfinite(f.preds[k]).all(axis=0)
                    t = pd.concat([scr.scr.per_protein_table(sp['group'].to_numpy()[m], pr[m], sp['y'].to_numpy()[m], K, tau) for pr in f.preds[k]]).groupby('group').mean(numeric_only=True)       # the hit rate of every seed, averaged
                    tabs[fam] = t['hit']
                ref = tabs['msr']
                for fam, t in tabs.items():
                    if fam == 'msr':
                        continue
                    d = (t - ref).dropna().to_numpy()
                    boot = d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))].mean(axis=1)           # same resamples for every comparison
                    rows.append(dict(set=setname, definition=key, K=K, family=fam, n_assays=len(d), diff=d.mean(), lo=np.quantile(boot, .025), hi=np.quantile(boot, .975)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'paired_vs_esm_msr.csv', index=False, float_format='%.4f')
    return out


def sensitivity(fr, info, B):
    """Hurt rate of the K picks (K = 1, 10, 100) under the half-SD definition, the variance definitions (5%, 10%), the floor and ladder definitions, and the earlier ones, for the enzymes and for all assays; the floor definition's assays are also
    scored with the other definitions (``same assays``) so that the three are compared on the same assays."""
    ks = [1, 10, 100]
    floor_g = lambda i, d: bool(i.loc[d, 'floor_ok'])                                                   # noqa: E731
    res_g = lambda i, d: bool(i.loc[d, 'floor_resolved'] and i.loc[d, 'floor_ok'])                      # noqa: E731
    noproxy = lambda i, d: i.loc[d, 'wt_kind'] != 'plateau proxy'                                       # noqa: E731
    defs = [('sd: 0.5 assay SD (the definition of the figures)', 'z_sd', -1, DROP_SD, 'use_wt', everything),
            ('floor: lost > 5% of the way to the floor', 'r_floor', +1, F_LOSS, 'use_floor', floor_g),
            ('floor: lost > 10% of the way', 'r_floor', +1, 0.10, 'use_floor', floor_g),
            ('floor: lost > 5%, floors with window >= 3 only', 'r_floor', +1, F_LOSS, 'use_floor', res_g),
            ('variance: sd^2 = 5% of the assay variance', 'z_var5', -1, TAU, 'use_wt', everything),
            ('variance: sd^2 = 10% of the assay variance', 'z_var10', -1, TAU, 'use_wt', everything),
            ('ladder: sd from paper / replicates / plateau', 'z', -1, TAU, 'use_wt', everything),
            ('same assays as the floor definition: 0.5 assay SD', 'z_sd', -1, DROP_SD, 'use_wt', floor_g),
            ('same assays as the floor definition: variance 5%', 'z_var5', -1, TAU, 'use_wt', floor_g),
            ('same assays as the floor definition: ladder', 'z', -1, TAU, 'use_wt', floor_g),
            ('same assays as the floor definition: variance 5%, floors with window >= 3', 'z_var5', -1, TAU, 'use_wt', res_g),
            ('same assays as the floor definition: ladder, floors with window >= 3', 'z', -1, TAU, 'use_wt', res_g),
            ('ladder, plateau-proxy assays dropped', 'z', -1, TAU, 'use_wt', noproxy),
            ('0.5 assay SD, plateau-proxy assays dropped', 'z_sd', -1, DROP_SD, 'use_wt', noproxy),
            ('variance 5%, plateau-proxy assays dropped', 'z_var5', -1, TAU, 'use_wt', noproxy),
            ('measured WT, plateau sd', 'z_plateau_sd', -1, TAU, 'use_wt', everything),
            ('old: plateau WT and plateau sd', 'z_old_proxy', -1, TAU, 'use', everything)]
    rows = []
    for setname, gs in (('enzymes', PG_ENZYME), ('all', set(info.index))):
        for name, lab, sign, tau, use_col, ok in defs:
            g = [d for d in info.index if d in gs and ok(info, d)]
            if not g or (setname == 'enzymes' and 'proxy assays dropped' in name):
                continue
            t = scr.per_protein('pg', ks, tau, B, mode='singles', fams=PG_FAM, fr=label_frame(fr, g, lab, sign, use_col))
            rows.append(t[t.metric.isin(['hit_rate', 'rand_hit'])].assign(set=setname, definition=name, n_assays=len(g)))
    out = pd.concat(rows, ignore_index=True)[['set', 'definition', 'n_assays', 'family', 'K', 'metric', 'value', 'lo', 'hi']]
    out.to_csv(OUT / 'sensitivity.csv', index=False, float_format='%.4f')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000, help='bootstrap replicates')
    a = ap.parse_args()
    fr, info = load_assays()
    info.to_csv(OUT / 'assay_definitions.csv', float_format='%.4f')
    fig_landscape(fr, info)
    fig_wt_vs_plateau(info)
    fig_floor_audit(info)
    sets = (('enzymes', PG_ENZYME, 'ProteinGym Activity: enzymes (18 assays), single mutants'), ('all', set(info.index), f'ProteinGym Activity: all {len(info)} held-out assays, single mutants'))
    for name, gs, title in sets:
        for fams, tag, mk in ((PG_FAM, '', True), (PG_FAM_ALL, '_all_models', False)):
            res = curves(fr, info, gs, fams, a.B)
            pd.concat([t.assign(definition=k, n_assays=n) for k, (t, n) in res.items() if len(t)], ignore_index=True).to_csv(OUT / f'screening_{name}{tag}.csv', index=False, float_format='%.4f')
            fig_curves(res, title, f'screening_{name}{tag}', fams, mk)
            if not tag:
                for key, (t, n) in res.items():
                    print(f'=== {name}: {key}: hurt rate of the K picks ({n} assays) ===')
                    v = t[(t.metric == 'hit_rate') & t.K.isin([1, 10, 100])].pivot_table(index='family', columns='K', values='value')
                    v.loc['random'] = t[(t.metric == 'rand_hit') & (t.family == 'msr') & t.K.isin([1, 10, 100])].set_index('K')['value']
                    print(v.round(3).to_string())
    pr = paired(fr, info)
    print('--- paired difference (model - ESM-MSR) in hurt rate, 95% interval over assays; * = interval excludes 0 ---')
    pr['txt'] = pr.apply(lambda r: f"{r['diff']:+.3f}{'*' if (r.lo > 0 or r.hi < 0) else ' '}", axis=1)
    print(pr.pivot_table(index=['set', 'definition', 'K'], columns='family', values='txt', aggfunc='first').to_string())
    s = sensitivity(fr, info, min(a.B, 100))
    for k in (10, 100):
        print(f'--- sensitivity, K = {k} ---')
        print(s[(s.metric == 'hit_rate') & (s.K == k)].pivot_table(index=['set', 'definition', 'n_assays'], columns='family', values='value', sort=False).round(3).to_string())
    cols = ['cls', 'wt_kind', 'floor_source', 'n_floor', 'span', 'window_over_noise', 'floor_resolved', 'loss5_over_sd', 'sd_share_of_variance', 'hurt_floor5', 'hurt_sd', 'hurt_var5', 'hurt_ladder']
    print(info[cols].round(3).to_string())


if __name__ == '__main__':
    main()
