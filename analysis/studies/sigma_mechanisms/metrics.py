"""Per-assay, per-site, 20-AA, tail-decoupling, and multi-property Pareto metrics on ``FINAL_p0`` (seeds 1–3).

Replaces the broken Oct-27 single-seed ``proteingym/sigma_tradeoff`` scripts:
  - Evaluates ``FINAL_p0`` across seeds 1–3 on the canonical **173 held-out ProteinGym assays** (+ ``DLG4``/``GRB2``
    binding DMS benchmarks for paired abundance-vs-binding Pareto curves).
  - Joins strictly onto ``PG.pgym_reference(heldout=True)`` (eliminating the 214-vs-176 ``NaN`` left-join crash and
    ``bool(np.nan) == True`` Tsuboyama-flag corruption).
  - Caches compact numeric arrays atomically in gitignored ``analysis/cache/sigma_mechanisms_p0_v1.pkl`` (zero tracked
    ``.pkl`` files in the repository).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from msr_analysis import proteingym as PG
from msr_analysis import stats as ST
from msr_analysis.boot import ci
from msr_analysis.paths import (
    CACHE,
    ESM3_S0_PGYM_COMB,
    FINAL_BENCH,
    FINAL_PGYM_COMB,
    FINAL_PGYM_COMB_S05,
    FINAL_PGYM_WT,
    FINAL_PGYM_WT_S05,
    MSR,
    atomic_pickle_dump,
)

CACHE_VERSION = 1
AAS = list('AVILMFWYCGPSTNQHDEKR')
AA_CLASS = {
    'P': 'Pro', 'G': 'Gly',
    **{a: 'hydrophobic' for a in 'AVILMFWYC'},
    **{a: 'polar' for a in 'STNQH'},
    **{a: 'charged' for a in 'DEKR'},
}
BULKY_HYDRO = frozenset('WCFIM')

# Paired (abundance/expression assay, function/binding assay, short protein label) on held-out proteins
PARETO_PAIRS = [
    ('RASK_HUMAN', 'KRAS', 'RASK_HUMAN_Weng_2022_abundance', 'RASK_HUMAN_Weng_2022_binding-DARPin_K55', 'ProteinGym'),
    ('PTEN_HUMAN', 'PTEN', 'PTEN_HUMAN_Matreyek_2021', 'PTEN_HUMAN_Mighell_2018', 'ProteinGym'),
    ('KCNE1_HUMAN', 'KCNE1', 'KCNE1_HUMAN_Muhammad_2023_expression', 'KCNE1_HUMAN_Muhammad_2023_function', 'ProteinGym'),
    ('KCNJ2_MOUSE', 'Kir2.1', 'KCNJ2_MOUSE_Coyote-Maestas_2022_surface', 'KCNJ2_MOUSE_Coyote-Maestas_2022_function', 'ProteinGym'),
    ('CP2C9_HUMAN', 'CYP2C9', 'CP2C9_HUMAN_Amorosi_2021_abundance', 'CP2C9_HUMAN_Amorosi_2021_activity', 'ProteinGym'),
    ('HXK4_HUMAN', 'GCK', 'HXK4_HUMAN_Gersing_2023_abundance', 'HXK4_HUMAN_Gersing_2022_activity', 'ProteinGym'),
    ('S22A1_HUMAN', 'OCT1', 'S22A1_HUMAN_Yee_2023_abundance', 'S22A1_HUMAN_Yee_2023_activity', 'ProteinGym'),
    ('VKOR1_HUMAN', 'VKOR1', 'VKOR1_HUMAN_Chiasson_2020_abundance', 'VKOR1_HUMAN_Chiasson_2020_activity', 'ProteinGym'),
    ('OXDA_RHOTO', 'DAAO', 'OXDA_RHOTO_Vanella_2023_expression', 'OXDA_RHOTO_Vanella_2023_activity', 'ProteinGym'),
    ('Q53Z42_HUMAN', 'HLA-A*02', 'Q53Z42_HUMAN_McShan_2019_expression', 'Q53Z42_HUMAN_McShan_2019_binding-TAPBPR', 'ProteinGym'),
    ('SPIKE_SARS2', 'Spike RBD', 'SPIKE_SARS2_Starr_2020_expression', 'SPIKE_SARS2_Starr_2020_binding', 'ProteinGym'),
    ('DLG4_HUMAN', 'PSD95-PDZ3', 'DLG4_HUMAN_Faure_2021', 'DLG4_HUMAN_Faure_2021_binding_domain', 'DMS benchmark'),
    ('GRB2_HUMAN', 'GRB2-SH3', 'GRB2_HUMAN_Faure_2021', 'GRB2_HUMAN_Faure_2021_binding_domain', 'DMS benchmark'),
]


def _pct_rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 0.5) / len(x)


def _within_site_rho(pos: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({'pos': pos, 'p': pred, 'y': y})
    g = df.groupby('pos')
    ok = g['pos'].transform('size') >= 5
    d = df[ok]
    if len(d) < 20:
        return np.nan
    g = d.groupby('pos')
    rx, ry = g['p'].rank(), g['y'].rank()
    dx = rx - rx.groupby(d['pos']).transform('mean')
    dy = ry - ry.groupby(d['pos']).transform('mean')
    num = (dx * dy).groupby(d['pos']).sum()
    den = np.sqrt((dx ** 2).groupby(d['pos']).sum() * (dy ** 2).groupby(d['pos']).sum())
    r = (num / den.replace(0, np.nan)).dropna()
    return float(r.mean()) if len(r) else np.nan


def _load_assay_wt_seeds(dms_id: str) -> list[pd.DataFrame]:
    """Load per-seed single- & multi-mutant WT-head frames across ``sigma in {0.0, 0.5, 1.0}`` for ``FINAL_p0``."""
    out = []
    p0_path = MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only' / f'{dms_id}.csv'
    if not p0_path.exists():
        p0_path = ESM3_S0_PGYM_COMB / f'{dms_id}.csv'
    if not p0_path.exists():
        return out
    d0 = pd.read_csv(p0_path, usecols=['mutant', 'DMS_score', 'wt_lora_pred']).drop_duplicates('mutant')
    d0 = d0.rename(columns={'DMS_score': 'y', 'wt_lora_pred': 's0.0'}).set_index('mutant')

    for sd in PG.SEEDS:
        p05 = FINAL_PGYM_WT_S05 / 'FINAL_p0' / f's{sd}' / f'{dms_id}.csv'
        p10 = FINAL_PGYM_WT / 'FINAL_p0' / f's{sd}' / f'{dms_id}.csv'
        if not p10.exists():
            p10 = FINAL_PGYM_COMB / 'FINAL_p0' / f's{sd}' / f'{dms_id}.csv'
        if not (p05.exists() and p10.exists()):
            continue
        d05 = pd.read_csv(p05, usecols=['mutant', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant')['wt_lora_pred'].rename('s0.5')
        d10 = pd.read_csv(p10, usecols=['mutant', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant')['wt_lora_pred'].rename('s1.0')
        joined = d0.join(d05, how='inner').join(d10, how='inner').reset_index()
        out.append(joined)
    return [d for d in out if len(d) == len(out[0])] if out else []


def compute_sigma_mechanisms_cache(refresh: bool = False) -> dict:
    """Compute or load cached per-assay mechanistic matrices across the 173 held-out ProteinGym assays."""
    cache_path = CACHE / 'sigma_mechanisms_p0_v1.pkl'
    if cache_path.exists() and not refresh:
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('version') == CACHE_VERSION:
            return cached

    ref = PG.pgym_reference(heldout=True, drop_large=True, include_fallback_small=False)
    dms_ids = list(ref.index)

    # Per-seed arrays keyed by metric -> (n_assays, 3)
    metric_keys = [
        'between_0.0', 'between_0.5', 'between_1.0',
        'within_0.0', 'within_0.5', 'within_1.0',
        'top10_pct_0.0', 'top10_pct_0.5', 'top10_pct_1.0',
        'bot10_pct_0.0', 'bot10_pct_0.5', 'bot10_pct_1.0',
        'top10_hydro_0.0', 'top10_hydro_0.5', 'top10_hydro_1.0',
        'top10_wcfim_0.0', 'top10_wcfim_0.5', 'top10_wcfim_1.0',
    ]
    per_seed = {k: np.full((len(dms_ids), 3), np.nan, dtype=float) for k in metric_keys}
    aa_meas = np.full((len(dms_ids), 20), np.nan, dtype=float)
    aa_shift_10 = np.full((len(dms_ids), 3, 20), np.nan, dtype=float)
    aa_shift_05 = np.full((len(dms_ids), 3, 20), np.nan, dtype=float)
    n_singles_arr = np.zeros(len(dms_ids), dtype=int)

    for ai, dms in enumerate(dms_ids):
        seeds_dfs = _load_assay_wt_seeds(dms)
        if not seeds_dfs:
            continue
        for si, df in enumerate(seeds_dfs):
            sing = df[~df['mutant'].astype(str).str.contains(':')].reset_index(drop=True)
            if si == 0:
                n_singles_arr[ai] = len(sing)
            if len(sing) < 25:
                continue
            pos = sing['mutant'].astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()
            to_aa = sing['mutant'].astype(str).str[-1].to_numpy()
            y = sing['y'].to_numpy(float)
            pm = _pct_rank(y)

            g_pos = pd.DataFrame({'pos': pos, 'y': y, 's0.0': sing['s0.0'], 's0.5': sing['s0.5'], 's1.0': sing['s1.0']}).groupby('pos').mean()
            for sg in (0.0, 0.5, 1.0):
                sc = sing[f's{sg}'].to_numpy(float)
                per_seed[f'between_{sg}'][ai, si] = float(spearmanr(g_pos[f's{sg}'], g_pos['y'])[0]) if len(g_pos) >= 5 else np.nan
                per_seed[f'within_{sg}'][ai, si] = _within_site_rho(pos, sc, y)

                if len(sing) >= 100:
                    n10 = max(10, len(sing) // 10)
                    order = np.argsort(-sc, kind='stable')
                    top_idx, bot_idx = order[:n10], order[-n10:]
                    per_seed[f'top10_pct_{sg}'][ai, si] = float(pm[top_idx].mean())
                    per_seed[f'bot10_pct_{sg}'][ai, si] = float(pm[bot_idx].mean())
                    top_aas = to_aa[top_idx]
                    per_seed[f'top10_hydro_{sg}'][ai, si] = float(np.mean([AA_CLASS.get(a) == 'hydrophobic' for a in top_aas]))
                    per_seed[f'top10_wcfim_{sg}'][ai, si] = float(np.mean([a in BULKY_HYDRO for a in top_aas]))

            if len(sing) >= 100:
                p0_pct = _pct_rank(sing['s0.0'].to_numpy(float))
                p05_pct = _pct_rank(sing['s0.5'].to_numpy(float))
                p10_pct = _pct_rank(sing['s1.0'].to_numpy(float))
                tmp = pd.DataFrame({'to': to_aa, 'm': pm, 'd10': p10_pct - p0_pct, 'd05': p05_pct - p0_pct})
                grp_aa = tmp.groupby('to').mean().reindex(AAS)
                if si == 0:
                    aa_meas[ai] = grp_aa['m'].to_numpy(float)
                aa_shift_10[ai, si] = grp_aa['d10'].to_numpy(float)
                aa_shift_05[ai, si] = grp_aa['d05'].to_numpy(float)

    # Also compute Pareto paired trajectories (deployed readout on PGym + DMS3 binding sets)
    sig_all, _ = PG.load_sigma_pgym(heldout=False, drop_large=False, include_fallback_small=True, include_dms3=True, refresh=False)
    rho_idx = sig_all.set_index('assay')
    held_pg = set(dms_ids)
    pareto_rows = []
    for prot, short_nm, abund_id, func_id, func_src in PARETO_PAIRS:
        if abund_id not in held_pg:
            continue
        if func_src == 'ProteinGym' and func_id not in held_pg:
            continue
        if abund_id not in rho_idx.index or func_id not in rho_idx.index:
            continue
        ra = rho_idx.loc[abund_id]
        rf = rho_idx.loc[func_id]
        pareto_rows.append(dict(
            protein=prot, short=short_nm, abund_assay=abund_id, func_assay=func_id, func_source=func_src,
            abund_0=float(ra['rho_dep_0.0']), abund_05=float(ra['rho_dep_0.5']), abund_1=float(ra['rho_dep_1.0']),
            func_0=float(rf['rho_dep_0.0']), func_05=float(rf['rho_dep_0.5']), func_1=float(rf['rho_dep_1.0']),
        ))

    payload = dict(
        version=CACHE_VERSION,
        dms_ids=dms_ids,
        categories=ref['category'].astype(str).tolist(),
        n_singles=n_singles_arr,
        per_seed=per_seed,
        aa_meas=aa_meas,
        aa_shift_10=aa_shift_10,
        aa_shift_05=aa_shift_05,
        pareto_df=pd.DataFrame(pareto_rows),
    )
    atomic_pickle_dump(payload, cache_path)
    return payload


def summarize_mechanisms(payload: dict, B: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build backing DataFrames with joint ``[assay × triplicate-seed]`` bootstrap 95% CIs and Holm p-values."""
    cats = np.asarray(payload['categories'], dtype=object)
    ps = payload['per_seed']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _summarize_mechanisms_inner(payload, cats, ps, B)


def _summarize_mechanisms_inner(payload: dict, cats: np.ndarray, ps: dict, B: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 1. Between-site vs Within-site Δρ per category & sigma
    site_rows = []
    for cat in PG.PG_CATEGORIES:
        idx = np.flatnonzero(cats == cat)
        dom_c, sw_c = ST.make_domain_seed_draws(len(idx), B=B, seed=41)
        for comp_kind in ('between', 'within'):
            base_mat = ps[f'{comp_kind}_0.0'][idx, :1]
            for sg in (0.5, 1.0):
                hi_mat = ps[f'{comp_kind}_{sg}'][idx]
                diff_mat = hi_mat - base_mat
                pt, reps = ST.boot_domain_seeds(diff_mat, B=B, counts=dom_c, seed_weights=sw_c)
                lo, hi = ci(reps)
                p_b = ST.boot_pvalue(reps, null=0.0, alternative='two-sided')
                pt0 = float(np.nanmean(base_mat))
                pt_sg = float(np.nanmean(np.nanmean(hi_mat, axis=1)))
                site_rows.append(dict(
                    category=cat, component=comp_kind, sigma=sg, n_assays=int(np.isfinite(diff_mat).any(axis=1).sum()),
                    rho_0=pt0, rho_sigma=pt_sg, delta=pt, lo=lo, hi=hi, p_boot=p_b, B=B,
                ))
    site_df = pd.DataFrame(site_rows)
    site_df['p_holm'] = ST.holm(site_df['p_boot'])

    # 2. 20-AA predicted percentile rank shift vs empirical category preference
    aa_meas = payload['aa_meas']
    aa_s10 = np.nanmean(payload['aa_shift_10'], axis=1)  # (n_assays, 20)
    aa_s05 = np.nanmean(payload['aa_shift_05'], axis=1)
    aa_rows = []
    stab_mask = np.isin(cats, ['Stability', 'Expression'])
    func_mask = np.isin(cats, ['Activity', 'Binding', 'OrganismalFitness'])
    for j, aa in enumerate(AAS):
        rec = dict(
            aa=aa, aa_class=AA_CLASS[aa],
            shift_10_all=float(np.nanmean(aa_s10[:, j])),
            shift_05_all=float(np.nanmean(aa_s05[:, j])),
            meas_tol_stability_expr=float(np.nanmean(aa_meas[stab_mask, j]) - 0.5),
            meas_tol_functional=float(np.nanmean(aa_meas[func_mask, j]) - 0.5),
        )
        for cat in PG.PG_CATEGORIES:
            cmask = cats == cat
            rec[f'shift_10_{cat}'] = float(np.nanmean(aa_s10[cmask, j]))
            rec[f'meas_tol_{cat}'] = float(np.nanmean(aa_meas[cmask, j]) - 0.5)
        aa_rows.append(rec)
    aa_df = pd.DataFrame(aa_rows)

    # 3. Top-10% forward selection vs Bottom-10% damage-filtering decoupling
    tail_rows = []
    for cat in PG.PG_CATEGORIES:
        idx = np.flatnonzero((cats == cat) & (payload['n_singles'] >= 100))
        dom_c, sw_c = ST.make_domain_seed_draws(len(idx), B=B, seed=53)
        for sg in (0.5, 1.0):
            # Top-10% forward selection gain: +Δ true fitness percentile (higher = better top picks)
            d_top = ps[f'top10_pct_{sg}'][idx] - ps['top10_pct_0.0'][idx, :1]
            pt_t, rp_t = ST.boot_domain_seeds(d_top, B=B, counts=dom_c, seed_weights=sw_c)
            lo_t, hi_t = ci(rp_t)

            # Bottom-10% damage-filtering gain: -Δ true fitness percentile of bottom-10% (positive = better rejection of bad mutants!)
            d_bot = -(ps[f'bot10_pct_{sg}'][idx] - ps['bot10_pct_0.0'][idx, :1])
            pt_b, rp_b = ST.boot_domain_seeds(d_bot, B=B, counts=dom_c, seed_weights=sw_c)
            lo_b, hi_b = ci(rp_b)

            # Bulky hydrophobic (WCFIM) fraction shift in top-10% picks
            d_wcfim = ps[f'top10_wcfim_{sg}'][idx] - ps['top10_wcfim_0.0'][idx, :1]
            pt_w, _ = ST.boot_domain_seeds(d_wcfim, B=B, counts=dom_c, seed_weights=sw_c)

            tail_rows.append(dict(
                category=cat, sigma=sg, n_assays=int(np.isfinite(d_top).any(axis=1).sum()),
                top10_selection_gain=pt_t, top10_lo=lo_t, top10_hi=hi_t, top10_p=ST.boot_pvalue(rp_t),
                bot10_filtering_gain=pt_b, bot10_lo=lo_b, bot10_hi=hi_b, bot10_p=ST.boot_pvalue(rp_b),
                top10_wcfim_shift=pt_w, B=B,
            ))
    tail_df = pd.DataFrame(tail_rows)

    pareto_df = payload['pareto_df'].copy()
    return site_df, aa_df, tail_df, pareto_df
