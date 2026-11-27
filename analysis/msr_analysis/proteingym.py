"""Authoritative ProteinGym loader, 173-assay holdout filter, joint [assay × seed] bootstrapper, and σ panel plotter.

Eliminates circular imports across ``01_summary``, ``05_proteingym``, ``22``, ``23``, ``98``,
``proteingym/p0_sigma``, ``combined_vs_wt``, ``ablation_vs_p0``, and ``leak_audit/22``.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .boot import ci
from .heldout import excluded
from .models import CATEGORY_COLOR
from .paths import (
    CACHE,
    ESM3_S0_PGYM_COMB,
    FINAL_BENCH,
    FINAL_PGYM_COMB,
    FINAL_PGYM_COMB_S05,
    FINAL_PGYM_WT,
    FINAL_PGYM_WT_S05,
    FINAL_PGYM_XMASK,
    MSR,
    PGYM,
    PGYM_REF_CSV,
    PUBLIC,
    REFERENCE,
    RETRAINED,
    atomic_pickle_dump,
    atomic_to_csv,
)
from .stats import boot_pvalue, fmt_p, holm, seed_sd

PG_CATEGORIES = ('Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness')
PG_CATEGORIES_ALPHA = ('Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability')
CATS = list(PG_CATEGORIES_ALPHA)
PG_CATEGORY_LABEL = {
    'Stability': 'Stability',
    'Expression': 'Expression / abundance',
    'Activity': 'Activity',
    'Binding': 'Binding',
    'OrganismalFitness': 'Organismal fitness',
}
PG_CATEGORY_SHORT = {
    'Stability': 'Stability',
    'Expression': 'Expression',
    'Activity': 'Activity',
    'Binding': 'Binding',
    'OrganismalFitness': 'Org. fitness',
}
PG_DROP = frozenset({
    'BRCA1_HUMAN_Findlay_2018',
    'BRCA2_HUMAN_Erwood_2022_HEK293T',
    'POLG_CXB3N_Mattenberger_2021',
    'SCN5A_HUMAN_Glazer_2019',
})
PG_STAB_LEAK = frozenset({'ESMtherm', 'ESM3ΔG'})
SUPPLEMENT_PG = ('ESMtherm',)
SEEDS = (1, 2, 3)
SIGMAS = (0.0, 0.5, 1.0)

_SEEDS_CACHE_VERSION = 3
_SIGMA_CACHE_VERSION = 2
_ASSAY_META_CACHE: dict[str, tuple[bool, bool]] = {}


# ----------------------------------------------------------------------------------------------
# Reference metadata & canonical 173 held-out assay filter
# ----------------------------------------------------------------------------------------------
def pgym_reference(
    heldout: bool = True,
    drop_large: bool = True,
    include_fallback_small: bool = False,
) -> pd.DataFrame:
    """Return ProteinGym assay metadata DataFrame indexed by ``DMS_id``.

    Columns: ``category``, ``uniprot``, ``homologous`` (4-part training-overlap rule), ``dropped_large`` (``PG_DROP``).
    With defaults ``(heldout=True, drop_large=True, include_fallback_small=False)``, returns the canonical
    **173 held-out assays** (``Activity=38, Binding=13, Expression=18, OrganismalFitness=67, Stability=37``).
    Passing ``include_fallback_small=True`` retains the 3 large assays scored via ``esm-msr-small`` fallback (176).
    """
    hom = excluded('proteingym')
    if PGYM_REF_CSV.exists():
        raw = pd.read_csv(PGYM_REF_CSV).set_index('DMS_id')
        df = pd.DataFrame({
            'category': raw['coarse_selection_type'].astype(str),
            'uniprot': raw['UniProt_ID'].astype(str) if 'UniProt_ID' in raw.columns else '',
        }, index=raw.index.astype(str))
    else:
        all_m = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
        df = pd.DataFrame({
            'category': all_m['category'].astype(str),
            'uniprot': [str(idx).split('_')[0] for idx in all_m.index],
        }, index=all_m.index.astype(str))
    df['homologous'] = df.index.isin(hom)
    df['dropped_large'] = df.index.isin(PG_DROP)

    mask = np.ones(len(df), dtype=bool)
    if heldout:
        mask &= ~df['homologous'].to_numpy()
    if drop_large and not include_fallback_small:
        mask &= ~df['dropped_large'].to_numpy()
    return df.loc[mask].copy()


def heldout_assay_ids(include_fallback_small: bool = False) -> list[str]:
    """Return the canonical list of held-out ProteinGym ``DMS_id``s (173 by default, 176 if ``include_fallback_small``)."""
    return list(pgym_reference(heldout=True, drop_large=True, include_fallback_small=include_fallback_small).index)


def heldout_category_counts(include_fallback_small: bool = False) -> dict[str, int]:
    """Return ``{category: count}`` for the held-out ProteinGym assays (sums to 173 by default)."""
    ref = pgym_reference(heldout=True, drop_large=True, include_fallback_small=include_fallback_small)
    vc = ref['category'].value_counts()
    return {c: int(vc.get(c, 0)) for c in PG_CATEGORIES_ALPHA}


def format_category_labels(
    counts: dict[str, int] | None = None,
    unfiltered_counts: dict[str, int] | None = None,
    multiline: bool = True,
) -> dict[str, str]:
    """Build dynamic axis/header labels with exact assay counts (e.g. ``'Activity\\n(38)'``, ``'DMS mean\\n(173/213)'``)."""
    if counts is None:
        counts = heldout_category_counts(include_fallback_small=False)
    if unfiltered_counts is None:
        unfilt_ref = pgym_reference(heldout=False, drop_large=True, include_fallback_small=False)
        vc = unfilt_ref['category'].value_counts()
        unfiltered_counts = {c: int(vc.get(c, 0)) for c in PG_CATEGORIES_ALPHA}
    sep = '\n' if multiline else ' '
    out = {}
    for c in PG_CATEGORIES_ALPHA:
        short = PG_CATEGORY_SHORT[c]
        n = counts.get(c, 0)
        nu = unfiltered_counts.get(c, n)
        cnt_txt = f'({n}/{nu})' if n != nu and not multiline else f'({n})'
        out[c] = f'{short}{sep}{cnt_txt}'
    tot = sum(counts.values())
    tot_u = sum(unfiltered_counts.values())
    out['PG avg'] = 'PG avg'
    out['DMS mean'] = f'DMS mean{sep}({tot}/{tot_u})' if tot != tot_u else f'DMS mean{sep}({tot})'
    return out


# ----------------------------------------------------------------------------------------------
# Per-assay file discovery & Spearman scoring
# ----------------------------------------------------------------------------------------------
def _safe_rho(pred, y) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 3:
        return np.nan
    p, t = p[ok], t[ok]
    if np.ptp(p) == 0 or np.ptp(t) == 0:
        return np.nan
    return float(spearmanr(p, t)[0])


def assay_has_multi(dms_id: str) -> tuple[bool, bool]:
    """Return ``(has_multi, has_double)`` for ProteinGym assay ``dms_id``."""
    if dms_id in _ASSAY_META_CACHE:
        return _ASSAY_META_CACHE[dms_id]
    cands = [
        RETRAINED / 'thermompnn' / 'predictions' / 'single_additive' / 'seed1' / 'proteingym' / f'{dms_id}.csv',
        FINAL_PGYM_WT / 'FINAL_p0' / 's1' / f'{dms_id}.csv',
        FINAL_PGYM_COMB / 'FINAL_p0' / 's1' / f'{dms_id}.csv',
    ]
    for p in cands:
        if p.exists():
            hdr = set(pd.read_csv(p, nrows=0).columns)
            if 'n_mut' in hdr:
                nm = pd.read_csv(p, usecols=['n_mut'])['n_mut'].to_numpy()
            elif 'mutant' in hdr:
                nm = pd.read_csv(p, usecols=['mutant'])['mutant'].astype(str).str.count(':').to_numpy() + 1
            else:
                continue
            res = (bool((nm > 1).any()), bool((nm == 2).any()))
            _ASSAY_META_CACHE[dms_id] = res
            return res
    return (False, False)


def _wt_csv_path(family: str, seed: int, dms_id: str, sigma: float = 1.0, include_fallback_small: bool = False) -> Path | None:
    if '@' in family:
        fam = family.split('@', 1)[0]
        p = FINAL_PGYM_XMASK / fam / f's{seed}' / f'{dms_id}.csv'
        return p if p.exists() else None
    sg = float(sigma)
    if sg == 0.0:
        p = MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only' / f'{dms_id}.csv'
        if p.exists():
            return p
        if include_fallback_small:
            fb = ESM3_S0_PGYM_COMB / f'{dms_id}.csv'
            return fb if fb.exists() else None
        return None
    if sg == 0.5:
        p = FINAL_PGYM_WT_S05 / family / f's{seed}' / f'{dms_id}.csv'
        return p if p.exists() else None
    cands = []
    if family == 'FINAL_p0' and seed == 1 and os.environ.get('MSR_PGYM_P0_WT'):
        cands.append(Path(os.environ['MSR_PGYM_P0_WT']).expanduser() / f'{dms_id}.csv')
    cands.extend([
        FINAL_PGYM_WT / family / f's{seed}' / f'{dms_id}.csv',
        FINAL_PGYM_COMB / family / f's{seed}' / f'{dms_id}.csv',
    ])
    for p in cands:
        if p.exists() and 'wt_lora_pred' in set(pd.read_csv(p, nrows=0).columns):
            return p
    return None


def _comb_csv_path(family: str, seed: int, dms_id: str, sigma: float = 1.0) -> Path | None:
    if '@' in family:
        return None
    sg = float(sigma)
    if sg == 0.0:
        p = ESM3_S0_PGYM_COMB / f'{dms_id}.csv'
        return p if p.exists() and 'combined_pred' in set(pd.read_csv(p, nrows=0).columns) else None
    if sg == 0.5:
        p = FINAL_PGYM_COMB_S05 / family / f's{seed}' / f'{dms_id}.csv'
        return p if p.exists() and 'combined_pred' in set(pd.read_csv(p, nrows=0).columns) else None
    cands = []
    if family == 'FINAL_p0' and seed == 1 and os.environ.get('MSR_PGYM_P0_COMB'):
        cands.append(Path(os.environ['MSR_PGYM_P0_COMB']).expanduser() / f'{dms_id}.csv')
    cands.append(FINAL_PGYM_COMB / family / f's{seed}' / f'{dms_id}.csv')
    for p in cands:
        if not p.exists():
            continue
        hdr = set(pd.read_csv(p, nrows=0).columns)
        if 'combined_pred' in hdr or ('wt_lora_pred' in hdr and 'mt_lora_pred' in hdr):
            return p
    return None


def _score_msr_csv(path: Path, head: str) -> float:
    hdr = set(pd.read_csv(path, nrows=0).columns)
    if 'DMS_score' not in hdr:
        return np.nan
    if head == 'wt':
        if 'wt_lora_pred' not in hdr:
            return np.nan
        d = pd.read_csv(path, usecols=['wt_lora_pred', 'DMS_score'])
        return _safe_rho(d['wt_lora_pred'].to_numpy(), d['DMS_score'].to_numpy())
    use = [c for c in ('combined_pred', 'wt_lora_pred', 'mt_lora_pred', 'DMS_score') if c in hdr]
    d = pd.read_csv(path, usecols=use)
    if 'combined_pred' in d:
        pred = d['combined_pred'].to_numpy(dtype=float)
    elif 'wt_lora_pred' in d and 'mt_lora_pred' in d:
        pred = 0.5 * (d['wt_lora_pred'].to_numpy(dtype=float) + d['mt_lora_pred'].to_numpy(dtype=float))
    else:
        return np.nan
    return _safe_rho(pred, d['DMS_score'].to_numpy(dtype=float))


def load_final_pgym_seeds(
    family: str = 'FINAL_p0',
    mode: str = 'deployed',
    sigma: float = 1.0,
    dms_ids=None,
    seeds=SEEDS,
    heldout: bool = True,
    include_fallback_small: bool = False,
) -> pd.DataFrame:
    """Return ``(n_assays, len(seeds))`` DataFrame of per-DMS Spearman rhos for ``family``.

    ``mode``:
      - ``'wt'``: WT head on all assays.
      - ``'comb'``: Dual-view combined prediction (``NaN`` where no combined run exists).
      - ``'deployed'``: combined prediction on multi-mutant assays (falling back to WT head only when no seed
        has a combined file for that assay), and WT head on single-mutant assays.
    """
    if mode not in ('wt', 'comb', 'deployed'):
        raise ValueError(f"mode must be 'wt', 'comb', or 'deployed', got {mode!r}")
    if dms_ids is None:
        dms_ids = list(pgym_reference(heldout=heldout, drop_large=True, include_fallback_small=include_fallback_small).index)
    dms_ids = list(dms_ids)
    seeds = tuple(seeds) if float(sigma) != 0.0 else (1,)
    out = pd.DataFrame(np.nan, index=pd.Index(dms_ids, name='DMS_id'), columns=list(seeds), dtype=float)

    for dms in dms_ids:
        if mode == 'wt':
            for s in seeds:
                p = _wt_csv_path(family, s, dms, sigma=sigma, include_fallback_small=include_fallback_small)
                if p is not None:
                    out.loc[dms, s] = _score_msr_csv(p, 'wt')
        elif mode == 'comb':
            for s in seeds:
                p = _comb_csv_path(family, s, dms, sigma=sigma)
                if p is not None:
                    out.loc[dms, s] = _score_msr_csv(p, 'comb')
        else:
            has_mm, _ = assay_has_multi(dms)
            if has_mm:
                comb_vals = {}
                for s in seeds:
                    p = _comb_csv_path(family, s, dms, sigma=sigma)
                    if p is not None:
                        v = _score_msr_csv(p, 'comb')
                        if np.isfinite(v):
                            comb_vals[s] = v
                if comb_vals:
                    for s, v in comb_vals.items():
                        out.loc[dms, s] = v
                    continue
            for s in seeds:
                p = _wt_csv_path(family, s, dms, sigma=sigma, include_fallback_small=include_fallback_small)
                if p is not None:
                    out.loc[dms, s] = _score_msr_csv(p, 'wt')
    return out


# ----------------------------------------------------------------------------------------------
# Sigma = {0.0, 0.5, 1.0} ProteinGym loader & paired property panel
# ----------------------------------------------------------------------------------------------
def load_sigma_pgym(
    dms_ids=None,
    heldout: bool = True,
    drop_large: bool = True,
    include_fallback_small: bool = False,
    include_dms3: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    """Compute (or load cached) per-assay Spearman rhos across ``sigma in (0.0, 0.5, 1.0)`` (WT & deployed).

    Returns ``(summary_df, per_seed_dict)`` where ``per_seed_dict[(readout, sigma)]`` holds per-seed rhos
    indexed by ``assay`` so bootstrap tests can resample both assays and triplicate seeds jointly.
    """
    cache_pkl = CACHE / 'pgym_sigma_seeds_v2.pkl'
    all_ref = pgym_reference(heldout=False, drop_large=False, include_fallback_small=True)

    full_df, full_seeds = None, None
    if cache_pkl.exists() and not refresh:
        with open(cache_pkl, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('version') == _SIGMA_CACHE_VERSION:
            full_df, full_seeds = cached['summary'], cached['per_seed']

    if full_df is None:
        all_ids = list(all_ref.index)
        per_seed: dict[tuple[str, float], dict[str, list[float]]] = {
            (rd, sg): {} for rd in ('wt', 'deployed') for sg in SIGMAS
        }
        rows = []
        for dms in all_ids:
            r_meta = all_ref.loc[dms]
            # Load variants intersected across all 3 sigmas per seed, matching pgsigma.load_assay / load_assay_dep
            wt_s, dep_s, heads_s, n_rows, has_mm = {sg: [] for sg in SIGMAS}, {sg: [] for sg in SIGMAS}, [], None, False
            fb_used = []
            for sg in (1.0, 0.0):
                p_prim = _wt_csv_path('FINAL_p0', 1, dms, sigma=sg, include_fallback_small=False)
                p_fb = _wt_csv_path('FINAL_p0', 1, dms, sigma=sg, include_fallback_small=True)
                if p_prim is None and p_fb is not None:
                    fb_used.append(str(sg))
            for sd in SEEDS:
                w_paths = {sg: _wt_csv_path('FINAL_p0', sd, dms, sigma=sg, include_fallback_small=True) for sg in SIGMAS}
                if not all(p is not None and p.exists() for p in w_paths.values()):
                    continue
                joined = None
                for sg, p in w_paths.items():
                    d = pd.read_csv(p, usecols=['mutant', 'DMS_score', 'wt_lora_pred']).drop_duplicates('mutant')
                    d = d.rename(columns={'wt_lora_pred': f'w_{sg}', 'DMS_score': 'y'}).set_index('mutant')
                    joined = d if joined is None else joined.join(d[[f'w_{sg}']], how='inner')
                joined = joined.reset_index()
                if len(joined) < 3:
                    continue
                n_rows = len(joined) if n_rows is None else n_rows
                has_mm = bool(joined['mutant'].astype(str).str.contains(':').any())
                for sg in SIGMAS:
                    wt_s[sg].append(_safe_rho(joined[f'w_{sg}'].to_numpy(), joined['y'].to_numpy()))

                dep_frame = joined.copy()
                h_map = {}
                for sg in SIGMAS:
                    cp = _comb_csv_path('FINAL_p0', sd, dms, sigma=sg) if has_mm else None
                    if cp is not None:
                        hdr = set(pd.read_csv(cp, nrows=0).columns)
                        if 'combined_pred' in hdr:
                            cd = pd.read_csv(cp, usecols=['mutant', 'combined_pred']).drop_duplicates('mutant').set_index('mutant')['combined_pred'].rename(f'c_{sg}')
                            dep_frame = dep_frame.join(cd, on='mutant', how='inner')
                            dep_frame[f'd_{sg}'] = dep_frame[f'c_{sg}']
                            h_map[sg] = 'combined'
                            continue
                    dep_frame[f'd_{sg}'] = dep_frame[f'w_{sg}']
                    h_map[sg] = 'wt_fallback' if has_mm else 'wt'
                for sg in SIGMAS:
                    dep_s[sg].append(_safe_rho(dep_frame[f'd_{sg}'].to_numpy(), dep_frame['y'].to_numpy()))
                heads_s.append(h_map)

            if not heads_s:
                continue
            row = dict(
                assay=dms,
                source='ProteinGym',
                uniprot=r_meta['uniprot'],
                category=r_meta['category'],
                n=int(n_rows or 0),
                train_overlap=bool(r_meta['homologous']),
                leaked=bool(r_meta['homologous']),
                dropped_large=bool(r_meta['dropped_large']),
                fallback=' '.join(fb_used),
                has_multi=has_mm,
                n_seeds=len(heads_s),
            )
            for sg in SIGMAS:
                w_vals = wt_s[sg] if sg != 0.0 else wt_s[sg][:1]
                per_seed[('wt', sg)][dms] = w_vals
                row[f'rho_{sg}'] = float(np.nanmean(w_vals))
                ks = [i for i, h in enumerate(heads_s) if h[sg] == 'combined'] or list(range(len(heads_s)))
                d_vals = [dep_s[sg][i] for i in ks]
                if sg == 0.0:
                    d_vals = d_vals[:1]
                per_seed[('deployed', sg)][dms] = d_vals
                row[f'rho_dep_{sg}'] = float(np.nanmean(d_vals))
                row[f'dep_head_{sg}'] = heads_s[ks[0]][sg]
            rows.append(row)

        # Also score DMS3 benchmark sets for optional inclusion
        from .registry import DMS3
        for name in DMS3:
            wt_s = {sg: [] for sg in SIGMAS}
            n_set = None
            for sd in SEEDS:
                files = {
                    1.0: FINAL_BENCH / name / 'final_checkpoints' / 'FINAL_p0' / f's{sd}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv',
                    0.5: FINAL_BENCH / name / 'final_checkpoints' / 'FINAL_p0' / f's{sd}' / 'best.ckpt_sigma0.5_unmasked_predictions.csv',
                    0.0: MSR / 'released_capped_w2a4_m16a16' / 'sigma0' / f'{name}.csv',
                }
                if not all(f.exists() for f in files.values()):
                    continue
                joined = None
                for sg, f in files.items():
                    d = pd.read_csv(f, usecols=['id', 'ddG_ML', 'wt_lora_pred']).drop_duplicates('id')
                    d = d.rename(columns={'wt_lora_pred': f'w_{sg}', 'ddG_ML': 'y'}).set_index('id')
                    joined = d if joined is None else joined.join(d[[f'w_{sg}']], how='inner')
                n_set = len(joined) if n_set is None else n_set
                for sg in SIGMAS:
                    wt_s[sg].append(_safe_rho(joined[f'w_{sg}'].to_numpy(), joined['y'].to_numpy()))
            if wt_s[1.0]:
                r_dms = dict(
                    assay=name, source='DMS benchmark', uniprot=name.split('_Faure')[0].split('_Kung')[0],
                    category='DMS benchmark', n=int(n_set or 0), train_overlap=False, leaked=False,
                    dropped_large=False, fallback='', has_multi=False, n_seeds=len(wt_s[1.0]),
                )
                for sg in SIGMAS:
                    vals = wt_s[sg] if sg != 0.0 else wt_s[sg][:1]
                    per_seed[('wt', sg)][name] = vals
                    per_seed[('deployed', sg)][name] = vals
                    r_dms[f'rho_{sg}'] = r_dms[f'rho_dep_{sg}'] = float(np.nanmean(vals))
                    r_dms[f'dep_head_{sg}'] = 'wt'
                rows.append(r_dms)

        full_df = pd.DataFrame(rows)
        full_seeds = {}
        for key, dmap in per_seed.items():
            max_s = max((len(v) for v in dmap.values()), default=1)
            mat = pd.DataFrame(
                {a: list(v) + [np.nan] * (max_s - len(v)) for a, v in dmap.items()},
                index=[f's{i + 1}' for i in range(max_s)],
            ).T
            full_seeds[key] = mat
        atomic_pickle_dump({'version': _SIGMA_CACHE_VERSION, 'summary': full_df, 'per_seed': full_seeds}, cache_pkl)

    sub = full_df.copy()
    if not include_dms3:
        sub = sub[sub['source'] == 'ProteinGym'].copy()
    if dms_ids is not None:
        keep_set = set(dms_ids)
        sub = sub[sub['assay'].isin(keep_set)].copy()
    else:
        if heldout:
            sub = sub[~sub['train_overlap'].astype(bool)].copy()
        if drop_large and not include_fallback_small:
            sub = sub[~sub['dropped_large'].astype(bool)].copy()
    sub = sub.reset_index(drop=True)
    for sg in (0.5, 1.0):
        sub[f'd_{sg}'] = sub[f'rho_dep_{sg}'] - sub['rho_dep_0.0']
    kept_assays = list(sub['assay'])
    sub_seeds = {k: df.reindex(kept_assays) for k, df in full_seeds.items()}
    return sub, sub_seeds


# ----------------------------------------------------------------------------------------------
# Full all-models ProteinGym scorer & joint [category-stratified assay × seed] bootstrapper
# ----------------------------------------------------------------------------------------------
PMPNN_RUNS = ('mpnn_score_s0', 'mpnn_score_s1', 'mpnn_score_s2')


def load_all_models_pgym(
    dms_ids=None,
    heldout: bool = False,
    drop_large: bool = True,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute (or load cached) per-DMS Spearman rhos (seed means AND per-seed DataFrames) across all models.

    Returns ``(seed_mean_df, per_seed_by_model)`` where ``per_seed_by_model[model_name]`` has shape
    ``(n_assays, n_seeds)`` (3 columns for triplicate models, 1 column for single-run models).
    """
    from .external import EXT_ORDER
    all_models = list(EXT_ORDER) + list(SUPPLEMENT_PG) + ['ESM-MSR σ=0.5', 'ProSST published']

    cache_pkl = CACHE / 'pgym_all_models_seeds_v2.pkl'
    cache_csv = CACHE / 'pg_all_models.csv'
    ref_213 = pgym_reference(heldout=False, drop_large=True, include_fallback_small=False)
    base_ids = list(ref_213.index)

    mean_df, seeds_map = None, None
    if cache_pkl.exists() and not refresh:
        with open(cache_pkl, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('version') == _SEEDS_CACHE_VERSION:
            mean_df, seeds_map = cached['mean_df'], cached['per_seed']

    if mean_df is None:
        a = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
        pub = pd.read_csv(PGYM / 'pgym_public_per_dms.csv').set_index('DMS_id')
        ro = pd.read_csv(PGYM / 'pgym_rosetta_singles_per_dms.csv').set_index('DMS_id')
        pm = pd.read_csv(PGYM / 'proteinmpnn_020_unmasked_fast_ensemble3_per_dms.csv').set_index('DMS_id')
        tmpd_ref = pd.read_csv(REFERENCE / 'pgym_thermompnn_corrected_s1_per_dms.csv').set_index('DMS_id')
        prosst_dir = PUBLIC / 'prosst' / 'proteingym'
        prosst_multi_dir = PUBLIC / 'prosst_multi' / 'dual_unmasked' / 'proteingym_multi'
        have_prosst_ours = all((prosst_dir / f'{d}.csv').exists() for d in base_ids)

        def _sd(*parts):
            return {k: RETRAINED.joinpath(*parts, f'seed{k}', 'proteingym') for k in SEEDS}

        tmp_sa_dir = _sd('thermompnn', 'predictions', 'single_additive')
        tmp_d_dir = _sd('thermompnn', 'predictions', 'epistatic_corrected')
        me1_dir = _sd('mutate_everything', 'predictions', 'stage1')
        me2_dir = _sd('mutate_everything', 'predictions', 'stage2')
        sp_dir = _sd('spurs', 'predictions')
        esm3dg_dir = PUBLIC / 'esm3dg' / 'proteingym'
        esmtherm_dir = PUBLIC / 'esmtherm' / 'proteingym'
        geo_dir = PUBLIC / 'geoddg' / 'faspr' / 'proteingym'

        def _align_me2(dm2, dt_spine):
            m_ren = dm2['mut_info'].isin(set(dt_spine['mut_type_renumbered'])).mean()
            m_mut = dm2['mut_info'].isin(set(dt_spine['mutant'])).mean()
            kcol = 'mut_type_renumbered' if m_ren >= m_mut else 'mutant'
            s_map = dm2.drop_duplicates('mut_info').set_index('mut_info')['ddg_pred']
            v2 = -dt_spine[kcol].map(s_map).to_numpy(dtype=float)
            return _safe_rho(v2, dt_spine['DMS_score'].to_numpy(dtype=float))

        def _score_rowid_dir(folder, dms, y_full):
            dp = pd.read_csv(folder / f'{dms}.csv', usecols=['row_id', 'pred'])
            rid = dp['row_id'].to_numpy(dtype=int)
            return _safe_rho(dp['pred'].to_numpy(dtype=float), y_full[rid])

        def _seed_vec(dirs, dms, pred_col):
            vals = []
            for k in SEEDS:
                f = dirs[k] / f'{dms}.csv'
                if f.exists():
                    d = pd.read_csv(f, usecols=[pred_col, 'DMS_score'])
                    vals.append(_safe_rho(d[pred_col].to_numpy(dtype=float), d['DMS_score'].to_numpy(dtype=float)))
                else:
                    vals.append(np.nan)
            return vals

        p0_dep = load_final_pgym_seeds('FINAL_p0', mode='deployed', sigma=1.0, dms_ids=base_ids, heldout=False)
        p0_s05 = load_final_pgym_seeds('FINAL_p0', mode='deployed', sigma=0.5, dms_ids=base_ids, heldout=False)

        per_model_rows: dict[str, dict[str, list[float]]] = {m: {} for m in all_models}
        meta_rows = {}
        for dms in base_ids:
            dt = pd.read_csv(
                tmp_sa_dir[1] / f'{dms}.csv',
                usecols=['mut_type_renumbered', 'mutant', 'n_mut', 'pred_single_additive', 'DMS_score'],
            )
            y_full = dt['DMS_score'].to_numpy(dtype=float)
            has_mm = bool((dt['n_mut'] > 1).any())
            has_dbl = bool((dt['n_mut'] == 2).any())
            has_comb_p0 = any(_comb_csv_path('FINAL_p0', k, dms, 1.0) is not None for k in SEEDS) if has_mm else False
            p0_head = 'combined' if (has_mm and has_comb_p0) else ('wt_fallback' if has_mm else 'wt')

            per_model_rows['ESM-MSR'][dms] = p0_dep.loc[dms].tolist()
            per_model_rows['ESM-MSR σ=0.5'][dms] = p0_s05.loc[dms].tolist()

            # ThermoMPNN / ThermoMPNN-D
            if has_dbl:
                d_seeds = []
                for k in SEEDS:
                    f = tmp_d_dir[k] / f'{dms}.csv'
                    if f.exists():
                        dtd = pd.read_csv(f, usecols=['n_mut', 'pred_single_additive', 'pred_combined', 'DMS_score'])
                        comb_t = np.where(dtd['n_mut'] == 2, dtd['pred_combined'], dtd['pred_single_additive'])
                        d_seeds.append(_safe_rho(comb_t, dtd['DMS_score'].to_numpy(dtype=float)))
                per_model_rows['ThermoMPNN'][dms] = d_seeds if d_seeds else [float(tmpd_ref.loc[dms, 'rho_tmpnnd'])]
            else:
                per_model_rows['ThermoMPNN'][dms] = _seed_vec(tmp_sa_dir, dms, 'pred_single_additive')

            # MutateEverything
            if has_mm:
                per_model_rows['MutateEverything'][dms] = [
                    _align_me2(pd.read_csv(me2_dir[k] / f'{dms}.csv', usecols=['mut_info', 'ddg_pred']), dt)
                    if (me2_dir[k] / f'{dms}.csv').exists() else np.nan
                    for k in SEEDS
                ]
            else:
                per_model_rows['MutateEverything'][dms] = _seed_vec(me1_dir, dms, 'pred_combined')

            # SPURS
            per_model_rows['SPURS'][dms] = _seed_vec(sp_dir, dms, 'SPURS_multi' if has_mm else 'SPURS_single_additive')

            # Zero-shot & public models
            per_model_rows['ProteinMPNN'][dms] = [float(pm.loc[dms, col]) for col in PMPNN_RUNS]
            pub_prosst = float(pub.loc[dms, 'ProSST-2048 rho_full'])
            per_model_rows['ProSST published'][dms] = [pub_prosst]
            pm_file = prosst_multi_dir / f'{dms}.csv.gz'
            if has_mm and pm_file.exists():
                dpm = pd.read_csv(pm_file, usecols=['comb_pred', 'DMS_score'])
                per_model_rows['ProSST'][dms] = [_safe_rho(dpm['comb_pred'].to_numpy(dtype=float), dpm['DMS_score'].to_numpy(dtype=float))]
            else:
                per_model_rows['ProSST'][dms] = [_score_rowid_dir(prosst_dir, dms, y_full)] if have_prosst_ours else [pub_prosst]

            e3_wt_col = 'ESM-MSR WT eps=0' if 'ESM-MSR WT eps=0' in a.columns else 'ESM-MSR WT sigma=0'
            e3_file = ESM3_S0_PGYM_COMB / f'{dms}.csv'
            if has_mm and e3_file.exists():
                de3 = pd.read_csv(e3_file, usecols=['combined_pred', 'DMS_score'])
                e3_val, e3_head = _safe_rho(de3['combined_pred'].to_numpy(), de3['DMS_score'].to_numpy()), 'combined'
            else:
                e3_val, e3_head = float(a.loc[dms, e3_wt_col]), 'wt'
            per_model_rows['ESM3-small-open'][dms] = [e3_val]

            per_model_rows['GeoDDG-3D'][dms] = [_score_rowid_dir(geo_dir, dms, y_full)]
            per_model_rows['ESM3ΔG'][dms] = [_score_rowid_dir(esm3dg_dir, dms, y_full)]
            per_model_rows['ESMtherm'][dms] = [_score_rowid_dir(esmtherm_dir, dms, y_full)]
            per_model_rows['Rosetta'][dms] = [float(ro.loc[dms, 'rho']) if dms in ro.index else np.nan]
            meta_rows[dms] = {
                'category': str(ref_213.loc[dms, 'category']),
                'homologous': bool(ref_213.loc[dms, 'homologous']),
                'has_multi': has_mm,
                'has_double': has_dbl,
                'msr_p0_head': p0_head,
                'esm3_0_head': e3_head,
            }

        seeds_map = {}
        mean_cols = {}
        for m, rmap in per_model_rows.items():
            max_s = max(len(v) for v in rmap.values())
            sdf = pd.DataFrame(
                {d: list(v) + [np.nan] * (max_s - len(v)) for d, v in rmap.items()},
                index=[f's{i + 1}' for i in range(max_s)],
            ).T
            seeds_map[m] = sdf
            with np.errstate(invalid='ignore'):
                mean_cols[m] = pd.Series(np.nanmean(sdf.to_numpy(dtype=float), axis=1), index=sdf.index)

        mean_df = pd.DataFrame.from_dict(meta_rows, orient='index')
        for m, ser in mean_cols.items():
            mean_df[m] = ser
        mean_df.index.name = 'DMS_id'
        atomic_pickle_dump({'version': _SEEDS_CACHE_VERSION, 'mean_df': mean_df, 'per_seed': seeds_map}, cache_pkl)
        atomic_to_csv(mean_df, cache_csv)

    target_ids = list(dms_ids) if dms_ids is not None else list(
        pgym_reference(heldout=heldout, drop_large=drop_large, include_fallback_small=False).index
    )
    out_mean = mean_df.reindex(target_ids).copy()
    out_seeds = {m: sdf.reindex(target_ids).copy() for m, sdf in seeds_map.items()}
    return out_mean, out_seeds


def boot_pgym_categories(
    per_seed_by_model: dict[str, pd.DataFrame | pd.Series],
    categories: pd.Series,
    B: int = 1000,
    seed: int = 2,
    columns: tuple[str, ...] = (*PG_CATEGORIES_ALPHA, 'PG avg', 'DMS mean'),
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Joint ``[category-stratified assay × triplicate-seed]`` bootstrapper across ProteinGym models.

    Resamples assays with replacement within each of the 5 ``PG_CATEGORIES_ALPHA`` and jointly resamples
    replicate seeds ``multinomial(ns, [1/ns]*ns)`` (shared across models and categories on each draw ``b``).
    Returns ``(summary_df, reps_by_model_and_col)``.
    """
    cats_ser = categories.astype(str)
    cat_indices = {c: np.flatnonzero((cats_ser == c).to_numpy()) for c in PG_CATEGORIES_ALPHA}
    rng = np.random.default_rng(seed)

    # Pre-draw stratified assay indices per category: shape (B, n_cat)
    assay_picks = {
        c: rng.integers(0, len(idx), size=(B, len(idx))) if len(idx) else np.zeros((B, 0), dtype=int)
        for c, idx in cat_indices.items()
    }
    # Pre-draw shared seed weights per replicate b for every distinct seed count ns > 1
    distinct_ns = sorted({
        (df.shape[1] if isinstance(df, pd.DataFrame) else 1)
        for df in per_seed_by_model.values()
    })
    seed_w_map: dict[int, np.ndarray] = {1: np.ones((B, 1), dtype=float)}
    for ns in [3] + [n for n in distinct_ns if n not in (0, 1, 3)]:
        seed_w_map[ns] = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(float) / ns

    rows = []
    reps_out: dict[str, dict[str, np.ndarray]] = {}
    for m, raw in per_seed_by_model.items():
        arr = raw.reindex(cats_ser.index).to_numpy(dtype=float) if isinstance(raw, (pd.DataFrame, pd.Series)) else np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        ns = arr.shape[1]
        sw = seed_w_map[ns]  # (B, ns)

        with np.errstate(invalid='ignore'):
            assay_mean = np.nanmean(arr, axis=1)
        fin = np.isfinite(arr)
        arr_zero = np.where(fin, arr, 0.0)
        # Per draw b, assay i: seed-resampled value (B, n_assays)
        num_all = np.einsum('bs,is->bi', sw, arr_zero)
        den_all = np.einsum('bs,is->bi', sw, fin.astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            draw_assay_vals = np.where(den_all > 0, num_all / den_all, np.nan)

        pt_col = {}
        rp_col = {}
        cat_counts = {}
        for c in PG_CATEGORIES_ALPHA:
            idx = cat_indices[c]
            vals_c = assay_mean[idx]
            cat_counts[c] = int(np.isfinite(vals_c).sum())
            pt_col[c] = float(np.nanmean(vals_c)) if cat_counts[c] else np.nan
            if len(idx):
                picked = np.take_along_axis(draw_assay_vals[:, idx], assay_picks[c], axis=1)
                with np.errstate(invalid='ignore'):
                    rp_col[c] = np.nanmean(picked, axis=1)
            else:
                rp_col[c] = np.full(B, np.nan)

        pt_col['PG avg'] = float(np.mean([pt_col[c] for c in PG_CATEGORIES_ALPHA]))
        rp_col['PG avg'] = np.mean([rp_col[c] for c in PG_CATEGORIES_ALPHA], axis=0)
        tot_n = sum(cat_counts.values())
        pt_col['DMS mean'] = float(np.nanmean(assay_mean)) if tot_n else np.nan
        rp_col['DMS mean'] = (
            sum(rp_col[c] * cat_counts[c] for c in PG_CATEGORIES_ALPHA) / tot_n
            if tot_n else np.full(B, np.nan)
        )
        reps_out[m] = rp_col

        seed_rhos_pg_avg = []
        if ns > 1:
            for s_idx in range(ns):
                c_means = [np.nanmean(arr[cat_indices[c], s_idx]) for c in PG_CATEGORIES_ALPHA if len(cat_indices[c])]
                seed_rhos_pg_avg.append(float(np.mean(c_means)))
        pg_seed_sd = seed_sd(seed_rhos_pg_avg)

        for col in columns:
            lo, hi = ci(rp_col[col])
            n_dms = cat_counts[col] if col in cat_counts else tot_n
            rows.append(dict(
                model=m,
                column=col,
                value=pt_col[col],
                lo=lo,
                hi=hi,
                n_dms=n_dms,
                n_seeds=ns,
                seed_sd_pg_avg=pg_seed_sd if col == 'PG avg' else np.nan,
            ))
    return pd.DataFrame(rows), reps_out


def sigma_delta_pvalues(
    df: pd.DataFrame,
    per_seed_dep: dict[tuple[str, float], pd.DataFrame] | None = None,
    B: int = 20000,
    seed: int = 3,
) -> pd.DataFrame:
    """Compute category × sigma (0.5, 1.0) paired ``Delta rho`` vs ``sigma = 0`` with Holm adjustment."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat in PG_CATEGORIES:
        g = df[df['category'] == cat]
        assays = list(g['assay'])
        n_a = len(assays)
        if n_a == 0:
            continue
        picks = rng.integers(0, n_a, size=(B, n_a))
        for sg in (0.5, 1.0):
            d_pt = (g[f'rho_dep_{sg}'] - g['rho_dep_0.0']).to_numpy(dtype=float)
            if per_seed_dep is not None and ('deployed', sg) in per_seed_dep and ('deployed', 0.0) in per_seed_dep:
                s_hi = per_seed_dep[('deployed', sg)].reindex(assays).to_numpy(dtype=float)
                s_0 = per_seed_dep[('deployed', 0.0)].reindex(assays).to_numpy(dtype=float)[:, :1]
                ns = s_hi.shape[1]
                sw = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(float) / ns if ns > 1 else np.ones((B, 1))
                fin = np.isfinite(s_hi)
                num = np.einsum('bs,is->bi', sw, np.where(fin, s_hi, 0.0))
                den = np.einsum('bs,is->bi', sw, fin.astype(float))
                with np.errstate(invalid='ignore', divide='ignore'):
                    hi_draw = np.where(den > 0, num / den, np.nan)
                delta_draw = hi_draw - s_0.T
                m = np.nanmean(np.take_along_axis(delta_draw, picks, axis=1), axis=1)
            else:
                m = d_pt[picks].mean(axis=1)
            p_b = boot_pvalue(m, null=0.0, alternative='two-sided')
            lo, hi = ci(m)
            p_w = float(wilcoxon(d_pt).pvalue) if len(d_pt) >= 5 and np.any(d_pt != 0) else np.nan
            rows.append(dict(
                category=cat,
                sigma=sg,
                n_assays=n_a,
                mean_delta=float(d_pt.mean()),
                median_delta=float(np.median(d_pt)),
                ci_lo=lo,
                ci_hi=hi,
                share_improved=float((d_pt > 0).mean()),
                p_boot=p_b,
                p_wilcoxon=p_w,
            ))
    R = pd.DataFrame(rows)
    if len(R):
        R['p_holm'] = holm(R['p_boot'])
        R['p_wilcoxon_holm'] = holm(R['p_wilcoxon'])
    return R


def draw_property_sigma_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    pval_df: pd.DataFrame,
    mark_leaks: bool = False,
    stats_without_leaks: bool = False,
    compact: bool = False,
) -> None:
    """Draw the paired ``sigma in {0, 0.5, 1}`` box/trajectory panel onto ``ax`` without mutating global rcParams."""
    from . import style

    rng = np.random.default_rng(1)
    gap, w = (0.7, 0.8) if compact else (0.9, 0.82)
    f_mean, f_head, f_tick, ms = (6.4, 7.4, 6.0, 2.5) if compact else (8.0, 10.5, 8.5, 3.2)
    ms_ex = 1.15 * ms
    excluded_col = '#9a9a96'

    pos, spans, x = {}, {}, 0.0
    for cat in PG_CATEGORIES:
        for k in range(3):
            pos[(cat, k)] = x
            x += 1.0
        spans[cat] = (pos[(cat, 0)], pos[(cat, 2)])
        x += gap

    ax.grid(axis='y', color=style.GRID, lw=0.7)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)

    for i, cat in enumerate(PG_CATEGORIES):
        x0, x1 = spans[cat]
        if i % 2 == 0:
            ax.axvspan(x0 - 0.5 - gap / 2, x1 + 0.5 + gap / 2, color=style.BAND, lw=0, zorder=0)
        g = df[df['category'] == cat]
        if g.empty:
            continue
        col = CATEGORY_COLOR[cat]
        d05 = (g['d_0.5'] if 'd_0.5' in g else (g['rho_dep_0.5'] - g['rho_dep_0.0'])).to_numpy(dtype=float)
        d10 = (g['d_1.0'] if 'd_1.0' in g else (g['rho_dep_1.0'] - g['rho_dep_0.0'])).to_numpy(dtype=float)
        Y = np.column_stack([np.zeros(len(g)), d05, d10])
        jit = (rng.random(len(g)) - 0.5) * (w * 0.56)
        leaked = g['leaked'].astype(bool).to_numpy() if 'leaked' in g else np.zeros(len(g), dtype=bool)
        lk = leaked if mark_leaks else np.zeros(len(g), dtype=bool)
        use = ~leaked if stats_without_leaks else np.ones(len(g), dtype=bool)
        dot = style.vivid(col) if mark_leaks else col

        for r_idx, y in enumerate(Y):
            ax.plot(
                [pos[(cat, k)] + jit[r_idx] for k in range(3)],
                y,
                color=excluded_col if lk[r_idx] else dot,
                alpha=0.55 if lk[r_idx] else 0.2,
                lw=0.45 if lk[r_idx] else 0.6,
                zorder=2 if lk[r_idx] else 2.2,
            )
        for k in range(3):
            c = pos[(cat, k)]
            y = Y[:, k]
            if k > 0 and use.any():
                q1, med, q3 = np.quantile(y[use], [0.25, 0.5, 0.75])
                ax.add_patch(plt.Rectangle((c - w / 2, q1), w, q3 - q1, fc=col, alpha=0.22, ec='none', zorder=1))
                ax.plot([c - w / 2, c + w / 2], [float(y[use].mean())] * 2, color=style.darken(col), lw=0.9, zorder=4, solid_capstyle='butt')
            if lk.any():
                ax.plot((c + jit)[lk], y[lk], 'x', ms=ms_ex, color=excluded_col, alpha=0.95, mew=0.5, zorder=2.8)
            if (~lk).any():
                ax.plot((c + jit)[~lk], y[~lk], 'o', ms=ms, color=dot, alpha=0.92 if mark_leaks else 0.8, mew=0, zorder=3.4)
            if use.any():
                med = float(np.median(y[use]))
                ax.plot([c - w / 2, c + w / 2], [med, med], color='#333333', lw=1.8, zorder=5)
                if k > 0:
                    ax.text(
                        c, med + 0.012, f'{med:+.3f}', ha='center', va='bottom',
                        fontsize=f_mean, fontweight='semibold', zorder=8,
                        bbox=dict(fc='white', ec='none', alpha=0.68, pad=0.1),
                    )
        tr = ax.get_xaxis_transform()
        n_txt = f'{int(use.sum())}' + (f' of {len(g)}' if use.sum() < len(g) else '')
        ax.plot([x0 - 0.44, x1 + 0.44], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.3)
        hdr = f'{PG_CATEGORY_SHORT[cat]}\n({n_txt})' if compact else f'{PG_CATEGORY_LABEL[cat]} ({n_txt})'
        ax.text((x0 + x1) / 2, 1.022, hdr, transform=tr, ha='center', va='bottom', fontsize=f_head, fontweight='bold', color='#333333', linespacing=1.05)

    ax.axhline(0, color='#8a8982', lw=1.0, zorder=1)
    xticks, xlabels = [], []
    for cat in PG_CATEGORIES:
        r_cat = pval_df[pval_df['category'] == cat].set_index('sigma') if len(pval_df) else pd.DataFrame()
        xticks.extend([pos[(cat, 0)], pos[(cat, 1)], pos[(cat, 2)]])
        p05 = fmt_p(r_cat.loc[0.5, 'p_holm'], prefix='p=') if 0.5 in r_cat.index else ''
        p10 = fmt_p(r_cat.loc[1.0, 'p_holm'], prefix='p=') if 1.0 in r_cat.index else ''
        xlabels.extend(['0', f'0.5\n{p05}'.rstrip(), f'1\n{p10}'.rstrip()])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=f_tick)
    ax.set_xlim(-0.8, x - gap - 0.2)
    ax.set_xlabel('σ (LoRA scale)')
    ax.set_ylabel('Δρ versus σ = 0')
