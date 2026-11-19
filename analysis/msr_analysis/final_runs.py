"""The ``FINAL_*`` ablation families: hyperparameter definitions, prediction paths, and spine loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import Frame, MSR_HEAD_COLS as HEAD_COLS, _first_site, add_msr_family_preds, build_spine
from .paths import (
    FINAL_BENCH,
    FINAL_PGYM_COMB,
    FINAL_PGYM_WT,
    FINAL_PGYM_XMASK,
    MSR_ROOT as ROOT,
    runner_folder,
)

REFERENCE = 'FINAL_p0'
FAMILIES = {
    'FINAL_p0':          dict(short='p0 (reference)', group='reference', change='reference recipe: unmasked WT and MT passes'),
    'FINAL_p1':          dict(short='p1 masked', group='masking', change='block_mask_wt/mt 0 -> 1 (independent single-site masking)'),
    'FINAL_p1_marg':     dict(short='p1 MT marginal', group='masking', change='p1 + MT pass marginal (block_marginal_mt)'),
    'FINAL_p0_pw0':      dict(short='no pair loss', group='loss', change='block_pair_weight 0.38 -> 0'),
    'FINAL_p0_norank':   dict(short='no rank loss', group='loss', change='lambda_rank_wt/mt 0.5 -> 0'),
    'FINAL_p0_rcthr0':   dict(short='rank censor 0', group='loss', change='rank_censor_threshold -1 -> 0'),
    'FINAL_p0_norange':  dict(short='no range-tagged cells', group='loss', change="Tobit regression without the assay's range-tagged (<-1 / >5) cells (--reg_exclude_range)"),
    'FINAL_p0_mse':      dict(short='MSE (no range-tagged)', group='loss', change='Tobit NLL -> plain MSE on ddG, range-tagged cells excluded (--reg_nll none)'),
    'FINAL_p0_noseqhwm': dict(short='no seq. head', group='architecture', change='incl_sequence_head_wt/mt True -> False'),
    'FINAL_p0_r2a4':     dict(short='MT LoRA r2', group='architecture', change='lora_rank/alpha_mt 16/16 -> 2/4'),
    'FINAL_p0_mega':     dict(short='full MegaScale', group='data', change='train split splits_oct06_capped -> mega_splits (evaluated on its own test split)'),
    'FINAL_p0@independent': dict(short='p0, tested masked', group='cross-mask', change='FINAL_p0 checkpoint evaluated with independent masking'),
    'FINAL_p1@unmasked':    dict(short='p1, tested unmasked', group='cross-mask', change='FINAL_p1 checkpoint evaluated unmasked'),
}
NATIVE_TAG = {'FINAL_p1': 'independent', 'FINAL_p1_marg': 'wt-independent_mt-marginal'}
SEEDS = (1, 2, 3)
FOLDER = {'tsuboyama_capped_val': 'splits_oct06_capped-val', 'tsuboyama_capped_test': 'splits_oct06_capped-test', 'ptmul': 'PTMUL'}
MEGA_FOLDER = {'tsuboyama_capped_val': 'mega_splits-val', 'tsuboyama_capped_test': 'mega_splits-test'}


def _split_family(family: str) -> tuple[str, str, bool]:
    if '@' in family:
        base, tag = family.split('@', 1)
        return base, tag, True
    return family, NATIVE_TAG.get(family, 'unmasked'), False


def _pred_file(family: str, seed: int, ds: str, sigma: str | float = '1.0') -> Path | None:
    fam, tag, _ = _split_family(family)
    folder = runner_folder(ds, mega_own=(fam == 'FINAL_p0_mega'))
    base = FINAL_BENCH / folder / 'final_checkpoints' / fam / f's{seed}'
    if not base.is_dir():
        return None
    sg = f'{float(sigma):.1f}'
    hits = [
        base / f'best.ckpt_sigma{sg}_{tag}_predictions.csv',
        base / f'best.ckpt_sigma{sg}_skip_reverse_{tag}_predictions.csv',
    ]
    hits = [p for p in hits if p.exists()]
    hits.sort(key=lambda p: 'skip_reverse' in p.name)
    return hits[0] if hits else None


def _own_split_spine(path: Path) -> pd.DataFrame:
    t = pd.read_csv(
        path,
        usecols=lambda c: c in ('id', 'mut_type', 'ddG_ML', 'dddG_ML', 'cens', 'mut_structure', 'code_wt'),
        low_memory=False,
    )
    s = pd.DataFrame({'id': t['id'].astype(str), 'row_id': np.arange(len(t))})
    s['y'] = t['ddG_ML'].astype(float)
    s['mut'] = t['mut_type'].astype(str)
    s['n_mut'] = s['mut'].str.count(':') + 1
    s['site1'] = s['mut'].map(_first_site)
    s['group'] = s['cluster'] = s['code'] = t['code_wt'].astype(str)
    for c in ('cens', 'mut_structure', 'code_wt', 'dddG_ML'):
        s[c] = t[c]
    s['use'] = (np.isfinite(s['y']) & (s['cens'] == 0)).to_numpy()
    s['wt_backbone'] = (s['mut_structure'] == '-').to_numpy()
    return s


def load_final_frame(
    ds: str,
    families=None,
    sigma: str | float = '1.0',
    seeds=SEEDS,
    mega_own_split: bool = False,
    heldout: bool = True,
) -> Frame | None:
    """Load ``FINAL_*`` predictions on ``ds`` (filtering Domainome to held-out domains when ``heldout=True``)."""
    fr = _load_final_frame(ds, families, sigma, seeds, mega_own_split)
    if heldout and fr is not None and ds == 'domainome':
        from .heldout import excluded
        keep = ~fr.spine['group'].astype(str).isin(excluded('domainome')).to_numpy()
        fr = Frame(
            fr.name,
            fr.spine.loc[keep].reset_index(drop=True),
            {k: v[..., keep] for k, v in fr.preds.items()},
            notes=dict(fr.notes, heldout=True),
        )
    return fr


def _load_final_frame(ds: str, families=None, sigma='1.0', seeds=SEEDS, mega_own_split: bool = False) -> Frame | None:
    families = list(families or FAMILIES)
    if mega_own_split:
        have = [(s, _pred_file('FINAL_p0_mega', s, ds, sigma)) for s in seeds]
        have = [(s, p) for s, p in have if p is not None]
        if not have:
            return None
        sp = _own_split_spine(have[0][1])
        fr = Frame(name=f'{ds}:mega_own', spine=sp, notes={'seeds': {'FINAL_p0_mega': [s for s, _ in have]}})
        add_msr_family_preds(fr.preds, sp, 'FINAL_p0_mega', [p for _, p in have])
        return fr
    sp = build_spine(ds)
    fr = Frame(name=ds, spine=sp, notes={'seeds': {}})
    for fam in families:
        if fam == 'FINAL_p0_mega' and ds in MEGA_FOLDER:
            continue
        have = [(s, _pred_file(fam, s, ds, sigma)) for s in seeds]
        have = [(s, p) for s, p in have if p is not None]
        if not have:
            continue
        fr.notes['seeds'][fam] = [s for s, _ in have]
        add_msr_family_preds(fr.preds, sp, fam, [p for _, p in have])
    return fr


# ----------------------------------------------------------------------------------------------
# ProteinGym per-DMS helpers (delegated to / compatible with msr_analysis.proteingym)
# ----------------------------------------------------------------------------------------------
MIN_PGYM_ASSAYS = 213


def pgym_per_dms(family: str, seed: int, mode: str = 'wt', dms_ids=None) -> pd.Series | None:
    """Return Series ``DMS_id -> Spearman rho`` for ``family``/``seed`` ('wt', 'comb', or 'deployed')."""
    if mode == 'deployed':
        from .proteingym import load_final_pgym_seeds
        df = load_final_pgym_seeds(family=family, mode='deployed', seeds=(seed,), dms_ids=dms_ids)
        return df[seed] if seed in df.columns and df[seed].notna().any() else None

    fam, _, cross = _split_family(family)
    if cross and mode != 'wt':
        return None
    base = (FINAL_PGYM_XMASK if cross else FINAL_PGYM_WT if mode == 'wt' else FINAL_PGYM_COMB) / fam / f's{seed}'
    col = 'wt_lora_pred' if mode == 'wt' else 'combined_pred'
    if not base.is_dir():
        return None
    files = sorted(p for p in base.glob('*.csv') if p.name != 'summary.csv')
    out = {}
    for p in files:
        hdr = pd.read_csv(p, nrows=0).columns
        if col not in hdr or 'DMS_score' not in hdr:
            continue
        d = pd.read_csv(p, usecols=[col, 'DMS_score'])
        out[p.stem] = float(spearmanr(d[col], d['DMS_score'], nan_policy='omit')[0])
    if mode == 'wt' and len(out) < MIN_PGYM_ASSAYS:
        return None
    if not out:
        return None
    s = pd.Series(out, dtype=float)
    return s.reindex(dms_ids) if dms_ids is not None else s


def pgym_per_dms_seeds(family: str, mode: str = 'wt', dms_ids=None, seeds=SEEDS) -> tuple[pd.Series | None, int]:
    """Return ``(seed_mean_series, n_seeds_used)`` across ``seeds``."""
    rs = [r for r in (pgym_per_dms(family, s, mode, dms_ids=dms_ids) for s in seeds) if r is not None]
    if not rs:
        return None, 0
    return pd.concat(rs, axis=1).mean(axis=1), len(rs)
