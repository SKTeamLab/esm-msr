"""The FINAL_* ablation families: what each one changes, where its benchmark predictions live, and a loader
that lays every family x seed x head onto a dataset's label spine.

The predictions are written by the testing queue (``testing_queue/run_testing_queue.py``, outside the repo), which runs
``inference_scripts/esm_msr_testing.py`` on each family's best checkpoint under its own (native) masking strategy:

    <root>/final_bench/predictions/<folder>/final_checkpoints/<FAMILY>/s<seed>/best.ckpt_sigma<s>[_skip_reverse]_<tag>_predictions.csv
    <root>/final_pgym_wt/<FAMILY>/s<seed>/<DMS_id>.csv          ProteinGym, WT head only (217 assays)
    <root>/final_pgym_comb/<FAMILY>/s<seed>/<DMS_id>.csv        ProteinGym, combined (p0 / p1 only, when run)

``<root>`` is ``MSR_ROOT`` (see ``registry.py``); ``MSR_FINAL_BENCH`` / ``MSR_FINAL_PGYM_WT`` / ``MSR_FINAL_PGYM_COMB``
override the three folders for someone who downloaded the predictions elsewhere.  Nothing here needs a GPU or a checkpoint.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Frame, _align_id, _first_site, build_spine, sum_singles
from .registry import ROOT

FINAL_BENCH = Path(os.environ.get('MSR_FINAL_BENCH', ROOT / 'final_bench' / 'predictions')).expanduser()
FINAL_PGYM_WT = Path(os.environ.get('MSR_FINAL_PGYM_WT', ROOT / 'final_pgym_wt')).expanduser()
FINAL_PGYM_COMB = Path(os.environ.get('MSR_FINAL_PGYM_COMB', ROOT / 'final_pgym_comb')).expanduser()
FINAL_PGYM_XMASK = Path(os.environ.get('MSR_FINAL_PGYM_XMASK', ROOT / 'final_pgym_wt_xmask')).expanduser()

# Every family differs from FINAL_p0 in exactly the hparams listed (checked against the checkpoints' hparams.yaml;
# block_row_chunk 20 -> 40 on the fir arms is a memory setting with an exact gradient, not an ablation).
REFERENCE = 'FINAL_p0'
FAMILIES = {
    'FINAL_p0':          dict(short='p0 (reference)', group='reference', change='reference recipe: unmasked WT and MT passes'),
    'FINAL_p1':          dict(short='p1 masked', group='masking', change='block_mask_wt/mt 0 -> 1 (independent single-site masking)'),
    'FINAL_p1_marg':     dict(short='p1 MT marginal', group='masking', change='p1 + MT pass marginal (block_marginal_mt)'),
    'FINAL_p0_pw0':      dict(short='no pair loss', group='loss', change='block_pair_weight 0.38 -> 0'),
    'FINAL_p0_norank':   dict(short='no rank loss', group='loss', change='lambda_rank_wt/mt 0.5 -> 0'),
    'FINAL_p0_rcthr0':   dict(short='rank censor 0', group='loss', change='rank_censor_threshold -1 -> 0'),
    'FINAL_p0_noseqhwm': dict(short='no seq. head', group='architecture', change='incl_sequence_head_wt/mt True -> False'),
    'FINAL_p0_r2a4':     dict(short='MT LoRA r2', group='architecture', change='lora_rank/alpha_mt 16/16 -> 2/4'),
    'FINAL_p0_mega':     dict(short='full MegaScale', group='data', change='train split splits_oct06_capped -> mega_splits '
                              '(not homology-restricted; its Tsuboyama numbers are on its own test split)'),
    # cross-masking: a checkpoint evaluated under the other family's masking (testing-queue tiers gamut_xmask / pgym_wt_xmask)
    'FINAL_p0@independent': dict(short='p0, tested masked', group='cross-mask', change='FINAL_p0 checkpoint evaluated with independent masking'),
    'FINAL_p1@unmasked':    dict(short='p1, tested unmasked', group='cross-mask', change='FINAL_p1 checkpoint evaluated unmasked'),
}
# the masking tag each family's prediction files carry (its native strategy unless the key names an override)
NATIVE_TAG = {'FINAL_p1': 'independent', 'FINAL_p1_marg': 'wt-independent_mt-marginal'}


def split_family(family):
    """'FINAL_p0@independent' -> ('FINAL_p0', 'independent', True); 'FINAL_p1' -> ('FINAL_p1', 'independent', False)."""
    if '@' in family:
        base, tag = family.split('@', 1)
        return base, tag, True
    return family, NATIVE_TAG.get(family, 'unmasked'), False
SEEDS = (1, 2, 3)

# dataset name in analysis/ -> folder name the testing harness writes
FOLDER = {'tsuboyama_capped_val': 'splits_oct06_capped-val', 'tsuboyama_capped_test': 'splits_oct06_capped-test',
          'ptmul': 'PTMUL'}
MEGA_FOLDER = {'tsuboyama_capped_val': 'mega_splits-val', 'tsuboyama_capped_test': 'mega_splits-test'}
HEAD_COLS = dict(wt='wt_lora_pred', mt='mt_lora_pred', comb='combined_pred')


def pred_file(family, seed, ds, sigma='1.0'):
    """The prediction CSV of one family/seed on one dataset (None when that run has not produced it)."""
    fam, tag, _ = split_family(family)
    folder = (MEGA_FOLDER if fam == 'FINAL_p0_mega' else {}).get(ds) or FOLDER.get(ds, ds)
    base = FINAL_BENCH / folder / 'final_checkpoints' / fam / f's{seed}'
    if not base.is_dir():
        return None
    sg = f'{float(sigma):.1f}'
    hits = [base / f'best.ckpt_sigma{sg}_{tag}_predictions.csv', base / f'best.ckpt_sigma{sg}_skip_reverse_{tag}_predictions.csv']
    hits = [p for p in hits if p.exists()]
    # prefer the full pass over a skip_reverse one when both exist
    hits.sort(key=lambda p: 'skip_reverse' in p.name)
    return hits[0] if hits else None


def own_split_spine(path):
    """A spine built from a prediction file's own label columns (FINAL_p0_mega on mega_splits, which has no task table)."""
    t = pd.read_csv(path, usecols=lambda c: c in ('id', 'mut_type', 'ddG_ML', 'dddG_ML', 'cens', 'mut_structure', 'code_wt'),
                    low_memory=False)
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


def _add_family(out, sp, tag, files):
    ids = sp['id'].to_numpy(dtype=object)
    arrs = {h: [] for h in HEAD_COLS}
    for p in files:
        hdr = pd.read_csv(p, nrows=0).columns
        d = pd.read_csv(p, usecols=['id'] + [c for c in HEAD_COLS.values() if c in hdr], low_memory=False)
        fid = d['id'].astype(str).to_numpy()
        for h, c in HEAD_COLS.items():
            arrs[h].append(_align_id(ids, fid, d[c].to_numpy()) if c in d else np.full(len(ids), np.nan))
    for h, lst in arrs.items():
        a = np.vstack(lst)
        if np.isfinite(a).any():
            out[f'{tag}:{h}'] = a
    if f'{tag}:comb' not in out and f'{tag}:mt' in out and f'{tag}:wt' in out:
        out[f'{tag}:comb'] = 0.5 * (out[f'{tag}:wt'] + out[f'{tag}:mt'])
    if (sp['n_mut'] > 1).any() and f'{tag}:wt' in out:
        out[f'{tag}:wt+'] = sum_singles(sp, out[f'{tag}:wt'])          # additive WT-singles prediction of a multi
        if f'{tag}:comb' in out:
            out[f'{tag}:epi'] = out[f'{tag}:comb'] - out[f'{tag}:wt+']  # comb_AB - WT_A - WT_B (README, analysis/)


def load_final_frame(ds, families=None, sigma='1.0', seeds=SEEDS, mega_own_split=False):
    """Frame for ``ds`` whose ``preds`` hold ``<family>:<head>`` (n_seeds, n) arrays for every family with at least one
    finished seed, and ``notes['seeds'][family]`` = the seeds present.  Heads: wt, mt, comb (= 1/2(MT + WT) when the
    file has no combined column), and on multi-mutant sets wt+ (sum of WT singles) and epi (comb - wt+).

    ``mega_own_split``: score FINAL_p0_mega alone on its own Tsuboyama split (spine from its file) instead."""
    families = list(families or FAMILIES)
    if mega_own_split:
        files = [p for p in (pred_file('FINAL_p0_mega', s, ds, sigma) for s in seeds) if p]
        if not files:
            return None
        sp = own_split_spine(files[0])
        fr = Frame(name=f'{ds}:mega_own', spine=sp, notes={'seeds': {'FINAL_p0_mega': [s for s in seeds if pred_file('FINAL_p0_mega', s, ds, sigma)]}})
        _add_family(fr.preds, sp, 'FINAL_p0_mega', files)
        return fr
    sp = build_spine(ds)
    fr = Frame(name=ds, spine=sp, notes={'seeds': {}})
    for fam in families:
        if fam == 'FINAL_p0_mega' and ds in MEGA_FOLDER:
            continue                                  # different test split: see mega_own_split
        have = [(s, pred_file(fam, s, ds, sigma)) for s in seeds]
        have = [(s, p) for s, p in have if p is not None]
        if not have:
            continue
        fr.notes['seeds'][fam] = [s for s, _ in have]
        _add_family(fr.preds, sp, fam, [p for _, p in have])
    return fr


# ----------------------------------------------------------------------------------------------
# ProteinGym (per-DMS Spearman from the per-assay files; summary.csv of a resumed run is incomplete)
# ----------------------------------------------------------------------------------------------
def pgym_per_dms(family, seed, mode='wt', dms_ids=None):
    """Series DMS_id -> Spearman rho of one family/seed.  mode 'wt' reads wt_lora_pred from final_pgym_wt,
    mode 'comb' reads combined_pred from final_pgym_comb.  Missing assays are NaN."""
    from scipy.stats import spearmanr
    fam, _, cross = split_family(family)
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
    s = pd.Series(out, dtype=float)
    return s.reindex(dms_ids) if dms_ids is not None else s
