"""The MGnify stability set scored with ESM-MSR (FINAL_p0): labels, predictions of the three heads, the Tobit-linked reading, homology classes.

MGnify (766,790 variants of 180,367 scaffolds of 60-80 residues; ``data/mgnify/mgnify_mapped.csv``) is a stability set of the same kind as Megascale that ESM-MSR never
trained on.  The testing queue's ``esm_msr_testing.py --mgnify`` run writes, per checkpoint, ``final_mgnify/<family>/s<seed>/all/predictions.csv.gz``: one row per variant with the WT head
(``wt_lora_pred``), the MT head (``mt_lora_pred``) and the combined prediction (``combined_pred`` = 1/2 (MT + WT)), all latent ddG.  The labels stored in a run are the v1 table's; the v2 table
re-derives the wild-type dG of 1,772 scaffolds, so they are replaced by joining the v2 columns on ``uid`` (predictions do not depend on them).

Two readings of every head, both returned by ``load_mgnify``:

  latent   the model's ddG as is (the form the BENCHMARKS tables score)
  linked   the Tobit link applied post hoc, ``h(dG_wt + latent) - dG_wt`` (``esm_msr.val_stats.tobit_link``; the constants are the ones the checkpoints were fit with: sigma_y 0.28, mu_f -0.14,
           sigma_f 0.5).  The link is monotone, so it cannot change a rank within one scaffold (``dG_wt`` is constant there); it changes ranks across scaffolds, and the scale (RMSE).

``MSR_ROOT`` (default ``~/playground/esm-msr-devel``) holds ``final_mgnify/``; ``MSR_MGNIFY_TABLE`` overrides the table path.  ``run='rdenovo'`` is the run on the 1,000 Rocklin de novo designs of the same table
(``final_mgnify/<family>/s<seed>/rdenovo``), ``run='all'`` the MGnify scaffolds.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Frame
from .registry import CACHE, REFERENCE, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402

RUNS = Path(os.environ.get('MSR_MGNIFY_RUNS', ROOT / 'final_mgnify')).expanduser()
RUNS_BY_SIGMA = {1.0: RUNS, 0.5: ROOT / 'final_mgnify_sigma0.5'}        # the LoRA scale of the run: sigma 1 is the deployed one; sigma 0.5 was run under the resource tracker (Nov 18) and fetched Nov 19
TABLE = Path(os.environ.get('MSR_MGNIFY_TABLE', ROOT / 'repo' / 'data' / 'mgnify' / 'mgnify_mapped.csv')).expanduser()
CLASSES = REFERENCE / 'mgnify_homology_classes.csv.gz'
from .tobit import TOBIT  # noqa: E402                        # hparams of FINAL_p0 s1-s3 (tobit_sigma_y / tobit_mu_f / tobit_sigma_f)
HEADS = dict(wt='wt_lora_pred', mt='mt_lora_pred', comb='combined_pred')
PRED_COLS = ['uid', 'code_wt', 'mut_type', 'n_mut', 'mgnify_split', 'has_all_singles', 'wt_lora_pred', 'mt_lora_pred', 'combined_pred', 'wt_lora_pred_additive', 'combined_pred_additive']
LABEL_COLS = ['dG_wt', 'dG_ML', 'ddG_ML', 'dddG_ML', 'cens']
CENS_LEGACY = {'default': 0, 'mut_below_neg1': -1, 'wt_out_of_range': -2, 'mut_above_pos5': 1}
STRATA = {'all': None, 'lowid': ['none', 'low', 'h'], 'clean': ['none', 'low'], 'h': ['h'], 'strong': ['H', 'sameprot']}     # homology of the scaffold to the capped training set (sequence evidence); lowid = without H and same protein, the rule of the Domainome low-id set


def finished_seeds(family='FINAL_p0', run='all', sigma=1.0):
    """Seeds whose run wrote its ``summary.json`` (the last file of a run)."""
    root = RUNS_BY_SIGMA[float(sigma)]
    return [s for s in (1, 2, 3) if (root / family / f's{s}' / run / 'summary.json').exists() and (root / family / f's{s}' / run / 'predictions.csv.gz').exists()]


def _labels():
    t = pd.read_csv(TABLE, usecols=['uid'] + LABEL_COLS, low_memory=False).set_index('uid')
    t['cens'] = t['cens'].map(lambda v: CENS_LEGACY.get(v, v)).astype(int) if t['cens'].dtype == object else t['cens']
    return t


def load_mgnify(family='FINAL_p0', run='all', seeds=None, refresh=False, heldout=True, sigma=1.0):
    """``_load_mgnify`` with the scaffolds (designs) that are homologous to the training set under the split's four-part rule (``msr_analysis.heldout``) marked unused: ``spine['use']`` is False for their rows,
    so every analysis that scores the measured rows sees held-out scaffolds only (``heldout=False`` keeps them, e.g. for the homology strata of ``18_mgnify.py``)."""
    fr = _load_mgnify(family, run, seeds, refresh, sigma)
    if heldout:
        from .heldout import excluded
        bad = excluded('rocklin_denovo' if run == 'rdenovo' else 'mgnify')
        sp = fr.spine.copy()
        sp['use'] = sp['use'].to_numpy() & ~sp['code_wt'].astype(str).isin(bad).to_numpy()
        fr = Frame(fr.name, sp, fr.preds, notes=dict(fr.notes, heldout=True))
    return fr


def _load_mgnify(family='FINAL_p0', run='all', seeds=None, refresh=False, sigma=1.0):
    """Frame of the MGnify run: ``spine`` (labels, ``use`` = measured rows, i.e. cens == 0) and ``preds['<head>']`` / ``preds['<head>~']`` (n_seeds, n) for the latent and the Tobit-linked reading."""
    root = RUNS_BY_SIGMA[float(sigma)]
    seeds = list(seeds or finished_seeds(family, run, sigma))
    if not seeds:
        raise FileNotFoundError(f'no finished MGnify run under {root}/{family}')
    pkl = CACHE / f'frame_mgnify_{family}_{run}_s{"".join(map(str, seeds))}{"" if float(sigma) == 1.0 else f"_sigma{sigma}"}.pkl'
    if pkl.exists() and not refresh:
        with open(pkl, 'rb') as fh:
            return pickle.load(fh)
    base, P = None, {h: [] for h in HEADS}
    for s in seeds:
        d = pd.read_csv(root / family / f's{s}' / run / 'predictions.csv.gz', usecols=lambda c: c in PRED_COLS, low_memory=False)
        if base is None:
            base = d
        else:
            d = d.set_index('uid').loc[base['uid']].reset_index()
        for h, c in HEADS.items():
            P[h].append(d[c].to_numpy(float))
    lab = _labels()
    sp = pd.DataFrame({'id': base['uid'].astype(str), 'row_id': np.arange(len(base))})
    for c in LABEL_COLS:
        sp[c] = base['uid'].map(lab[c]).to_numpy()
    sp['y'] = sp['ddG_ML'].astype(float)
    sp['mut'] = base['mut_type'].astype(str).to_numpy()
    sp['n_mut'] = base['n_mut'].to_numpy()
    sp['site1'] = sp['mut'].str.extract(r'(-?\d+)')[0].astype(float).fillna(-1).astype(int)
    sp['group'] = sp['cluster'] = sp['code'] = sp['code_wt'] = base['code_wt'].to_numpy()
    sp['split'] = base['mgnify_split'].to_numpy()
    sp['has_all_singles'] = base['has_all_singles'].to_numpy()
    sp['use'] = (np.isfinite(sp['y']) & (sp['cens'] == 0)).to_numpy()
    if run == 'rdenovo':                                # the Rocklin de novo designs: no homology classes (they are not MGnify scaffolds)
        sp['cls'] = 'n/a'
    else:
        cls = pd.read_csv(CLASSES if CLASSES.exists() else ROOT / 'leak_audit' / 'out' / 'mgnify_homology_classes.csv', usecols=['code_wt', 'cls']).set_index('code_wt')['cls']
        sp['cls'] = sp['code_wt'].map(cls).fillna('none')
    h = tobit_link(**TOBIT)
    dg = sp['dG_wt'].to_numpy(float)
    preds = {}
    for k, a in P.items():
        a = np.vstack(a)
        preds[k] = a
        preds[k + '~'] = h(dg[None, :] + a) - dg[None, :]
    fr = Frame(f'mgnify_{family}', sp, preds, notes=dict(seeds=seeds, family=family, run=run, sigma=float(sigma), tobit=TOBIT))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(pkl, 'wb') as fh:
        pickle.dump(fr, fh, protocol=4)
    return fr
