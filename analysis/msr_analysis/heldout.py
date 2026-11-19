"""The held-out benchmark proteins: every protein NOT homologous to the training set under the split's own four-part rule.

``reference/tsuboyama_rule_flags.csv`` (``leak_audit/21_tsuboyama_rule.py``) holds, for every ProteinGym assay, Domainome domain, MGnify scaffold and Rocklin design, the four values of the rule that built the
Homology-Restricted MegaScale split (``docs/SPLITTING_AND_HOMOLOGY_AUDIT.md``) against the training libraries and whether any criterion is met:

    MMseqs2 E-value <= 1e-3   |   identity > 0.40 (BLOSUM62 local, matches / shorter length)   |   Foldseek E-value <= 1e-3   |   Foldseek TM-score > 0.70 (shorter-chain normalised)

Training only (the validation libraries are not used).  Every analysis that scores one of these benchmarks must leave the flagged proteins out through this module, so that the filter is the same
everywhere.  The thermostability sets and the functional DMS targets were quarantined against both the training and the validation libraries when the split was built and need no filter here.

    from msr_analysis.heldout import excluded, heldout_mask
    bad = excluded('proteingym')          # set of DMS ids
"""
import pandas as pd

from .registry import REFERENCE

DATASETS = ('proteingym', 'domainome', 'mgnify', 'rocklin_denovo')
_CACHE = {}


def flags():
    if 'f' not in _CACHE:
        f = pd.read_csv(REFERENCE / 'tsuboyama_rule_flags.csv', dtype={'item_id': str})
        _CACHE['f'] = f
    return _CACHE['f']


def excluded(dataset):
    """Item ids (DMS id / domain id / code_wt) of ``dataset`` that are homologous to the training set under the four-part rule."""
    assert dataset in DATASETS, dataset
    f = flags()
    return set(f.loc[(f['dataset'] == dataset) & f['homologous'], 'item_id'])


def heldout_mask(dataset, ids):
    """Boolean numpy array over ``ids``: True where the protein is held out (not flagged)."""
    bad = excluded(dataset)
    return pd.Series([str(i) for i in ids]).map(lambda i: i not in bad).to_numpy()


def criteria(dataset):
    """Per-item table of the four values and flags (item_id as index)."""
    f = flags()
    return f[f['dataset'] == dataset].set_index('item_id')
