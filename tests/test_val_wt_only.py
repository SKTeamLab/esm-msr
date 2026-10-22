"""`--val_wt_only`: validate the WT head alone in a run whose MT adapter is not trained.

A WT-only sweep freezes the MT adapter at its initialisation, so every MT prediction is a forward through noise.
`--val_wt_only` drops those passes - the MT half of the main validation forward and the MT probe of the cycle passes
- and reports the WT head under all three names (mt == comb == wt), which is a true statement about such a model
rather than an undefined column or a noisy one. The ctx level is the WT adapter read in reverse (`wt_rev`), so it
survives: it needs `--val_cycle_passes`, not the MT head.
"""
import types

import torch

from esm_msr import config
from esm_msr.models import MSRModel
from esm_msr.training import ESM3EpistasisLightningModule as M

from _stubs import IdentityLink

import sys

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_the_flag_parses():
    assert parse().val_wt_only is False
    a = parse('--val_wt_only')
    assert a.val_wt_only is True and a.val_cycle_passes is True      # ctx still available


# ----------------------------------------------------------------------------------------------------------------
# the cycle pass can drop its MT probe
# ----------------------------------------------------------------------------------------------------------------

class _CycleModel:
    training = False

    def __init__(self):
        self.calls = []

    def forward_partitioned(self, batch, pass_type, mask_strategy=None, cached_wt_esm3=None):
        self.calls.append(pass_type)
        return {'pred_calibrated': torch.full((2,), 1.0 if pass_type == 'wt' else 2.0)}


def _batch(B=2):
    f = lambda v: torch.full((B,), float(v))
    return {'wt_sequence_tokens': torch.zeros(B, 4, dtype=torch.long),
            'mt_sequence_tokens': torch.ones(B, 4, dtype=torch.long),
            'wt_id': torch.zeros(B, 4, dtype=torch.long), 'mt_id': torch.zeros(B, 4, dtype=torch.long),
            'ddG': torch.linspace(-2.0, 0.0, B), 'dddG': f(0.0), 'dG_wt': f(2.0), 'y_floor': f(-0.2),
            'cens': torch.zeros(B, dtype=torch.long), 'cens_src': torch.zeros(B, dtype=torch.long),
            'mut_mask': torch.ones(B, 1, dtype=torch.bool), 'subset_type': ['single'] * B,
            'pdb': ['1ABC'] * B, 'mutations': ['A:1:V'] * B}


def test_forward_cycle_can_skip_the_mt_probe():
    m = _CycleModel()
    out = MSRModel.forward_cycle(m, _batch(), include_mt=False)
    assert set(out) == {'wt_rev'} and m.calls == ['wt']
    m2 = _CycleModel()
    out2 = MSRModel.forward_cycle(m2, _batch(), include_mt=True)
    assert set(out2) == {'wt_rev', 'mt_fwd'} and m2.calls == ['wt', 'mt']


# ----------------------------------------------------------------------------------------------------------------
# validation_step honours it
# ----------------------------------------------------------------------------------------------------------------

class _HP(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def get(self, k, default=None):
        return dict.get(self, k, default)


class _ValModel:
    def __init__(self, wt_only):
        self.wt_only = wt_only
        self.calls = []

    def forward_batch(self, batch, mask_strategy=None):
        self.calls.append('batch')
        v = torch.tensor([0.5, 0.5])
        return {'wt_lora_pred': v, 'mt_lora_pred': v + 1.0, 'combined_pred': v + 0.5,
                'epi_pred': v}

    def forward_partitioned(self, batch, pass_type, mask_strategy=None, cached_wt_esm3=None):
        self.calls.append(pass_type)
        return {'pred_calibrated': torch.tensor([0.5, 0.5])}

    def forward_cycle(self, batch, mask_strategy=None, include_mt=True):
        self.calls.append('cycle_mt' if include_mt else 'cycle_wt')
        out = {'wt_rev': torch.tensor([0.25, 0.25])}
        if include_mt:
            out['mt_fwd'] = torch.tensor([0.75, 0.75])
        return out


def _stub(val_wt_only=True, val_cycle_passes=True):
    s = types.SimpleNamespace()
    s.hparams = _HP(val_wt_only=val_wt_only, val_cycle_passes=val_cycle_passes)
    s.model = _ValModel(val_wt_only)
    s.link_head = IdentityLink()
    s.mask_wt = s.mask_mt = None
    s.mask_strategy = {'wt': None, 'mt': None}
    from collections import defaultdict
    s.validation_step_outputs = defaultdict(list)
    return s


def _run(stub):
    M.validation_step(stub, _batch(), 0)
    return stub.model.calls, stub.validation_step_outputs[0][0]


def test_wt_only_runs_one_forward_and_no_mt_cycle():
    calls, rec = _run(_stub(val_wt_only=True))
    assert calls == ['wt', 'cycle_wt'], calls                 # no 'batch' forward, no MT probe
    # the WT head is reported under all three names
    assert (rec['wt_scores'] == rec['mt_scores']).all()
    assert (rec['wt_scores'] == rec['comb_scores']).all()
    # the ctx control survives, the MT probe does not
    assert 'wt_rev_scores' in rec and 'mt_fwd_scores' not in rec


def test_the_default_still_runs_both_passes():
    calls, rec = _run(_stub(val_wt_only=False))
    assert calls == ['batch', 'cycle_mt'], calls
    assert 'wt_rev_scores' in rec and 'mt_fwd_scores' in rec
    assert not (rec['wt_scores'] == rec['mt_scores']).all()    # the MT half is real here


def test_wt_only_without_the_cycle_passes_has_no_ctx_either():
    calls, rec = _run(_stub(val_wt_only=True, val_cycle_passes=False))
    assert calls == ['wt'] and 'wt_rev_scores' not in rec and 'mt_fwd_scores' not in rec
