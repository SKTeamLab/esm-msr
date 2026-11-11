"""Validation reads BOTH adapters with every mutated site masked on its own, the partner's mutant residue visible.

    path 1 = WT(a) + MT(B|A)      path 2 = WT(b) + MT(A|B)      mean of the two == the reported 0.5 * (WT + MT) of that regime

On the stand-in backbone of test_pass_masking.py (logits depend on the context a pass can see), with the model's per-library logit cache switched on the way validation
switches it on, these tests pin the arithmetic, the claim that makes the strategy cheap (every unique masked sequence is computed once per library), the regimes
validation_step reads, and the rules for the unmasked copy (only when a head trains on visible sites).
"""
import sys
import types
from collections import defaultdict

import pytest
import torch

from esm_msr import config
from esm_msr.training import ESM3EpistasisLightningModule as M, validation_regimes

from test_pass_masking import MASK, _double, _model

INDEP = {'wt': 'independent', 'mt': 'independent'}
NONE = {'wt': None, 'mt': None}


def _cat(items):
    return {k: torch.cat([it[k] for it in items]) for k in items[0]}


def _single(wt_seq=(3, 4, 5, 6, 7), site=1, wt=8, mut=9):
    seq = torch.tensor([list(wt_seq)])
    mt = seq.clone()
    mt[0, site] = mut
    return {'wt_sequence_tokens': seq, 'mt_sequence_tokens': mt, 'mut_pos': torch.tensor([[site, 0]]),
            'wt_id': torch.tensor([[wt, 0]]), 'mt_id': torch.tensor([[mut, 0]]), 'mut_mask': torch.tensor([[True, False]])}


def test_the_mean_of_the_two_paths_is_the_reported_prediction():
    m, b = _model(), _double()
    m.eval()
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    path1, path2 = wt[0] + mt[1], wt[1] + mt[0]
    comb = m.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    assert float(comb[0]) == pytest.approx(float(0.5 * (path1 + path2)), rel=1e-5)
    assert float(path1 - path2) != 0.0                                  # the two ways round the cycle disagree when the partner matters


def test_calibration_is_applied_per_site_and_the_paths_still_average_to_the_combined_prediction():
    class Cal(torch.nn.Module):
        def __init__(self, s, b):
            super().__init__()
            self.s, self.b = s, b

        def forward(self, y, n_mut=None):
            return self.s * y + self.b * (1 if n_mut is None else n_mut)
    m, b = _model(), _double()
    m.eval()
    m.calibration_head_wt, m.calibration_head_mt = Cal(0.5, 0.1), Cal(0.3, -0.2)
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    p1 = (0.5 * wt[0] + 0.1) + (0.3 * mt[1] - 0.2)
    p2 = (0.5 * wt[1] + 0.1) + (0.3 * mt[0] - 0.2)
    comb = m.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    assert float(comb[0]) == pytest.approx(float(0.5 * (p1 + p2)), rel=1e-5)      # per-site calibration sums to the double's calibration


def _grid():
    """Four doubles of ONE library: mutations A1/A2 at site 1 crossed with B1/B2 at site 3."""
    return _cat([_double(a=(1, 8, 9), b=(3, 8, 9)), _double(a=(1, 8, 9), b=(3, 8, 20)),
                 _double(a=(1, 8, 21), b=(3, 8, 9)), _double(a=(1, 8, 21), b=(3, 8, 20))])


def test_every_unique_masked_sequence_is_computed_once_per_library():
    m = _model()
    m.eval()
    rows = []
    inner = m._get_esm3_outputs

    def counting(seq, *a, **k):
        rows.append(int(seq.shape[0]))
        return inner(seq, *a, **k)
    m._get_esm3_outputs = counting
    m._masked_logit_cache = {}
    b = _grid()
    b['pdb'] = ['LIB'] * 4
    with torch.inference_mode():
        m.forward_batch(b, mask_strategy=INDEP)                         # what validation_step reads
    # WT: the wild-type sequence with site 1 masked, with site 3 masked = 2.  MT: (A1, site 3 masked), (A2, site 3 masked), (B1, site 1 masked),
    # (B2, site 1 masked) = 4.  Naive: 4 doubles x 4 passes = 16.
    assert sum(rows) == 6, rows


def test_the_cache_does_not_change_the_numbers():
    b = _grid()
    m1, m2 = _model(), _model()
    m1.eval(), m2.eval()
    m1._masked_logit_cache = {}                                         # cached: the batch carries a library id
    b['pdb'] = ['LIB'] * 4
    with torch.inference_mode():
        cached = m1.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    del b['pdb']                                                        # uncached: no library id, so every row is a forward
    with torch.inference_mode():
        plain = m2.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    assert torch.allclose(cached, plain)


# ---------------------------------------------------------------------------------------------------------------- which regimes validation reads

class _HP(dict):
    __getattr__ = dict.__getitem__

    def get(self, k, default=None):
        return dict.get(self, k, default)


def _hp(**kw):
    return _HP({'block_mask_wt': 1.0, 'block_mask_mt': 1.0, **kw})


def test_the_main_regime_is_masked_and_the_unmasked_copy_comes_only_when_a_head_trains_on_visible_sites():
    assert validation_regimes(_hp()) == [('', INDEP)]
    assert validation_regimes(_hp(block_mask_mt=0.0)) == [('', INDEP), ('unmasked_', NONE)]
    assert validation_regimes(_hp(block_mask_wt=0.0)) == [('', INDEP), ('unmasked_', NONE)]
    assert validation_regimes(_hp(block_mask_wt=0.5, block_mask_mt=0.75)) == [('', INDEP), ('unmasked_', NONE)]


def _stub(model, regimes):
    s = types.SimpleNamespace(model=model, validation_step_outputs=defaultdict(list), obs_link=lambda v: v, val_regimes=regimes)
    s.val_mask_strategy = regimes[0][1]
    return s


def _validate(hp):
    m = _model()
    m.eval()
    seen = []
    inner = m.forward_partitioned

    def spy(batch, pass_type, mask_strategy=None, **k):
        seen.append((pass_type, mask_strategy))
        return inner(batch, pass_type=pass_type, mask_strategy=mask_strategy, **k)
    m.forward_partitioned = spy
    s = _stub(m, validation_regimes(hp))
    b = _cat([_double(), _single()])
    b['ddG'] = torch.zeros(2)
    M.validation_step(s, b, 0)
    return seen, s.validation_step_outputs[0][0]


def test_validation_reads_every_site_masked_on_its_own_whatever_the_run_trained_with():
    seen, rec = _validate(_hp())
    assert seen and all(st == 'independent' for _, st in seen), seen
    assert 'regimes' not in rec


def test_the_unmasked_copy_reads_neither_adapter_masked_and_leaves_the_main_regime_alone():
    seen0, rec0 = _validate(_hp())
    seen, rec = _validate(_hp(block_mask_mt=0.0))
    assert {x for x in seen if x[1] is None} and {x for x in seen if x[1] == 'independent'}, seen             # both regimes were read
    e = rec['regimes']['unmasked_']
    assert e['wt_scores'].shape == e['mt_scores'].shape == e['comb_scores'].shape == (2,)
    assert (rec['comb_scores'] == rec0['comb_scores']).all()                                                  # the main numbers do not move
    assert not (e['comb_scores'] == rec['comb_scores']).all()                                                 # and the unmasked ones differ from them


def test_the_retired_validation_flags_are_gone():
    for gone in ('--val_masking_strategy', '--val_unmasked', '--val_masked', '--val_cycle_passes', '--monitor_metric', '--skip_val'):
        with pytest.raises(SystemExit):
            config.parse_arguments(['--experiment_name', 'x', gone, 'masked'] if gone == '--val_masking_strategy' else ['--experiment_name', 'x', gone])
