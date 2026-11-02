"""``--val_four_pass``: the four-pass thermodynamic cycle of a double, read on the independent masking of both adapters.

    path 1 = WT(a) + MT(B|A)      path 2 = WT(b) + MT(A|B)      mean of the two == the reported 0.5 * (WT + MT) of that regime
    gap    = path 1 - path 2      the cycle's closing error

On the stand-in backbone of test_pass_masking.py (logits depend on the context a pass can see), with the model's per-library logit cache
switched on the way validation switches it on, these tests pin the arithmetic and the claim that makes the strategy cheap: every unique
masked sequence is computed once per library.
"""
import types

import pytest
import torch

from esm.utils.constants import esm3 as C

from esm_msr.training import ESM3EpistasisLightningModule as M

from test_pass_masking import MASK, _double, _model

INDEP = {'wt': 'independent', 'mt': 'independent'}


def _cat(items):
    return {k: torch.cat([it[k] for it in items]) for k in items[0]}


def _single(wt_seq=(3, 4, 5, 6, 7), site=1, wt=8, mut=9):
    seq = torch.tensor([list(wt_seq)])
    mt = seq.clone()
    mt[0, site] = mut
    return {'wt_sequence_tokens': seq, 'mt_sequence_tokens': mt, 'mut_pos': torch.tensor([[site, 0]]),
            'wt_id': torch.tensor([[wt, 0]]), 'mt_id': torch.tensor([[mut, 0]]), 'mut_mask': torch.tensor([[True, False]])}


def _stub(model):
    return types.SimpleNamespace(model=model)


def _four(m, batch):
    with torch.inference_mode():
        return M._four_pass(_stub(m), batch)


def test_the_mean_of_the_two_paths_is_the_reported_prediction_and_the_gap_is_their_difference():
    m, b = _model(), _double()
    m.eval()
    c = _four(m, b)
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    path1, path2 = wt[0] + mt[1], wt[1] + mt[0]
    assert c['comb'][0] == pytest.approx(float(0.5 * (path1 + path2)), rel=1e-5)
    assert c['gap'][0] == pytest.approx(float(path1 - path2), rel=1e-5)
    assert c['gap'][0] != 0.0                                         # the two ways round the cycle disagree when the partner matters
    assert c['wt'][0] == pytest.approx(float(wt.sum())) and c['mt'][0] == pytest.approx(float(mt.sum()))


def test_it_is_the_independent_independent_regime_of_forward_batch():
    m = _model()
    m.eval()
    b = _cat([_double(), _single(), _double(b=(3, 8, 20))])
    c = _four(m, b)
    ref = m.forward_batch(b, mask_strategy=INDEP)
    for k, r in (('wt', 'wt_lora_pred'), ('mt', 'mt_lora_pred'), ('comb', 'combined_pred')):
        assert torch.allclose(torch.as_tensor(c[k]), ref[r].float(), atol=1e-5), k


def test_a_single_has_no_cycle():
    m = _model()
    m.eval()
    c = _four(m, _cat([_single(), _double()]))
    assert torch.isnan(torch.as_tensor(c['gap'][0])) and not torch.isnan(torch.as_tensor(c['gap'][1]))


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
    c = _four(m, b)
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    p1 = (0.5 * wt[0] + 0.1) + (0.3 * mt[1] - 0.2)
    p2 = (0.5 * wt[1] + 0.1) + (0.3 * mt[0] - 0.2)
    assert c['gap'][0] == pytest.approx(float(p1 - p2), rel=1e-5)
    assert c['comb'][0] == pytest.approx(float(0.5 * (p1 + p2)), rel=1e-5)        # per-site calibration sums to the double's calibration


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
    c1 = _four(m, b)
    # WT: the wild-type sequence with site 1 masked, with site 3 masked = 2.  MT: (A1, site 3 masked), (A2, site 3 masked), (B1, site 1 masked),
    # (B2, site 1 masked) = 4.  Naive: 4 doubles x 4 passes = 16.  The WT head's other two regimes add one forward each: the unmasked
    # wild-type sequence, and the marginal one with both sites masked.
    assert sum(rows) == 6 + 1 + 1, rows
    n = len(rows)
    c2 = _four(m, b)                                                    # a second visit to the library costs a lookup
    assert len(rows) == n
    for k in c1:
        assert torch.allclose(torch.as_tensor(c1[k]), torch.as_tensor(c2[k]), equal_nan=True)


def test_the_cache_does_not_change_the_numbers():
    b = _grid()
    b['pdb'] = ['LIB'] * 4
    m1, m2 = _model(), _model()
    m1.eval(), m2.eval()
    m1._masked_logit_cache = {}                                         # cached
    del b['pdb']                                                        # uncached: no library id, so every row is a forward
    c_plain = _four(m2, b)
    b['pdb'] = ['LIB'] * 4
    c_cached = _four(m1, b)
    for k in c_plain:
        assert torch.allclose(torch.as_tensor(c_plain[k]), torch.as_tensor(c_cached[k]), equal_nan=True), k


def test_the_wt_head_is_also_scored_unmasked_and_marginal_and_the_ensemble_is_their_mean_with_the_cycle():
    m = _model()
    m.eval()
    b = _cat([_double(), _single(), _double(b=(3, 8, 20))])
    c = _four(m, b)
    un = m.forward_partitioned(b, pass_type='wt', mask_strategy=None)['pred_calibrated']
    mg = m.forward_partitioned(b, pass_type='wt', mask_strategy='marginal')['pred_calibrated']
    assert torch.allclose(torch.as_tensor(c['wt_unmasked']), un.float()) and torch.allclose(torch.as_tensor(c['wt_marginal']), mg.float())
    assert torch.allclose(torch.as_tensor(c['ens']), 0.5 * (torch.as_tensor(c['comb']) + torch.as_tensor(c['wt_unmasked'])))
    assert not torch.allclose(torch.as_tensor(c['wt_unmasked']), torch.as_tensor(c['wt_marginal']))          # three regimes, three different scores
