"""``--val_masking independent`` (the default): validation reads BOTH adapters with every mutated site masked on its own, the partner's mutant
residue visible, and reports the closing error of the double's thermodynamic cycle.

    path 1 = WT(a) + MT(B|A)      path 2 = WT(b) + MT(A|B)      mean of the two == the reported 0.5 * (WT + MT) of that regime
    gap    = path 1 - path 2      the cycle's closing error

On the stand-in backbone of test_pass_masking.py (logits depend on the context a pass can see), with the model's per-library logit cache
switched on the way validation switches it on, these tests pin the arithmetic, the regime validation_step reads, and the claim that makes the
strategy cheap: every unique masked sequence is computed once per library.
"""
import sys
import types
from collections import defaultdict

import pytest
import torch

from esm_msr import config
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


def _gap(m, batch):
    with torch.inference_mode():
        return M._cycle_gap(_stub(m), batch)


def test_the_mean_of_the_two_paths_is_the_reported_prediction_and_the_gap_is_their_difference():
    m, b = _model(), _double()
    m.eval()
    gap = _gap(m, b)
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    path1, path2 = wt[0] + mt[1], wt[1] + mt[0]
    comb = m.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    assert float(comb[0]) == pytest.approx(float(0.5 * (path1 + path2)), rel=1e-5)
    assert gap[0] == pytest.approx(float(path1 - path2), rel=1e-5)
    assert gap[0] != 0.0                                              # the two ways round the cycle disagree when the partner matters


def test_a_single_has_no_cycle():
    m = _model()
    m.eval()
    gap = _gap(m, _cat([_single(), _double()]))
    assert torch.isnan(torch.as_tensor(gap[0])) and not torch.isnan(torch.as_tensor(gap[1]))


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
    gap = _gap(m, b)
    wt = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['unsummed_llr'][0]
    mt = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['unsummed_llr'][0]
    p1 = (0.5 * wt[0] + 0.1) + (0.3 * mt[1] - 0.2)
    p2 = (0.5 * wt[1] + 0.1) + (0.3 * mt[0] - 0.2)
    assert gap[0] == pytest.approx(float(p1 - p2), rel=1e-5)
    comb = m.forward_batch(b, mask_strategy=INDEP)['combined_pred']
    assert float(comb[0]) == pytest.approx(float(0.5 * (p1 + p2)), rel=1e-5)      # per-site calibration sums to the double's calibration


def _grid():
    """Four doubles of ONE library: mutations A1/A2 at site 1 crossed with B1/B2 at site 3."""
    return _cat([_double(a=(1, 8, 9), b=(3, 8, 9)), _double(a=(1, 8, 9), b=(3, 8, 20)),
                 _double(a=(1, 8, 21), b=(3, 8, 9)), _double(a=(1, 8, 21), b=(3, 8, 20))])


def test_every_unique_masked_sequence_is_computed_once_per_library_and_the_gap_rides_on_validations_own_passes():
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
    n = len(rows)
    g1 = _gap(m, b)                                                      # the cycle gap costs a lookup
    g2 = _gap(m, b)
    assert len(rows) == n and torch.allclose(torch.as_tensor(g1), torch.as_tensor(g2), equal_nan=True)


def test_the_cache_does_not_change_the_numbers():
    b = _grid()
    b['pdb'] = ['LIB'] * 4
    m1, m2 = _model(), _model()
    m1.eval(), m2.eval()
    m1._masked_logit_cache = {}                                         # cached
    del b['pdb']                                                        # uncached: no library id, so every row is a forward
    g_plain = _gap(m2, b)
    b['pdb'] = ['LIB'] * 4
    g_cached = _gap(m1, b)
    assert torch.allclose(torch.as_tensor(g_plain), torch.as_tensor(g_cached), equal_nan=True)


# ---------------------------------------------------------------------------------------------------------------- which regime validation reads

class _HP(dict):
    __getattr__ = dict.__getitem__


def _validation_stub(model, val_masking, train_wt=None, train_mt=None, unmasked=False):
    s = types.SimpleNamespace(model=model, hparams=_HP(mt_direct_pred=False, val_wt_only=False, val_cycle_passes=False, val_masking=val_masking,
                                                       val_unmasked=unmasked),
                              validation_step_outputs=defaultdict(list), link_head=None)
    s.val_independent = val_masking == 'independent'
    s.mask_wt, s.mask_mt = train_wt, train_mt
    s.val_mask_wt, s.val_mask_mt = ('independent', 'independent') if s.val_independent else (train_wt, train_mt)
    s.val_mask_strategy = {'wt': s.val_mask_wt, 'mt': s.val_mask_mt}
    s._cycle_gap = types.MethodType(M._cycle_gap, s)
    return s


def _validate(val_masking, unmasked=False, **train):
    m = _model()
    m.eval()
    seen = []
    inner = m.forward_partitioned

    def spy(batch, pass_type, mask_strategy=None, **k):
        seen.append((pass_type, mask_strategy))
        return inner(batch, pass_type=pass_type, mask_strategy=mask_strategy, **k)
    m.forward_partitioned = spy
    s = _validation_stub(m, val_masking, unmasked=unmasked, **train)
    b = _cat([_double(), _single()])
    b['ddG'] = torch.zeros(2)
    M.validation_step(s, b, 0)
    return seen, s.validation_step_outputs[0][0]


def test_validation_reads_every_site_masked_on_its_own_by_default_whatever_the_run_trained_with():
    seen, rec = _validate('independent')                 # an unmasked-trained run: validation still reads the independent regime
    assert seen and all(st == 'independent' for _, st in seen), seen
    assert 'gap_scores' in rec and len(rec['gap_scores']) == 2


def test_val_masking_train_is_the_old_validation_with_no_cycle_gap():
    seen, rec = _validate('train', train_wt='marginal', train_mt='independent')
    assert ('wt', 'marginal') in seen and ('mt', 'independent') in seen and ('wt', 'independent') not in seen, seen
    assert 'gap_scores' not in rec
    seen, rec = _validate('train')                       # nothing set: unmasked, as before
    assert seen and all(st is None for _, st in seen), seen


def test_the_flag_defaults_to_independent_and_the_four_pass_flag_is_gone():
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
    old = sys.argv
    try:
        sys.argv = argv
        assert config.parse_arguments().val_masking == 'independent'
        sys.argv = argv + ['--val_masking', 'train']
        assert config.parse_arguments().val_masking == 'train'
        for bad in (['--val_masking', 'marginal'], ['--val_four_pass']):
            sys.argv = argv + bad
            with pytest.raises(SystemExit):
                config.parse_arguments()
    finally:
        sys.argv = old


def test_val_unmasked_adds_the_regime_with_neither_adapter_masked_and_leaves_the_default_alone():
    seen0, rec0 = _validate('independent')
    seen, rec = _validate('independent', unmasked=True)
    assert 'combU_scores' not in rec0
    assert [x for x in seen if x[1] is None] and {x for x in seen if x[1] == 'independent'}, seen             # both regimes were read
    assert rec['wtU_scores'].shape == rec['mtU_scores'].shape == rec['combU_scores'].shape == (2,)
    assert (rec['comb_scores'] == rec0['comb_scores']).all()                                                  # the default numbers do not move
    assert not (rec['combU_scores'] == rec['comb_scores']).all()                                              # and the unmasked ones differ from them


def test_the_unmasked_flag_is_off_by_default():
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
    old = sys.argv
    try:
        sys.argv = argv
        assert config.parse_arguments().val_unmasked is False
        sys.argv = argv + ['--val_unmasked']
        assert config.parse_arguments().val_unmasked is True
    finally:
        sys.argv = old
