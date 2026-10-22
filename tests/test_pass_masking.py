"""Per-pass sequence masking: ``--mask_strategy_wt`` / ``--mask_strategy_mt``.

The motivating pairing is marginal on the WT pass and independent on the MT pass. On the real ``forward_partitioned`` with a stand-in backbone
whose logits depend on the context it can see, these tests pin what each strategy lets a pass see: ``marginal`` hides every mutated site
(so a double's two readouts cannot depend on each other: no interaction by construction), ``independent`` masks one site at a time and leaves
the partner visible (the conditional masked marginal, the readout of ddG(A|B)).
"""
import types

import pytest
import torch

from esm.utils.constants import esm3 as C

from esm_msr import config
from esm_msr.models import MSRModel, _per_pass, pass_mask_strategy

MASK = C.SEQUENCE_MASK_TOKEN
V = 40                                                    # vocabulary of the stand-in


class _Out:
    def __init__(self, logits):
        self.sequence_logits = logits


def _model():
    """An MSRModel without a backbone: ``_get_esm3_outputs`` is replaced by logits[b, i, v] = base[v] * (sum of the visible tokens)."""
    m = MSRModel.__new__(MSRModel)
    torch.nn.Module.__init__(m)
    m.log_likelihood, m.adapter_mode = False, 'dual'
    m.peft_wt, m.peft_mt = object(), object()
    m.native_cond_mt_input = False
    m.base = torch.arange(V, dtype=torch.float32)

    def outputs(seq, coords=None, tokens=None, plddt=None, active_model=None):
        visible = torch.where(seq == MASK, torch.zeros_like(seq), seq).float().sum(dim=1)            # [B]
        return _Out(m.base.view(1, 1, V) * visible.view(-1, 1, 1).expand(-1, seq.shape[1], 1))
    m._get_esm3_outputs = outputs
    return m


def _double(wt_seq=(3, 4, 5, 6, 7), a=(1, 8, 9), b=(3, 8, 9)):
    """A double at 0-based sequence positions a[0] and b[0]: wt residue ids and mutant ids are a[1]/a[2] and b[1]/b[2] (stand-in tokens)."""
    wt = torch.tensor([list(wt_seq)])
    mt = wt.clone()
    pa, pb = a[0], b[0]
    mt[0, pa], mt[0, pb] = a[2], b[2]
    return {'wt_sequence_tokens': wt, 'mt_sequence_tokens': mt, 'mut_pos': torch.tensor([[pa, pb]]),
            'wt_id': torch.tensor([[a[1], b[1]]]), 'mt_id': torch.tensor([[a[2], b[2]]]), 'mut_mask': torch.tensor([[True, True]])}


def _llr(m, batch, pass_type, strategy):
    return m.forward_partitioned(batch, pass_type=pass_type, mask_strategy=strategy)['unsummed_llr'][0]


def test_pass_strategies_resolve_independently():
    hp = {'mask_strategy_wt': 'marginal', 'mask_strategy_mt': 'independent'}
    assert pass_mask_strategy(hp, 'wt') == 'marginal' and pass_mask_strategy(hp, 'mt') == 'independent'
    assert pass_mask_strategy({'mask_strategy_mt': 'none'}, 'mt') is None
    assert pass_mask_strategy({}, 'wt') is None and pass_mask_strategy({'mask_strategy_wt': None}, 'wt') is None


def test_a_dict_or_a_string_reaches_each_pass():
    assert _per_pass({'wt': 'marginal', 'mt': None}, 'wt') == 'marginal' and _per_pass({'wt': 'marginal', 'mt': None}, 'mt') is None
    assert _per_pass('independent', 'wt') == 'independent' == _per_pass('independent', 'mt')       # inference passes one string
    assert _per_pass(None, 'mt') is None


def test_the_flags_exist_per_pass_and_the_global_one_is_retired():
    a = config.parse_arguments(['--experiment_name', 'x', '--mask_strategy_mt', 'independent'])
    assert a.mask_strategy_mt == 'independent' and a.mask_strategy_wt is None
    with pytest.raises(SystemExit):
        config.parse_arguments(['--experiment_name', 'x', '--mask_strategy', 'marginal'])
    with pytest.raises(SystemExit):
        config.parse_arguments(['--experiment_name', 'x', '--mask_strategy_mt', 'chain'])


def test_marginal_hides_both_sites_from_each_other():
    m, b = _model(), _double()
    visible_both_masked = float(b['mt_sequence_tokens'][0].sum() - b['mt_sequence_tokens'][0, 1] - b['mt_sequence_tokens'][0, 3])
    llr = _llr(m, b, 'mt', 'marginal')
    # llr_i = (base[mt_id] - base[wt_id]) * context with BOTH sites masked: the same context for both
    assert llr[0] == pytest.approx((9 - 8) * visible_both_masked) and llr[1] == pytest.approx((9 - 8) * visible_both_masked)


def test_independent_leaves_the_partner_visible_on_the_mt_pass():
    m, b = _model(), _double()
    seq = b['mt_sequence_tokens'][0]
    total = float(seq.sum())
    llr = _llr(m, b, 'mt', 'independent')
    # site A's forward masks only A: the context still contains B's MUTANT residue (and vice versa)
    assert llr[0] == pytest.approx((9 - 8) * (total - float(seq[1])))
    assert llr[1] == pytest.approx((9 - 8) * (total - float(seq[3])))
    assert llr[0] != pytest.approx(float(_llr(m, b, 'mt', 'marginal')[0]))


def test_independent_on_the_mt_pass_changes_when_the_partners_residue_changes():
    """The point of the pairing: the readout of A depends on what B was mutated to, i.e. the model CAN represent an interaction."""
    m = _model()
    b1, b2 = _double(b=(3, 8, 9)), _double(b=(3, 8, 20))
    assert float(_llr(m, b1, 'mt', 'independent')[0]) != pytest.approx(float(_llr(m, b2, 'mt', 'independent')[0]))
    # ... and under marginal it cannot: with both sites masked the partner's residue is invisible
    assert float(_llr(m, b1, 'mt', 'marginal')[0]) == pytest.approx(float(_llr(m, b2, 'mt', 'marginal')[0]))


def test_independent_on_the_wild_type_is_the_sum_of_the_singles():
    """Reading a double on the WILD-TYPE sequence one site at a time sees only wild type at the partner: exactly each single's own readout."""
    m, b = _model(), _double()
    wt = b['wt_sequence_tokens'][0]
    total = float(wt.sum())
    llr = _llr(m, b, 'wt', 'independent')
    assert llr[0] == pytest.approx((9 - 8) * (total - float(wt[1]))) and llr[1] == pytest.approx((9 - 8) * (total - float(wt[3])))


def test_unmasked_is_unchanged_by_the_per_pass_plumbing():
    m, b = _model(), _double()
    plain = m.forward_partitioned(b, pass_type='mt')['pred_raw']
    assert torch.equal(plain, m.forward_partitioned(b, pass_type='mt', mask_strategy=None)['pred_raw'])


def test_forward_batch_applies_each_pass_its_own_strategy():
    m = _model()
    m.eval()
    seen = {}

    def fake(batch, pass_type, cached_wt_esm3=None, mask_strategy=None):
        seen[pass_type] = mask_strategy
        z = torch.zeros(1)
        return {'pred_calibrated': z, 'pred_raw': z}
    m.forward_partitioned = fake
    m.calibration_head_wt = m.calibration_head_mt = None
    m.forward_batch({'mut_mask': torch.ones(1, 2, dtype=torch.bool)}, mask_strategy={'wt': 'marginal', 'mt': 'independent'})
    assert seen == {'wt': 'marginal', 'mt': 'independent'}
    m.forward_batch({'mut_mask': torch.ones(1, 2, dtype=torch.bool)}, mask_strategy='independent')
    assert seen == {'wt': 'independent', 'mt': 'independent'}
