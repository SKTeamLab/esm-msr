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


def test_independent_skips_singles_on_second_slot_and_deduplicates_states_with_exact_grads():
    """Mixed single+double batch from one library: singles have no i=1 evaluation, singles at the same site share 1
    masked forward, doubles sharing (partner_pos, partner_aa, mask_pos) share 1 masked forward, and parameter
    gradients match per-item evaluation to floating-point precision."""
    m = _model()
    scale_param = torch.nn.Parameter(torch.tensor(1.5))
    call_rows = []

    def outputs_grad(seq, coords=None, tokens=None, plddt=None, active_model=None):
        call_rows.append(int(seq.shape[0]))
        visible = torch.where(seq == MASK, torch.zeros_like(seq), seq).float().sum(dim=1) * scale_param
        return _Out(m.base.view(1, 1, V) * visible.view(-1, 1, 1).expand(-1, seq.shape[1], 1))

    m._get_esm3_outputs = outputs_grad

    # 4 items on WT=[3,4,5,6,7]:
    #   item 0: single at pos 1 (8->9)   -> masks pos 1 on WT
    #   item 1: single at pos 1 (8->12)  -> masks pos 1 on WT (DUPLICATE of item 0!)
    #   item 2: double at (1:8->9,  3:8->15) -> i=0 masks 1 with pos 3=15; i=1 masks 3 with pos 1=9
    #   item 3: double at (1:8->14, 3:8->15) -> i=0 masks 1 with pos 3=15 (DUPLICATE of item 2 i=0!); i=1 masks 3 with pos 1=14
    wt = torch.tensor([[3, 4, 5, 6, 7]] * 4)
    mt = wt.clone()
    mt[0, 1] = 9
    mt[1, 1] = 12
    mt[2, 1], mt[2, 3] = 9, 15
    mt[3, 1], mt[3, 3] = 14, 15
    batch = {
        'wt_sequence_tokens': wt,
        'mt_sequence_tokens': mt,
        'mut_pos': torch.tensor([[1, 0], [1, 0], [1, 3], [1, 3]]),
        'wt_id': torch.tensor([[8, 0], [8, 0], [8, 8], [8, 8]]),
        'mt_id': torch.tensor([[9, 0], [12, 0], [9, 15], [14, 15]]),
        'mut_mask': torch.tensor([[True, False], [True, False], [True, True], [True, True]]),
        'pdb': ['1ABC'] * 4,
    }

    out = m.forward_partitioned(batch, pass_type='mt', mask_strategy='independent')
    # Total valid (b,i) = 6, but 2 are duplicates -> only 4 unique sequences evaluated in 1 call (instead of 2*4=8 rows)
    assert call_rows == [4]

    weights = torch.tensor([1.0, 2.0, -0.5, 0.75])
    loss_batched = (out['pred_raw'] * weights).sum()
    loss_batched.backward()
    grad_batched = scale_param.grad.clone()

    # Compare against running each item b=0..3 individually
    scale_param.grad = None
    indiv_preds = []
    for b_i in range(4):
        sub = {k: (v[b_i:b_i+1] if torch.is_tensor(v) else [v[b_i]]) for k, v in batch.items()}
        indiv_preds.append(m.forward_partitioned(sub, pass_type='mt', mask_strategy='independent')['pred_raw'][0])
    indiv_stack = torch.stack(indiv_preds)
    assert torch.allclose(out['pred_raw'], indiv_stack, atol=1e-5)
    loss_indiv = (indiv_stack * weights).sum()
    loss_indiv.backward()
    assert torch.allclose(grad_batched, scale_param.grad, atol=1e-5)


def test_score_screening_batch_supports_asymmetric_per_pass_masking():
    """score_screening_batch with {'wt': None, 'mt': 'independent'} matches forward_batch."""
    m = _model()
    m.eval()
    m.calibration_head_wt = m.calibration_head_mt = None

    wt_1 = torch.tensor([[3, 4, 5, 6, 7]])
    mut_pos = torch.tensor([[1, 0], [1, 3], [1, 3]])
    wt_id = torch.tensor([[4, 0], [4, 6], [4, 6]])
    mt_id = torch.tensor([[9, 0], [9, 15], [14, 15]])
    mut_mask = torch.tensor([[True, False], [True, True], [True, True]])

    strat = {'wt': None, 'mt': 'independent'}
    scr = m.score_screening_batch(
        wt_sequence_tokens=wt_1, mut_pos=mut_pos, wt_id=wt_id, mt_id=mt_id,
        mut_mask=mut_mask, mask_strategy=strat, batch_size=8, quiet=True
    )

    wt_B = wt_1.expand(3, -1).clone()
    mt_B = wt_B.clone()
    for b in range(3):
        for i in torch.where(mut_mask[b])[0]:
            mt_B[b, mut_pos[b, i]] = mt_id[b, i]
    fb = m.forward_batch({
        'wt_sequence_tokens': wt_B, 'mt_sequence_tokens': mt_B,
        'mut_pos': mut_pos, 'wt_id': wt_id, 'mt_id': mt_id, 'mut_mask': mut_mask,
    }, mask_strategy=strat)

    assert torch.allclose(scr['wt_lora_pred'], fb['wt_lora_pred'], atol=1e-5)
    assert torch.allclose(scr['mt_lora_pred'], fb['mt_lora_pred'], atol=1e-5)
    assert torch.allclose(scr['combined_pred'], fb['combined_pred'], atol=1e-5)
