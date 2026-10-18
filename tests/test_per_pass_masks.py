"""--mask_strategy_wt / --mask_strategy_mt, and the adapter override they are built for.

Why per-pass masking exists: the single --mask_strategy masks BOTH passes, so under 'marginal' the MT pass on a
double never sees the partner's residue and cannot represent a residue-residue interaction at all. Measured on
the marginal-trained checkpoints: cell_rank and flipacc exactly 0 and 0.5 for every view. Masking the WT pass
(pair context, no interaction by construction) while the MT pass reads the conditional masked marginal is the
combination that has never been run.
"""
import sys
import types

import pytest
import torch

from esm_msr.models import MASK_STRATEGIES, MSRModel, _per_pass, pass_mask_strategy

SCALE = {'wt': 1.0, 'mt': 2.0}
V = 33


class Head(torch.nn.Module):
    def __init__(self, s, b):
        super().__init__()
        self.s, self.b = s, b

    def forward(self, y, n_mut=None):
        return self.s * y + self.b * (1 if n_mut is None else 0)


class FakeModel:
    """Only what forward_partitioned reads: a backbone whose logit at a site depends on the token shown there and
    on the other sites' tokens, so a masked partner is visibly different from a mutant one."""
    adapter_mode = 'dual'
    mask_structure = False

    def __init__(self):
        self.calibration_head_wt, self.calibration_head_mt = Head(0.5, 0.1), Head(0.25, -0.2)
        self.peft_wt, self.peft_mt = 'wt', 'mt'
        self.used = []

    def _process_logits(self, x):
        return x

    def _get_esm3_outputs(self, seq, coords, struct, plddt, active_model=None):
        self.used.append(active_model)
        onehot = torch.nn.functional.one_hot(seq, V).float() * SCALE[active_model]
        ctx = seq.float().sum(1, keepdim=True) - seq.float()
        return types.SimpleNamespace(sequence_logits=onehot + 0.01 * ctx.unsqueeze(-1) * torch.arange(V).float())


WT_SEQ = [4, 5, 6, 7, 8, 9]
A, B = (1, 12), (3, 15)


def item(muts):
    mt = list(WT_SEQ)
    for p, t in muts:
        mt[p] = t
    pos = [p for p, _ in muts] + [0] * (2 - len(muts))
    return {'wt_sequence_tokens': torch.tensor([WT_SEQ]), 'mt_sequence_tokens': torch.tensor([mt]),
            'mut_pos': torch.tensor([pos]), 'wt_id': torch.tensor([[WT_SEQ[p] for p in pos]]),
            'mt_id': torch.tensor([[t for _, t in muts] + [0] * (2 - len(muts))]),
            'mut_mask': torch.tensor([[True] * len(muts) + [False] * (2 - len(muts))]),
            'coords': None, 'structure_tokens': None, 'plddt': None}


def raw(batch, pass_type, mask=None, adapter=None, model=None):
    m = model or FakeModel()
    return MSRModel.forward_partitioned(m, batch, pass_type=pass_type, mask_strategy=mask, adapter=adapter)['pred_raw']


def test_a_pair_masked_double_is_the_same_input_whichever_sequence_it_came_from():
    """Both sites masked: the WT-sequence and MT-sequence inputs are identical, so the pass cannot know the
    partner's identity - the property that makes this view interaction-free by construction."""
    d = item([A, B])
    torch.testing.assert_close(raw(d, 'wt', 'marginal'), raw(d, 'mt', 'marginal', adapter='wt'), atol=0, rtol=0)


def test_independent_masking_on_the_mutated_sequence_does_see_the_partner():
    d, a, b = item([A, B]), item([A]), item([B])
    assert not torch.allclose(raw(d, 'mt', 'independent'), raw(a, 'mt', 'independent') + raw(b, 'mt', 'independent'), atol=1e-3)


def test_the_adapter_override_picks_the_lora_while_the_pass_picks_the_sequence():
    m = FakeModel()
    MSRModel.forward_partitioned(m, item([A]), pass_type='mt', adapter='wt')
    assert m.used == ['wt']


def test_without_an_override_the_adapter_follows_the_pass():
    m = FakeModel()
    MSRModel.forward_partitioned(m, item([A]), pass_type='mt')
    MSRModel.forward_partitioned(m, item([A]), pass_type='wt', mask_strategy='marginal')
    assert m.used == ['mt', 'wt']


def test_an_unknown_adapter_is_refused():
    with pytest.raises(AssertionError, match='adapter'):
        MSRModel.forward_partitioned(FakeModel(), item([A]), pass_type='wt', adapter='fused')


def test_a_wild_type_cache_cannot_serve_the_other_adapter():
    with pytest.raises(AssertionError, match='cached_wt_esm3'):
        MSRModel.forward_partitioned(FakeModel(), item([A]), pass_type='wt', adapter='mt', cached_wt_esm3={'seq': None})


@pytest.mark.parametrize('hp, want', [
    ({}, (None, None)),
    ({'mask_strategy': 'marginal'}, ('marginal', 'marginal')),                                    # the old shared flag
    ({'mask_strategy': 'marginal', 'mask_strategy_mt': 'independent'}, ('marginal', 'independent')),
    ({'mask_strategy': 'marginal', 'mask_strategy_wt': 'none'}, (None, 'marginal')),              # 'none' unmasks one pass
    ({'mask_strategy_wt': 'marginal', 'mask_strategy_mt': 'independent'}, ('marginal', 'independent')),
])
def test_each_pass_resolves_its_own_strategy(hp, want):
    assert (pass_mask_strategy(hp, 'wt'), pass_mask_strategy(hp, 'mt')) == want


def test_forward_batch_hands_each_pass_its_own_strategy():
    calls = []

    def fp(batch, pass_type, cached_wt_esm3=None, mask_strategy=None):
        calls.append((pass_type, mask_strategy))
        return {'pred_calibrated': torch.zeros(1), 'pred_raw': torch.zeros(1)}
    m = types.SimpleNamespace(training=False, adapter_mode='dual', forward_partitioned=fp,
                              hparams={}, combine=None)
    try:
        MSRModel.forward_batch(m, item([A]), mask_strategy={'wt': 'marginal', 'mt': 'independent'})
    except Exception:
        pass            # forward_batch does more than the two calls; only the strategies it passed matter here
    assert calls[:2] == [('wt', 'marginal'), ('mt', 'independent')]
    assert _per_pass(None, 'wt') is None and _per_pass('marginal', 'mt') == 'marginal'


def test_the_defaults_change_nothing():
    from esm_msr import config
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        a = vars(config.parse_arguments())
    finally:
        sys.argv = argv
    assert a['mask_strategy_wt'] is None and a['mask_strategy_mt'] is None
    assert pass_mask_strategy(a, 'wt') is None and pass_mask_strategy(a, 'mt') is None
    assert MASK_STRATEGIES == ('marginal', 'independent')
