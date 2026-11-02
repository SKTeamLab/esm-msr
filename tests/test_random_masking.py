"""``--mask_strategy_<pass> random``: every mutated site is masked independently with probability ``--mask_prob`` on a TRAINING pass.

The consolidation of the masked and the unmasked setup into one: a double then meets all four combinations of masked and visible sites, so
one model covers both regimes. The readout is the plain one (mt - wt logit at each mutated site) on the partly masked sequence. The
stand-in backbone is the one test_pass_masking.py uses (logits depend on the context a pass can see).
"""
import os
import sys
import tempfile
from collections import Counter

import pytest
import torch
import yaml

from esm_msr import config, inference
from esm_msr.models import pass_mask_strategy

from test_pass_masking import MASK, _double, _model

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def _llr(m, batch, pass_type, strategy):
    return m.forward_partitioned(batch, pass_type=pass_type, mask_strategy=strategy)['unsummed_llr'][0]


def _recording(m):
    """Wrap the stand-in backbone so each forward records the sequence it was given."""
    seen, inner = [], m._get_esm3_outputs

    def outputs(seq, *a, **k):
        seen.append(seq.clone())
        return inner(seq, *a, **k)
    m._get_esm3_outputs = outputs
    return seen


# ---------------------------------------------------------------------------------------------------------------- the flags

def test_random_is_a_third_strategy_with_its_probability_defaulting_to_half():
    a = parse('--mask_strategy_wt', 'random', '--mask_strategy_mt', 'random')
    assert (a.mask_strategy_wt, a.mask_strategy_mt, a.mask_prob) == ('random', 'random', 0.5)
    assert parse('--mask_strategy_mt', 'random', '--mask_prob', '0.3').mask_prob == 0.3
    assert parse().mask_prob == 0.5 and parse().mask_strategy_wt is None
    assert pass_mask_strategy({'mask_strategy_wt': 'random'}, 'wt') == 'random'
    for bad in ('-0.1', '1.5'):
        with pytest.raises(SystemExit):
            parse('--mask_prob', bad)


def test_a_checkpoint_trained_with_random_masks_is_scored_unmasked_by_default():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'hparams.yaml')
        yaml.safe_dump({'mask_strategy_wt': 'random', 'mask_strategy_mt': 'random'}, open(p, 'w'))
        assert inference.parse_hparams_to_lora_config(p)['mask_strategy'] is None
        yaml.safe_dump({'mask_strategy_wt': 'random', 'mask_strategy_mt': 'independent'}, open(p, 'w'))
        assert inference.parse_hparams_to_lora_config(p)['mask_strategy'] == {'wt': None, 'mt': 'independent'}


# ---------------------------------------------------------------------------------------------------------------- what a pass sees

@pytest.mark.parametrize('pass_type', ['wt', 'mt'])
def test_probability_one_is_marginal_and_probability_zero_is_unmasked(pass_type):
    m, b = _model(), _double()
    m.train()
    m.random_mask_prob = 1.0
    assert torch.allclose(_llr(m, b, pass_type, 'random'), _llr(m, b, pass_type, 'marginal'))
    m.random_mask_prob = 0.0
    assert torch.allclose(_llr(m, b, pass_type, 'random'), _llr(m, b, pass_type, None))


@pytest.mark.parametrize('pass_type', ['wt', 'mt'])
def test_an_eval_pass_runs_unmasked_whatever_the_probability(pass_type):
    m, b = _model(), _double()
    m.eval()
    m.random_mask_prob = 1.0
    seen = _recording(m)
    assert torch.allclose(_llr(m, b, pass_type, 'random'), _llr(m, b, pass_type, None))
    assert all(int((s == MASK).sum()) == 0 for s in seen)


def test_only_mutated_sites_are_masked_each_with_probability_p_independently():
    torch.manual_seed(0)
    m, b = _model(), _double()
    m.train()
    m.random_mask_prob = 0.5
    seen = _recording(m)
    base = b['mt_sequence_tokens'][0]
    for _ in range(600):
        m.forward_partitioned(b, pass_type='mt', mask_strategy='random')
    pos = b['mut_pos'][0].tolist()
    others = [i for i in range(base.numel()) if i not in pos]
    assert all(torch.equal(s[0, others], base[others]) for s in seen)             # nothing but a mutated site is ever touched
    masked = torch.stack([s[0, pos] == MASK for s in seen])                        # [600, 2]
    assert masked.float().mean().item() == pytest.approx(0.5, abs=0.06)
    combos = Counter(map(tuple, masked.tolist()))
    for c in [(False, False), (False, True), (True, False), (True, True)]:         # all four regimes occur, about a quarter each
        assert combos[c] / 600 == pytest.approx(0.25, abs=0.07), combos


def test_the_two_passes_draw_their_own_masks():
    torch.manual_seed(1)
    m, b = _model(), _double()
    m.train()
    seen = _recording(m)
    wt_masks, mt_masks = [], []
    for _ in range(60):
        m.forward_partitioned(b, pass_type='wt', mask_strategy='random')
        m.forward_partitioned(b, pass_type='mt', mask_strategy='random')
        wt_masks.append(seen[-2][0, b['mut_pos'][0]] == MASK)
        mt_masks.append(seen[-1][0, b['mut_pos'][0]] == MASK)
    assert any(not torch.equal(x, y) for x, y in zip(wt_masks, mt_masks))


def test_a_visible_partner_lets_the_mt_pass_see_what_it_was_mutated_to_and_a_masked_one_does_not():
    """The point of 0 < p < 1 on the MT pass: sometimes the readout of A sees B's mutant residue (an interaction can show), sometimes not."""
    torch.manual_seed(2)
    m = _model()
    m.train()
    same = differ = 0
    for _ in range(200):
        state = torch.get_rng_state()
        l1 = float(_llr(m, _double(b=(3, 8, 9)), 'mt', 'random')[0])
        torch.set_rng_state(state)                                                  # the same mask draw for a double whose partner differs
        l2 = float(_llr(m, _double(b=(3, 8, 20)), 'mt', 'random')[0])
        same, differ = same + (l1 == pytest.approx(l2)), differ + (l1 != pytest.approx(l2))
    assert same > 0 and differ > 0                                                  # equal when B is masked, different when B is visible
