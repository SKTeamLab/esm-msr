"""``--substitution_calibration`` (a learned from x to offset per head, in raw-LLR units) and ``--last_n_layers_{wt,mt}`` (LoRA only in the last N blocks).

Stand-in backbone as in test_pass_masking: logits[b, i, v] = v * (sum of the visible tokens), so every LLR is a known number. The canonical residues are tokens 4..23.
"""
import re

import pytest
import torch

from esm_msr.models import MSRModel

V = 40
CAN = list(range(4, 24))


class _Out:
    def __init__(self, logits):
        self.sequence_logits = logits


def _model(table_wt=None, table_mt=None):
    m = MSRModel.__new__(MSRModel)
    torch.nn.Module.__init__(m)
    m.log_likelihood, m.adapter_mode = False, 'dual'
    m.peft_wt, m.peft_mt = object(), object()
    m.native_cond_mt_input = False
    m.register_buffer('canonical_idx_tensor', torch.tensor(CAN))
    t2c = torch.full((V,), -1, dtype=torch.long)
    t2c[torch.tensor(CAN)] = torch.arange(20)
    m.register_buffer('token_to_canonical', t2c, persistent=False)
    m.substitution_calibration = table_wt is not None
    if m.substitution_calibration:
        m.calibration_head_wt_sub = torch.nn.Parameter(table_wt.clone())
        m.calibration_head_mt_sub = torch.nn.Parameter(table_mt.clone())
        m.register_buffer('_sub_offdiag', 1.0 - torch.eye(20), persistent=False)
    m.base = torch.arange(V, dtype=torch.float32)
    m._get_esm3_outputs = lambda seq, *a, **k: _Out(m.base.view(1, 1, V) * seq.float().sum(dim=1).view(-1, 1, 1).expand(-1, seq.shape[1], 1))
    return m


def _double():
    wt = torch.tensor([[4, 5, 6, 7, 8]])
    mt = wt.clone(); mt[0, 1], mt[0, 3] = 9, 12
    return {'wt_sequence_tokens': wt, 'mt_sequence_tokens': mt, 'mut_pos': torch.tensor([[1, 3]]),
            'wt_id': torch.tensor([[5, 7]]), 'mt_id': torch.tensor([[9, 12]]), 'mut_mask': torch.tensor([[True, True]])}


def _tables():
    g = torch.Generator().manual_seed(0)
    return torch.randn(20, 20, generator=g), torch.randn(20, 20, generator=g)


@pytest.mark.parametrize('side', ['wt', 'mt'])
def test_table_is_added_per_mutation_and_diagonal_is_ignored(side):
    tw, tm = _tables()
    tw.fill_diagonal_(5.0); tm.fill_diagonal_(5.0)
    plain, with_t = _model(), _model(tw, tm)
    b = _double()
    a = plain.forward_partitioned(b, pass_type=side, mask_strategy=None)['unsummed_llr'][0]
    c = with_t.forward_partitioned(b, pass_type=side, mask_strategy=None)['unsummed_llr'][0]
    T = tw if side == 'wt' else tm
    expect = torch.stack([T[5 - 4, 9 - 4], T[7 - 4, 12 - 4]])
    assert torch.allclose(c - a, expect, atol=1e-4)
    # the calibration head's input is the sum, so the per-item shift is the sum of the two entries
    assert torch.allclose(with_t.forward_partitioned(b, pass_type=side, mask_strategy=None)['pred_raw'] - plain.forward_partitioned(b, pass_type=side, mask_strategy=None)['pred_raw'], expect.sum().view(1), atol=1e-4)


def test_zero_table_changes_nothing():
    z = torch.zeros(20, 20)
    b = _double()
    for side in ('wt', 'mt'):
        assert torch.equal(_model().forward_partitioned(b, pass_type=side, mask_strategy=None)['unsummed_llr'],
                           _model(z, z).forward_partitioned(b, pass_type=side, mask_strategy=None)['unsummed_llr'])


@pytest.mark.parametrize('side', ['wt', 'mt'])
def test_site_llr_row_shifts_by_the_from_row_and_the_wild_type_column_stays_zero(side):
    tw, tm = _tables()
    plain, with_t = _model(), _model(tw, tm)
    seqs = torch.tensor([[4, 5, 6, 7, 8], [4, 5, 6, 7, 8]])
    read = torch.tensor([1, 3]); wt_at = torch.tensor([5, 7])
    a = plain.forward_site_llr(seqs, read, wt_at, side)
    c = with_t.forward_site_llr(seqs, read, wt_at, side)
    T = tw if side == 'wt' else tm
    T = T * (1 - torch.eye(20))
    assert torch.allclose(c - a, T[wt_at - 4], atol=1e-4)
    assert torch.all(c[torch.arange(2), wt_at - 4] == 0)


def test_table_parameters_are_calibration_parameters_for_the_optimizer_and_checkpoints():
    tw, tm = _tables()
    names = [n for n, _ in _model(tw, tm).named_parameters()]
    assert names == ['calibration_head_wt_sub', 'calibration_head_mt_sub']
    wt, mt = names
    assert 'calibration' in wt and 'mt' not in wt.lower()           # training.configure_optimizers routes on these substrings
    assert 'calibration' in mt and 'mt' in mt.lower()


@pytest.mark.parametrize('n,blocks', [(0, range(48)), (4, range(44, 48)), (48, range(48))])
def test_last_n_layers_restricts_the_lora_targets(n, blocks):
    m = MSRModel.__new__(MSRModel)
    cfg = m._create_lora_config({'lora_rank': 2, 'lora_alpha': 4, 'last_n_layers': n, 'target_mode': 'expanded'})
    rx = re.compile(cfg.target_modules)
    hit = lambda i, leaf: rx.match(f'transformer.blocks.{i}.{leaf}') is not None
    for i in range(48):
        for leaf in ('attn.layernorm_qkv.1', 'ffn.1', 'ffn.3'):
            assert hit(i, leaf) == (i in blocks), (i, leaf)
