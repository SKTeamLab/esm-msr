"""Block training (``esm_msr.blocks`` and the ``_block_*`` steps of the Lightning module).

The claim everything rests on: a pair block's 19 x 19 cells, assembled from 2 wild-type and 38 mutant passes, equal what the per-item
``independent`` strategy reads for each double on its own; a singles block's [B, 20] cell equals the per-item masked single.
The backbone is the context-dependent stand-in of test_pass_masking.py (its logits depend on the tokens a pass can see).
"""
import types

import numpy as np
import pytest
import torch

from esm_msr import blocks, config
from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M

from esm_msr.link import SaturatingLink
from _stubs import IdentityLink
from test_pass_masking import _model

CANON = torch.tensor(blocks.CANON_IDS)
V = 40


def _stand_in():
    m = _model()
    m.canonical_idx_tensor = CANON
    return m


# a 9-token protein of canonical residues; sites 2 and 5 are the pair under test
SEQ = torch.tensor([CANON[k].item() for k in (0, 3, 5, 7, 9, 11, 13, 15, 17)])
I, J = 2, 5
WT_I, WT_J = int(SEQ[I]), int(SEQ[J])


def _item(pos, mts, y, **kw):
    pos = list(pos)
    d = {'subset_type': 'single' if len(pos) == 1 else 'double', 'mut_pos': np.array(pos), 'wt_id': np.array([int(SEQ[p]) for p in pos]),
         'mt_id': np.array(mts), 'ddG': y, 'cens': 0, 'cens_bound': float('nan'), 'dG_meas': 1.0, 'dG_wt': 2.0, 'y_floor': 0.0,
         'ddG_A': float('nan'), 'ddG_B': float('nan'),
         'wt_sequence_tokens': SEQ.numpy(), 'coords_orig': np.zeros((1, 9, 37, 3), dtype=np.float32), 'structure_tokens_orig': np.zeros(9, dtype=np.int64)}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------------------------------------------- the label tensors

def test_the_alphabet_matches_the_models_and_the_wild_type_reads_zero_in_it():
    assert blocks.AAS == 'ACDEFGHIKLMNPQRSTVWY' and len(set(blocks.CANON_IDS)) == 20
    assert all(blocks.TOKEN_TO_AA[t] == k for k, t in enumerate(blocks.CANON_IDS))


def test_build_library_lays_singles_and_doubles_into_dense_tensors():
    a, b = blocks.CANON_IDS[1], blocks.CANON_IDS[2]
    items = [_item([I], [a], -1.0), _item([I], [b], 0.5), _item([J], [a], 2.0, cens=-1, cens_bound=0.3),
             _item([J, I], [b, a], 0.7, ddG_A=0.1, ddG_B=0.2)]                # given in reversed position order: the builder sorts
    lib = blocks.build_library('LIB', items)
    assert lib.single_pos.tolist() == [I, J] and lib.pairs == [(I, J)]
    assert lib.singles['y'].shape == (2, 20) and lib.pair_data['y'].shape == (1, 20, 20)
    assert lib.singles['y'][0, 1] == -1.0 and lib.singles['y'][0, 2] == 0.5 and np.isnan(lib.singles['y'][0, 3])
    assert lib.singles['cens'][1, 1] == -1 and lib.singles['bound'][1, 1] == pytest.approx(0.3) and lib.singles['cens'][0, 1] == 0
    # the double's cell is [residue at i, residue at j] whichever order the item listed them in
    assert lib.pair_data['y'][0, 1, 2] == pytest.approx(0.7) and np.isnan(lib.pair_data['y'][0, 2, 1])
    assert lib.pair_data['yA'][0, 1, 2] == pytest.approx(0.1) and lib.pair_data['yB'][0, 1, 2] == pytest.approx(0.2)
    assert lib.wt_tok[I] == WT_I and lib.wt_tok[J] == WT_J and lib.coords.shape == (1, 1, 9, 37, 3) and lib.structure_tokens.shape == (1, 9)


def test_non_canonical_and_triple_items_are_left_out_and_an_empty_library_is_none():
    assert blocks.build_library('X', [_item([I], [99], 1.0)]) is None
    assert blocks.build_library('X', [dict(_item([I], [blocks.CANON_IDS[0]], 1.0), subset_type='native_cond')]) is None


# ---------------------------------------------------------------------------------------------------------------- the passes of a block

def test_a_singles_block_masks_one_site_per_row():
    s = blocks.single_sequences(SEQ, torch.tensor([1, 4, 7]))
    assert s.shape == (3, 9)
    for r, p in enumerate((1, 4, 7)):
        assert s[r, p] == blocks.MASK and (s[r] != blocks.MASK).sum() == 8 and torch.equal(s[r][s[r] != blocks.MASK], SEQ[SEQ.new_tensor([k for k in range(9) if k != p])])


def test_a_pair_block_has_two_wt_passes_and_thirty_eight_mutant_passes():
    wt_rows, row_seqs, col_seqs, Xa, Ya = blocks.pair_sequences(SEQ, I, J, WT_I, WT_J)
    assert wt_rows.shape == (2, 9) and row_seqs.shape == (19, 9) and col_seqs.shape == (19, 9) and len(Xa) == len(Ya) == 19
    assert wt_rows[0, I] == blocks.MASK and wt_rows[0, J] == WT_J and wt_rows[1, J] == blocks.MASK and wt_rows[1, I] == WT_I
    assert (row_seqs[:, J] == blocks.MASK).all() and (col_seqs[:, I] == blocks.MASK).all()
    assert row_seqs[:, I].tolist() == [blocks.CANON_IDS[a] for a in Xa] and WT_I not in row_seqs[:, I].tolist()
    assert col_seqs[:, J].tolist() == [blocks.CANON_IDS[a] for a in Ya]


def _reference_cell(m, x, y):
    """What the per-item independent strategy reads for the double (x at I, y at J): the WT sum and the MT sum."""
    mt_seq = SEQ.clone()
    mt_seq[I], mt_seq[J] = blocks.CANON_IDS[x], blocks.CANON_IDS[y]
    b = {'wt_sequence_tokens': SEQ.unsqueeze(0), 'mt_sequence_tokens': mt_seq.unsqueeze(0), 'mut_pos': torch.tensor([[I, J]]),
         'wt_id': torch.tensor([[WT_I, WT_J]]), 'mt_id': torch.tensor([[blocks.CANON_IDS[x], blocks.CANON_IDS[y]]]), 'mut_mask': torch.tensor([[True, True]])}
    w = m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['pred_calibrated'][0]
    t = m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['pred_calibrated'][0]
    return float(w), float(t)


def test_every_cell_of_a_pair_block_equals_the_per_item_independent_readout():
    m = _stand_in()
    wt_rows, row_seqs, col_seqs, Xa, Ya = blocks.pair_sequences(SEQ, I, J, WT_I, WT_J)
    llr = lambda seqs, pos, wt: m.forward_site_llr(seqs, torch.as_tensor(pos), torch.as_tensor(wt), 'wt')
    llr_w = llr(wt_rows, [I, J], [WT_I, WT_J])
    llr_m = llr(torch.cat([row_seqs, col_seqs]), [J] * 19 + [I] * 19, [WT_J] * 19 + [WT_I] * 19)
    wt_raw, mt_raw = blocks.assemble_pair(llr_w[0], llr_w[1], llr_m[:19], llr_m[19:], Xa, Ya)
    assert wt_raw.shape == mt_raw.shape == (19, 19)
    for r, x in enumerate(Xa):
        for c, y in enumerate(Ya):
            w, t = _reference_cell(m, x, y)
            assert float(wt_raw[r, c]) == pytest.approx(w, rel=1e-5, abs=1e-4), (x, y)
            assert float(mt_raw[r, c]) == pytest.approx(t, rel=1e-5, abs=1e-4), (x, y)


def test_a_masked_site_reads_the_wild_type_as_exactly_zero_and_matches_the_per_item_single():
    m = _stand_in()
    pos = torch.tensor([1, 4])
    llr = m.forward_site_llr(blocks.single_sequences(SEQ, pos), pos, SEQ[pos], 'wt')
    assert llr.shape == (2, 20)
    for r, p in enumerate((1, 4)):
        wt_aa = blocks.TOKEN_TO_AA[int(SEQ[p])]
        assert float(llr[r, wt_aa]) == 0.0
        for a in (0, 5, 19):
            mt = SEQ.clone(); mt[p] = blocks.CANON_IDS[a]
            b = {'wt_sequence_tokens': SEQ.unsqueeze(0), 'mt_sequence_tokens': mt.unsqueeze(0), 'mut_pos': torch.tensor([[p]]),
                 'wt_id': SEQ[p].view(1, 1), 'mt_id': torch.tensor([[blocks.CANON_IDS[a]]]), 'mut_mask': torch.tensor([[True]])}
            ref = float(m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['pred_calibrated'][0])
            assert float(llr[r, a]) == pytest.approx(ref, rel=1e-5, abs=1e-4)


# ---------------------------------------------------------------------------------------------------------------- the sampler and dataset

def _lib(name, n_pos=5, pairs=((2, 5),)):
    items = [_item([p], [blocks.CANON_IDS[a]], float(p + a)) for p in range(n_pos) for a in range(3) if blocks.CANON_IDS[a] != int(SEQ[p])]
    items += [_item(list(pr), [blocks.CANON_IDS[1], blocks.CANON_IDS[2]], 0.5) for pr in pairs]
    return blocks.build_library(name, items)


def test_an_epoch_is_every_singles_block_then_every_pair_block_and_is_reproducible():
    libs = [_lib('A', 5, ((2, 5),)), _lib('B', 7, ((1, 3), (2, 4)))]
    s = blocks.BlockSampler(libs, 2, seed=3)
    d = list(s)
    assert len(d) == len(s) == (3 + 4) + 3 and s.n_single_blocks() == 7 and s.n_pair_blocks() == 3
    kinds = [x[0] for x in d]
    assert kinds == ['s'] * 7 + ['p'] * 3
    assert sorted(p for x in d if x[0] == 's' and x[1] == 0 for p in x[2]) == list(range(5))        # every position once
    assert all(len(x[2]) <= 2 for x in d if x[0] == 's')
    assert list(blocks.BlockSampler(libs, 2, seed=3)) == d
    s.set_epoch(1)
    e = list(s)
    assert e != d                                                                                     # a new epoch regroups the positions ...
    cover = lambda xs: sorted((x[1], p) for x in xs if x[0] == 's' for p in x[2])
    assert cover(e) == cover(d) and sorted(x for x in e if x[0] == 'p') == sorted(x for x in d if x[0] == 'p')     # ... of the same cells


def test_a_block_is_one_library_and_the_dataset_returns_tensors():
    libs = [_lib('A', 5), _lib('B', 6)]
    ds = blocks.BlockDataset(libs)
    sb = ds[('s', 1, (0, 2, 3))]
    assert sb['block'] == 'single' and sb['pdb'] == 'B' and sb['positions'].shape == (3,) and sb['y'].shape == (3, 20)
    assert sb['seq'].shape == (9,) and sb['wt_ids'].tolist() == [int(SEQ[p]) for p in sb['positions'].tolist()]
    pb = ds[('p', 0, 0)]
    assert pb['block'] == 'pair' and pb['positions'].tolist() == [2, 5] and pb['y'].shape == (20, 20) and pb['yA'].shape == (20, 20)


# ---------------------------------------------------------------------------------------------------------------- the losses

class _Head(torch.nn.Module):
    use_scale = True

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def bias_for(self, n):
        return self.bias * n.to(self.bias.dtype)

    def forward(self, y, n=None):
        return self.scale * y + self.bias_for(n)


class _Net(torch.nn.Module):
    """Linear stand-in: the raw LLR of row r, residue a is w_side * (a + 1 + 0.1 r); a wild-type cell is not special-cased here."""

    def __init__(self):
        super().__init__()
        self.w_wt, self.w_mt = torch.nn.Parameter(torch.tensor(0.1)), torch.nn.Parameter(torch.tensor(0.1))
        self.calibration_head_wt, self.calibration_head_mt = _Head(), _Head()
        self.calls = []

    def forward_site_llr(self, seqs, read_pos, wt_at, side, coords=None, struct=None):
        R = seqs.shape[0]
        self.calls.append((side, R, torch.is_grad_enabled()))
        w = self.w_wt if side == 'wt' else self.w_mt
        return w * (torch.arange(20).float().view(1, 20) + 1 + 0.1 * torch.arange(R).float().view(R, 1))


def _hp(**kw):
    d = dict(tf_mix=1.0, lambda_reg_wt=1.0, lambda_rank_wt=1.0, lambda_reg_mt=1.0, lambda_rank_mt=1.0, censor_threshold=None, block_positions=8,
             joint_doubles='off')
    d.update(kw)

    class HP(dict):
        __getattr__ = dict.__getitem__

        def get(self, k, default=None):
            return dict.get(self, k, default)
    return HP(d)


def _module(**hp):
    s = types.SimpleNamespace()
    s.hparams = _hp(**hp)
    s.model = _Net()
    s.link_head = IdentityLink()
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.manual_backward = lambda loss, retain_graph=False: loss.backward()
    for name in ('_reg_loss', '_compute_rank_loss', '_block_cal', '_block_llr', '_block_ctx', '_block_reg_rank', '_block_backward',
                 '_block_forced', '_block_singles', '_block_pair', '_block_step'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    return s


def _blk(kind, **over):
    lib = _lib('A', 5, ((2, 5),))
    ds = blocks.BlockDataset([lib])
    b = ds[('s', 0, (0, 1, 2))] if kind == 'single' else ds[('p', 0, 0)]
    b.update(over)
    return b


def test_a_singles_block_trains_both_heads_and_its_wt_loss_is_the_mean_squared_error_on_the_observed_cells():
    s = _module(lambda_rank_wt=0.0, lambda_rank_mt=0.0)
    b = _blk('single')
    logs = M._block_step(s, b)
    y = b['y']
    obs = torch.isfinite(y)
    llr = s.model.w_wt.detach() * (torch.arange(20).float().view(1, 20) + 1 + 0.1 * torch.arange(3).float().view(3, 1))
    assert logs['L_reg_wt'] == pytest.approx(float(((llr - y)[obs] ** 2).mean()), rel=1e-5)
    assert s.model.w_wt.grad is not None and s.model.w_mt.grad is not None and float(s.model.w_wt.grad.abs()) > 0
    assert [c[0] for c in s.model.calls] == ['wt', 'mt'] and all(c[1] == 3 for c in s.model.calls)            # B rows each, 2 passes


def test_a_singles_block_is_ranked_as_one_list_of_b_times_twenty_cells():
    s = _module(lambda_reg_wt=0.0, lambda_reg_mt=0.0)
    seen = []
    class R(ListMLELoss):
        def forward_censored(self, p, t, m, c):
            seen.append(tuple(p.shape))
            return super().forward_censored(p, t, m, c)
    s.crit_rank_wt = s.crit_rank_mt = R()
    M._block_step(s, _blk('single'))
    assert seen and all(sh == (1, 60) for sh in seen)


def test_a_pair_block_uses_forty_passes_and_regresses_the_cycle_prediction():
    s = _module(lambda_rank_mt=0.0)
    b = _blk('pair')
    logs = M._block_step(s, b)
    assert sorted((c[0], c[1]) for c in s.model.calls) == [('mt', 38), ('wt', 2)]
    assert 'L_reg_combined' in logs and logs['L_reg_combined'] > 0
    assert s.model.w_mt.grad is not None and (s.model.w_wt.grad is None or float(s.model.w_wt.grad.abs()) == 0.0)   # joint off: WT half detached


def test_joint_doubles_make_the_pair_blocks_wt_half_live_and_tf_mix_one_takes_the_gradient_away():
    for mix, expect in ((0.0, True), (1.0, False)):
        s = _module(joint_doubles='both', tf_mix=mix, lambda_rank_mt=0.0)
        b = _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2))              # every cell has a usable additive label
        M._block_step(s, b)
        g = s.model.w_wt.grad
        assert (g is not None and float(g.abs()) > 0) == expect, (mix, g)


def test_pair_rank_lists_cover_every_cell_once_in_rows_or_columns():
    s = _module(lambda_reg_mt=0.0, block_positions=8)
    seen = []
    class R(ListMLELoss):
        def forward_censored(self, p, t, m, c):
            seen.append(p.shape[1])
            return super().forward_censored(p, t, m, c)
    s.crit_rank_mt = R()
    torch.manual_seed(0)
    sizes = set()
    for _ in range(6):
        seen.clear()
        M._block_step(s, _blk('pair'))
        sizes.add(tuple(sorted(seen)))
        assert sum(seen) == 361                                                                  # 19 x 19 cells, each in exactly one list
    assert all(max(t) <= 8 * 19 for t in sizes)                                                  # at most B rows (or columns) per list


def test_a_frozen_wt_head_is_read_without_a_graph_and_gets_no_loss():
    s = _module()
    s.peft_manager.wt_path_is_frozen = True
    M._block_step(s, _blk('single'))
    assert s.model.calls[0] == ('wt', 3, False) and s.model.w_wt.grad is None


def test_the_flags():
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
    import sys
    old = sys.argv
    try:
        sys.argv = argv + ['--block_training', '--block_positions', '16']
        a = config.parse_arguments()
        assert a.block_training is True and a.block_positions == 16
        sys.argv = argv
        assert config.parse_arguments().block_training is False
        for bad in (['--block_training', '--mt_direct_pred'], ['--block_training', '--adapter_mode', 'fused'], ['--block_training', '--block_positions', '0']):
            sys.argv = argv + bad
            with pytest.raises(SystemExit):
                config.parse_arguments()
    finally:
        sys.argv = old


def test_the_real_link_gives_finite_gradients_although_most_cells_are_unobserved():
    """Unobserved cells have no dG_wt: the link must not see NaN there, or 0 * NaN in backward poisons every parameter."""
    for kind in ('single', 'pair'):
        s = _module(joint_doubles='both', tf_mix=0.5)
        s.link_head = SaturatingLink(tau=0.59, floor=0.0)
        b = _blk(kind, yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2)) if kind == 'pair' else _blk(kind)
        M._block_step(s, b)
        grads = [p.grad for p in list(s.model.parameters()) if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads), kind
        assert float(s.model.w_mt.grad.abs()) > 0
