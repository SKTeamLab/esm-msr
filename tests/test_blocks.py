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

from esm_msr.link import LeakyFloorLink, SaturatingLink
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
        out = w * (torch.arange(20).float().view(1, 20) + 1 + 0.1 * torch.arange(R).float().view(R, 1))
        wt_aa = torch.as_tensor([blocks.TOKEN_TO_AA[int(t)] for t in wt_at])
        return out * (torch.arange(20).view(1, 20) != wt_aa.view(R, 1)).float()                       # the wild type reads exactly 0


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
    for name in ('_reg_loss', '_compute_rank_loss', '_block_cal', '_block_cal_sg', '_block_s_wt', '_block_llr', '_block_ctx', '_block_reg_rank',
                 '_block_backward', '_block_forced', '_block_singles', '_block_singles_llr', '_block_pair', '_block_step', '_block_cond_legs'):
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


@pytest.mark.parametrize('ident,rows', [(False, 38), (True, 40)])
def test_a_pair_block_uses_thirty_eight_passes_forty_with_the_identity_and_regresses_the_cycle_prediction(ident, rows):
    s = _module(lambda_rank_mt=0.0, block_identity=ident)
    b = _blk('pair')
    logs = M._block_step(s, b)
    assert sorted((c[0], c[1]) for c in s.model.calls) == [('mt', rows), ('wt', 2)]
    assert 'L_reg_combined' in logs and logs['L_reg_combined'] > 0
    assert s.model.w_mt.grad is not None and (s.model.w_wt.grad is None or float(s.model.w_wt.grad.abs()) == 0.0)   # joint off: WT half detached


def test_joint_doubles_make_the_pair_blocks_wt_half_live_and_tf_mix_one_takes_the_gradient_away():
    for mix, expect in ((0.0, True), (1.0, False)):
        s = _module(joint_doubles='both', tf_mix=mix, lambda_rank_mt=0.0)
        b = _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2))              # every cell has a usable additive label
        M._block_step(s, b)
        g = s.model.w_wt.grad
        assert (g is not None and float(g.abs()) > 0) == expect, (mix, g)


def test_block_list_rows_cuts_the_lists_into_rows_of_positions():
    class R(ListMLELoss):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward_censored(self, p, t, m, c):
            self.seen.append(p.shape[1])
            return super().forward_censored(p, t, m, c)
    s = _module(lambda_reg_wt=0.0, lambda_reg_mt=0.0, block_list_rows=1)
    s.crit_rank_wt = s.crit_rank_mt = rec = R()
    M._block_step(s, _blk('single'))
    assert rec.seen == [20] * 6                                                    # 3 positions x 20 cells, per head, one list each
    s = _module(lambda_reg_wt=0.0, lambda_reg_mt=0.0, block_list_rows=2)
    s.crit_rank_wt = s.crit_rank_mt = rec = R()
    M._block_step(s, _blk('single'))
    assert sorted(rec.seen) == [20, 20, 40, 40]                                    # 3 positions in lists of up to 2
    s = _module(lambda_reg_mt=0.0, block_positions=8, block_list_rows=2)
    s.crit_rank_mt = rec = R()
    M._block_step(s, _blk('pair'))
    assert rec.seen == [40] * 10                                                    # the 20 x 20 matrix in 10 lists of 2 rows or columns


def test_rank_norm_ref_scales_a_list_to_the_reference_length_not_its_own():
    class Const:
        def forward_censored(self, p, t, m, c):
            return p.sum() * 0.0 + 7.0
    s = _module()
    n = 40
    mask = torch.ones(n, dtype=torch.bool)
    mask[:10] = False                                                              # 30 valid members
    args = (torch.randn(n), torch.arange(n).float(), mask, torch.zeros(n, dtype=torch.long), n, Const())
    own, _, _ = M._compute_rank_loss(s, *args)
    ref, _, _ = M._compute_rank_loss(s, *args, norm_to=16)
    assert float(own) == pytest.approx(7.0 * 40 / 30) and float(ref) == pytest.approx(7.0 * 16 / 30)


def test_the_list_flags_default_to_the_old_behaviour_and_are_validated():
    import sys
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', '--block_training']
    old = sys.argv
    try:
        sys.argv = argv
        a = config.parse_arguments()
        assert a.block_list_rows == 0 and a.block_rank_norm == 'sum'
        sys.argv = argv + ['--block_list_rows', '1', '--block_rank_norm', 'ref']
        a = config.parse_arguments()
        assert a.block_list_rows == 1 and a.block_rank_norm == 'ref'
        sys.argv = argv + ['--block_list_rows', '-1']
        with pytest.raises(SystemExit):
            config.parse_arguments()
    finally:
        sys.argv = old


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
        assert sum(seen) == 400                                                                  # 20 x 20 cells (the identity row and column too), each in exactly one list
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


@pytest.mark.parametrize('link', [lambda: SaturatingLink(tau=0.59, floor=0.0), lambda: LeakyFloorLink(tau=0.59, knee=0.0, alpha=0.10, clamp=-1.0)],
                         ids=['softplus', 'leaky'])
def test_the_real_link_gives_finite_gradients_although_most_cells_are_unobserved(link):
    """Unobserved cells have no dG_wt: the link must not see NaN there, or 0 * NaN in backward poisons every parameter."""
    for kind in ('single', 'pair'):
        s = _module(joint_doubles='both', tf_mix=0.5)
        s.link_head = link()
        b = _blk(kind, yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2)) if kind == 'pair' else _blk(kind)
        M._block_step(s, b)
        grads = [p.grad for p in list(s.model.parameters()) if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads), kind
        assert float(s.model.w_mt.grad.abs()) > 0


def test_block_tf_detach_wt_default_keeps_the_wt_head_out_of_the_combined_loss():
    for kind in ('single', 'pair'):
        s = _module(lambda_reg_wt=0.0, lambda_rank_wt=0.0, tf_mix=0.0)
        M._block_step(s, _blk(kind))
        assert s.model.w_wt.grad is None and float(s.model.w_mt.grad.abs()) > 0, kind
        s = _module(lambda_reg_wt=0.0, lambda_rank_wt=0.0, tf_mix=0.0, tf_detach_wt=False)
        M._block_step(s, _blk(kind))
        assert s.model.w_wt.grad is not None and float(s.model.w_wt.grad.abs()) > 0, kind


def test_block_clashes_with_the_per_item_only_losses_are_refused():
    import sys
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', '--block_training']
    old = sys.argv
    try:
        sys.argv = argv + ['--lambda_cond_mt', '7.5']                         # the conditional legs ARE defined for blocks
        assert config.parse_arguments().lambda_cond_mt == 7.5
        for bad in (['--reg_nll', 'tobit'], ['--wt_supervise_doubles'], ['--mt_single_loss_weight', '0.1']):
            sys.argv = argv + bad
            with pytest.raises(SystemExit):
                config.parse_arguments()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------------------------------------------- identity, conditional legs, chunks

class _Rec(ListMLELoss):
    """Records every list the rank loss is handed: predictions, labels and the member mask."""

    def __init__(self):
        super().__init__()
        self.lists = []

    def forward_censored(self, p, t, m, c):
        self.lists.append((p[0].detach().clone(), t[0].clone(), m[0].clone()))
        return super().forward_censored(p, t, m, c)


def test_pair_labels_put_the_singles_on_the_identity_row_and_column_and_zero_on_the_wild_type():
    b = _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2))
    wi, wj = blocks.TOKEN_TO_AA[WT_I], blocks.TOKEN_TO_AA[int(SEQ[5])]
    wi = blocks.TOKEN_TO_AA[int(SEQ[2])]
    lab, A, B, ident, full = blocks.pair_labels(b, list(range(20)), list(range(20)), wi, wj)
    assert lab['y'].shape == (20, 20) and ident.sum() == 39 and bool(ident[wi, wj])
    assert lab['y'][wi, wj] == 0.0 and lab['cens'][wi, wj] == 0
    for x in (0, 1, 2):                                                  # (x, wt_j) is the single x at position 2
        assert lab['y'][x, wj] == pytest.approx(float(b['si_y'][x])) and float(A[x, wj]) == pytest.approx(float(b['si_y'][x])) and float(B[x, wj]) == 0.0
    assert torch.isnan(lab['y'][wi, 3]) or float(lab['y'][wi, 3]) == pytest.approx(float(b['sj_y'][3]), nan_ok=True)
    assert torch.isfinite(lab['dG_wt']).all() and torch.isfinite(lab['y_floor']).all()      # no cell hands the link a NaN
    lab19, *_ = blocks.pair_labels(b, [a for a in range(20) if a != wi], [a for a in range(20) if a != wj], wi, wj)
    assert lab19['y'].shape == (19, 19)


def test_the_identity_cells_of_a_pair_block_are_the_per_item_singles():
    m = _stand_in()
    wt_rows, row_seqs, col_seqs, Xa, Ya = blocks.pair_sequences(SEQ, I, J, WT_I, WT_J, identity=True)
    assert len(Xa) == len(Ya) == 20 and row_seqs.shape == (20, 9)
    llr = lambda seqs, pos, wt: m.forward_site_llr(seqs, torch.as_tensor(pos), torch.as_tensor(wt), 'wt')
    llr_w = llr(wt_rows, [I, J], [WT_I, WT_J])
    llr_m = llr(torch.cat([row_seqs, col_seqs]), [J] * 20 + [I] * 20, [WT_J] * 20 + [WT_I] * 20)
    wt_raw, mt_raw = blocks.assemble_pair(llr_w[0], llr_w[1], llr_m[:20], llr_m[20:], Xa, Ya)
    wi, wj = blocks.TOKEN_TO_AA[WT_I], blocks.TOKEN_TO_AA[WT_J]
    assert float(wt_raw[wi, wj]) == 0.0 and float(mt_raw[wi, wj]) == 0.0                 # the wild type itself reads 0
    for x in (0, 3, 17):                                                                   # the single x at i: column wt_j
        mt = SEQ.clone(); mt[I] = blocks.CANON_IDS[x]
        b = {'wt_sequence_tokens': SEQ.unsqueeze(0), 'mt_sequence_tokens': mt.unsqueeze(0), 'mut_pos': torch.tensor([[I]]),
             'wt_id': torch.tensor([[WT_I]]), 'mt_id': torch.tensor([[blocks.CANON_IDS[x]]]), 'mut_mask': torch.tensor([[True]])}
        if x == wi:
            continue
        w = float(m.forward_partitioned(b, pass_type='wt', mask_strategy='independent')['pred_calibrated'][0])
        t = float(m.forward_partitioned(b, pass_type='mt', mask_strategy='independent')['pred_calibrated'][0])
        assert float(wt_raw[x, wj]) == pytest.approx(w, rel=1e-5, abs=1e-4) and float(mt_raw[x, wj]) == pytest.approx(t, rel=1e-5, abs=1e-4)


def test_a_singles_list_holds_the_wild_type_with_label_and_prediction_zero_when_asked():
    for ident, extra in ((True, 3), (False, 0)):
        s = _module(lambda_reg_wt=0.0, lambda_reg_mt=0.0, block_identity=ident)
        s.crit_rank_wt = s.crit_rank_mt = rec = _Rec()
        b = _blk('single')
        M._block_step(s, b)
        p, t, m = rec.lists[0]
        n_obs = int(torch.isfinite(b['y']).sum())
        assert int(m.sum()) == n_obs + extra and p.shape == (60,)
        if ident:
            wi = torch.as_tensor([blocks.TOKEN_TO_AA[int(w)] for w in b['wt_ids']])
            cells = torch.arange(3) * 20 + wi
            assert m[cells].all() and (t[cells] == 0).all() and (p[cells] == 0).all()


def test_the_identity_cell_has_no_regression_and_the_regression_count_is_unchanged():
    logs = {}
    for ident in (True, False):
        s = _module(lambda_rank_wt=0.0, lambda_rank_mt=0.0, block_identity=ident)
        logs[ident] = M._block_step(s, _blk('single'))
        logs[ident]['n'] = s._block_cnts['reg_wt']
    assert logs[True]['n'] == logs[False]['n'] and logs[True]['L_reg_wt'] == pytest.approx(logs[False]['L_reg_wt'])


def test_a_pair_block_ranks_the_singles_and_the_wild_type_with_the_doubles():
    s = _module(lambda_reg_mt=0.0, lambda_reg_wt=0.0, lambda_rank_wt=0.0, block_positions=20)
    s.crit_rank_mt = rec = _Rec()
    b = _blk('pair')
    M._block_step(s, b)
    assert len(rec.lists) == 1 and rec.lists[0][0].shape == (400,)
    n_doubles = int(torch.isfinite(b['y']).sum())
    n_singles = int(torch.isfinite(b['si_y']).sum()) + int(torch.isfinite(b['sj_y']).sum())
    assert int(rec.lists[0][2].sum()) == n_doubles + n_singles + 1                       # + the wild type


def _cond_setup(**hp):
    s = _module(lambda_rank_mt=0.0, lambda_rank_wt=0.0, lambda_reg_wt=0.0, lambda_reg_mt=0.0, lambda_cond_mt=1.0, **hp)
    return s, _blk('pair')


def test_a_conditional_leg_is_the_partner_single_plus_the_leg_against_the_doubles_own_label():
    s, b = _cond_setup(cond_partner_min_dg=1.0)
    logs = M._block_step(s, b)
    # The one double (x = aa 1 at site 2, y = aa 2 at site 5) has label 0.5. Row pass k = 1 reads site 5 after the substitution at site 2: the
    # partner single is the label of (site 2, aa 1) = 2 + 1, the leg is w_mt * (y + 1 + 0.1 k) = 0.1 * (2 + 1 + 0.1). Site 5 has no singles, so no column leg.
    expected = (3.0 + 0.1 * (2 + 1 + 0.1 * 1) - 0.5) ** 2
    assert logs['L_reg_cond'] == pytest.approx(expected, rel=1e-5)
    assert s.model.w_mt.grad is not None and float(s.model.w_mt.grad.abs()) > 0 and s.model.w_wt.grad is None


def test_the_partner_cutoff_removes_the_legs_whose_partner_is_not_stable_enough():
    s, b = _cond_setup(cond_partner_min_dg=10.0)                              # the partner has dG = 2.0 + 3.0 = 5.0 < 10
    logs = M._block_step(s, b)
    assert 'L_reg_cond' not in logs
    s, b = _cond_setup(cond_partner_min_dg=4.9)                               # just below the partner's 5.0
    assert 'L_reg_cond' in M._block_step(s, b)


def test_a_conditional_list_holds_the_partner_single_as_its_identity_cell():
    s = _module(lambda_reg_mt=0.0, lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_rank_mt=1.0, lambda_cond_mt=1.0, block_positions=19)
    b = _blk('pair')
    wt_rows, row_seqs, col_seqs, Xa, Ya = blocks.pair_sequences(b['seq'], 2, 5, WT_I, int(SEQ[5]), identity=True)
    wi, wj = blocks.TOKEN_TO_AA[int(SEQ[2])], blocks.TOKEN_TO_AA[int(SEQ[5])]
    lab, A, Bm, ident_cells, full = blocks.pair_labels(b, Xa, Ya, wi, wj)
    llr_m = s.model.forward_site_llr(torch.cat([row_seqs, col_seqs]), torch.as_tensor([5] * 20 + [2] * 20), torch.as_tensor([int(SEQ[5])] * 20 + [WT_I] * 20), 'mt')
    s.crit_rank_mt = rec = _Rec()
    s._block_sums, s._block_cnts = defaultdict_float(), defaultdict_float()
    M._block_cond_legs(s, b, llr_m, Xa, Ya, wi, wj, full, ident_cells)
    assert rec.lists                                                              # row legs whose partner single (site 2) is observed
    p, t, m = rec.lists[0]
    cols = torch.arange(len(p)).view(-1, 20)
    ident_cols = cols[:, wj]
    assert m[ident_cols].any()                                                    # the identity cell is a member of the list ...
    ok = ident_cols[m[ident_cols]]
    single_x = [float(b['si_y'][a]) for a in range(20) if a != wi and torch.isfinite(b['si_y'][a])]
    assert sorted(float(t[k]) for k in ok.tolist()) == sorted(single_x)           # ... labelled with the partner single, and predicted as it (leg 0)


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


def test_row_chunks_give_the_same_gradient_as_all_rows_at_once():
    def content_llr(net):
        """A stand-in whose rows depend on their CONTENT (the default one on the row's index within the call, which a chunk renumbers)."""
        def f(seqs, read_pos, wt_at, side, coords=None, struct=None):
            w = net.w_wt if side == 'wt' else net.w_mt
            out = w * (torch.arange(20).float().view(1, 20) + 1 + 0.01 * seqs.sum(dim=1).float().view(-1, 1))
            wt_aa = torch.as_tensor([blocks.TOKEN_TO_AA[int(t)] for t in wt_at])
            return out * (torch.arange(20).view(1, 20) != wt_aa.view(-1, 1)).float()
        return f
    grads = []
    for chunk in (0, 7):
        torch.manual_seed(5)
        s = _module(lambda_cond_mt=1.0, joint_doubles='both', tf_mix=0.5, block_row_chunk=chunk)
        s.model.forward_site_llr = content_llr(s.model)
        M._block_step(s, _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2)))
        grads.append((s.model.w_mt.grad.clone(), s.model.w_wt.grad.clone(), s.model.calibration_head_mt.scale.grad.clone()))
    for a, b_ in zip(*grads):
        assert torch.allclose(a, b_, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------------------------------------------- --block_mask_wt / --block_mask_mt

def _content_net(net):
    """A stand-in whose logits depend on the CONTENT of a sequence and on the position read, and that remembers the sequences it was handed."""
    seen = []

    def f(seqs, read_pos, wt_at, side, coords=None, struct=None):
        seen.append((side, seqs.clone(), read_pos.clone()))
        w = net.w_wt if side == 'wt' else net.w_mt
        base = (torch.arange(20).float().view(1, 20) + 1 + 0.01 * seqs.sum(dim=1).float().view(-1, 1) + 0.3 * read_pos.float().view(-1, 1))
        wt_aa = torch.as_tensor([blocks.TOKEN_TO_AA[int(t)] for t in wt_at])
        return w * base * (torch.arange(20).view(1, 20) != wt_aa.view(-1, 1)).float()
    net.forward_site_llr = f
    return seen


def test_the_unmasked_sequence_helpers():
    seq = SEQ.clone()
    seqs = blocks.single_sequences(seq, torch.tensor([1, 4, 6]))
    out = blocks.unmasked_rows(seqs, seq, torch.tensor([True, False, True]))
    assert torch.equal(out[0], seq) and torch.equal(out[2], seq) and int(out[1, 4]) == blocks.MASK
    vs, vp, vw, vr, vc = blocks.mutant_single_sequences(seq, torch.tensor([1, 4]), torch.tensor([int(seq[1]), int(seq[4])]))
    assert vs.shape == (38, len(seq)) and vp.tolist() == [1] * 19 + [4] * 19 and vr.tolist() == [0] * 19 + [1] * 19
    assert all(int(vs[k, vp[k]]) == blocks.CANON_IDS[int(vc[k])] != int(seq[vp[k]]) for k in range(38))       # the substitution is visible, never the wild type
    assert not (vs == blocks.MASK).any()
    ps, pr = blocks.pair_unmasked_sequences(seq, I, J, [0, 1, 2], [3, 4])
    assert ps.shape == (12, len(seq)) and pr.tolist() == [I, J] * 6
    assert torch.equal(ps[0], ps[1]) and not torch.equal(ps[1], ps[2])                                         # a cell's sequence is listed twice in a row
    assert int(ps[4, I]) == blocks.CANON_IDS[1] and int(ps[4, J]) == blocks.CANON_IDS[3]                      # cell 2 = (second x, first y): rows 4 and 5


def test_mask_wt_zero_reads_the_plain_wild_type_for_every_position_and_the_default_stays_masked():
    s = _module(block_mask_wt=0.0, lambda_rank_wt=0.0, lambda_rank_mt=0.0)
    seen = _content_net(s.model)
    M._block_step(s, _blk('single'))
    wt = [x for x in seen if x[0] == 'wt'][0][1]
    mt = [x for x in seen if x[0] == 'mt'][0][1]
    assert not (wt == blocks.MASK).any() and all(torch.equal(r, wt[0]) for r in wt)                            # unmasked: one sequence for every position
    assert (mt == blocks.MASK).sum(dim=1).tolist() == [1, 1, 1]                                                # the MT head is still masked (p = 1)


def test_mask_mt_zero_reads_each_substitution_as_its_own_visible_sequence_and_assembles_the_block():
    s = _module(block_mask_mt=0.0, lambda_rank_wt=0.0, lambda_rank_mt=0.0)
    seen = _content_net(s.model)
    b = _blk('single')
    M._block_step(s, b)
    side, seqs, read = [x for x in seen if x[0] == 'mt'][0]
    assert seqs.shape[0] == 3 * 19 and not (seqs == blocks.MASK).any()
    assert s.model.w_mt.grad is not None and float(s.model.w_mt.grad.abs()) > 0
    # the assembled [B, 20] is the visible readout of each substitution, with the wild-type column exactly 0
    s2 = _module(block_mask_mt=0.0)
    _content_net(s2.model)
    pos, wt_ids = b['positions'], b['wt_ids']
    llr = M._block_singles_llr(s2, b, blocks.single_sequences(b['seq'], pos), pos, wt_ids, 'mt', grad=True)
    assert llr.shape == (3, 20)
    for r in range(3):
        wa = blocks.TOKEN_TO_AA[int(wt_ids[r])]
        assert float(llr[r, wa]) == 0.0
        for a in (0, 5, 19):
            if a == wa:
                continue
            one = blocks.CANON_IDS[a]
            sq = b['seq'].clone()
            sq[pos[r]] = one
            ref = s2.model.forward_site_llr(sq.unsqueeze(0), pos[r:r + 1], wt_ids[r:r + 1], 'mt')[0, a]
            assert float(llr[r, a]) == pytest.approx(float(ref), rel=1e-6)


def test_a_probability_between_draws_masked_and_visible_positions_in_one_block():
    torch.manual_seed(1)
    s = _module(block_mask_mt=0.5)
    seen = _content_net(s.model)
    b = _blk('single')
    llr = M._block_singles_llr(s, b, blocks.single_sequences(b['seq'], b['positions']), b['positions'], b['wt_ids'], 'mt', grad=True)
    assert llr.shape == (3, 20) and torch.isfinite(llr).all()
    n_rows = sum(x[1].shape[0] for x in seen)                                                                  # k masked rows + 19 per visible row
    assert n_rows in (3, 3 + 18, 3 + 36, 57), n_rows


def test_pair_block_with_mask_wt_zero_is_one_wild_type_forward_and_mask_mt_zero_reads_both_sites_of_every_cell():
    s = _module(block_mask_wt=0.0, block_mask_mt=0.0, lambda_cond_mt=1.0, block_positions=8)
    seen = _content_net(s.model)
    M._block_step(s, _blk('pair', yA=torch.full((20, 20), 0.3), yB=torch.full((20, 20), 0.2)))
    wt = [x for x in seen if x[0] == 'wt'][0][1]
    mt = [x for x in seen if x[0] == 'mt'][0][1]
    assert not (wt == blocks.MASK).any() and torch.equal(wt[0], wt[1])
    assert mt.shape[0] == 2 * 400 and not (mt == blocks.MASK).any()                                           # 20 x 20 cells, each read at i and at j
    assert s.model.w_mt.grad is not None and float(s.model.w_mt.grad.abs()) > 0                               # trained, with the cond legs skipped for this block


def test_the_unmasked_pair_cell_is_the_sum_of_the_two_visible_readouts():
    s = _module(block_mask_mt=0.0, lambda_rank_mt=0.0, lambda_rank_wt=0.0, lambda_reg_wt=0.0)
    seen = _content_net(s.model)
    fed = []
    head = s.model.calibration_head_mt
    orig = head.forward
    head.forward = lambda y, n=None: (fed.append(y.detach().clone()), orig(y, n))[1]                         # what the MT head is handed: the raw cells
    M._block_step(s, _blk('pair'))
    mt_raw = fed[0]
    assert mt_raw.shape == (20, 20)
    f = lambda sq, pos, wt: s.model.forward_site_llr(sq.unsqueeze(0), torch.tensor([pos]), torch.tensor([wt]), 'mt')[0]
    for x, y in ((3, 7), (0, 19), (11, 2)):
        sq = SEQ.clone()
        sq[I], sq[J] = blocks.CANON_IDS[x], blocks.CANON_IDS[y]
        want = float(f(sq, I, WT_I)[x] + f(sq, J, WT_J)[y])               # the mutation at each site VISIBLE, the other site's mutation in place too
        assert float(mt_raw[x, y]) == pytest.approx(want, rel=1e-5), (x, y)
    wi, wj = blocks.TOKEN_TO_AA[WT_I], blocks.TOKEN_TO_AA[WT_J]
    assert float(mt_raw[wi, wj]) == 0.0                                                                       # the wild type reads 0 at both sites
    side, seqs, read = [x for x in seen if x[0] == 'mt'][0]
    c = (3 * 20 + 7) * 2
    assert torch.equal(seqs[c], seqs[c + 1]) and read[c].item() == I and read[c + 1].item() == J


def test_the_masking_probabilities_are_validated():
    import sys
    argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', '--block_training']
    old = sys.argv
    try:
        sys.argv = argv
        a = config.parse_arguments()
        assert a.block_mask_wt == 1.0 and a.block_mask_mt == 1.0
        sys.argv = argv + ['--block_mask_wt', '0', '--block_mask_mt', '0.25']
        a = config.parse_arguments()
        assert a.block_mask_wt == 0.0 and a.block_mask_mt == 0.25
        for bad in (['--block_mask_mt', '1.5'], ['--block_mask_wt', '-0.1']):
            sys.argv = argv + bad
            with pytest.raises(SystemExit):
                config.parse_arguments()
    finally:
        sys.argv = old


def test_identical_rows_are_one_forward_and_give_the_numbers_of_separate_ones():
    m = _stand_in()
    rows = []
    inner = m._get_esm3_outputs

    def counting(seq, *a, **k):
        rows.append(int(seq.shape[0]))
        return inner(seq, *a, **k)
    m._get_esm3_outputs = counting
    pos = torch.tensor([1, 4, 6, 4])
    wt = SEQ[pos]
    same = SEQ.unsqueeze(0).repeat(4, 1)                                  # an unmasked pass: the same sequence read at four positions
    together = m.forward_site_llr(same, pos, wt, 'wt')
    assert rows == [1], rows
    for r in range(4):
        alone = m.forward_site_llr(same[r:r + 1], pos[r:r + 1], wt[r:r + 1], 'wt')
        assert torch.allclose(together[r], alone[0], atol=1e-6)
    rows.clear()
    masked = blocks.single_sequences(SEQ, pos[:3])                        # rows that differ take the plain path: one forward each, in the given order
    m.forward_site_llr(masked, pos[:3], wt[:3], 'wt')
    assert rows == [3], rows
