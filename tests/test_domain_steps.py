"""--block_domain_steps: one optimizer step per library, singles + w * mean(pairs), with --grad_clip_per_head clipping each head on its own."""
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from esm_msr import blocks, config, training


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x'] + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


def lib(name, positions, n_pairs):
    return blocks.Library(name=name, seq=torch.zeros(1), coords=torch.zeros(1), structure_tokens=torch.zeros(1), wt_tok={},
                          single_pos=np.arange(positions), pairs=[(q, q + 1) for q in range(n_pairs)])


LIBS = [lib('a', 3, 0), lib('b', 2, 2), lib('c', 0, 1), lib('d', 0, 0)]


def test_the_flags_refuse_what_would_mean_nothing():
    assert parse('--block_training', '--block_domain_steps', '--block_pair_weight', '0.1').block_pair_weight == 0.1
    for bad in (['--block_domain_steps'],
                ['--block_training', '--block_pair_weight', '0.5'],
                ['--block_training', '--block_domain_steps', '--block_pair_weight', '-1'],
                ['--block_training', '--block_domain_steps', '--grad_accum_blocks', '2'],
                ['--grad_clip_per_head', '--grad_clip_norm', '0'],
                ['--val_every_epochs', '0']):
        with pytest.raises(SystemExit):
            parse(*bad)


def test_an_epoch_is_one_closed_step_per_library_with_every_position_and_pair():
    s = blocks.BlockSampler(LIBS, 2, seed=1, domain_steps=True)
    d = list(s)
    assert len(d) == len(s) == 5                             # a: s; b: s p p; c: p; d: nothing
    assert s.n_steps() == sum(x[3] for x in d) == 3
    by_lib = {}
    for x in d:
        by_lib.setdefault(x[1], []).append(x)
    assert set(by_lib) == {0, 1, 2}
    assert [x[0] for x in by_lib[1]] == ['s', 'p', 'p'] and [x[3] for x in by_lib[1]] == [False, False, True]
    assert by_lib[0] == [('s', 0, (0, 1, 2), True)] and by_lib[2] == [('p', 2, 0, True)]
    assert {x[2] for x in by_lib[1][1:]} == {0, 1}
    li = [x[1] for x in d]
    assert [k for k in range(1, len(li)) if li[k] != li[k - 1]] == [k + 1 for k, x in enumerate(d[:-1]) if x[3]]     # a library's blocks stay together


def test_no_pairs_means_singles_steps_only_and_the_order_follows_the_epoch():
    s = blocks.BlockSampler(LIBS, 2, seed=1, pairs=False, domain_steps=True)
    assert s.n_steps() == len(s) == 2 and all(x[0] == 's' and x[3] for x in s)
    s = blocks.BlockSampler([lib(str(k), 1, 0) for k in range(30)], 2, seed=1, domain_steps=True)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first
    s.set_epoch(0)
    assert list(s) == first


def test_the_plain_sampler_is_unchanged():
    s = blocks.BlockSampler(LIBS, 2, seed=1)
    assert len(s) == s.n_steps() == 2 + 1 + 3 and all(len(x) == 3 for x in s)


class Sched:
    last_epoch = 10 ** 9
    def __init__(self): self.n = 0
    def step(self): self.n += 1


def module(blocks_grads, clip=0.0, per_head=False):
    mod = training.ESM3EpistasisLightningModule.__new__(training.ESM3EpistasisLightningModule)
    nn.Module.__init__(mod)
    mod._device = torch.device('cpu')
    hp = dict(block_domain_steps=True, grad_accum_blocks=1, grad_clip_norm=clip, grad_clip_per_head=per_head, freeze_wt_adapter=False, freeze_mt_adapter=False,
              lr_warmup_steps=0, lr_schedule='cosine')
    mod._hparams = types.SimpleNamespace(**hp, get=lambda k, d=None: hp.get(k, d))
    mod.model = nn.Module()
    mod.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, has_transitioned=False, enforce_freezing=lambda *a, **k: None)
    wt, mt = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([{'params': [wt], 'lr': 1.0, 'name': 'lora_wt'}, {'params': [mt], 'lr': 1.0, 'name': 'lora_mt'}], lr=1.0)
    sched, calls = Sched(), {'i': 0}
    mod.optimizers = lambda: opt
    mod.lr_schedulers = lambda: sched
    logged = []
    mod.log = lambda name, value, **k: logged.append((name, float(value)))
    mod._log_lrs = lambda: None
    mod.log_calibration_head = lambda **k: None

    def block_step(b):
        gw, gm = blocks_grads[calls['i']]; calls['i'] += 1
        for p, g in ((wt, gw), (mt, gm)):
            if g is not None:
                g = torch.tensor([g])
                p.grad = g if p.grad is None else p.grad + g
        return {'L_reg_combined': float(gm or 0)}
    mod._block_step = block_step
    return mod, wt, mt, sched, logged


def test_the_step_comes_at_the_end_of_the_library_on_the_summed_gradient_and_logs_apart():
    mod, wt, mt, sched, logged = module([(1., 2.), (None, 3.), (None, 5.)])
    mod.training_step({'block': 'single', 'step_end': False}, 0)
    mod.training_step({'block': 'pair', 'step_end': False}, 1)
    assert sched.n == 0 and wt.item() == 0 and not logged
    mod.training_step({'block': 'pair', 'step_end': True}, 2)
    assert sched.n == 1 and wt.item() == -1 and mt.item() == -10
    assert ('train/L_reg_combined', 2.) in logged and ('train/L_reg_combined_pair', 4.) in logged


def test_one_clip_lets_the_doubles_shrink_the_wt_step_and_per_head_does_not():
    grads = [(1., 0.), (None, 100.)]
    mod, wt, mt, *_ = module(grads, clip=1.0, per_head=False)
    mod.training_step({'block': 'single', 'step_end': False}, 0)
    mod.training_step({'block': 'pair', 'step_end': True}, 1)
    assert abs(wt.item()) < 0.011
    mod, wt, mt, _, logged = module(grads, clip=1.0, per_head=True)
    mod.training_step({'block': 'single', 'step_end': False}, 0)
    mod.training_step({'block': 'pair', 'step_end': True}, 1)
    assert wt.item() == pytest.approx(-1.0) and mt.item() == pytest.approx(-1.0)
    assert dict(logged)['norm_grad/pre_clip_mt'] == pytest.approx(100.0) and dict(logged)['norm_grad/pre_clip_wt'] == pytest.approx(1.0)


def test_the_dataset_tells_a_block_its_library_and_whether_it_closes_the_step():
    b = blocks.BlockDataset([blocks.Library(name='x', seq=torch.zeros(1), coords=torch.zeros(1), structure_tokens=torch.zeros(1), wt_tok={0: 1, 1: 2},
                                            single_pos=np.array([]), pairs=[(0, 1)] * 4,
                                            pair_data={f: np.zeros((4, 20, 20)) for f in blocks.SINGLE_FIELDS})])
    assert b[('p', 0, 1, True)]['step_end'] is True and b[('p', 0, 1, True)]['n_lib_pairs'] == 4
    assert 'step_end' not in b[('p', 0, 1)]                  # the plain sampler's descriptors


def test_a_pair_blocks_gradient_is_weighted_by_its_share_of_the_library_and_its_logged_loss_is_not():
    from test_blocks import _blk, _module
    s = _module(lambda_rank_mt=0.0)
    plain = M_step(s, _blk('pair'))
    g0 = s.model.w_mt.grad.clone()
    s = _module(lambda_rank_mt=0.0, block_pair_weight=0.1)
    scaled = M_step(s, _blk('pair', n_lib_pairs=4, step_end=True))
    assert s.model.w_mt.grad.item() == pytest.approx(g0.item() * 0.1 / 4, rel=1e-5)
    assert scaled['L_reg_combined'] == pytest.approx(plain['L_reg_combined'])


def M_step(s, b):
    return training.ESM3EpistasisLightningModule._block_step(s, b)


def test_the_cosine_length_counts_domain_steps():
    from test_audit_next_features import _make_stub_module
    mod = _make_stub_module(lr_schedule='cosine', lr_warmup_steps=4, num_epochs=1, block_domain_steps=True)

    class S:
        def n_steps(self): return 90
        def __len__(self): return 180
    mod._fabric, mod._jit_is_scripting = None, False
    mod._trainer = types.SimpleNamespace(estimated_stepping_batches=180, train_dataloader=[types.SimpleNamespace(sampler=S())])
    opts, scheds = mod.configure_optimizers()
    for _ in range(90):
        scheds[0]['scheduler'].step()
    g = opts[0].param_groups[0]
    assert g['lr'] / g['initial_lr'] == pytest.approx(0.05, rel=0.05)


@pytest.mark.parametrize('p', [0.0, 0.5])
def test_visible_sites_compose_with_the_domain_weight_and_the_row_chunk(p):
    """The masking grid (p = 0.5 and 0 on both heads, row chunk 20) runs on domain steps: a visible-site pair block is weighted w / n_pairs exactly as a masked one is, and
    a visible-site singles block still trains both heads (the same random draw is replayed for each of the two compared steps)."""
    from test_blocks import _blk, _module
    kw = dict(lambda_rank_mt=0.0, block_mask_wt=p, block_mask_mt=p, block_row_chunk=20)
    s = _module(**kw)
    torch.manual_seed(3)
    plain = M_step(s, _blk('pair'))
    g0 = s.model.w_mt.grad.clone()
    s = _module(**kw, block_pair_weight=0.38)
    torch.manual_seed(3)
    scaled = M_step(s, _blk('pair', n_lib_pairs=4, step_end=True))
    assert float(g0.abs()) > 0
    assert s.model.w_mt.grad.item() == pytest.approx(g0.item() * 0.38 / 4, rel=1e-4)
    assert scaled['L_reg_combined'] == pytest.approx(plain['L_reg_combined'])
    s = _module(**kw)
    torch.manual_seed(3)
    logs = M_step(s, _blk('single'))
    assert all(np.isfinite(v) for v in logs.values())
    assert float(s.model.w_wt.grad.abs()) > 0 and float(s.model.w_mt.grad.abs()) > 0
