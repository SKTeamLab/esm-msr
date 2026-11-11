"""--grad_accum_blocks K: K blocks' gradients add up in .grad, one optimizer step follows the K-th, on the mean gradient (the gradient clip is 1.0 per head)."""
import types

import pytest
import torch
import torch.nn as nn

from esm_msr import config, training


def parse(*extra):
    return config.parse_arguments(['--experiment_name', 'x'] + list(extra))


def test_the_flag_parses_defaults_to_one_and_must_be_positive():
    assert parse().grad_accum_blocks == 1
    assert parse('--grad_accum_blocks', '4').grad_accum_blocks == 4
    with pytest.raises(SystemExit):
        parse('--grad_accum_blocks', '0')


class Sched:
    def __init__(self): self.n = 0
    def step(self): self.n += 1


def module(accum, grads):
    """A stub with one trainable vector and a _block_step that deposits ``grads[i]`` for block i and reports its 'loss'."""
    mod = training.ESM3EpistasisLightningModule.__new__(training.ESM3EpistasisLightningModule)
    nn.Module.__init__(mod)
    mod._device = torch.device('cpu')
    hp = dict(grad_accum_blocks=accum, lr_warmup_steps=0)
    mod._hparams = types.SimpleNamespace(**hp, get=lambda k, d=None: hp.get(k, d))
    mod.model = nn.Module()
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{'params': [p], 'lr': 1.0, 'name': 'g'}], lr=1.0)
    sched, calls = Sched(), {'i': 0}
    mod.optimizers = lambda: opt
    mod.lr_schedulers = lambda: sched
    logged = []
    mod.log = lambda name, value, **k: logged.append((name, float(value)))
    mod._log_lrs = lambda: None
    mod.log_calibration_head = lambda **k: None

    def block_step(b):
        g = grads[calls['i']]; calls['i'] += 1
        p.grad = g.clone() if p.grad is None else p.grad + g
        return {'L_x': float(g.sum())}
    mod._block_step = block_step
    return mod, p, opt, sched, logged


def run(mod, n):
    for i in range(n):
        mod.training_step({'block': 'single'}, i)


def test_k_blocks_make_one_step_on_their_mean_gradient():
    grads = [torch.tensor([.1, 0.]), torch.tensor([.3, .2]), torch.tensor([.2, .4]), torch.tensor([.6, .2])]
    mod, p, opt, sched, logged = module(2, grads)
    run(mod, 4)
    # SGD lr 1: first step -mean(g0, g1) = -[.2, .1], second -mean(g2, g3) = -[.4, .3] (both norms below the clip of 1)
    assert torch.allclose(p, torch.tensor([-.6, -.4]))
    assert sched.n == 2 and p.grad is None
    losses = [v for n, v in logged if n == 'train/L_x']
    assert losses == [pytest.approx(.3), pytest.approx(.7)]         # the mean of each window's two block losses


def test_k_equal_one_is_the_unchanged_step():
    grads = [torch.tensor([.1, 0.]), torch.tensor([.3, .2])]
    mod, p, opt, sched, logged = module(1, grads)
    run(mod, 2)
    assert torch.allclose(p, torch.tensor([-.4, -.2])) and sched.n == 2


def test_an_unfinished_window_carries_across_calls_and_takes_no_step():
    grads = [torch.tensor([.1, .1]), torch.tensor([.1, .1]), torch.tensor([.1, .1])]
    mod, p, opt, sched, logged = module(3, grads)
    run(mod, 2)
    assert torch.equal(p, torch.zeros(2)) and sched.n == 0 and not logged
    run(mod, 1)
    assert torch.allclose(p, torch.tensor([-.1, -.1])) and sched.n == 1


def test_the_clip_sees_the_mean_gradient():
    grads = [torch.tensor([3., 0.]), torch.tensor([3., 0.]), torch.tensor([3., 0.]), torch.tensor([3., 0.])]
    mod, p, opt, sched, logged = module(4, grads)
    run(mod, 4)
    assert torch.allclose(p, torch.tensor([-1., 0.]))                # mean [3, 0] has norm 3, clipped to 1


def test_the_cosine_length_counts_optimizer_steps():
    from test_audit_next_features import _make_stub_module
    base = {}
    for accum in (1, 4):
        mod = _make_stub_module(lr_warmup_steps=4, num_epochs=1, grad_accum_blocks=accum)    # no trainer: 900 blocks
        opts, scheds = mod.configure_optimizers()
        for _ in range(230):
            scheds[0]['scheduler'].step()
        g = opts[0].param_groups[0]
        base[accum] = g['lr'] / g['initial_lr']
    assert base[4] == pytest.approx(0.05, rel=0.05)           # 900 blocks / 4 = 225 optimizer steps: the cosine has finished
    assert base[1] > 0.4                                     # the same 230 steps are a quarter of the way through 900
