import math
import types
import numpy as np
import pytest
import torch

from esm_msr import config, losses, training


def test_tobit_loss_gradcheck_mu_f_and_sigma_f():
    """Exact autograd wrt z, mu_f, and sigma_f matches finite differences across uncensored and censored items."""
    torch.manual_seed(0)
    z = torch.tensor([-1.5, -0.5, 0.2, 1.5, 3.0, -1.2, 5.2], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([-0.4, -0.2, 0.3, 1.4, 2.9, float('nan'), float('nan')], dtype=torch.float64)
    bound = torch.tensor([float('nan')] * 5 + [-1.0, 5.0], dtype=torch.float64)
    cens = torch.tensor([0, 0, 0, 0, 0, -1, 1], dtype=torch.long)

    mu_f = torch.tensor(-0.45, dtype=torch.float64, requires_grad=True)
    sigma_f = torch.tensor(0.55, dtype=torch.float64, requires_grad=True)

    def fn(z_, mu_, sig_):
        return losses.tobit_regression_loss(z_, y, bound, cens, sigma_y=0.28, mu_f=mu_, sigma_f=sig_).sum()

    assert torch.autograd.gradcheck(fn, (z, mu_f, sigma_f), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_tobit_loss_recovers_synthetic_mu_f_and_sigma_f():
    """Jointly optimizing (mu_f, sigma_f) via tobit_regression_loss recovers true noisy-floor parameters."""
    rng = np.random.default_rng(42)
    n = 6000
    sigma_y_true = 0.28
    mu_f_true = -0.85
    sigma_f_true = 0.70

    z_np = rng.uniform(-3.0, 3.5, size=n)
    x_np = z_np + rng.normal(0.0, sigma_y_true, size=n)
    f_np = rng.normal(mu_f_true, sigma_f_true, size=n)
    y_np = np.maximum(x_np, f_np)

    z = torch.tensor(z_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    bound = torch.full_like(y, float('nan'))
    cens = torch.zeros(n, dtype=torch.long)

    mu_bounds = (-2.0, 1.0)
    sig_bounds = (0.10, 2.0)
    mu_raw = training._init_bounded_param(-0.3, mu_bounds)
    sig_raw = training._init_bounded_param(0.3, sig_bounds)

    opt = torch.optim.Adam([mu_raw, sig_raw], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        mu_f = training._eval_bounded_param(mu_raw, mu_bounds)
        sigma_f = training._eval_bounded_param(sig_raw, sig_bounds)
        loss = losses.tobit_regression_loss(z, y, bound, cens, sigma_y=sigma_y_true, mu_f=mu_f, sigma_f=sigma_f).mean()
        loss.backward()
        opt.step()

    mu_est = float(training._eval_bounded_param(mu_raw, mu_bounds).item())
    sig_est = float(training._eval_bounded_param(sig_raw, sig_bounds).item())
    assert abs(mu_est - mu_f_true) < 0.05, f"Expected mu_f ~ {mu_f_true}, got {mu_est}"
    assert abs(sig_est - sigma_f_true) < 0.05, f"Expected sigma_f ~ {sigma_f_true}, got {sig_est}"


def test_bounded_param_init_and_limits():
    bounds = (-2.0, 1.0)
    p = training._init_bounded_param(-0.3, bounds)
    assert math.isclose(float(training._eval_bounded_param(p, bounds).item()), -0.3, abs_tol=1e-5)

    lo_val = float(training._eval_bounded_param(torch.nn.Parameter(torch.tensor(-50.0)), bounds).item())
    hi_val = float(training._eval_bounded_param(torch.nn.Parameter(torch.tensor(50.0)), bounds).item())
    assert bounds[0] <= lo_val <= bounds[0] + 1e-6
    assert bounds[1] - 1e-6 <= hi_val <= bounds[1]

    with pytest.raises(ValueError):
        training._init_bounded_param(-0.3, (1.0, -2.0))
    with pytest.raises(ValueError):
        training._init_bounded_param(-3.0, bounds)


def _make_stub_module(learn_mu=True, learn_sigma=True, warmup=10, reg_nll='tobit'):
    """Construct a bare ESM3EpistasisLightningModule without loading ESM3 weights."""
    m = training.ESM3EpistasisLightningModule.__new__(training.ESM3EpistasisLightningModule)
    torch.nn.Module.__init__(m)
    m._tobit_sigma_y = 0.28
    m._tobit_mu_f_init = -0.3
    m._tobit_sigma_f_init = 0.3
    m._tf_prior_sd = 0.7
    m.reg_nll = reg_nll
    m._learn_tobit_mu_f = bool(learn_mu) and reg_nll == 'tobit'
    m._learn_tobit_sigma_f = bool(learn_sigma) and reg_nll == 'tobit'
    m._tobit_mu_f_bounds = (-2.0, 1.0)
    m._tobit_sigma_f_bounds = (0.10, 2.0)
    m._tobit_warmup_steps = warmup
    m.tobit_mu_f_raw = training._init_bounded_param(-0.3, m._tobit_mu_f_bounds) if m._learn_tobit_mu_f else None
    m.tobit_sigma_f_raw = training._init_bounded_param(0.3, m._tobit_sigma_f_bounds) if m._learn_tobit_sigma_f else None
    m._trainer = None
    m._opt_steps = -1
    return m


def test_warmup_freezes_tobit_grads_then_unfreezes():
    m = _make_stub_module(learn_mu=True, learn_sigma=True, warmup=5)
    pred = torch.tensor([-2.0, -1.0, 0.5, 2.0], requires_grad=True)
    target = torch.tensor([-0.3, -0.2, 0.5, 2.0])
    bound = torch.full_like(target, float('nan'))
    cens = torch.zeros(4, dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)
    dgwt = torch.zeros(4)

    # During warmup (step 0 < 5): no gradient into tobit_*_raw, but pred still gets gradient
    m._opt_steps = -1  # step = 0
    loss, _, _ = m._reg_loss(pred, target, bound, cens, mask, 4.0, dgwt)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert m.tobit_mu_f_raw.grad is None
    assert m.tobit_sigma_f_raw.grad is None

    # After warmup (step 5 >= 5): both tobit_mu_f_raw and tobit_sigma_f_raw receive finite nonzero gradients
    pred.grad = None
    m._opt_steps = 4  # step = 5
    loss2, _, _ = m._reg_loss(pred, target, bound, cens, mask, 4.0, dgwt)
    loss2.backward()
    assert m.tobit_mu_f_raw.grad is not None and float(m.tobit_mu_f_raw.grad.abs()) > 0.0
    assert m.tobit_sigma_f_raw.grad is not None and float(m.tobit_sigma_f_raw.grad.abs()) > 0.0


def test_dynamic_properties_and_checkpoint_filter():
    m = _make_stub_module(learn_mu=True, learn_sigma=True, warmup=0)
    link0 = m.obs_link
    val0 = float(link0(np.array([-3.0]))[0])

    # Shift mu_f toward -1.0
    with torch.no_grad():
        m.tobit_mu_f_raw.copy_(training._init_bounded_param(-1.0, m._tobit_mu_f_bounds))
        m.tobit_sigma_f_raw.copy_(training._init_bounded_param(0.8, m._tobit_sigma_f_bounds))

    tp = m.tobit_params
    assert math.isclose(tp['mu_f'], -1.0, abs_tol=1e-5)
    assert math.isclose(tp['sigma_f'], 0.8, abs_tol=1e-5)
    assert math.isclose(m.tf_post['mu_f'], -1.0, abs_tol=1e-5)
    assert math.isclose(m.tf_post['sigma_f'], 0.8, abs_tol=1e-5)

    val1 = float(m.obs_link(np.array([-3.0]))[0])
    assert val1 < val0 - 0.5  # lower floor -> lower expected reading at deep latent dG

    # on_save_checkpoint retains tobit_*_raw alongside lora/calibration and strips base weights
    ckpt = {
        'state_dict': {
            'model.base.weight': torch.zeros(1),
            'model.lora_A': torch.ones(1),
            'model.calibration_head_wt.scale': torch.ones(1),
            'tobit_mu_f_raw': m.tobit_mu_f_raw.detach().clone(),
            'tobit_sigma_f_raw': m.tobit_sigma_f_raw.detach().clone(),
        }
    }
    m.on_save_checkpoint(ckpt)
    kept = set(ckpt['state_dict'].keys())
    assert kept == {'model.lora_A', 'model.calibration_head_wt.scale', 'tobit_mu_f_raw', 'tobit_sigma_f_raw'}


def test_disabled_flags_backward_compatible():
    m = _make_stub_module(learn_mu=False, learn_sigma=False, warmup=0)
    assert m.tobit_mu_f_raw is None
    assert m.tobit_sigma_f_raw is None
    assert m.tobit_params == {'sigma_y': 0.28, 'mu_f': -0.3, 'sigma_f': 0.3}
    assert m._tobit_params_for_loss() == {'sigma_y': 0.28, 'mu_f': -0.3, 'sigma_f': 0.3}


def test_config_cli_flags():
    ns_default = config.parse_arguments(['--experiment_name', 'test_run'])
    assert ns_default.learn_tobit_mu_f is False
    assert ns_default.learn_tobit_sigma_f is False
    assert ns_default.tobit_lr_mult == 5.0
    assert ns_default.tobit_warmup_steps is None
    assert ns_default.tobit_mu_f_bounds == [-2.0, 1.0]
    assert ns_default.tobit_sigma_f_bounds == [0.10, 2.0]

    ns = config.parse_arguments([
        '--experiment_name', 'test_run',
        '--learn_tobit_mu_f', '--learn_tobit_sigma_f',
        '--tobit_lr_mult', '10.0', '--tobit_warmup_steps', '250',
        '--tobit_mu_f_bounds', '-1.5', '0.5',
        '--tobit_sigma_f_bounds', '0.15', '1.5',
    ])
    assert ns.learn_tobit_mu_f is True
    assert ns.learn_tobit_sigma_f is True
    assert ns.tobit_lr_mult == 10.0
    assert ns.tobit_warmup_steps == 250
    assert ns.tobit_mu_f_bounds == [-1.5, 0.5]
    assert ns.tobit_sigma_f_bounds == [0.15, 1.5]
