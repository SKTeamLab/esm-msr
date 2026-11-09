"""Unit tests for the November 2026 design-audit fixes, validation statistics, and two-conditional masked supervision."""
import types
import numpy as np
import pytest
import torch
import torch.nn as nn

from esm_msr import config, epi_metrics as E, training
from esm_msr.losses import ListMLELoss


class _Group(nn.Module):
    """A model with one parameter per optimizer group, named the way configure_optimizers sorts them."""

    def __init__(self):
        super().__init__()
        self.wt_adapter_name, self.mt_adapter_name = 'wt_adapter', 'mt_adapter'
        self.wt_adapter_lora = nn.Parameter(torch.randn(8))
        self.mt_adapter_lora = nn.Parameter(torch.randn(8))
        self.calibration_head_wt = nn.Linear(1, 1)
        self.calibration_head_mt = nn.Linear(1, 1)


def _make_stub_module(**hp_overrides):
    """Only what configure_optimizers reads: the hparams, the model, the parameters that arrived trainable."""
    mod = training.ESM3EpistasisLightningModule.__new__(training.ESM3EpistasisLightningModule)
    nn.Module.__init__(mod)
    defaults = dict(learning_rate=2e-4, lr_warmup_steps=10, num_epochs=4, grad_accum_blocks=1, block_domain_steps=False)
    defaults.update(hp_overrides)
    mod._hparams = types.SimpleNamespace(**defaults, get=lambda k, d=None: defaults.get(k, d))
    mod.model = _Group()
    mod._initially_trainable = {n for n, _ in mod.model.named_parameters()}
    return mod


def test_the_schedule_is_a_linear_warmup_then_a_cosine_decay_to_the_floor_ratio():
    mod = _make_stub_module(lr_warmup_steps=4, num_epochs=1)
    opts, scheds = mod.configure_optimizers()
    opt, sch = opts[0], scheds[0]['scheduler']
    lrs = []
    for _ in range(20):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    assert lrs[3] > lrs[0]                              # warmup rises first, then the cosine decays smoothly
    assert lrs[-1] < lrs[3]
    names = {g['name'] for g in opt.param_groups}
    assert names == {'lora_wt', 'lora_mt', 'calib_wt', 'calib_mt'}
    base = {g['name']: g['initial_lr'] for g in opt.param_groups}
    assert base['calib_wt'] == pytest.approx(5.0 * base['lora_wt'])       # the calibration heads learn at config.CALIB_LR_MULT times the adapters' rate


def test_epi_metrics_1x_dddg_and_preserved_cell_rank():
    """Verify comb = 0.5*(WT + MT) fixes the 2x epistasis bias & achieves 1.000 oracle cell_rank_comb while epi retains raw MT."""
    rng = np.random.default_rng(42)
    keys, st, ddg, dddg, dg_wt, wt, mt, comb = [], [], [], [], [], [], [], []
    a = rng.normal(-0.5, 0.5, 5)
    b = rng.normal(-0.5, 0.5, 5)
    eps = rng.normal(0.3, 0.4, (5, 5))
    c = 3.0
    for i in range(5):
        keys.append((('L1', 10, chr(65 + i)),))
        st.append('single'); ddg.append(a[i]); dddg.append(np.nan); dg_wt.append(c)
        wt.append(a[i]); mt.append(a[i]); comb.append(a[i])
    for j in range(5):
        keys.append((('L1', 20, chr(70 + j)),))
        st.append('single'); ddg.append(b[j]); dddg.append(np.nan); dg_wt.append(c)
        wt.append(b[j]); mt.append(b[j]); comb.append(b[j])
    for i in range(5):
        for j in range(5):
            y_true = a[i] + b[j] + eps[i, j]
            w_ab = a[i] + b[j]
            # Ensemble trained so 0.5*(WT + MT) == y_true => MT = w_ab + 2*eps
            m_ab = w_ab + 2.0 * eps[i, j]
            c_ab = 0.5 * (w_ab + m_ab)
            keys.append((('L1', 10, chr(65 + i)), ('L1', 20, chr(70 + j))))
            st.append('double'); ddg.append(y_true); dddg.append(eps[i, j]); dg_wt.append(c)
            wt.append(w_ab); mt.append(m_ab); comb.append(c_ab)
    tbl = E.Table({'mut_key': keys, 'subset_type': st, 'cens': np.zeros(len(keys), int),
                   'ddG': ddg, 'dddG': dddg, 'dG_wt': dg_wt, 'wt': wt, 'mt': mt, 'comb': comb}, link=None)
    res = E.compute(tbl)
    diag = E.compute_clean_diagnostics(tbl)
    assert abs(res['epi_global_bias_comb']) < 1e-9
    assert abs(res['epi_global_err_comb']) < 1e-9
    assert abs(diag['epi_clean_bias_comb']) < 1e-9
    assert abs(diag['epi_clean_bias_epi']) < 1e-9
    assert abs(diag['wt_doubles_clean_slope'] - 1.0) < 0.35
    assert res['epi_cell_rank_comb'] > 0.999999
    assert 'epi_cell_rank_mt_legacy' in diag and diag['epi_cell_rank_mt_legacy'] < res['epi_cell_rank_comb']
    assert abs(res['epi_cell_rank_epi'] - diag['epi_cell_rank_mt_legacy']) < 1e-12


def test_library_classifier_and_debiased_rho():
    assert training.classify_megascale_library('1F0M') == 'natural_wt'
    assert training.classify_megascale_library('2KT8_G54S') == 'mutant_bg'
    assert training.classify_megascale_library('v2K43S_2KVV') == 'mutant_bg'
    assert training.classify_megascale_library('HEEH_KT_rd6_0280') == 'designed'
    assert training.classify_megascale_library('r6_560_TrROS_Hall') == 'designed'

    rng = np.random.default_rng(7)
    aas = list('ACDEFGHIKLMNPQRSTVWY')
    recs = []
    for lib in ('1F0M', '1I2T'):
        pairs = [(aas[i % 20], aas[(i * 3 + 1) % 20]) for i in range(60)]
        true_struct = rng.normal(0, 1.0, 60)
        aa_bias = np.array([0.5 * (ord(m) - ord(w)) / 10.0 for w, m in pairs])
        gt = true_struct + aa_bias
        wt = 0.8 * true_struct + 2.0 * aa_bias + rng.normal(0, 0.2, 60)
        recs.append((lib, wt, gt, pairs))
    rho = training.compute_nat_debiased_rho(recs)
    assert np.isfinite(rho) and rho > 0.7


def test_epi_metrics_credits_marginal_wt_pair_and_line_effects():
    """Verify epi_metrics credits a standalone marginal-masked WT head (wt == mt == comb) for pair + line epistasis while cell_rank == 0."""
    rng = np.random.default_rng(11)
    keys, st, ddg, dddg, dg_wt, wt, mt, comb = [], [], [], [], [], [], [], []
    for p in range(10):
        lib, c = ('L1', 2.5) if p % 2 == 0 else ('L2', 3.0)
        a = rng.normal(-0.6, 0.5, 5)
        b = rng.normal(-0.6, 0.5, 5)
        m_pair = rng.normal(0.0, 0.5)
        r_line = rng.normal(0.0, 0.35, 5)
        s_line = rng.normal(0.0, 0.35, 5)
        eps_cell = rng.normal(0.0, 0.25, (5, 5))
        eps_cell = eps_cell - eps_cell.mean(1, keepdims=True) - eps_cell.mean(0, keepdims=True) + eps_cell.mean()
        for i in range(5):
            keys.append(((lib, 10 * p + 1, chr(65 + i)),))
            st.append('single'); ddg.append(a[i]); dddg.append(np.nan); dg_wt.append(c)
            wt.append(a[i]); mt.append(a[i]); comb.append(a[i])
        for j in range(5):
            keys.append(((lib, 10 * p + 5, chr(70 + j)),))
            st.append('single'); ddg.append(b[j]); dddg.append(np.nan); dg_wt.append(c)
            wt.append(b[j]); mt.append(b[j]); comb.append(b[j])
        for i in range(5):
            for j in range(5):
                ep_marg = m_pair + r_line[i] + s_line[j]
                ep_true = ep_marg + eps_cell[i, j]
                y_true = a[i] + b[j] + ep_true
                y_marg = a[i] + b[j] + ep_marg
                keys.append(((lib, 10 * p + 1, chr(65 + i)), (lib, 10 * p + 5, chr(70 + j))))
                st.append('double'); ddg.append(y_true); dddg.append(ep_true); dg_wt.append(c)
                # Standalone marginal WT (--val_wt_only): mt is NaN, wt == comb == y_marg on doubles, == a[i]/b[j] on singles
                wt.append(y_marg); mt.append(np.nan); comb.append(y_marg)
    tbl = E.Table({'mut_key': keys, 'subset_type': st, 'cens': np.zeros(len(keys), int),
                   'ddG': ddg, 'dddG': dddg, 'dG_wt': dg_wt, 'wt': wt, 'mt': mt, 'comb': comb}, link=None)
    res = E.compute(tbl)
    assert res['epi_pair_rho_comb'] > 0.85
    assert res['epi_line_rho_comb'] > 0.85
    assert res['epi_beyond_lib_comb'] > 0.85
    assert res['epi_cell_rank_comb'] == 0.0
