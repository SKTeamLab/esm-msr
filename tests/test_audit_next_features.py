"""Unit tests for the November 2026 design-audit fixes, validation statistics, and two-conditional masked supervision."""
import types
import numpy as np
import pytest
import torch
import torch.nn as nn

from esm_msr import config, epi_metrics as E, training
from esm_msr.data import ProteinCyclingBatchSampler
from esm_msr.link import SaturatingLink
from esm_msr.losses import ListMLELoss


class _DummyCalHead(nn.Module):
    def __init__(self, scale: float, bias: float, per_mutation_bias: bool = True):
        super().__init__()
        self.use_scale = True
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.per_mutation_bias = per_mutation_bias

    def bias_for(self, n_mut: torch.Tensor) -> torch.Tensor:
        return self.bias * n_mut.float() if self.per_mutation_bias else self.bias.expand_as(n_mut.float())

    def forward(self, x: torch.Tensor, n_mut: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias_for(n_mut)


class _DummyPEFTManager:
    wt_path_is_frozen = False
    mt_path_is_frozen = False
    has_transitioned = False
    initially_trainable = set()

    def enforce_freezing(self, *args, **kwargs):
        pass


class _DummyModel(nn.Module):
    def __init__(self, mask_mt='independent', s_wt=0.25, b_wt=0.10, s_mt=0.15, b_mt=-0.20):
        super().__init__()
        self.adapter_mode = 'dual'
        self.wt_adapter_name = 'wt_adapter'
        self.mt_adapter_name = 'mt_adapter'
        self.calibration_head_wt = _DummyCalHead(s_wt, b_wt)
        self.calibration_head_mt = _DummyCalHead(s_mt, b_mt)
        self.lora_wt_weight = nn.Parameter(torch.randn(8))
        self.lora_mt_weight = nn.Parameter(torch.randn(8))
        self._mask_mt = mask_mt

    def forward_partitioned(self, batch, pass_type='wt', mask_strategy=None):
        B, K = batch['mut_mask'].shape
        n_mut = batch['mut_mask'].sum(dim=1)
        idx = torch.arange(B, device=batch['ddG'].device, dtype=torch.float32)
        if pass_type == 'wt':
            u = torch.stack([(idx + 1.0) * 0.1 * self.lora_wt_weight[j % 8] for j in range(K)], dim=1)
            u = torch.where(batch['mut_mask'], u, torch.zeros_like(u))
            raw = u.sum(dim=1)
            cal = self.calibration_head_wt(raw, n_mut)
        else:
            u = torch.stack([(idx + 1.0) * 0.1 * self.lora_mt_weight[(j + 2) % 8] for j in range(K)], dim=1)
            u = torch.where(batch['mut_mask'], u, torch.zeros_like(u))
            raw = u.sum(dim=1)
            cal = self.calibration_head_mt(raw, n_mut)
        return {'pred_calibrated': cal, 'pred_raw': raw, 'unsummed_llr': u}


def _make_stub_module(**hp_overrides):
    mod = training.ESM3EpistasisLightningModule.__new__(training.ESM3EpistasisLightningModule)
    nn.Module.__init__(mod)
    defaults = dict(
        adapter_mode='dual',
        lambda_reg_wt=15.0,
        lambda_rank_wt=0.5,
        lambda_reg_mt=15.0,
        lambda_rank_mt=0.5,
        lambda_cond_mt=0.0,
        cond_partner_min_dg=1.0,
        mt_single_loss_weight=1.0,
        rank_calibrated_stopgrad=False,
        latent_censor_hinge=False,
        rank_list_size=4,
        micro_batch_size=8,
        censor_threshold=-1.0,
        tf_mix=1.0,
        native_cond_for_wt=True,
        native_cond_for_mt=False,
        mt_direct_pred=False,
        rank_group_by='none',
        huber_delta=0.0,
        learning_rate=2e-4,
        calib_lr_mult=5.0,
        lr_warmup_steps=10,
        lr_schedule='cosine',
        cosine_min_lr_ratio=0.05,
        num_epochs=4,
        grad_clip_norm=1.0,
        freeze_wt_adapter=False,
        freeze_mt_adapter=False,
    )
    defaults.update(hp_overrides)
    mod._hparams = types.SimpleNamespace(**defaults, get=lambda k, d=None: defaults.get(k, d))
    mod.model = _DummyModel()
    mod.peft_manager = _DummyPEFTManager()
    mod.peft_manager.initially_trainable = {n for n, _ in mod.model.named_parameters()}
    mod.crit_rank_wt = ListMLELoss()
    mod.crit_rank_mt = ListMLELoss()
    mod.crit_reg = nn.MSELoss(reduction='none')
    mod.link_head = SaturatingLink(tau=0.59, floor=0.0)
    mod.mask_wt = None
    mod.mask_mt = 'independent'
    mod.manual_backward = lambda loss: loss.backward()
    return mod


def _make_batch(B=8):
    mut_mask = torch.zeros(B, 2, dtype=torch.bool)
    mut_mask[:, 0] = True
    mut_mask[B // 2 :, 1] = True  # first half singles, second half doubles
    dG_wt = torch.full((B,), 3.0)
    ddG = torch.tensor([-0.5, -1.2, -2.5, -4.5, -0.8, -1.5, -2.2, -4.8], dtype=torch.float32)[:B]
    ddG_A = torch.tensor([float('nan')] * (B // 2) + [-0.4, -0.8, -1.0, -2.8])[:B]
    ddG_B = torch.tensor([float('nan')] * (B // 2) + [-0.3, -0.5, -0.9, -2.5])[:B]
    dG_meas = dG_wt + ddG
    cens = torch.where(dG_meas <= -1.0, torch.tensor(-1), torch.tensor(0))
    cens_bound = torch.where(cens < 0, torch.full((B,), -1.0) - dG_wt, torch.full((B,), float('nan')))
    return {
        'ddG': ddG,
        'ddG_A': ddG_A,
        'ddG_B': ddG_B,
        'dG_wt': dG_wt,
        'y_floor': torch.zeros(B),
        'dG_meas': dG_meas,
        'cens': cens,
        'cens_src': torch.zeros(B, dtype=torch.long),
        'cens_bound': cens_bound,
        'mut_mask': mut_mask,
        'subset_type': ['single'] * (B // 2) + ['double'] * (B - B // 2),
    }


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


def test_stopgrad_calibrated_rank_loss_never_touches_calibration_params():
    """With rank_calibrated_stopgrad=True and lambda_reg=0, calibration scales/biases must receive zero gradient."""
    mod = _make_stub_module(lambda_reg_wt=0.0, lambda_reg_mt=0.0, rank_calibrated_stopgrad=True)
    batch = _make_batch(8)
    logs = mod._compose_losses_streaming_and_backward(batch)
    assert 'L_rank_wt' in logs and 'L_rank_combined' in logs
    assert mod.model.calibration_head_wt.scale.grad is None or mod.model.calibration_head_wt.scale.grad.abs().item() == 0.0
    assert mod.model.calibration_head_mt.scale.grad is None or mod.model.calibration_head_mt.scale.grad.abs().item() == 0.0
    assert mod.model.lora_wt_weight.grad is not None and mod.model.lora_wt_weight.grad.abs().sum().item() > 0.0
    assert mod.model.lora_mt_weight.grad is not None and mod.model.lora_mt_weight.grad.abs().sum().item() > 0.0


def test_two_conditional_masked_supervision_and_single_downweighting():
    """Verify lambda_cond_mt produces finite gradients on MT adapter + calibration without extra forward passes."""
    mod = _make_stub_module(lambda_cond_mt=7.5, mt_single_loss_weight=0.1, rank_calibrated_stopgrad=True, latent_censor_hinge=True)
    batch = _make_batch(8)
    logs = mod._compose_losses_streaming_and_backward(batch)
    assert 'L_cond_mt' in logs and logs['L_cond_mt'] > 0.0
    assert torch.isfinite(mod.model.lora_mt_weight.grad).all()
    assert mod.model.lora_mt_weight.grad.abs().sum().item() > 0.0


def test_sampler_drop_last_false_retains_tail_batches():
    class _MockDS(list):
        pass

    ds = _MockDS([{'pdb': '1AAA', 'subset_type': 'single'} for _ in range(300)])
    s_drop = ProteinCyclingBatchSampler([ds], batch_size=256, train_list=['1AAA'], rng_seed=1, drop_last=True)
    s_keep = ProteinCyclingBatchSampler([ds], batch_size=256, train_list=['1AAA'], rng_seed=1, drop_last=False, min_batch_size=32)
    assert len(list(s_drop)) == 1
    batches = list(s_keep)
    assert len(batches) == 2 and sum(len(b) for b in batches) == 300


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


def test_new_cli_flags_parse_and_cosine_scheduler_steps():
    args = config.parse_arguments([
        '--experiment_name', 'unit_test',
        '--rank_calibrated_stopgrad',
        '--mt_single_loss_weight', '0.1',
        '--lambda_cond_mt', '7.5',
        '--latent_censor_hinge',
        '--grad_clip_norm', '1.0',
        '--calib_lr_mult', '5.0',
        '--lr_schedule', 'cosine',
        '--no-drop_last',
        '--min_batch_size', '32',
        '--no-incl_sequence_head_wt',
    ])
    assert args.rank_calibrated_stopgrad is True
    assert args.mt_single_loss_weight == pytest.approx(0.1)
    assert args.lambda_cond_mt == pytest.approx(7.5)
    assert args.latent_censor_hinge is True
    assert args.grad_clip_norm == pytest.approx(1.0)
    assert args.calib_lr_mult == pytest.approx(5.0)
    assert args.lr_schedule == 'cosine'
    assert args.drop_last is False
    assert args.incl_sequence_head_wt is False

    mod = _make_stub_module(lr_schedule='cosine', lr_warmup_steps=4, num_epochs=1)
    opts, scheds = mod.configure_optimizers()
    opt, sch = opts[0], scheds[0]['scheduler']
    lrs = []
    for _ in range(20):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    # Warmup rises first, then cosine decays smoothly
    assert lrs[3] > lrs[0]
    assert lrs[-1] < lrs[3]


def test_wt_supervise_doubles_with_marginal_masking():
    """Verify --wt_supervise_doubles trains WT on true double labels through link & censoring, both standalone and joint."""
    # Make a doubles-only batch: without wt_supervise_doubles WT produces no loss; with wt_supervise_doubles WT learns cleanly.
    batch = _make_batch(8)
    batch['mut_mask'] = torch.ones((8, 2), dtype=torch.bool)
    batch['subset_type'] = ['double'] * 8

    mod_off = _make_stub_module(mask_strategy_wt='marginal', wt_supervise_doubles=False, lambda_reg_mt=0.0, lambda_rank_mt=0.0)
    logs_off = mod_off._compose_losses_streaming_and_backward(batch)
    assert 'L_reg_wt' not in logs_off
    assert mod_off.model.lora_wt_weight.grad is None or mod_off.model.lora_wt_weight.grad.abs().sum().item() == 0.0

    mod_on = _make_stub_module(mask_strategy_wt='marginal', wt_supervise_doubles=True, rank_calibrated_stopgrad=True,
                               lambda_cond_mt=7.5, mt_single_loss_weight=0.1)
    mod_on.mask_wt = 'marginal'
    logs_on = mod_on._compose_losses_streaming_and_backward(batch)
    assert 'L_reg_wt' in logs_on and 'L_rank_wt' in logs_on
    assert mod_on.model.lora_wt_weight.grad is not None and mod_on.model.lora_wt_weight.grad.abs().sum().item() > 0.0
    assert mod_on.model.lora_mt_weight.grad is not None and mod_on.model.lora_mt_weight.grad.abs().sum().item() > 0.0


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
