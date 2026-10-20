"""``--tf_mix``: how much of a double's WT half is the MEASURED additive label and how much is the WT head.

Teacher forcing replaces the WT half of a multi-mutant's combined prediction with the measured additive
expectation (ddG_A + ddG_B) so the MT head only learns the correction. That label carries the measurement noise of
both singles and the floor clipping of either; the WT head is smooth but biased. ``--tf_mix m`` blends them on the
calibrated scale: m = 1.0 is the released behaviour, 0.0 never teacher-forces, 0.5 the arithmetic mean.

The tests below read the blend out of an exact loss value: with the stub's WT pass at +2.0, MT pass at 0.0, label
+1.0 and target -1.0, the combined prediction is 0.5*(mix*1 + (1-mix)*2) and the squared error is deterministic.
"""
import sys
import types

import pytest
import torch

from esm_msr import config
from esm_msr.losses import ListMLELoss
from esm_msr.training import ESM3EpistasisLightningModule as M

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra):
    argv = sys.argv
    try:
        sys.argv = BASE + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ----------------------------------------------------------------------------------------------------------------
# the flag
# ----------------------------------------------------------------------------------------------------------------

def test_default_is_one_and_in_range_only():
    assert parse().tf_mix == 1.0
    assert parse('--tf_mix', '0.5').tf_mix == 0.5
    for bad in ('-0.1', '1.5'):
        with pytest.raises(SystemExit):
            parse('--tf_mix', bad)


# ----------------------------------------------------------------------------------------------------------------
# the blend, read out of the loss
# ----------------------------------------------------------------------------------------------------------------

class _CalHead(torch.nn.Module):
    use_scale = False
    bias = None

    def bias_for(self, n_mut):
        return 0.0


class _EnsembleModel(torch.nn.Module):
    lora_mode = 'ensemble'

    def __init__(self, wt_value, mt_value):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self.wt_value, self.mt_value = float(wt_value), float(mt_value)
        self.calibration_head_wt = _CalHead() if isinstance(_CalHead, type) else _CalHead()

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        B = int(micro['ddG'].shape[0])
        v = self.wt_value if pass_type == 'wt' else self.mt_value
        out = torch.full((B,), v) * self.w[0]          # keeps the graph alive: the loss must require grad
        return {'pred_calibrated': out, 'pred_raw': out}


class _HP(dict):
    """hparams is read-only on LightningModule; this stub mirrors AttributeDict, which RAISES on a missing key."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


    def get(self, k, default=None):
        return dict.get(self, k, default)


def _batch(doubles=True, B=6, additive=True):
    f = lambda v: torch.full((B,), float(v))
    mask = torch.ones(B, 2 if doubles else 1, dtype=torch.bool)
    return {
        'ddG': f(-1.0),                                      # the target every item is scored against
        'ddG_additive': f(1.0) if additive else f(float('nan')),
        'wt_sequence_tokens': torch.zeros(B, 40, dtype=torch.long), 'mut_mask': mask,
        'subset_type': ['double'] * B if doubles else ['single'] * B,
        'dG_wt': f(2.0), 'y_floor': f(-0.2), 'dG_meas': f(1.0),
        'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': f(float('nan')),
        'ddG_A': f(float('nan')), 'ddG_B': f(float('nan')),
    }


def _stub(tf_mix=None, wt_value=2.0, mt_value=0.0):
    base = dict(subset_size=4, micro_batch_size=3, micro_batch_tokens=0, double_weight=1.0, cond_weight=1.0,
                native_cond_weight=1.0, wt_double_target='latent', mt_reg_mask='all', mask_strategy=None,
                lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0,
                combined_mode='mean', detach_regression=False)
    if tf_mix is not None:
        base['tf_mix'] = tf_mix
    s = types.SimpleNamespace()
    s.hparams = _HP(base)
    s.model = _EnsembleModel(wt_value, mt_value)
    s.ns_head = None
    s.link_head = None
    s.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    s.crit_reg = torch.nn.MSELoss(reduction='none')
    s.crit_rank_wt = s.crit_rank_mt = ListMLELoss()
    s.manual_backward = lambda loss, retain_graph=False: None      # no graph needed: only the loss value is read
    for name in ('_compute_rank_loss', '_reg_loss'):
        setattr(s, name, types.MethodType(getattr(M, name), s))
    s._micro_batch = M._micro_batch
    return s


def _combined_loss(stub, batch=None):
    logs = M._compose_losses_streaming_and_backward(stub, _batch() if batch is None else batch)
    return float(logs['L_reg_combined'])


def test_mix_one_is_exactly_the_released_behaviour():
    """The default and an explicit 1.0 must be the same number - the flag cannot perturb the current recipe."""
    a = _combined_loss(_stub())                       # no tf_mix in hparams at all (an old checkpoint's hparams)
    b = _combined_loss(_stub(tf_mix=1.0))
    assert a == pytest.approx(b, abs=1e-7)
    assert a == pytest.approx((0.5 * 1.0 + 1.0) ** 2, abs=1e-6)         # forced = label 1.0, MT 0.0 -> 2.25


def test_the_blend_is_linear_in_the_measured_label():
    for mix in (0.0, 0.25, 0.5, 0.75):
        forced = mix * 1.0 + (1.0 - mix) * 2.0                          # label 1.0 vs WT head 2.0
        expected = (0.5 * forced + 0.5 * 0.0 - (-1.0)) ** 2
        assert _combined_loss(_stub(tf_mix=mix)) == pytest.approx(expected, abs=1e-6), mix


def test_mix_zero_means_no_teacher_forcing():
    """At 0.0 the WT half is the WT head's own prediction, on the doubles that have a label too."""
    assert _combined_loss(_stub(tf_mix=0.0)) == pytest.approx((0.5 * 2.0 + 1.0) ** 2, abs=1e-6)


def test_singles_are_mixed_like_doubles():
    """Since 2026-10-19 the mix applies to singles too, so 0.0 really is a no-teacher-forcing arm for the whole
    stream. A single's label is its own ddG (-1.0); the WT head says 2.0."""
    for mix in (0.0, 0.25, 0.5, 1.0):
        forced = mix * -1.0 + (1.0 - mix) * 2.0
        expected = (0.5 * forced + 0.5 * 0.0 - (-1.0)) ** 2
        assert _combined_loss(_stub(tf_mix=mix), _batch(doubles=False)) == pytest.approx(expected, abs=1e-6), mix
    assert _combined_loss(_stub(tf_mix=0.0), _batch(doubles=False)) == pytest.approx(4.0, abs=1e-6)


def test_items_without_a_usable_label_fall_back_to_the_wt_head_for_any_mix():
    a = _combined_loss(_stub(tf_mix=0.0), _batch(additive=False))
    b = _combined_loss(_stub(tf_mix=0.5), _batch(additive=False))
    c = _combined_loss(_stub(tf_mix=1.0), _batch(additive=False))
    assert a == pytest.approx(b, abs=1e-7) and b == pytest.approx(c, abs=1e-7)
    assert a == pytest.approx((0.5 * 2.0 + 1.0) ** 2, abs=1e-6)
