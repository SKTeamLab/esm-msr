"""The frozen-WT / MT-only / doubles-only paradigm.

``--frozen_wt_mt_only`` is a named combination of primitive flags: a PRE-TRAINED WT path loaded from a checkpoint
(``--init_wt_from_checkpoint``), held fixed (``--freeze_wt_adapter``), with the training stream restricted to
multi-mutants (``--no-incl_singles``) so only the MT head (plus the link and the nonspecific term) learns. The
validation loaders are built independently and always keep singles and doubles, so ``rho_singles_wt`` stays a guard.

What is tested here: the flag combination is derived and validated at parse time, and the loader copies exactly
the WT path - never the MT adapter, the link or the nonspecific term, which must not be smuggled in from the
checkpoint the WT head came from.
"""
import inspect
import sys
import types

import pytest
import torch

from esm_msr import config
from esm_msr.peft_manager import PEFTStateManager
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
# the flag combination
# ----------------------------------------------------------------------------------------------------------------

def test_the_paradigm_needs_a_pretrained_wt(tmp_path):
    with pytest.raises(SystemExit):
        parse('--frozen_wt_mt_only')


def test_a_missing_checkpoint_is_refused():
    with pytest.raises(SystemExit):
        parse('--init_wt_from_checkpoint', '/nonexistent/wt.ckpt')


def test_the_paradigm_derives_its_flags(tmp_path):
    ckpt = tmp_path / 'dc17.ckpt'
    torch.save({'state_dict': {}}, ckpt)
    a = parse('--frozen_wt_mt_only', '--init_wt_from_checkpoint', str(ckpt))
    assert a.freeze_wt_adapter is True
    # singles are KEPT since 2026-10-19: a doubles-only stream is floor-limited and not more informative, and the
    # MT head learns contextual single effects too. --subsets double gives the old stream.
    assert a.subsets == ['single', 'double']
    assert a.lambda_reg_wt == 0.0 and a.lambda_rank_wt == 0.0
    # the MT objective is untouched
    assert a.lambda_reg_mt == pytest.approx(1.0) and a.lambda_rank_mt == pytest.approx(1.0)


def test_a_plain_warm_start_is_allowed_and_keeps_the_wt_trainable(tmp_path):
    ckpt = tmp_path / 'wt.ckpt'
    torch.save({'state_dict': {}}, ckpt)
    a = parse('--init_wt_from_checkpoint', str(ckpt))
    assert a.freeze_wt_adapter is False and a.subsets == ['single', 'double']


# ----------------------------------------------------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------------------------------------------------

class _StubModel(torch.nn.Module):
    wt_adapter_name = 'wt_adapter'
    mt_adapter_name = 'mt_adapter'

    def __init__(self):
        super().__init__()
        self.peft_wt_adapter = torch.nn.Parameter(torch.zeros(2))         # the WT adapter
        self.calibration_head_wt = torch.nn.Parameter(torch.tensor([0.0]))  # the WT readout
        self.peft_mt_adapter = torch.nn.Parameter(torch.ones(2))           # must NOT be loaded
        self.link_head_tau = torch.nn.Parameter(torch.tensor([0.59]))      # must NOT be loaded
        self.ns_head_values = torch.nn.Parameter(torch.zeros(3))           # must NOT be loaded

    def _log_trainable_parameters(self):
        pass


def _stub(model=None):
    s = types.SimpleNamespace()
    s.model = _StubModel() if model is None else model
    s.peft_manager = types.SimpleNamespace(wt_parameter_names=PEFTStateManager.wt_parameter_names.__get__(s))
    return s


def _save(tmp_path, state, name='wt.ckpt', nested=True):
    p = tmp_path / name
    torch.save({'state_dict': state} if nested else state, p)
    return str(p)


def test_only_the_wt_path_is_loaded(tmp_path):
    stub = _stub()
    src = {'peft_wt_adapter': torch.tensor([1.5, -2.5]), 'calibration_head_wt': torch.tensor([0.12]),
           'peft_mt_adapter': torch.tensor([9.0, 9.0]), 'link_head.tau': torch.tensor([0.8]),
           'ns_head.values': torch.full((3,), 7.0)}
    n = M._load_wt_weights(stub, _save(tmp_path, src))
    assert n == ['peft_wt_adapter', 'calibration_head_wt']
    torch.testing.assert_close(stub.model.peft_wt_adapter, torch.tensor([1.5, -2.5]))
    torch.testing.assert_close(stub.model.calibration_head_wt, torch.tensor([0.12]))
    # MT adapter, link and nonspecific term are untouched
    torch.testing.assert_close(stub.model.peft_mt_adapter, torch.ones(2))
    assert float(stub.model.link_head_tau) == pytest.approx(0.59)
    assert torch.all(stub.model.ns_head_values == 0)


def test_lightning_checkpoint_prefix_and_converged_file_both_work(tmp_path):
    stub = _stub()
    src = {'model.peft_wt_adapter': torch.tensor([3.0, 4.0]), 'model.calibration_head_wt': torch.tensor([-0.2])}
    M._load_wt_weights(stub, _save(tmp_path, src))
    torch.testing.assert_close(stub.model.peft_wt_adapter, torch.tensor([3.0, 4.0]))

    converged = tmp_path / 'converged_wt_epoch_3_step_900.pt'
    torch.save({'state_dict': {'peft_wt_adapter': torch.tensor([-1.0, 1.0]),
                               'calibration_head_wt': torch.tensor([0.5])}}, converged)
    M._load_wt_weights(stub, str(converged))
    torch.testing.assert_close(stub.model.peft_wt_adapter, torch.tensor([-1.0, 1.0]))


def test_nothing_matching_is_an_error(tmp_path):
    stub = _stub()
    with pytest.raises(RuntimeError, match='no WT parameter'):
        M._load_wt_weights(stub, _save(tmp_path, {'peft_mt_adapter': torch.ones(2)}))


def test_a_shape_mismatch_is_skipped_but_a_partial_load_succeeds(tmp_path):
    stub = _stub()
    src = {'peft_wt_adapter': torch.zeros(5),                    # wrong shape -> skipped
           'calibration_head_wt': torch.tensor([0.25])}
    n = M._load_wt_weights(stub, _save(tmp_path, src))
    assert n == ['calibration_head_wt']
    torch.testing.assert_close(stub.model.calibration_head_wt, torch.tensor([0.25]))
    assert torch.all(stub.model.peft_wt_adapter == 0)             # unchanged


def test_dtype_and_device_are_followed(tmp_path):
    """A checkpoint saved in bf16 or on CPU must land in the parameter's own dtype/device, not replace it."""
    stub = _stub()
    stub.model.peft_wt_adapter.data = stub.model.peft_wt_adapter.data.to(torch.float32)
    src = {'peft_wt_adapter': torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
           'calibration_head_wt': torch.tensor([0.1])}
    M._load_wt_weights(stub, _save(tmp_path, src))
    assert stub.model.peft_wt_adapter.dtype == torch.float32
    torch.testing.assert_close(stub.model.peft_wt_adapter, torch.tensor([1.0, 2.0]))


# ----------------------------------------------------------------------------------------------------------------
# the link travels with the WT model
# ----------------------------------------------------------------------------------------------------------------

class _Link(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_tau = torch.nn.Parameter(torch.tensor([0.59]))
        self.register_buffer('floor_min', torch.tensor([-1.0]))
        self.register_buffer('global_floor', torch.tensor([0.0]))


def test_the_link_is_loaded_with_the_wt_model(tmp_path):
    """A frozen WT adapter with a different link is a different model: the floors are the observed-scale readout."""
    stub = _stub()
    stub.link_head = _Link()
    src = {'peft_wt_adapter': torch.tensor([1.0, 2.0]), 'calibration_head_wt': torch.tensor([0.1]),
           'link_head.raw_tau': torch.tensor([0.8]), 'link_head.floor_min': torch.tensor([-0.5]),
           'link_head.global_floor': torch.tensor([-0.136])}
    M._load_wt_weights(stub, _save(tmp_path, src))          # also caches the state dict for the link
    n = M._load_link_weights(stub, 'cached')
    assert n == ['raw_tau', 'floor_min', 'global_floor']
    assert float(stub.link_head.raw_tau) == pytest.approx(0.8)
    assert float(stub.link_head.floor_min) == pytest.approx(-0.5)
    assert float(stub.link_head.global_floor) == pytest.approx(-0.136)


def test_a_checkpoint_without_a_link_leaves_it_as_configured(tmp_path):
    """The released checkpoints and converged_wt_*.pt files have no link_head.* - that is not an error."""
    stub = _stub()
    stub.link_head = _Link()
    path = _save(tmp_path, {'peft_wt_adapter': torch.tensor([1.0, 2.0])})
    M._load_wt_weights(stub, path)
    assert M._load_link_weights(stub, path) == []
    assert float(stub.link_head.raw_tau) == pytest.approx(0.59)


def test_the_init_hooks_the_link_to_the_wt_model():
    """Guard the wiring itself: the loader must run under --init_wt_from_checkpoint, link included. (The link has no parameters to freeze:
    its numbers are buffers, so nothing can move them.)"""
    src = inspect.getsource(M.__init__)
    assert 'init_wt_from_checkpoint' in src and '_load_link_weights' in src


# ----------------------------------------------------------------------------------------------------------------
# one naming rule for "the WT path"
# ----------------------------------------------------------------------------------------------------------------

def test_freezing_and_loading_agree_on_what_the_wt_path_is():
    stub = _stub()
    names = stub.peft_manager.wt_parameter_names()
    assert names == ['peft_wt_adapter', 'calibration_head_wt']

    mgr = PEFTStateManager(stub.model)
    mgr.freeze_wt_components()
    frozen = sorted(n for n, p in stub.model.named_parameters() if not p.requires_grad)
    assert frozen == sorted(names)
    # the MT path is untouched by a WT freeze
    assert all(p.requires_grad for n, p in stub.model.named_parameters() if n == 'peft_mt_adapter')
