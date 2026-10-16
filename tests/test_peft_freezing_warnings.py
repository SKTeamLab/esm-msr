"""
``PEFTStateManager.enforce_freezing`` runs on every optimizer step while anything is frozen, so what it logs has to
be worth logging. It used to warn once per tensor per step for every parameter whose ``baseline_requires_grad`` was
False and whose name matched no adapter or calibration head - which is the whole frozen pretrained base. A run that
froze its WT path at epoch 4 therefore wrote 1140 warnings per step; 528960 of them before it was noticed.

A frozen base weight is the normal state and says nothing. A parameter that arrived TRAINABLE and is now frozen
without matching a known component is a real problem - its optimizer group cannot be identified, so
``zero_lrs`` will not zero its learning rate - and that deserves exactly one line.
"""
import logging

import pytest
import torch

from esm_msr.peft_manager import PEFTStateManager


class FakeModel:
    """Just the surface PEFTStateManager touches: named_parameters, the adapter names, and the log hook."""
    wt_adapter_name, mt_adapter_name = 'wt_adapter', 'mt_adapter'

    def __init__(self, trainable, frozen):
        self._params = {}
        for n in trainable:
            self._params[n] = torch.nn.Parameter(torch.zeros(2), requires_grad=True)
        for n in frozen:
            p = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self._params[n] = p
        self.logged = 0

    def named_parameters(self):
        return list(self._params.items())

    def _log_trainable_parameters(self):
        self.logged += 1


BASE = [f'peft_mt.base_model.model.transformer.blocks.{i}.attn.out_proj.weight' for i in range(40)] + \
       ['peft_mt.base_model.model.transformer.norm.weight', 'model.base_model.model.encoder.sequence_embed.weight']
ADAPTERS = ['peft_wt.base_model.model.blocks.0.wt_adapter.lora_A.weight',
            'peft_mt.base_model.model.blocks.0.mt_adapter.lora_A.weight',
            'calibration_head_wt.raw_scale', 'calibration_head_wt.bias', 'calibration_head_mt.raw_scale']


class FakeOpt:
    def __init__(self):
        self.param_groups = [{'name': n, 'lr': 2e-4, 'weight_decay': 0.01}
                             for n in ('lora_wt', 'lora_mt', 'calib_wt', 'calib_mt', 'link')]


def test_the_frozen_pretrained_base_is_not_warned_about(caplog):
    """The 1140-per-step case: a base that was already frozen when the manager was built."""
    m = FakeModel(trainable=ADAPTERS, frozen=BASE)
    mgr = PEFTStateManager(m)
    mgr.freeze_wt_components()
    opt = FakeOpt()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):                                     # as many steps as it takes; it is per-step in training
            mgr.enforce_freezing(opt, zero_lrs=True)
    assert [r.message for r in caplog.records] == [], "warned about the frozen base"


def test_the_wt_groups_are_still_identified_and_zeroed():
    """The warning change must not touch what enforce_freezing actually does."""
    m = FakeModel(trainable=ADAPTERS, frozen=BASE)
    mgr = PEFTStateManager(m)
    mgr.freeze_wt_components()
    opt = FakeOpt()
    mgr.enforce_freezing(opt, zero_lrs=True)
    lrs = {g['name']: g['lr'] for g in opt.param_groups}
    assert lrs['lora_wt'] == 0.0 and lrs['calib_wt'] == 0.0, lrs
    assert lrs['lora_mt'] == 2e-4 and lrs['calib_mt'] == 2e-4 and lrs['link'] == 2e-4, lrs
    assert all(not m._params[n].requires_grad for n in ADAPTERS if 'wt' in n)


def test_a_genuinely_unmatched_frozen_parameter_warns_exactly_once(caplog):
    """A parameter that arrived trainable, was frozen, and matches nothing: its LR cannot be zeroed, so say so."""
    odd = 'some_new_head.weight'
    m = FakeModel(trainable=ADAPTERS + [odd], frozen=BASE)
    mgr = PEFTStateManager(m)
    mgr.baseline_requires_grad[odd] = False
    opt = FakeOpt()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            mgr.enforce_freezing(opt, zero_lrs=True)
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 1, msgs
    assert odd in msgs[0] and 'will not be zeroed' in msgs[0]


def test_many_unmatched_parameters_give_one_line_not_one_each(caplog):
    odd = [f'some_new_head.{i}.weight' for i in range(9)]
    m = FakeModel(trainable=ADAPTERS + odd, frozen=BASE)
    mgr = PEFTStateManager(m)
    for n in odd:
        mgr.baseline_requires_grad[n] = False
    with caplog.at_level(logging.WARNING):
        mgr.enforce_freezing(FakeOpt(), zero_lrs=True)
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 1, msgs
    assert msgs[0].startswith('9 parameter(s)') and '+4 more' in msgs[0], msgs[0]


def test_freezing_the_mt_path_is_also_quiet(caplog):
    m = FakeModel(trainable=ADAPTERS, frozen=BASE)
    mgr = PEFTStateManager(m)
    mgr.freeze_mt_components()
    with caplog.at_level(logging.WARNING):
        mgr.enforce_freezing(FakeOpt(), zero_lrs=True)
    assert [r.message for r in caplog.records] == []
