"""A WT-only arm must not reopen the MT path at the WT -> MT transition.

`_stop_if_nothing_is_training` ends a run when every optimizer group that still holds trainable parameters is at
lr 0. `_transition_to_mt` is what makes that true: it freezes the WT path and then `enforce_freezing(zero_lrs=True)`
zeroes the frozen groups' LRs. If it unconditionally unfreezes the MT path, a WT-only arm (`--freeze_mt_adapter`)
leaves a live group at base LR and the arm runs its remaining epochs forward-only with no loss.

It did: 2026-10-19, `wt_r8a11` froze the WT after decay #3 and then ran epoch 7 at 4.9 GB with `lr/lora_mt` 2e-4 and
never stopped. The guard existed in the retired pin as a hand patch and was never upstreamed, so the consolidation
onto devel-revision dropped it.
"""
import torch
from types import SimpleNamespace

from esm_msr.training import ESM3EpistasisLightningModule


class _Recorder:
    def __init__(self):
        self.calls = []
        self.has_transitioned = False
        self.mt_path_is_frozen = False

    def freeze_wt_components(self):
        self.calls.append('freeze_wt')

    def freeze_mt_components(self):
        self.calls.append('freeze_mt')
        self.mt_path_is_frozen = True

    def unfreeze_mt_components(self):
        self.calls.append('unfreeze_mt')
        self.mt_path_is_frozen = False

    def enforce_freezing(self, optimizers, zero_lrs=True):
        self.calls.append(f'enforce(zero_lrs={zero_lrs})')


def _bare(freeze_mt_adapter):
    """The module shell `_transition_to_mt` needs, without ESM3, a trainer or a GPU."""
    obj = object.__new__(ESM3EpistasisLightningModule)
    torch.nn.Module.__init__(obj)
    obj.peft_manager = _Recorder()
    obj._hparams = SimpleNamespace(freeze_mt_adapter=freeze_mt_adapter)
    obj._save_converged_wt_weights = lambda: None
    obj.optimizers = lambda: []
    return obj


def test_wt_only_arm_refreezes_the_mt_path_at_the_transition():
    obj = _bare(freeze_mt_adapter=True)
    ESM3EpistasisLightningModule._transition_to_mt(obj)
    assert obj.peft_manager.calls == ['freeze_wt', 'freeze_mt', 'enforce(zero_lrs=True)']
    assert obj.peft_manager.has_transitioned
    assert obj.peft_manager.mt_path_is_frozen, "a WT-only arm must leave the MT path frozen so the stop can fire"


def test_a_normal_arm_still_opens_the_mt_path():
    obj = _bare(freeze_mt_adapter=False)
    ESM3EpistasisLightningModule._transition_to_mt(obj)
    assert obj.peft_manager.calls == ['freeze_wt', 'unfreeze_mt', 'enforce(zero_lrs=True)']
    assert not obj.peft_manager.mt_path_is_frozen, "the normal handover must still thaw MT"


def test_transition_does_not_rethaw_already_frozen_mt_path():
    obj = _bare(freeze_mt_adapter=False)
    obj.peft_manager.mt_path_is_frozen = True
    ESM3EpistasisLightningModule._transition_to_mt(obj)
    assert obj.peft_manager.calls == ['freeze_wt', 'enforce(zero_lrs=True)']
    assert obj.peft_manager.mt_path_is_frozen, "an MT path that already plateau-froze must stay frozen when WT transitions later"


def test_fused_mode_freezes_at_wt_transition_and_skips_mt_plateau():
    obj = _bare(freeze_mt_adapter=False)
    obj.model = SimpleNamespace(adapter_mode='fused')
    ESM3EpistasisLightningModule._mt_plateau_step(obj, {'epi_beyond_rho_mt': [0.5, 0.4]})
    assert obj.peft_manager.calls == []
    ESM3EpistasisLightningModule._transition_to_mt(obj)
    assert obj.peft_manager.calls == ['freeze_wt', 'freeze_mt', 'enforce(zero_lrs=True)']
