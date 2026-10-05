"""WT-head early stopping: restore the best WT adapter, freeze it, leave the MT head alone."""
import types
import unittest

import torch
from torch import nn

try:
    from tests.test_compose_censoring import HP
except ImportError:  # run from inside tests/
    from test_compose_censoring import HP


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt_adapter = nn.Linear(2, 2, bias=False)
        self.calibration_head_wt = nn.Linear(1, 1, bias=False)
        self.mt_adapter = nn.Linear(2, 2, bias=False)

    def set_all(self, v):
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(v)


def make_stub(patience=2):
    from esm_msr import training
    cls = next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_wt_early_stop'))
    events = []
    stub = types.SimpleNamespace()
    stub.hparams = HP(wt_early_stop_patience=patience)
    stub.model = Mini()
    stub._trainable_param_names = {n for n, _ in stub.model.named_parameters()}
    stub.peft_manager = types.SimpleNamespace(
        has_transitioned=False, freeze_wt_components=lambda: events.append('freeze_wt'),
        unfreeze_mt_components=lambda: events.append('unfreeze_mt'), enforce_freezing=lambda *a, **k: events.append('enforce'))
    stub.optimizers = lambda: None
    stub._save_converged_wt_weights = lambda: events.append('save')
    stub._wt_param_names = types.MethodType(cls._wt_param_names, stub)
    stub.step = types.MethodType(cls._wt_early_stop, stub)
    return stub, events


class TestWtEarlyStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr.training  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def test_triggers_after_patience_and_restores_the_best_wt_state_only(self):
        stub, events = make_stub(patience=2)
        stub.model.set_all(1.0)
        stub.step(0.60)                                   # baseline: best so far, snapshot of the all-ones state
        stub.model.set_all(2.0)
        stub.step(0.70)                                   # improvement: snapshot of the all-twos state
        stub.model.set_all(3.0)
        stub.step(0.69)                                   # 1 validation without a gain
        self.assertEqual(events, [])
        stub.model.set_all(4.0)
        stub.step(0.695)                                  # 2nd: trigger
        self.assertEqual(events, ['save', 'freeze_wt', 'unfreeze_mt', 'enforce'])
        self.assertTrue(stub.peft_manager.has_transitioned)
        self.assertTrue(torch.all(stub.model.wt_adapter.weight == 2.0))                 # restored to the best state
        self.assertTrue(torch.all(stub.model.calibration_head_wt.weight == 2.0))
        self.assertTrue(torch.all(stub.model.mt_adapter.weight == 4.0))                 # the MT head is not touched

    def test_a_gain_resets_the_counter(self):
        stub, events = make_stub(patience=2)
        for v, m in ((1.0, 0.60), (2.0, 0.59), (3.0, 0.65), (4.0, 0.64)):
            stub.model.set_all(v)
            stub.step(m)
        self.assertEqual(events, [])                      # never two in a row without a gain
        self.assertEqual(stub.peft_manager.wt_patience_counter, 1)

    def test_a_gain_smaller_than_1e4_does_not_count(self):
        stub, events = make_stub(patience=1)
        stub.step(0.6000)
        stub.step(0.60005)
        self.assertEqual(events, ['save', 'freeze_wt', 'unfreeze_mt', 'enforce'])


if __name__ == '__main__':
    unittest.main()
