"""--per_mutation_bias applies the calibration bias once per mutated position instead of once per item.

The LLR sum the head calibrates is itself a sum over the item's positions, so a per-item bias makes a double's
prediction sit one bias below the sum of its own singles (measured at 0.1065 kcal/mol on devel-revision:
L_AB - L_A - L_B = -b). Anything that INVERTS the calibration has to invert the same thing, which is why
training's teacher-forcing de-calibration goes through bias_for() rather than reading .bias."""
import types
import unittest

import torch


def head(per_mutation, bias=0.1, scale=0.3):
    from esm_msr.models import CalibrationHead
    return CalibrationHead(init_scale=scale, init_bias=bias, per_mutation=per_mutation)


class TestPerMutationBias(unittest.TestCase):
    def test_a_per_item_head_adds_the_same_bias_to_singles_and_doubles(self):
        h = head(False)
        n = torch.tensor([1, 2, 3])
        raw = torch.zeros(3)
        out = h(raw, n)
        self.assertTrue(torch.allclose(out, torch.full((3,), 0.1), atol=1e-6), out)

    def test_a_per_mutation_head_scales_the_bias_by_the_mutation_count(self):
        h = head(True)
        n = torch.tensor([1, 2, 3])
        out = h(torch.zeros(3), n)
        self.assertTrue(torch.allclose(out, torch.tensor([0.1, 0.2, 0.3]), atol=1e-6), out)

    def test_the_double_then_matches_the_sum_of_its_singles(self):
        # The whole point: with a per-item bias a double is one bias short of its two singles.
        for per_mutation, expected_gap in ((False, -0.1), (True, 0.0)):
            h = head(per_mutation)
            raw_a = torch.tensor([2.0]); raw_b = torch.tensor([3.0])
            single_a = h(raw_a, torch.tensor([1]))
            single_b = h(raw_b, torch.tensor([1]))
            double = h(raw_a + raw_b, torch.tensor([2]))
            gap = float(double - single_a - single_b)
            self.assertAlmostEqual(gap, expected_gap, places=6,
                                   msg=f"per_mutation={per_mutation}: gap {gap}")

    def test_bias_for_refuses_to_guess_a_mutation_count(self):
        with self.assertRaises(AssertionError):
            head(True).bias_for(None)

    def test_a_per_item_head_needs_no_count(self):
        self.assertAlmostEqual(float(head(False).bias_for(None)), 0.1, places=6)

    def test_a_head_with_no_bias_is_zero_either_way(self):
        for pm in (False, True):
            self.assertEqual(head(pm, bias=None).bias_for(torch.tensor([2])), 0.0)

    def test_the_decalibration_training_uses_round_trips(self):
        # tf_additive_labels are de-calibrated as (label - bias_for(n_mut)) / scale before being injected as a
        # raw prediction; that has to invert forward() exactly, per mutation count.
        h = head(True)
        raw = torch.tensor([1.5, -2.0, 0.25])
        n = torch.tensor([1, 2, 3])
        cal = h(raw, n)
        back = (cal - h.bias_for(n)) / h.scale
        self.assertTrue(torch.allclose(back, raw, atol=1e-5), back)


class TestStopWhenNothingTrains(unittest.TestCase):
    def _stub(self, groups, frozen=(True, True)):
        from esm_msr import training
        cls = training.ESM3EpistasisLightningModule
        p = torch.nn.Parameter(torch.zeros(1))
        for g in groups:
            g.setdefault('params', [p])
        opt = types.SimpleNamespace(param_groups=groups)
        stub = types.SimpleNamespace(
            optimizers=lambda: opt,
            trainer=types.SimpleNamespace(should_stop=False, current_epoch=7),
            peft_manager=types.SimpleNamespace(wt_path_is_frozen=frozen[0], mt_path_is_frozen=frozen[1]))
        stub._stop_if_nothing_is_training = types.MethodType(cls._stop_if_nothing_is_training, stub)
        return stub

    def test_it_stops_when_every_group_is_at_zero(self):
        stub = self._stub([{'name': 'lora_wt', 'lr': 0.0}, {'name': 'lora_mt', 'lr': 0.0}])
        stub._stop_if_nothing_is_training()
        self.assertTrue(stub.trainer.should_stop)

    def test_it_keeps_going_while_one_group_still_has_a_rate(self):
        stub = self._stub([{'name': 'lora_wt', 'lr': 0.0}, {'name': 'lora_mt', 'lr': 1.8e-5}], frozen=(True, False))
        stub._stop_if_nothing_is_training()
        self.assertFalse(stub.trainer.should_stop)

    def test_a_zero_lr_group_of_frozen_parameters_does_not_hold_the_run_open(self):
        # requires_grad False AND lr 0: on_validation_end re-applies baseline requires_grad every epoch, so the
        # lr is the durable signal.
        p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        stub = self._stub([{'name': 'lora_mt', 'lr': 0.0, 'params': [p]}])
        stub._stop_if_nothing_is_training()
        self.assertTrue(stub.trainer.should_stop)

    def test_a_rate_on_frozen_parameters_does_not_hold_it_open_either(self):
        p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        stub = self._stub([{'name': 'link', 'lr': 0.005, 'params': [p]}])
        stub._stop_if_nothing_is_training()
        self.assertTrue(stub.trainer.should_stop)


if __name__ == '__main__':
    unittest.main()
