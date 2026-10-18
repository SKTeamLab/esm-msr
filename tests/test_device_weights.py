"""The frozen base weights are deduplicated and cast before the FIRST forward pass, which is the pre-fit validation and
not the start of training (devel-revision 8a89f01: preparing them in on_train_start fragmented the allocator), and again
at EVERY later stage hook, because a stage boundary re-breaks the storage sharing.

main() validates before it fits; Lightning returns the module to the CPU when the validate stage tears down; and the
move back for fit gives every parameter its own storage again. A once-per-process flag therefore left a second copy of
the frozen bf16 base live for the whole run - measured at 2.73 GB on devel-revision (dc8_s2 mem/allocated_gb 5.706
against 2.972 for its --skip_val resume, which re-shared at on_train_start and was the only path that was ever right),
and confirmed live in dc14, which logs "570 had been duplicated" at both hooks and peaks at 24.90 GB reserved where
every fresh run before it peaked at 27.607."""
import types
import unittest

import torch


class TestPrepareDeviceWeights(unittest.TestCase):
    def _stub(self):
        from esm_msr import training
        cls = training.ESM3EpistasisLightningModule
        calls = []
        stub = types.SimpleNamespace(
            _reshare_base_params_on_device=lambda: calls.append(1),
            peft_manager=types.SimpleNamespace(has_transitioned=False),
            hparams=types.SimpleNamespace(freeze_wt_adapter=False, freeze_mt_adapter=False))
        stub._prepare_device_weights = types.MethodType(cls._prepare_device_weights, stub)
        return cls, stub, calls

    def test_the_first_validation_prepares_the_weights(self):
        cls, stub, calls = self._stub()
        cls.on_validation_start(stub)
        self.assertEqual(len(calls), 1)

    def test_every_later_stage_hook_prepares_them_again(self):
        # The regression this guards: with a once-only flag, on_train_start was a no-op after the pre-fit
        # validation, which is precisely where the sharing has just been broken again.
        cls, stub, calls = self._stub()
        cls.on_validation_start(stub)
        cls.on_train_start(stub)
        cls.on_validation_start(stub)
        self.assertEqual(len(calls), 3)

    def test_with_skip_val_training_still_prepares_them(self):
        cls, stub, calls = self._stub()
        cls.on_train_start(stub)
        self.assertEqual(len(calls), 1)


class TestReshareRepairsBrokenSharing(unittest.TestCase):
    def _stub(self):
        from esm_msr import training
        cls = training.ESM3EpistasisLightningModule
        wt, mt = torch.nn.Module(), torch.nn.Module()
        wt.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        mt.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        stub = types.SimpleNamespace(model=types.SimpleNamespace(peft_wt=wt, peft_mt=mt),
                                     _cast_frozen_linears_bf16=lambda: None)
        stub._reshare_base_params_on_device = types.MethodType(cls._reshare_base_params_on_device, stub)
        return stub, wt, mt

    def test_it_repoints_mt_at_wt_storage(self):
        stub, wt, mt = self._stub()
        self.assertNotEqual(wt.weight.data.data_ptr(), mt.weight.data.data_ptr())
        stub._reshare_base_params_on_device()
        self.assertEqual(wt.weight.data.data_ptr(), mt.weight.data.data_ptr())

    def test_a_repair_logs_at_info_and_an_already_shared_pass_does_not(self):
        stub, wt, mt = self._stub()
        with self.assertLogs(level='INFO') as got:
            stub._reshare_base_params_on_device()
        self.assertIn('1 had been duplicated', '\n'.join(got.output))
        with self.assertNoLogs(level='INFO'):   # nothing to repair the second time
            stub._reshare_base_params_on_device()

    def test_a_single_adapter_model_is_left_alone(self):
        from esm_msr import training
        cls = training.ESM3EpistasisLightningModule
        stub = types.SimpleNamespace(model=types.SimpleNamespace(peft_wt=None, peft_mt=None),
                                     _cast_frozen_linears_bf16=lambda: self.fail("should not cast"))
        stub._reshare_base_params_on_device = types.MethodType(cls._reshare_base_params_on_device, stub)
        stub._reshare_base_params_on_device()   # returns before touching anything


if __name__ == '__main__':
    unittest.main()
