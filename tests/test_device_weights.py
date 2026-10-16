"""The frozen base weights are deduplicated and cast before the FIRST forward pass, which is the pre-fit validation, not
the start of training, and only once (devel-revision 8a89f01: preparing them in on_train_start fragmented the allocator)."""
import types
import unittest


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

    def test_the_first_validation_prepares_the_weights_and_training_does_not_redo_it(self):
        cls, stub, calls = self._stub()
        cls.on_validation_start(stub)
        self.assertEqual(len(calls), 1)
        cls.on_train_start(stub)
        cls.on_validation_start(stub)
        self.assertEqual(len(calls), 1)

    def test_with_skip_val_training_still_prepares_them(self):
        cls, stub, calls = self._stub()
        cls.on_train_start(stub)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
