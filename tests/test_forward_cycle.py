"""MSRModel.forward_cycle: the two passes forward_batch does not run, with the right inputs and signs."""
import types
import unittest

import torch

from esm_msr.models import MSRModel


class Recorder:
    training = False

    def __init__(self):
        self.calls = []

    def forward_partitioned(self, batch, pass_type, cached_wt_esm3=None, mask_strategy=None):
        self.calls.append((pass_type, batch, mask_strategy))
        return {'pred_calibrated': torch.tensor([1.5, -0.5]) if pass_type == 'wt' else torch.tensor([0.25, 2.0])}


def batch():
    return {'wt_sequence_tokens': torch.tensor([[1, 2, 3], [1, 2, 3]]), 'mt_sequence_tokens': torch.tensor([[1, 9, 3], [1, 2, 8]]),
            'wt_id': torch.tensor([[2], [3]]), 'mt_id': torch.tensor([[9], [8]]), 'mut_pos': torch.tensor([[2], [3]]),
            'mut_mask': torch.tensor([[True], [True]])}


class TestForwardCycle(unittest.TestCase):
    def test_the_wt_adapter_reads_the_mutated_sequence_about_the_reverse_mutation(self):
        r, b = Recorder(), batch()
        out = MSRModel.forward_cycle(r, b, mask_strategy='marginal')
        kind, rev, ms = r.calls[0]
        self.assertEqual(kind, 'wt'); self.assertEqual(ms, 'marginal')
        self.assertTrue(torch.equal(rev['wt_sequence_tokens'], b['mt_sequence_tokens']))
        self.assertTrue(torch.equal(rev['wt_id'], b['mt_id'])); self.assertTrue(torch.equal(rev['mt_id'], b['wt_id']))
        self.assertTrue(torch.equal(out['wt_rev'], -torch.tensor([1.5, -0.5])))            # the reverse ddG, negated into a forward one

    def test_the_mt_adapter_reads_the_wild_type_sequence_about_the_forward_mutation(self):
        r, b = Recorder(), batch()
        out = MSRModel.forward_cycle(r, b)
        kind, fwd, _ = r.calls[1]
        self.assertEqual(kind, 'mt')
        self.assertTrue(torch.equal(fwd['mt_sequence_tokens'], b['wt_sequence_tokens']))
        self.assertTrue(torch.equal(fwd['wt_id'], b['wt_id'])); self.assertTrue(torch.equal(fwd['mt_id'], b['mt_id']))
        self.assertTrue(torch.equal(out['mt_fwd'], torch.tensor([0.25, 2.0])))

    def test_only_the_requested_legs_are_run(self):
        r = Recorder(); out = MSRModel.forward_cycle(r, batch(), legs=('wt_rev',))
        self.assertEqual(list(out), ['wt_rev']); self.assertEqual([c[0] for c in r.calls], ['wt'])
        r = Recorder(); out = MSRModel.forward_cycle(r, batch(), legs=('mt_fwd',))
        self.assertEqual(list(out), ['mt_fwd']); self.assertEqual([c[0] for c in r.calls], ['mt'])

    def test_the_input_batch_is_not_modified(self):
        r, b = Recorder(), batch(); before = {k: v.clone() for k, v in b.items()}
        MSRModel.forward_cycle(r, b)
        for k in b:
            self.assertTrue(torch.equal(b[k], before[k]), k)

    def test_training_mode_is_refused(self):
        r = Recorder(); r.training = True
        with self.assertRaises(AssertionError):
            MSRModel.forward_cycle(r, batch())


if __name__ == '__main__':
    unittest.main()
