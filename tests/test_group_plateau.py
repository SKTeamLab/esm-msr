import unittest

import torch

from esm_msr.training import GroupPlateau


def make():
    a, b, c = (torch.nn.Parameter(torch.zeros(1)) for _ in range(3))
    opt = torch.optim.SGD([{'params': [a], 'lr': 1e-3, 'name': 'lora_wt'}, {'params': [b], 'lr': 1e-3, 'name': 'lora_mt'},
                           {'params': [c], 'lr': 5e-3, 'name': 'link'}])
    return opt, GroupPlateau(opt, ('lora_wt',)), GroupPlateau(opt, ('lora_mt',))


class TestGroupPlateau(unittest.TestCase):
    def test_each_head_cuts_only_its_own_groups(self):
        opt, wt, mt = make()
        for v in (0.5, 0.6, 0.6, 0.6):        # wt: gain, then two validations without one -> cut
            wt.step(v)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-4)
        self.assertEqual((opt.param_groups[1]['lr'], opt.param_groups[2]['lr']), (1e-3, 5e-3))
        for v in (0.1, 0.2, 0.3):             # mt keeps improving: no cut
            mt.step(v)
        self.assertEqual(opt.param_groups[1]['lr'], 1e-3)

    def test_a_frozen_group_stays_at_zero(self):
        opt, wt, _ = make()
        opt.param_groups[0]['lr'] = 0.0
        for v in (0.5, 0.5, 0.5, 0.5):
            wt.step(v)
        self.assertEqual(opt.param_groups[0]['lr'], 0.0)


if __name__ == '__main__':
    unittest.main()
