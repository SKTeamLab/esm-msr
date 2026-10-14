"""--censor_hinge_scale latent: a variant reported below the range is bounded on the latent scale (at or below the link's floor knee), so it
can always be satisfied by the variant's own prediction and the link gets no gradient from it. With 'observed' (the default before 2026-10-13) the bound -1
cannot be met while the link's floor sits above it, and the hinge drags the link."""
import unittest

import torch

try:
    from tests.test_compose_censoring import run
except ImportError:  # run from inside tests/
    from test_compose_censoring import run

from esm_msr import censoring
from esm_msr.link import MonotoneLink


def batch(feat_dead):
    """Four WT-context singles (one list), dG_wt 2; item 0 reported '<-1' (bound on the ddG scale -3), predicted latent ddG feat_dead."""
    feat = torch.tensor([feat_dead, -0.4, 0.3, -1.0])
    ddG = torch.tensor([-3.0, -0.5, 0.4, -1.1])
    cens = torch.tensor([-1, 0, 0, 0]); bound = torch.tensor([-3.0, float('nan'), float('nan'), float('nan')])
    return {'ddG': ddG, 'feat': feat, 'mut_mask': torch.ones(4, 1, dtype=torch.bool), 'subset_type': ['single'] * 4, 'flip_key': [''] * 4,
            'reg_ok': torch.ones(4, dtype=torch.bool), 'cens': cens, 'cens_bound': bound, 'cens_src': torch.tensor([1, 0, 0, 0]),
            'ddG_additive': torch.full((4,), float('nan')), 'dddG': torch.full((4,), float('nan')), 'dG_wt': torch.full((4,), 2.0),
            'bg_offset': torch.zeros(4)}


def floor_link():
    return MonotoneLink(lo=-0.5, hi=5.0, tau_lo=0.1, tau_hi=0.1)          # floor above the reported bound -1


HP = dict(lambda_rank_wt=0.0, lambda_mt_colrank=0.0, lambda_reg_mt_master=0.0, wt_list_size=4)


def hinge_grad(link):
    return sum(float(p.grad.abs().sum()) for p in link.parameters() if p.grad is not None)


class TestLatentHinge(unittest.TestCase):
    def test_a_variant_deep_below_the_floor_costs_nothing_on_the_latent_scale(self):
        link = floor_link()
        out, _ = run(batch(-6.0), link_head=link, censor_hinge_scale='latent', **HP)             # latent dG 2 - 6 = -4, below the knee -0.5
        self.assertEqual(out.get('L_reg_wt_cens', 0.0), 0.0)

    def test_on_the_observed_scale_the_same_variant_drags_the_link(self):
        link = floor_link()
        out, _ = run(batch(-6.0), link_head=link, censor_hinge_scale='observed', **HP)           # observed ddG ~ -2.5 > bound -3: violated
        self.assertGreater(out.get('L_reg_wt_cens', 0.0), 0.0)
        self.assertGreater(hinge_grad(link), 0.0)

    def test_a_variant_predicted_above_the_floor_is_pushed_down_and_the_link_is_left_alone(self):
        link = floor_link()
        out, stub = run(batch(0.0), link_head=link, censor_hinge_scale='latent', lambda_reg_wt=1.0, **{k: v for k, v in HP.items()})
        self.assertGreater(out.get('L_reg_wt_cens', 0.0), 0.0)
        self.assertNotEqual(float(stub.model.w.grad[0]), 0.0)
        # the hinge alone gives the link nothing: the ordinary items are taken out of the linked losses (no dG_wt)
        link2 = floor_link()
        b = batch(0.0)
        b['dG_wt'][1:] = float('nan')
        out2, _ = run(b, link_head=link2, censor_hinge_scale='latent', **HP)
        self.assertGreater(out2.get('L_reg_wt_cens', 0.0), 0.0)
        self.assertEqual(hinge_grad(link2), 0.0)
        link3 = floor_link()
        run(b, link_head=link3, censor_hinge_scale='observed', **HP)                             # the observed hinge does drag it
        self.assertGreater(hinge_grad(link3), 0.0)

    def test_the_latent_bound_is_the_inverse_inside_the_range_and_the_knee_beyond_it(self):
        link = floor_link()
        crit = torch.nn.MSELoss(reduction='none')
        # bound 1.0 (inside the range): the latent bound is h^-1(1.0) ~ 1.0; a latent of 1.5 violates it by about 0.5
        L = censoring.latent_censored_regression_loss(crit, link, torch.tensor([1.5]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([-1]))
        self.assertAlmostEqual(float(L), 0.25, places=2)
        # bound -1 (below the floor): the latent bound is the knee -0.5
        L = censoring.latent_censored_regression_loss(crit, link, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([-1]))
        self.assertAlmostEqual(float(L), 0.25, places=4)


if __name__ == '__main__':
    unittest.main()
