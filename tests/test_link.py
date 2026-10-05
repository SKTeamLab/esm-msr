import unittest

import torch

from esm_msr.link import MonotoneLink
try:
    from tests.test_compose_censoring import make_batch, run
except ImportError:  # run from inside tests/
    from test_compose_censoring import make_batch, run


class TestMonotoneLink(unittest.TestCase):
    def test_monotone_and_bounded_for_any_temperatures(self):
        z = torch.linspace(-40, 40, 4001)
        for tl, th in ((0.1, 0.1), (0.05, 2.0), (2.0, 0.05), (1.0, 1.0)):
            h = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=tl, tau_hi=th)(z)
            self.assertTrue(torch.all(h[1:] >= h[:-1] - 1e-6), (tl, th))
        for tl, th in ((0.1, 0.1), (0.05, 0.5), (0.5, 0.05), (1.0, 1.0)):      # temperatures small against the span of 6
            h = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=tl, tau_hi=th)(z)
            self.assertLessEqual(float(h.max()), 5.0 + 1e-2)
            self.assertGreaterEqual(float(h.min()), -1.0 - 1e-2)

    def test_plateaus_at_the_floor_and_ceiling_and_follows_the_identity_inside(self):
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.1, tau_hi=0.1)
        self.assertAlmostEqual(float(link(torch.tensor(-30.0))), -1.0, places=3)
        self.assertAlmostEqual(float(link(torch.tensor(30.0))), 5.0, places=3)
        for z in (0.5, 2.0, 3.5):
            self.assertAlmostEqual(float(link(torch.tensor(z))), z, places=2)

    def test_sharper_temperature_approaches_a_hard_clamp(self):
        z = torch.linspace(-6, 10, 200)
        soft = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.01, tau_hi=0.01)(z)
        self.assertLess(float((soft - z.clamp(-1.0, 5.0)).abs().max()), 0.02)

    def test_a_double_far_below_the_floor_is_reported_at_the_floor(self):
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.2, tau_hi=0.2)
        # dG_wt 3, latent -9: latent dG = -6 -> observed about -1, i.e. observed ddG about -4
        obs = link.obs_ddG(torch.tensor([-9.0, -1.0]), torch.tensor([3.0, 3.0]))
        self.assertAlmostEqual(float(obs[0]), -4.0, places=2)
        self.assertAlmostEqual(float(obs[1]), -1.0, places=1)

    def test_background_offset_scores_a_conditional_as_its_double(self):
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05)
        # ddG(A|B) = -2 on a background whose own ddG is -3 (dG_wt = 2): double at latent dG -3 -> floors at -1, observed ddG -3
        obs = link.obs_ddG(torch.tensor([-2.0]), torch.tensor([2.0]), torch.tensor([-3.0]))
        self.assertAlmostEqual(float(obs), -3.0, places=2)

    def test_gradients_reach_every_parameter_and_unknown_dG_wt_is_nan(self):
        link = MonotoneLink()
        out = link.obs_ddG(torch.tensor([-8.0, 7.0, 0.3]), torch.tensor([2.0, 2.0, 2.0]))
        out.sum().backward()
        for n, p in link.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), n)
        self.assertTrue(torch.isnan(link.obs_ddG(torch.tensor([0.0]), torch.tensor([float('nan')]))).all())

    def test_fixed_bounds_are_not_trainable(self):
        link = MonotoneLink(learn_bounds=False)
        trainable = {n for n, p in link.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {'raw_tau_lo', 'raw_tau_hi'})


class TestLinkInTheLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr.training  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def test_a_flat_link_reproduces_the_calibrated_regression_exactly(self):
        # bounds far outside the data and a tiny temperature make h the identity: the link run must equal the plain run
        ident = MonotoneLink(lo=-100.0, hi=100.0, tau_lo=1e-3, tau_hi=1e-3, learn_bounds=False)
        plain, _ = run(make_batch(with_cens=False))
        linked, _ = run(make_batch(with_cens=False), link_head=ident, link='softclamp')
        for k in ('L_reg_wt', 'L_reg_mt', 'L_rank_wt', 'L_rank_mt'):
            self.assertAlmostEqual(plain[k], linked[k], places=3, msg=k)

    def test_saturation_is_absorbed_by_the_link_not_charged_to_the_prediction(self):
        # A single measured at the floor (observed ddG = -3 with dG_wt 2, i.e. dG = -1). A latent far below it is penalised by the
        # plain regression but fits perfectly through the link.
        batch = make_batch(with_cens=False)
        batch['ddG'][:12] = -3.0
        batch['feat'][:12] = -9.0               # latent -9 (w = 1)
        plain, _ = run(batch, lambda_rank_wt=0.0, lambda_rank_mt=0.0, lambda_reg_mt=0.0)
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05, learn_bounds=False)
        linked, _ = run(batch, link_head=link, link='softclamp', lambda_rank_wt=0.0, lambda_rank_mt=0.0, lambda_reg_mt=0.0)
        self.assertGreater(plain['L_reg_wt'], 20.0)
        self.assertLess(linked['L_reg_wt'], 0.05)

    def test_reg_ok_is_ignored_while_the_link_is_on(self):
        batch = make_batch(with_cens=False)
        batch['reg_ok'][:] = False                 # would withhold every cond item from the plain regression
        plain, _ = run(batch)
        self.assertNotIn('L_reg_mt', plain)
        link = MonotoneLink(lo=-1.0, hi=5.0, learn_bounds=False)
        linked, _ = run(batch, link_head=link, link='softclamp')
        self.assertIn('L_reg_mt', linked)

    def test_items_without_a_dG_wt_are_left_out_of_the_regression_but_keep_their_rank_terms(self):
        batch = make_batch(with_cens=False)
        batch['dG_wt'][:] = float('nan')
        link = MonotoneLink(learn_bounds=False)
        out, _ = run(batch, link_head=link, link='softclamp')
        self.assertNotIn('L_reg_wt', out)
        self.assertNotIn('L_reg_mt', out)
        self.assertIn('L_rank_wt', out)
        self.assertIn('L_rank_mt', out)

    def test_conditional_items_are_scored_as_the_double_they_came_from(self):
        # cond item: ddG(A|B) = -1 on a background with ddG_B = -2 (so ddG_AB = -3); dG_wt 2, so the double's dG is -1: at the floor.
        batch = make_batch(with_cens=False)
        batch['bg_offset'][12:] = -2.0
        batch['ddG'][12:] = -1.0
        batch['feat'][12:] = -1.0                   # MT latent for ddG(A|B)
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05, learn_bounds=False)
        out, _ = run(batch, link_head=link, link='softclamp', lambda_rank_wt=0.0, lambda_rank_mt=0.0, lambda_reg_wt=0.0)
        # latent double dG = 2 - 2 - 1 = -1 -> h = -1 -> observed ddG -3 = target ddG_AB: no residual
        self.assertLess(out['L_reg_mt'], 0.01)


if __name__ == '__main__':
    unittest.main()
