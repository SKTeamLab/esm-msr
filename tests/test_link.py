"""--link on the released objective: the link itself, and how the composition uses it. A link that is the identity over the data reproduces
the released losses; saturation is absorbed by the link instead of being charged to the prediction; the WT regression of a double onto the
sum of its singles stays latent; items without a dG_wt leave the linked regressions and keep their rank terms; checkpoints keep the link."""
import types
import unittest

import torch

from esm_msr.link import MonotoneLink
from esm_msr.losses import ListMLELoss


class TestMonotoneLink(unittest.TestCase):
    def test_monotone_and_bounded(self):
        z = torch.linspace(-40, 40, 4001)
        for tl, th in ((0.1, 0.1), (0.05, 0.5), (0.5, 0.05), (1.0, 1.0)):
            h = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=tl, tau_hi=th)(z)
            self.assertTrue(torch.all(h[1:] >= h[:-1] - 1e-6), (tl, th))
            self.assertLessEqual(float(h.max()), 5.0 + 1e-2)
            self.assertGreaterEqual(float(h.min()), -1.0 - 1e-2)

    def test_a_double_far_below_the_floor_is_reported_at_the_floor(self):
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.2, tau_hi=0.2)
        obs = link.obs_ddG(torch.tensor([-9.0, -1.0]), torch.tensor([3.0, 3.0]))
        self.assertAlmostEqual(float(obs[0]), -4.0, places=2)
        self.assertAlmostEqual(float(obs[1]), -1.0, places=1)

    def test_gradients_reach_every_parameter(self):
        link = MonotoneLink()
        link.obs_ddG(torch.tensor([-8.0, 7.0, 0.3]), torch.tensor([2.0, 2.0, 2.0])).sum().backward()
        for n, p in link.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n)


class CalHead(torch.nn.Module):
    use_scale, bias = False, None


class StubModel(torch.nn.Module):
    """pred = feature * w[pass]: one parameter per pass, so the gradient of each pass is visible."""
    lora_mode = 'ensemble'

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.calibration_head_wt = CalHead()

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, detach_calibration=False):
        v = micro['feat'] * self.w[0 if pass_type == 'wt' else 1]
        return {'pred_calibrated': v, 'pred_raw': v}


def make_batch(ddG, feat, n_mut, dG_wt, ddG_additive=None, floor_offset=None):
    B = len(ddG)
    mm = torch.zeros(B, 2, dtype=torch.bool)
    for i, n in enumerate(n_mut):
        mm[i, :n] = True
    add = torch.tensor(ddG_additive if ddG_additive is not None else [float('nan')] * B, dtype=torch.float32)
    return {'ddG': torch.tensor(ddG, dtype=torch.float32), 'feat': torch.tensor(feat, dtype=torch.float32), 'mut_mask': mm,
            'valid_dddG_mask': torch.tensor([n == 2 for n in n_mut]), 'ddG_additive': add,
            'dddG': torch.full((B,), float('nan')), 'subset_type': ['single' if n == 1 else 'double' for n in n_mut],
            'dG_wt': torch.tensor(dG_wt, dtype=torch.float32), 'mutations': [[('A', i, 'G')] * n for i, n in enumerate(n_mut)],
            'dG_floor_offset': torch.tensor(floor_offset if floor_offset is not None else [0.0] * B, dtype=torch.float32)}


def run(batch, link=None, **hp):
    from esm_msr import training
    cls = training.ESM3EpistasisLightningModule
    h = dict(subset_size=4, micro_batch_size=64, mt_reg_mask='all', zero_epistasis_for_singles=True, detach_ensemble_input=True,
             mask_strategy=None, detach_regression=False, lambda_reg_wt=1.0, lambda_rank_wt=0.0, lambda_reg_combined=1.0,
             lambda_rank_combined=0.0, lambda_epi_combined=0.0, double_weight=1.0, mut_ctx_weight=0.5, reversion_weight=0.5)
    bias = h.pop('bias', None) if 'bias' in h else hp.pop('bias', None)
    ns = hp.pop('ns', None)
    h.update(hp)
    model = StubModel()
    if bias is not None:                                     # a WT calibration bias (the stub's predictions do not add it themselves)
        head = CalHead(); head.bias = torch.tensor(float(bias)); model.calibration_head_wt = head
    stub = types.SimpleNamespace(hparams=types.SimpleNamespace(**h), model=model, link_head=link, ns_head=ns,
                                 peft_manager=types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False),
                                 crit_reg=torch.nn.MSELoss(reduction='none'), crit_rank_wt=ListMLELoss(), crit_rank_combined=ListMLELoss())
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    stub._compute_rank_loss = types.MethodType(cls._compute_rank_loss, stub)
    out = cls._compose_losses_streaming_and_backward(stub, batch)
    return out, stub


IDENTITY = lambda: MonotoneLink(lo=-60.0, hi=60.0, tau_lo=0.01, tau_hi=0.01)     # the identity on any dG the tests use


class TestComposition(unittest.TestCase):
    def setUp(self):
        # three singles and a double (its additive target is the sum of two singles), all in range
        self.batch = make_batch(ddG=[-0.5, -1.2, 0.3, -2.0], feat=[-0.4, -1.0, 0.1, -1.5], n_mut=[1, 1, 1, 2], dG_wt=[2.0] * 4,
                                ddG_additive=[float('nan')] * 3 + [-1.7])

    def test_an_identity_link_reproduces_the_released_losses_and_gradients(self):
        plain, s0 = run(self.batch)
        linked, s1 = run(self.batch, link=IDENTITY())
        for k in ('L_reg_wt', 'L_reg_combined'):
            self.assertAlmostEqual(plain[k], linked[k], places=5)
        torch.testing.assert_close(s0.model.w.grad, s1.model.w.grad, atol=1e-5, rtol=1e-4)

    def test_saturation_is_absorbed_by_the_link_not_charged_to_the_prediction(self):
        # a double measured at the floor (dG_wt 2, observed ddG -3 = dG -1) and predicted far below it: no regression error through the link
        b = make_batch(ddG=[-3.0], feat=[-9.0], n_mut=[2], dG_wt=[2.0], ddG_additive=[-9.0])
        plain, _ = run(b, lambda_reg_wt=0.0)
        linked, _ = run(b, link=MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05), lambda_reg_wt=0.0)
        self.assertGreater(plain['L_reg_combined'], 30.0)
        self.assertLess(linked['L_reg_combined'], 1e-3)

    def test_the_wt_regression_of_a_double_stays_latent(self):
        # the double's additive target (-9) is below the floor; through the link it would cost ~0, latent it costs (feat - target)^2
        b = make_batch(ddG=[-3.0], feat=[-7.0], n_mut=[2], dG_wt=[2.0], ddG_additive=[-9.0])
        out, _ = run(b, link=MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05), lambda_reg_combined=0.0)
        self.assertAlmostEqual(out['L_reg_wt'], 4.0, places=4)

    def test_a_single_is_regressed_through_the_link(self):
        b = make_batch(ddG=[-3.0], feat=[-9.0], n_mut=[1], dG_wt=[2.0])
        out, _ = run(b, link=MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.05, tau_hi=0.05), lambda_reg_combined=0.0)
        self.assertLess(out['L_reg_wt'], 1e-3)

    def test_items_without_a_dG_wt_leave_the_regressions_but_keep_their_rank_terms(self):
        b = make_batch(ddG=[-0.5, -1.2, 0.3, -2.0], feat=[-0.4, -1.0, 0.1, -1.5], n_mut=[1, 1, 1, 1], dG_wt=[float('nan')] * 4)
        out, stub = run(b, link=IDENTITY(), lambda_rank_wt=1.0, lambda_rank_combined=1.0)
        self.assertNotIn('L_reg_wt', out)
        self.assertNotIn('L_reg_combined', out)
        self.assertIn('L_rank_wt', out)
        self.assertIn('L_rank_combined', out)
        self.assertTrue(torch.isfinite(stub.model.w.grad).all())

    def test_the_link_learns_from_the_regressions(self):
        link = MonotoneLink()
        run(self.batch, link=link)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in link.parameters()), 0.0)


class TestInverse(unittest.TestCase):
    def test_round_trip_inside_the_range_and_clamped_where_h_is_flat(self):
        link = MonotoneLink(lo=-1.0, hi=5.0, tau_lo=0.5, tau_hi=0.5)
        z = torch.linspace(-0.5, 4.0, 50)
        torch.testing.assert_close(link.inverse(link(z)), z, atol=2e-4, rtol=1e-4)
        self.assertAlmostEqual(float(link.inverse(torch.tensor(-1.5))), -3.0, places=4)          # below the floor: lo - 2
        self.assertLess(float(link.inverse(torch.tensor(-0.8))), -1.3)                           # compressed near the floor: deeper
        sharp = MonotoneLink(lo=0.0, hi=5.0, tau_lo=0.05, tau_hi=0.05)
        self.assertAlmostEqual(float(sharp.inverse(torch.tensor(-0.9))), -2.0, places=4)          # below a sharp floor too


def tf_batch(ddG_A, ddG_B, feat, dG_wt=2.0, single_feats=()):
    """One double (singles ddG_A, ddG_B; measured double ddG_A + ddG_B - 0.3) plus optional singles, all in one library."""
    n = 1 + len(single_feats)
    b = make_batch(ddG=[ddG_A + ddG_B - 0.3] + [-0.5] * len(single_feats), feat=[feat] + list(single_feats), n_mut=[2] + [1] * len(single_feats),
                   dG_wt=[dG_wt] * n, ddG_additive=[ddG_A + ddG_B] + [float('nan')] * len(single_feats))
    b['ddG_A'] = torch.tensor([ddG_A] + [float('nan')] * len(single_feats))
    b['ddG_B'] = torch.tensor([ddG_B] + [float('nan')] * len(single_feats))
    return b


class TestTeacherForcingSingles(unittest.TestCase):
    def test_latent_link_with_an_identity_link_is_the_released_forcing(self):
        b = tf_batch(-1.2, -0.4, -1.5, single_feats=(-0.3, -1.1, 0.2))
        m, s0 = run(b, link=IDENTITY(), tf_singles='measured')
        l, s1 = run(b, link=IDENTITY(), tf_singles='latent_link')
        self.assertAlmostEqual(m['L_reg_combined'], l['L_reg_combined'], places=5)
        torch.testing.assert_close(s0.model.w.grad, s1.model.w.grad, atol=1e-5, rtol=1e-4)

    def test_latent_link_takes_a_near_floor_single_deeper_by_at_most_the_cap(self):
        # floor at 0: single A measured at dG -0.9 (below it, not invertible) -> capped at 0.5 below its measurement, ddG_A -3.4;
        # B (dG 1.5) is in range and stays where it is
        link = MonotoneLink(lo=0.0, hi=5.0, tau_lo=0.05, tau_hi=0.05)
        b = tf_batch(-2.9, -0.5, -5.0)
        lat = -3.4 + float(link.inverse(torch.tensor(1.5)) - 2.0)
        self.assertAlmostEqual(lat, -3.9, places=3)
        out, _ = run(b, link=link, tf_singles='latent_link', lambda_reg_combined=0.0)
        self.assertAlmostEqual(out['L_reg_wt'], (-5.0 - lat) ** 2, places=4)                    # the WT double target is the latent sum
        wide, _ = run(b, link=link, tf_singles='latent_link', lambda_reg_combined=0.0, tf_latent_max_shift=1.5)
        self.assertAlmostEqual(wide['L_reg_wt'], (-5.0 - (lat - 1.0)) ** 2, places=4)          # the cap sets how deep

    def test_inside_the_range_the_inverse_link_sets_the_depth(self):
        # a soft knee: a single at dG 0.2 is moved by less than the cap, exactly to h^-1
        link = MonotoneLink(lo=-0.5, hi=5.0, tau_lo=0.5, tau_hi=0.5)
        b = tf_batch(-1.8, -0.5, -5.0)                                                         # dG_A 0.2, dG_B 1.5
        inv = lambda dg: float(link.inverse(torch.tensor(dg))) - 2.0
        lat = inv(0.2) + inv(1.5)
        self.assertGreater(inv(0.2), -1.8 - 0.5)                                                  # within the cap
        self.assertLess(inv(0.2), -1.8 - 0.05)                                                    # but deeper than measured
        out, _ = run(b, link=link, tf_singles='latent_link', lambda_reg_combined=0.0)
        self.assertAlmostEqual(out['L_reg_wt'], (-5.0 - lat) ** 2, places=4)
        meas, _ = run(b, link=link, tf_singles='measured', lambda_reg_combined=0.0)
        self.assertAlmostEqual(meas['L_reg_wt'], (-5.0 + 2.3) ** 2, places=4)                    # released: the measured sum

    def test_predicted_forcing_reads_no_single_label_and_adds_one_bias_per_extra_mutation(self):
        link = IDENTITY()
        b1 = tf_batch(-1.2, -0.4, -1.5)
        b2 = tf_batch(-0.1, 0.3, -1.5)                                                          # other single labels, same double
        b2['ddG'] = b1['ddG'].clone()
        o1, s1 = run(b1, link=link, tf_singles='predicted', lambda_reg_wt=0.0, bias=0.5)
        o2, s2 = run(b2, link=link, tf_singles='predicted', lambda_reg_wt=0.0, bias=0.5)
        torch.testing.assert_close(s1.model.w.grad, s2.model.w.grad)
        # WT pass = MT pass = feat (-1.5); forced = WT + bias (0.5); combined = 0.5 (-1.0) + 0.5 (-1.5) = -1.25 against ddG_AB -1.9
        self.assertAlmostEqual(o1['L_reg_combined'], (-1.25 + 1.9) ** 2, places=4)
        m1, _ = run(b1, link=link, tf_singles='measured', lambda_reg_wt=0.0, bias=0.5)
        m2, _ = run(b2, link=link, tf_singles='measured', lambda_reg_wt=0.0, bias=0.5)
        self.assertNotAlmostEqual(m1['L_reg_combined'], m2['L_reg_combined'], places=3)          # the released forcing does read them


class TestNonspecificTerm(unittest.TestCase):
    def test_piecewise_linear_and_constant_beyond_the_knots(self):
        from esm_msr.link import NonspecificTerm
        f = NonspecificTerm(lo=-1.0, hi=1.0, step=1.0)
        with torch.no_grad():
            f.values.copy_(torch.tensor([0.4, 0.2, 0.0]))
        torch.testing.assert_close(f(torch.tensor([-5.0, -1.0, -0.5, 0.5, 1.0, 7.0])), torch.tensor([0.4, 0.4, 0.3, 0.1, 0.0, 0.0]))

    def _ns(self, value):
        from esm_msr.link import NonspecificTerm
        f = NonspecificTerm(lo=-3.0, hi=5.0, step=0.5)
        with torch.no_grad():
            f.values.fill_(value)
        return f

    def test_it_shifts_doubles_only_and_reads_the_additive_expectation(self):
        # double: forced WT half = measured additive -1.6, MT = feat -1.5; f = 0.4 everywhere -> combined = -1.55 + 0.4 against ddG_AB -1.9
        b = tf_batch(-1.2, -0.4, -1.5)
        out, _ = run(b, link=IDENTITY(), ns=self._ns(0.4), lambda_reg_wt=0.0)
        self.assertAlmostEqual(out['L_reg_combined'], (-1.55 + 0.4 + 1.9) ** 2, places=4)
        s = make_batch(ddG=[-0.5], feat=[-0.3], n_mut=[1], dG_wt=[2.0])                     # a single: no f
        o1, _ = run(s, link=IDENTITY(), ns=self._ns(0.4), lambda_reg_wt=0.0)
        o0, _ = run(s, link=IDENTITY(), lambda_reg_wt=0.0)
        self.assertAlmostEqual(o1['L_reg_combined'], o0['L_reg_combined'], places=6)

    def test_f_learns_from_doubles_and_never_trains_the_wt_half(self):
        ns = self._ns(0.0)
        b = tf_batch(-1.2, -0.4, -1.5, single_feats=(-0.3, -1.1))
        _, stub = run(b, link=IDENTITY(), ns=ns, lambda_reg_wt=0.0)
        self.assertGreater(float(ns.values.grad.abs().sum()), 0.0)
        ns2 = self._ns(0.0)
        s = make_batch(ddG=[-0.5, -1.0], feat=[-0.3, -0.8], n_mut=[1, 1], dG_wt=[2.0, 2.0])
        _, stub2 = run(s, link=IDENTITY(), ns=ns2, lambda_reg_wt=0.0)
        self.assertTrue(ns2.values.grad is None or float(ns2.values.grad.abs().sum()) == 0.0)
        self.assertEqual(float(stub2.model.w.grad[0]), 0.0)                                    # the WT pass gets nothing from the combined losses

    def test_validation_adds_f_to_comb_and_twice_f_to_the_mt_readout(self):
        from esm_msr import training
        ns = self._ns(0.3)
        stub = types.SimpleNamespace(ns_head=ns, model=StubModel())
        out = {'combined_pred': torch.tensor([1.0, 2.0]), 'mt_lora_pred': torch.tensor([1.0, 2.0]), 'wt_lora_pred': torch.tensor([1.0, 2.0]),
               'epi_pred': torch.tensor([0.0, 0.0])}
        batch = {'mut_mask': torch.tensor([[True, False], [True, True]])}
        r = training.ESM3EpistasisLightningModule._apply_nonspecific(stub, out, batch, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(r['combined_pred'], torch.tensor([1.0, 2.3]))
        torch.testing.assert_close(r['mt_lora_pred'], torch.tensor([1.0, 2.6]))
        torch.testing.assert_close(r['wt_lora_pred'], torch.tensor([1.0, 2.0]))


class TestCheckpoint(unittest.TestCase):
    def test_the_link_is_kept_in_the_filtered_checkpoint(self):
        from esm_msr import training
        stub = types.SimpleNamespace(peft_manager=types.SimpleNamespace())
        ck = {'state_dict': {'link_head.lo': torch.tensor(0.0), 'ns_head.values': torch.zeros(3), 'model.peft_wt.lora_A': torch.zeros(1), 'model.esm.blocks.0.w': torch.zeros(1)}}
        training.ESM3EpistasisLightningModule.on_save_checkpoint(stub, ck)
        self.assertEqual(set(ck['state_dict']), {'link_head.lo', 'ns_head.values', 'model.peft_wt.lora_A'})


if __name__ == '__main__':
    unittest.main()
