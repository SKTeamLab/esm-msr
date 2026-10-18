"""Smoke test of the real loss composition with censored items, on a stub model (no GPU, no backbone).

Checks the plumbing that is easy to get wrong: censored singles reach the WT rank and regression terms, a '>5' / '<-1' item
is penalised only on the wrong side of its bound, flip columns accept censored members, and with nothing censored the
composition equals the uncensored one exactly.
"""
import types
import unittest

import torch

from esm_msr.losses import ListMLELoss


class HP(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def make_hp(**kw):
    hp = HP(wt_list_size=4, micro_batch_size=16, lambda_reg_wt=1.0, lambda_rank_wt=1.0, lambda_reg_mt_master=1.0, lambda_mt_colrank=1.0,
            mt_comp_offset=1.0, mt_comp_subst=1.0, mt_comp_int=1.0, flip_list_min=3, mask_strategy=None, subfloor_rank_only=True, cond_weight=1.0, native_cond_weight=1.0,
            mt_single_anchor_weight=0.0, mt_single_anchor_frac=1.0, censor_reg_weight=1.0, censor_floor_hinge=False,
            include_out_of_range=False, censor_floor=None)
    hp.update(kw)
    return hp


class StubModel(torch.nn.Module):
    """pred = scale * feature + bias; the same affine map for both passes, with separate parameters."""
    dedup_backbone = False

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))     # wt, mt

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        i = 0 if pass_type == 'wt' else 1
        raw = micro['feat'] * self.w[i]
        return {'pred_calibrated': raw, 'pred_raw': raw}


def make_batch(n_single=12, with_cens=True):
    """12 WT-context singles (3 lists of 4) and one flip column of 6 cond items."""
    g = torch.Generator().manual_seed(0)
    B = n_single + 6
    feat = torch.randn(B, generator=g)
    ddG = torch.randn(B, generator=g)
    cens = torch.zeros(B, dtype=torch.long)
    bound = torch.full((B,), float('nan'))
    src = torch.zeros(B, dtype=torch.long)
    if with_cens:
        cens[1], cens[6], cens[9] = -1, 1, -1            # dead / hyper / dead singles
        bound[1], bound[6], bound[9] = -3.0, 2.0, -3.0
        src[1], src[6], src[9] = 1, 1, 1
        cens[n_single + 4] = -1                          # one dead member of the flip column
        bound[n_single + 4] = -2.0
        src[n_single + 4] = 1
        ddG[1], ddG[6], ddG[9], ddG[n_single + 4] = -3.0, 2.0, -3.0, -2.0
    return {
        'ddG': ddG, 'feat': feat, 'mut_mask': torch.ones(B, 1, dtype=torch.bool),
        'subset_type': ['single'] * n_single + ['cond'] * 6,
        'flip_key': [''] * n_single + ['colA'] * 6,
        'reg_ok': torch.ones(B, dtype=torch.bool), 'cens': cens, 'cens_bound': bound, 'cens_src': src,
        'ddG_additive': torch.full((B,), float('nan')), 'dddG': torch.full((B,), float('nan')),
        'dG_wt': torch.full((B,), 2.0), 'bg_offset': torch.zeros(B),
    }


def run(batch, link_head=None, **hp_kw):
    from esm_msr import training
    cls = next(v for v in vars(training).values() if isinstance(v, type) and hasattr(v, '_compose_losses_streaming_and_backward'))
    stub = types.SimpleNamespace()
    stub.hparams = make_hp(**hp_kw)
    stub.model = StubModel()
    stub.link_head = link_head
    stub.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    stub.crit_reg = torch.nn.MSELoss(reduction='none')
    stub.crit_rank_wt, stub.crit_rank_mt = ListMLELoss(), ListMLELoss()
    stub._warned_unrouted = True
    stub.global_step = 0
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_compute_flip_loss', '_compute_block_components', '_aligned_chunks', '_subset_weights', '_plan_units'):
        setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    out = cls._compose_losses_streaming_and_backward(stub, batch)
    return out, stub


class TestComposeCensoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr.training  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def test_censored_batch_runs_and_reports_the_new_terms(self):
        out, stub = run(make_batch(), include_out_of_range=True)
        self.assertIn('L_reg_wt_cens', out)
        self.assertIn('L_reg_mt_cens', out)
        self.assertTrue(torch.isfinite(stub.model.w.grad).all())
        self.assertEqual(stub._cens_diag, (3, 1))

    def test_no_censored_item_gives_exactly_the_uncensored_result(self):
        a, sa = run(make_batch(with_cens=False))
        b_batch = make_batch(with_cens=False)
        b_batch.pop('cens'), b_batch.pop('cens_bound'), b_batch.pop('cens_src')     # an old batch without the fields
        b, sb = run(b_batch)
        self.assertNotIn('L_reg_wt_cens', a)
        for k in a:
            self.assertEqual(float(a[k]), float(b[k]), k)
        self.assertTrue(torch.equal(sa.model.w.grad, sb.model.w.grad))

    def test_censored_single_does_not_enter_the_ordinary_regression(self):
        # Make the dead single's stored label absurd; it must not influence reg_wt (only the hinge uses its bound).
        base = make_batch()
        weird = make_batch()
        weird['ddG'][1] = 123.0
        a, _ = run(base)
        b, _ = run(weird)
        self.assertEqual(float(a['L_reg_wt']), float(b['L_reg_wt']))

    def test_hinge_weight_zero_leaves_censored_items_out_of_the_regression(self):
        out, _ = run(make_batch(), censor_reg_weight=0.0)
        self.assertNotIn('L_reg_wt_cens', out)
        self.assertNotIn('L_reg_mt_cens', out)

    def test_rank_terms_still_run_with_censored_members(self):
        out, _ = run(make_batch())
        self.assertIn('L_rank_wt', out)
        self.assertIn('L_rank_mt', out)


if __name__ == '__main__':
    unittest.main()


class TestMtGateAndMicroBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import esm_msr.training  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"training module not importable: {e}")

    def test_mt_rank_loss_trains_even_when_the_mt_regression_is_off(self):
        # the MT unit used to exist only when lambda_reg_mt_master > 0, so rank-only (or interaction-only) MT training silently trained nothing
        out, stub = run(make_batch(with_cens=False), lambda_reg_mt_master=0.0, lambda_mt_colrank=1.0)
        self.assertIn('L_rank_mt', out)
        self.assertNotIn('L_reg_mt', out)
        self.assertTrue(stub.model.w.grad[1].abs() > 0)

    def test_micro_batch_size_is_not_rounded_to_the_wt_list_size(self):
        from unittest import mock
        sizes = []
        orig = StubModel.forward_partitioned

        def spy(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
            if pass_type == 'mt':
                sizes.append(int(micro['feat'].shape[0]))
            return orig(self, micro, pass_type, mask_strategy=mask_strategy, cached_wt_esm3=cached_wt_esm3)

        with mock.patch.object(StubModel, 'forward_partitioned', spy):
            run(make_batch(), lambda_mt_colrank=0.0, micro_batch_size=5, wt_list_size=4)       # 6 MT rows: 5 + 1, not 4 + 2
        self.assertEqual(sizes, [5, 1])
