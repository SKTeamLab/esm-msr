"""--link_floor_file: a per-library offset on the link's FLOOR KNEE only.

The assay infers dG against a K50 predicted for the unfolded sequence, so an unfolded variant reads the error of that prediction: a constant
of the library. The offset supplies it as data (no parameters). What must hold: offset 0 reproduces the shared-floor link exactly (so every
earlier run is reproducible), the floor moves by the offset, the ceiling and the in-range identity do not move, the inverse puts a censored
variant's bound on its OWN library's knee, and the epistasis metrics score each double through its own offset.
"""
import unittest

import numpy as np
import torch

from esm_msr import epi_metrics
from esm_msr.link import MAX_FLOOR_OFFSET, MonotoneLink, numpy_link


def link(lo=-0.5, hi=5.0, tl=0.4, th=0.4):
    return MonotoneLink(lo=lo, hi=hi, tau_lo=tl, tau_hi=th)


class TestFloorOffset(unittest.TestCase):
    def test_offset_zero_and_none_reproduce_the_shared_floor_link(self):
        h, z = link(), torch.linspace(-20, 20, 2001)
        for off in (None, 0.0, torch.zeros_like(z)):
            self.assertTrue(torch.allclose(h(z, off), h(z), atol=0), f'offset {off} changed the link')

    def test_the_floor_moves_by_the_offset_and_the_ceiling_does_not(self):
        h = link()
        far_lo, far_hi = torch.tensor([-1e3]), torch.tensor([1e3])
        base = float(h(far_lo))
        for off in (-0.4, 0.3, 0.65):
            self.assertAlmostEqual(float(h(far_lo, off)) - base, off, places=4)
            self.assertAlmostEqual(float(h(far_hi, off)), float(h(far_hi)), places=6)

    def test_in_range_predictions_are_untouched(self):
        """The offset must not act as a per-library calibration shift: well inside the range h(z) ~ z whatever the offset."""
        h, z = link(), torch.tensor([1.5, 2.0, 3.0, 4.0])
        for off in (-0.6, 0.6):
            self.assertLess(float((h(z, off) - h(z)).abs().max()), 0.01)

    def test_still_monotone_and_bounded_with_offsets(self):
        h, z = link(), torch.linspace(-30, 30, 3001)
        for off in (-MAX_FLOOR_OFFSET, -0.5, 0.5, MAX_FLOOR_OFFSET):
            v = h(z, off)
            self.assertTrue(torch.all(v[1:] >= v[:-1] - 1e-6), off)
            self.assertLessEqual(float(v.max()), 5.0 + 1e-2)

    def test_a_wild_offset_is_clamped_and_never_crosses_the_ceiling(self):
        h = torch.tensor([0.0])
        for off in (1e4, -1e4, float('nan')):
            v = float(link()(h, off))
            self.assertTrue(np.isfinite(v))
        self.assertAlmostEqual(float(link()(torch.tensor([0.0]), 50.0)), float(link()(torch.tensor([0.0]), MAX_FLOOR_OFFSET)), places=6)

    def test_nan_offset_falls_back_to_the_shared_floor(self):
        h, z = link(), torch.tensor([-3.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(h(z, torch.full_like(z, float('nan'))), h(z), atol=1e-6))

    def test_per_item_offsets_are_applied_item_by_item(self):
        h = link()
        z = torch.tensor([-5.0, -5.0, -5.0])
        off = torch.tensor([-0.3, 0.0, 0.3])
        got = h(z, off)
        want = torch.tensor([float(h(z[:1], float(o))) for o in off])
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_obs_ddG_uses_the_offset(self):
        h = link(lo=-0.5, tl=0.2, th=0.2)
        lat, dgw = torch.tensor([-9.0]), torch.tensor([3.0])
        self.assertAlmostEqual(float(h.obs_ddG(lat, dgw, lo_offset=0.5)) - float(h.obs_ddG(lat, dgw)), 0.5, places=3)

    def test_inverse_puts_a_censored_bound_on_the_librarys_own_knee(self):
        """--censor_hinge_scale latent bounds a dead variant at the inverse of the reported limit. With a per-library floor that is the
        library's own knee, so the bound is as demanding as that library's readout allows."""
        h = link(lo=-0.5, tl=0.1, th=0.1)
        bound = torch.tensor([-1.0])                                    # below every floor here: the inverse returns the knee
        for off in (-0.4, 0.0, 0.45):
            z = float(h.inverse(bound, below_lo=0.0, above_hi=0.0, lo_offset=off))
            self.assertAlmostEqual(z, -0.5 + off, places=4)
        # inside the range the inverse is exact and barely depends on the offset
        inside = torch.tensor([2.0])
        self.assertAlmostEqual(float(h.inverse(inside, lo_offset=0.4)), float(h.inverse(inside)), places=3)

    def test_numpy_link_matches_the_torch_link(self):
        h = link()
        lo, hi, tl, th = (float(v) for v in h.params())
        g = numpy_link(lo, hi, tl, th)
        z = np.linspace(-12, 12, 501)
        for off in (0.0, -0.35, 0.4):
            want = h(torch.tensor(z, dtype=torch.float32), off).detach().numpy()
            self.assertLess(float(np.abs(g(z, off) - want).max()), 2e-3, off)
        self.assertLess(float(np.abs(g(z) - h(torch.tensor(z, dtype=torch.float32)).detach().numpy()).max()), 2e-3)


def items_for(offsets):
    """Two singles and their double in each of two libraries, deep below the floor, with the given per-library floor offsets."""
    keys, st, ddG, dddG, dgw, wt, foff = [], [], [], [], [], [], []
    for li, (c, off) in enumerate(zip((3.0, 3.0), offsets)):
        for p, r, y, lat in ((10, 'A', -2.6, -2.6), (20, 'C', -2.4, -2.4)):
            keys.append(((f'lib{li}', 'X', p, r),)); st.append('single'); ddG.append(y); dddG.append(np.nan)
            dgw.append(c); wt.append(lat); foff.append(off)
        keys.append(((f'lib{li}', 'X', 10, 'A'), (f'lib{li}', 'X', 20, 'C')))
        st.append('double'); ddG.append(-2.9); dddG.append(2.1); dgw.append(c); wt.append(-5.0); foff.append(off)
    a = lambda v: np.asarray(v, float)
    return {'mut_key': keys, 'subset_type': st, 'cens': np.zeros(len(st), int), 'ddG': a(ddG), 'dddG': a(dddG), 'dG_wt': a(dgw),
            'wt': a(wt), 'mt': a(wt), 'comb': a(wt), 'wt_rev': None, 'dG_floor_offset': a(foff)}


class TestMetricsUseTheOffset(unittest.TestCase):
    def test_the_table_scores_each_double_through_its_own_offset(self):
        h = numpy_link(-0.5, 5.0, 0.3, 0.3)
        flat = epi_metrics.Table(items_for((0.0, 0.0)), h, heads=('comb',))
        split = epi_metrics.Table(items_for((-0.4, 0.4)), h, heads=('comb',))
        self.assertEqual(flat.n, 2)
        self.assertTrue(np.allclose(flat.floor_off, 0.0))
        self.assertTrue(np.allclose(np.sort(split.floor_off), [-0.4, 0.4]))
        # the two libraries' doubles sit at their own floors, so their observed predictions differ by the offsets
        self.assertGreater(float(np.ptp(split.P['comb'])), 0.5)
        self.assertLess(float(np.ptp(flat.P['comb'])), 1e-6)

    def test_a_dump_without_offsets_is_scored_as_before(self):
        h = numpy_link(-0.5, 5.0, 0.3, 0.3)
        it = items_for((0.0, 0.0)); it.pop('dG_floor_offset')
        t = epi_metrics.Table(it, h, heads=('comb',))
        self.assertTrue(np.allclose(t.floor_off, 0.0))
        self.assertTrue(np.allclose(t.P['comb'], epi_metrics.Table(items_for((0.0, 0.0)), h, heads=('comb',)).P['comb']))

    def test_a_one_argument_link_still_works(self):
        """Links built elsewhere (older analysis code) take only z; the Table must not pass an offset to them."""
        one_arg = lambda z: np.asarray(z, float) * 1.0
        t = epi_metrics.Table(items_for((-0.4, 0.4)), one_arg, heads=('comb',))
        self.assertEqual(t.n, 2)
        self.assertTrue(np.isfinite(t.P['comb']).all())


if __name__ == '__main__':
    unittest.main()


class TestTheStepUsesTheOffset(unittest.TestCase):
    """The offset must reach the losses, not only the link object: a double measured at its library's own floor costs nothing when the floor
    offset says so, and the same prediction costs something when it does not."""

    def setUp(self):
        from tests.test_link import make_batch, run                     # noqa: PLC0415
        self.make_batch, self.run = make_batch, run

    def test_a_double_at_its_librarys_floor_is_free_only_with_the_offset(self):
        # dG_wt 2, measured ddG -1.6 (absolute dG +0.4): at the floor of a library whose floor is 0.4, above a shared floor of -0.5
        h = lambda: MonotoneLink(lo=-0.5, hi=5.0, tau_lo=0.15, tau_hi=0.15)
        kw = dict(ddG=[-1.6], feat=[-9.0], n_mut=[2], dG_wt=[2.0], ddG_additive=[-9.0])
        with_off, _ = self.run(self.make_batch(floor_offset=[0.9], **kw), link=h())
        without, _ = self.run(self.make_batch(**kw), link=h())
        self.assertLess(with_off['L_reg_combined'], 0.02)
        self.assertGreater(without['L_reg_combined'], 0.3)

    def test_a_zero_offset_batch_matches_a_batch_without_the_field(self):
        kw = dict(ddG=[-0.5, -1.2, 0.3, -2.0], feat=[-0.4, -1.0, 0.1, -1.5], n_mut=[1, 1, 1, 2], dG_wt=[2.0] * 4, ddG_additive=[float('nan')] * 3 + [-1.4])
        a, _ = self.run(self.make_batch(floor_offset=[0.0] * 4, **kw), link=MonotoneLink(lo=-0.5))
        b = self.make_batch(**kw); b.pop('dG_floor_offset')
        c, _ = self.run(b, link=MonotoneLink(lo=-0.5))
        for k in a:
            if isinstance(a[k], float):
                self.assertAlmostEqual(a[k], c[k], places=6, msg=k)


class TestOffsetLengthMismatch(unittest.TestCase):
    """A field that validation did not pool for every item used to reach the Table as a short array and raise IndexError deep inside it,
    which the metrics' try/except turned into 'epistasis metrics skipped' and no epistasis numbers for the whole run. The Table now ignores
    an offset array that does not match the items, and training raises a named error before writing a dump."""

    def test_a_short_or_empty_offset_array_is_ignored_not_misapplied(self):
        h = numpy_link(-0.5, 5.0, 0.3, 0.3)
        good = epi_metrics.Table(items_for((0.0, 0.0)), h, heads=('comb',))
        for bad in ([], [0.3], [0.3] * 4):
            it = items_for((-0.4, 0.4))
            it['dG_floor_offset'] = np.asarray(bad, float)
            t = epi_metrics.Table(it, h, heads=('comb',))
            self.assertEqual(t.n, good.n, bad)
            self.assertTrue(np.allclose(t.floor_off, 0.0), bad)
            self.assertTrue(np.allclose(t.P['comb'], good.P['comb']), bad)
