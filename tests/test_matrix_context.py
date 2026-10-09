"""--pack_pair_matrices: the sampler keeps every oriented pair matrix in one batch, and the whole-matrix losses (components, confident
reversals) streamed over several micro-batches through the prediction cache give exactly the gradient of the same losses computed in one
micro-batch that holds the whole batch."""
import types
import unittest

import numpy as np
import torch

try:
    from tests.test_compose_censoring import make_hp
    from tests.test_sampler import DummyDataset
except ImportError:  # run from inside tests/
    from test_compose_censoring import make_hp
    from test_sampler import DummyDataset

from esm_msr.data import ProteinCyclingBatchSampler
from esm_msr.link import MonotoneLink
from esm_msr.losses import ListMLELoss


class PerItemModel(torch.nn.Module):
    """pred = feature * scale + theta[item]: one free parameter per batch row, so any wrong per-row gradient shows."""
    dedup_backbone = False

    def __init__(self, n):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.theta = torch.nn.Parameter(torch.zeros(n))

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        i = 0 if pass_type == 'wt' else 1
        raw = micro['feat'] * self.w[i] + (self.theta[micro['row']] if i == 1 else 0.0)
        return {'pred_calibrated': raw, 'pred_raw': raw}


def matrix_batch(n_rows=6, partners='ACDE', n_singles=5, seed=0):
    """One oriented matrix of cond items (rows: scored substitutions, columns: partner residues) with interaction and reversals, plus singles."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 1.0, n_rows), rng.normal(0, 0.5, len(partners))
    eps = rng.normal(0, 1.0, (n_rows, len(partners)))
    keys, rid, t, feat = [], [], [], []
    for i in range(n_rows):
        for j, c in enumerate(partners):
            keys.append(f'LIB|5|7{c}'); rid.append(i); t.append(a[i] + b[j] + eps[i, j]); feat.append(a[i] + b[j] + 0.3 * rng.normal())
    for k in range(n_singles):
        keys.append(''); rid.append(100 + k); t.append(rng.normal()); feat.append(rng.normal())
    B = len(t)
    st = ['cond'] * (n_rows * len(partners)) + ['single'] * n_singles
    return {
        'ddG': torch.tensor(t, dtype=torch.float32), 'feat': torch.tensor(feat, dtype=torch.float32), 'row': torch.arange(B),
        'mut_mask': torch.ones(B, 1, dtype=torch.bool), 'subset_type': st, 'flip_key': keys, 'mt_id': torch.tensor(rid).view(-1, 1),
        'reg_ok': torch.ones(B, dtype=torch.bool), 'cens': torch.zeros(B, dtype=torch.long), 'cens_bound': torch.full((B,), float('nan')),
        'cens_src': torch.zeros(B, dtype=torch.long), 'dG_wt': torch.full((B,), 2.0), 'bg_offset': torch.zeros(B),
    }


def run(batch, link=None, **hp_kw):
    from esm_msr import training
    cls = training.ESM3EpistasisLightningModule
    stub = types.SimpleNamespace()
    stub.hparams = make_hp(**{'flip_delta': 0.6, 'flip_scale': 0.25, 'mt_comp_offset': 10.0, 'mt_comp_subst': 10.0, 'mt_comp_int': 10.0,
                              'lambda_mt_flip': 1.0, 'flip_list_min': 3, **hp_kw})
    stub.model = PerItemModel(len(batch['ddG']))
    stub.link_head = link
    stub.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
    stub.crit_reg = torch.nn.MSELoss(reduction='none')
    stub.crit_rank_wt, stub.crit_rank_mt = ListMLELoss(), ListMLELoss()
    stub._warned_unrouted = True
    stub.global_step = 0
    stub.manual_backward = lambda loss, retain_graph=False: loss.backward(retain_graph=retain_graph)
    for name in ('_compute_rank_loss', '_compute_flip_loss', '_compute_block_components', '_aligned_chunks', '_subset_weights', '_plan_units',
                 '_matrix_context'):
        setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    stub._reversal_tetrads = cls._reversal_tetrads
    out = cls._compose_losses_streaming_and_backward(stub, batch)
    return out, stub


class TestStreamedWholeMatrixGradient(unittest.TestCase):
    def _compare(self, link=None):
        batch = matrix_batch()
        one, s1 = run(batch, link=link, pack_pair_matrices=True, micro_batch_size=64, mt_single_anchor_weight=0.0)
        many, s2 = run(batch, link=link, pack_pair_matrices=True, micro_batch_size=6, mt_single_anchor_weight=0.0)
        self.assertGreater(s1._rev_diag[0], 0)                          # the matrix has confident reversals
        self.assertEqual(s1._rev_diag[0], s2._rev_diag[0])
        for p1, p2 in ((s1.model.theta, s2.model.theta), (s1.model.w, s2.model.w)):
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-4)
        self.assertGreater(float(s1.model.theta.grad.abs().sum()), 0.0)
        for k in ('L_comp_off', 'L_comp_subst', 'L_comp_int', 'L_flip_mt', 'L_reg_mt'):      # logged once per batch, whatever the cut
            self.assertAlmostEqual(one[k], many[k], places=5, msg=k)

    def test_identical_to_one_micro_batch(self):
        self._compare()

    def test_identical_to_one_micro_batch_through_the_link(self):
        self._compare(link=MonotoneLink())

    def test_one_micro_batch_matches_the_unpacked_path(self):
        batch = matrix_batch()
        packed, s1 = run(batch, pack_pair_matrices=True, micro_batch_size=64, mt_single_anchor_weight=0.0)
        plain, s2 = run(batch, pack_pair_matrices=False, micro_batch_size=64, mt_single_anchor_weight=0.0)
        torch.testing.assert_close(s1.model.theta.grad, s2.model.theta.grad, atol=1e-5, rtol=1e-4)
        self.assertAlmostEqual(packed['L_comp_int'], plain['L_comp_int'], places=5)

    def test_without_packing_a_cut_matrix_loses_its_whole_matrix_gradient(self):
        batch = matrix_batch()
        _, s1 = run(batch, pack_pair_matrices=False, micro_batch_size=64, mt_single_anchor_weight=0.0)
        _, s2 = run(batch, pack_pair_matrices=False, micro_batch_size=6, mt_single_anchor_weight=0.0)
        self.assertFalse(torch.allclose(s1.model.theta.grad, s2.model.theta.grad, atol=1e-4))


class TestSamplerPacking(unittest.TestCase):
    def test_every_oriented_matrix_lands_whole_in_one_batch(self):
        items = []
        for pair in ('5|7', '5|9', '7|5', '12|30'):                     # four oriented matrices of 19 x 19
            for c in range(19):
                for r in range(19):
                    items.append({'flip_key': f'P|{pair}{chr(65 + c)}', 'subset_type': 'cond', 'r': r})
        items += [{'flip_key': '', 'subset_type': 'single'} for _ in range(300)]
        ds = DummyDataset('1ABC', items)
        sampler = ProteinCyclingBatchSampler(datasets=[ds], batch_size=400, train_list=['1ABC'], rng_seed=1, flip_pair_groups=19)
        seen = {}
        for bi, b in enumerate(sampler):
            for i in b:
                fk = ds[i]['flip_key']
                if fk:
                    seen.setdefault(fk.rsplit('|', 1)[0] + '|' + fk.rsplit('|', 1)[1][:-1], set()).add(bi)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(len(v) == 1 for v in seen.values()), seen)


if __name__ == '__main__':
    unittest.main()


class LinearModel(torch.nn.Module):
    """pred = Linear(features): the weight is an fp32 leaf that autocast casts (and caches) like the LoRA matrices."""
    dedup_backbone = False

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)

    def forward_partitioned(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        x = torch.stack([micro['feat'], micro['row'].float() / 100.0], dim=1)
        raw = self.lin(x).float().squeeze(1)
        return {'pred_calibrated': raw, 'pred_raw': raw}


class TestAutocastCache(unittest.TestCase):
    def test_the_cache_pass_does_not_cut_the_gradient_under_autocast(self):
        """Regression: the no-grad cache pass under bf16 autocast used to leave cached weight casts without autograd history, so the
        live forwards of the step gave the adapter no gradient (found by the grad-share probe on the GPU)."""
        batch = matrix_batch()
        from esm_msr import training
        cls = training.ESM3EpistasisLightningModule
        stub = types.SimpleNamespace()
        stub.hparams = make_hp(flip_delta=0.6, flip_scale=0.25, mt_comp_offset=10.0, mt_comp_subst=10.0, mt_comp_int=10.0, lambda_mt_flip=1.0,
                               flip_list_min=3, pack_pair_matrices=True, micro_batch_size=6, mt_single_anchor_weight=0.0)
        stub.model = LinearModel()
        stub.link_head = None
        stub.peft_manager = types.SimpleNamespace(wt_path_is_frozen=False, mt_path_is_frozen=False)
        stub.crit_reg = torch.nn.MSELoss(reduction='none')
        stub.crit_rank_wt, stub.crit_rank_mt = ListMLELoss(), ListMLELoss()
        stub._warned_unrouted = True
        stub.global_step = 0
        seen = []
        def backward(loss, retain_graph=False):
            seen.append(bool(loss.requires_grad))
            loss.backward(retain_graph=retain_graph)
        stub.manual_backward = backward
        for name in ('_compute_rank_loss', '_compute_flip_loss', '_compute_block_components', '_aligned_chunks', '_subset_weights', '_plan_units',
                     '_matrix_context'):
            setattr(stub, name, types.MethodType(getattr(cls, name), stub))
        stub._reversal_tetrads = cls._reversal_tetrads
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cls._compose_losses_streaming_and_backward(stub, batch)
        self.assertTrue(seen and all(seen))
        self.assertIsNotNone(stub.model.lin.weight.grad)
        self.assertGreater(float(stub.model.lin.weight.grad.abs().sum()), 0.0)
