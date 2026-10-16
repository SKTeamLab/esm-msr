"""
``--micro_batch_size`` must change speed and memory only: the rank loss a batch contributes, and its gradient, must
not depend on how the batch is chopped into micro-batches.

``_compute_rank_loss`` scores a block of ``B/list_size`` lists and the caller weights the result by
``n_list / global_num_lists``, so the batch's rank term is a sum over micro-batches. For that sum to be
chop-independent, each list has to be normalised by its OWN valid length and an invalid list has to contribute
zero while still counting. Two things can break it:

  * scaling the micro-batch's mean list loss by ``list_size / mean(valid length)`` makes a list's weight depend on
    its neighbours in the chop;
  * ``forward_censored`` averages over VALID lists while ``n_list`` counts ALL of them, so a micro-batch holding
    one valid list and three invalid ones hands that one list four times its share.

Both matter here: training uses ``--micro_batch_size 64`` with ``--subset_size 16``, i.e. four lists per
micro-batch, and 27% of training doubles have no additive label, so WT lists really do carry masked members.
"""
import types

import numpy as np
import pytest
import torch

from esm_msr import training
from esm_msr.losses import ListMLELoss

LIST_SIZE = 16


def make_block(n_lists=4, seed=0, censored=True, starve_last=True):
    """``(pred, targets, mask, cens)`` for ``n_lists`` lists: masked members, censored members, and one starved list."""
    rng = np.random.default_rng(seed)
    n = n_lists * LIST_SIZE
    targets = torch.tensor(rng.normal(-1.0, 1.2, n), dtype=torch.float32)
    pred = torch.tensor(rng.normal(-1.0, 1.0, n), dtype=torch.float32, requires_grad=True)
    mask = torch.tensor(rng.random(n) > 0.27, dtype=torch.bool)          # the masked-double rate
    cens = torch.zeros(n, dtype=torch.long)
    if censored:
        cens[torch.tensor(rng.random(n) < 0.10)] = -1                    # floor / gap censoring
        cens[torch.tensor(rng.random(n) < 0.03)] = +1                    # the '>5' rows
    if starve_last:
        mask[-LIST_SIZE:] = False
        mask[-1] = True                                                  # a list with a single valid member: invalid, counts as 0
    return pred, targets, mask, cens


def batch_rank_term(pred, targets, mask, cens, micro):
    """The batch's rank term, assembled exactly as _compose_losses_streaming_and_backward assembles it."""
    B = pred.shape[0]
    global_num_lists = max(1, B // LIST_SIZE)
    mb = (micro // LIST_SIZE) * LIST_SIZE                                # the same rounding _micro_batch applies
    total, n_seen = 0.0, 0
    for start in range(0, B, mb):
        sl = slice(start, min(start + mb, B))
        L, _, n_list = training.ESM3EpistasisLightningModule._compute_rank_loss(
            None, pred[sl], targets[sl], mask[sl], cens[sl], LIST_SIZE, ListMLELoss())
        if L is not None:
            total = total + L * (n_list / global_num_lists)
            n_seen += n_list
    assert n_seen == global_num_lists, f"{n_seen} lists scored, expected {global_num_lists}"
    return total


def grad_of(micro, **kw):
    pred, targets, mask, cens = make_block(**kw)
    total = batch_rank_term(pred, targets, mask, cens, micro)
    total.backward()
    return float(total.detach()), pred.grad.clone()


@pytest.mark.parametrize('micro', [32, 64])
def test_the_batch_rank_term_does_not_depend_on_the_chop(micro):
    """One list per micro-batch is the reference: it is the only chop with no cross-list averaging to get wrong."""
    ref_loss, ref_grad = grad_of(16)
    loss, grad = grad_of(micro)
    assert ref_grad.abs().sum() > 0.0, "degenerate block: nothing to compare"
    assert loss == pytest.approx(ref_loss, abs=1e-5, rel=1e-5), f"micro={micro}: {loss} vs {ref_loss}"
    torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('micro', [32, 64])
def test_chop_invariance_holds_without_censoring_too(micro):
    ref_loss, ref_grad = grad_of(16, censored=False)
    loss, grad = grad_of(micro, censored=False)
    assert loss == pytest.approx(ref_loss, abs=1e-5, rel=1e-5)
    torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('micro', [32, 64])
def test_a_starved_list_contributes_zero_and_still_counts(micro):
    """A list with fewer than two valid members cannot be ranked. It must weigh nothing, not redistribute weight."""
    pred, targets, mask, cens = make_block(starve_last=True)
    with_starved = batch_rank_term(pred, targets, mask, cens, micro)
    mask3 = mask[:-LIST_SIZE]
    three = batch_rank_term(pred[:-LIST_SIZE], targets[:-LIST_SIZE], mask3, cens[:-LIST_SIZE], micro)
    # 4 lists, one of them starved, vs the same 3 real lists alone: the starved list adds nothing to the numerator,
    # so the two differ only by global_num_lists (4 vs 3).
    assert float(with_starved) == pytest.approx(float(three) * 3.0 / 4.0, abs=1e-5, rel=1e-5)


def test_each_list_is_normalised_by_its_own_length():
    """Two lists, one full and one half-masked: the half-masked one must be scaled by its own length, not the mean."""
    pred, targets, mask, cens = make_block(n_lists=2, seed=7, censored=False, starve_last=False)
    mask[:] = True
    mask[LIST_SIZE + LIST_SIZE // 2:] = False                            # list 0 full, list 1 half
    crit = ListMLELoss()
    view = lambda t: t.view(-1, LIST_SIZE)
    per_list = [crit.forward_censored(view(pred)[i:i + 1], view(targets)[i:i + 1], view(mask)[i:i + 1], view(cens)[i:i + 1])
                for i in range(2)]
    n_valid = view(mask).float().sum(dim=-1)
    expected = sum(float(L) * (LIST_SIZE / float(k)) for L, k in zip(per_list, n_valid)) / 2.0
    L, _, n_list = training.ESM3EpistasisLightningModule._compute_rank_loss(
        None, pred, targets, mask, cens, LIST_SIZE, crit)
    assert n_list == 2
    assert float(L) == pytest.approx(expected, abs=1e-6, rel=1e-6)
