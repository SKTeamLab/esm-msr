"""Ranking and regression objectives, both aware of the assay's floor.

A measurement at the assay's floor says "this variant is at most this stable" and nothing more: its value is the floor's, not the variant's,
and its order among its equally-dead peers is noise. Both objectives here are built to use exactly that much of such an item and no more:

* :meth:`ListMLELoss.forward_censored` ties the censored members of a list at the bottom (or top) - the bottom block stays in the
  denominators, the top block is scored as "placed first in any order" - so the model is still taught where they rank against the
  measurable variants, while never scoring their order among themselves. Always forward (top-weighted) Plackett-Luce.
* :func:`hinge_regression_loss` penalises a censored item only when the prediction falls on the wrong side of its bound, so predicting "at or
  below the floor" for something measured at the floor is free, and only claiming it is *better* than the floor costs anything.

``cens`` is -1 (lower-censored), 0 (ordinary) or +1 (upper-censored) throughout; see ``esm_msr.censoring`` for where it comes from.
"""
import itertools

import torch


class ListMLELoss(torch.nn.Module):
    """
    ListMLE (Plackett-Luce listwise NLL) with boolean masking for irregular list sizes.
    """
    def __init__(self, eps: float = 1e-10, invert: bool = False):
        """
        Args:
            eps: floor on the Plackett-Luce denominators, for numerical stability.
            invert: rank ascending instead of descending.
        """
        super().__init__()
        self.eps = eps
        self.invert = invert

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            predictions: scores, (Batch, List_Size).
            ground_truths: targets, (Batch, List_Size); higher ranks higher.
            mask: False marks entries to ignore, (Batch, List_Size).
        """
        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
        elif mask.shape != predictions.shape:
            raise AssertionError(f"Mask shape {mask.shape} must strictly match predictions shape {predictions.shape}.")
        if self.invert:
            predictions, ground_truths = -predictions, -ground_truths
        loss, valid = self._per_list(predictions, ground_truths, mask, mask)
        if valid.sum() == 0:
            # Whole micro-batch masked out: keep a 0.0 with gradients attached so backward does not crash.
            return predictions.sum() * 0.0
        return loss[valid].mean()

    def _per_list(self, predictions: torch.Tensor, ground_truths: torch.Tensor, mask: torch.Tensor,
                  score: torch.Tensor):
        """
        Plackett-Luce NLL of each list, descending order, with a tied block at the bottom.

        ``mask`` marks the list's members; ``score`` (a subset) marks those whose own term counts. Members in
        ``mask & ~score`` sort after every scored member, stay in the denominators and contribute no term of
        their own, so their order among themselves is free. Returns ``(loss [B], valid [B])``; a list is valid
        when it has at least 2 members and at least one scored member.
        """
        predictions, ground_truths = predictions.float(), ground_truths.float()
        gt = ground_truths.clone()
        gt[mask & ~score] = torch.finfo(gt.dtype).min
        gt[~mask] = float('-inf')
        _, idx = gt.sort(descending=True, dim=-1)
        ordered_pred = predictions.gather(-1, idx)
        ordered_mask = mask.gather(-1, idx)
        ordered_score = (score & mask).gather(-1, idx)

        # Log-sum-exp over the live members only, so masked entries cannot inflate the max.
        safe = ordered_pred.clone()
        safe[~ordered_mask] = float('-inf')
        max_pred, _ = safe.max(dim=-1, keepdim=True)
        max_pred = torch.where(max_pred == float('-inf'), torch.zeros_like(max_pred), max_pred)
        # Zero the masked terms by sending them to -inf BEFORE exp, not by multiplying after: a masked entry's
        # score is arbitrary, and exp() of it can overflow to inf, which 0 * inf would turn into NaN.
        shifted = torch.where(ordered_mask, ordered_pred - max_pred, torch.full_like(ordered_pred, float('-inf')))
        exp_pred = torch.exp(shifted)
        denom = torch.flip(torch.cumsum(torch.flip(exp_pred, dims=(-1,)), dim=-1), dims=(-1,))
        denom = torch.clamp(denom, min=self.eps)
        score_w = ordered_score.float()
        log_probs = (ordered_pred - max_pred - torch.log(denom)) * score_w
        valid = (mask.sum(dim=-1) > 1) & (score & mask).any(dim=-1)
        return -log_probs.sum(dim=-1), valid

    def forward_censored(self, predictions: torch.Tensor, ground_truths: torch.Tensor,
                         mask: torch.Tensor, cens: torch.Tensor) -> torch.Tensor:
        """
        ListMLE with the censored members' mutual order left unscored. ``cens`` is -1, 0 or +1 per member:

        * ``-1`` lower-censored (true value at or below its bound): ranks below every uncensored member; the order
          within the block is not scored, because the assay cannot resolve it.
        * ``+1`` upper-censored (at or above its bound): ranks above every uncensored member, likewise unresolved.
        * ``0`` ordinary: ordered by ``ground_truths``.

        Always forward (top-weighted) Plackett-Luce. Under PL the probability of "upper block first in any order, then
        the rest in this order" factorises exactly into P(upper block before everyone else) x PL(rest), so a list is
        scored as the forward NLL of its ordinary members (the lower block tied at the bottom: no terms of its own,
        kept in the denominators) plus ``-log P(upper block first)`` (:meth:`_top_block_nll`). Until 2026-10-20 a list
        holding any upper-censored member switched to REVERSE Plackett-Luce (and a list with both kinds averaged the
        two), which is exact for the block but bottom-weights every other member of that list. With no censored
        member this equals :meth:`forward`.
        """
        if self.invert:
            raise AssertionError("forward_censored is not supported together with invert=True.")
        if mask.shape != predictions.shape or cens.shape != predictions.shape:
            raise AssertionError("mask and cens must match predictions' shape.")
        lo, hi = (cens < 0) & mask, (cens > 0) & mask
        ordinary = mask & ~lo & ~hi
        loss, valid = self._per_list(predictions, ground_truths, mask & ~hi, ordinary)
        has_hi = hi.any(dim=-1)
        if bool(has_hi.any()):
            loss = loss + self._top_block_nll(predictions, mask, hi)
            # a list whose upper block has anything below it is informative even with no ordinary pair
            valid = valid | (has_hi & (mask & ~hi).any(dim=-1))
        if valid.sum() == 0:
            return predictions.sum() * 0.0
        return loss[valid].mean()

    MAX_EXACT_BLOCK = 4     # 4! = 24 orders; a larger upper block is scored in its current-score order

    def _top_block_nll(self, predictions: torch.Tensor, mask: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """
        ``-log P(the upper-censored block occupies the first places, in any order)`` per list, 0 where there is none.

        ``P = sum over the block's orders sigma of prod_k w_sigma(k) / (remaining block weight + W_rest)``, with
        ``w = exp(score)`` and ``W_rest`` the weight of every other member, ordinary and lower-censored. Summed exactly in
        log-space for blocks of up to ``MAX_EXACT_BLOCK``; a larger block uses the single order by current score, a lower
        bound on P (so an upper bound on the loss). Only lists that hold the block are looped over - they are rare.
        """
        pf = predictions.float()            # the same precision _per_list scores in, so the two terms add cleanly
        rows = []
        for b in range(pf.shape[0]):
            if not bool(hi[b].any()):
                rows.append(pf.new_zeros(()))
                continue
            s = pf[b]
            top, rest = s[hi[b]], s[mask[b] & ~hi[b]]
            if rest.numel() == 0:
                rows.append(pf.new_zeros(()))
                continue
            log_rest = torch.logsumexp(rest, dim=0).reshape(1)
            m = top.numel()
            orders = (itertools.permutations(range(m)) if m <= self.MAX_EXACT_BLOCK
                      else [tuple(torch.argsort(top.detach(), descending=True).tolist())])
            terms = []
            for order in orders:
                order = list(order)
                lp = top.new_zeros(())
                for k in range(m):
                    lp = lp + top[order[k]] - torch.logsumexp(torch.cat([top[order[k:]], log_rest]), dim=0)
                terms.append(lp)
            rows.append(-torch.logsumexp(torch.stack(terms), dim=0))
        return torch.stack(rows)


def hinge_regression_loss(crit, pred: torch.Tensor, target: torch.Tensor, bound: torch.Tensor,
                          cens: torch.Tensor) -> torch.Tensor:
    """
    Regression that treats a censored measurement as a bound, element-wise (same reduction-'none' convention as ``crit``).

    An uncensored item (``cens == 0``) is scored against ``target`` as usual. A lower-censored item - one the assay
    could only pin at its floor - is scored against ``bound`` and *only when the prediction sits above it*: saying
    "at or below the floor" about something measured at the floor is free, and only claiming it is better than the
    floor is penalised. An upper-censored item is the mirror image. No value is ever imposed that was not measured.

    ``bound`` is on the item's own ddG scale (``cens_bound``), so one number serves singles, doubles and conditional
    items alike; where it is NaN the item contributes nothing rather than poisoning the sum.
    """
    censored = cens != 0
    finite_bound = torch.isfinite(bound)
    # A censored item with no computable bound is dropped by `keep`; zero it here too, because a NaN target would
    # survive the multiply (0 * NaN = NaN) and poison the whole sum.
    safe_bound = torch.where(finite_bound, bound, torch.zeros_like(bound))
    tgt = torch.where(censored, safe_bound, target)
    L = crit(pred, tgt)
    wrong_side = ((cens < 0) & (pred > safe_bound)) | ((cens > 0) & (pred < safe_bound))
    keep = (~censored) | (wrong_side & finite_bound)
    return L * keep.to(L.dtype)
