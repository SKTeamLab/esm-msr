"""Ranking and regression objectives, both aware of the assay's floor.

A measurement at the assay's floor says "this variant is at most this stable" and nothing more: its value is the floor's, not the variant's,
and its order among its equally-dead peers is noise. Both objectives here are built to use exactly that much of such an item and no more:

* :meth:`ListMLELoss.forward_censored` ties the censored members of a list at the bottom (or top) - the bottom block stays in the
  denominators, the top block is scored as "placed first in any order" - so the model is still taught where they rank against the
  measurable variants, while never scoring their order among themselves. Always forward (top-weighted) Plackett-Luce.
* :func:`tobit_regression_loss` is the same idea without a threshold: the exact likelihood of a reading that is the larger of the variant's own
  stability and a noisy per-library unfolded floor, so a near-floor reading bounds the latent from above smoothly instead of pinning it.
* :func:`hinge_regression_loss` penalises a censored item only when the prediction falls on the wrong side of its bound, so predicting "at or
  below the floor" for something measured at the floor is free, and only claiming it is *better* than the floor costs anything.

``cens`` is -1 (lower-censored), 0 (ordinary) or +1 (upper-censored) throughout; see ``esm_msr.censoring`` for where it comes from.
"""
import itertools
import math

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


TOBIT_SIGMA_Y, TOBIT_MU_F, TOBIT_SIGMA_F = 0.28, 0.05, 0.38     # kcal/mol: assay noise, mean and spread of the unfolded floor
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
TF_PRIOR_SD = 0.75                                              # kcal/mol: the width of the prior on a single's latent dG around the WT head's read (--tf_label posterior)


def tobit_regression_loss(z: torch.Tensor, y: torch.Tensor, bound: torch.Tensor, cens: torch.Tensor,
                          sigma_y: float = TOBIT_SIGMA_Y, mu_f: float = TOBIT_MU_F, sigma_f: float = TOBIT_SIGMA_F) -> torch.Tensor:
    """
    Noisy-floor Tobit negative log-likelihood, element-wise, on ABSOLUTE dG (``z`` the latent dG the model claims, ``y`` the reading).

    A reading is ``Y = max(X, F)`` with ``X ~ N(z, sigma_y^2)`` (the variant's own stability, measured with noise) and ``F ~ N(mu_f, sigma_f^2)``
    the unfolded floor of that library and protease (the composition model's error for the sequence). Then

        p(y | z) = phi_y(y - z) * Phi_f(y - mu_f)  +  phi_f(y - mu_f) * Phi_y(y - z)

    The first term is "the variant was measured"; the second is "the floor was read, and the variant sits below it". Far above the floor
    ``Phi_f -> 1`` and ``phi_f -> 0`` and this is a Gaussian, i.e. plain MSE. Near and below it the second term turns the loss into a smooth upper bound
    on z: a dead double at z = -3 read at +0.05 costs almost nothing, a false-stable z = +1.5 is expensive, and no reading below the floor
    has to be inverted or censored by a threshold. One item's minimiser is its maximum-likelihood point (a reading of 0 gives z = -0.18, barely better than z = -30); E[z | y] needs a
    prior as well, see :func:`tobit_posterior_mean`. Below z* = y - (sigma_y / sigma_f) (y - mu_f) the floor branch owns the item and the loss is flat, at any reading.

    ``cens`` as everywhere: 0 an ordinary reading scored with the mixture; -1 / +1 a bound (``bound`` absolute dG, e.g. the assay's '<-1' / '>5' tags), scored
    with the survival term ``-log Phi((b - z) / sigma_y)`` / ``-log Phi((z - b) / sigma_y)`` (the floor factor does not depend on z). A censored
    item whose bound is not finite contributes nothing.

    Scaled by ``2 * sigma_y^2`` and shifted by the Gaussian constant so that, away from the floor, it equals ``(y - z)^2`` exactly: the same magnitude as the
    MSE it replaces, so the loss weights tuned for MSE carry over.
    """
    z, y, bound = z.float(), y.float(), bound.float()
    scale = 2.0 * sigma_y ** 2
    lo, hi = cens < 0, cens > 0
    num = (cens == 0) & torch.isfinite(y)
    y_s = torch.where(num, y, torch.zeros_like(y))
    t1 = -0.5 * ((y_s - z) / sigma_y) ** 2 - math.log(sigma_y) - _LOG_SQRT_2PI + torch.special.log_ndtr((y_s - mu_f) / sigma_f)
    t2 = -0.5 * ((y_s - mu_f) / sigma_f) ** 2 - math.log(sigma_f) - _LOG_SQRT_2PI + torch.special.log_ndtr((y_s - z) / sigma_y)
    L_num = scale * (-torch.logaddexp(t1, t2) - (math.log(sigma_y) + _LOG_SQRT_2PI))
    b_ok = torch.isfinite(bound)
    b_s = torch.where(b_ok, bound, torch.zeros_like(bound))
    L_lo = -scale * torch.special.log_ndtr((b_s - z) / sigma_y)
    L_hi = -scale * torch.special.log_ndtr((z - b_s) / sigma_y)
    zero = torch.zeros_like(L_num)
    return torch.where(num, L_num, zero) + torch.where(lo & b_ok, L_lo, zero) + torch.where(hi & b_ok, L_hi, zero)


@torch.no_grad()
def tobit_posterior_mean(y: torch.Tensor, m: torch.Tensor, cens: torch.Tensor, s: float = TF_PRIOR_SD,
                         sigma_y: float = TOBIT_SIGMA_Y, mu_f: float = TOBIT_MU_F, sigma_f: float = TOBIT_SIGMA_F) -> torch.Tensor:
    """
    E[z | reading] on ABSOLUTE dG, element-wise: the prior ``z ~ N(m, s^2)`` updated by the likelihood of :func:`tobit_regression_loss`, in closed form. Nothing is sampled.

    The likelihood has two branches, so the posterior is a mixture of two pieces with known means, with r^2 = s^2 + sigma_y^2 and a = (y - m) / r:

        measured:  the prior times a Gaussian, mean (m sigma_y^2 + y s^2) / r^2,                  weight phi(a) / r * Phi_f(y - mu_f)
        floor:     the prior times Phi((y - z) / sigma_y), mean m - (s^2 / r) lambda(a),          weight phi_f(y - mu_f) * Phi(a)

    with lambda = phi / Phi. So a reading well above the floor is averaged with the prior (88% reading at s = 0.75), and a reading at the floor keeps the part of the prior below it: a
    deep prior stays deep, an optimistic one is cut down to the floor. ``cens`` -1 / +1 (a '<-1' / '>5' tag, with ``y`` its bound) gives the survival term's truncated mean, and a NaN ``y``
    (no reading) returns the prior. Continuous in y, with no threshold, inverse or NaN to handle.
    """
    y_s = torch.where(torch.isfinite(y), y, m)
    r = math.sqrt(s ** 2 + sigma_y ** 2)
    a = (y_s - m) / r
    lam = lambda t: torch.exp(-0.5 * t ** 2 - _LOG_SQRT_2PI - torch.special.log_ndtr(t))
    f = (y_s - mu_f) / sigma_f
    log_w1 = -0.5 * a ** 2 - math.log(r) + torch.special.log_ndtr(f)
    log_w2 = -0.5 * f ** 2 - math.log(sigma_f) + torch.special.log_ndtr(a)
    w1 = torch.exp(log_w1 - torch.logaddexp(log_w1, log_w2))
    z = w1 * (m * sigma_y ** 2 + y_s * s ** 2) / r ** 2 + (1.0 - w1) * (m - s ** 2 / r * lam(a))
    z = torch.where(cens < 0, m - s ** 2 / r * lam(a), z)
    z = torch.where(cens > 0, m + s ** 2 / r * lam(-a), z)
    return torch.where(torch.isfinite(y), z, m)
