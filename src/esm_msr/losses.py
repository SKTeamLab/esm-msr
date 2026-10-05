import torch


class ListMLELoss(torch.nn.Module):
    """
    Implements the Listwise Maximum Likelihood Estimation (ListMLE) loss function,
    with support for dynamic boolean masking to handle irregular list sizes.
    """
    def __init__(self, eps: float = 1e-10, invert: bool = False):
        """
        Initializes the loss module.

        Args:
            eps (float): A small epsilon value to prevent log(0) for numerical stability.
            invert (bool): If True, inverts the predictions.
        """
        super().__init__()
        self.eps = eps
        self.invert = invert

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor, mask: torch.Tensor | None = None,
                score_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculates the masked ListMLE loss.

        ``score_mask`` (opt-in; ``None`` leaves the computation exactly as it always was)
        turns this into a *censored* Plackett-Luce likelihood. Items with
        ``mask & ~score_mask`` are censored: their true order among themselves is unknown (e.g.
        measurements pinned at an assay's dynamic-range floor), but they are known to rank
        below every scored item. They are placed after all scored items, stay in the
        denominators - so each scored item is still pushed above every censored one - and
        contribute no numerator term of their own, so permuting them costs nothing. A list
        with no scored item carries no information and is skipped.

        Args:
            predictions (torch.Tensor): A tensor of scores predicted by the model (Batch, List_Size).
            ground_truths (torch.Tensor): A tensor of ground-truth values (Batch, List_Size).
            mask (torch.Tensor, optional): Boolean tensor where False indicates elements to ignore (Batch, List_Size).

        Returns:
            torch.Tensor: The calculated ListMLE loss as a scalar tensor.
        """
        predictions = predictions.float()
        ground_truths = ground_truths.float()
        
        if self.invert:
            predictions = predictions * -1

        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
        elif mask.shape != predictions.shape:
            raise AssertionError(f"Mask shape {mask.shape} must strictly match predictions shape {predictions.shape}.")

        if score_mask is not None:
            if score_mask.shape != predictions.shape:
                raise AssertionError(f"score_mask shape {score_mask.shape} must strictly match predictions shape {predictions.shape}.")
            if self.invert:
                raise AssertionError("score_mask (censored ListMLE) is not supported together with invert=True.")

        # 1. Push masked ground truths to -infinity so they are sorted to the very end of the list
        gt_masked = ground_truths.clone()
        if score_mask is not None:
            # Censored items sort after every scored item but before the masked-out ones.
            gt_masked[mask & ~score_mask] = torch.finfo(gt_masked.dtype).min
        gt_masked[~mask] = float('-inf')

        # `descending=True` means HIGHER ground_truth values are ranked higher.
        _, indices = gt_masked.sort(descending=not self.invert, dim=-1)

        # 2. Reorder predictions and the mask to match the ground-truth permutation
        ordered_predictions = predictions.gather(-1, indices)
        ordered_mask = mask.gather(-1, indices)

        # 3. --- Numerically Stable Log-Sum-Exp Calculation ---
        # Mask out invalid predictions so they do not artificially inflate the max_prediction
        ordered_preds_safe = ordered_predictions.clone()
        ordered_preds_safe[~ordered_mask] = float('-inf')
        
        max_predictions, _ = ordered_preds_safe.max(dim=-1, keepdim=True)
        
        # Guard against rows where ALL elements are masked (max becomes -inf, causing NaN gradients)
        # Avoid in-place replacement on max_predictions to be safe
        max_predictions = torch.where(
            max_predictions == float('-inf'), 
            torch.zeros_like(max_predictions), 
            max_predictions
        )

        exp_predictions = torch.exp(ordered_predictions - max_predictions)

        # This prevents the "modified by an inplace operation" ExpBackward0 RuntimeError.
        exp_predictions = exp_predictions * ordered_mask.float()

        # Plackett-Luce denominator: cumulative sum from bottom of list to top
        exp_predictions_rev = torch.flip(exp_predictions, dims=(-1,))
        cum_sum_exp_predictions_rev = torch.cumsum(exp_predictions_rev, dim=-1)
        cum_sum_exp_predictions = torch.flip(cum_sum_exp_predictions_rev, dims=(-1,))

        cum_sum_exp_predictions = torch.clamp(cum_sum_exp_predictions, min=self.eps)

        # 4. --- Loss Calculation ---
        log_probs = ordered_predictions - max_predictions - torch.log(cum_sum_exp_predictions)
        
        if score_mask is None:
            log_probs = log_probs * ordered_mask.float()
        else:
            ordered_score = (score_mask & mask).gather(-1, indices)
            log_probs = log_probs * ordered_score.float()

        # Sum the log probabilities per list
        list_loss = -torch.sum(log_probs, dim=-1)
        
        # 5. --- Safe Batch Averaging ---
        # Ranking requires at least 2 valid items to form a meaningful permutation.
        valid_lists = ordered_mask.sum(dim=-1) > 1
        if score_mask is not None:
            valid_lists = valid_lists & (score_mask & mask).any(dim=-1)
        
        if valid_lists.sum() == 0:
            # If the entire micro-batch was masked out, return a 0.0 tensor with attached gradients to prevent crashes
            return (predictions.sum() * 0.0)
            
        return list_loss[valid_lists].mean()


    # ------------------------------------------------------------------ two-sided censoring

    def _per_list(self, predictions: torch.Tensor, ground_truths: torch.Tensor, mask: torch.Tensor,
                  score: torch.Tensor):
        """
        Plackett-Luce NLL of each list, descending order, with a tied block at the bottom.

        ``mask`` marks the list's members; ``score`` (a subset) marks those whose own term counts.
        Members in ``mask & ~score`` sort after every scored member, stay in the denominators and
        contribute no term, so their order among themselves is free. Returns ``(loss [B], valid [B])``;
        a list is valid when it has at least 2 members and at least one scored member.
        """
        predictions, ground_truths = predictions.float(), ground_truths.float()
        gt = ground_truths.clone()
        gt[mask & ~score] = torch.finfo(gt.dtype).min
        gt[~mask] = float('-inf')
        _, idx = gt.sort(descending=True, dim=-1)
        ordered_pred = predictions.gather(-1, idx)
        ordered_mask = mask.gather(-1, idx)
        ordered_score = (score & mask).gather(-1, idx)

        safe = ordered_pred.clone()
        safe[~ordered_mask] = float('-inf')
        max_pred, _ = safe.max(dim=-1, keepdim=True)
        max_pred = torch.where(max_pred == float('-inf'), torch.zeros_like(max_pred), max_pred)
        exp_pred = torch.exp(ordered_pred - max_pred) * ordered_mask.float()
        denom = torch.flip(torch.cumsum(torch.flip(exp_pred, dims=(-1,)), dim=-1), dims=(-1,))
        denom = torch.clamp(denom, min=self.eps)
        log_probs = (ordered_pred - max_pred - torch.log(denom)) * ordered_score.float()
        valid = (mask.sum(dim=-1) > 1) & (score & mask).any(dim=-1)
        return -log_probs.sum(dim=-1), valid

    def forward_censored(self, predictions: torch.Tensor, ground_truths: torch.Tensor,
                         mask: torch.Tensor, cens: torch.Tensor) -> torch.Tensor:
        """
        ListMLE with two-sided censoring. ``cens`` is -1, 0 or +1 per member:

        * ``-1`` lower-censored (true value at or below its bound): ranks below every uncensored
          member; their order among themselves is not scored.
        * ``+1`` upper-censored (true value at or above its bound): ranks above every uncensored member.
        * ``0`` ordinary: ordered by ``ground_truths``.

        A tied block at the bottom is exact in forward Plackett-Luce (drop the block's own terms, keep it in
        the denominators); a tied block at the top is exact in *reverse* Plackett-Luce (worst first, via negated
        scores). A list with only lower-censored (or no censored) members uses the forward form, a list
        with only upper-censored members the reverse form, and a list with both averages the two, each
        pass leaving out the other pass's censored members. With no censored member this equals ``forward``.
        """
        if self.invert:
            raise AssertionError("forward_censored is not supported together with invert=True.")
        if mask.shape != predictions.shape or cens.shape != predictions.shape:
            raise AssertionError("mask and cens must match predictions' shape.")
        lo, hi = (cens < 0) & mask, (cens > 0) & mask
        ordinary = mask & ~lo & ~hi
        lf, vf = self._per_list(predictions, ground_truths, mask & ~hi, ordinary)
        lr, vr = self._per_list(-predictions, -ground_truths, mask & ~lo, ordinary)
        has_lo, has_hi = lo.any(dim=-1), hi.any(dim=-1)
        both = has_lo & has_hi
        use_rev_only = has_hi & ~has_lo
        loss = torch.where(use_rev_only, lr, lf)
        valid = torch.where(use_rev_only, vr, vf)
        loss = torch.where(both & vf & vr, 0.5 * (lf + lr), loss)
        loss = torch.where(both & ~vf & vr, lr, loss)
        valid = torch.where(both, vf | vr, valid)
        if valid.sum() == 0:
            return predictions.sum() * 0.0
        return loss[valid].mean()
