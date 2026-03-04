import torch
from typing import Optional, Literal


class ListMLELoss(torch.nn.Module):
    """
    Implements the Listwise Maximum Likelihood Estimation (ListMLE) loss function.

    This loss is designed for learning-to-rank tasks. It evaluates the entire
    list of items at once and aims to maximize the probability of the
    ground-truth permutation. This version contains the corrected denominator calculation.
    """
    def __init__(self, eps: float = 1e-10, invert=False):
        """
        Initializes the loss module.

        Args:
            eps (float): A small epsilon value to prevent log(0) for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.invert = invert

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the ListMLE loss.

        Args:
            predictions (torch.Tensor): A tensor of scores predicted by the model for a list of items.
            ground_truths (torch.Tensor): A tensor of ground-truth values that define the correct
                                          ranking. For this implementation, a higher value in this
                                          tensor is considered a better rank.

        Returns:
            torch.Tensor: The calculated ListMLE loss as a scalar tensor.
        """
        # Ensure inputs are float tensors for calculation
        predictions = predictions.float()
        if self.invert:
            predictions = predictions * -1

        ground_truths = ground_truths.float()

        # Get the correct permutation indices by sorting the ground-truth values.
        # `descending=True` means HIGHER ground_truth values are ranked higher.
        # This now matches the expected "higher is better" logic.
        _, indices = ground_truths.sort(descending=not self.invert, dim=-1)

        # Reorder the model's predictions to match the ground-truth permutation
        ordered_predictions = predictions.gather(-1, indices)

        # --- Numerically Stable Log-Sum-Exp Calculation ---

        # Subtract the maximum prediction before exponentiating to prevent overflow
        max_predictions, _ = ordered_predictions.max(dim=-1, keepdim=True)
        exp_predictions = torch.exp(ordered_predictions - max_predictions)

        # The Plackett-Luce model sums over the remaining items. To achieve this,
        # we flip the tensor, compute the standard cumulative sum, and flip it back.
        exp_predictions_rev = torch.flip(exp_predictions, dims=(-1,))
        cum_sum_exp_predictions_rev = torch.cumsum(exp_predictions_rev, dim=-1)
        cum_sum_exp_predictions = torch.flip(cum_sum_exp_predictions_rev, dims=(-1,))

        # Add epsilon to prevent taking the log of zero, which would result in -inf
        cum_sum_exp_predictions = torch.clamp(cum_sum_exp_predictions, min=self.eps)

        # --- Loss Calculation ---

        # Compute the log probability for each item in the ordered list.
        # This corresponds to the inner term of the summation in the ListMLE formula.
        log_probs = ordered_predictions - max_predictions - torch.log(cum_sum_exp_predictions)

        # The final loss is the negative sum of these log probabilities (negative log-likelihood),
        # averaged over the batch.
        loss = -torch.sum(log_probs, dim=-1).mean()

        return loss


class AsymmetricHuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0, penalty_factor=3.0, reduction='none'):
        super().__init__()
        self.delta = delta
        self.penalty_factor = penalty_factor # Penalize "dangerous" errors 3x more
        self.reduction = reduction

    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        
        # Standard Huber Logic
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=pred.device))
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic**2 + self.delta * linear
        
        # Asymmetry Logic (Higher = More Stable):
        # Case A: Pred > Target (e.g. Pred=5, Truth=-1). 
        #         We are "Optimistic" (predicting stable when it isn't). HIGH PENALTY.
        # Case B: Pred < Target (e.g. Pred=-1, Truth=5). 
        #         We are "Pessimistic" (missing a hit). LOW PENALTY.
        
        # mask = 1 where we are "too optimistic" (pred is higher/better than truth)
        mask = (pred > target).float() 
        weighted_loss = base_loss * (1.0 + (self.penalty_factor - 1.0) * mask)
        
        if self.reduction == 'none':
            return weighted_loss
        else:
            return weighted_loss.mean()
    

class ListMLELoss_enhanced(torch.nn.Module):
    """
    Enhanced ListMLE loss with configurable tie-breaking strategies.
    
    This loss function extends the standard ListMLE loss by providing control over
    how ties in ground truth rankings are handled, including random tie-breaking
    and tolerance-based grouping of near-equal values.
    """
    
    def __init__(
        self,
        eps: float = 1e-10,
        invert: bool = False,
        tie_break: Literal["stable", "random", "average"] = "random", #"stable",
        tolerance: Optional[float] = 0.05, #None,
        random_seed: Optional[int] = 1 #None
    ):
        """
        Initialize the enhanced ListMLE loss.
        
        Args:
            eps: Small value for numerical stability
            invert: Whether to invert rankings (lower is better)
            tie_break: Strategy for breaking ties:
                - "stable": Deterministic, preserves original order (default, matches current behavior)
                - "random": Random shuffling of tied groups
                - "average": Average loss over all possible orderings (approximated)
            tolerance: If provided, values within this tolerance are treated as tied
            random_seed: Optional seed for reproducible randomness in tie-breaking
        """
        super().__init__()
        self.eps = eps
        self.invert = invert
        self.tie_break = tie_break
        self.tolerance = tolerance
        self.random_seed = random_seed
        
    def _identify_tie_groups(self, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Identify groups of tied or near-tied values.
        
        Returns a tensor where each element indicates which tie group it belongs to.
        """
        # Handle both 1D and 2D inputs
        if ground_truths.dim() == 1:
            ground_truths = ground_truths.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, n_items = ground_truths.shape
        tie_groups = torch.zeros_like(ground_truths, dtype=torch.long)
        
        for b in range(batch_size):
            sorted_vals, sorted_idx = ground_truths[b].sort(descending=not self.invert)
            
            group_id = 0
            tie_groups[b, sorted_idx[0]] = group_id
            
            for i in range(1, n_items):
                if self.tolerance is None:
                    # Exact equality check
                    if sorted_vals[i] != sorted_vals[i-1]:
                        group_id += 1
                else:
                    # Tolerance-based check
                    if abs(sorted_vals[i] - sorted_vals[i-1]) > self.tolerance:
                        group_id += 1
                        
                tie_groups[b, sorted_idx[i]] = group_id
        
        if squeeze_output:
            tie_groups = tie_groups.squeeze(0)
                
        return tie_groups
    
    def _random_shuffle_within_groups(
        self, 
        ground_truths: torch.Tensor, 
        tie_groups: torch.Tensor
    ) -> torch.Tensor:
        """
        Randomly shuffle items within each tie group.
        """
        if self.random_seed is not None:
            gen = torch.Generator(device=ground_truths.device)
            gen.manual_seed(self.random_seed + self._call_count)
            self._call_count += 1
        else:
            gen = None

        # Handle both 1D and 2D inputs
        if ground_truths.dim() == 1:
            ground_truths = ground_truths.unsqueeze(0)
            tie_groups = tie_groups.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, n_items = ground_truths.shape
        shuffled = ground_truths.clone()
            
        for b in range(batch_size):
            unique_groups = torch.unique(tie_groups[b])
            
            for group in unique_groups:
                mask = tie_groups[b] == group
                group_indices = torch.where(mask)[0]
                
                if len(group_indices) > 1:
                    # Add small random noise to break ties
                    noise = torch.rand(len(group_indices), generator=gen, device=ground_truths.device) * 1e-6
                    if self.invert:
                        noise = -noise
                    shuffled[b, group_indices] = shuffled[b, group_indices] + noise
        
        if squeeze_output:
            shuffled = shuffled.squeeze(0)
                    
        return shuffled
    
    def _compute_average_loss_approximation(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        tie_groups: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate the average loss over all possible orderings of tied items.
        
        This is done by computing multiple random permutations and averaging.
        """
        n_samples = 5  # Number of random permutations to sample
        total_loss = 0
        
        for _ in range(n_samples):
            shuffled_gt = self._random_shuffle_within_groups(ground_truths, tie_groups)
            loss = self._compute_base_loss(predictions, shuffled_gt)
            total_loss += loss
            
        return total_loss / n_samples
    
    def _compute_base_loss(
        self, 
        predictions: torch.Tensor, 
        ground_truths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the base ListMLE loss (extracted from original forward method).
        """
        # Get the correct permutation indices
        _, indices = ground_truths.sort(descending=not self.invert, dim=-1)
        
        # Reorder predictions
        ordered_predictions = predictions.gather(-1, indices)
        
        # Numerically stable log-sum-exp
        max_predictions, _ = ordered_predictions.max(dim=-1, keepdim=True)
        exp_predictions = torch.exp(ordered_predictions - max_predictions)
        
        # Corrected cumulative sum
        exp_predictions_rev = torch.flip(exp_predictions, dims=(-1,))
        cum_sum_exp_predictions_rev = torch.cumsum(exp_predictions_rev, dim=-1)
        cum_sum_exp_predictions = torch.flip(cum_sum_exp_predictions_rev, dims=(-1,))
        
        cum_sum_exp_predictions = torch.clamp(cum_sum_exp_predictions, min=self.eps)
        
        # Compute log probabilities
        log_probs = ordered_predictions - max_predictions - torch.log(cum_sum_exp_predictions)
        
        # Return mean loss
        return -torch.sum(log_probs, dim=-1).mean()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        ground_truths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the enhanced ListMLE loss.
        
        Args:
            predictions: Model predictions for ranking (1D or 2D tensor)
            ground_truths: Ground truth values (1D or 2D tensor, higher = better rank by default)
            
        Returns:
            The calculated loss
        """
        # Handle 1D inputs by adding batch dimension
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            ground_truths = ground_truths.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        # Ensure float tensors
        predictions = predictions.float()
        if self.invert:
            predictions =  predictions * -1
            
        ground_truths = ground_truths.float()
        
        # Handle tie-breaking based on strategy
        if self.tie_break == "stable":
            # Default behavior - use stable sort
            loss = self._compute_base_loss(predictions, ground_truths)
            
        elif self.tie_break == "random":
            # Identify tie groups and shuffle within them
            tie_groups = self._identify_tie_groups(ground_truths)
            shuffled_gt = self._random_shuffle_within_groups(ground_truths, tie_groups)
            loss = self._compute_base_loss(predictions, shuffled_gt)
            
        elif self.tie_break == "average":
            # Approximate average over all permutations
            tie_groups = self._identify_tie_groups(ground_truths)
            loss = self._compute_average_loss_approximation(predictions, ground_truths, tie_groups)
            
        else:
            raise ValueError(f"Unknown tie_break strategy: {self.tie_break}")
            
        # If input was 1D, return scalar loss
        if squeeze_batch and loss.numel() == 1:
            loss = loss.squeeze()
            
        return loss