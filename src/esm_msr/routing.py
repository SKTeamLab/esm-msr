"""
Subset taxonomy and WT/MT head routing for the dual-adapter model.

Every dataset item carries a ``subset_type``. This module is the single source of
truth for which adapter ("head") an item trains, how it is predicted at
validation time, and whether its target is a direct measurement or derived.

Sign convention (Tsuboyama ddG_ML): ddG = dG(after) - dG(before), dG = unfolding
free energy, so positive ddG = stabilizing. Each pass scores
``LLR = logit(to_residue) - logit(from_residue)`` at the mutated position(s), which
is regressed (through a calibration head) onto ddG.

Which residue is *visible* at the scored position depends on the pass:

* WT pass  - input is ``wt_sequence_tokens`` (the before-state), adapter ``peft_wt``.
  The from-residue is visible ("wt-marginal").
* MT pass  - input is ``mt_sequence_tokens`` (the more-mutated state), adapter
  ``peft_mt``. For forward items the to-residue is visible ("mt-marginal").

Subsets, as produced by ``data.ProteinStructureMutationEpistasisDataset``:

================  ===========  ====================================  ==========================
subset_type       head         MT-pass input / scored mutation       target
================  ===========  ====================================  ==========================
single            WT           (WT pass) wild type, X                ddG_X           (measured)
double            ensemble     AB, A and B                           ddG_AB          (measured)
native_mut_ctx    MT           bg+X, X   (code '1A0N_L7S')           ddG(X | bg)     (measured)
mut_ctx_rev       MT           AB, A -> wt (reversion)               ddG_B - ddG_AB  (derived)
mut_ctx           MT           AB, B   (context A)                   ddG_AB - ddG_A  (derived)
reversion         none         wild type, X -> wt                    -ddG_X          (derived)
================  ===========  ====================================  ==========================

Notes:

* ``mut_ctx_rev`` and ``mut_ctx`` carry the *same* information. "Revert A in AB"
  feeds the MT pass sequence AB and position A with LLR = logit(wtA) - logit(mtA)
  and target ddG_B - ddG_AB; "A given B" feeds the same sequence and position with
  the negated LLR and the negated target. Only the structure masking differs.
  Enable one or the other, not both, or every double is counted four times.
* Both derive from a double and its two singles, so each target's noise is the
  sum of two measurements' noise and is correlated with the double's own label.
* ``reversion`` has no head under the dual design: its WT pass sees a mutant
  sequence (violating "WT = real wild-type context") and its MT pass sees the
  wild-type sequence. The same physics is learned by scoring forward singles in
  the MT pass (see ``lambda_mt_single_anchor`` in training).
* Doubles are predicted as 0.5 * WT + 0.5 * MT. With WT = sum_i ddG(i | wt) and
  MT = sum_i ddG(i | all other mutations present), this is the average of the
  two thermodynamic paths A->AB and B->AB, and for any number of mutations it is
  exact whenever epistasis is at most pairwise (trapezoid rule on the hypercube).
"""
from typing import Iterable, Optional, Sequence

import torch

WT_HEAD_SUBSETS = frozenset({'single'})
MT_HEAD_SUBSETS = frozenset({'native_mut_ctx', 'mut_ctx_rev', 'mut_ctx'})
ENSEMBLE_SUBSETS = frozenset({'double'})
UNROUTED_SUBSETS = frozenset({'reversion'})

# Items whose target is a direct assay measurement (not a difference of two).
MEASURED_SUBSETS = frozenset({'single', 'double', 'native_mut_ctx'})
# Items whose target is built from a double and its singles.
DOUBLE_DERIVED_SUBSETS = frozenset({'double', 'mut_ctx', 'mut_ctx_rev'})

ALL_SUBSETS = WT_HEAD_SUBSETS | MT_HEAD_SUBSETS | ENSEMBLE_SUBSETS | UNROUTED_SUBSETS


def head_for(subset_type: str) -> Optional[str]:
    """Return 'wt', 'mt', 'ensemble', or None (unrouted) for a subset type."""
    if subset_type in WT_HEAD_SUBSETS:
        return 'wt'
    if subset_type in MT_HEAD_SUBSETS:
        return 'mt'
    if subset_type in ENSEMBLE_SUBSETS:
        return 'ensemble'
    return None


def subset_mask(subset_types: Sequence[str], subsets: Iterable[str], device=None) -> torch.Tensor:
    """Boolean tensor marking which entries of ``subset_types`` belong to ``subsets``."""
    subsets = frozenset(subsets)
    return torch.as_tensor([s in subsets for s in subset_types], dtype=torch.bool, device=device)


def combine_rule_from_hparams(hparams: dict) -> str:
    """
    Choose ``MSRModel.combine_rule`` for a checkpoint from its training hparams.

    Checkpoints trained with the combined (teacher-forced 0.5*WT + 0.5*MT) losses expect
    every item to be averaged ('average'); checkpoints trained with separate heads and no
    combined loss expect per-subset routing ('routed').
    """
    combined = any(float(hparams.get(k, 0.0) or 0.0) > 0 for k in
                   ('lambda_reg_combined', 'lambda_rank_combined', 'lambda_epi_combined'))
    separate = float(hparams.get('lambda_reg_mt', 0.0) or 0.0) > 0
    return 'routed' if separate and not combined else 'average'
