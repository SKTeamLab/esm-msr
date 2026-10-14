"""
Subset taxonomy and WT/MT head routing for the dual-adapter model.

Every dataset item carries a ``subset_type``. This module is the single source of truth for which adapter ("head") an item trains, how it is
predicted at validation time, and whether its target is a direct measurement or derived.

Sign convention (Tsuboyama ddG_ML): ddG = dG(after) - dG(before), dG = unfolding free energy, so positive ddG = stabilizing. Each pass scores
``LLR = logit(to_residue) - logit(from_residue)`` at the mutated position(s), which is regressed (through a calibration head) onto ddG.

Which residue is *visible* at the scored position depends on the pass:

* WT pass  - input is ``wt_sequence_tokens`` (the before-state), adapter ``peft_wt``. The from-residue is visible ("wt-marginal").
* MT pass  - input is ``mt_sequence_tokens`` (the after-state, carrying every mutation of the item), adapter ``peft_mt``. The to-residue is
  visible ("mt-marginal").

Subsets, as produced by ``data.MutationStabilityDataset``:

===========  ==========  =====================================  ==========================
subset_type  head        scored mutation / conditioning          target
===========  ==========  =====================================  ==========================
single       WT          X, on the real WT structure             ddG_X           (measured)
double       ensemble    A and B, on the WT structure            ddG_AB          (measured)
native_cond  MT          X in a mutant background (code          ddG(X | bg)     (measured)
                         '1A0N_L7S'), WT structure
cond         MT          A, given B present                      ddG_AB - ddG_B  (derived)
===========  ==========  =====================================  ==========================

Notes:

* ``cond`` is one item per ordered pair of a double, so a double with both singles measured yields ddG(A|B) and ddG(B|A). Its target derives
  from a double and its two singles, so each one's noise is the sum of two measurements' noise and is correlated with the double's own label.
* ``native_cond`` is the only *measured* conditional: a library whose code carries a mutation suffix ('1A0N_L7S' = 1A0N measured in the L7S
  background) has no wild-type counterpart, so every measurement in it is already a conditional effect and belongs to the MT head.
* ``forward_batch`` reports 0.5 * WT + 0.5 * MT for every item, whatever owns it in training. With WT = sum_i ddG(i | wt) and
  MT = sum_i ddG(i | all other mutations present), that is the average of the two thermodynamic paths A->AB and B->AB, and for any number of
  mutations it is exact whenever epistasis is at most pairwise (trapezoid rule on the hypercube).
"""
from typing import Iterable, Optional, Sequence

import torch

WT_HEAD_SUBSETS = frozenset({'single'})
MT_HEAD_SUBSETS = frozenset({'native_cond', 'cond'})
ENSEMBLE_SUBSETS = frozenset({'double'})

# Items whose target is a direct assay measurement (not a difference of two).
MEASURED_SUBSETS = frozenset({'single', 'double', 'native_cond'})
# Items whose target is built from a double and its singles.
DOUBLE_DERIVED_SUBSETS = frozenset({'double', 'cond'})
# Items whose target is a conditional effect ddG(X | background) - the MT head's task.
CONDITIONAL_SUBSETS = MT_HEAD_SUBSETS

ALL_SUBSETS = WT_HEAD_SUBSETS | MT_HEAD_SUBSETS | ENSEMBLE_SUBSETS


def head_for(subset_type: str) -> Optional[str]:
    """'wt', 'mt', 'ensemble', or None for an unknown subset."""
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
