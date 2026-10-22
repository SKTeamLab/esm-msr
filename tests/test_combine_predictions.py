"""The reported prediction is the mean of the two heads. ``route`` (WT for singles, MT for the rest) and ``sum`` (MT learns ddG - wt) were
removed: neither ever ran in a completed arm, and ``sum`` asks the LLR-sum head for something it cannot output."""
import pytest
import torch

from esm_msr.models import MSRModel, combine_predictions

WT = torch.tensor([-1.0, -2.0, 0.5, 3.0])
MT = torch.tensor([-1.5, 0.5, -0.25, 1.0])


def test_mean_is_the_released_formula_bit_for_bit():
    torch.testing.assert_close(combine_predictions(WT, MT), 0.5 * WT + 0.5 * MT, atol=0, rtol=0)


def test_gradients_flow_to_both_heads():
    wt, mt = WT.clone().requires_grad_(True), MT.clone().requires_grad_(True)
    combine_predictions(wt, mt).sum().backward()
    assert bool((wt.grad != 0).all()) and bool((mt.grad != 0).all())


def test_the_removed_modes_are_refused_by_the_model():
    for kw in ({'combined_mode': 'route'}, {'combined_mode': 'sum'}, {'lora_mode': 'corrector'}):
        with pytest.raises(AssertionError, match='removed'):
            MSRModel(lora_config={}, **kw)
