"""--reg_nll none (plain MSE on ddG) and --reg_exclude_range.

MSE is the squared error on ddG against the matched label: identical to the squared error on absolute dG with the library's dG_wt added to both sides (the offset cancels), and
equal to the Tobit loss far above the floor. The assay's range-tagged cells ('<-1' / '>5') leave the regression and its denominator (always under mse, on request under tobit);
floor-censored numbers (--rank_censor_threshold) stay.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esm_msr import censoring, training  # noqa: E402
from esm_msr.losses import tobit_regression_loss  # noqa: E402

CLS = next(v for k, v in vars(training).items() if isinstance(v, type) and hasattr(v, '_reg_loss'))
PARAMS = dict(sigma_y=0.28, mu_f=-0.14, sigma_f=0.5)
NAN = float('nan')


def reg(kind, pred, target, bound, cens, dgwt, mask=None):
    me = SimpleNamespace(reg_nll=kind, tobit_params=PARAMS)
    mask = torch.ones(len(pred), dtype=torch.bool) if mask is None else mask
    return CLS._reg_loss(me, pred, target, bound, cens, mask, float(max(1, int(mask.sum()))), dgwt)


def test_mse_is_the_squared_error_on_ddg_whatever_the_library_offset():
    torch.manual_seed(0)
    pred, target = torch.randn(50), torch.randn(50)
    z = torch.zeros(50)
    nan = torch.full((50,), NAN)
    _, v0, n = reg('none', pred, target, nan, z, torch.zeros(50))
    _, v1, _ = reg('none', pred, target, nan, z, torch.rand(50) * 6 - 1)           # any dG_wt: the loss does not see it
    assert n == 50
    assert abs(v0 - float(((pred - target) ** 2).sum())) < 1e-4
    assert abs(v0 - v1) < 1e-6
    # ... which is what adding dG_wt to both sides gives (the Tobit code's own form of the same quantity)
    off = torch.rand(50) * 6 - 1
    assert abs(float((((off + pred) - (off + target)) ** 2).sum()) - v0) < 1e-3


def test_mse_matches_tobit_far_above_the_floor_but_not_below_it():
    far = reg('tobit', torch.tensor([0.5, -0.2]), torch.tensor([0.1, 0.0]), torch.full((2,), NAN), torch.zeros(2), torch.full((2,), 4.0))[1]
    assert abs(far - reg('none', torch.tensor([0.5, -0.2]), torch.tensor([0.1, 0.0]), torch.full((2,), NAN), torch.zeros(2), torch.full((2,), 4.0))[1]) < 1e-3
    # a dead item (dG_wt 0, reading at the floor 0.05) predicted at -3: Tobit forgives it, MSE does not
    args = (torch.tensor([-3.0]), torch.tensor([0.05]), torch.full((1,), NAN), torch.zeros(1), torch.zeros(1))
    assert reg('tobit', *args)[1] < reg('none', *args)[1] - 5.0


def test_mse_refuses_a_range_tagged_item_and_has_gradient_elsewhere():
    pred = torch.tensor([0.3, 0.3], requires_grad=True)
    t, v, _ = reg('none', pred, torch.tensor([0.0, NAN]), torch.full((2,), NAN), torch.zeros(2), torch.zeros(2))
    assert abs(v - 0.09) < 1e-6                                   # the non-finite label contributes nothing
    t.backward()
    assert pred.grad[0].abs() > 0 and pred.grad[1] == 0
    with pytest.raises(AssertionError):
        reg('none', pred.detach(), torch.zeros(2), torch.zeros(2), torch.tensor([0.0, -1.0]), torch.zeros(2))


def _lab():
    # five cells: ordinary, range-tagged dead ('<-1'), range-tagged high ('>5'), floor-censored number, ordinary without a dG_wt
    return dict(y=torch.tensor([0.5, -1.0, 5.0, 0.0, 1.0]), dG_wt=torch.tensor([2.0, 2.0, 2.0, 2.0, NAN]),
                cens=torch.tensor([0.0, -1.0, 1.0, -1.0, 0.0]),
                cens_src=torch.tensor([censoring.SRC_NONE, censoring.SRC_RANGE, censoring.SRC_RANGE, censoring.SRC_FLOOR, censoring.SRC_NONE]),
                bound=torch.tensor([NAN, -1.0, 5.0, NAN, NAN]))


@pytest.mark.parametrize('kind,exclude,kept', [('tobit', False, [0, 1, 2, 3]), ('tobit', True, [0, 3]), ('none', False, [0, 3]), ('none', True, [0, 3])])
def test_range_tagged_cells_leave_the_regression_only_when_asked_or_under_mse(kind, exclude, kept):
    me = SimpleNamespace(reg_nll=kind, reg_exclude_range=exclude)
    c = CLS._block_ctx(me, _lab(), [])
    assert torch.nonzero(c.obs_reg).flatten().tolist() == kept     # the cell without dG_wt was never in the regression; the floor-censored number stays
    assert int(c.obs_rank.sum()) == 5                                # the rank lists keep every cell with a label
    denom = float(max(1, int(c.obs_reg.sum())))
    assert denom == len(kept)                                         # the denominator follows the kept cells
