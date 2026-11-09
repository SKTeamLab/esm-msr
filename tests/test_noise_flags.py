"""--tf_prior_sd and --tobit_sigma_y / _mu_f / _sigma_f default to the constants of losses.py, so a run that does not pass them trains exactly as before, and they do reach the label."""
import sys

import pytest
import torch

from esm_msr import config, losses
from esm_msr.training import tf_single_label


def parse(*extra):
    old = sys.argv
    try:
        sys.argv = ['training.py', '--experiment_name', 'x'] + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = old


def test_the_defaults_are_the_constants_of_losses():
    a = parse()
    assert (a.tf_prior_sd, a.tobit_sigma_y, a.tobit_mu_f, a.tobit_sigma_f) == (losses.TF_PRIOR_SD, losses.TOBIT_SIGMA_Y, losses.TOBIT_MU_F, losses.TOBIT_SIGMA_F)


def test_non_positive_widths_are_refused():
    for bad in ('--tf_prior_sd', '--tobit_sigma_y', '--tobit_sigma_f'):
        with pytest.raises(SystemExit):
            parse(bad, '0')
    assert parse('--tobit_mu_f', '-0.1').tobit_mu_f == -0.1                      # a mean may be anything


def test_the_posterior_label_uses_the_run_s_constants():
    y, wt_lat = torch.tensor([1.0, 0.05, 2.0]), torch.tensor([0.2, 0.8, 1.0])
    dg, cens, src = torch.full((3,), 2.0), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    default = tf_single_label(y, cens, src, wt_lat, dg)
    explicit = tf_single_label(y, cens, src, wt_lat, dg,
                               post=dict(s=losses.TF_PRIOR_SD, sigma_y=losses.TOBIT_SIGMA_Y, mu_f=losses.TOBIT_MU_F, sigma_f=losses.TOBIT_SIGMA_F))
    wide = tf_single_label(y, cens, src, wt_lat, dg, post=dict(s=1.5, sigma_y=losses.TOBIT_SIGMA_Y, mu_f=losses.TOBIT_MU_F, sigma_f=losses.TOBIT_SIGMA_F))
    assert torch.equal(default, explicit)
    assert not torch.allclose(default, wide)                                       # a wider prior trusts the reading more
    assert (wide - y).abs().sum() < (default - y).abs().sum()
