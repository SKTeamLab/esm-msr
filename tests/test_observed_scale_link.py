"""The observed-scale link of validation: the Tobit expected reading of a Tobit-trained run, on the in-distribution libraries only.

* no link for a run trained with plain MSE (--reg_nll none): it regressed the observed ddG, its predictions are read as they are;
* the Tobit link, with the run's own constants, for a Tobit run (--reg_nll tobit, the default and every older hparams);
* never for an external benchmark set, even when its batch carries a finite dG_wt;
* the dump's link_summary says which (kind none / tobit), so an offline rescore agrees.
"""
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esm_msr import training, val_stats  # noqa: E402

CLS = next(v for k, v in vars(training).items() if isinstance(v, type) and hasattr(v, 'validation_step'))
HP = dict(tobit_sigma_y=0.28, tobit_mu_f=-0.14, tobit_sigma_f=0.5)


def test_link_rule_follows_the_regression_loss():
    h = training.observed_scale_link({**HP, 'reg_nll': 'tobit'})
    ref = val_stats.tobit_link(0.28, -0.14, 0.5)
    z = np.array([-3.0, -0.5, 0.0, 1.0, 4.0])
    assert np.allclose(h(z), ref(z))
    assert training.observed_scale_link(HP) is not None                       # no reg_nll in older hparams: Tobit
    assert training.observed_scale_link({**HP, 'reg_nll': 'none'}) is None
    # the link follows the run's own constants (the deep-floor ablation reads through its own floor)
    deep = training.observed_scale_link({**HP, 'tobit_mu_f': -1.0, 'tobit_sigma_f': 1.0})
    assert deep(np.array([-3.0]))[0] < ref(np.array([-3.0]))[0] - 0.3


class _Model:
    def __init__(self):
        self._masked_logit_cache = {}

    def forward_batch(self, batch, mask_strategy=None):
        n = len(batch['ddG'])
        v = torch.linspace(-3, 1, n)
        return {'wt_lora_pred': v, 'mt_lora_pred': v, 'combined_pred': v}


def _step(reg_nll, loader, dG_wt):
    me = SimpleNamespace(model=_Model(), val_mask_strategy='unmasked', val_regimes=[('', 'unmasked')], validation_step_outputs=defaultdict(list), device='cpu',
                         val_dataloader_names=['train_lib', 'ssym'], external_val_sets={'ssym'}, obs_link=training.observed_scale_link({**HP, 'reg_nll': reg_nll}))
    n = 6
    batch = dict(ddG=torch.linspace(-2, 1, n), dG_wt=torch.full((n,), float(dG_wt)), cens=torch.zeros(n, dtype=torch.long), cens_src=torch.zeros(n, dtype=torch.long),
                 dddG=torch.full((n,), float('nan')), subset_type=['single'] * n, pdb=['x'] * n, mutations=['A1G'] * n)
    CLS.validation_step(me, batch, 0, loader)
    return me.validation_step_outputs[loader][0]


def test_validation_reads_through_the_link_only_for_a_tobit_run_on_an_in_distribution_library():
    rec = _step('tobit', 0, 1.0)
    assert 'obs' in rec and set(rec['obs']) == {'wt', 'comb'}
    assert not np.allclose(rec['obs']['comb'], rec['comb_scores'])             # saturated at the floor: differs from the latent
    h = val_stats.tobit_link(0.28, -0.14, 0.5)
    assert np.allclose(rec['obs']['comb'], h(1.0 + rec['comb_scores'].astype(np.float64)) - 1.0)


def test_no_link_for_an_mse_run():
    assert 'obs' not in _step('none', 0, 1.0)


def test_no_link_for_an_external_set_even_with_a_finite_dG_wt():
    assert 'obs' not in _step('tobit', 1, 1.0)
    assert 'obs' not in _step('none', 1, 1.0)
