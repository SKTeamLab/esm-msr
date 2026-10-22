"""Shared stand-ins for the tests that drive ``ESM3EpistasisLightningModule`` methods on a stub ``self`` (no backbone, CPU)."""
import types

import torch


class IdentityLink:
    """A link that does nothing: observed == latent. The tests that read an exact loss value out of the composition use it so the
    arithmetic stays checkable by hand; the real link has its own tests (test_link*.py, test_floor_hinge_knee.py)."""

    def effective_floor(self, y_floor):
        return torch.as_tensor(y_floor, dtype=torch.float32)

    def __call__(self, z, y_floor):
        return z

    def obs_ddG(self, latent, dG_wt, y_floor):
        return latent

    def latent_ddG(self, observed, dG_wt, y_floor):
        return observed

    def summary(self):
        return {'tau': 0.59, 'floor_min': 0.0, 'floor_max': 0.001, 'global_floor': 0.0}


def compose_hparams(**over):
    """The hparams ``_compose_losses_streaming_and_backward`` reads, at the standing recipe's values."""
    hp = dict(rank_list_size=4, micro_batch_size=3, native_cond_for_wt=True, native_cond_for_mt=False, censor_threshold=None, tf_mix=1.0,
              lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0, huber_delta=0.0,
              freeze_wt_adapter=False, freeze_mt_adapter=False)
    hp.update(over)
    return hp


def with_identity_link(stub):
    """Give a hand-built stub ``self`` the pieces the composition now always uses."""
    stub.link_head = IdentityLink()
    stub.mask_wt = stub.mask_mt = None
    stub.mask_strategy = {'wt': None, 'mt': None}
    return stub
