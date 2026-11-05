"""Shared stand-ins for the tests that drive ``ESM3EpistasisLightningModule`` methods on a stub ``self`` (no backbone, CPU)."""


def compose_hparams(**over):
    """The hparams ``_compose_losses_streaming_and_backward`` reads, at the standing recipe's values."""
    hp = dict(rank_list_size=4, micro_batch_size=3, censor_threshold=None, tf_mix=1.0,
              lambda_reg_wt=0.0, lambda_rank_wt=0.0, lambda_reg_mt=1.0, lambda_rank_mt=0.0, huber_delta=0.0,
              freeze_wt_adapter=False, freeze_mt_adapter=False)
    hp.update(over)
    return hp


def with_masks(stub):
    """Give a hand-built stub ``self`` the pieces the composition always uses."""
    stub.mask_wt = stub.mask_mt = None
    stub.mask_strategy = {'wt': None, 'mt': None}
    return stub
