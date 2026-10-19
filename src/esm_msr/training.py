import json
import os
import logging
import warnings
from collections import defaultdict
import gc



import numpy as np
import torch 
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import CometLogger, CSVLogger
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin

from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from esm_msr.models import MSRModel, combine_predictions
from esm_msr import censoring, epi_metrics, stats, utils
from esm_msr.link import SaturatingLink, NonspecificTerm
from esm_msr.losses import ListMLELoss, hinge_regression_loss
from esm_msr.preprocess_megascale import setup_dataloaders
from esm_msr.peft_manager import PEFTStateManager
from esm_msr.config import parse_arguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') 

def alternating_phase(batch_idx: int, has_single: bool, has_multi: bool) -> str:
    """
    Which head this batch trains when --alternate_heads is on: 'wt' (singles) or 'mt' (multi-mutants).

    Alternates on the batch index so the two heads get equal turns, and falls back to the other head when this
    batch cannot serve its turn - every batch is ONE library, and the median training library holds no
    multi-mutants at all, so an unconditional alternation would waste every other step on most libraries.
    Returns 'none' when the batch has neither, which the caller skips.
    """
    if not has_single and not has_multi:
        return 'none'
    turn = 'wt' if batch_idx % 2 == 0 else 'mt'
    if turn == 'mt' and not has_multi:
        return 'wt'
    if turn == 'wt' and not has_single:
        return 'mt'
    return turn


EXTERNAL_SUMMARY_METRICS = ('rho_combined', 'rmse_combined')


def summarise_external_benchmarks(per_loader: dict, external: set) -> dict:
    """OOD validation loaders summarised as ``val_<metric>_external``: the mean of rho_combined and rmse_combined.

    ``per_loader`` maps loader name -> {metric: value}. These libraries are deliberately kept out of every aggregate
    (``val_*_avg``, the pooled global ranking, the epistasis levels): they are out of the training distribution, so
    including them makes the aggregate incomparable across runs and inconsistent with what the model was trained on.
    Their per-loader columns are still logged by the caller, so nothing diagnostic is lost.
    """
    out = {}
    for metric in EXTERNAL_SUMMARY_METRICS:
        vals = [v[metric] for name, v in per_loader.items()
                if name in external and metric in v and np.isfinite(v[metric])]
        if vals:
            out[f'val_{metric}_external'] = float(np.mean(vals))
    return out


class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        mt_lora_config = {
            "lora_rank": self.hparams.lora_rank_mt, "lora_alpha": self.hparams.lora_alpha_mt, "lora_dropout": self.hparams.lora_dropout_mt,
            "target_mode": self.hparams.target_mode_mt, "use_dora": self.hparams.use_dora_mt, "seed": self.hparams.seed,
            "incl_structure_encoder": self.hparams.incl_structure_encoder_mt, "last_n_layers": self.hparams.last_n_layers_mt,
            "incl_sequence_head": self.hparams.incl_sequence_head_mt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_mt,
        }

        wt_lora_config = {
            "lora_rank": self.hparams.lora_rank_wt, "lora_alpha": self.hparams.lora_alpha_wt, "lora_dropout": self.hparams.lora_dropout_wt,
            "target_mode": self.hparams.target_mode_wt, "use_dora": self.hparams.use_dora_wt, "seed": self.hparams.seed,
            "incl_structure_encoder": self.hparams.incl_structure_encoder_wt, "last_n_layers": self.hparams.last_n_layers_wt,
            "incl_sequence_head": self.hparams.incl_sequence_head_wt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_wt,
        }

        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': self.hparams.seed}

        self.model = MSRModel(
            lora_config=lora_config, shared_scale_init=self.hparams.shared_scale_init, shared_bias_init=self.hparams.shared_bias_init, adapter_mode=self.hparams.adapter_mode,
            lora_mode=self.hparams.lora_mode, model_dtype=torch.float32,
            combined_mode=getattr(self.hparams, 'combined_mode', None) or 'mean',
            per_mutation_bias=bool(getattr(self.hparams, 'per_mutation_bias', False))
        )
        
        self.peft_manager = PEFTStateManager(self.model)
        # --init_wt_from_checkpoint: copy a pre-trained WT model in BEFORE anything freezes it, so that
        # --freeze_wt_adapter / --freeze_link (or --frozen_wt_mt_only) hold exactly the loaded model fixed.
        if self.hparams.get('init_wt_from_checkpoint'):
            self._load_wt_weights(self.hparams.init_wt_from_checkpoint)
        # on_save_checkpoint deliberately keeps only the lora / calibration / link_head entries, so a
        # checkpoint is missing every frozen base weight. Lightning's default strict load would then refuse
        # to resume from our own checkpoints; the base comes from the pretrained weights either way.
        self.strict_loading = False

        if self.hparams.freeze_wt_adapter: self.peft_manager.freeze_wt_components()
        if self.hparams.freeze_mt_adapter: self.peft_manager.freeze_mt_components()

        self.crit_rank_wt = ListMLELoss() if self.hparams.lambda_rank_wt > 0 else None
        self.crit_rank_mt = ListMLELoss() if self.hparams.lambda_rank_mt > 0 else None

        if self.hparams.reg_loss == 'huber':
            self.crit_reg = nn.HuberLoss(reduction='none', delta=self.hparams.huber_delta)
        else:
            self.crit_reg = nn.MSELoss(reduction='none')

        # The observed-scale link. Regression goes through it; ranking does not, because it is monotone.
        self.link_head = SaturatingLink(
            tau=self.hparams.link_tau, floor_min=self.hparams.link_floor_min, floor_max=self.hparams.link_floor_max,
            learn_tau=self.hparams.learn_link_tau, learn_floor_bounds=self.hparams.learn_link_floor_bounds,
            global_floor_init=self.hparams.get('dataset_floor', float('nan')),
            learn_global_floor=self.hparams.learn_global_floor,
            ceiling=self.hparams.get('link_ceiling'), learn_ceiling=self.hparams.get('learn_link_ceiling', True),
            tau_hi=self.hparams.get('link_tau_hi'),
        ) if self.hparams.link else None
        if self.link_head is not None:
            logging.info(f"Observed-scale link enabled: {self.link_head.summary()}")
            # The link is part of the pre-trained WT model, not a separate piece: a frozen WT adapter paired with a
            # link that still moves (or with a different floor) is a different model than the one being frozen.
            if self.hparams.get('init_wt_from_checkpoint'):
                self._load_link_weights(self.hparams.init_wt_from_checkpoint)
            if self.hparams.get('freeze_link', False):
                self._freeze_link()
        # --nonspecific_term fx: nonspecific epistasis as its own learned term, f(x) (esm_msr.link.NonspecificTerm), added to a
        # multi-mutant's combined latent BEFORE the link so the link keeps modelling the readout alone. Lives on the Lightning
        # module, not on self.model, so the released model's state and inference are unchanged; checkpoints keep it as ns_head.*.
        self.ns_head = None
        if self.hparams.get('nonspecific_term', 'none') == 'fx':
            if self.link_head is None:
                raise ValueError("--nonspecific_term fx needs --link: f acts on the latent scale, before the link")
            if (self.hparams.get('combined_mode') or 'mean') != 'mean':
                raise ValueError("--nonspecific_term fx is defined for --combined_mode mean only: f enters the combined "
                                 "prediction as a latent term shared between the WT and MT halves, which is what makes "
                                 "comb = (WT + MT) / 2 hold at validation")
            self.ns_head = NonspecificTerm(lo=float(self.hparams.get('ns_knot_lo', -3.0)),
                                           hi=float(self.hparams.get('ns_knot_hi', 5.0)),
                                           step=float(self.hparams.get('ns_knot_step', 0.5)))
            logging.info(f"Nonspecific epistasis term enabled: {len(self.ns_head.values)} knots, "
                         f"{float(self.ns_head.knots[0])}..{float(self.ns_head.knots[-1])} kcal/mol additive dG, "
                         f"lr {float(self.hparams.get('ns_lr', 5e-3))}")

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = self.hparams.get('val_dataloader_names', ['val'])
        self.external_val_sets = {s.strip() for s in str(self.hparams.get('external_val_sets', '') or '').split(',')
                                  if s.strip()}
        if self.external_val_sets:
            logging.info(f"OOD validation sets held OUT of every aggregate (diagnostics only): "
                         f"{sorted(self.external_val_sets)}; summarised as val_rho_combined_external and "
                         f"val_rmse_combined_external")

    def _cast_frozen_linears_bf16(self):
        """
        Store the frozen ESM3 transformer Linear weights in bf16 on the training
        device.

        Under bf16-mixed autocast, F.linear computes in bf16: inputs and fp32
        weights are cast to bf16 for every call. For frozen (requires_grad=False)
        base weights autocast does NOT cache that cast, so every forward re-casts
        ~1.4B params (~300 extra kernels per forward) and a ~2.8 GB bf16 copy of
        the weights is held per live autograd graph. Storing the frozen weights in
        bf16 up front removes both: the forward math is numerically identical
        (autocast would have rounded the fp32 weights to bf16 anyway), the frozen
        weights are never updated, and the re-share hook runs afterwards so MT
        ends up pointing at the same bf16 storage. Only the frozen
        transformer-block Linears are cast; embeddings, layernorms, LoRA params,
        and the output heads stay fp32.
        """
        peft_wt = getattr(self.model, 'peft_wt', None)
        if peft_wt is None:
            return
        n = 0
        for name, m in peft_wt.named_modules():
            if 'transformer.blocks.' not in name or 'lora_' in name:
                continue
            if isinstance(m, torch.nn.Linear) and m.weight is not None \
                    and m.weight.dtype == torch.float32 and not m.weight.requires_grad:
                m.weight.data = m.weight.data.to(torch.bfloat16)
                n += 1
        logging.info(f"[bf16] cast {n} frozen transformer Linear weights to bf16 in peft_wt")

    def _reshare_base_params_on_device(self):
        """
        Re-share the frozen ESM3 base weights between the WT and MT PEFT copies on
        the training device.

        MSRModel.add_loras_to_esm3 (dual mode) re-shares the base parameters on the
        CPU via `p_mt.data = p_wt.data`, but `nn.Module.to(device)` moves each
        Parameter object independently, silently breaking that storage sharing and
        leaving the 1.4B base model duplicated on the GPU (~5.8 GB wasted in fp32).
        This hook re-applies the name-matched re-share after the device move and
        after checkpoint restore (which likewise breaks the sharing). Safe because
        every re-shared parameter is frozen (requires_grad=False, no optimizer
        updates ever write to it).
        """
        peft_wt = getattr(self.model, 'peft_wt', None)
        peft_mt = getattr(self.model, 'peft_mt', None)
        if peft_wt is None or peft_mt is None:
            return  # single-adapter mode: nothing to re-share
        # Cast the frozen WT transformer Linears to bf16 first so the re-share
        # below points MT at the same bf16 storage (no extra copy).
        self._cast_frozen_linears_bf16()

        def _clean(n: str) -> str:
            return n.replace("base_model.model.", "").replace(".base_layer.", ".").replace(".original_module.", ".")

        wt_base = {_clean(n): p for n, p in peft_wt.named_parameters()
                   if "lora_" not in n and "dora_" not in n}
        n_shared, n_repointed, dev = 0, 0, None
        for name, p in peft_mt.named_parameters():
            if "lora_" in name or "dora_" in name:
                continue
            src = wt_base.get(_clean(name))
            if src is not None:
                if p.data.data_ptr() != src.data.data_ptr():
                    n_repointed += 1
                p.data = src.data
                n_shared += 1
                dev = src.device
        # Quiet when the sharing was already intact, so calling this from every stage does not add a line per
        # validation, and loud when it had to repair something, because that is a 2.7 GB event.
        msg = (f"[reshare] dual-adapter base weights re-shared on {dev}: {n_shared} params deduplicated "
               f"({n_repointed} had been duplicated; transformer Linears stored bf16, rest fp32)")
        logging.info(msg) if n_repointed else logging.debug(msg)

    def _prepare_device_weights(self):
        """
        Deduplicate and down-cast the frozen base weights, once, before ANY forward pass.

        This has to happen before the first forward, not just before training. ``main()`` calls
        ``trainer.validate`` before ``trainer.fit``, so the hook this used to live in (``on_train_start``) fired too
        late: that first validation ran with the base weights still duplicated between the two adapters AND still in
        fp32, which is roughly 11 GB of parameters instead of 3. That sets the CUDA caching allocator's high-water
        mark at its maximum, and the reshape that follows frees those fp32 storages into the allocator's CACHE
        rather than back to the driver - so training then allocates its activations inside a large, badly
        fragmented reservation. The symptom is a monotone slowdown (2.2 s/step rising past 12 s/step by step 260)
        with the card pinned near 100%, which looks like a leak and is really fragmentation.

        Do NOT call empty_cache here. An earlier version did, on the theory that the freed fp32 storages should go
        back to the driver. Measured at micro_batch_size 64, with the GPU exclusive, it cost 3.7 GiB of DEVICE
        memory while changing torch's own accounting not at all:

            main-baseline            torch peak 23.10  reserved 23.61  nvidia-smi 27.38 GiB
            here, with empty_cache   torch peak 23.10  reserved 23.63  nvidia-smi 31.07 GiB
            here, without it         torch peak 23.10  reserved 23.61  nvidia-smi 27.38 GiB

        Under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (which every launcher here sets), empty_cache
        releases the expandable segment's pages and the next allocation re-expands into fresh mappings; the WSL2
        driver keeps the released ones, so the process footprint grows by roughly what was freed. The allocator
        reuses those blocks perfectly well on its own - reserved is 23.61 either way - so the call bought nothing
        and spent 3.7 GiB of the 31.84 GiB card, which is the difference between 4.4 GiB of headroom at
        micro_batch_size 64 and 0.8.

        The reshape ordering above is still the point of this method and still necessary.

        Called from every stage hook, and deliberately NOT behind a once-only flag. The flag this used to carry
        made the re-share fire exactly once per process, which is wrong whenever a stage boundary moves the
        module: ``main()`` validates before it fits, so a fresh run's first hook is ``on_validation_start``;
        Lightning returns the module to the CPU when that stage tears down; and the move back for ``fit`` gives
        every parameter its own storage again - the breakage this exists to undo, one stage later, with the flag
        blocking the repair. Measured, dc8_s2 against its own resume (which passes --skip_val and so re-shares
        at ``on_train_start`` instead): mem/allocated_gb 5.706 GB against 2.972, i.e. a second copy of the
        frozen bf16 base carried for the whole run, and the same 2.73 GB again on reserved and on peak. Only
        runs that skip the zero-shot validation were ever correct. Re-sharing is a dict build and ~570 pointer
        assignments, so running it once per stage costs nothing worth guarding.
        """
        self._reshare_base_params_on_device()

    def on_validation_start(self):
        self._prepare_device_weights()

    def on_train_start(self):
        self._prepare_device_weights()
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def _compute_rank_loss(self, pred, targets, mask, cens, list_size, crit_fn):
        """
        ListMLE over the batch chopped into lists of ``list_size``, censored members tied at the bottom.

        The batch is one library (the sampler guarantees it), so a list is a random sample of that library's
        variants and the ranking is always within-library.

        Mean over this micro-batch's lists of each list's loss scaled by ``list_size / its own valid length``; a
        list with fewer than two valid members contributes 0 and still counts. The caller weights the result by
        ``n_list / global_num_lists``, so the sum over micro-batches is the batch's rank term whatever
        ``--micro_batch_size`` is - a micro-batch of one list being the degenerate case.

        Scoring the block in one call and dividing by the micro-batch's MEAN length does not have that property,
        and it was wrong in two ways at once (tests/test_rank_list_normalisation.py; main-baseline 1ca241e):
        a list's weight picked up its neighbours' lengths, and ``forward_censored`` averages over VALID lists
        while ``n_list`` counts all of them, so a micro-batch holding one valid list and three starved ones gave
        that list four times its share. At ``--micro_batch_size 64`` with ``--subset_size 16`` - four lists per
        micro-batch, with 27% of training doubles carrying no additive label - that was a 6.7% error in the
        batch's rank term, and a different one for every chop.
        """
        if mask.dtype != torch.bool:
            raise AssertionError(f"rank mask must be bool, got {mask.dtype}.")
        valid_len = (pred.shape[0] // list_size) * list_size
        if valid_len <= 0:
            return None, 0.0, 0
        view = lambda t: t[:valid_len].view(-1, list_size)
        pred_rank, targ_rank, mask_rank, cens_rank = view(pred), view(targets), view(mask), view(cens)
        num_lists = valid_len // list_size
        # Per list, so that neither the scaling nor forward_censored's own mean can mix lists. A starved list
        # returns a gradient-attached zero, which is what lets it count in the mean without contributing.
        per_list = torch.stack([crit_fn.forward_censored(pred_rank[i:i + 1], targ_rank[i:i + 1],
                                                         mask_rank[i:i + 1], cens_rank[i:i + 1])
                                for i in range(num_lists)])
        n_valid = mask_rank.float().sum(dim=-1).clamp(min=1.0)
        scaled_loss = (per_list * (list_size / n_valid)).mean()
        return scaled_loss, per_list.detach().mean().item(), num_lists

    @staticmethod
    def _micro_batch(B: int, seq_len: int, list_size: int, hard_cap: int, budget: int) -> int:
        """
        How many items to put through one forward/backward, capped so that memory does not track sequence length.

        Activation memory scales with (micro batch x sequence length), and every batch here is ONE library, so a
        fixed micro batch costs whatever that library's length happens to be. Measured on this model at bf16:
        3.0 GB of resident weights plus about 5.4 MB per token, so micro_batch_size 64 costs 23 GB at the median
        training length of 49 and extrapolates past the 31.8 GiB card on the longest libraries (74). Exceeding the
        card does not raise OOM under WSL2 - the driver silently spills to host memory over PCIe - so the only
        symptom is throughput collapsing, which is what made this hard to see.

        ``--micro_batch_tokens`` is therefore the real budget and ``--micro_batch_size`` only an upper bound: short
        libraries still get the full 64, long ones are cut down. The result is rounded to a multiple of
        ``list_size``, because the rank loss reshapes the micro batch into lists of exactly that size.
        """
        hard_cap = max(1, int(hard_cap))
        budget = int(budget or 0)
        n = hard_cap if budget <= 0 else min(hard_cap, max(1, budget // max(1, seq_len)))
        n = max(list_size, (n // list_size) * list_size)      # whole lists only
        return min(B, n)

    def _reg_loss(self, pred, target, bound, cens, mask, weights, global_w_sum):
        """
        Weighted regression over ``mask``, one-sided wherever the measurement is only a bound.

        Returns ``(loss_term, summed_value, summed_weight)``. A censored item is scored against ``bound`` and only
        when the prediction sits on the wrong side of it (``losses.hinge_regression_loss``), so a variant the assay
        pinned at its floor cannot be pushed to a value it never had.
        """
        if mask.dtype != torch.bool:
            raise AssertionError(f"regression mask must be bool, got {mask.dtype}: an integer mask indexes rows "
                                 f"instead of selecting them, which fails silently.")
        L = hinge_regression_loss(self.crit_reg, pred[mask], target[mask], bound[mask], cens[mask])
        w = weights[mask]
        return (L * w).sum() / global_w_sum, float((L * w).sum().item()), float(w.sum().item())

    def _compose_losses_streaming_and_backward(self, batch: dict) -> dict:
        device, B = batch['ddG'].device, int(batch['ddG'].shape[0])
        list_size = max(1, int(self.hparams.subset_size))
        mb = self._micro_batch(B, int(batch['wt_sequence_tokens'].shape[1]), list_size,
                               getattr(self.hparams, 'micro_batch_size', 32),
                               getattr(self.hparams, 'micro_batch_tokens', 0))
        
        st_all = batch.get('subset_type', ['single'] * B)
        w_all = torch.ones(B, dtype=torch.float32, device=device)
        for subset, weight in (('double', self.hparams.double_weight),
                               ('cond', self.hparams.cond_weight),
                               ('native_cond', self.hparams.native_cond_weight)):
            idx = torch.as_tensor([i for i, t in enumerate(st_all) if t == subset], device=device, dtype=torch.long)
            if idx.numel() > 0:
                w_all[idx] = weight

        global_w_sum = w_all.sum().clamp_min(1e-9)
        global_num_lists = max(1, B // list_size)

        # --link: regression on the observed scale, observed ddG = h(dG_wt + latent) - dG_wt, with the knee at the
        # item's own library floor. An item whose library has no measured dG_wt (the external benchmarks, which this
        # assay never measured) cannot be put on that scale, so it leaves the linked regressions but keeps its rank
        # terms, which the link would not have changed anyway.
        link = self.link_head
        dGwt_all = batch.get('dG_wt', torch.full((B,), float('nan'), device=device)).float()
        floor_all = batch.get('y_floor', torch.full((B,), float('nan'), device=device)).float()
        link_ok_all = torch.isfinite(dGwt_all) if link is not None else torch.ones(B, dtype=torch.bool, device=device)
        cens_all = batch.get('cens', torch.zeros(B, dtype=torch.long, device=device)).long()
        bound_all = batch.get('cens_bound', torch.full((B,), float('nan'), device=device)).float()

        # A bound below what the link can even report is not a bound the model can honour. The floor source already
        # bounds an item at exactly the link's floor, but the two-protease source bounds it at its OWN reading, and
        # that reading is frequently BELOW the floor (on 1QP2, 92% of the gap-censored items, by a median 0.41
        # kcal/mol) - precisely because the fit that produced it was unreliable. Left alone the hinge is then
        # violated no matter what the model predicts, contributing an irreducible squared error (mean 0.20 there)
        # and pushing the latent down forever instead of stopping once the prediction saturates.
        #
        # Raising the bound to the link's floor makes it satisfiable: the model discharges it by predicting at or
        # below the floor, the penalty decays to zero as it does, and the two censoring sources end up with
        # identical semantics. Detached, so a bound never trains the floor it is derived from.
        #
        # The ceiling needs the mirror image once --link_ceiling is on. A '>5' row is bounded at 5 - dG_wt, but
        # what the link can actually emit tops out at ceiling - dG_wt, and the ceiling is LEARNED: the moment it
        # settles below 5 that bound is unreachable for every such item, the hinge is violated whatever the model
        # predicts, and the only way the gradient can respond is to push the latent up without limit. Clamping
        # both ways keeps every bound inside the range the link can report.
        if link is not None:
            # The floor on the ITEM'S OWN ddG scale. The offset between an item's label and the absolute dG its
            # measurement describes is a constant of the item, so the same shift that defines cens_bound converts
            # the floor: for a single or a double it is just -dG_wt, and for a conditional item it also carries the
            # partner's single. Computed from the batch rather than assumed, so it stays correct if cond is enabled.
            offset = batch.get('ddG').float() - batch.get('dG_meas', torch.full((B,), float('nan'), device=device)).float()
            floor_ddG = (link.effective_floor(floor_all).detach() + offset)
            # Only where the floor is computable: torch.maximum propagates NaN, which would destroy a good bound.
            bound_all = torch.where(torch.isfinite(floor_ddG), torch.maximum(bound_all, floor_ddG), bound_all)
            ceil_t = link.ceiling()
            if ceil_t is not None:
                ceil_ddG = ceil_t.detach() + offset
                bound_all = torch.where(torch.isfinite(ceil_ddG), torch.minimum(bound_all, ceil_ddG), bound_all)

        # --alternate_heads: one head per batch. Decided once for the whole batch, from the batch's own
        # composition, so every micro-batch of it takes the same turn.
        phase = None
        if getattr(self.hparams, 'alternate_heads', False):
            _single_all = torch.as_tensor([t in ('single', 'native_cond') for t in st_all], device=device)
            phase = alternating_phase(getattr(self, '_batch_idx', 0), bool(_single_all.any()), bool((~_single_all).any()))
            if phase == 'none':
                return {}

        sums, cnts = defaultdict(float), defaultdict(float)
        n_tf_drop = 0
        n_cens_lo = int((cens_all < 0).sum().item())
        n_cens_hi = int((cens_all > 0).sum().item())

        for start in range(0, B, mb):
            # Drop the PREVIOUS micro-batch's tensors before this one's forward allocates. manual_backward frees a
            # graph's saved activations, so this matters only for a micro-batch that produced no loss and therefore
            # ran no backward: its graph then stays alive through these names and peak memory doubles. That happens
            # once the WT path is frozen (--freeze_wt_after_epoch / --wt_freeze_after_decays) or when a unit's
            # masks come up empty. Fixed on devel as acf6026, where a lossless unit pinned ~20 GB.
            wt_out = wt_pred_cal = wt_pred_raw = mt_out = mt_pred_cal = mt_pred_raw = None
            p_obs = p_wt = p_comb = base_wt_cal = base_wt_raw = forced_wt_cal = forced_wt_raw = None
            combined_pred_raw = combined_pred_cal = total_wt = total_mt = L_rank = term = None
            wt_losses = mt_losses = None

            idx = torch.arange(start, min(start + mb, B), device=device)
            micro, w_mb = utils.slice_batch_by_index(batch, idx), w_all[idx]
            ddG_mb = micro['ddG'].float() 
            dGwt_mb, floor_mb, link_ok = dGwt_all[idx], floor_all[idx], link_ok_all[idx]
            cens_mb, bound_mb = cens_all[idx], bound_all[idx]
            n_mut_mb = micro['mut_mask'].sum(dim=1)
            is_single = n_mut_mb == 1
            reg_mask = ~is_single if self.hparams.mt_reg_mask == 'doubles' else torch.ones_like(is_single, dtype=torch.bool)
            
            # --wt_double_target: the two self-consistent pairings for a multi-mutant's WT-path target.
            #
            #   'linked' - the WT pass's LLR sum through the link, against the item's own measured ddG_AB. The link
            #              turns the additive latent into the observed ddG the double would have under pure
            #              saturation, which is a measured quantity. Additivity is still enforced by the
            #              architecture (the pass can only emit a sum of per-position terms), not by the label.
            #   'latent' - the unlinked LLR sum against the sum of the measured singles, ddG_A + ddG_B.
            #
            # Mixing them is the one broken combination: a sum of two observed singles is not anything the readout
            # can report (two singles at -2.5 give -5.0, while a linked prediction floors near y_floor - dG_wt), so
            # a linked prediction could never match it and the gradient would drive the latent down without limit.
            wt_double_linked = self.hparams.wt_double_target == 'linked'
            wt_targets = ddG_mb.clone()
            valid_wt_mask = torch.isfinite(ddG_mb)
            if not wt_double_linked:
                add = micro['ddG_additive'].float() if 'ddG_additive' in micro else torch.full_like(ddG_mb, float('nan'))
                multi = ~is_single
                wt_targets = torch.where(multi, add, wt_targets)
                valid_wt_mask = valid_wt_mask & (is_single | torch.isfinite(add))

            # --no-wt_trains_doubles: leave multi-mutants to the MT head entirely. They still teacher-force the
            # WT half of the combined prediction, so the WT path still SHAPES a double's prediction - it just
            # takes no gradient from one. The rank mask already drops them when the target is not 'linked'.
            if not getattr(self.hparams, 'wt_trains_doubles', True):
                valid_wt_mask = valid_wt_mask & is_single

            # Teacher forcing for ENSEMBLE mode: the measured additive expectation replaces the WT half of the
            # combined latent, so the MT head only has to learn the correction.
            tf_additive_labels = torch.full_like(ddG_mb, float('nan'))
            if 'ddG_additive' in micro:
                tf_additive_labels = micro['ddG_additive'].clone().float()
            tf_additive_labels[is_single] = ddG_mb[is_single]
            has_add_label = ~torch.isnan(tf_additive_labels)

            # --tf_delink: the label is on the OBSERVED scale and it is about to be added to a LATENT prediction.
            # Under --combined_mode sum that mismatch enters at full weight rather than half, so put the label on
            # the latent scale first. The inverse is NONLINEAR, so a double's label is de-linked ONE SINGLE AT A
            # TIME and then summed - de-linking ddG_A + ddG_B as a sum is a different and wrong number. Detached:
            # a label must never train the link it was inverted through. A single at or below its library's floor
            # has no invertible latent value and returns NaN, which drops it from teacher forcing - the same
            # population tf_saturated below removes, now by construction rather than by threshold.
            if link is not None and getattr(self.hparams, 'tf_delink', False):
                lat = lambda o: link.latent_ddG(o.float(), dGwt_mb, floor_mb).detach()
                if 'ddG_A' in micro and 'ddG_B' in micro:
                    tf_lat = lat(micro['ddG_A']) + lat(micro['ddG_B'])
                else:
                    tf_lat = torch.full_like(ddG_mb, float('nan'))
                tf_additive_labels = torch.where(is_single, lat(ddG_mb), tf_lat)
                has_add_label = torch.isfinite(tf_additive_labels) & link_ok

            # ... but only where that measured sum is a usable stand-in for the LATENT sum. A single measured at its
            # library's floor reports the floor, not its own effect, so injecting it into a latent average biases the
            # combined prediction by however much that single was compressed. Such doubles fall back to the WT head's
            # own latent.
            if 'ddG_A' in micro and 'ddG_B' in micro:
                a_floored = (dGwt_mb + micro['ddG_A'].float()) <= floor_mb
                b_floored = (dGwt_mb + micro['ddG_B'].float()) <= floor_mb
                tf_saturated = (a_floored | b_floored) & ~is_single
                n_tf_drop += int(tf_saturated.sum().item())
                has_add_label = has_add_label & ~tf_saturated

            # The WT half of the combined prediction is always detached (--no-detach_ensemble_input is retired),
            # so the WT graph never has to survive its own backward. That also drops a retain_graph.
            retain_wt = False

            # =============================================================
            # PHASE 1: WILD-TYPE PASS
            # =============================================================
            # On an 'mt' turn the WT half of the combined prediction comes entirely from the teacher-forcing
            # label (the MT losses are restricted to labelled items below), so the WT forward is not needed and
            # is skipped - that is where the saving in --alternate_heads comes from. The zeros are never read.
            if phase == 'mt':
                wt_pred_cal = wt_pred_raw = torch.zeros_like(ddG_mb)
            else:
                wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=self.hparams.mask_strategy)
                wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()

            cens_wt_rank = cens_mb if wt_double_linked else torch.where(is_single, cens_mb, torch.zeros_like(cens_mb))
            wt_losses = []
            if self.hparams.lambda_reg_wt > 0:
                if link is not None:
                    # Assembled in latent (the pass's own sum over the item's positions), linked once, at the end.
                    # Under 'latent' a multi-mutant's target is itself a latent sum, so it stays off the link.
                    p_obs = link.obs_ddG(wt_pred_cal, dGwt_mb, floor_mb)
                    if wt_double_linked:
                        p_wt, reg_wt_mask = p_obs, valid_wt_mask & link_ok
                    else:
                        # Under 'latent' a multi-mutant's target is itself a latent sum, so it stays off the link
                        # and needs no dG_wt - hence it is kept even where link_ok is False.
                        p_wt = torch.where(is_single, p_obs, wt_pred_cal)
                        reg_wt_mask = valid_wt_mask & (link_ok | ~is_single)
                else:
                    p_wt, reg_wt_mask = wt_pred_cal, valid_wt_mask
                cens_wt = cens_mb if wt_double_linked else torch.where(is_single, cens_mb, torch.zeros_like(cens_mb))
                if reg_wt_mask.any():
                    term, val, wsum = self._reg_loss(p_wt, wt_targets, bound_mb, cens_wt, reg_wt_mask, w_mb, global_w_sum)
                    wt_losses.append(self.hparams.lambda_reg_wt * term)
                    sums['reg_wt'] += val; cnts['reg_wt'] += wsum

            if self.hparams.lambda_rank_wt > 0 and self.crit_rank_wt is not None:
                L_rank, val, n_list = self._compute_rank_loss(wt_pred_raw, wt_targets, valid_wt_mask, cens_wt_rank,
                                                              list_size, self.crit_rank_wt)
                if L_rank is not None:
                    wt_losses.append(self.hparams.lambda_rank_wt * L_rank * (n_list / global_num_lists))
                    sums['rank_wt'] += val * n_list; cnts['rank_wt'] += n_list

            if phase != 'mt' and wt_losses and not self.peft_manager.wt_path_is_frozen and (self.hparams.lambda_rank_wt > 0 or self.hparams.lambda_reg_wt > 0):
                total_wt = sum(wt_losses)
                if not torch.isfinite(total_wt): raise AssertionError("WT Loss evaluated to NaN/Inf.")
                if not total_wt.requires_grad: raise AssertionError("WT Loss detached from PyTorch Graph! Cannot call backward.")
                self.manual_backward(total_wt, retain_graph=retain_wt)
                    
            if phase != 'wt' and not self.peft_manager.mt_path_is_frozen and (self.hparams.lambda_rank_mt > 0 or self.hparams.lambda_reg_mt > 0):
                # =============================================================
                # PHASE 2: MUTANT & COMBINED PASS
                # =============================================================
                mt_out = self.model.forward_partitioned(micro, pass_type='mt', mask_strategy=self.hparams.mask_strategy)
                mt_pred_cal, mt_pred_raw = mt_out['pred_calibrated'].float(), mt_out['pred_raw'].float()
                
                base_wt_cal, base_wt_raw = wt_pred_cal.detach(), wt_pred_raw.detach()

                # Teacher forcing override for ENSEMBLE mode
                if self.model.lora_mode == 'ensemble':
                    if not has_add_label.any():
                        logging.warning("Ensemble mode teacher-forcing enabled, but no additive labels or single mutations are present in the batch.")

                    # Locate the appropriate calibration head
                    cal_head = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
                    if cal_head is None:
                        raise AssertionError("Ensemble mode failed: Could not locate 'calibration_head_wt' or 'calibration_head_fused' on self.model for de-calibration.")
                    
                    # Invert the calibration: raw = (cal - bias) / scale
                    s = 1.0 if not cal_head.use_scale else cal_head.scale.detach()
                    # bias_for, not cal_head.bias: with --per_mutation_bias the head added bias * n_mut, so a
                    # scalar inverse here would under-correct every multi-mutant and silently bias the label
                    # the MT head is teacher-forced against.
                    b = cal_head.bias_for(n_mut_mb)
                    b = b.detach() if torch.is_tensor(b) else b
                    decalibrated_label = (tf_additive_labels - b) / s
                    
                    forced_wt_cal = torch.where(has_add_label, tf_additive_labels, base_wt_cal)
                    forced_wt_raw = torch.where(has_add_label, decalibrated_label, base_wt_raw)

                    # --tf_mix: how much of the WT half of a MULTI-mutant's combined prediction comes from the
                    # measured additive label rather than from the WT head's own prediction. 1.0 (default) is the
                    # released behaviour - the label wherever one is usable; 0.0 never teacher-forces; 0.5 is the
                    # arithmetic mean of the two on the calibrated scale. The label carries the measurement noise
                    # of its two singles (and the floor clipping of either), the WT head is smooth but biased, so
                    # the mix trades variance for bias. An item with no usable label is unaffected for any mix
                    # (both terms of the blend are then the WT pass), and SINGLES are never mixed: a single's
                    # forced half is its own measurement, which is also its target.
                    _tf_mix = self.hparams.get('tf_mix', 1.0)          # not `or 1.0`: 0.0 is a legal, meaningful mix
                    tf_mix = 1.0 if _tf_mix is None else float(_tf_mix)
                    if tf_mix < 1.0:
                        w_cal = torch.where(is_single, torch.ones_like(forced_wt_cal),
                                            torch.full_like(forced_wt_cal, tf_mix))
                        forced_wt_cal = w_cal * forced_wt_cal + (1.0 - w_cal) * base_wt_cal
                        forced_wt_raw = w_cal * forced_wt_raw + (1.0 - w_cal) * base_wt_raw
                elif self.model.lora_mode == 'corrector':
                    forced_wt_cal, forced_wt_raw = base_wt_cal, base_wt_raw
                else:
                    raise AssertionError(f"Unknown lora_mode: {self.model.lora_mode}")

                cmode = getattr(self.hparams, 'combined_mode', None) or 'mean'
                # Under 'route' a SINGLE's combined prediction is the WT branch alone, and teacher forcing has
                # already replaced that branch with the measured label (tf_additive_labels[is_single] = ddG). The
                # single's loss is then identically zero and carries no gradient - harmless only because
                # --mt_reg_mask doubles masks singles out of the combined losses entirely. Paired with 'all' it
                # would silently train on nothing but doubles while appearing to train on everything.
                if cmode == 'route' and self.hparams.mt_reg_mask != 'doubles':
                    raise AssertionError("--combined_mode route requires --mt_reg_mask doubles: a single's "
                                         "combined prediction is the teacher-forced label itself, so its loss is "
                                         "identically zero and must be masked out rather than silently ignored.")
                if phase == 'mt':
                    # no WT forward was run, so only items whose WT half is a label can be scored
                    reg_mask = reg_mask & has_add_label
                combined_pred_raw = combine_predictions(forced_wt_raw, mt_pred_raw, cmode, is_single)
                combined_pred_cal = combine_predictions(forced_wt_cal, mt_pred_cal, cmode, is_single)

                ns = getattr(self, 'ns_head', None)
                if ns is not None:
                    # Nonspecific epistasis f(x) on multi-mutants only: x = dG_wt + the additive latent the forcing uses (the
                    # forced WT half; detached). Where that half is the WT pass of the multi-mutant, it carries the calibration
                    # bias ONCE per item - add one bias per extra mutation. Under --per_mutation_bias the head already added
                    # bias * n_mut, so there is nothing left to add; where the half is a measured additive label (teacher
                    # forcing) there is no bias in it at all. Detached: f never trains the WT half; its own gradient comes from
                    # the combined losses, which is also why it cannot leak into the WT adapter.
                    ch = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
                    s_ns = 1.0 if ch is None or not ch.use_scale else ch.scale.detach()
                    b_ns = 0.0 if ch is None or ch.bias is None or getattr(self.hparams, 'per_mutation_bias', False) else ch.bias.detach()
                    from_pass = ~has_add_label if self.model.lora_mode == 'ensemble' else torch.ones_like(is_single)
                    x_hat = (dGwt_mb + forced_wt_cal
                             + torch.where(from_pass, b_ns * (n_mut_mb - 1.0), torch.zeros_like(forced_wt_cal))).detach()
                    f_ns = torch.where((n_mut_mb > 1) & link_ok, ns(x_hat), torch.zeros_like(x_hat))
                    # Added after the combination, so the MT readout keeps 2 f: comb = (WT + MT) / 2 still holds.
                    combined_pred_cal = combined_pred_cal + f_ns
                    combined_pred_raw = combined_pred_raw + f_ns / s_ns

                mt_losses = []

                # Mut pass strictly relies on combined losses
                if self.hparams.lambda_reg_mt > 0:
                    if link is not None:
                        p_comb = link.obs_ddG(combined_pred_cal, dGwt_mb, floor_mb)
                        reg_c_mask = reg_mask & link_ok
                    else:
                        p_comb, reg_c_mask = combined_pred_cal, reg_mask
                    if reg_c_mask.any():
                        term, val, wsum = self._reg_loss(p_comb, ddG_mb, bound_mb, cens_mb, reg_c_mask, w_mb, global_w_sum)
                        mt_losses.append(self.hparams.lambda_reg_mt * term)
                        sums['reg_combined'] += val; cnts['reg_combined'] += wsum

                if self.hparams.lambda_rank_mt > 0 and self.crit_rank_mt is not None:
                    L_rank, val, n_list = self._compute_rank_loss(combined_pred_raw, ddG_mb, reg_mask, cens_mb,
                                                                  list_size, self.crit_rank_mt)
                    if L_rank is not None:
                        mt_losses.append(self.hparams.lambda_rank_mt * L_rank * (n_list / global_num_lists))
                        sums['rank_combined'] += val * n_list; cnts['rank_combined'] += n_list

                if mt_losses:
                    total_mt = sum(mt_losses)
                    if not torch.isfinite(total_mt): raise AssertionError("MT Loss evaluated to NaN/Inf.")
                    if not total_mt.requires_grad: raise AssertionError("MT Loss detached from PyTorch Graph! Cannot call backward.")
                    self.manual_backward(total_mt)

        logs = {}
        for k in ('reg_wt', 'rank_wt', 'reg_combined', 'rank_combined'):
            if cnts[k] > 0: logs[f'L_{k}'] = sums[k] / max(1, cnts[k])
        self._cens_diag = (n_cens_lo, n_cens_hi, n_tf_drop)
        return logs
    
    def _log_lrs(self):
        opts = self.trainer.optimizers
        for opt in opts:
            for i, g in enumerate(opt.param_groups):
                name = g.get("name", f"group{i}")
                self.log(f"lr/{name}", float(g["lr"]), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def log_calibration_head(self, on_step=True, on_epoch=False, prog_bar=False, logger=True):
        head_suffixes = ["wt", "mt", "fused", "shared"]

        for suffix in head_suffixes:
            head = getattr(self.model, f"calibration_head_{suffix}", None)
            if head is None:
                continue
                
            for param in ("scale", "bias"):
                val = getattr(head, param, None)
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    try:
                        self.log(
                            f"calibration_heads/{suffix}_{param}",
                            val.detach().mean().item() if val.numel() > 1 else val.detach().item(),
                            on_step=on_step,
                            on_epoch=on_epoch,
                            prog_bar=prog_bar,
                            logger=logger,
                        )
                    except Exception:
                        # never break logging loop; skip malformed tensors
                        continue

    def training_step(self, batch: dict, batch_idx: int):
        self._batch_idx = int(batch_idx)
        self.model.train()
        logs = self._compose_losses_streaming_and_backward(utils._normalize_batch(batch))

        # The temperature anchor is a penalty on two parameters, not a per-item loss, so it is applied once per
        # optimizer step rather than inside the micro-batch loop (where it would be counted once per micro-batch).
        if self.link_head is not None and float(self.hparams.get('link_tau_penalty', 0.0) or 0.0) > 0:
            pen = self.link_head.tau_penalty(float(self.hparams.link_tau_penalty))
            if pen.requires_grad:
                self.manual_backward(pen)
                logs['link_tau_penalty'] = float(pen.detach())

        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=False)

        optim = self.optimizers()
        
        max_norm = getattr(self.hparams, "grad_clip_norm", None)
        if max_norm and max_norm > 0:
            self.clip_gradients(optim, gradient_clip_val=max_norm, gradient_clip_algorithm="norm")
            
        param_id_to_name = {id(p): name for name, p in self.model.named_parameters()}
        for i, g in enumerate(optim.param_groups):
            group_name = g.get("name", f"group{i}")
            named_params = [(param_id_to_name.get(id(p), f"param_{j}"), p) for j, p in enumerate(g["params"])]
            
            w_norm = utils.l2_weight_norm(named_params)
            g_norm = utils.l2_grad_norm(named_params)
            s_norm = utils.group_step_norm(named_params, float(g["lr"]))
            
            self.log(f"norm_weight/{group_name}", w_norm, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log(f"norm_grad/{group_name}", g_norm, on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log(f"norm_step/{group_name}", s_norm, on_step=True, on_epoch=False, logger=True, sync_dist=True)

        optim.step()
        optim.zero_grad(set_to_none=True)
        
        sch_warmup = self.lr_schedulers()   # one config, so Lightning hands back the scheduler itself
        total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
        if sch_warmup.last_epoch < total_warmup_steps:
            sch_warmup.step()

        self._log_lrs()
        self.log_calibration_head(on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)

        if getattr(self, '_cens_diag', None) is not None:
            self.log("train/cens_lower_items", float(self._cens_diag[0]), on_step=True)
            self.log("train/cens_upper_items", float(self._cens_diag[1]), on_step=True)
            self.log("train/tf_saturated_dropped", float(self._cens_diag[2]), on_step=True)

        if getattr(self, 'ns_head', None) is not None:
            # The learned curve, so it can be watched as it forms during a run (the same points are in the val dumps as ns_term).
            with torch.no_grad():
                for xv in (-1.0, 0.0, 1.0, 2.0, 3.0):
                    self.log(f"ns/f_at_x{xv:+.0f}",
                             float(self.ns_head(torch.tensor([xv], device=self.ns_head.values.device))[0]), on_step=True)
            
        if torch.cuda.is_available():
            gb = 1.0 / (1024 ** 3)
            self.log("mem/allocated_gb", torch.cuda.memory_allocated() * gb, on_step=True)
            self.log("mem/reserved_gb", torch.cuda.memory_reserved() * gb, on_step=True, prog_bar=True)
            self.log("mem/peak_allocated_gb", torch.cuda.max_memory_allocated() * gb, on_step=True)

        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self.log("amp_scale", self.trainer.precision_plugin.scaler.get_scale(), on_step=True)

        return torch.tensor(0.0, device=self.device)

    def _apply_nonspecific(self, out_dict, batch, dG_wt):
        """With --nonspecific_term: a multi-mutant's combined prediction gains f(x) and its MT readout 2 f(x) (so comb = (WT + MT) / 2
        still holds), x = dG_wt + the WT pass (the additive latent) + one calibration bias per extra mutation - none when the head
        already applies one per mutated position (--per_mutation_bias). Singles and items without dG_wt are untouched."""
        ns = getattr(self, 'ns_head', None)
        if ns is None:
            return out_dict
        dev = out_dict['combined_pred'].device
        mm = batch['mut_mask'].to(dev)
        n_mut = mm.sum(dim=1).float()
        dgw = dG_wt.to(dev).float()
        ok = (n_mut > 1) & torch.isfinite(dgw)
        ch = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
        b = 0.0 if ch is None or ch.bias is None or getattr(self.hparams, 'per_mutation_bias', False) else ch.bias.detach()
        with torch.no_grad():
            x_hat = torch.where(ok, dgw, torch.zeros_like(dgw)) + out_dict['wt_lora_pred'].float() + b * (n_mut - 1.0)
            f = torch.where(ok, ns(x_hat), torch.zeros_like(x_hat))
        out = dict(out_dict)
        out['combined_pred'] = out_dict['combined_pred'].float() + f
        out['mt_lora_pred'] = out_dict['mt_lora_pred'].float() + 2.0 * f
        out['epi_pred'] = out_dict['epi_pred'].float() + f
        return out

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        wt_only = bool(self.hparams.get('val_wt_only', False))
        if wt_only and not getattr(self, '_warned_val_wt_only', False):
            self._warned_val_wt_only = True
            logging.info("validation: WT head only (--val_wt_only) - no MT forward and no MT cycle pass; the mt and "
                         "comb columns report the WT head")
        if wt_only:
            # The MT adapter is untrained in this run (WT-only sweep, MT frozen at init), so an MT forward is a pass
            # through noise. Report the WT head under all three names rather than leave the columns undefined or fill
            # them with that noise: mt == comb == wt is then a true statement about this model.
            with torch.inference_mode():
                w = self.model.forward_partitioned(batch, pass_type='wt',
                                                   mask_strategy=self.hparams.mask_strategy)['pred_calibrated'].float()
            out_dict = {'wt_lora_pred': w, 'mt_lora_pred': w, 'combined_pred': w,
                        'epi_pred': torch.zeros_like(w)}
        else:
            with torch.inference_mode():
                out_dict = self.model.forward_batch(batch, mask_strategy=self.hparams.mask_strategy)

        B = int(out_dict['wt_lora_pred'].shape[0])
        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        # --nonspecific_term: before anything else reads the predictions, so the observed-scale block below and every metric
        # downstream see the term exactly as training does.
        dG_wt = batch['dG_wt'].float() if torch.is_tensor(batch.get('dG_wt')) else torch.full((B,), float('nan'))
        out_dict = self._apply_nonspecific(out_dict, batch, dG_wt)
        nan_col = lambda: np.full(B, np.nan, dtype=np.float32)
        get = lambda k: batch[k].detach().cpu().float().numpy() if torch.is_tensor(batch.get(k)) else nan_col()

        # The epistasis metrics need the LATENT scores and apply the link themselves. The observed scale is carried
        # separately, for the rmse metrics only, so they stay comparable with a run that has no link.
        obs = {}
        if self.link_head is not None and torch.is_tensor(batch.get('dG_wt')):
            dgw, foff = batch['dG_wt'].float(), batch['y_floor'].float()
            ok = torch.isfinite(dgw)
            z = torch.where(ok, dgw, torch.zeros_like(dgw))
            with torch.inference_mode():
                for key, src in (('wt', out_dict['wt_lora_pred']), ('comb', out_dict['combined_pred'])):
                    v = src.float()
                    obs[key] = torch.where(ok, self.link_head.obs_ddG(v, z, foff), v).detach().cpu().numpy()

        cyc = {}
        if self.hparams.val_cycle_passes:
            with torch.inference_mode():
                c = self.model.forward_cycle(batch, mask_strategy=self.hparams.mask_strategy,
                                             include_mt=not wt_only)
            cyc = {'wt_rev_scores': c['wt_rev'].detach().cpu().float().numpy()}
            if 'mt_fwd' in c:
                cyc['mt_fwd_scores'] = c['mt_fwd'].detach().cpu().float().numpy()

        rec = {
            **cyc,
            'wt_scores': out_dict['wt_lora_pred'].detach().cpu().float().numpy(),
            'mt_scores': out_dict['mt_lora_pred'].detach().cpu().float().numpy(),
            'comb_scores': out_dict['combined_pred'].detach().cpu().float().numpy(),
            'ground_truths': ddG.detach().cpu().float().numpy() if ddG is not None else nan_col(),
            'dddG_truths': get('dddG'),
            'subset_types': list(batch.get('subset_type', ['single'] * B)),
            'cens': batch['cens'].detach().cpu().numpy() if torch.is_tensor(batch.get('cens')) else np.zeros(B, dtype=np.int64),
            'cens_src': (batch['cens_src'].detach().cpu().numpy() if torch.is_tensor(batch.get('cens_src'))
                         else np.zeros(B, dtype=np.int64)),
            'dG_wt': get('dG_wt'),
            'y_floor': get('y_floor'),
            'pdb': list(batch.get('pdb', [])),
            'mutations': list(batch.get('mutations', [])),
        }
        if obs:
            rec['obs'] = obs
        self.validation_step_outputs[dataloader_idx].append(rec)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = defaultdict(list)

    _VAL_PROGRESS_BAR = ('rho_combined', 'rho_wt_valid')

    @staticmethod
    def _mut_keys(pdb, mutations):
        """
        One hashable key per item: a tuple of ``(library, position, mutant residue)`` per mutation.

        ``epi_metrics`` pairs a double with its two singles by these keys and requires the library to be part of
        them, so that identical mutations in different libraries are never matched up.
        """
        keys = []
        for i, m in enumerate(mutations):
            lib = pdb[i] if isinstance(pdb, (list, tuple)) and i < len(pdb) else (pdb if isinstance(pdb, str) else 'unknown')
            parts = m.split(':') if isinstance(m, str) else m
            try:
                if parts and isinstance(parts[0], (list, tuple)):          # (wt, pos, mt) triples
                    keys.append(tuple((str(lib), int(q[1]), str(q[2])) for q in parts))
                else:                                                      # 'A12G' strings
                    keys.append(tuple((str(lib), int(str(q)[1:-1]), str(q)[-1]) for q in parts))
            except (ValueError, IndexError, TypeError):
                keys.append(None)
        return keys

    def on_validation_epoch_end(self):
        avg_metrics = defaultdict(list)
        pooled = defaultdict(list)

        for dataloader_idx, valid_outputs in self.validation_step_outputs.items():
            name = (self.val_dataloader_names[dataloader_idx] if dataloader_idx < len(self.val_dataloader_names)
                    else f"unknown_dl_{dataloader_idx}")

            cat = lambda key: (np.concatenate([np.atleast_1d(np.asarray(o[key])) for o in valid_outputs])
                               if valid_outputs else np.array([]))
            wt, mt, comb = cat('wt_scores'), cat('mt_scores'), cat('comb_scores')
            gt, cens, dddG = cat('ground_truths'), cat('cens'), cat('dddG_truths')
            cens_src = cat('cens_src')
            subset_types = [t for o in valid_outputs for t in o['subset_types']]
            obs = None
            if valid_outputs and 'obs' in valid_outputs[0]:
                obs = {k: np.concatenate([o['obs'][k] for o in valid_outputs]) for k in valid_outputs[0]['obs']}

            if not len(gt):
                continue

            # Score the CANONICAL population: every item the assay actually fitted, censored or not. Floor- and
            # gap-censored items are scored like any other, because censoring belongs in the loss and excluding
            # them makes the numbers incomparable to a run that censors nothing. The rows the assay only bounded
            # (--censor_range) are held back instead: their label IS their bound and is a library constant, so
            # scoring them would compare predictions against a flat line. They are still dumped in full below.
            canon = cens_src != censoring.SRC_RANGE
            sub = lambda a: np.asarray(a)[canon]
            metrics = stats.compute_metrics(
                sub(wt), sub(mt), sub(comb), sub(gt), [t for t, k in zip(subset_types, canon) if k],
                obs=None if obs is None else {k: sub(v) for k, v in obs.items()})
            is_external = name in self.external_val_sets
            for metric, val in metrics.items():
                if val is not None and np.isfinite(val):
                    self.log(f"val_{metric}/{name}", float(val), on_epoch=True, sync_dist=True)
                    if not is_external:
                        avg_metrics[metric].append(float(val))
            if is_external:
                # OOD benchmark: scored and logged above, but out of every aggregate - avg_metrics, the pooled global
                # ranking and the epistasis hierarchy - and dropped before the pooling below. Only its own rho/rmse
                # summary survives (summarise_external_benchmarks).
                ext_per_loader[name] = {k: metrics[k] for k in EXTERNAL_SUMMARY_METRICS if k in metrics}
                continue

            # Pool for the global ranking and for the epistasis hierarchy, which needs every library at once.
            keys = [k for o in valid_outputs for k in self._mut_keys(o['pdb'], o['mutations'])]
            pooled['mut_key'].extend(keys)
            pooled['subset_type'].extend(subset_types)
            pooled['loader'].extend([name] * len(keys))
            pooled['n_mut'].extend([len(k) if k else 0 for k in keys])
            for field, arr in (('wt_scores', wt), ('mt_scores', mt), ('comb_scores', comb),
                               ('ground_truths', gt), ('dddG', dddG), ('cens', cens), ('cens_src', cens_src),
                               ('dG_wt', cat('dG_wt')), ('y_floor', cat('y_floor'))):
                pooled[field].extend(np.asarray(arr).tolist())
            for field in ('wt_rev_scores', 'mt_fwd_scores'):
                have = valid_outputs and field in valid_outputs[0]
                pooled[field].extend(cat(field).tolist() if have else [float('nan')] * len(keys))

        for key, values in avg_metrics.items():
            if values:
                self.log(f'val_{key}_avg', float(np.nanmean(values)), on_epoch=True,
                         prog_bar=key in self._VAL_PROGRESS_BAR, sync_dist=True)

        # OOD benchmarks, as their own section: mean rho/rmse over the external loaders, never inside an aggregate.
        for _k, _v in summarise_external_benchmarks(ext_per_loader, self.external_val_sets).items():
            self.log(_k, _v, on_epoch=True, sync_dist=True)

        # Pooled across every library at once: one global ranking, which punishes a model whose per-library scales
        # disagree even when each library's own ordering is fine.
        if pooled['ground_truths']:
            g_wt = np.array(pooled['wt_scores']); g_comb = np.array(pooled['comb_scores'])
            g_gt = np.array(pooled['ground_truths']); g_src = np.array(pooled['cens_src'])
            ok = np.isfinite(g_gt) & (g_src != censoring.SRC_RANGE)
            if ok.any():
                self.log("val_rho_ungrouped_wt", stats.safe_spearman(g_wt[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
                self.log("val_rho_ungrouped_combined", stats.safe_spearman(g_comb[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
                self.log("val_ndcg@k=96_ungrouped_combined", stats.safe_ndcg_k96(g_comb[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
                self.log("val_ndcg@k>0_ungrouped_combined", stats.safe_ndcg_t0(g_comb[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
            # The epistasis scalars are pooled over every library at once, so they are not per-library values and
            # never entered avg_metrics. Merge them in as single-element lists AFTER the val_<key>_avg logging loop
            # above - so nothing is logged twice - and the plateau steps can then read any metric by name.
            for _k, _v in (self._log_epistasis(pooled) or {}).items():
                if np.isfinite(_v):
                    avg_metrics[_k].append(float(_v))

        if self.link_head is not None:
            for k, v in self.link_head.summary().items():
                self.log(f"link/{k}", v, on_epoch=True, sync_dist=True)

        if not self.trainer.sanity_checking and self.trainer.current_epoch==self.hparams.freeze_wt_after_epoch and not self.peft_manager.has_transitioned:
            logging.info(f"Epoch {self.hparams.freeze_wt_after_epoch} ended! Transitioning to MT training.")
            self._transition_to_mt()

        if not self.trainer.sanity_checking:
            total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
            # After warmup only: during warmup LambdaLR rewrites every group's lr each step and would
            # overwrite these decays.
            if self.trainer.global_step >= total_warmup_steps:
                self._wt_plateau_step(avg_metrics)
                self._mt_plateau_step(avg_metrics)
                self._stop_if_nothing_is_training()

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _log_epistasis(self, pooled):
        """
        Logs ``val_epi_<metric>_<head>`` (``esm_msr.epi_metrics``: every metric for comb, mt and ctx, the naive and
        global ones also for the additive control add), pooled over every validation library, and with --val_dump
        writes ``val_dump_<tag>.npz`` for offline rescoring.

        Heads: comb ((WT + ~MT)/2, the reported prediction), mt (~MT), ctx ((WT + ~WT)/2, needs --val_cycle_passes),
        add (WT, additive by construction - the control for what saturation alone can earn). With --link the metrics
        put each head on the observed scale through the run's link, bending at each library's own floor. Never raises.
        """
        try:
            keys = pooled['mut_key']
            if not keys or all(k is None for k in keys):
                return {}
            f = lambda name: np.asarray(pooled[name], dtype=float)
            n_items = len(keys)
            for field in ('dG_wt', 'y_floor', 'ground_truths', 'dddG', 'wt_scores', 'mt_scores', 'comb_scores',
                          'cens', 'cens_src'):
                got = len(np.atleast_1d(np.asarray(pooled.get(field, []))))
                if got != n_items:    # a field on_validation_epoch_end failed to pool: fail loudly, never mis-score quietly
                    raise ValueError(f"pooled '{field}' has {got} values for {n_items} items")
            items = {'mut_key': keys, 'subset_type': list(pooled['subset_type']), 'cens': f('cens').astype(int),
                     'cens_src': f('cens_src').astype(int),
                     'ddG': f('ground_truths'), 'dddG': f('dddG'), 'dG_wt': f('dG_wt'), 'y_floor': f('y_floor'),
                     'wt': f('wt_scores'), 'mt': f('mt_scores'), 'comb': f('comb_scores'), 'wt_rev': f('wt_rev_scores')}

            log_dir = getattr(self.logger, 'log_dir', None) or getattr(self.logger, 'save_dir', None)
            if self.hparams.val_dump and log_dir:
                tag = 'zs' if self.trainer.sanity_checking or self.global_step == 0 else f'e{self.current_epoch}'
                os.makedirs(log_dir, exist_ok=True)
                extra = {'link_summary': np.array(json.dumps(self.link_head.summary()))} if self.link_head is not None else {}
                if getattr(self, 'ns_head', None) is not None:
                    extra['ns_term'] = np.array(json.dumps(self.ns_head.table()))
                np.savez_compressed(
                    os.path.join(log_dir, f'val_dump_{tag}.npz'),
                    wt=items['wt'], mt=items['mt'], comb=items['comb'], wt_rev=items['wt_rev'],
                    mt_fwd=f('mt_fwd_scores'), dddG=items['dddG'], ground_truth=items['ddG'],
                    dG_wt=items['dG_wt'], y_floor=items['y_floor'], cens=items['cens'],
                    cens_src=items['cens_src'],
                    n_mut=np.asarray(pooled['n_mut']), subset_type=np.asarray(items['subset_type'], dtype=str),
                    loader=np.asarray(pooled['loader']),
                    mut_key=np.asarray([json.dumps(k) for k in keys]), **extra)

            # The dump above carries every mutant, so any of this can be rescored offline. The metrics are
            # computed on the canonical population only - the same one the per-library metrics use - so that a
            # run with --censor_range stays directly comparable to one without.
            canon = items['cens_src'] != censoring.SRC_RANGE
            scored = {k: ([v[i] for i in np.flatnonzero(canon)] if isinstance(v, list) else np.asarray(v)[canon])
                      for k, v in items.items()}
            link = self.link_head.numpy() if self.link_head is not None else None
            res = epi_metrics.compute(epi_metrics.Table(scored, link))
            for name, val in res.items():
                if np.isfinite(val):
                    self.log(f"val_{name}", float(val), on_epoch=True,
                             prog_bar=name == 'epi_cell_rank_mt', sync_dist=True)
            return res                  # the caller merges these into avg_metrics so a plateau can watch them
        except Exception as e:
            logging.warning(f"epistasis metrics skipped: {e}", exc_info=True)
        return {}

    def _wt_plateau_step(self, avg_metrics: dict) -> None:
        """
        Decay the WT path's own learning rate when ITS validation metric stops improving, and optionally freeze it.

        The shared ReduceLROnPlateau this replaced watched ``val_rho_combined_avg``, which the MT path keeps
        improving long after the WT head has stopped - ``rho_wt_valid`` typically peaks around epoch 2 while the
        combined metric climbs for several more. That meant the WT adapter kept taking full-rate steps on a metric
        it was no longer improving, so it drifted (weakly overfitting), and the target the MT path teacher-forces
        against moved underneath it. Keyed on the WT metric, both stop. (That shared scheduler is gone entirely
        now; see ``configure_optimizers``.)

        Only the ``lora_wt`` and ``calib_wt`` groups are touched, so the MT path and the link are unaffected.
        """
        patience = int(getattr(self.hparams, 'wt_plateau_patience', 0) or 0)
        if patience <= 0 or self.peft_manager.has_transitioned:
            return
        key = self.hparams.wt_convergence_metric
        if not avg_metrics.get(key):
            # Loudly, and once: a scheduler that is configured but silently inert is worse than one that errors.
            # dc12 ran 8 epochs with its MT plateau doing nothing, because _log_epistasis returned None and the
            # epistasis scalars never reached avg_metrics - and not one line said so.
            if not getattr(self, '_warned_no_wt_metric', False):
                self._warned_no_wt_metric = True
                logging.warning(f"--wt_plateau_patience is set but '{key}' is not among the validation "
                                f"metrics {sorted(avg_metrics)}, so the WT plateau will never fire.")
            return
        current = float(np.nanmean(avg_metrics[key]))
        if not np.isfinite(current):
            return

        best = getattr(self, '_wt_plateau_best', -float('inf'))
        if current > best + 1e-4:
            self._wt_plateau_best, self._wt_plateau_bad = current, 0
            return
        self._wt_plateau_bad = getattr(self, '_wt_plateau_bad', 0) + 1
        if self._wt_plateau_bad < patience:
            return

        self._wt_plateau_bad = 0
        factor, min_lr = float(self.hparams.wt_plateau_factor), float(self.hparams.wt_plateau_min_lr)
        touched = []
        for g in self.optimizers().param_groups:
            if g.get('name') in ('lora_wt', 'calib_wt'):
                new = max(min_lr, float(g['lr']) * factor)
                touched.append(f"{g['name']}: {float(g['lr']):.2e} -> {new:.2e}")
                g['lr'] = new
        self._wt_decays = getattr(self, '_wt_decays', 0) + 1
        logging.info(f"WT plateau on {key} (best={best:.4f}, now={current:.4f}), decay #{self._wt_decays}: "
                     f"{', '.join(touched)}")

        after = int(getattr(self.hparams, 'wt_freeze_after_decays', 0) or 0)
        if after and self._wt_decays >= after:
            logging.info(f"WT path frozen after {self._wt_decays} decays; its predictions are now fixed, so the "
                         f"teacher-forced target the MT path uses stops moving.")
            self._transition_to_mt()

    def _mt_plateau_step(self, avg_metrics: dict) -> None:
        """
        Decay the MT path's own learning rate when ITS metric stops improving. The mirror of _wt_plateau_step, and
        it touches only the ``lora_mt`` and ``calib_mt`` groups, so the WT path and the link are unaffected.

        Keyed by default on ``epi_beyond_rho_mt``, the MT head's correlation on the part of a double that is not
        explained by the two singles - what the MT path exists to learn. Measured across dc6/dc7/dc8_s1 it peaks
        early and then falls back without regaining the peak: dc6 peaked at epoch 4 (+0.5395) and ended -0.0190
        lower, dc8 peaked at epoch 3 (+0.5331) and ended -0.0079 lower.

        BUT it is noisy epoch to epoch, with swings of +-0.03 and no trend, and dc7_s1 is a counter-example: it
        peaked at its LAST epoch (+0.5367, the highest value of the three runs) after dipping twice on the way. A
        patience of 1 would have decayed it at epoch 4 and cost it that. So patience 1 wants a gentler
        --mt_plateau_factor than the WT path's 0.1, or patience 2; see docs/devel_conservative.md.
        """
        patience = int(getattr(self.hparams, 'mt_plateau_patience', 0) or 0)
        if patience <= 0 or self.peft_manager.mt_path_is_frozen:
            return
        key = getattr(self.hparams, 'mt_convergence_metric', None) or 'epi_beyond_rho_mt'
        if not avg_metrics.get(key):
            # Loudly, and once: a scheduler that is configured but silently inert is worse than one that errors.
            # dc12 ran 8 epochs with its MT plateau doing nothing, because _log_epistasis returned None and the
            # epistasis scalars never reached avg_metrics - and not one line said so.
            if not getattr(self, '_warned_no_mt_metric', False):
                self._warned_no_mt_metric = True
                logging.warning(f"--mt_plateau_patience is set but '{key}' is not among the validation "
                                f"metrics {sorted(avg_metrics)}, so the MT plateau will never fire.")
            return
        current = float(np.nanmean(avg_metrics[key]))
        if not np.isfinite(current):
            return

        best = getattr(self, '_mt_plateau_best', -float('inf'))
        if current > best + 1e-4:
            self._mt_plateau_best, self._mt_plateau_bad = current, 0
            return
        self._mt_plateau_bad = getattr(self, '_mt_plateau_bad', 0) + 1
        if self._mt_plateau_bad < patience:
            return

        self._mt_plateau_bad = 0
        factor, min_lr = float(self.hparams.mt_plateau_factor), float(self.hparams.mt_plateau_min_lr)
        touched = []
        for g in self.optimizers().param_groups:
            if g.get('name') in ('lora_mt', 'calib_mt'):
                new = max(min_lr, float(g['lr']) * factor)
                touched.append(f"{g['name']}: {float(g['lr']):.2e} -> {new:.2e}")
                g['lr'] = new
        self._mt_decays = getattr(self, '_mt_decays', 0) + 1
        logging.info(f"MT plateau on {key} (best={best:.4f}, now={current:.4f}), decay #{self._mt_decays}: "
                     f"{', '.join(touched)}")

        after = int(getattr(self.hparams, 'mt_freeze_after_decays', 0) or 0)
        if after and self._mt_decays >= after:
            logging.info(f"MT path frozen after {self._mt_decays} decays.")
            self.peft_manager.freeze_mt_components()
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def _stop_if_nothing_is_training(self) -> None:
        """
        End the run once every optimizer group that still holds trainable parameters is at lr 0.

        With --wt_freeze_after_decays and --mt_freeze_after_decays both set, each head stops itself after its own
        decay budget, and the MT head is normally last (rho_wt_valid peaks around epoch 2 while beyond_rho_mt
        climbs for several more). The remaining epochs would re-run the same weights for the same score, so the
        epoch budget can be set generously and is spent only where it is used.

        Keyed on the learning rates rather than requires_grad because ``on_validation_end`` re-applies the
        baseline requires_grad every epoch, while freezing zeroes the group's lr for good
        (``enforce_freezing(zero_lrs=True)``).
        """
        opts = self.optimizers()
        opts = opts if isinstance(opts, list) else [opts]
        for opt in opts:
            for g in getattr(opt, 'optimizer', opt).param_groups:
                if float(g.get('lr', 0.0)) > 0.0 and any(p.requires_grad for p in g['params']):
                    return
        if not self.trainer.should_stop:
            logging.info(f"Every trainable group is at lr 0 after epoch {self.trainer.current_epoch} "
                         f"(WT frozen: {self.peft_manager.wt_path_is_frozen}, "
                         f"MT frozen: {self.peft_manager.mt_path_is_frozen}); stopping early.")
            self.trainer.should_stop = True

    def _transition_to_mt(self):
        """Freeze the WT path and open the MT path, saving the converged WT weights first."""
        self._save_converged_wt_weights()
        self.peft_manager.freeze_wt_components()
        self.peft_manager.unfreeze_mt_components()
        self.peft_manager.has_transitioned = True
        self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def on_validation_end(self):
        self.peft_manager.apply_baseline_requires_grad()

    def configure_optimizers(self):
        base_lr = float(self.hparams.learning_rate)
        wd = float(getattr(self.hparams, "weight_decay", 0.0))
        
        lora_wt_params, lora_mt_params, other_params, calib_wt_params, calib_mt_params = [], [], [], [], []
        adapter_mode = getattr(self.model, 'adapter_mode', 'dual')
        
        # Grab the robust strings injected by PEFT
        wt_name = getattr(self.model, 'wt_adapter_name', 'wt_adapter').lower()
        mt_name = getattr(self.model, 'mt_adapter_name', 'mt_adapter').lower()

        seen = set()
        for name, p in self.model.named_parameters():
            if id(p) in seen: continue
            
            # Build the SAME groups whatever happens to be frozen right now, keying on what arrived trainable.
            # A currently-frozen parameter still gets its group: enforce_freezing zeroes that group's lr and
            # weight_decay and clears the grads, so the group is inert.
            #
            # Keying on the CURRENT freeze state is what broke --resume_from_checkpoint. on_load_checkpoint
            # re-freezes the WT path, and Lightning calls it BEFORE configure_optimizers, so the optimizer came
            # out with 3 groups where the checkpoint had 5 and the restore died with "loaded state dict has a
            # different number of parameter groups". For a run that freezes nothing at construction - every dc
            # run - this selects exactly the same parameters as before.
            if name not in self.peft_manager.initially_trainable:
                continue

            seen.add(id(p))
            lname = name.lower()
            
            if "calibration" in lname:
                if "mt" in lname: calib_mt_params.append(p)
                else: calib_wt_params.append(p)
            elif mt_name in lname and adapter_mode == 'dual': lora_mt_params.append(p)
            elif wt_name in lname or "default" in lname: lora_wt_params.append(p)
            else: other_params.append(p)

        main_groups = []
        if lora_wt_params: main_groups.append({"params": lora_wt_params, "lr": base_lr, "weight_decay": wd, "name": "lora_wt"})
        if lora_mt_params: main_groups.append({"params": lora_mt_params, "lr": base_lr, "weight_decay": wd, "name": "lora_mt"})
        if calib_wt_params: main_groups.append({"params": calib_wt_params, "lr": base_lr * self.hparams.calib_lr_mult, "weight_decay": 0.0, "name": "calib_wt"})
        if calib_mt_params: main_groups.append({"params": calib_mt_params, "lr": base_lr * self.hparams.calib_lr_mult, "weight_decay": 0.0, "name": "calib_mt"})
        if other_params: main_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd, "name": "other"})
        # The link lives on the LightningModule, not inside self.model, so the loop above never sees it.
        if self.link_head is not None:
            link_params = [p for p in self.link_head.parameters() if p.requires_grad]
            if link_params:
                main_groups.append({"params": link_params, "lr": float(self.hparams.link_lr),
                                    "weight_decay": 0.0, "name": "link"})
        # f(x) likewise lives on the LightningModule. Its own group with no warm-up and no plateau decay, like the link's.
        if getattr(self, 'ns_head', None) is not None:
            main_groups.append({"params": list(self.ns_head.parameters()), "lr": float(self.hparams.get('ns_lr', 5e-3)),
                                "weight_decay": 0.0, "name": "nonspecific"})

        if not main_groups:
            logging.warning("No param groups found; defaulting to all trainables.")
            main_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd, "name": "all"}]

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=True, weight_decay=wd)

        lambdas_main = []
        for g in main_groups:
            if "calib" in g["name"] or g["name"] in ("link", "nonspecific"): lambdas_main.append(lambda step: 1.0)
            elif g["name"] == "lora_mt":
                lambdas_main.append(lambda step, delay=self.hparams.mt_lora_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))
            else:
                lambdas_main.append(lambda step, delay=self.hparams.calib_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))

        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        # Warmup is the only shared scheduler. The ReduceLROnPlateau that used to sit here watched
        # val_rho_combined_avg and owned EVERY group in opt_main, so once --wt_plateau_patience and
        # --mt_plateau_patience gave each head a schedule on its own metric it became a second, redundant
        # decay on both of them, triggered by a metric neither head is being tuned against. Measured on
        # dc12r_s2: lora_mt went 2e-4 -> 6e-5 (MT decay 1) -> 1.8e-5 (MT decay 2) -> 1.8e-6 (this scheduler,
        # on rho_combined wobbling 0.8398 -> 0.8392 -> 0.8393) -> 5.4e-7 (MT decay 3). A factor of 370 in
        # three epochs, so --mt_plateau_factor 0.3 was never the schedule the MT head actually got.
        #
        # The 'other' and 'link' groups consequently hold their warmed-up lr for the whole run: this was
        # their only decay, and there is no metric specific to them to plateau on.
        return [opt_main], [{"scheduler": warmup_main, "interval": "step", "frequency": 1}]

    def _load_wt_weights(self, path):
        """Copy the WT path (adapter + calibration head) out of a checkpoint, in place.

        Accepts either a full Lightning checkpoint (the queue's arms: ``checkpoint['state_dict']``, keys prefixed
        ``model.``) or a ``converged_wt_*.pt`` file written by ``_save_converged_wt_weights`` (``{'state_dict': ...}``,
        bare model-parameter names). Only parameters named by ``peft_manager.wt_parameter_names()`` are touched, so
        the MT adapter, the link and the nonspecific term are never silently overwritten: the link and the term are
        configured by their own flags (an old learned link would otherwise be smuggled in), and the MT path must
        start fresh. Raises when nothing matches, because a path typo must not quietly train a random WT head.
        """
        obj = torch.load(path, map_location='cpu')
        src = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
        if not isinstance(src, dict):
            raise RuntimeError(f"--init_wt_from_checkpoint: {path} does not contain a state dict")
        self._init_wt_src = src            # re-used by _load_link_weights; the file is 225 MB, read it once
        current = {n: p for n, p in self.model.named_parameters()}
        loaded, missing, mismatched = [], [], []
        with torch.no_grad():
            for name in self.peft_manager.wt_parameter_names():
                key = name if name in src else ('model.' + name if ('model.' + name) in src else None)
                if key is None:
                    missing.append(name)
                    continue
                value = src[key]
                if not torch.is_tensor(value) or tuple(value.shape) != tuple(current[name].shape):
                    mismatched.append(name)
                    continue
                current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
                loaded.append(name)
        if not loaded:
            raise RuntimeError(f"--init_wt_from_checkpoint: no WT parameter of {path} matched this model "
                               f"({len(missing)} names missing, {len(mismatched)} with a different shape). "
                               f"Expected keys like 'model.peft_wt...' / 'calibration_head_wt...'.")
        logging.info(f"WT path warm-started from {path}: {len(loaded)} tensors loaded"
                     + (f", {len(missing)} name(s) not present" if missing else "")
                     + (f", {len(mismatched)} shape mismatch(es) skipped ({', '.join(mismatched[:3])}...)" if mismatched else ""))
        return loaded

    def _freeze_link(self):
        """--freeze_link: nothing in the link trains, because the link is the last stage of the pre-trained WT model."""
        for p in self.link_head.parameters():
            p.requires_grad = False
        logging.info("Link frozen (--freeze_link): its parameters are part of the pre-trained WT model")

    def _load_link_weights(self, path):
        """Copy the link (``link_head.*``) out of the same checkpoint the WT adapter came from.

        The link is the last stage of the WT readout (latent -> observed ddG), so a frozen WT model without its link
        is not the model that was measured: its floors, temperature and global floor all shape the observed-scale
        prediction the WT loss and the rmse guard are computed from. A checkpoint that has no ``link_head.*`` keys
        (the released model, or a ``converged_wt_*.pt``) leaves the link as its own flags configured it - that is
        logged, not an error.
        """
        src = getattr(self, '_init_wt_src', None)
        if src is None:
            obj = torch.load(path, map_location='cpu')
            src = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
            if not isinstance(src, dict):
                return []
        targets = dict(self.link_head.named_parameters())
        targets.update(self.link_head.named_buffers())
        loaded = []
        with torch.no_grad():
            for name, tensor in targets.items():
                key = 'link_head.' + name
                if key not in src:
                    continue
                value = src[key]
                if torch.is_tensor(value) and tuple(value.shape) == tuple(tensor.shape):
                    tensor.copy_(value.to(device=tensor.device, dtype=tensor.dtype))
                    loaded.append(name)
        if loaded:
            logging.info(f"Link warm-started from {path}: {len(loaded)}/{len(targets)} tensor(s) ({', '.join(loaded)})")
        else:
            logging.info(f"Link NOT loaded from {path}: no link_head.* keys in it (the link stays as its flags set it)")
        return loaded

    def _save_converged_wt_weights(self):
        """Extracts and saves only the WT adapter and calibration head at the moment of convergence."""
        
        logging.info("Saving converged WT adapter weights before transition...")
        wt_state_dict = {}
        
        for name, param in self.model.named_parameters():
            # Grab components specific to the WT pass (works for both dual and fused modes)
            if any(x in name for x in ['peft_wt', 'wt_adapter', 'calibration_head_wt', 'default', 'calibration_head_fused']):
                wt_state_dict[name] = param.detach().cpu()
                
        if wt_state_dict:
            save_dir = self.trainer.default_root_dir if self.trainer.default_root_dir else "."
            if self.logger and hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                save_dir = self.logger.log_dir
            elif self.logger and hasattr(self.logger, 'save_dir') and self.logger.save_dir:
                save_dir = self.logger.save_dir
                
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"converged_wt_epoch_{self.current_epoch}_step_{self.global_step}.pt")
            
            # Wrap in 'state_dict' key to match standard PyTorch Lightning load formats
            torch.save({'state_dict': wt_state_dict}, save_path)
            logging.info(f"Successfully saved {len(wt_state_dict)} WT tensors to: {save_path}")
        else:
            logging.warning("Attempted to save converged WT weights, but found 0 matching tensors.")
    
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Intercepts the checkpoint before it saves to disk and strips out 
        the frozen ESM3 backbone, saving only the trainable PEFT parameters.
        """
        state_dict = checkpoint.get('state_dict', {})
        
        # 'link_head' matters: its parameters (global_floor, raw_tau, and the ceiling when enabled) define the
        # observed scale, so a checkpoint without them cannot reproduce the predictions the run was scored on, and
        # link.numpy_link_from_state_dict - which exists for exactly that - returns None for every checkpoint.
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'lora' in k.lower() or 'calibration' in k.lower() or 'link_head' in k.lower() or 'ns_head' in k.lower()
        }
        
        if not filtered_state_dict:
            logging.warning("Checkpoint filtering caught an empty state_dict. Check parameter naming.")
            
        checkpoint['state_dict'] = filtered_state_dict
        
        # Save the transition state from the PEFT manager so it persists across preemptions
        checkpoint['transition_state'] = {
            'has_transitioned': getattr(self.peft_manager, 'has_transitioned', False),
            # _wt_plateau_step's own counters. Without these a resume restarts the WT plateau from scratch, so a
            # run preempted mid-decay would decay again from a reset patience and follow a different schedule.
            # Harmless once the WT path has frozen (the step returns early on has_transitioned), but not before.
            'wt_plateau_best': getattr(self, '_wt_plateau_best', -float('inf')),
            'wt_plateau_bad': getattr(self, '_wt_plateau_bad', 0),
            'wt_decays': getattr(self, '_wt_decays', 0),
            'mt_plateau_best': getattr(self, '_mt_plateau_best', -float('inf')),
            'mt_plateau_bad': getattr(self, '_mt_plateau_bad', 0),
            'mt_decays': getattr(self, '_mt_decays', 0),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Restores the custom transition state and re-enforces parameter freezing
        if the model was preempted after the WT validation convergence trigger.
        """
        transition_state = checkpoint.get('transition_state', {})
        self.peft_manager.has_transitioned = transition_state.get('has_transitioned', False)
        self._wt_plateau_best = transition_state.get('wt_plateau_best', -float('inf'))
        self._wt_plateau_bad = transition_state.get('wt_plateau_bad', 0)
        self._wt_decays = transition_state.get('wt_decays', 0)
        self._mt_plateau_best = transition_state.get('mt_plateau_best', -float('inf'))
        self._mt_plateau_bad = transition_state.get('mt_plateau_bad', 0)
        self._mt_decays = transition_state.get('mt_decays', 0)
        
        # Re-apply freezing states if we resumed after convergence
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter:
            self.peft_manager.freeze_wt_components()

        if self.peft_manager.has_transitioned:
            self.peft_manager.unfreeze_mt_components()
     
        if self.hparams.freeze_mt_adapter:
            self.peft_manager.freeze_mt_components()


class OfflineCometLogger(CometLogger):
    """CometLogger whose experiment is written to disk (COMET_OFFLINE_DIRECTORY) for `comet upload` later.

    Lightning's own ``online=False`` is not that: it sets ``disabled=True`` and the logger silently discards everything.
    """
    def _create_experiment(self) -> None:
        import comet_ml
        self._experiment = comet_ml.start(api_key=self._api_key, workspace=self._workspace, project=self._project_name,
                                          online=False, experiment_config=comet_ml.ExperimentConfig(**self._kwargs))
        self._experiment_key = self._experiment.get_key()
        self._project_name = self._experiment.project_name


def format_validation_results(results) -> str:
    """
    The metrics of one validation, once, from ``trainer.validate``'s per-dataloader list of logged dicts.

    Lightning hands every dataloader a copy of everything logged in ``on_validation_epoch_end`` (all of this module's validation metrics)
    and prints one column per dataloader, so with one loader per protein each value appeared once per protein. Here the dicts are merged
    (dropping Lightning's ``/dataloader_idx_N`` suffixes): per-protein metrics (``val_<metric>/<protein>``) become one table with a row per
    protein, and everything else a sorted list of name / value.
    """
    merged = {}
    for d in results or []:
        for k, v in d.items():
            merged.setdefault(k.split('/dataloader_idx_')[0], float(v.item() if hasattr(v, 'item') else v))
    per_protein, scalars = defaultdict(dict), {}
    for k, v in merged.items():
        if '/' in k:
            metric, protein = k.split('/', 1)
            per_protein[protein][metric] = v
        else:
            scalars[k] = v
    lines = []
    if per_protein:
        metrics = sorted({m for row in per_protein.values() for m in row})
        width = max(len('protein'), *(len(p) for p in per_protein))
        lines.append('  '.join(['protein'.ljust(width)] + [m.rjust(max(len(m), 8)) for m in metrics]))
        for protein in sorted(per_protein):
            row = per_protein[protein]
            lines.append('  '.join([protein.ljust(width)] + [(f'{row[m]:.4f}' if m in row else '').rjust(max(len(m), 8)) for m in metrics]))
        lines.append('')
    if scalars:
        width = max(len(k) for k in scalars)
        lines += [f'{k.ljust(width)}  {scalars[k]:.4f}' for k in sorted(scalars)]
    return '\n'.join(lines)


def main():
    args = parse_arguments()
    pl.seed_everything(args.seed)

    if args.offline_model:
        os.environ['INFRA_PROVIDER'] = "1"

    try:
        tokenizer = EsmSequenceTokenizer("cpu")
        structure_encoder = ESM3_structure_encoder_v0("cpu")
    except Exception as e:
        logging.error(f"Failed to load base model for tokenizer/encoder: {e}. Exiting.")
        return

    train_loaders, val_loaders, bench_loaders, train_names, val_names, bench_names = setup_dataloaders(args, tokenizer, structure_encoder, add_benchmarks_to_val=True)

    if torch.cuda.is_available() and args.gpus > 0:
         accelerator, devices, model_device, strategy = "gpu", args.gpus, 'cuda:0', args.strategy if args.gpus > 1 else 'auto'
    else:
         accelerator, devices, model_device, strategy = "cpu", 1, 'cpu', 'auto'
         
    try:
        lightning_model = ESM3EpistasisLightningModule(
            **vars(args), train_dataloader_names=train_names, val_dataloader_names=val_names, tokenizer=tokenizer, model_device=model_device
        )
    except Exception as e:
         logging.error(f"Failed to initialize Lightning Module: {e}.", exc_info=True)
         return

    if args.load_lora_checkpoint:
        try:
            lightning_model.model.load_lora_weights(args.load_lora_checkpoint, load_wt_only=args.load_wt_only)
        except AttributeError:
            raise NotImplementedError("load_lora_weights is not implemented on MSRModel.")
            
    loggers = []
    if args.log_dir:
        csv_logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name, version=args.version)
        loggers.append(csv_logger)
        checkpoint_dir = csv_logger.log_dir if hasattr(csv_logger, 'log_dir') else os.path.join(args.log_dir, args.experiment_name, csv_logger.version)
    else:
        checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment_name, args.version or "default_version")

    if args.comet_api_key:
        comet_kwargs = dict(api_key=args.comet_api_key, project=args.comet_project_name,
                            name=f"{args.experiment_name}-{args.version or 'run'}")
        if getattr(args, 'comet_experiment_key', None):
            # Comet has no notion of resuming on its own: without this a resumed run opens a SECOND
            # experiment, and the run's curves end up split across two of them.
            comet_kwargs.update(experiment_key=args.comet_experiment_key, mode='get_or_create')
        if args.comet_offline:
            os.environ['COMET_OFFLINE_DIRECTORY'] = os.path.abspath(os.path.join(args.log_dir or '.', 'comet_offline'))
            comet_logger = OfflineCometLogger(**comet_kwargs)
        else:
            comet_logger = CometLogger(**comet_kwargs)
        loggers.append(comet_logger)

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename=args.checkpoint_filename, save_top_k=args.save_top_k, monitor=args.monitor_metric, mode=args.monitor_mode, save_last=True),
        TQDMProgressBar(refresh_rate=min(10, args.log_every_n_steps))
    ]

    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=args.early_stopping_metric, patience=args.early_stopping_patience, mode=args.monitor_mode, verbose=True))

    trainer_kwargs = {
        "max_epochs": args.num_epochs, "accelerator": accelerator, "devices": devices, "strategy": strategy,
        "logger": loggers if loggers else False, "callbacks": callbacks, "enable_checkpointing": True, 
        "num_sanity_val_steps": args.num_sanity_val_steps, "log_every_n_steps": args.log_every_n_steps, "check_val_every_n_epoch": args.check_val_every_n_epoch,
    }

    if args.precision == "16-mixed":
        trainer_kwargs["plugins"] = [MixedPrecisionPlugin(precision="16-mixed", device=model_device, scaler=torch.cuda.amp.GradScaler(init_scale=1024.0))]
    else:
        trainer_kwargs["precision"] = args.precision

    try:
        trainer = Trainer(**trainer_kwargs)
    except Exception as e:
        logging.error(f"Trainer init failed: {e}", exc_info=True)
        return

    if not args.skip_val:
        # verbose=False: Lightning's own table repeats every epoch-level metric under every dataloader (one per protein), so the
        # zero-shot validation is printed once here instead
        results = trainer.validate(lightning_model, dataloaders=val_loaders, verbose=False)
        if trainer.is_global_zero:
            logging.info("Zero-shot validation:\n" + format_validation_results(results))
    ckpt = getattr(args, 'resume_from_checkpoint', None)
    if ckpt:
        logging.info(f"Resuming from {ckpt}")
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders, ckpt_path=ckpt)
    if args.comet_offline and args.comet_api_key and trainer.is_global_zero:
        comet_logger.experiment.end()  # writes the archive to upload with `comet upload`

if __name__ == "__main__":
    main()