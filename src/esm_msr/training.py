import json
import itertools
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

from esm_msr.models import MSRModel, combine_predictions, pass_mask_strategy
from esm_msr import censoring, epi_metrics, stats, utils
from esm_msr.link import SaturatingLink
from esm_msr.losses import ListMLELoss, hinge_regression_loss
from esm_msr.preprocess_megascale import setup_dataloaders
from esm_msr.peft_manager import PEFTStateManager
from esm_msr.config import parse_arguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high')

# The LoRA spec that is no longer a flag. inference.parse_hparams_to_lora_config defaults the same values for a checkpoint whose
# hparams do not name them, so nothing needs recording.
LORA_CONSTANTS = dict(target_mode='expanded', last_n_layers=0, incl_structure_encoder=False, incl_sequence_head=True,
                      use_dora=False, unfreeze_layernorms=False)
SHARED_SCALE_INIT, SHARED_BIAS_INIT = 0.3, 0.0       # the calibration heads' starting scale and bias
CALIB_LR_MULT = 20.0                                  # the calibration heads learn this much faster than the adapters
SAVE_TOP_K = 3
# command-line values that say where things are, not what the run is: kept out of hparams.yaml (and so out of Comet)
UNLOGGED_ARGS = ('log_dir', 'cache_path', 'comet', 'comet_project_name', 'comet_offline', 'comet_experiment_key')


def build_link(hp) -> SaturatingLink:
    """The observed-scale link of a run: one floor and one knee width, both set by flags and neither learned."""
    return SaturatingLink(tau=hp.link_tau, floor=hp.link_floor)


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


def teacher_forcing_unusable(dG_wt, ddG_A, ddG_B, cut):
    """Multi-mutants whose measured additive label cannot stand in for the latent sum: a single read at or below ``cut``
    (the library floor, the learned global floor, or --censor_threshold) reports where the assay saturates, not its own
    effect, and de-linking it is undefined at the floor and explosively noisy just above it."""
    return ((dG_wt + ddG_A) <= cut) | ((dG_wt + ddG_B) <= cut)


def reachable_bounds(link, bound, y_floor, offset):
    """Censoring bounds (on each item's own ddG scale) moved inside what the link can report; see the comment at the call site.

    Raised to the knee's reading h(floor) = floor + tau*ln2, so a lower-censored item's hinge holds exactly when its latent is at or below
    the floor. ``offset`` = ddG - dG_meas converts absolute dG to the item's scale. Detached: a bound never trains the link it comes from.
    """
    knee_ddG = link(link.effective_floor(y_floor), y_floor).detach() + offset
    # Only where computable: torch.maximum propagates NaN, which would destroy a good bound.
    return torch.where(torch.isfinite(knee_ddG), torch.maximum(bound, knee_ddG), bound)


class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(self, val_dataloader_names=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['val_dataloader_names'])
        hp = self.hparams

        lora = lambda side: {"lora_rank": hp[f'lora_rank_{side}'], "lora_alpha": hp[f'lora_alpha_{side}'],
                             "lora_dropout": hp[f'lora_dropout_{side}'], "seed": hp.seed,
                             "target_mode": hp.get(f'target_mode_{side}', LORA_CONSTANTS['target_mode']),
                             **{k: v for k, v in LORA_CONSTANTS.items() if k != 'target_mode'}}
        lora_config = {'wt_config': lora('wt'), 'mt_config': lora('mt'), 'seed': hp.seed}

        self.model = MSRModel(
            lora_config=lora_config, shared_scale_init=SHARED_SCALE_INIT, shared_bias_init=SHARED_BIAS_INIT, adapter_mode=hp.adapter_mode,
            model_dtype=torch.float32, per_mutation_bias=True, tie_mt_scale=bool(hp.tie_mt_scale),
            native_cond_mt_input=bool(hp.native_cond_for_mt),
            mask_native_cond_struct=bool(hp.get('mask_native_cond_struct', False)),
            fused_antisymmetric=bool(hp.get('fused_antisymmetric', False)),
        )

        self.peft_manager = PEFTStateManager(self.model)
        # --init_wt_from_checkpoint: copy a pre-trained WT model in BEFORE anything freezes it, so that
        # --frozen_wt_mt_only holds exactly the loaded model fixed.
        if hp.get('init_wt_from_checkpoint'):
            self._load_wt_weights(hp.init_wt_from_checkpoint)
        # on_save_checkpoint deliberately keeps only the lora / calibration / link_head entries, so a
        # checkpoint is missing every frozen base weight. Lightning's default strict load would then refuse
        # to resume from our own checkpoints; the base comes from the pretrained weights either way.
        self.strict_loading = False

        if hp.freeze_wt_adapter: self.peft_manager.freeze_wt_components()
        if hp.freeze_mt_adapter: self.peft_manager.freeze_mt_components()

        self.crit_rank_wt = ListMLELoss() if hp.lambda_rank_wt > 0 else None
        self.crit_rank_mt = ListMLELoss() if hp.lambda_rank_mt > 0 else None
        self.crit_reg = nn.HuberLoss(reduction='none', delta=float(hp.huber_delta)) if hp.huber_delta > 0 else nn.MSELoss(reduction='none')

        # The observed-scale link. Regression goes through it; ranking does not, because it is monotone. Nothing in it is learned.
        self.link_head = build_link(hp)
        logging.info(f"Observed-scale link: {self.link_head.summary()}")
        # The link is part of the pre-trained WT model, not a separate piece: a frozen WT adapter paired with a different floor or
        # width is a different model than the one being frozen.
        if hp.get('init_wt_from_checkpoint'):
            self._load_link_weights(hp.init_wt_from_checkpoint)

        # sequence masking, per pass (models.pass_mask_strategy)
        self.mask_wt, self.mask_mt = pass_mask_strategy(hp, 'wt'), pass_mask_strategy(hp, 'mt')
        self.mask_strategy = {'wt': self.mask_wt, 'mt': self.mask_mt}
        if self.mask_wt or self.mask_mt:
            logging.info(f"sequence masking: WT pass {self.mask_wt or 'none'}, MT pass {self.mask_mt or 'none'}")

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = list(val_dataloader_names) if val_dataloader_names else ['val']
        self.external_val_sets = {s.strip() for s in str(hp.get('external_val_sets', '') or '').split(',') if s.strip()}
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
        peft_wt = getattr(self.model, 'peft_wt', None) or getattr(self.model, 'peft_fused', None)    # fused mode has the one adapter
        if peft_wt is None:
            return
        try:
            if 'bf16' not in str(self.trainer.precision):       # the CPU smoke runs in fp32 (ESM_MSR_PRECISION=32); production is bf16-mixed
                return
        except RuntimeError:
            pass
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
            self._cast_frozen_linears_bf16()      # single-adapter (fused) mode: nothing to re-share, but the frozen base is still stored bf16
            return
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
        per_list = torch.stack([crit_fn.forward_censored(
            pred_rank[i:i + 1], targ_rank[i:i + 1], mask_rank[i:i + 1], cens_rank[i:i + 1]
        ) for i in range(num_lists)])
        n_valid = mask_rank.float().sum(dim=-1).clamp(min=1.0)
        scaled_loss = (per_list * (list_size / n_valid)).mean()
        return scaled_loss, per_list.detach().mean().item(), num_lists

    def _compute_partner_rank_loss(self, pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                   cens: torch.Tensor, list_size: int, crit_fn: ListMLELoss,
                                   micro: dict, epi_pred: torch.Tensor | None = None,
                                   epi_targ: torch.Tensor | None = None, epi_mask: torch.Tensor | None = None):
        """
        Contrastive within-partner / within-pair ListMLE for the MT path:

        1. Orders the micro-batch rows so doubles sharing a position pair ``(pos_0, pos_1)`` or primary partner
           mutation ``(pos_0, mt_0)`` land in the same ``list_size`` slice before calling :meth:`_compute_rank_loss`.
        2. Forms explicit fixed-partner and fixed-pair epistatic-residual lists from doubles in the micro-batch
           that share a partner allele ``(pos, mt_id)`` or position pair ``(pos_0, pos_1)``, plus the micro-batch's
           epistatic-residual list over ``epi_mask``, directly optimizing ``cell_rank`` and ``epi_beyond_rho``.
        """
        mut_pos = micro.get('mut_pos')
        mt_id = micro.get('mt_id')
        mut_mask = micro.get('mut_mask')
        if mut_pos is not None and mt_id is not None and mut_mask is not None and pred.shape[0] > 1:
            pos_cpu = mut_pos.detach().cpu().tolist()
            mt_cpu = mt_id.detach().cpu().tolist()
            mm_cpu = mut_mask.detach().cpu().tolist()
            keys = []
            for r in range(pred.shape[0]):
                v = [(pos_cpu[r][j], mt_cpu[r][j]) for j, m in enumerate(mm_cpu[r]) if m]
                v.sort()
                is_dbl = 0 if len(v) >= 2 else 1
                p0 = v[0][0] if len(v) >= 1 else 999999
                m0 = v[0][1] if len(v) >= 1 else 999999
                p1 = v[1][0] if len(v) >= 2 else 999999
                m1 = v[1][1] if len(v) >= 2 else 999999
                keys.append((is_dbl, p0, m0, p1, m1, r))
            keys.sort()
            perm = torch.tensor([k[-1] for k in keys], dtype=torch.long, device=pred.device)
            base_loss, base_val, n_list = self._compute_rank_loss(
                pred[perm], targets[perm], mask[perm], cens[perm], list_size, crit_fn
            )
        else:
            base_loss, base_val, n_list = self._compute_rank_loss(
                pred, targets, mask, cens, list_size, crit_fn
            )

        if base_loss is None or epi_pred is None or epi_targ is None or epi_mask is None:
            return base_loss, base_val, n_list

        valid_dbl = torch.where(epi_mask)[0]
        if valid_dbl.numel() < 2:
            return base_loss, base_val, n_list

        groups = []
        if mut_pos is not None and mt_id is not None and mut_mask is not None:
            by_partner = defaultdict(list)
            by_pair = defaultdict(list)
            dbl_list = valid_dbl.detach().cpu().tolist()
            for r in dbl_list:
                v = sorted((pos_cpu[r][j], mt_cpu[r][j]) for j, m in enumerate(mm_cpu[r]) if m)
                for allele in v:
                    by_partner[allele].append(r)
                if len(v) >= 2:
                    by_pair[(v[0][0], v[1][0])].append(r)
            for g in itertools.chain(by_pair.values(), by_partner.values()):
                if len(g) >= 2:
                    groups.append(g[:list_size])

        dbl_all = valid_dbl.detach().cpu().tolist()
        for st in range(0, len(dbl_all), list_size):
            chunk = dbl_all[st:st + list_size]
            if len(chunk) >= 2:
                groups.append(chunk)

        if not groups:
            return base_loss, base_val, n_list

        epi_losses = []
        for g in groups:
            g_idx = torch.tensor(g, dtype=torch.long, device=pred.device)
            k = g_idx.numel()
            p_g = epi_pred[g_idx].unsqueeze(0)
            t_g = epi_targ[g_idx].unsqueeze(0)
            m_g = torch.ones((1, k), dtype=torch.bool, device=pred.device)
            c_g = cens[g_idx].unsqueeze(0)
            l_g = crit_fn.forward_censored(p_g, t_g, m_g, c_g)
            epi_losses.append(l_g * (list_size / float(k)))

        epi_loss_mean = torch.stack(epi_losses).mean()
        comb_loss = 0.5 * base_loss + 0.5 * epi_loss_mean
        return comb_loss, float(comb_loss.detach().item()), n_list

    @staticmethod
    def _micro_batch(B: int, list_size: int, hard_cap: int) -> int:
        """
        How many items to put through one forward/backward: ``--micro_batch_size``, rounded down to a multiple of ``list_size`` (the rank
        loss reshapes the micro batch into lists of exactly that size) and no more than the batch. Activation memory scales with
        micro batch x sequence length, so the right cap depends on the card: 64 costs about 23 GB at the median training length of 49 on
        a 32 GB one. Exceeding the card does not raise OOM under WSL2 - the driver silently spills to host memory over PCIe - so the only
        symptom is throughput collapsing.
        """
        n = max(1, int(hard_cap))
        n = max(list_size, (n // list_size) * list_size)      # whole lists only
        return min(B, n)

    def _reg_loss(self, pred, target, bound, cens, mask, denom):
        """
        Regression over ``mask``, one-sided wherever the measurement is only a bound.

        Returns ``(loss_term, summed_value, count)``: the sum of the per-item losses over ``denom`` (the whole batch's size, so that the
        micro-batches add up to the batch's mean whatever ``--micro_batch_size`` is), the same sum on its own, and the number of items.
        A censored item is scored against ``bound`` and only when the prediction sits on the wrong side of it
        (``losses.hinge_regression_loss``), so a variant the assay pinned at its floor cannot be pushed to a value it never had.
        """
        if mask.dtype != torch.bool:
            raise AssertionError(f"regression mask must be bool, got {mask.dtype}: an integer mask indexes rows "
                                 f"instead of selecting them, which fails silently.")
        L = hinge_regression_loss(self.crit_reg, pred[mask], target[mask], bound[mask], cens[mask])
        return L.sum() / denom, float(L.sum().item()), float(L.numel())

    def _compose_losses_streaming_and_backward(self, batch: dict) -> dict:
        hp = self.hparams
        device, B = batch['ddG'].device, int(batch['ddG'].shape[0])
        list_size = max(1, int(hp.rank_list_size))
        mb = self._micro_batch(B, list_size, hp.micro_batch_size)

        st_all = batch.get('subset_type', ['single'] * B)
        is_native_cond_all = torch.as_tensor([t == 'native_cond' for t in st_all], device=device)
        global_num_lists = max(1, B // list_size)

        # Regression on the observed scale: observed ddG = h(dG_wt + latent) - dG_wt, with the knee at the item's floor. An item whose
        # library has no measured dG_wt (the external benchmarks, which this assay never measured) cannot be put on that scale, so it
        # leaves the linked regressions but keeps its rank terms, which the link would not have changed anyway.
        link = self.link_head
        dGwt_all = batch.get('dG_wt', torch.full((B,), float('nan'), device=device)).float()
        floor_all = batch.get('y_floor', torch.full((B,), float('nan'), device=device)).float()
        link_ok_all = torch.isfinite(dGwt_all)
        cens_all = batch.get('cens', torch.zeros(B, dtype=torch.long, device=device)).long()
        bound_all = batch.get('cens_bound', torch.full((B,), float('nan'), device=device)).float()

        # A bound below what the link can even report is not a bound the model can honour. A softplus link only reaches the floor itself
        # as the latent goes to -inf, so a bound AT the floor is never satisfiable: every floor-censored item would keep a penalty and a
        # push toward -inf. The bound is raised to the link's KNEE, h(floor) = floor + tau*ln2 on the observed scale: the reading of a
        # variant whose latent dG sits exactly at the floor, i.e. half unfolded. There the hinge says exactly what the censoring means -
        # the latent is at or below the floor - and switches off once that holds. Detached, so a bound never trains the floor.
        # The floor is on the ITEM'S OWN ddG scale: the offset between an item's label and the absolute dG its measurement describes is
        # a constant of the item, the same shift that defines cens_bound.
        offset = batch.get('ddG').float() - batch.get('dG_meas', torch.full((B,), float('nan'), device=device)).float()
        bound_all = reachable_bounds(link, bound_all, floor_all, offset)

        sums, cnts = defaultdict(float), defaultdict(float)
        n_tf_drop = 0
        n_cens_lo = int((cens_all < 0).sum().item())
        n_cens_hi = int((cens_all > 0).sum().item())
        nc_for_wt, nc_for_mt = bool(hp.native_cond_for_wt), bool(hp.native_cond_for_mt)
        t_single = hp.get('censor_threshold')
        tf_mix = float(hp.tf_mix)
        rank_group_by = str(hp.get('rank_group_by', 'none') or 'none')

        for start in range(0, B, mb):
            # Drop the PREVIOUS micro-batch's tensors before this one's forward allocates. manual_backward frees a graph's saved
            # activations, so this matters only for a micro-batch that produced no loss and therefore ran no backward: its graph then
            # stays alive through these names and peak memory doubles. That happens once the WT path is frozen or when a unit's masks
            # come up empty. Fixed on devel as acf6026, where a lossless unit pinned ~20 GB.
            wt_out = wt_pred_cal = wt_pred_raw = mt_out = mt_pred_cal = mt_pred_raw = None
            p_obs = p_wt = p_comb = base_wt_cal = base_wt_raw = forced_wt_cal = forced_wt_raw = None
            combined_pred_raw = combined_pred_cal = total_wt = total_mt = L_rank = term = None
            wt_losses = mt_losses = None

            idx = torch.arange(start, min(start + mb, B), device=device)
            micro = utils.slice_batch_by_index(batch, idx)
            ddG_mb = micro['ddG'].float()
            dGwt_mb, floor_mb, link_ok = dGwt_all[idx], floor_all[idx], link_ok_all[idx]
            cens_mb, bound_mb = cens_all[idx], bound_all[idx]
            n_mut_mb = micro['mut_mask'].sum(dim=1)
            is_single = n_mut_mb == 1
            is_native_cond = is_native_cond_all[idx]
            denom = float(B)

            # native_cond routing is independent per head: with --native_cond_for_mt the mutant-background item joins the MT/combined
            # loss; without it, it is not an MT item at all.
            reg_mask = torch.ones_like(is_single, dtype=torch.bool) if nc_for_mt else ~is_native_cond

            # The WT head trains on singles only. Multi-mutants still teacher-force the WT half of the combined prediction below, so the
            # WT path still SHAPES a double's prediction - it just takes no gradient from one. ... and whether it trains on native_cond
            # is its own flag (the released default is on, via the 'single' label; with --native_cond_for_mt it is a native_cond item
            # and this mask decides).
            wt_targets = ddG_mb
            valid_wt_mask = torch.isfinite(ddG_mb) & is_single
            if not nc_for_wt:
                valid_wt_mask = valid_wt_mask & ~is_native_cond

            # Teacher forcing: the measured additive expectation replaces the WT half of the combined latent, so the MT head only has to
            # learn the correction. The label is on the OBSERVED scale and is about to be added to a LATENT prediction, so it is put on
            # the latent scale first. The inverse is NONLINEAR, so a double's label is de-linked ONE SINGLE AT A TIME and then summed -
            # de-linking ddG_A + ddG_B as a sum is a different and wrong number. Detached: a label must never train the link it was
            # inverted through. A single at or below the link's floor has no invertible latent value and returns NaN, which drops it from
            # teacher forcing.
            lat = lambda o: link.latent_ddG(o.float(), dGwt_mb, floor_mb).detach()
            if 'ddG_A' in micro and 'ddG_B' in micro:
                tf_lat = lat(micro['ddG_A']) + lat(micro['ddG_B'])
            else:
                tf_lat = torch.full_like(ddG_mb, float('nan'))
                if not getattr(self, '_warned_tf_delink_missing', False):
                    self._warned_tf_delink_missing = True
                    logging.warning("The batch has no ddG_A/ddG_B columns, so the additive label cannot be de-linked: teacher forcing is "
                                    "unavailable for this batch and the WT pass is used.")
            tf_additive_labels = torch.where(is_single, lat(ddG_mb), tf_lat)
            has_add_label = torch.isfinite(tf_additive_labels) & link_ok

            # ... but only where that measured sum is a usable stand-in for the LATENT sum. A single measured at the floor reports the
            # floor, not its own effect, so injecting it into a latent average biases the combined prediction by however much that single
            # was compressed. Such doubles fall back to the WT head's own latent. The cut is --censor_threshold, the same one that
            # censored the single, and a censored single gives no label for ITSELF either (its label is a bound).
            cut = torch.full_like(floor_mb, float(t_single)) if t_single is not None else floor_mb
            if 'ddG_A' in micro and 'ddG_B' in micro:
                tf_saturated = teacher_forcing_unusable(dGwt_mb, micro['ddG_A'].float(), micro['ddG_B'].float(), cut) & ~is_single
                n_tf_drop += int(tf_saturated.sum().item())
                has_add_label = has_add_label & ~tf_saturated
            if t_single is not None:
                has_add_label = has_add_label & ~(is_single & (cens_mb != 0))

            # =============================================================
            # PHASE 1: WILD-TYPE PASS
            # =============================================================
            wt_frozen = bool(self.peft_manager.wt_path_is_frozen)
            need_wt_losses = (hp.lambda_reg_wt > 0 or hp.lambda_rank_wt > 0)
            if wt_frozen and hasattr(self.model, 'peft_wt') and isinstance(self.model.peft_wt, torch.nn.Module):
                self.model.peft_wt.eval()

            if wt_frozen and not need_wt_losses and tf_mix >= 1.0 and bool(has_add_label.all()):
                wt_pred_cal = torch.zeros_like(tf_additive_labels)
                wt_pred_raw = torch.zeros_like(tf_additive_labels)
            elif wt_frozen and not need_wt_losses:
                with torch.no_grad():
                    wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=self.mask_wt)
                    wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()
                if torch.cuda.is_available():
                    torch.clear_autocast_cache()
            else:
                wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=self.mask_wt)
                wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()

            cens_wt = torch.where(is_single, cens_mb, torch.zeros_like(cens_mb))
            wt_losses = []
            if hp.lambda_reg_wt > 0:
                # Assembled in latent (the pass's own sum over the item's positions), linked once, at the end. A multi-mutant's target is
                # itself a latent sum, so it stays off the link and needs no dG_wt - hence it is kept even where link_ok is False.
                p_obs = link.obs_ddG(wt_pred_cal, dGwt_mb, floor_mb)
                p_wt = torch.where(is_single, p_obs, wt_pred_cal)
                reg_wt_mask = valid_wt_mask & (link_ok | ~is_single)
                if reg_wt_mask.any():
                    term, val, wsum = self._reg_loss(p_wt, wt_targets, bound_mb, cens_wt, reg_wt_mask, denom)
                    wt_losses.append(hp.lambda_reg_wt * term)
                    sums['reg_wt'] += val; cnts['reg_wt'] += wsum

            if hp.lambda_rank_wt > 0 and self.crit_rank_wt is not None:
                L_rank, val, n_list = self._compute_rank_loss(
                    wt_pred_raw, wt_targets, valid_wt_mask, cens_wt, list_size, self.crit_rank_wt
                )
                if L_rank is not None:
                    wt_losses.append(hp.lambda_rank_wt * L_rank * (n_list / global_num_lists))
                    sums['rank_wt'] += val * n_list; cnts['rank_wt'] += n_list

            if wt_losses and not self.peft_manager.wt_path_is_frozen and (hp.lambda_rank_wt > 0 or hp.lambda_reg_wt > 0):
                total_wt = sum(wt_losses)
                if not torch.isfinite(total_wt): raise AssertionError("WT Loss evaluated to NaN/Inf.")
                if not total_wt.requires_grad: raise AssertionError("WT Loss detached from PyTorch Graph! Cannot call backward.")
                self.manual_backward(total_wt)

            if not self.peft_manager.mt_path_is_frozen and (hp.lambda_rank_mt > 0 or hp.lambda_reg_mt > 0):
                # =============================================================
                # PHASE 2: MUTANT & COMBINED PASS
                # =============================================================
                mt_out = self.model.forward_partitioned(micro, pass_type='mt', mask_strategy=self.mask_mt)
                mt_pred_cal, mt_pred_raw = mt_out['pred_calibrated'].float(), mt_out['pred_raw'].float()

                # The WT half of the combined prediction is detached: the WT graph never has to survive its own backward.
                base_wt_cal, base_wt_raw = wt_pred_cal.detach(), wt_pred_raw.detach()

                if not has_add_label.any():
                    logging.warning("Teacher forcing is on, but no additive labels or single mutations are present in the batch.")

                # Invert the calibration: raw = (cal - bias) / scale
                cal_head = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
                if cal_head is None:
                    raise AssertionError("Could not locate 'calibration_head_wt' or 'calibration_head_fused' on self.model for de-calibration.")
                s = 1.0 if not cal_head.use_scale else cal_head.scale.detach()
                # bias_for, not cal_head.bias: the head adds bias * n_mut, so a scalar inverse here would under-correct every multi-mutant
                # and silently bias the label the MT head is teacher-forced against.
                b = cal_head.bias_for(n_mut_mb)
                b = b.detach() if torch.is_tensor(b) else b
                decalibrated_label = (tf_additive_labels - b) / s

                forced_wt_cal = torch.where(has_add_label, tf_additive_labels, base_wt_cal)
                forced_wt_raw = torch.where(has_add_label, decalibrated_label, base_wt_raw)

                # --tf_mix: how much of an item's WT half comes from the measurement rather than from the WT head's own prediction. 1.0
                # is the label wherever one is usable; 0.0 never teacher-forces; 0.5 is the arithmetic mean of the two on the calibrated
                # scale. The label carries the measurement noise (two singles for a double, one for a single) and the floor clipping, the
                # WT head is smooth but biased, so the mix trades variance for bias. It applies to SINGLES as well as doubles: with the
                # label in place the MT head is a total/interaction predictor, with the WT pass in place it compensates the WT head's
                # error. An item with no usable label is unaffected for any mix (both terms of the blend are then the WT pass).
                if tf_mix < 1.0:
                    w_cal = torch.full_like(forced_wt_cal, tf_mix)
                    forced_wt_cal = w_cal * forced_wt_cal + (1.0 - w_cal) * base_wt_cal
                    forced_wt_raw = w_cal * forced_wt_raw + (1.0 - w_cal) * base_wt_raw

                combined_pred_raw = combine_predictions(forced_wt_raw, mt_pred_raw)
                combined_pred_cal = combine_predictions(forced_wt_cal, mt_pred_cal)

                mt_losses = []

                # Mut pass strictly relies on combined losses
                if hp.lambda_reg_mt > 0:
                    p_comb = link.obs_ddG(combined_pred_cal, dGwt_mb, floor_mb)
                    reg_c_mask = reg_mask & link_ok
                    if reg_c_mask.any():
                        term, val, wsum = self._reg_loss(p_comb, ddG_mb, bound_mb, cens_mb, reg_c_mask, denom)
                        mt_losses.append(hp.lambda_reg_mt * term)
                        sums['reg_combined'] += val; cnts['reg_combined'] += wsum

                if hp.lambda_rank_mt > 0 and self.crit_rank_mt is not None:
                    if getattr(self.model, 'adapter_mode', 'dual') == 'fused' and getattr(self.model, 'fused_antisymmetric', False):
                        cal_mt_h = getattr(self.model, 'calibration_head_fused', None)
                        b_m = cal_mt_h.bias_for(n_mut_mb) if cal_mt_h is not None else 0.0
                        b_m = b_m.detach() if torch.is_tensor(b_m) else b_m
                        rank_pred_in = combine_predictions(forced_wt_cal / s, mt_pred_raw - (b_m / s))
                    else:
                        rank_pred_in = combined_pred_raw
                    if rank_group_by == 'partner':
                        if 'ddG_A' in micro and 'ddG_B' in micro:
                            epi_targ_mb = ddG_mb - (micro['ddG_A'].float() + micro['ddG_B'].float())
                            epi_mask_mb = reg_mask & ~is_single & has_add_label & torch.isfinite(epi_targ_mb)
                            epi_pred_mb = mt_pred_raw - forced_wt_raw
                        else:
                            epi_targ_mb = epi_mask_mb = epi_pred_mb = None
                        L_rank, val, n_list = self._compute_partner_rank_loss(
                            rank_pred_in, ddG_mb, reg_mask, cens_mb, list_size, self.crit_rank_mt,
                            micro=micro, epi_pred=epi_pred_mb, epi_targ=epi_targ_mb, epi_mask=epi_mask_mb
                        )
                    else:
                        L_rank, val, n_list = self._compute_rank_loss(
                            rank_pred_in, ddG_mb, reg_mask, cens_mb, list_size, self.crit_rank_mt
                        )
                    if L_rank is not None:
                        mt_losses.append(hp.lambda_rank_mt * L_rank * (n_list / global_num_lists))
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

    def on_train_epoch_start(self) -> None:
        """Synchronize the train batch sampler with trainer's current_epoch for resumable dataloading."""
        trainer = getattr(self, 'trainer', None)
        if trainer is None:
            return
        epoch = getattr(trainer, 'current_epoch', 0)
        train_loader = getattr(trainer, 'train_dataloader', None)
        if train_loader is not None:
            batch_sampler = getattr(train_loader, 'batch_sampler', None)
            if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
                batch_sampler.set_epoch(epoch)

    def training_step(self, batch: dict, batch_idx: int):
        self.model.train()
        if self.peft_manager.wt_path_is_frozen and hasattr(self.model, 'peft_wt') and isinstance(self.model.peft_wt, torch.nn.Module):
            self.model.peft_wt.eval()
        logs = self._compose_losses_streaming_and_backward(utils._normalize_batch(batch))

        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=False)

        optim = self.optimizers()

        log_every = int(getattr(self.trainer, 'log_every_n_steps', 25) or 25) if getattr(self, '_trainer', None) is not None else 1
        if batch_idx == 0 or (batch_idx + 1) % log_every == 0:
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
        if sch_warmup.last_epoch < self.hparams.lr_warmup_steps:
            sch_warmup.step()

        self._log_lrs()
        self.log_calibration_head(on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)

        if getattr(self, '_cens_diag', None) is not None:
            self.log("train/cens_lower_items", float(self._cens_diag[0]), on_step=True)
            self.log("train/cens_upper_items", float(self._cens_diag[1]), on_step=True)
            self.log("train/tf_saturated_dropped", float(self._cens_diag[2]), on_step=True)

        if torch.cuda.is_available():
            gb = 1.0 / (1024 ** 3)
            self.log("mem/allocated_gb", torch.cuda.memory_allocated() * gb, on_step=True)
            self.log("mem/reserved_gb", torch.cuda.memory_reserved() * gb, on_step=True, prog_bar=True)
            self.log("mem/peak_allocated_gb", torch.cuda.max_memory_allocated() * gb, on_step=True)

        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        if getattr(self, '_val_cache_dl_idx', None) != dataloader_idx:
            self._val_cache_dl_idx = dataloader_idx
            self.model._masked_logit_cache = {}
        elif getattr(self.model, '_masked_logit_cache', None) is None:
            self.model._masked_logit_cache = {}

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
                w = self.model.forward_partitioned(batch, pass_type='wt', mask_strategy=self.mask_wt)['pred_calibrated'].float()
            out_dict = {'wt_lora_pred': w, 'mt_lora_pred': w, 'combined_pred': w,
                        'epi_pred': torch.zeros_like(w)}
        else:
            with torch.inference_mode():
                out_dict = self.model.forward_batch(batch, mask_strategy=self.mask_strategy)

        B = int(out_dict['wt_lora_pred'].shape[0])
        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        nan_col = lambda: np.full(B, np.nan, dtype=np.float32)
        get = lambda k: batch[k].detach().cpu().float().numpy() if torch.is_tensor(batch.get(k)) else nan_col()

        # The epistasis metrics need the LATENT scores and apply the link themselves. The observed scale is carried
        # separately, for the rmse metrics only, so they stay comparable with a run that has no link.
        obs = {}
        if torch.is_tensor(batch.get('dG_wt')):
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
                c = self.model.forward_cycle(batch, mask_strategy=self.mask_strategy,
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

    _VAL_PROGRESS_BAR = ('rho_combined', 'rho_singles_wt')

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
        ext_per_loader = {}          # OOD loaders: kept out of avg_metrics/pooled, survived by _external summaries

        for dataloader_idx, valid_outputs in self.validation_step_outputs.items():
            name = (self.val_dataloader_names[dataloader_idx] if dataloader_idx < len(self.val_dataloader_names)
                    else f"unknown_dl_{dataloader_idx}")

            cat = lambda key: (np.concatenate([np.atleast_1d(np.asarray(o[key])) for o in valid_outputs])
                               if valid_outputs else np.array([]))
            wt, mt, comb = cat('wt_scores'), cat('mt_scores'), cat('comb_scores')
            gt, cens, dddG = cat('ground_truths'), cat('cens'), cat('dddG_truths')
            cens_src = cat('cens_src')
            # Score every run on the released labelling, where a mutant-background library's measurements are
            # singles. --native_cond_for_wt / --native_cond_for_mt relabel them 'native_cond' to decide which head
            # trains on them and what the MT pass sees, and that routing must not move the yardstick: with the label
            # left in place rho_singles_wt drops the five mutant-background validation libraries (the frozen gold
            # head reads 0.8536 on the other 20 against 0.8510 on all 25). The dump and the epistasis pool get the
            # same canonical label.
            subset_types = ['single' if t == 'native_cond' else t for o in valid_outputs for t in o['subset_types']]
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
            # The epistasis scalars are pooled over every library at once, so they are not per-library values and
            # never entered avg_metrics. Merge them in as single-element lists AFTER the val_<key>_avg logging loop
            # above - so nothing is logged twice - and the plateau steps can then read any metric by name.
            for _k, _v in (self._log_epistasis(pooled) or {}).items():
                if np.isfinite(_v):
                    avg_metrics[_k].append(float(_v))

        for k, v in self.link_head.summary().items():
            self.log(f"link/{k}", v, on_epoch=True, sync_dist=True)

        if not self.trainer.sanity_checking:
            # After warmup only: during warmup LambdaLR rewrites every group's lr each step and would overwrite these decays.
            if self.trainer.global_step >= self.hparams.lr_warmup_steps:
                self._wt_plateau_step(avg_metrics)
                self._mt_plateau_step(avg_metrics)
                self._stop_if_nothing_is_training()

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _log_epistasis(self, pooled):
        """
        Logs ``val_epi_<metric>_<head>`` (``esm_msr.epi_metrics``: the full epistasis hierarchy for epi and ctx,
        naive_rho also for comb and add, and global calibration/scale for epi), pooled over every validation library,
        and with --val_dump writes ``val_dump_<tag>.npz`` for offline rescoring.

        Heads: epi (~MT, evaluated relative to the WT additive baseline), ctx ((WT + ~WT)/2, needs --val_cycle_passes),
        comb ((WT + ~MT)/2, reported prediction, naive_rho only), add (WT, additive by construction - the control for
        what saturation alone can earn). With --link the metrics put each head on the observed scale through the run's
        link, bending at each library's own floor. Never raises.
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
                extra = {'link_summary': np.array(json.dumps(self.link_head.summary()))}
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
            link = self.link_head.numpy()
            res = epi_metrics.compute(epi_metrics.Table(scored, link))
            for name, val in res.items():
                if np.isfinite(val):
                    self.log(f"val_{name}", float(val), on_epoch=True,
                             prog_bar=name in ('epi_cell_rank_epi', 'epi_cell_rank_mt'), sync_dist=True)
            return res                  # the caller merges these into avg_metrics so a plateau can watch them
        except Exception as e:
            logging.warning(f"epistasis metrics skipped: {e}", exc_info=True)
        return {}

    def _plateau_step(self, head: str, avg_metrics: dict) -> None:
        """
        Decay one head's own learning rate when ITS validation metric stops improving, and freeze it after ``--freeze_after_decays`` decays.

        Each head gets its own plateau because a shared one (on ``val_rho_combined_avg``) kept the WT adapter taking full-rate steps on a
        metric the MT path was still improving, so it drifted and the target the MT path teacher-forces against moved underneath it; and a
        shared scheduler that owned every parameter group compounded patience-1 triggers into a 370x drop in three epochs (dc12r_s2).
        The WT head watches ``rho_singles_wt`` and touches only the ``lora_wt`` / ``calib_wt`` groups; the MT head watches
        ``--mt_convergence_metric`` (default ``epi_beyond_rho_epi``: the epistasis component's correlation on the part of a double the WT head's
        singles do not explain) and touches only ``lora_mt`` / ``calib_mt``. That metric is noisy epoch to epoch (swings of +-0.03, no trend), so
        patience 1 wants the gentle factor 0.3 it has by default.
        """
        patience = int(self.hparams.plateau_patience or 0)
        done = self.peft_manager.has_transitioned if head == 'wt' else self.peft_manager.mt_path_is_frozen
        if patience <= 0 or done:
            return
        key = 'rho_singles_wt' if head == 'wt' else self.hparams.mt_convergence_metric
        if head != 'wt' and key == 'epi_beyond_rho_mt' and key not in avg_metrics and 'epi_beyond_rho_epi' in avg_metrics:
            key = 'epi_beyond_rho_epi'
        if not avg_metrics.get(key):
            # Loudly, and once: a scheduler that is configured but silently inert is worse than one that errors.
            # dc12 ran 8 epochs with its MT plateau doing nothing, because _log_epistasis returned None and the
            # epistasis scalars never reached avg_metrics - and not one line said so.
            if not getattr(self, f'_warned_no_{head}_metric', False):
                setattr(self, f'_warned_no_{head}_metric', True)
                logging.warning(f"--plateau_patience is set but '{key}' is not among the validation metrics {sorted(avg_metrics)}, "
                                f"so the {head.upper()} plateau will never fire.")
            return
        current = float(np.nanmean(avg_metrics[key]))
        if not np.isfinite(current):
            return

        best = getattr(self, f'_{head}_plateau_best', -float('inf'))
        if current > best + 1e-4:
            setattr(self, f'_{head}_plateau_best', current); setattr(self, f'_{head}_plateau_bad', 0)
            return
        bad = getattr(self, f'_{head}_plateau_bad', 0) + 1
        setattr(self, f'_{head}_plateau_bad', bad)
        if bad < patience:
            return

        setattr(self, f'_{head}_plateau_bad', 0)
        factor = float(self.hparams.plateau_factor)
        touched = []
        for g in self.optimizers().param_groups:
            if g.get('name') in (f'lora_{head}', f'calib_{head}'):
                new = float(g['lr']) * factor
                touched.append(f"{g['name']}: {float(g['lr']):.2e} -> {new:.2e}")
                g['lr'] = new
        decays = getattr(self, f'_{head}_decays', 0) + 1
        setattr(self, f'_{head}_decays', decays)
        logging.info(f"{head.upper()} plateau on {key} (best={best:.4f}, now={current:.4f}), decay #{decays}: {', '.join(touched)}")

        after = int(self.hparams.freeze_after_decays or 0)
        if after and decays >= after:
            if head == 'wt':
                logging.info(f"WT path frozen after {decays} decays; its predictions are now fixed, so the "
                             f"teacher-forced target the MT path uses stops moving.")
                self._transition_to_mt()
            else:
                logging.info(f"MT path frozen after {decays} decays.")
                self.peft_manager.freeze_mt_components()
                self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def _wt_plateau_step(self, avg_metrics: dict) -> None:
        ESM3EpistasisLightningModule._plateau_step(self, 'wt', avg_metrics)

    def _mt_plateau_step(self, avg_metrics: dict) -> None:
        if getattr(getattr(self, 'model', None), 'adapter_mode', 'dual') == 'fused':
            return
        ESM3EpistasisLightningModule._plateau_step(self, 'mt', avg_metrics)

    def _stop_if_nothing_is_training(self) -> None:
        """
        End the run once every optimizer group that still holds trainable parameters is at lr 0.

        With --wt_freeze_after_decays and --mt_freeze_after_decays both set, each head stops itself after its own
        decay budget, and the MT head is normally last (rho_singles_wt peaks around epoch 2 while beyond_rho_mt
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
        """Freeze the WT path and open the MT path, saving the converged WT weights first.

        In a WT-only arm (``--freeze_mt_adapter``, both MT lambdas 0, doubles dropped) the MT path must NOT be
        reopened: ``enforce_freezing`` below then zeroes the MT groups' learning rates, which is the condition
        ``_stop_if_nothing_is_training`` reads, so the arm stops at the freeze instead of spending its remaining
        epoch budget forward-only. It did not, and 2026-10-19 cost an epoch: r8a11 froze the WT after decay #3 and
        then ran epoch 7 at 4.9 GB with ``lr/lora_mt`` still 2e-4 and no loss. The check existed in the retired
        ``pin_dc20_series`` as a hand patch but was never upstreamed, so consolidating onto devel-revision dropped
        it (see docs/STATE.md pitfall 14). The warm-up LambdaLR also rewrites every group's lr, so the zeroing at
        ``on_train_start`` is undone during warm-up; this transition happens after it and therefore sticks.
        """
        self._save_converged_wt_weights()
        self.peft_manager.freeze_wt_components()
        if self.hparams.freeze_mt_adapter or getattr(getattr(self, 'model', None), 'adapter_mode', 'dual') == 'fused':
            self.peft_manager.freeze_mt_components()
        elif not getattr(self.peft_manager, 'mt_path_is_frozen', False):
            self.peft_manager.unfreeze_mt_components()
        self.peft_manager.has_transitioned = True
        self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def on_validation_end(self):
        self.model._masked_logit_cache = None
        self._val_cache_dl_idx = None
        self.peft_manager.apply_baseline_requires_grad()

    def configure_optimizers(self):
        base_lr = float(self.hparams.learning_rate)
        wd = float(self.hparams.weight_decay)

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
        if calib_wt_params: main_groups.append({"params": calib_wt_params, "lr": base_lr * CALIB_LR_MULT, "weight_decay": 0.0, "name": "calib_wt"})
        if calib_mt_params: main_groups.append({"params": calib_mt_params, "lr": base_lr * CALIB_LR_MULT, "weight_decay": 0.0, "name": "calib_mt"})
        if other_params: main_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd, "name": "other"})
        # The link has no trainable parameters, so it has no group.

        if not main_groups:
            logging.warning("No param groups found; defaulting to all trainables.")
            main_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd, "name": "all"}]

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=True, weight_decay=wd)

        # Linear warm-up of every adapter group; the calibration heads start at their full rate.
        warmup = self.hparams.lr_warmup_steps
        lambdas_main = [(lambda step: 1.0) if "calib" in g["name"] else
                        (lambda step, warmup=warmup: min(1.0, max(1e-4, step / max(1, warmup)))) for g in main_groups]

        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        # Warmup is the only shared scheduler. The ReduceLROnPlateau that used to sit here watched val_rho_combined_avg and owned
        # EVERY group in opt_main, so once each head got a plateau on its own metric it became a second, redundant decay on both
        # of them, triggered by a metric neither head is being tuned against (dc12r_s2: lora_mt went 2e-4 -> 1.8e-6 in three epochs
        # where --plateau_factor 0.3 asked for 5.4e-6 in three decays).
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
            if 'lora' in k.lower() or 'calibration' in k.lower() or 'link_head' in k.lower()
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
        trainer = getattr(self, 'trainer', None)
        checkpoint['dataloader_state'] = {
            'epoch': getattr(trainer, 'current_epoch', 0) if trainer else 0,
            'global_step': getattr(trainer, 'global_step', 0) if trainer else 0,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Restores the custom transition state and re-enforces parameter freezing
        if the model was preempted after the WT validation convergence trigger.
        """
        dataloader_state = checkpoint.get('dataloader_state', {})
        if dataloader_state:
            logging.info(f"Resumed dataloader state: checkpoint from epoch {dataloader_state.get('epoch')}, step {dataloader_state.get('global_step')}")
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


def _comet_api_key():
    """The Comet key from the environment, else ~/.comet.config; None if neither has one. Never from the command line."""
    key = os.environ.get('COMET_API_KEY')
    if key:
        return key
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(os.path.expanduser('~/.comet.config'))
        return cfg.get('comet', 'api_key', fallback=None)
    except Exception:
        return None


def main():
    args = parse_arguments()
    pl.seed_everything(args.seed)

    train_loaders, val_loaders, bench_loaders, train_names, val_names, bench_names = setup_dataloaders(args, add_benchmarks_to_val=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # hparams.yaml (and Comet) record what the run IS, not where things are on this machine or how to reach Comet
    hparams = {k: v for k, v in vars(args).items() if k not in UNLOGGED_ARGS}
    try:
        lightning_model = ESM3EpistasisLightningModule(val_dataloader_names=val_names, **hparams)
    except Exception as e:
        logging.error(f"Failed to initialize Lightning Module: {e}.", exc_info=True)
        return

    csv_logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name, version=args.version)
    loggers = [csv_logger]
    checkpoint_dir = csv_logger.log_dir

    if args.comet or args.comet_offline:
        key = _comet_api_key()
        if key:
            comet_kwargs = dict(api_key=key, project=args.comet_project_name, name=f"{args.experiment_name}-{args.version or 'run'}")
            if args.comet_experiment_key:
                # Comet has no notion of resuming on its own: without this a resumed run opens a SECOND
                # experiment, and the run's curves end up split across two of them.
                comet_kwargs.update(experiment_key=args.comet_experiment_key, mode='get_or_create')
            if args.comet_offline:
                offline_dir = os.path.abspath(os.path.join(args.log_dir or '.', 'comet_offline'))
                os.makedirs(offline_dir, exist_ok=True)           # comet_ml does not create it, and refuses to start without it
                os.environ['COMET_OFFLINE_DIRECTORY'] = offline_dir
                comet_logger = OfflineCometLogger(**comet_kwargs)
            else:
                comet_logger = CometLogger(**comet_kwargs)
            loggers.append(comet_logger)
        else:
            logging.warning("No Comet key (COMET_API_KEY or ~/.comet.config): this run logs to CSV only.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename=f"{{epoch:02d}}-{{{args.monitor_metric}:.4f}}", save_top_k=SAVE_TOP_K,
                        monitor=args.monitor_metric, mode='max', save_last=True),
        TQDMProgressBar(refresh_rate=10)
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=args.monitor_metric, patience=args.early_stopping_patience, mode='max', verbose=True))

    try:
        trainer = Trainer(max_epochs=args.num_epochs, accelerator=accelerator, devices=1, logger=loggers, callbacks=callbacks,
                          enable_checkpointing=True, num_sanity_val_steps=0, log_every_n_steps=25, check_val_every_n_epoch=1,
                          precision=os.environ.get('ESM_MSR_PRECISION', 'bf16-mixed'))   # bf16-mixed always; fp32 only for the CPU smoke test
    except Exception as e:
        logging.error(f"Trainer init failed: {e}", exc_info=True)
        return

    if not args.skip_val:
        # verbose=False: Lightning's own table repeats every epoch-level metric under every dataloader (one per protein), so the
        # zero-shot validation is printed once here instead
        results = trainer.validate(lightning_model, dataloaders=val_loaders, verbose=False)
        if trainer.is_global_zero:
            logging.info("Zero-shot validation:\n" + format_validation_results(results))
    ckpt = args.resume_from_checkpoint
    if ckpt:
        logging.info(f"Resuming from {ckpt}")
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders, ckpt_path=ckpt)
    if args.comet_offline and len(loggers) > 1 and trainer.is_global_zero:
        loggers[1].experiment.end()  # writes the archive to upload with `comet upload`

if __name__ == "__main__":
    main()