import os
import logging
import warnings
from collections import Counter, defaultdict
import gc

from typing import List, Dict, Any, Optional, Tuple

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

from esm_msr.models import MSRModel
from esm_msr import utils
from esm_msr.losses import ListMLELoss, ListMLELoss_enhanced, AsymmetricHuberLoss
from esm_msr.preprocess_megascale import setup_dataloaders
from esm_msr.peft_manager import PEFTStateManager
from esm_msr.config import parse_arguments
from esm_msr import stats
from esm_msr import routing

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') 

class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters(ignore=['tokenizer'])
        
        mt_lora_config = {
            "lora_rank": self.hparams.lora_rank_mt, "lora_alpha": self.hparams.lora_alpha_mt, "lora_dropout": self.hparams.lora_dropout_mt,
            "target_mode": self.hparams.target_mode_mt, "use_dora": self.hparams.use_dora_mt, "seed": self.hparams.seed,
            "last_n_layers": self.hparams.last_n_layers_mt,
            "incl_sequence_head": self.hparams.incl_sequence_head_mt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_mt,
        }

        wt_lora_config = {
            "lora_rank": self.hparams.lora_rank_wt, "lora_alpha": self.hparams.lora_alpha_wt, "lora_dropout": self.hparams.lora_dropout_wt,
            "target_mode": self.hparams.target_mode_wt, "use_dora": self.hparams.use_dora_wt, "seed": self.hparams.seed,
            "last_n_layers": self.hparams.last_n_layers_wt,
            "incl_sequence_head": self.hparams.incl_sequence_head_wt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_wt,
        }

        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': self.hparams.seed}

        self.model = MSRModel(
            lora_config=lora_config, shared_scale_init=self.hparams.shared_scale_init, shared_bias_init=self.hparams.shared_bias_init, adapter_mode=self.hparams.adapter_mode,
            lora_mode=self.hparams.lora_mode, model_dtype=torch.float32,
            dedup_backbone=self.hparams.get('dedup_backbone', True),
            mask_structure=self.hparams.get('mask_structure', False),
        )
        # Everything trainable at construction (adapters, calibration heads, unfrozen
        # layernorms); used to keep checkpoints adapter-only.
        self._trainable_param_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        self._warned_unrouted = False

        self.peft_manager = PEFTStateManager(self.model)

        if self.hparams.freeze_wt_adapter: self.peft_manager.freeze_wt_components()
        if self.hparams.freeze_mt_adapter: self.peft_manager.freeze_mt_components()

        def _get_rank_loss():
            if self.hparams.rank_loss == 'listmle': return ListMLELoss(invert=self.hparams.invert_list_loss)
            elif self.hparams.rank_loss == 'listmle_enhanced': return ListMLELoss_enhanced()
            return None

        self.crit_rank_wt = _get_rank_loss() if self.hparams.lambda_rank_wt > 0 else None
        self.crit_rank_combined = _get_rank_loss() if self.hparams.lambda_rank_combined > 0 else None
        self.crit_rank_mt = _get_rank_loss() if self.hparams.lambda_rank_mt > 0 else None
        if self.hparams.get('censor_floor', None) is not None and self.hparams.lambda_rank_mt > 0 \
                and not isinstance(self.crit_rank_mt, ListMLELoss):
            raise AssertionError("--censor_floor requires --rank_loss listmle (censored Plackett-Luce).")

        if self.hparams.reg_loss == 'huber':
            self.crit_reg = nn.HuberLoss(reduction='none', delta=self.hparams.huber_delta)
        elif self.hparams.reg_loss == 'mse':
            self.crit_reg = nn.MSELoss(reduction='none')
        elif self.hparams.reg_loss == 'asymmetric':
            self.crit_reg = AsymmetricHuberLoss()

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = self.hparams.get('val_dataloader_names', ['val'])

    def on_train_start(self):
        self._reshare_base_params_on_device()
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

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
        n_shared, dev = 0, None
        for name, p in peft_mt.named_parameters():
            if "lora_" in name or "dora_" in name:
                continue
            src = wt_base.get(_clean(name))
            if src is not None:
                p.data = src.data
                n_shared += 1
                dev = src.device
        logging.info(f"[reshare] dual-adapter base weights re-shared on {dev}: {n_shared} params "
                     f"deduplicated (transformer Linears stored bf16, rest fp32)")

    def _compute_rank_loss(self, pred, targets, mask, list_size, crit_fn):
        valid_len = (pred.shape[0] // list_size) * list_size
        if valid_len > 0:
            pred_rank = pred[:valid_len].view(-1, list_size)
            targ_rank = targets[:valid_len].view(-1, list_size)
            mask_rank = mask[:valid_len].view(-1, list_size)
            L_raw = crit_fn(pred_rank, targ_rank, mask=mask_rank)
            avg_len = mask_rank.float().sum(dim=-1).mean()
            scaled_loss = L_raw * (list_size / avg_len.clamp(min=1.0))
            num_lists = valid_len // list_size
            return scaled_loss, L_raw.detach(), num_lists
        return None, 0.0, 0

    def _compute_flip_loss(self, pred, targets, valid, flip_keys, crit_fn, min_len, censored=None):
        """Within-column rank loss on the MT pass - the flip-signature objective.

        A *flip column* is one scored position with one fixed partner identity, over the
        substitutions available at that position. Inside a column the conditional target
        ddG(A|B) differs from the double's ddG_AB only by the constant ddG_B, so ordering by
        either is the same ordering; imposing it is therefore equivalent to reproducing the
        measured ordering of the double-mutant phenotypes.

        Why this and not regression: an ordering within a column is invariant to any monotone
        function of the underlying stability, so the assay's saturating response and its
        dynamic-range floor cannot be fitted by it at all. A regression on ddG can, and does.

        One honest caveat, because it decides how to read the metric. Under strict additivity
        every column shares ONE ordering, so a purely additive model already satisfies much of
        this loss; what it cannot satisfy is the per-column *deviation* from that consensus,
        which is the identity-dependent part. So this loss is artifact-immune and includes the
        interaction term, but is not exclusively about it. The matching validation metric
        (``val_rho_flip``) double-centres away the consensus and therefore IS exclusive - use
        the loss to train and the metric to judge.

        Columns are variable length, so they are padded into a [G, L] block with a mask
        rather than reshaped.

        ``censored`` (optional bool per row) marks items pinned at the assay floor. They stay
        in their column - every uncensored member is still ranked above them - but their order
        among themselves is not scored (censored Plackett-Luce, ``ListMLELoss`` ``score_mask``).
        A column counts toward ``min_len`` by total membership but needs >= 1 uncensored member
        to carry any information.
        """
        groups = {}
        for i, k in enumerate(flip_keys):
            if k and bool(valid[i]):
                groups.setdefault(k, []).append(i)
        groups = [g for g in groups.values() if len(g) >= min_len
                  and (censored is None or not all(bool(censored[i]) for i in g))]
        if not groups:
            return None, 0.0, 0
        L = max(len(g) for g in groups)
        G = len(groups)
        dev = pred.device
        p = torch.zeros(G, L, device=dev, dtype=pred.dtype)
        t = torch.zeros(G, L, device=dev, dtype=pred.dtype)
        m = torch.zeros(G, L, device=dev, dtype=torch.bool)
        sm = torch.ones(G, L, device=dev, dtype=torch.bool)
        for gi, g in enumerate(groups):
            idx = torch.as_tensor(g, device=dev, dtype=torch.long)
            p[gi, :len(g)] = pred[idx]
            t[gi, :len(g)] = targets[idx]
            m[gi, :len(g)] = True
            if censored is not None:
                sm[gi, :len(g)] = ~censored[idx]
        if censored is not None:
            L_raw = crit_fn(p, t, mask=m, score_mask=sm)
        else:
            L_raw = crit_fn(p, t, mask=m)
        avg_len = m.float().sum(dim=-1).mean()
        scaled = L_raw * (L / avg_len.clamp(min=1.0))
        n_unc = int((m & sm).sum())
        self._flip_diag = (G, float(avg_len), int(sum(len(g) for g in groups)), n_unc)
        return scaled, L_raw.detach(), G

    def _subset_weights(self, subset_types, device) -> torch.Tensor:
        """Per-item loss weight from its subset type (1.0 for singles and unknown types)."""
        hp = self.hparams
        weight_by_subset = {
            'double': hp.double_weight,
            'reversion': hp.reversion_weight,
            'cond': hp.cond_weight,
            'native_cond': hp.native_cond_weight,
        }
        return torch.tensor([float(weight_by_subset.get(routing.canonical_subset(s), 1.0)) for s in subset_types],
                            dtype=torch.float32, device=device)

    def _plan_units(self, batch: dict, wt_rows, mt_rows, comb_rows, mb: int):
        """
        Group a batch into work units, each of which runs exactly ONE adapter.

        A unit is ``(kind, rows)`` with kind 'wt', 'mt' or 'combined'. 'wt' and 'mt' units
        touch a single adapter, so no unit alternates between them in the inner loop. Only
        the legacy combined objective needs both passes on the same rows, because its loss
        couples them; those rows become 'combined' units.

        Singles can appear in both a 'wt' unit (wild-type-context regression) and an 'mt'
        unit (the single anchor). That is the same two forward passes as before, just split
        so each runs alone.

        WT units are sized by how many *distinct* backbone inputs they contain rather than
        by micro_batch_size: every item of a library shares one wild-type sequence, so with
        ``dedup_backbone`` the whole WT block is a single forward and splitting it would
        only repeat that forward.
        """
        units = []
        if len(comb_rows):
            units += [('combined', comb_rows[s:s + mb]) for s in range(0, len(comb_rows), mb)]
        if len(wt_rows):
            size = mb
            if getattr(self.model, 'dedup_backbone', False):
                first, _ = self.model._unique_rows(batch['wt_sequence_tokens'][wt_rows],
                                                   batch['coords'][wt_rows],
                                                   batch['structure_tokens'][wt_rows])
                if first.numel() <= mb:
                    size = int(len(wt_rows))
            units += [('wt', wt_rows[s:s + size]) for s in range(0, len(wt_rows), size)]
        if len(mt_rows):
            if getattr(self.hparams, 'flip_group_units', True) and self.hparams.lambda_rank_mt > 0:
                # Order MT rows so one flip column lands contiguously. The loss groups by
                # column inside a micro-batch, so without this a column of ~19 is scattered
                # across the batch and most micro-batches hold only a fragment of each. Same
                # items and the same number of forwards - only the grouping changes. Items
                # with no column (anchored singles) sort to the end so they do not split one.
                keys = getattr(self, '_last_flip_keys', None)
                if keys is not None and len(keys):
                    # Reorder in place as a tensor; downstream slicing requires one.
                    order = sorted(range(len(mt_rows)),
                                   key=lambda i: (keys[int(mt_rows[i])] == '', keys[int(mt_rows[i])]))
                    mt_rows = mt_rows[torch.as_tensor(order, dtype=torch.long, device=mt_rows.device)]
            units += [('mt', mt_rows[s:s + mb]) for s in range(0, len(mt_rows), mb)]
        return units

    def _compose_losses_streaming_and_backward(self, batch: dict) -> dict:
        """
        Computes every loss for one batch and back-propagates each work unit immediately, so
        at most one unit's activations are alive at a time.

        Targets and masks are derived once for the whole batch, then units are planned by
        :meth:`_plan_units` so that each one drives a single adapter. Routing comes from
        ``esm_msr.routing``:

        * WT head - ``single`` items, plus multi-mutants whose target becomes ddG_A + ddG_B
          (the WT pass on a double sums wild-type-context effects, so it must not be taught
          the epistasis).
        * MT head - ``cond`` and ``native_cond`` on their own ddG, plus, when
          ``mt_single_anchor_weight > 0``, ordinary singles (the zero-background case of the
          MT task, with clean measured labels).
        * Legacy ``lambda_*_combined`` teacher-force 0.5*label + 0.5*MT on the WT block.
          Algebraically this regresses the MT pass of a double onto
          2*ddG_AB - (ddG_A + ddG_B) = ddG(A|B) + ddG(B|A), the sum of the two conditional
          effects that the ``cond`` subset supervises one at a time.
        * Unrouted subsets (``reversion``) are skipped.
        """
        hp = self.hparams
        device, B = batch['ddG'].device, int(batch['ddG'].shape[0])
        list_size = max(1, int(hp.subset_size))
        mb = min(B, max(list_size, (int(hp.get('micro_batch_size', 32)) // list_size) * list_size))

        st_all = list(batch.get('subset_type', ['single'] * B))
        flip_keys = list(batch.get('flip_key', [''] * B))
        if len(flip_keys) != B:
            flip_keys = [''] * B
        w_all = self._subset_weights(st_all, device)
        global_w_sum = w_all.sum().clamp_min(1e-9)
        global_num_lists = max(1, B // list_size)
        # How many flip columns the whole batch offers, so each micro-batch's contribution
        # is weighted by its share (mirrors global_num_lists for the ListMLE terms).
        self._last_flip_keys = flip_keys
        cens_all = batch.get('censored')
        use_cens = hp.get('censor_floor', None) is not None and torch.is_tensor(cens_all)
        cens_all = cens_all.to(device) if use_cens else None
        _fk = Counter(k for i, k in enumerate(flip_keys) if k)
        _unc = Counter(k for i, k in enumerate(flip_keys) if k and not (use_cens and bool(cens_all[i])))
        global_num_flip = max(1, sum(1 for k, c in _fk.items() if c >= int(hp.flip_list_min) and _unc[k] > 0))

        need_combined = hp.lambda_rank_combined > 0 or hp.lambda_reg_combined > 0 or hp.lambda_epi_combined > 0
        anchor_w = float(hp.get('mt_single_anchor_weight', 0.0) or 0.0)
        if anchor_w > 0 and hp.lambda_reg_mt <= 0:
            raise AssertionError('mt_single_anchor_weight > 0 requires lambda_reg_mt > 0.')
        wt_frozen, mt_frozen = self.peft_manager.wt_path_is_frozen, self.peft_manager.mt_path_is_frozen

        # ---------------- per-item targets and masks, derived once ----------------
        ddG = batch['ddG'].float()
        n_mut = batch['mut_mask'].sum(dim=1)
        nan = torch.full_like(ddG, float('nan'))
        ddG_add = batch['ddG_additive'].float() if 'ddG_additive' in batch else nan
        dddG = batch['dddG'].float() if 'dddG' in batch else nan

        is_wt_subset = routing.subset_mask(st_all, routing.WT_HEAD_SUBSETS, device)
        is_mt_subset = routing.subset_mask(st_all, routing.MT_HEAD_SUBSETS, device)
        wt_block = is_wt_subset | routing.subset_mask(st_all, routing.ENSEMBLE_SUBSETS, device)

        if not self._warned_unrouted:
            unrouted = sorted({s for s in st_all if routing.head_for(s) is None})
            if unrouted:
                logging.warning(f"Subsets {unrouted} have no head in the dual-adapter design and are "
                                f"excluded from all losses (see esm_msr.routing).")
                self._warned_unrouted = True

        # WT head: measured ddG for single mutations, additive ddG for multi-mutants.
        wt_targets = torch.where(n_mut >= 2, ddG_add, ddG)
        wt_ok = wt_block & torch.isfinite(wt_targets)

        # MT head: its own subsets, plus anchored singles. Anchored singles are the bulk of
        # the MT pass's backbone rows, so `mt_single_anchor_frac` subsamples them per step;
        # weights are scaled by 1/frac to leave the anchor's expected contribution unchanged.
        mt_w = torch.where(is_mt_subset, w_all, torch.zeros_like(w_all))
        if anchor_w > 0:
            anchor_rows = is_wt_subset
            frac = float(hp.get('mt_single_anchor_frac', 1.0) or 1.0)
            if not 0.0 < frac <= 1.0:
                raise AssertionError(f'mt_single_anchor_frac must be in (0, 1]; got {frac}.')
            if frac < 1.0 and bool(anchor_rows.any()):
                gen = torch.Generator(device='cpu').manual_seed(int(self.global_step) * 7919 + 13)
                keep = torch.rand(int(anchor_rows.sum()), generator=gen).to(device) < frac
                sampled = torch.zeros_like(anchor_rows)
                sampled[anchor_rows.nonzero(as_tuple=True)[0][keep]] = True
                anchor_rows = sampled
            mt_w = torch.where(anchor_rows, w_all * (anchor_w / frac), mt_w)
        mt_ok = (mt_w > 0) & torch.isfinite(ddG)

        # Items whose absolute target is trustworthy enough for the REGRESSION terms.
        # Double-derived items below the dynamic-range floor are reported by the assay
        # without any flag and roughly half of them are unidentifiable fits, so with
        # --subfloor_rank_only their ordering is still used (it is informative) while their
        # value is withheld from the regression (it is not).
        # Items whose absolute target is trustworthy enough for the REGRESSION terms. The
        # dataset marks sub-floor double-derived items reg_ok=False (see
        # MutationStabilityDataset._drop_unreachable): the assay reports them without any
        # flag, roughly half are unidentifiable fits, and a third are genuine compensation.
        # Their ordering is informative, their value is not, so they stay in the rank losses
        # and are withheld here.
        reg_keep = batch.get('reg_ok')
        reg_keep = (reg_keep.to(device) if torch.is_tensor(reg_keep)
                    else torch.ones(B, dtype=torch.bool, device=device))

        # Legacy combined objective.
        tf_labels = torch.where(n_mut == 1, ddG, ddG_add)
        has_tf_label = torch.isfinite(tf_labels)
        comb_ok = wt_block & ((n_mut >= 2) if hp.mt_reg_mask == 'doubles' else torch.ones_like(wt_block))
        epi_ok = wt_block & (n_mut >= 2) & torch.isfinite(dddG)
        if not hp.zero_epistasis_for_singles:
            epi_ok = epi_ok | is_wt_subset
        epi_targets = torch.where(n_mut >= 2, torch.nan_to_num(dddG), torch.zeros_like(dddG))

        # ---------------- plan ----------------
        idx = torch.arange(B, device=device)
        train_wt = (not wt_frozen) and (hp.lambda_reg_wt > 0 or hp.lambda_rank_wt > 0)
        train_mt_reg = (not mt_frozen) and hp.lambda_reg_mt > 0
        run_combined = (not mt_frozen) and need_combined

        comb_rows = idx[wt_block] if run_combined else idx[:0]
        wt_rows = idx[:0] if (run_combined or not train_wt) else idx[wt_ok]
        mt_rows = idx[mt_ok] if train_mt_reg else idx[:0]
        units = self._plan_units(batch, wt_rows, mt_rows, comb_rows, mb)

        zero = torch.zeros((), device=device)
        sums, cnts = defaultdict(lambda: zero), defaultdict(lambda: zero)

        for kind, rows in units:
            micro, w_mb = utils.slice_batch_by_index(batch, rows), w_all[rows]
            m_wt_ok, m_mt_ok = wt_ok[rows], mt_ok[rows]
            losses_wt, losses_mt = [], []

            # ---- WT pass (for 'wt' and 'combined' units) ----
            wt_pred_cal = wt_pred_raw = None
            if kind in ('wt', 'combined'):
                retain = (kind == 'combined') and not hp.detach_ensemble_input
                wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=hp.mask_strategy,
                                                        detach_calibration=hp.detach_regression)
                wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()
                del wt_out

                if train_wt and m_wt_ok.any():
                    t = wt_targets[rows]
                    if hp.lambda_reg_wt > 0:
                        L = self.crit_reg(wt_pred_cal[m_wt_ok], t[m_wt_ok]) * w_mb[m_wt_ok]
                        losses_wt.append(hp.lambda_reg_wt * L.sum() / global_w_sum)
                        sums['reg_wt'] = sums['reg_wt'] + L.sum().detach()
                        cnts['reg_wt'] = cnts['reg_wt'] + w_mb[m_wt_ok].sum()
                    if hp.lambda_rank_wt > 0 and self.crit_rank_wt is not None:
                        L_rank, val, n_list = self._compute_rank_loss(
                            wt_pred_raw, torch.nan_to_num(t), m_wt_ok, list_size, self.crit_rank_wt)
                        if L_rank is not None:
                            losses_wt.append(hp.lambda_rank_wt * L_rank * (n_list / global_num_lists))
                            sums['rank_wt'] = sums['rank_wt'] + val * n_list
                            cnts['rank_wt'] = cnts['rank_wt'] + n_list

                if losses_wt:
                    total = sum(losses_wt)
                    if not torch.isfinite(total): raise AssertionError("WT Loss evaluated to NaN/Inf.")
                    if not total.requires_grad: raise AssertionError("WT Loss detached from PyTorch Graph! Cannot call backward.")
                    self.manual_backward(total, retain_graph=retain)
                if not retain and wt_pred_cal is not None:
                    wt_pred_cal, wt_pred_raw = wt_pred_cal.detach(), wt_pred_raw.detach()

            if kind == 'wt':
                continue

            # ---- MT pass (for 'mt' and 'combined' units) ----
            mt_out = self.model.forward_partitioned(micro, pass_type='mt', mask_strategy=hp.mask_strategy,
                                                    detach_calibration=hp.detach_regression)
            mt_pred_cal, mt_pred_raw = mt_out['pred_calibrated'].float(), mt_out['pred_raw'].float()
            del mt_out

            if kind == 'combined':
                base_wt_cal = wt_pred_cal.detach() if hp.detach_ensemble_input else wt_pred_cal
                base_wt_raw = wt_pred_raw.detach() if hp.detach_ensemble_input else wt_pred_raw
                tf, has_tf = tf_labels[rows], has_tf_label[rows]

                if self.model.lora_mode == 'ensemble':
                    cal_head = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
                    if cal_head is None:
                        raise AssertionError("Ensemble mode failed: Could not locate 'calibration_head_wt' or "
                                             "'calibration_head_fused' on self.model for de-calibration.")
                    # Invert the calibration: raw = (cal - bias) / scale
                    s = 1.0 if not cal_head.use_scale else cal_head.scale.detach()
                    b = 0.0 if cal_head.bias is None else cal_head.bias.detach()
                    forced_cal = torch.where(has_tf, tf, base_wt_cal)
                    forced_raw = torch.where(has_tf, (tf - b) / s, base_wt_raw)
                elif self.model.lora_mode == 'corrector':
                    forced_cal, forced_raw = base_wt_cal, base_wt_raw
                else:
                    raise AssertionError(f"Unknown lora_mode: {self.model.lora_mode}")

                m_comb, m_epi = comb_ok[rows], epi_ok[rows]
                if hp.lambda_reg_combined > 0 and m_comb.any():
                    L = self.crit_reg(0.5 * forced_cal[m_comb] + 0.5 * mt_pred_cal[m_comb], ddG[rows][m_comb]) * w_mb[m_comb]
                    losses_mt.append(hp.lambda_reg_combined * L.sum() / global_w_sum)
                    sums['reg_combined'] = sums['reg_combined'] + L.sum().detach()
                    cnts['reg_combined'] = cnts['reg_combined'] + w_mb[m_comb].sum()
                if hp.lambda_rank_combined > 0 and self.crit_rank_combined is not None:
                    L_rank, val, n_list = self._compute_rank_loss(
                        0.5 * forced_raw + 0.5 * mt_pred_raw, ddG[rows], m_comb, list_size, self.crit_rank_combined)
                    if L_rank is not None:
                        losses_mt.append(hp.lambda_rank_combined * L_rank * (n_list / global_num_lists))
                        sums['rank_combined'] = sums['rank_combined'] + val * n_list
                        cnts['rank_combined'] = cnts['rank_combined'] + n_list
                if hp.lambda_epi_combined > 0 and m_epi.any():
                    epi_pred = 0.5 * mt_pred_cal - 0.5 * forced_cal
                    L = self.crit_reg(epi_pred[m_epi], epi_targets[rows][m_epi]) * w_mb[m_epi]
                    losses_mt.append(hp.lambda_epi_combined * L.sum() / global_w_sum)
                    sums['epi_combined'] = sums['epi_combined'] + L.sum().detach()
                    cnts['epi_combined'] = cnts['epi_combined'] + w_mb[m_epi].sum()

            elif kind == 'mt':
                if m_mt_ok.any() and hp.lambda_reg_mt > 0:
                    w = mt_w[rows]
                    reg_ok = m_mt_ok & reg_keep[rows] if hp.subfloor_rank_only else m_mt_ok
                    if reg_ok.any():
                        L = self.crit_reg(mt_pred_cal[reg_ok], ddG[rows][reg_ok]) * w[reg_ok]
                        losses_mt.append(hp.lambda_reg_mt * L.sum() / global_w_sum)
                        sums['reg_mt'] = sums['reg_mt'] + L.sum().detach()
                        cnts['reg_mt'] = cnts['reg_mt'] + w[reg_ok].sum()
                if hp.lambda_rank_mt > 0 and self.crit_rank_mt is not None:
                    fk_rows = [flip_keys[int(r)] for r in rows]
                    L_flip, val, n_grp = self._compute_flip_loss(
                        mt_pred_raw, ddG[rows], m_mt_ok, fk_rows,
                        self.crit_rank_mt, hp.flip_list_min,
                        censored=cens_all[rows] if use_cens else None)
                    if L_flip is not None:
                        losses_mt.append(hp.lambda_rank_mt * L_flip * (n_grp / max(global_num_flip, 1)))
                        sums['rank_mt'] = sums['rank_mt'] + val * n_grp
                        cnts['rank_mt'] = cnts['rank_mt'] + n_grp

            if losses_mt:
                total = sum(losses_mt)
                if not torch.isfinite(total): raise AssertionError("MT Loss evaluated to NaN/Inf.")
                if not total.requires_grad: raise AssertionError("MT Loss detached from PyTorch Graph! Cannot call backward.")
                self.manual_backward(total)

        # One host sync for all logged values instead of one per unit and loss term.
        keys = [k for k in ('reg_wt', 'rank_wt', 'reg_combined', 'rank_combined', 'epi_combined', 'reg_mt', 'rank_mt') if k in cnts]
        if not keys:
            return {}
        vals = torch.stack([torch.stack([torch.as_tensor(sums[k], device=device, dtype=torch.float32),
                                         torch.as_tensor(cnts[k], device=device, dtype=torch.float32)]) for k in keys]).tolist()
        return {f'L_{k}': s_ / c_ for k, (s_, c_) in zip(keys, vals) if c_ > 0}

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
        self.model.train()
        logs = self._compose_losses_streaming_and_backward(utils._normalize_batch(batch))
        
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=False)

        optim = self.optimizers()
        
        max_norm = getattr(self.hparams, "grad_clip_norm", None)
        if max_norm and max_norm > 0:
            self.clip_gradients(optim, gradient_clip_val=max_norm, gradient_clip_algorithm="norm")
            
        # Norm diagnostics only at the log cadence (log_every_n_steps): the
        # param->name dict build plus three norm reductions over every parameter
        # group ran every step and forced host syncs that dominated step time.
        if self.global_step % self.trainer.log_every_n_steps == 0:
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
        
        sch_warmup, sch_plateau = self.lr_schedulers()
        total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
        if sch_warmup.last_epoch < total_warmup_steps:
            sch_warmup.step()

        self._log_lrs()
        self.log_calibration_head(on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)
        if getattr(self, '_flip_diag', None) is not None:
            g, alen, nitems, n_unc = self._flip_diag
            self.log("train/flip_cols", float(g), on_step=True)
            self.log("train/flip_len", float(alen), on_step=True)
            self.log("train/flip_items", float(nitems), on_step=True)
            self.log("train/flip_uncensored", float(n_unc), on_step=True)
            self._flip_diag = None
            
        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self.log("amp_scale", self.trainer.precision_plugin.scaler.get_scale(), on_step=True)

        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        with torch.inference_mode():
            out_dict = self.model.forward_batch(batch, mask_strategy=self.hparams.mask_strategy)

        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        dddG = batch.get('dddG')
        n_items = int(out_dict['wt_lora_pred'].shape[0])

        def _np(t):
            return t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.full(n_items, float(t))

        mt_id = batch.get('mt_id')
        row_id = (mt_id[:, 0].detach().cpu().numpy() if torch.is_tensor(mt_id) and mt_id.ndim == 2
                  else np.full(n_items, -1))
        self.validation_step_outputs[dataloader_idx].append({
            'wt_scores': _np(out_dict['wt_lora_pred']),
            'mt_scores': _np(out_dict['mt_lora_pred']),
            'comb_scores': _np(out_dict['combined_pred']),
            'ground_truths': _np(ddG) if ddG is not None else np.full(n_items, np.nan),
            'dddG': _np(dddG) if dddG is not None else np.full(n_items, np.nan),
            'subset_type': list(batch.get('subset_type', ['single'] * n_items)),
            # For val_rho_flip: the column key, and the substitution identity that indexes
            # the row within that column.
            'flip_key': list(batch.get('flip_key', [''] * n_items)),
            'row_id': row_id,
            # Hashable per-item mutation tuple, so rho_epi_full can pair each double with its
            # two singles (comb_AB - comb_A - comb_B).
            'mut_key': [tuple(tuple(m) for m in muts) for muts in batch.get('mutations', [()] * n_items)],
        })

    def on_validation_epoch_start(self):
        self.validation_step_outputs = defaultdict(list)

    def on_validation_epoch_end(self):
        """
        Logs, per dataloader, the three head metrics and the calibration RMSE from
        ``stats.compute_metrics``, their means across dataloaders (``*_avg``, which the
        checkpoint monitor and the plateau scheduler read), and the same metrics pooled
        over every item of every dataloader (``*_pooled``), which weights proteins by
        their size instead of equally.
        """
        per_loader, pooled = {}, defaultdict(list)

        for dataloader_idx, outputs in self.validation_step_outputs.items():
            name = (self.val_dataloader_names[dataloader_idx]
                    if dataloader_idx < len(self.val_dataloader_names) else f"unknown_dl_{dataloader_idx}")
            if not outputs:
                continue

            cols = {k: np.concatenate([np.asarray(o[k]).reshape(-1) for o in outputs])
                    for k in ('wt_scores', 'mt_scores', 'comb_scores', 'ground_truths', 'dddG')}
            subset_types = [s for o in outputs for s in o['subset_type']]
            mut_keys = [k for o in outputs for k in o.get('mut_key', [])]
            if len(mut_keys) != len(subset_types):
                mut_keys = None

            per_loader[name] = stats.compute_metrics(
                cols['wt_scores'], cols['mt_scores'], cols['comb_scores'],
                cols['ground_truths'], subset_types, dddG=cols['dddG'], mut_keys=mut_keys)

            # Identity-dependent interaction, scored on the MT pass. This is the only
            # validation number that is specific to what the MT adapter exists for: it is
            # exactly zero for an additive readout and unaffected by the assay's monotone
            # response, so unlike rho_combined it cannot be satisfied by learning saturation.
            fk = [k for o in outputs for k in o.get('flip_key', [])]
            rid = np.concatenate([np.asarray(o['row_id']).reshape(-1) for o in outputs]) \
                if all('row_id' in o for o in outputs) else np.array([])
            if len(fk) == len(cols['mt_scores']) and len(rid) == len(fk):
                rho_flip, n_pairs, n_cells = stats.flip_signature_rho(
                    cols['mt_scores'], cols['ground_truths'], fk, rid,
                    min_len=int(self.hparams.flip_list_min))
                per_loader[name]['rho_flip'] = rho_flip
                if n_pairs:
                    self.log(f"val_flip_pairs/{name}", float(n_pairs), on_epoch=True, sync_dist=True)
            else:
                per_loader[name]['rho_flip'] = float('nan')

            for k, v in cols.items():
                pooled[k].append(v)
            pooled['subset_type'].extend(subset_types)
            # Mutations are numbered per library, so tag them with the loader to keep pooled
            # singles from pairing with another protein's doubles.
            pooled['mut_key'].extend(
                [tuple((name,) + m for m in k) for k in mut_keys] if mut_keys is not None
                else [None] * len(subset_types))

        for name, metrics in per_loader.items():
            for metric, val in metrics.items():
                if not np.isnan(val):
                    self.log(f"val_{metric}/{name}", val, on_epoch=True, sync_dist=True)

        avg_metrics = {}
        for metric in ('rho_wt_valid', 'rho_wt_all', 'rho_mt_valid', 'rho_mt_all',
                       'rho_combined', 'rmse_combined', 'rho_flip', 'rho_epi_fast', 'rho_epi_full'):
            vals = [m[metric] for m in per_loader.values() if metric in m and not np.isnan(m[metric])]
            if vals:
                avg_metrics[metric] = float(np.mean(vals))
                self.log(f"val_{metric}_avg", avg_metrics[metric], on_epoch=True, prog_bar=True, sync_dist=True)

        if pooled['subset_type']:
            pooled_metrics = stats.compute_metrics(
                np.concatenate(pooled['wt_scores']), np.concatenate(pooled['mt_scores']),
                np.concatenate(pooled['comb_scores']), np.concatenate(pooled['ground_truths']),
                pooled['subset_type'], dddG=np.concatenate(pooled['dddG']),
                mut_keys=None if any(k is None for k in pooled['mut_key']) else pooled['mut_key'])
            for metric, val in pooled_metrics.items():
                if not np.isnan(val):
                    self.log(f"val_{metric}_pooled", val, on_epoch=True, sync_dist=True)

        if not self.trainer.sanity_checking and self.hparams.freeze_wt_on_convergence and not self.peft_manager.has_transitioned:
            target_metric_key = self.hparams.wt_convergence_metric
            if target_metric_key in avg_metrics:
                current_val = float(avg_metrics[target_metric_key])
                if current_val > getattr(self.peft_manager, 'wt_best_metric', -float('inf')) + 1e-4:
                    self.peft_manager.wt_best_metric = current_val
                    self.peft_manager.wt_patience_counter = 0
                else:
                    self.peft_manager.wt_patience_counter = getattr(self.peft_manager, 'wt_patience_counter', 0) + 1
                
                if self.peft_manager.wt_patience_counter >= self.hparams.wt_convergence_patience:
                    logging.info(f"Convergence reached! Transitioning to MT training.")
                    self._save_converged_wt_weights()
                    self.peft_manager.freeze_wt_components()
                    if hasattr(self.peft_manager, 'unfreeze_mt_components'):
                        self.peft_manager.unfreeze_mt_components()
                    else:
                        logging.warning("PEFTStateManager missing 'unfreeze_mt_components'. Please implement this method.")
                    self.peft_manager.has_transitioned = True
                    self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

        if not self.trainer.sanity_checking and self.trainer.current_epoch==self.hparams.freeze_wt_after_epoch and not self.peft_manager.has_transitioned:
            logging.info(f"Epoch {self.hparams.freeze_wt_after_epoch} ended! Transitioning to MT training.")
            self._save_converged_wt_weights()
            self.peft_manager.freeze_wt_components()
            if hasattr(self.peft_manager, 'unfreeze_mt_components'):
                self.peft_manager.unfreeze_mt_components()
            else:
                logging.warning("PEFTStateManager missing 'unfreeze_mt_components'. Please implement this method.")
            self.peft_manager.has_transitioned = True
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

        if not self.trainer.sanity_checking:
            schedulers = self.lr_schedulers()
            if schedulers is not None:
                sch_warmup, sch_plateau = schedulers
                total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
                if self.trainer.global_step >= total_warmup_steps and 'rho_combined' in avg_metrics:
                    sch_plateau.step(avg_metrics['rho_combined'])

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

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

            # Never hand the frozen backbone to the optimizer (with freeze_wt_on_convergence
            # the baseline check below is bypassed, which used to put all ~1.4B frozen base
            # weights into the 'other' group and its per-step norm logging).
            if name not in self._trainable_param_names:
                continue
            # Rely on the manager's baseline configuration to determine valid groups
            if not self.peft_manager.baseline_requires_grad.get(name, True) and not self.hparams.freeze_wt_on_convergence:
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

        if not main_groups:
            logging.warning("No param groups found; defaulting to all trainables.")
            main_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd, "name": "all"}]

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=True, weight_decay=wd)

        lambdas_main = []
        for g in main_groups:
            if "calib" in g["name"]: lambdas_main.append(lambda step: 1.0)
            elif g["name"] == "lora_mt":
                lambdas_main.append(lambda step, delay=self.hparams.mt_lora_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))
            else:
                lambdas_main.append(lambda step, delay=self.hparams.calib_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))

        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        plateau_main = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_main, mode='max', factor=0.1, patience=1, min_lr=1e-7)

        return [opt_main], [{"scheduler": warmup_main, "interval": "step", "frequency": 1}, {"scheduler": plateau_main, "interval": "epoch", "frequency": 1, "monitor": "val_rho_combined_avg"}]

    def _save_converged_wt_weights(self):
        """Extracts and saves only the WT adapter and calibration head at the moment of convergence."""
        
        logging.info("Saving converged WT adapter weights before transition...")
        wt_state_dict = {}
        
        for name, param in self.model.named_parameters():
            # WT adapter + its calibration head only. (Matching on 'peft_wt' used to grab every
            # frozen backbone weight in the WT PEFT wrapper as well: a ~5.6 GB file.)
            if name not in self._trainable_param_names:
                continue
            if any(x in name for x in ['wt_adapter', 'calibration_head_wt', 'default', 'calibration_head_fused']):
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
        
        trainable_keys = {f"model.{n}" for n in self._trainable_param_names}
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'lora' in k.lower() or 'calibration' in k.lower() or k in trainable_keys
        }
        
        if not filtered_state_dict:
            logging.warning("Checkpoint filtering caught an empty state_dict. Check parameter naming.")
            
        checkpoint['state_dict'] = filtered_state_dict
        
        # Save the transition state from the PEFT manager so it persists across preemptions
        checkpoint['transition_state'] = {
            'has_transitioned': getattr(self.peft_manager, 'has_transitioned', False),
            'wt_best_metric': getattr(self.peft_manager, 'wt_best_metric', -float('inf')),
            'wt_patience_counter': getattr(self.peft_manager, 'wt_patience_counter', 0)
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Restores the custom transition state and re-enforces parameter freezing
        if the model was preempted after the WT validation convergence trigger.
        """
        transition_state = checkpoint.get('transition_state', {})
        self.peft_manager.has_transitioned = transition_state.get('has_transitioned', False)
        self.peft_manager.wt_best_metric = transition_state.get('wt_best_metric', -float('inf'))
        self.peft_manager.wt_patience_counter = transition_state.get('wt_patience_counter', 0)
        
        # Re-apply freezing states if we resumed after convergence
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter:
            self.peft_manager.freeze_wt_components()

        if self.peft_manager.has_transitioned:
            if hasattr(self.peft_manager, 'unfreeze_mt_components'):
                self.peft_manager.unfreeze_mt_components()
            else:
                logging.warning("PEFTStateManager missing 'unfreeze_mt_components'. Please implement this method.")
     
        if self.hparams.freeze_mt_adapter:
            self.peft_manager.freeze_mt_components()


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
        comet_kwargs = {
            "api_key": args.comet_api_key,
            "project": args.comet_project_name,
            "name": f"{args.experiment_name}-{args.version or 'run'}",
        }
        if getattr(args, "comet_experiment_key", None):
            comet_kwargs["experiment_key"] = args.comet_experiment_key
        loggers.append(CometLogger(**comet_kwargs))

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename=args.checkpoint_filename, save_top_k=args.save_top_k, monitor=args.monitor_metric, mode=args.monitor_mode, save_last=True),
        TQDMProgressBar(refresh_rate=min(10, args.log_every_n_steps))
    ]

    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=args.early_stopping_metric, patience=args.early_stopping_patience, mode=args.monitor_mode, verbose=True))

    trainer_kwargs = {
        "max_epochs": args.num_epochs, "min_epochs": args.min_epochs, "accelerator": accelerator, "devices": devices, "strategy": strategy,
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

    if not args.skip_val: trainer.validate(lightning_model, dataloaders=val_loaders, ckpt_path=args.ckpt_path)
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()