import os
import re
import json
import logging
import warnings
from collections import defaultdict
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
from esm_msr import stats, epi_metrics, censoring

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') 

# A library measured in a mutant background ('2KT8_V15A'): its singles are conditional effects, not wild-type-context ddGs, so rho_wt_valid
# leaves it out (as devel does, where they are 'native_cond').
NATIVE_BACKGROUND_CODE_RE = re.compile(r'_[A-Z][0-9]+[A-Z]$')

EXTERNAL_SUMMARY_METRICS = ('rho_combined', 'rmse_combined')


def summarise_external_benchmarks(per_loader: dict, external: set) -> dict:
    """OOD validation loaders summarised as ``val_<metric>_external``: the mean of rho_combined and rmse_combined.

    ``per_loader`` maps loader name -> {metric: value}. These libraries are deliberately kept out of every aggregate
    (``val_*_avg``, the pooled ungrouped metrics, the epistasis levels): they are out of the training distribution, so
    including them makes the aggregate incomparable across runs and inconsistent with what the model was trained on.
    Their per-loader columns are still logged by the caller, so nothing diagnostic is lost. Same contract and the same
    metric names as devel-revision's helper of this name, so the two branches' aggregates line up.
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
        # OOD benchmarks: scored and logged per loader, held out of every aggregate (see summarise_external_benchmarks)
        self.external_val_sets = {s.strip() for s in str(self.hparams.get('external_val_sets', '') or '').split(',')
                                  if s.strip()}
        if self.external_val_sets:
            logging.info(f"external validation benchmarks held out of val_*_avg, the pooled metrics and the "
                         f"epistasis levels: {sorted(self.external_val_sets)}; summarised as "
                         f"val_rho_combined_external and val_rmse_combined_external")
        
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
            lora_mode=self.hparams.lora_mode, model_dtype=torch.float32
        )
        
        self.peft_manager = PEFTStateManager(self.model)

        if self.hparams.freeze_wt_adapter: self.peft_manager.freeze_wt_components()
        if self.hparams.freeze_mt_adapter: self.peft_manager.freeze_mt_components()

        def _get_rank_loss():
            if self.hparams.rank_loss == 'listmle': return ListMLELoss(invert=self.hparams.invert_list_loss)
            elif self.hparams.rank_loss == 'listmle_enhanced': return ListMLELoss_enhanced()
            return None

        self.crit_rank_wt = _get_rank_loss() if self.hparams.lambda_rank_wt > 0 else None
        self.crit_rank_combined = _get_rank_loss() if self.hparams.lambda_rank_combined > 0 else None

        if self.hparams.reg_loss == 'huber':
            self.crit_reg = nn.HuberLoss(reduction='none', delta=self.hparams.huber_delta)
        elif self.hparams.reg_loss == 'mse':
            self.crit_reg = nn.MSELoss(reduction='none')
        elif self.hparams.reg_loss == 'asymmetric':
            self.crit_reg = AsymmetricHuberLoss()

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = self.hparams.get('val_dataloader_names', ['val'])

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

        ``main()`` calls ``trainer.validate`` before ``trainer.fit``, so doing this only in ``on_train_start`` is too late:
        the first validation runs with the base weights still duplicated between the two adapters AND still fp32, roughly
        11 GB of parameters instead of 3. That sets the CUDA caching allocator's high-water mark at its maximum, and the
        re-share then frees those fp32 storages into the allocator's cache, so training allocates its activations inside a
        large, fragmented reservation: a monotone slowdown (2.2 s/step rising past 12 s/step by step 260 on devel-revision)
        that looks like a leak. Do NOT add empty_cache here: under expandable_segments on WSL2 it cost 3.7 GiB of device
        memory and changed torch's own accounting not at all (devel-revision 1e0d5b2).

        Called from every stage hook, and deliberately NOT behind a once-only flag. The flag this used to carry made
        the re-share fire exactly once per process, which is wrong because of the very ``validate``-before-``fit``
        ordering described above: the re-share ran at ``on_validation_start``, Lightning returned the module to the CPU
        when that stage tore down, the move back for ``fit`` gave every parameter its own storage again, and the flag
        then blocked the repair at ``on_train_start``. Measured on devel-revision, dc8_s2 against its own resume (which
        passes --skip_val and so re-shared at ``on_train_start`` instead, the only path that was ever correct):
        mem/allocated_gb 5.706 GB against 2.972, with the same 2.73 GB on peak_allocated and on reserved. Confirmed
        live in dc14, which logs ``570 had been duplicated`` at BOTH hooks and peaks at 24.90 GB reserved where every
        fresh run before it peaked at 27.607.

        Re-sharing is a dict build and ~570 pointer assignments, so it needs no guard.
        """
        self._reshare_base_params_on_device()

    def on_validation_start(self):
        self._prepare_device_weights()

    def on_train_start(self):
        self._prepare_device_weights()
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

    def _compute_rank_loss(self, pred, targets, mask, list_size, crit_fn):
        """
        Mean over the micro-batch's lists of each list's loss scaled by list_size / (its own valid length); a list with fewer than two valid
        items contributes 0 and still counts. Weighted by num_lists / global_num_lists by the caller, the sum over micro-batches is then the
        same whatever --micro_batch_size is (a micro-batch of one list is the released behaviour exactly).
        """
        valid_len = (pred.shape[0] // list_size) * list_size
        if valid_len > 0:
            pred_rank = pred[:valid_len].view(-1, list_size)
            targ_rank = targets[:valid_len].view(-1, list_size)
            mask_rank = mask[:valid_len].view(-1, list_size)
            num_lists = valid_len // list_size
            per_list = torch.stack([crit_fn(pred_rank[i:i + 1], targ_rank[i:i + 1], mask=mask_rank[i:i + 1]) for i in range(num_lists)])
            n_valid = mask_rank.float().sum(dim=-1).clamp(min=1.0)
            scaled_loss = (per_list * (list_size / n_valid)).mean()
            return scaled_loss, per_list.detach().mean().item(), num_lists
        return None, 0.0, 0

    def _compose_losses_streaming_and_backward(self, batch: dict) -> dict:
        device, B = batch['ddG'].device, int(batch['ddG'].shape[0])
        list_size = max(1, int(self.hparams.subset_size))
        mb = min(B, max(list_size, (getattr(self.hparams, 'micro_batch_size', 32) // list_size) * list_size))
        
        st_all = batch.get('subset_type', ['single'] * B)
        w_all = torch.ones(B, dtype=torch.float32, device=device)
        
        idx_double_all = torch.as_tensor([i for i, s in enumerate(st_all) if s == 'double'], device=device)
        idx_mut_ctx_all = torch.as_tensor([i for i, s in enumerate(st_all) if s == 'mut_ctx'], device=device)
        idx_reversion_all = torch.as_tensor([i for i, s in enumerate(st_all) if s == 'reversion'], device=device)
        idx_onb_all = torch.as_tensor([i for i, s in enumerate(st_all) if s == 'over_and_back'], device=device)
        
        if idx_double_all.numel() > 0: w_all[idx_double_all] = self.hparams.double_weight
        if idx_mut_ctx_all.numel() > 0: w_all[idx_mut_ctx_all] = self.hparams.mut_ctx_weight
        if idx_reversion_all.numel() > 0: w_all[idx_reversion_all] = self.hparams.reversion_weight
        if idx_onb_all.numel() > 0: w_all[idx_onb_all] = self.hparams.over_and_back_weight

        global_w_sum = w_all.sum().clamp_min(1e-9)
        global_num_lists = max(1, B // list_size)

        sums, cnts = defaultdict(float), defaultdict(float)

        for start in range(0, B, mb):
            # Drop the PREVIOUS micro-batch's tensors before this one's forward allocates. manual_backward frees a graph's saved
            # activations, so this matters only for a pass that produced no loss and therefore ran no backward: its graph then stays
            # alive through these names while the next forward allocates a second copy, and peak memory doubles. That happens once
            # the WT path is frozen (--freeze_wt_after_epoch / --freeze_wt_on_convergence), when both WT lambdas are 0, or when a
            # unit's masks come up empty. Ported from devel-revision 5c5e9e9 (devel acf6026, where a lossless unit pinned ~20 GB).
            wt_out = wt_pred_cal = wt_pred_raw = mt_out = mt_pred_cal = mt_pred_raw = epi_pred = None
            base_wt_cal = base_wt_raw = forced_wt_cal = forced_wt_raw = combined_pred_raw = combined_pred_cal = None
            total_wt = total_mt = L = L_rank = decalibrated_label = wt_losses = mt_losses = None

            idx = torch.arange(start, min(start + mb, B), device=device)
            micro, w_mb = utils.slice_batch_by_index(batch, idx), w_all[idx]
            ddG_mb = micro['ddG'].float() 
            is_single, valid_double = micro['mut_mask'].sum(dim=1) == 1, micro.get('valid_dddG_mask', micro['mut_mask'].sum(dim=1) > 1) 
            reg_mask = ~is_single if self.hparams.mt_reg_mask == 'doubles' else torch.ones_like(is_single, dtype=torch.bool)
            
            wt_targets, valid_wt_mask = ddG_mb.clone(), torch.ones_like(ddG_mb, dtype=torch.bool)
            
            # Prepare hybrid additive targets for teacher forcing (ddG_additive for multi, true ddG for singles)
            tf_additive_labels = torch.full_like(ddG_mb, float('nan'))
            if 'ddG_additive' in micro:
                tf_additive_labels = micro['ddG_additive'].clone().float()
                
            # Explicitly set singles to use their true ddG as the additive expectation
            tf_additive_labels[is_single] = ddG_mb[is_single]
            has_add_label = ~torch.isnan(tf_additive_labels)
                
            if valid_double.any() and 'ddG_additive' in micro:
                valid_wt_mask[valid_double & torch.isnan(micro['ddG_additive'])] = False
                valid_add_doubles = valid_double & ~torch.isnan(micro['ddG_additive'])
                wt_targets[valid_add_doubles] = micro['ddG_additive'][valid_add_doubles].float()

            epi_targets = torch.zeros_like(ddG_mb)
            epi_mask = torch.zeros_like(is_single, dtype=torch.bool) if self.hparams.zero_epistasis_for_singles else is_single.clone()
            if valid_double.any() and 'dddG' in micro:
                valid_epi_doubles = valid_double & ~torch.isnan(micro['dddG'])
                epi_mask |= valid_epi_doubles
                epi_targets[epi_mask & valid_double] = micro['dddG'][epi_mask & valid_double].float()

            retain_wt = not self.hparams.detach_ensemble_input

            # =============================================================
            # PHASE 1: WILD-TYPE PASS
            # =============================================================
            wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=self.hparams.mask_strategy, detach_calibration=self.hparams.detach_regression)
            wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()

            wt_losses = []
            if self.hparams.lambda_reg_wt > 0 and valid_wt_mask.any():
                L = self.crit_reg(wt_pred_cal[valid_wt_mask], wt_targets[valid_wt_mask])
                wt_losses.append(self.hparams.lambda_reg_wt * (L * w_mb[valid_wt_mask]).sum() / global_w_sum)
                sums['reg_wt'] += float((L * w_mb[valid_wt_mask]).sum().item()); cnts['reg_wt'] += float(w_mb[valid_wt_mask].sum().item())

            if self.hparams.lambda_rank_wt > 0 and self.crit_rank_wt is not None:
                L_rank, val, n_list = self._compute_rank_loss(wt_pred_raw, wt_targets, valid_wt_mask, list_size, self.crit_rank_wt)
                if L_rank is not None:
                    wt_losses.append(self.hparams.lambda_rank_wt * L_rank * (n_list / global_num_lists))
                    sums['rank_wt'] += val * n_list; cnts['rank_wt'] += n_list

            if wt_losses and not self.peft_manager.wt_path_is_frozen and (self.hparams.lambda_rank_wt > 0 or self.hparams.lambda_reg_wt > 0):
                total_wt = sum(wt_losses)
                if not torch.isfinite(total_wt): raise AssertionError("WT Loss evaluated to NaN/Inf.")
                if not total_wt.requires_grad: raise AssertionError("WT Loss detached from PyTorch Graph! Cannot call backward.")
                self.manual_backward(total_wt, retain_graph=retain_wt)
                    
            if not self.peft_manager.mt_path_is_frozen and (self.hparams.lambda_rank_combined > 0 or self.hparams.lambda_reg_combined > 0 or self.hparams.lambda_epi_combined > 0):
                # =============================================================
                # PHASE 2: MUTANT & COMBINED PASS
                # =============================================================
                mt_out = self.model.forward_partitioned(micro, pass_type='mt', mask_strategy=self.hparams.mask_strategy, detach_calibration=self.hparams.detach_regression)
                mt_pred_cal, mt_pred_raw = mt_out['pred_calibrated'].float(), mt_out['pred_raw'].float()
                
                detach_ens = getattr(self.hparams, 'detach_ensemble_input', True)
                base_wt_cal = wt_pred_cal.detach() if detach_ens else wt_pred_cal
                base_wt_raw = wt_pred_raw.detach() if detach_ens else wt_pred_raw

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
                    b = 0.0 if cal_head.bias is None else cal_head.bias.detach()
                    decalibrated_label = (tf_additive_labels - b) / s
                    
                    forced_wt_cal = torch.where(has_add_label, tf_additive_labels, base_wt_cal)
                    forced_wt_raw = torch.where(has_add_label, decalibrated_label, base_wt_raw)
                elif self.model.lora_mode == 'corrector':
                    forced_wt_cal, forced_wt_raw = base_wt_cal, base_wt_raw
                else:
                    raise AssertionError(f"Unknown lora_mode: {self.model.lora_mode}")

                combined_pred_raw = 0.5 * forced_wt_raw + 0.5 * mt_pred_raw
                combined_pred_cal = 0.5 * forced_wt_cal + 0.5 * mt_pred_cal
                epi_pred = 0.5 * mt_pred_cal - 0.5 * forced_wt_cal
                
                mt_losses = []

                # Mut pass strictly relies on combined losses
                if self.hparams.lambda_reg_combined > 0 and reg_mask.any():
                    L = self.crit_reg(combined_pred_cal[reg_mask], ddG_mb[reg_mask])
                    mt_losses.append(self.hparams.lambda_reg_combined * (L * w_mb[reg_mask]).sum() / global_w_sum)
                    sums['reg_combined'] += float((L * w_mb[reg_mask]).sum().item()); cnts['reg_combined'] += float(w_mb[reg_mask].sum().item())

                if self.hparams.lambda_rank_combined > 0 and self.crit_rank_combined is not None:
                    L_rank, val, n_list = self._compute_rank_loss(combined_pred_raw, ddG_mb, reg_mask, list_size, self.crit_rank_combined)
                    if L_rank is not None:
                        mt_losses.append(self.hparams.lambda_rank_combined * L_rank * (n_list / global_num_lists))
                        sums['rank_combined'] += val * n_list; cnts['rank_combined'] += n_list

                if self.hparams.lambda_epi_combined > 0 and epi_mask.any():
                    L = self.crit_reg(epi_pred[epi_mask], epi_targets[epi_mask])
                    mt_losses.append(self.hparams.lambda_epi_combined * (L * w_mb[epi_mask]).sum() / global_w_sum)
                    sums['epi_combined'] += float((L * w_mb[epi_mask]).sum().item()); cnts['epi_combined'] += float(w_mb[epi_mask].sum().item())

                if mt_losses:
                    total_mt = sum(mt_losses)
                    if not torch.isfinite(total_mt): raise AssertionError("MT Loss evaluated to NaN/Inf.")
                    if not total_mt.requires_grad: raise AssertionError("MT Loss detached from PyTorch Graph! Cannot call backward.")
                    self.manual_backward(total_mt)

        logs = {}
        log_keys = ['reg_wt', 'rank_wt', 'reg_combined', 'rank_combined', 'epi_combined']
        for k in log_keys:
            if cnts[k] > 0: logs[f'L_{k}'] = sums[k] / max(1, cnts[k])
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
        self.model.train()
        logs = self._compose_losses_streaming_and_backward(utils._normalize_batch(batch))
        
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
        
        sch_warmup, sch_plateau = self.lr_schedulers()
        total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
        if sch_warmup.last_epoch < total_warmup_steps:
            sch_warmup.step()

        self._log_lrs()
        self.log_calibration_head(on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)
            
        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self.log("amp_scale", self.trainer.precision_plugin.scaler.get_scale(), on_step=True)

        # GPU memory every step, so allocator fragmentation (reserved climbing away from allocated) shows up in Comet
        if torch.cuda.is_available():
            gb = 1.0 / (1024 ** 3)
            self.log("mem/allocated_gb", torch.cuda.memory_allocated() * gb, on_step=True)
            self.log("mem/reserved_gb", torch.cuda.memory_reserved() * gb, on_step=True, prog_bar=True)
            self.log("mem/peak_allocated_gb", torch.cuda.max_memory_allocated() * gb, on_step=True)

        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        with torch.inference_mode():
            out_dict = self.model.forward_batch(batch, mask_strategy=getattr(self.hparams, 'mask_strategy', None))

        B = int(out_dict['wt_lora_pred'].shape[0])
        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        nan_col = lambda: np.full(B, np.nan, dtype=np.float32)
        get = lambda k: batch[k].detach().cpu().float().numpy() if torch.is_tensor(batch.get(k)) else nan_col()

        cyc = {}
        if getattr(self.hparams, 'val_cycle_passes', False):
            with torch.inference_mode():
                c = self.model.forward_cycle(batch, mask_strategy=getattr(self.hparams, 'mask_strategy', None))
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
            # singles.
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

        # Ungrouped metrics across all valid in-distribution samples
        if pooled['ground_truths']:
            g_wt = np.array(pooled['wt_scores']); g_comb = np.array(pooled['comb_scores'])
            g_gt = np.array(pooled['ground_truths']); g_src = np.array(pooled['cens_src'])
            ok = np.isfinite(g_gt) & (g_src != censoring.SRC_RANGE)
            if ok.any():
                self.log("val_rho_ungrouped_wt", stats.safe_spearman(g_wt[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
                self.log("val_rho_ungrouped_combined", stats.safe_spearman(g_comb[ok], g_gt[ok]), on_epoch=True, sync_dist=True)
            for _k, _v in (self._log_epistasis(pooled) or {}).items():
                if np.isfinite(_v):
                    avg_metrics[_k].append(float(_v))

        if hasattr(self, 'link_head'):
            for k, v in self.link_head.summary().items():
                self.log(f"link/{k}", v, on_epoch=True, sync_dist=True)

        if not self.trainer.sanity_checking and self.hparams.freeze_wt_on_convergence and not self.peft_manager.has_transitioned:
            target_metric_key = self.hparams.wt_convergence_metric
            if target_metric_key in avg_metrics:
                current_val = float(np.nanmean(avg_metrics[target_metric_key]))
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
                if self.trainer.global_step >= total_warmup_steps and avg_metrics['rho_combined']:
                    sch_plateau.step(np.nanmean(avg_metrics['rho_combined']))

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
        
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'lora' in k.lower() or 'calibration' in k.lower()
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
        loggers.append(CometLogger(api_key=args.comet_api_key, project=args.comet_project_name, name=f"{args.experiment_name}-{args.version or 'run'}"))

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
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders)

if __name__ == "__main__":
    main()