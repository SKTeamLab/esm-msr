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
from esm_msr import stats, epi_metrics, link as link_mod

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') 

# A library measured in a mutant background ('2KT8_V15A'): its singles are conditional effects, not wild-type-context ddGs, so rho_wt_valid
# leaves it out (as devel does, where they are 'native_cond').
NATIVE_BACKGROUND_CODE_RE = re.compile(r'_[A-Z][0-9]+[A-Z]$')

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
            lora_mode=self.hparams.lora_mode, model_dtype=torch.float32
        )
        
        self.peft_manager = PEFTStateManager(self.model)

        # Monotone saturating link (esm_msr.link): latent stability -> the assay's observed dG. Lives on the Lightning module, not on
        # self.model, so the released model's state and inference are unchanged; checkpoints keep it as link_head.*.
        self.link_head = None
        # --nonspecific_term fx: f(x) added to a multi-mutant's latent before the link (esm_msr.link.NonspecificTerm); kept as ns_head.*
        self.ns_head = None
        if self.hparams.get('nonspecific_term', 'none') == 'fx':
            self.ns_head = link_mod.NonspecificTerm(lo=float(self.hparams.get('ns_knot_lo', -3.0)), hi=float(self.hparams.get('ns_knot_hi', 5.0)),
                                                    step=float(self.hparams.get('ns_knot_step', 0.5)))
        if self.hparams.get('link', 'none') != 'none':
            self.link_head = link_mod.MonotoneLink(
                lo=self.hparams.link_lo, hi=self.hparams.link_hi, tau_lo=self.hparams.link_tau, tau_hi=self.hparams.link_tau,
                learn_bounds=bool(self.hparams.link_learn_bounds))

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

    def on_train_start(self):
        self._reshare_base_params_on_device()
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

        # --link: regression on the observed scale, observed ddG = h(dG_wt + latent) - dG_wt. Items without a measured dG_wt leave the
        # linked regressions (they keep their rank terms).
        link = self.link_head
        use_link = link is not None
        dGwt_all = batch['dG_wt'].float() if torch.is_tensor(batch.get('dG_wt')) else torch.full((B,), float('nan'), device=device)
        link_ok_all = torch.isfinite(dGwt_all) if use_link else torch.ones(B, dtype=torch.bool, device=device)
        dGwt_all = torch.where(torch.isfinite(dGwt_all), dGwt_all, torch.zeros_like(dGwt_all))
        # --link_floor_file: the library's own floor knee (0 = the shared floor). Data, like dG_wt; never a parameter.
        floor_all = batch['dG_floor_offset'].float() if torch.is_tensor(batch.get('dG_floor_offset')) else torch.zeros(B, device=device)
        floor_all = torch.where(torch.isfinite(floor_all), floor_all, torch.zeros_like(floor_all))

        sums, cnts = defaultdict(float), defaultdict(float)

        for start in range(0, B, mb):
            idx = torch.arange(start, min(start + mb, B), device=device)
            micro, w_mb = utils.slice_batch_by_index(batch, idx), w_all[idx]
            dGwt_mb, link_ok = dGwt_all[idx], link_ok_all[idx]
            floor_mb = floor_all[idx]
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

            tf_mode = getattr(self.hparams, 'tf_singles', 'measured')
            if tf_mode == 'latent_link' and use_link:
                # --tf_singles latent_link: each measured single back on the latent scale, h^-1(dG_wt + ddG_X) - dG_wt (detached), so a
                # near-floor single whose measurement is compressed is forced at its deeper latent value; a double's label is the sum of its two
                # at most --tf_latent_max_shift below the measurement: where h is flat (at or below the floor) the inverse is unbounded, and
                # the evidence (lift test, docs/epistasis_metrics_report.html) supports about 0.4-0.5 kcal/mol of compression at dG 0-0.5
                max_shift = float(getattr(self.hparams, 'tf_latent_max_shift', 0.5))
                with torch.no_grad():
                    inv = lambda y: torch.maximum(link.inverse(dGwt_mb + y, below_lo=50.0, lo_offset=floor_mb) - dGwt_mb, y - max_shift)
                    lat = torch.where(is_single, inv(ddG_mb), inv(micro['ddG_A'].float()) + inv(micro['ddG_B'].float()))
                tf_additive_labels = torch.where(has_add_label & link_ok & torch.isfinite(lat), lat, tf_additive_labels)

            if valid_double.any() and 'ddG_additive' in micro:
                valid_wt_mask[valid_double & torch.isnan(micro['ddG_additive'])] = False
                valid_add_doubles = valid_double & ~torch.isnan(micro['ddG_additive'])
                # the sum of the two singles: measured, or (latent_link) the same latent sum the forcing uses
                wt_targets[valid_add_doubles] = tf_additive_labels[valid_add_doubles]

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
            reg_wt_mask, p_wt = valid_wt_mask, wt_pred_cal
            if use_link:
                # singles are measurements of the item: through the link; a double's target is the sum of its singles: latent, as before
                p_wt = torch.where(is_single, link.obs_ddG(wt_pred_cal, dGwt_mb, lo_offset=floor_mb), wt_pred_cal)
                reg_wt_mask = valid_wt_mask & (link_ok | ~is_single)
            if self.hparams.lambda_reg_wt > 0 and reg_wt_mask.any():
                L = self.crit_reg(p_wt[reg_wt_mask], wt_targets[reg_wt_mask])
                wt_losses.append(self.hparams.lambda_reg_wt * (L * w_mb[reg_wt_mask]).sum() / global_w_sum)
                sums['reg_wt'] += float((L * w_mb[reg_wt_mask]).sum().item()); cnts['reg_wt'] += float(w_mb[reg_wt_mask].sum().item())

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
                    if tf_mode == 'predicted':
                        # --tf_singles predicted: the model's own latent single predictions, no label. The WT pass of a multi-mutant is
                        # scale * sum(LLR) + bias; the sum of its singles' predictions carries one bias per mutation
                        n_mut = micro['mut_mask'].sum(dim=1).float()
                        forced_wt_cal = base_wt_cal.detach() + b * (n_mut - 1.0)
                        forced_wt_raw = (forced_wt_cal - b) / s
                    else:
                        decalibrated_label = (tf_additive_labels - b) / s
                        forced_wt_cal = torch.where(has_add_label, tf_additive_labels, base_wt_cal)
                        forced_wt_raw = torch.where(has_add_label, decalibrated_label, base_wt_raw)
                elif self.model.lora_mode == 'corrector':
                    forced_wt_cal, forced_wt_raw = base_wt_cal, base_wt_raw
                else:
                    raise AssertionError(f"Unknown lora_mode: {self.model.lora_mode}")

                combined_pred_raw = 0.5 * forced_wt_raw + 0.5 * mt_pred_raw
                combined_pred_cal = 0.5 * forced_wt_cal + 0.5 * mt_pred_cal
                ns = getattr(self, 'ns_head', None)
                if ns is not None:
                    # nonspecific epistasis f(x) on multi-mutants only: x = dG_wt + the additive latent the forcing uses (the forced WT half;
                    # where that half is the WT pass of the multi-mutant, which carries the calibration bias once, add one bias per extra
                    # mutation). Detached: f does not train the WT half. Its own gradient comes from the combined losses.
                    ch = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
                    s_ns = 1.0 if ch is None or not ch.use_scale else ch.scale.detach()
                    b_ns = 0.0 if ch is None or ch.bias is None else ch.bias.detach()
                    n_mut = micro['mut_mask'].sum(dim=1).float()
                    from_pass = torch.ones_like(n_mut, dtype=torch.bool) if self.model.lora_mode != 'ensemble' else (
                        torch.zeros_like(n_mut, dtype=torch.bool) if tf_mode == 'predicted' else ~has_add_label)
                    x_hat = (dGwt_mb + forced_wt_cal + torch.where(from_pass, b_ns * (n_mut - 1.0), torch.zeros_like(n_mut))).detach()
                    f_ns = torch.where((n_mut > 1) & link_ok, ns(x_hat), torch.zeros_like(x_hat))
                    combined_pred_cal = combined_pred_cal + f_ns
                    combined_pred_raw = combined_pred_raw + f_ns / s_ns
                epi_pred = 0.5 * mt_pred_cal - 0.5 * forced_wt_cal
                
                mt_losses = []

                # Mut pass strictly relies on combined losses
                reg_c_mask, p_comb = reg_mask, combined_pred_cal
                if use_link:
                    # the teacher-forced combined latent (measured additive half + MT half) observed through the link
                    p_comb = link.obs_ddG(combined_pred_cal, dGwt_mb, lo_offset=floor_mb)
                    reg_c_mask = reg_mask & link_ok
                if self.hparams.lambda_reg_combined > 0 and reg_c_mask.any():
                    L = self.crit_reg(p_comb[reg_c_mask], ddG_mb[reg_c_mask])
                    mt_losses.append(self.hparams.lambda_reg_combined * (L * w_mb[reg_c_mask]).sum() / global_w_sum)
                    sums['reg_combined'] += float((L * w_mb[reg_c_mask]).sum().item()); cnts['reg_combined'] += float(w_mb[reg_c_mask].sum().item())

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
        if self.link_head is not None:
            for k, v in self.link_head.summary().items():
                self.log(f"link/{k}", v, on_step=True)
        if getattr(self, 'ns_head', None) is not None:
            with torch.no_grad():
                for xv in (-1.0, 0.0, 1.0, 2.0, 3.0):
                    self.log(f"ns/f_at_x{xv:+.0f}", float(self.ns_head(torch.tensor([xv], device=self.ns_head.values.device))[0]), on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)
            
        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self.log("amp_scale", self.trainer.precision_plugin.scaler.get_scale(), on_step=True)

        return torch.tensor(0.0, device=self.device)

    def _apply_nonspecific(self, out_dict, batch, dG_wt):
        """With --nonspecific_term: a multi-mutant's combined prediction gains f(x) and its MT readout 2 f(x) (so comb = (WT + MT) / 2 still
        holds), x = dG_wt + the WT pass (additive) + one calibration bias per extra mutation. Singles and items without dG_wt are untouched."""
        ns = getattr(self, 'ns_head', None)
        if ns is None:
            return out_dict
        dev = out_dict['combined_pred'].device
        mm = batch['mut_mask'].to(dev)
        n_mut = mm.sum(dim=1).float()
        dgw = dG_wt.to(dev).float()
        ok = (n_mut > 1) & torch.isfinite(dgw)
        ch = getattr(self.model, 'calibration_head_wt', getattr(self.model, 'calibration_head_fused', None))
        b = 0.0 if ch is None or ch.bias is None else ch.bias.detach()
        with torch.no_grad():
            x_hat = torch.where(ok, dgw, torch.zeros_like(dgw)) + out_dict['wt_lora_pred'].float() + b * (n_mut - 1.0)
            f = torch.where(ok, ns(x_hat), torch.zeros_like(x_hat))
        out = dict(out_dict)
        out['combined_pred'] = out_dict['combined_pred'].float() + f
        out['mt_lora_pred'] = out_dict['mt_lora_pred'].float() + 2.0 * f
        out['epi_pred'] = out_dict['epi_pred'].float() + f
        return out

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        with torch.inference_mode():
            out_dict = self.model.forward_batch(batch, mask_strategy=self.hparams.mask_strategy)

        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        dddG = batch.get('dddG', None)

        n_items = int(out_dict['wt_lora_pred'].shape[0])
        dG_wt = batch['dG_wt'].float() if torch.is_tensor(batch.get('dG_wt')) else torch.full((n_items,), float('nan'))
        floor_off = batch['dG_floor_offset'].float() if torch.is_tensor(batch.get('dG_floor_offset')) else torch.zeros(n_items)
        out_dict = self._apply_nonspecific(out_dict, batch, dG_wt)
        obs = {}
        if self.link_head is not None:
            # observed-scale predictions, h(dG_wt + latent) - dG_wt, for the rmse metrics only (rank metrics read the latent scores; the
            # epistasis metrics apply the link themselves). Items without a dG_wt (the external benchmarks, not measured by this assay) keep
            # their latent prediction.
            dgw = dG_wt.to(out_dict['combined_pred'].device)
            foff = floor_off.to(dgw.device)
            ok = torch.isfinite(dgw)
            z = torch.where(ok, dgw, torch.zeros_like(dgw))
            with torch.inference_mode():
                for k, v in (('wt_obs_scores', out_dict['wt_lora_pred']), ('comb_obs_scores', out_dict['combined_pred'])):
                    v = v.float()
                    obs[k] = torch.where(ok, self.link_head.obs_ddG(v, z, lo_offset=foff), v).detach().cpu().numpy()

        cyc = {}
        if getattr(self.hparams, 'val_cycle_passes', False):
            with torch.inference_mode():
                c = self.model.forward_cycle(batch, mask_strategy=self.hparams.mask_strategy)
            cyc = {'wt_rev_scores': c['wt_rev'].detach().cpu().float().numpy(), 'mt_fwd_scores': c['mt_fwd'].detach().cpu().float().numpy()}

        self.validation_step_outputs[dataloader_idx].append({
            **cyc, **obs,
            'dG_wt': dG_wt.detach().cpu().numpy(),
            'dG_floor_offset': floor_off.detach().cpu().numpy(),
            'subset_type': list(batch.get('subset_type', ['single'] * n_items)),
            'mt_scores': out_dict['mt_lora_pred'].detach().cpu().float().numpy(),
            'wt_scores': out_dict['wt_lora_pred'].detach().cpu().float().numpy(),
            'comb_scores': out_dict['combined_pred'].detach().cpu().float().numpy(),
            'epi_scores': out_dict['epi_pred'].detach().cpu().float().numpy(),
            'ground_truths': ddG.detach().cpu().float().numpy() if ddG is not None else np.array([]),
            'dddG_truths': dddG.detach().cpu().float().numpy() if dddG is not None else np.array([]),
            'pdb': batch.get('pdb', []),
            'mutations': batch.get('mutations', [])
        })

    def _log_epistasis(self, pooled):
        """
        Logs val_epi_<metric>_<head> (esm_msr.epi_metrics: every metric for comb, mt and ctx, the naive and global ones also for the additive
        control add), pooled over every validation library, and writes val_dump_<tag>.npz (rescore offline with baseline/epi_from_dump.py).
        Heads: comb ((WT + ~MT)/2, the reported prediction), mt (~MT), ctx ((WT + ~WT)/2, needs --val_cycle_passes), add (WT). With --link the
        metrics put each head on the observed scale through the run's link; without it the predictions are taken as observed. Never raises.
        """
        try:
            keys = pooled['mut_key']
            if not keys or any(k is None for k in keys):
                return
            f = lambda name: np.asarray(pooled[name], dtype=float)
            items = {'mut_key': keys, 'subset_type': list(pooled['subset_type']), 'cens': None, 'ddG': f('ground_truths'), 'dddG': f('dddG'),
                     'dG_wt': f('dG_wt'), 'wt': f('wt_scores'), 'mt': f('mt_scores'), 'comb': f('comb_scores'), 'wt_rev': f('wt_rev_scores'),
                     'dG_floor_offset': f('dG_floor_offset')}
            log_dir = getattr(self.logger, 'log_dir', None) or getattr(self.logger, 'save_dir', None)
            if log_dir:
                tag = 'zs' if self.trainer.sanity_checking or self.global_step == 0 else f'e{self.current_epoch}'
                os.makedirs(log_dir, exist_ok=True)
                extra = {'link_summary': np.array(json.dumps(self.link_head.summary()))} if self.link_head is not None else {}
                if getattr(self, 'ns_head', None) is not None:
                    extra['ns_term'] = np.array(json.dumps(self.ns_head.table()))
                np.savez_compressed(os.path.join(log_dir, f'val_dump_{tag}.npz'), wt=items['wt'], mt=items['mt'], comb=items['comb'],
                                    wt_rev=items['wt_rev'], mt_fwd=f('mt_fwd_scores'), dddG=items['dddG'], n_mut=np.asarray(pooled['n_mut']),
                                    ground_truth=items['ddG'], dG_wt=items['dG_wt'], subset_type=np.asarray(items['subset_type'], dtype=str),
                                    loader=np.asarray(pooled['loader']), mut_key=np.asarray([json.dumps(k) for k in keys]),
                                    dG_floor_offset=items['dG_floor_offset'], **extra)
            link = self.link_head.numpy() if self.link_head is not None else None
            res = epi_metrics.compute(epi_metrics.Table(items, link))
            for name, val in res.items():
                if np.isfinite(val):
                    self.log(f"val_{name}", float(val), on_epoch=True, prog_bar=name == 'epi_cell_rank_mt', sync_dist=True)
        except Exception as e:
            logging.warning(f"epistasis metrics skipped: {e}", exc_info=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_metrics = defaultdict(list)
        
        # Pooled storage to compute ungrouped/global metrics at the end
        pooled_data = defaultdict(list)
        
        for dataloader_idx, valid_outputs in self.validation_step_outputs.items():
            val_loader_name = self.val_dataloader_names[dataloader_idx] if dataloader_idx < len(self.val_dataloader_names) else f"unknown_dl_{dataloader_idx}"

            single_dict_comb = {}
            wt_scores_list, comb_scores_list, epi_scores_list = [], [], []
            ground_truths_list, dddG_truths_list, mut_lens_list = [], [], []
            
            flat_mut_parts, flat_pdb_ids = [], []
            mt_scores_list, wt_rev_list, mt_fwd_list = [], [], []
            dG_wt_list, subset_list, wt_obs_list, comb_obs_list = [], [], [], []

            for o in valid_outputs:
                pdb_list, muts_list = o.get('pdb', []), o.get('mutations', [])
                for i, m_val in enumerate(muts_list):
                    mut_parts = m_val.split(':') if isinstance(m_val, str) else m_val
                    pdb_id = pdb_list[i] if (isinstance(pdb_list, (list, tuple, np.ndarray)) and i < len(pdb_list)) else (pdb_list if isinstance(pdb_list, str) else "unknown")

                    wt_scores_list.append(o['wt_scores'][i])
                    mt_scores_list.append(o['mt_scores'][i])
                    wt_rev_list.append(o['wt_rev_scores'][i] if 'wt_rev_scores' in o else np.nan)
                    mt_fwd_list.append(o['mt_fwd_scores'][i] if 'mt_fwd_scores' in o else np.nan)
                    dG_wt_list.append(o['dG_wt'][i] if 'dG_wt' in o else np.nan)
                    subset_list.append(o['subset_type'][i] if 'subset_type' in o else ('single' if len(mut_parts) == 1 else 'double'))
                    wt_obs_list.append(o['wt_obs_scores'][i] if 'wt_obs_scores' in o else np.nan)
                    comb_obs_list.append(o['comb_obs_scores'][i] if 'comb_obs_scores' in o else np.nan)
                    comb_scores_list.append(o['comb_scores'][i])
                    epi_scores_list.append(o['epi_scores'][i])
                    ground_truths_list.append(o['ground_truths'][i] if i < len(o['ground_truths']) else np.nan)
                    dddG_truths_list.append(o['dddG_truths'][i] if i < len(o['dddG_truths']) else np.nan)
                    mut_lens_list.append(len(mut_parts))
                    
                    flat_mut_parts.append(mut_parts)
                    flat_pdb_ids.append(pdb_id)
                    
                    if len(mut_parts) == 1:
                        key = (pdb_id, str(mut_parts[0]))
                        single_dict_comb[key] = o['comb_scores'][i]

            epi_scores_list_full = []

            for i, mut_parts in enumerate(flat_mut_parts):
                if len(mut_parts) > 1:
                    pdb_id = flat_pdb_ids[i]
                    add_comb = sum([single_dict_comb.get((pdb_id, str(m)), np.nan) for m in mut_parts])
                    epi_scores_list_full.append(comb_scores_list[i] - add_comb)
                else:
                    epi_scores_list_full.append(0.0)
                        
            wt_scores, comb_scores = np.array(wt_scores_list), np.array(comb_scores_list)
            epi_scores, epi_scores_full, ground_truths, dddG_truths = np.array(epi_scores_list), np.array(epi_scores_list_full), np.array(ground_truths_list), np.array(dddG_truths_list)
            idx_singles, idx_doubles = np.array(mut_lens_list) == 1, np.array(mut_lens_list) >= 2
            
            valid_dddG_mask = idx_doubles & ~np.isnan(dddG_truths) & ~np.isnan(epi_scores)

            metrics = stats.compute_metrics(
                wt_scores, comb_scores, epi_scores, epi_scores_full,
                ground_truths, dddG_truths, idx_singles, idx_doubles, valid_dddG_mask
            )
            if self.link_head is not None:
                # with --link, calibration is judged on the assay's observed scale (rank metrics are unchanged: h is monotone)
                for pathway, ob in (('wt', np.array(wt_obs_list)), ('combined', np.array(comb_obs_list))):
                    ok = np.isfinite(ob) & np.isfinite(ground_truths)
                    metrics['rmse'][pathway] = float(np.sqrt(np.mean((ob[ok] - ground_truths[ok]) ** 2))) if ok.any() else np.nan
            # the WT head on plain singles in the library's own wild-type context (devel's rho_wt_valid; NaN for a mutant-background library)
            sel = np.array([st == 'single' for st in subset_list]) & idx_singles & np.isfinite(ground_truths)
            rho_wt_valid = np.nan if NATIVE_BACKGROUND_CODE_RE.search(str(val_loader_name)) else stats.safe_spearman(wt_scores[sel], ground_truths[sel])
            if not np.isnan(rho_wt_valid):
                self.log(f"val_rho_wt_valid/{val_loader_name}", rho_wt_valid, on_epoch=True, sync_dist=True)
                avg_metrics['rho_wt_valid'].append(rho_wt_valid)
            
            # Aggregate pooled data for ungrouped metric calculation
            pooled_data['wt_scores'].extend(wt_scores_list)
            pooled_data['comb_scores'].extend(comb_scores_list)
            pooled_data['ground_truths'].extend(ground_truths_list)
            # for the epistasis metrics (esm_msr.epi_metrics): per-item values of every pass, the mutation count and a library-tagged key
            pooled_data['mt_scores'].extend(mt_scores_list)
            pooled_data['wt_rev_scores'].extend(wt_rev_list)
            pooled_data['mt_fwd_scores'].extend(mt_fwd_list)
            pooled_data['dG_wt'].extend(dG_wt_list)
            pooled_data['subset_type'].extend(subset_list)
            pooled_data['dddG'].extend(dddG_truths_list)
            pooled_data['n_mut'].extend(mut_lens_list)
            pooled_data['loader'].extend([val_loader_name] * len(mut_lens_list))
            pooled_data['mut_key'].extend([tuple((val_loader_name,) + tuple(m) for m in mp) if all(isinstance(m, (tuple, list)) for m in mp) else None
                                           for mp in flat_mut_parts])

            # Log metrics avoiding MUT-specific grouped ones and grouped NDCGs
            for metric_type in ['rho', 'rho_singles', 'rho_doubles', 'rho_dddG_heuristic', 'rho_dddG', 'rmse', 'ndcg@k=96', 'ndcg>0']: 
                if metric_type in metrics:
                    for pathway, val in metrics[metric_type].items():
                        if pathway == 'mt': continue # Skip all standalone mut pass metrics
                        if not np.isnan(val): self.log(f"val_{metric_type}_{pathway}/{val_loader_name}", val, on_epoch=True, sync_dist=True)

            for metric_type, pathways in metrics.items(): 
                for pathway, val in pathways.items():     
                    if not np.isnan(val): avg_metrics[f"{metric_type}_{pathway}"].append(val)

        for key, values in avg_metrics.items():
            if values: self.log(f'val_{key}_avg', np.nanmean(values), on_epoch=True, prog_bar=True, sync_dist=True)

        # Compute Ungrouped metrics globally across all valid samples
        if pooled_data['ground_truths']:
            g_wt = np.array(pooled_data['wt_scores'])
            g_comb = np.array(pooled_data['comb_scores'])
            g_gt = np.array(pooled_data['ground_truths'])
            
            valid_mask = ~np.isnan(g_gt)
            if valid_mask.any():
                ungrouped_rho = {
                    'wt': stats.safe_spearman(g_wt[valid_mask], g_gt[valid_mask]),
                    'combined': stats.safe_spearman(g_comb[valid_mask], g_gt[valid_mask])
                }
                ungrouped_ndcg_k96 = {
                    'wt': stats.safe_ndcg_k96(g_wt[valid_mask], g_gt[valid_mask]),
                    'combined': stats.safe_ndcg_k96(g_comb[valid_mask], g_gt[valid_mask])
                }
                ungrouped_ndcg_t0 = {
                    'wt': stats.safe_ndcg_t0(g_wt[valid_mask], g_gt[valid_mask]),
                    'combined': stats.safe_ndcg_t0(g_comb[valid_mask], g_gt[valid_mask])
                }
                
                for pathway, val in ungrouped_rho.items():
                    if not np.isnan(val): self.log(f"val_rho_ungrouped_{pathway}", val, on_epoch=True, sync_dist=True)
                for pathway, val in ungrouped_ndcg_k96.items():
                    if not np.isnan(val): self.log(f"val_ndcg@k=96_ungrouped_{pathway}", val, on_epoch=True, sync_dist=True)
                for pathway, val in ungrouped_ndcg_t0.items():
                    if not np.isnan(val): self.log(f"val_ndcg@k>0_ungrouped_{pathway}", val, on_epoch=True, sync_dist=True)

        ESM3EpistasisLightningModule._log_epistasis(self, pooled_data)

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
        if self.link_head is not None:
            link_params = [p for p in self.link_head.parameters() if p.requires_grad]
            main_groups.append({"params": link_params, "lr": float(self.hparams.link_lr), "weight_decay": 0.0, "name": "link"})
        if getattr(self, 'ns_head', None) is not None:
            main_groups.append({"params": list(self.ns_head.parameters()), "lr": float(self.hparams.get('ns_lr', 5e-3)), "weight_decay": 0.0,
                                "name": "nonspecific"})

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=True, weight_decay=wd)

        lambdas_main = []
        for g in main_groups:
            if "calib" in g["name"] or g["name"] in ("link", "nonspecific"): lambdas_main.append(lambda step: 1.0)
            elif g["name"] == "lora_mt":
                lambdas_main.append(lambda step, delay=self.hparams.mt_lora_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))
            else:
                lambdas_main.append(lambda step, delay=self.hparams.calib_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))

        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        # the link's group keeps its learning rate (its min_lr is its lr); every other group as released
        min_lrs = [float(g["lr"]) if g["name"] in ("link", "nonspecific") else 1e-7 for g in main_groups]
        plateau_main = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_main, mode='max', factor=0.1, patience=1, min_lr=min_lrs)

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
            if 'lora' in k.lower() or 'calibration' in k.lower() or k.startswith('link_head.') or k.startswith('ns_head.')
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

    if not args.skip_val: trainer.validate(lightning_model, dataloaders=val_loaders)
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders)

if __name__ == "__main__":
    main()