# -*- coding: utf-8 -*-

import os
import argparse
import math
import warnings
import logging
from collections import defaultdict
import comet_ml
import gc

from typing import List, Dict, Any, Optional, Tuple

from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_squared_error

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
from pytorch_lightning.loggers import CometLogger, CSVLogger

from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from esm_msr.models import ESM3LoRAModel
from esm_msr import utils

from esm_msr.data import ( # Data loading utilities
    ProteinStructureMutationEpistasisDatasetChainRule,
    collate_fn_chainrule,
    ProteinCyclingDataLoader,
    SubsetRestrictedProteinCyclingDataLoader,
    PooledDataLoader
)
from esm_msr.losses import ListMLELoss, ListMLELoss_enhanced, AsymmetricHuberLoss

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') # or 'medium'


class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(
        self,
        # --- core model cfg ---
        lora_rank: int = 6,
        lora_alpha: int = 12,
        lora_dropout: float = 0.15,
        include_structure_encoder: bool = False,
        use_qkv_lora: bool = False,
        freeze_lora: bool = False,
        calibration: str = 'linear',
        use_dora: bool = False,
        mean_ll: bool = False,
        use_plddt: bool = False,
        # --- calibration cfg ---
        shared_scale=0.33,
        shared_bias=0,
        single_scale=None,
        single_bias=None,
        mutctx_scale=None,
        mutctx_bias=None,
        reversion_scale=None,
        reversion_bias=None, 
        # --- optimization ---
        learning_rate: float = 3e-5,
        lr_warmup_steps: int = 2000,
        lr_total_steps: Optional[int] = None,
        batch_size: int = 256,
        effective_batch_size: int = 256,
        precision: str = "16-mixed",
        # --- losses & sampling ---
        lambda_rank: float = 1.0,
        lambda_reg: float = 0.2,
        lambda_epi: float = 0.0,
        subset_size: int = 16,
        invert_list_loss: bool = False,
        rank_loss: str = 'listmle',
        reg_loss: str = 'huber', 
        epi_loss: str = 'huber',  # 'huber' | 'mse',
        rank_tau_init: float | None = None,
        double_weight: float = 1.0,
        mut_ctx_weight: float = 0.5,
        reversion_weight: float = 0.1,
        lambda_head_tie: float = 0.1,
        singles_only: bool = False,
        detach_regression: bool = False,
        # --- masking ---
        mask_sequence_fraction: float=0.0,
        mask_structure_fraction: float=0.0,
        mask_coords_fraction: float=0.0,
        # --- scoring --- 
        additive_condition: str = 'wt',
        effect_strategy: str = 'chain_rule_avg',
        # --- misc ---
        seed: int = 42,
        train_dataloader_names: List[str] = ['train'],
        val_dataloader_names: List[str] = ['val'],
        resume_global_step: int = 0,
        model_device: str = 'cuda:0',
        tokenizer = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.rng = np.random.default_rng(seed)

        # Validate tokenizer assumptions once at startup
        self.tokenizer = tokenizer

        # Model
        base_model = ESM3_sm_open_v0()
        peft_model = utils.add_lora_to_esm3(
            base_model,
            lora_rank=self.hparams.lora_rank,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_mode='all' if self.hparams.use_qkv_lora else 'expanded',
            use_dora=self.hparams.use_dora,
            seed=seed,
            include_structure_encoder=self.hparams.include_structure_encoder
        )
        self.model = ESM3LoRAModel(
            peft_model, 
            freeze_lora=freeze_lora,
            shared_scale=shared_scale,
            shared_bias=shared_bias,
            single_scale=single_scale,
            single_bias=single_bias,
            mutctx_scale=mutctx_scale,
            mutctx_bias=mutctx_bias,
            reversion_scale=reversion_scale,
            reversion_bias=reversion_bias,
            use_plddt=use_plddt,         
            ).to(model_device) #, add_epistasis_head=(effect_strategy=='epistasis_head'))
        
        utils.print_trainable_parameters(self.model)
        self._train_epi_preds = []
        self._train_epi_truths= []

        # Losses
        if rank_loss == 'listmle':
            self.crit_rank = ListMLELoss(invert=self.hparams.invert_list_loss) if self.hparams.lambda_rank>0 else None
            self.crit_rank2 = ListMLELoss(invert=not self.hparams.invert_list_loss) if self.hparams.lambda_rank>0 else None
        elif rank_loss == 'listmle_enhanced':
            self.crit_rank = ListMLELoss_enhanced() if self.hparams.lambda_rank>0 else None

        if reg_loss == 'huber':
            self.crit_reg = nn.HuberLoss(reduction='none')
        elif reg_loss == 'mse':
            self.crit_reg = nn.MSELoss(reduction='none')
        elif reg_loss == 'asymmetric':
            self.crit_reg = AsymmetricHuberLoss()

        if epi_loss == 'listmle':
            self.crit_epi = ListMLELoss() if self.hparams.lambda_epi>0 else None
        elif epi_loss == 'mse':
            self.crit_epi = nn.MSELoss()
        else:
            epi_loss = 'huber'
            self.crit_epi = nn.HuberLoss()
        self.crit_const = nn.MSELoss()

        # Training config
        self.automatic_optimization = False
        self.accumulate_grad_batches = max(1, effective_batch_size // batch_size)
        self.validation_step_outputs = defaultdict(list)
        self.first_batch = True

        # learnable raw temp -> tau = softplus(raw) + tau_min
        if rank_tau_init is not None:
            self.tau0 = torch.tensor(rank_tau_init, dtype=torch.float32)
            self.rank_tau_raw = nn.Parameter(torch.log(torch.expm1(self.tau0.clamp_min(1e-8))))
        else:
            self.tau0 = None

        flank_seq=getattr(self.hparams, "flank_seq", 0)
        flank_struct=getattr(self.hparams, "flank_struct", 0)
        flank_coords=getattr(self.hparams, "flank_coords", 0)

        logging.info(f'Masking sequence at mutated position(s): {self.hparams.mask_sequence_pos}')
        logging.info(f'Masking {int(mask_sequence_fraction*100)} percent of sequence tokens')
        logging.info(f'Masking {flank_seq} surrounding residue identities')
        logging.info(f'Masking structure tokens at mutated position(s): {self.hparams.mask_structure_pos}')
        logging.info(f'Masking {int(mask_structure_fraction*100)} percent of structure tokens')
        logging.info(f'Masking {flank_struct} surrounding structure tokens')
        logging.info(f'Masking all coordinates at mutated position(s): {self.hparams.mask_coords_pos}')
        logging.info(f'Masking {int(mask_coords_fraction*100)} percent of coords')
        logging.info(f'Masking {flank_coords} surrounding structure tokens')
        logging.info(f'Using double mutant prediction strategy: {self.hparams.effect_strategy}')
        logging.info(f'Replacing non-mutated residue when evaluating additive doubles: {additive_condition}')

    def get_rank_tau(self):
        # keep positive, bounded away from 0
        return torch.nn.functional.softplus(self.rank_tau_raw) + 1e-3

    def _calibrate(self, uncal, head='single', active=True, allow_shared_head=True, detach_reg=False):
        if detach_reg:
            uncal = uncal.detach()

        attr_name = f'calibration_head_{head}'
        
        if hasattr(self.model, attr_name) and active:
            cal_head = getattr(self.model, attr_name)
            cal = cal_head(uncal)
        elif not hasattr(self.model, attr_name) and active and not allow_shared_head:
            raise AssertionError(f'Requested to calibrate via {head}, but no calibration head is present')
        elif hasattr(self.model, 'calibration_head_shared') and active and allow_shared_head:
            cal_head = getattr(self.model, 'calibration_head_shared')
            cal = cal_head(uncal)
        elif not active:
            cal = uncal
        else:
            raise AssertionError('Model may have no calibration heads; add by specifying init weights')
        return cal
    
    def calibrate_split_paths_singles(
        self,
        batch_in: Dict[str, Any],
        preds_uncal: torch.Tensor,
        detach_reg: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply calibration AFTER a raw forward.
        - Uses per-route heads if available; falls back to shared head.
        - Keeps NaNs in slots where a quantity is undefined (e.g., additive for singles/multis).
        - `pred_*` are not mutated in-place; returns new tensors.
        """
        device = preds_uncal.device
        if batch_in.get('subset_type'):
            st = batch_in['subset_type']
            idx_single = torch.as_tensor([i for i, s in enumerate(st) if s == 'single'], device=device, dtype=torch.long)
            idx_double = torch.as_tensor([i for i, s in enumerate(st) if s == 'double'], device=device, dtype=torch.long)
            idx_mut_ctx = torch.as_tensor([i for i, s in enumerate(st) if s == 'mut_ctx'], device=device, dtype=torch.long)
            idx_reversion = torch.as_tensor([i for i, s in enumerate(st) if s == 'reversion'], device=device, dtype=torch.long)
        else:
            # must be validation set. There should be only singles. Doubles are calibrated separately.
            muts = batch_in['mutations']
            idx_single = torch.tensor([i for i, m in enumerate(muts) if len(m) == 1], device=device, dtype=torch.long)
            idx_double = torch.tensor([i for i, m in enumerate(muts) if len(m) == 2], device=device, dtype=torch.long)  
            idx_mut_ctx = torch.tensor([], device=device, dtype=torch.long)
            idx_reversion = torch.tensor([], device=device, dtype=torch.long)
        assert idx_double.numel() == 0, "Doubles should not be passed to calibrate_split_paths_singles"    
        assert len(torch.cat([idx_single, idx_mut_ctx, idx_reversion])) == len(batch_in['mutations']), (idx_single, idx_mut_ctx, idx_reversion)

        # clone to avoid in-place surprises
        preds_cal = preds_uncal.clone()

        # ---- Singles and non-epistasis doubles path (unchanged intent)
        if idx_single.numel() > 0 and torch.isfinite(preds_cal[idx_single]).any():
            block = preds_cal[idx_single]
            preds_cal.index_copy_(0, idx_single, self._calibrate(block, head='single', allow_shared_head=True, detach_reg=detach_reg))

        if idx_mut_ctx.numel() > 0 and torch.isfinite(preds_cal[idx_mut_ctx]).any():
            block = preds_cal[idx_mut_ctx]
            preds_cal.index_copy_(0, idx_mut_ctx, self._calibrate(block, head='mut_ctx', allow_shared_head=True, detach_reg=detach_reg))
            
        if idx_reversion.numel() > 0 and torch.isfinite(preds_cal[idx_reversion]).any():
            block = preds_cal[idx_reversion]
            preds_cal.index_copy_(0, idx_reversion, self._calibrate(block, head='reversion', allow_shared_head=True, detach_reg=detach_reg))

        return preds_cal

    # =============================
    # Doubles: two-stage scoring (effect + additive) with epistasis as their delta
    # =============================
    def forward_doubles_and_calibrate(
        self,
        batch_doubles_orig: Dict[str, Any],
        *,
        # masking config (decided by caller)
        mask_sequence_pos: bool = True,
        mask_structure_pos: bool = False,
        mask_coords_pos: bool = False,
        mask_sequence_fraction: float = 0.0, 
        mask_structure_fraction: float = 0.0, 
        mask_coords_fraction: float = 0.0, 
        flank_seq: int = 0,
        flank_struct: int = 0,
        flank_coords: int = 0,
        struct_mask_id: int = 0,
        # strategies
        effect_strategy: str = "chain_rule_avg",   # {"direct_forward", "chain_rule_random", "chain_rule_avg"}
        detach_reg: bool = False,
        compute_epi: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generally used for validation only
        Two-stage doubles scorer:
        1) EFFECT PREDICTION (double-mutation effect):
            - "direct_forward":     forward(AB) after apply_masks (no extra preprocessing)
            - "chain_rule_random":  (ΔA|B=WT + ΔB|A=mut) OR (ΔB|A=WT + ΔA|B=mut)
            - "chain_rule_avg":     0.5*[ (ΔA|B=WT + ΔB|A=mut) + (ΔB|A=WT + ΔA|B=mut) ]

        2) EPISTASIS CORRECTION (additive baseline for ΔA + ΔB):
            - additive_condition in {"mask","wt","mut",None} via one helper that only varies the condition.
            - If None, no additive or epistasis predictions (saves compute)

        Returns:
            (effect_pred [Nd], additive_pred [Nd], epi_pred [Nd]), where epi_pred = effect_pred - additive_pred.

        Notes:
        • This function will apply sequence/structure/coords masks as requested by calling `apply_masks(...)`.
        • The conditional singles builder rewrites mutations to length-1 per row (scored site),
            and accepts unmasked inputs (it applies the required masking/fills itself).
        • Structure/coords are left untouched here (sequence-only conditioning).
        """
        # ---- sanity: doubles only ----
        if 'mutations' not in batch_doubles_orig:
            raise AssertionError("forward_doubles_and_calibrate expects 'mutations' in batch.")
        for m in batch_doubles_orig['mutations']:
            assert len(m) == 2, "forward_doubles_and_calibrate expects doubles-only rows."

        # ---- Stage 0: build AB-masked batch from originals (consistency anchor) ----
        batch_AB = utils.apply_masks(
            batch_doubles_orig,
            mask_sequence_pos=mask_sequence_pos, 
            mask_structure_pos=mask_structure_pos,
            mask_coords_pos=mask_coords_pos, 
            mask_sequence_fraction=mask_sequence_fraction, 
            mask_structure_fraction=mask_structure_fraction, 
            mask_coords_fraction=mask_coords_fraction, 
            flank_seq=flank_seq,
            flank_struct=flank_struct,
            flank_coords=flank_coords,
            struct_mask_id=struct_mask_id,
            from_originals=True,
            tokenizer=self.tokenizer,
            skip=self.hparams.mean_ll
        )

        if self.first_batch:
            self.first_batch = False
            import pickle
            logging.info('Writing first batch to file')
            with open('./first_batch.pkl', 'wb') as file:
                pickle.dump(batch_AB, file, protocol=None, fix_imports=True, buffer_callback=None)

        additive_pred_raw = None
        additive_pred_cal = None
        epi_pred_cal = None
        dA = None
        dB = None
        dA_cal = None
        dB_cal = None

        if compute_epi:

            A_wt = utils.make_conditional_batch_doubles(batch_AB, which='A', condition='wt', tokenizer=self.tokenizer)
            B_wt = utils.make_conditional_batch_doubles(batch_AB, which='B', condition='wt', tokenizer=self.tokenizer)

            with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                dA = self.model.forward(A_wt).view(-1)
                dB = self.model.forward(B_wt).view(-1)
                additive_pred_raw = dA + dB

                dA_cal = self._calibrate(dA, head='single', allow_shared_head=True, detach_reg=detach_reg)
                dB_cal = self._calibrate(dB, head='single', allow_shared_head=True, detach_reg=detach_reg)
                additive_pred_cal = dA_cal + dB_cal

        if effect_strategy == "direct_forward":

            # exactly one forward on AB after apply_masks
            with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                effect_pred_uncal = self.model.forward(batch_AB).view(-1)
                effect_pred_cal = self._calibrate(effect_pred_uncal, head='double', allow_shared_head=True, detach_reg=detach_reg)

            if compute_epi:
                epi_pred_cal = effect_pred_cal - additive_pred_cal

        elif effect_strategy == "chain_rule_random":
            # Choose random path for this batch
            use_path1 = torch.rand(1) > 0.5
            
            if use_path1:
                # Path 1: B → AB
                B_wt = utils.make_conditional_batch_doubles(batch_AB, which='B', condition='wt', tokenizer=self.tokenizer)
                A_Bmt = utils.make_conditional_batch_doubles(batch_AB, which='A', condition='mut', tokenizer=self.tokenizer)

                with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                    if not compute_epi:
                        dB = self.model.forward(B_wt).view(-1)
                        dB_cal = self._calibrate(dB, head='single', allow_shared_head=True, detach_reg=detach_reg)
                    dA_Bmt_raw = self.model.forward(A_Bmt).view(-1)
                    dA_Bmt_cal = self._calibrate(dA_Bmt_raw, head='mut_ctx', allow_shared_head=True, detach_reg=detach_reg)
                    effect_pred_uncal = dA_Bmt_raw + dB
                    effect_pred_cal = dA_Bmt_cal + dB_cal

                if compute_epi:
                    epi_pred_cal = dA_Bmt_cal - dA_cal
        
            else:
                # Path 1: A → AB
                A_wt = utils.make_conditional_batch_doubles(batch_AB, which='A', condition='wt', tokenizer=self.tokenizer)
                B_Amt = utils.make_conditional_batch_doubles(batch_AB, which='B', condition='mut', tokenizer=self.tokenizer)

                with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                    if not compute_epi:
                        dA = self.model.forward(A_wt).view(-1)
                        dA_cal = self._calibrate(dA, head='single', allow_shared_head=True, detach_reg=detach_reg)
                    dB_Amt_raw = self.model.forward(B_Amt).view(-1)
                    dB_Amt_cal = self._calibrate(dB_Amt_raw, head='mut_ctx', allow_shared_head=True, detach_reg=detach_reg)
                    effect_pred_uncal = dB_Amt_raw + dA
                    effect_pred_cal = dB_Amt_cal + dA_cal

                if compute_epi:
                    epi_pred_cal = dB_Amt_cal - dB_cal                 
                         
        elif effect_strategy == "chain_rule_avg":
            # Build all four conditional batches once from AB

            A_Bmt = utils.make_conditional_batch_doubles(batch_AB, which='A', condition='mut', tokenizer=self.tokenizer)
            B_Amt = utils.make_conditional_batch_doubles(batch_AB, which='B', condition='mut', tokenizer=self.tokenizer)

            if not compute_epi:
                A_wt = utils.make_conditional_batch_doubles(batch_AB, which='A', condition='wt', tokenizer=self.tokenizer)
                B_wt = utils.make_conditional_batch_doubles(batch_AB, which='B', condition='wt', tokenizer=self.tokenizer)

                with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                    dA = self.model.forward(A_wt).view(-1)
                    dB = self.model.forward(B_wt).view(-1)
                    additive_pred_raw = dA + dB

                    dA_cal = self._calibrate(dA, head='single', allow_shared_head=True, detach_reg=detach_reg)
                    dB_cal = self._calibrate(dB, head='single', allow_shared_head=True, detach_reg=detach_reg)
                    additive_pred_cal = dA_cal + dB_cal                

            with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                dA_Bmt_raw = self.model.forward(A_Bmt).view(-1)
                dB_Amt_raw = self.model.forward(B_Amt).view(-1)
                dA_Bmt_cal = self._calibrate(dA_Bmt_raw, head='mut_ctx', allow_shared_head=True, detach_reg=detach_reg)
                dB_Amt_cal = self._calibrate(dB_Amt_raw, head='mut_ctx', allow_shared_head=True, detach_reg=detach_reg)

                # order-averaged chain rule
                effect_pred_uncal = 0.5 * (dB_Amt_raw + dA_Bmt_raw + additive_pred_raw)
                effect_pred_cal = 0.5 * (dB_Amt_cal + dA_Bmt_cal + additive_pred_cal)
                epi_pred_cal = effect_pred_cal - additive_pred_cal

        else:
            raise ValueError("effect_strategy must be one of {'direct_forward', 'chain_rule_random', chain_rule_avg'}")
        
        return effect_pred_uncal, effect_pred_cal, additive_pred_cal, epi_pred_cal

    # =============================
    # Mixed-batch convenience: forward_split_paths_and_calibrate
    # =============================
    def forward_split_paths_and_calibrate(
        self,
        batch_in: Dict[str, Any],
        *,
        # global masking config (used for singles/multis; doubles own strategy inside forward_doubles_and_calibrate)
        # doubles strategy
        effect_strategy: str = "chain_rule_avg",
        validation: bool = False,
        detach_reg: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Put in a *mixed* batch and get back predictions in the same order.
        - If the batch does *not* have masked tensors, this will mask them from *_orig* using the
        provided config.
        - Doubles are routed to `forward_doubles_and_calibrate` (with chosen strategy).
        - Returns: (pred [B], pred_epi [B]) where pred_epi is NaN for non-doubles.
        """
        device = batch_in['ddG'].device
        B = int(batch_in['ddG'].shape[0]) if 'ddG' in batch_in else \
            int(batch_in['sequence_tokens'].shape[0])

        # Partition indices by mutation count
        muts = batch_in['mutations']
        assert len(muts) == B, "mutations length must equal batch size."

        # Partition indices by mutation count
        idx_single = torch.tensor([i for i, m in enumerate(muts) if len(m) == 1], device=device, dtype=torch.long)
        idx_double = torch.tensor([i for i, m in enumerate(muts) if len(m) == 2], device=device, dtype=torch.long)

        pred_uncal = torch.full((len(muts),), float('nan'), dtype=torch.float32, device=device)
        pred_cal = torch.full((len(muts),), float('nan'), dtype=torch.float32, device=device)
        pred_add = torch.full((len(muts),), float('nan'), dtype=torch.float32, device=device)
        pred_epi = torch.full((len(muts),), float('nan'), dtype=torch.float32, device=device)

        # Singles
        # All singles go through the same forward, but are calibrated depending on their subset type (e.g. mut_ctx, reversion)
        if idx_single.numel() > 0:
            b_s = utils.slice_batch_by_index(batch_in, idx_single)
            # Mask per config (singles only have one position)
            b_s_m = utils.apply_masks(
                b_s,
                mask_sequence_pos=self.hparams.mask_sequence_pos, 
                mask_structure_pos=self.hparams.mask_structure_pos, 
                mask_coords_pos=self.hparams.mask_coords_pos, 
                mask_sequence_fraction=self.hparams.mask_sequence_fraction, #if not validation else 0.0, 
                mask_structure_fraction=self.hparams.mask_structure_fraction, #if not validation else 0.0, 
                mask_coords_fraction=self.hparams.mask_coords_fraction, #if not validation else 0.0, 
                flank_seq=self.hparams.flank_seq if not validation else 0, 
                flank_struct=self.hparams.flank_struct if not validation else 0,  
                flank_coords=self.hparams.flank_coords if not validation else 0, 
                tokenizer=self.tokenizer, 
                from_originals=True,
                skip=self.hparams.mean_ll
            )
            with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                p_s_uncal = self.model.forward(b_s_m).view(-1).to(torch.float32)
                p_s_cal = self.calibrate_split_paths_singles(b_s_m, p_s_uncal, detach_reg=detach_reg)
            pred_uncal.index_copy_(0, idx_single, p_s_uncal)    
            pred_cal.index_copy_(0, idx_single, p_s_cal)

        # Doubles
        # Calibration depends on the doubles strategy used, and should only occur during validation
        if idx_double.numel() > 0:
            b_d = utils.slice_batch_by_index(batch_in, idx_double)
            out_d_cal = self.forward_doubles_and_calibrate(
                b_d,
                mask_sequence_pos=self.hparams.mask_sequence_pos, 
                mask_structure_pos=self.hparams.mask_structure_pos, 
                mask_coords_pos=self.hparams.mask_coords_pos, 
                mask_sequence_fraction=self.hparams.mask_sequence_fraction, #if not validation else 0.0, 
                mask_structure_fraction=self.hparams.mask_structure_fraction, #if not validation else 0.0, 
                mask_coords_fraction=self.hparams.mask_coords_fraction, #if not validation else 0.0, 
                flank_seq=self.hparams.flank_seq if not validation else 0, 
                flank_struct=self.hparams.flank_struct if not validation else 0,  
                flank_coords=self.hparams.flank_coords if not validation else 0, 
                effect_strategy=effect_strategy,
                detach_reg=detach_reg,
                compute_epi=self.hparams.lambda_epi > 0 or validation
            )
            p_d_uncal, p_d_cal, p_a_cal, p_e_cal = out_d_cal
            pred_cal.index_copy_(0, idx_double, p_d_cal.view(-1).to(torch.float32))
            pred_uncal.index_copy_(0, idx_double, p_d_uncal.view(-1).to(torch.float32))   
            if p_a_cal is not None:
                pred_add.index_copy_(0, idx_double, p_a_cal.view(-1).to(torch.float32))
            if p_e_cal is not None:
                pred_epi.index_copy_(0, idx_double, p_e_cal.view(-1).to(torch.float32))

        return pred_uncal, pred_cal, pred_add, pred_epi

    def _compose_losses_streaming_and_backward(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Fused, streaming losses with flexible masking & doubles handling.
        - Mask once per micro-batch via forward_split_paths_and_calibrate.
        - Reuse predictions for rank/reg/epi to avoid extra forwards.
        """
        device = batch['ddG'].device
        logs: Dict[str, float] = {}

        B = int(batch['ddG'].shape[0])
        ddG = utils._get_label(batch, 'ddG', device=device)
        dddG_all = utils._get_label(batch, 'dddG', device=device)

        subset_size = int(self.hparams.subset_size)
        mb = max(1, subset_size) if subset_size > 0 else max(1, min(B, 128))

        rank_sum_combined = reg_sum = epi_sum = 0.0
        rank_cnt_combined = reg_cnt = epi_cnt = 0

        calibration_loss = 0

        num_splits = math.ceil(B / mb)

        for start in range(0, B, mb):
            end = min(start + mb, B)
            idx = torch.arange(start, end, device=device)
            micro = utils.slice_batch_by_index(batch, idx)

            st = micro['subset_type']  # e.g., list of strings length mb
            idx_single = torch.as_tensor([i for i, s in enumerate(st) if s == 'single'], device=device, dtype=torch.int)
            idx_double = torch.as_tensor([i for i, s in enumerate(st) if s == 'double'], device=device, dtype=torch.int)
            idx_mut_ctx = torch.as_tensor([i for i, s in enumerate(st) if s == 'mut_ctx'], device=device, dtype=torch.int)
            idx_reversion = torch.as_tensor([i for i, s in enumerate(st) if s == 'reversion'], device=device, dtype=torch.int)

            # One forward for all heads
            with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                pred_uncal, pred_cal, add_cal, epi_cal = \
                    self.forward_split_paths_and_calibrate(
                    micro,
                    effect_strategy=self.hparams.effect_strategy,
                    validation=False,
                    detach_reg=self.hparams.detach_regression
                )

            if self.tau0 is not None:
                pred_rank = pred_uncal / self.get_rank_tau()
            else:
                pred_rank = pred_uncal

            mb_losses = []
            ddG_mb = micro['ddG']

            # -------- 2a) Ranking --------
            if self.hparams.lambda_rank > 0 and self.crit_rank is not None and ddG is not None:
                with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):

                    L_rank = self.crit_rank(pred_rank, ddG_mb)

                    mb_losses.append(self.hparams.lambda_rank * L_rank)

                    rank_sum_combined += float(L_rank.detach().item())
                    rank_cnt_combined += 1

            # -------- 2b) Regression --------
            if self.hparams.lambda_reg > 0 and ddG is not None:
                with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):

                    L_reg_vec = self.crit_reg(pred_cal, ddG_mb)

                    w = torch.ones_like(L_reg_vec)
                    if idx_double.any():
                        w[idx_double] = self.hparams.double_weight
                    if idx_mut_ctx.any():
                        w[idx_mut_ctx] = self.hparams.mut_ctx_weight
                    if idx_reversion.any():
                        w[idx_reversion] = self.hparams.reversion_weight  
                    L_reg = self.hparams.lambda_reg * (w * L_reg_vec).sum() / w.sum().clamp_min(1e-9)
                mb_losses.append(L_reg)
                reg_sum += float(L_reg.detach().item()); reg_cnt += 1

            # -------- 2c) Epistasis (teacher forcing when training with doubles) --------
            # Simplified: if dddG is present and finite, use it. No need to explicitly pre-filter doubles.
            if self.hparams.lambda_epi > 0 and dddG_all is not None:
                dddG_mb = micro['dddG']
                valid_epi = torch.isfinite(dddG_mb[idx_double])
                if valid_epi.any():
                    with torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
                        L_epi = self.crit_epi(epi_cal[idx_double][valid_epi], dddG_mb[idx_double][valid_epi])

                    self._train_epi_preds.append(epi_cal[idx_double][valid_epi].detach())
                    self._train_epi_truths.append(dddG_mb[idx_double][valid_epi].detach())

                    mb_losses.append(self.hparams.lambda_epi * L_epi)
                    epi_sum += float(L_epi.detach().item()); epi_cnt += 1

            # -------- Single backward for this micro-batch --------
            if mb_losses:
                loss_mb = (sum(mb_losses)) / (self.accumulate_grad_batches * num_splits)
                self._backward_now(loss_mb)

        logs = {
            'L_rank':      (rank_sum_combined / max(1, rank_cnt_combined)) if rank_cnt_combined else 0.0,
            'L_reg': (reg_sum  / max(1, reg_cnt)) if reg_cnt else 0.0,
            'L_epi':       (epi_sum  / max(1, epi_cnt)) if epi_cnt else 0.0
        }

        return logs

    def _backward_now(self, loss: torch.Tensor | None):
        if loss is None:
            return
        self._did_backward = True
        self.manual_backward(loss)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Manual-optimization training step that:
        1) Normalizes and asserts residue-axis alignment across modalities.
        2) Streams fused losses in micro-batches (VRAM-safe), doing one backward per micro-batch.
        3) Steps the optimizer only when gradient accumulation boundary is reached.
        4) Logs per-step metrics from the fused loss composer.

        Requirements:
        - self.automatic_optimization == False  (Lightning manual opt)
        - self._compose_losses_streaming_and_backward() performs autocast + manual_backward internally
        """
        # --------- Housekeeping / mode ----------
        if getattr(self, "automatic_optimization", None) is not False:
            warnings.warn("Expected manual optimization (automatic_optimization=False).", RuntimeWarning)

        self.model.train()

        # --------- Batch normalization & shape sanity ----------
        batch = utils._normalize_batch(batch)  # your existing normalizer
        # Fail early if the dataloader produces misaligned residue lengths.
        utils._assert_L_alignment(
            batch,
            use_orig=True,
            where="train_step(orig)"
        )

        # ---------- Gradient accumulation bookkeeping ----------
        self.batch_accumulation_count = getattr(self, 'batch_accumulation_count', 0) + 1
        is_optimizer_step = (self.batch_accumulation_count % self.accumulate_grad_batches) == 0

        # Reset "did we backprop at least once" flag; _compose_losses_* will set this.
        self._did_backward = False

        # ---------- Fused streaming losses (does the micro-batching + backward internally) ----------
        logs = self._compose_losses_streaming_and_backward(batch)

        # ---------- Optimizer step (only at accumulation boundary, and only if we actually did backward) ----------
        optim = self.optimizers()
        if is_optimizer_step and self._did_backward:
            # Optional gradient clipping (Lightning can help unscale internally)
            max_norm = getattr(self.hparams, "grad_clip_norm", None)
            if max_norm and max_norm > 0:
                # Lightning utility that works with precision plugin
                self.clip_gradients(
                    optim,
                    gradient_clip_val=max_norm,
                    gradient_clip_algorithm="norm",
                )

            optim.step()
            optim.zero_grad(set_to_none=True)

            sch = self.lr_schedulers()
            if sch:
                if isinstance(sch, (list, tuple)):
                    for s in sch: s.step()
                else:
                    sch.step()
            self._log_lrs()
        else:
            # No grads created in this accumulation window — skip stepping/updating entirely.
            pass

        # ---------- Logging ----------
        # Per-step logs: ranking / regression / epistasis, as produced by the fused composer.
        for k, v in logs.items():
            # on_step=True gives you live progress-bar updates; on_epoch can be added if desired.
            self.log(f"train/{k}", v, on_step=True)

        if self.tau0 is not None:
            self.log("rank/tau", self.get_rank_tau().detach(), on_step=True, prog_bar=True)
        self.log_calibration_heads(on_step=True)

        # Every N steps, compute correlation and clear
        if (self.global_step % 50 == 0) and (self.global_step != 0) and len(self._train_epi_preds) > 0:
            try:
                preds = torch.cat(self._train_epi_preds)
                truths = torch.cat(self._train_epi_truths)
                rho, _ = spearmanr(preds.cpu(), truths.cpu())
                print('train epi rho:', rho)
                self.log('train/spearman_epistasis_rolling', rho, on_step=True)
                try:
                    rmse = np.sqrt(mean_squared_error(preds.cpu(), truths.cpu()))
                    print('train epi rmse:', rmse,)
                    self.log('train/rmse_epistasis_rolling', rmse, on_step=True)
                except Exception as e:
                    print(e)
                self._train_epi_preds.clear()
                self._train_epi_truths.clear()
            except:
                pass

        # Periodic aggressive cleanup
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Emergency cleanup if memory is high
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if mem_used > 0.9:  # Over 90% of peak
                gc.collect()
                torch.cuda.empty_cache()

        # Lightning requires a tensor return; value unused when manual optimization is enabled.
        return torch.tensor(0.0, device=self.device)
    
    def on_before_optimizer_step(self, optimizer):
        # grads are unscaled here
        lora = utils._select_lora_params(self.model.named_parameters())
        lr_dict = self._get_lrs()
        if not self.hparams.freeze_lora:
            self.log_dict({
                "lora/grad_norm": utils.l2_grad_norm(lora),
                "lora/weight_norm": utils.l2_weight_norm(lora),
                "lora/step_norm": utils.group_step_norm(lora, lr_dict['lora'])
            })

    # =============================
    # Validation (unchanged surface; now uses compat)
    # =============================
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """
        Compute ddG loss + store:
        - pred_all (all items)
        - subset (doubles with finite dddG): pred_add, pred_all, ddG
        - epistasis: pred_epi + dddG (only doubles with finite dddG)
        """
        batch = utils._normalize_batch(batch)
        device = batch['ddG'].device
        utils._assert_L_alignment(
            batch,
            use_orig=True,
            where="val_step(orig)"
        )
        self.model.eval()

        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=torch.is_autocast_enabled()):
            pred_uncal, pred_cal, add_cal, epi_cal = self.forward_split_paths_and_calibrate(
                batch,
                effect_strategy="chain_rule_avg", # average of both paths
                validation=True # remove additional masking
            )
            ddG = utils._get_label(batch, 'ddG', device=device)

        out = {
            'scores': pred_cal.detach().cpu().numpy(),
            'ground_truths': ddG.detach().cpu().numpy() if ddG is not None else np.array([]),
            'dataloader_idx': dataloader_idx,
            'pdb': batch['pdb'],
            'mutations': batch['mutations']
        }

        # ----- Doubles with finite dddG only -----
        idx_double = utils._double_indices(batch, device=device)  # long tensor
        idx_dddG = utils._double_indices(batch, finite_dddG=True, device=device)
        if idx_double.numel() > 0:            
            dddG = utils._get_label(batch, 'dddG', device)  # must exist because idx_dddG checks it

            # NOTE: pred_add/pred_epi contain NaNs for non-doubles; we slice by idx_dddG
            out['epi_pred']   = epi_cal.index_select(0, idx_dddG).detach().cpu().numpy()
            out['epi_gt']     = dddG.index_select(0, idx_dddG).detach().cpu().numpy()

            # NEW: additive & total vs ddG, same subset
            out['add_pred']   = add_cal.index_select(0, idx_double).detach().cpu().numpy()
            out['double_pred']   = pred_cal.index_select(0, idx_double).detach().cpu().numpy()
            out['ddg_subset'] = ddG.index_select(0, idx_double).detach().cpu().numpy()
        else:
            # Always store arrays (not scalars) to avoid zero-dim concatenate errors
            out['epi_pred']   = np.array([], dtype=np.float32)
            out['epi_gt']     = np.array([], dtype=np.float32)
            out['epi_pdb']    = np.array([], dtype=object)
            out['add_pred']   = np.array([], dtype=np.float32)
            out['double_pred']   = np.array([], dtype=np.float32)
            out['ddg_subset'] = np.array([], dtype=np.float32)

        self.validation_step_outputs[dataloader_idx].append(out)

    # inside your LightningModule
    def log_calibration_heads(self, on_step=True, on_epoch=False, prog_bar=False, logger=True):
        head_names = [
            "single",
            "mut_ctx",
            "reversion",
            "double",
            "epi",
            "shared",
        ]
        for name in head_names:
            head = getattr(self.model, f"calibration_head_{name}", None)
            if head is None:
                continue  # skip missing heads
            elif name in ['single', 'double']:
                name += 's'
            for param in ("scale", "bias"):
                val = getattr(head, param, None)
                if val is None:
                    continue
                # ensure it's a tensor (some heads may store raw_scale instead)
                if isinstance(val, torch.Tensor):
                    try:
                        self.log(
                            f"calibration_heads/{name}_{param}",
                            val.detach().mean().item() if val.numel() > 1 else val.detach().item(),
                            on_step=on_step,
                            on_epoch=on_epoch,
                            prog_bar=prog_bar,
                            logger=logger,
                        )
                        #print(name, param, val.detach().mean().item())
                    except Exception:
                        # never break logging loop; skip malformed tensors
                        continue

    def on_validation_epoch_end(self) -> None:
        """Extend aggregation with additive/total vs ddG on doubles-with-finite-dddG subset,
        in addition to your existing epistasis metrics and overall metrics.
        """
        #self.log_calibration_heads(on_epoch=True)

        # ====== ORIGINAL METRICS (unchanged) ======
        all_dataloader_scores = []
        all_dataloader_epis = []
        all_dataloader_ground_truths = []
        avg_metrics = defaultdict(list)

        for dataloader_idx, valid_outputs in self.validation_step_outputs.items():

            val_loader_name = self.hparams.val_dataloader_names[dataloader_idx] \
                if dataloader_idx < len(self.hparams.val_dataloader_names) else f"unknown_dl_{dataloader_idx}"

            #losses = torch.stack([o['loss'] for o in valid_outputs])
            scores = np.concatenate([o['scores'] for o in valid_outputs])
            ground_truths = np.concatenate([o['ground_truths'] for o in valid_outputs])

            try:
                if len(np.unique(scores)) < 2 or len(np.unique(ground_truths)) < 2:
                    rho = 0.0
                else:
                    rho, _ = spearmanr(scores, ground_truths)
                if np.isnan(rho):
                    rho = 0.0
            except ValueError:
                rho = 0.0
            rmse = np.sqrt(mean_squared_error(ground_truths, scores))
            dfs = pd.DataFrame({'scores': scores, 'ground_truths': ground_truths})
            ndcg = utils.compute_ndcg_flexible(dfs, 'scores', 'ground_truths', threshold=0)

            gt_pos = ground_truths > 1e-6
            sc_pos = scores > 1e-6
            f1 = f1_score(gt_pos, sc_pos, zero_division=0)
            tp = np.sum(sc_pos & gt_pos)
            ppv = tp / np.sum(sc_pos) if np.sum(sc_pos) > 0 else 0.0
            sens = tp / np.sum(gt_pos) if np.sum(gt_pos) > 0 else 0.0
            gain = ground_truths[sc_pos].sum()
            mean_score = scores.mean()

            self.log(f'val_spearman_rho/{val_loader_name}', rho, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'val_ndcg/{val_loader_name}', ndcg, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'val_net_gain/{val_loader_name}', gain, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'val_rmse/{val_loader_name}', rmse, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

            avg_metrics['rho'].append(rho)
            avg_metrics['ndcg'].append(ndcg)
            avg_metrics['rmse'].append(rmse)
            avg_metrics['f1'].append(f1)
            avg_metrics['gain'].append(gain)
            avg_metrics['sensitivity'].append(sens)
            avg_metrics['precision'].append(ppv)
            avg_metrics['mean_score'].append(mean_score)
            all_dataloader_scores.append(scores)
            all_dataloader_ground_truths.append(ground_truths)

            # ====== NEW: gather epistasis arrays (as you had) ======
            epi_preds, epi_targets = [], []
            for o in valid_outputs:
                if isinstance(o, dict) and ('epi_pred' in o) and ('epi_gt' in o):
                    if o['epi_pred'].size and o['epi_gt'].size:
                        epi_preds.append(o['epi_pred'])
                        epi_targets.append(o['epi_gt'])

            if epi_preds:
                epi_preds = np.concatenate(epi_preds)
                epi_targets = np.concatenate(epi_targets)

                finite = np.isfinite(epi_targets)
                epi_preds_f = epi_preds[finite]
                epi_targets_f = epi_targets[finite]

                if epi_targets_f.size == 0:
                    continue
                if len(np.unique(epi_targets_f)) < 2 or len(np.unique(epi_preds_f)) < 2:
                    rho_p = 0.0
                else:
                    rho_p = spearmanr(epi_preds_f, epi_targets_f)[0]
                    if np.isnan(rho_p): rho_p = 0.0
                rmse_p = float(np.sqrt(mean_squared_error(epi_targets_f, epi_preds_f)))

                self.log(f'val_epi_rho/{val_loader_name}', float(rho_p), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val_epi_rmse/{val_loader_name}', rmse_p, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                all_dataloader_epis.append(epi_preds)
                avg_metrics['epi_rho'].append(rho_p)

            # ====== NEW: Additive vs ddG and All vs ddG on the same subset ======
            add_preds, double_preds, ddg_subsets = [], [], []
            for o in valid_outputs:
                if isinstance(o, dict) and all(k in o for k in ('add_pred','double_pred','ddg_subset')):
                    # Each is an array (may be empty); only append non-empty to avoid 0-d concatenation issues
                    if o['add_pred'].size and o['double_pred'].size and o['ddg_subset'].size:
                        add_preds.append(o['add_pred'])
                        double_preds.append(o['double_pred'])
                        ddg_subsets.append(o['ddg_subset'])

            if add_preds:
                add_preds = np.concatenate(add_preds)
                double_preds = np.concatenate(double_preds)
                ddg_subsets = np.concatenate(ddg_subsets)

                # Spearman & RMSE: additive vs ddG
                try:
                    rho_add = 0.0 if (len(np.unique(add_preds)) < 2 or len(np.unique(ddg_subsets)) < 2) \
                            else (spearmanr(add_preds, ddg_subsets)[0] or 0.0)
                    if np.isnan(rho_add): rho_add = 0.0
                except ValueError:
                    rho_add = 0.0
                rmse_add = float(np.sqrt(mean_squared_error(ddg_subsets, add_preds)))

                # Spearman & RMSE: all vs ddG
                try:
                    rho_double = 0.0 if (len(np.unique(double_preds)) < 2 or len(np.unique(ddg_subsets)) < 2) \
                            else (spearmanr(double_preds, ddg_subsets)[0] or 0.0)
                    if np.isnan(rho_double): rho_double = 0.0
                except ValueError:
                    rho_double = 0.0
                rmse_double = float(np.sqrt(mean_squared_error(ddg_subsets, double_preds)))

                self.log(f'val_add_vs_ddG_rho/{val_loader_name}', rho_add, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val_add_vs_ddG_rmse/{val_loader_name}', rmse_add, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val_double_vs_ddG_rho/{val_loader_name}', rho_double, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val_double_vs_ddG_rmse/{val_loader_name}', rmse_double, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            
        # ====== ORIGINAL OVERALL AVERAGES (unchanged) ======
        if avg_metrics['rho']:
            overall_avg_rho = np.mean(avg_metrics['rho'])
            overall_median_rho = np.median(avg_metrics['rho'])
            overall_avg_ndcg = np.mean(avg_metrics['ndcg'])
            overall_median_ndcg = np.median(avg_metrics['ndcg'])
            overall_avg_epi_rho = np.mean(avg_metrics['epi_rho'])

            self.log('val_rho_avg', overall_avg_rho, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_rho_median', overall_median_rho, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_ndcg_avg', overall_avg_ndcg, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_ndcg_median', overall_median_ndcg, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            self.log('val_epi_rho_avg', overall_avg_epi_rho, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            if all_dataloader_scores and all_dataloader_ground_truths:
                all_scores_flat = np.concatenate(all_dataloader_scores)
                all_gt_flat = np.concatenate(all_dataloader_ground_truths)
                ungrouped_rho, _ = spearmanr(all_scores_flat, all_gt_flat)
                ungrouped_rmse = np.sqrt(mean_squared_error(all_gt_flat, all_scores_flat))

                gt_pos_u = all_gt_flat > 1e-6
                sc_pos_u = all_scores_flat > 1e-6
                ungrouped_f1 = f1_score(gt_pos_u, sc_pos_u, zero_division=0.0)
                tp_u = np.sum(sc_pos_u & gt_pos_u)
                predicted_pos_u = np.sum(sc_pos_u)
                actual_pos_u = np.sum(gt_pos_u)
                ungrouped_prec = tp_u / predicted_pos_u if predicted_pos_u > 0 else 0.0
                ungrouped_sens = tp_u / actual_pos_u if actual_pos_u > 0 else 0.0
                ungrouped_gain = all_gt_flat[sc_pos_u].sum()
                ungrouped_mean = all_scores_flat.mean()

                try:
                    all_epis_flat = np.concatenate(all_dataloader_epis)
                    epi_mag = np.mean(np.abs(all_epis_flat))
                    self.log('val_mean_epi_mag', epi_mag, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                except ValueError:
                    pass

                self.log('val_rho_ungrouped', ungrouped_rho, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('val_rmse_ungrouped', ungrouped_rmse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('val_f1', ungrouped_f1, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log('val_gain_total', ungrouped_gain, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log('val_sensitivity', ungrouped_sens, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log('val_precision', ungrouped_prec, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log('val_mean_score', ungrouped_mean, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # clear as before
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

    def on_fit_start(self):
        # how many optimizer *steps* were already completed before resuming
        self.step_offset = int(getattr(self.hparams, "resume_global_step", 0) or 0)
        k = self.adjusted_step()

        if k <= 0:
            return
        for cfg in self.trainer.lr_scheduler_configs:
            cfg.scheduler.last_epoch = k - 1

    def adjusted_step(self) -> int:
        # absolute step since the original run began
        return self.step_offset + int(self.global_step)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Schedules:
        - LoRA/Other: warmup -> cosine to 0 by main_end; then constant tail = base_lr * lr_tail_lora_mult
        - Calibration: 1 during warmup; 0 after
        """
        base_lr = float(self.hparams.learning_rate)
        wd = float(getattr(self.hparams, "weight_decay", 0.01))

        # Tunables (with sensible defaults)
        lr_warmup_steps   = int(getattr(self.hparams, "lr_warmup_steps", 0))
        lr_total_steps    = getattr(self.hparams, "lr_total_steps", None)
        lr_tail_steps     = int(getattr(self.hparams, "lr_tail_steps", 0))          # tail length
        lr_tail_lora_mult = float(getattr(self.hparams, "lr_tail_lora_mult", 0.1))  # LoRA LR multiplier in tail
        calib_lr_mult     = float(getattr(self.hparams, "calib_lr_mult", 20.0))
        rank_lr_mult      = float(getattr(self.hparams, "rank_lr_mult", 200.0))

        k_resume = int(getattr(self.hparams, "resume_global_step", 0) or 0)
        if k_resume == 0:
            burn_steps = 0
        else:
            burn_steps  = int(getattr(self.hparams, "resume_burn_in_steps", 500) or 0)
        burn_scale  = float(getattr(self.hparams, "resume_burn_scale", 0.3))

        def burn_mult(step: int) -> float:
            # Applies only when resuming with a burn-in configured
            if burn_steps <= 0:
                return 1.0
            return burn_scale if step < (k_resume + burn_steps) else 1.0

        def lmul(f):
            # wraps a base lambda so it’s scaled during burn-in
            return (lambda s: burn_mult(s) * f(s))

        # --- Collect params by group ---
        lora_params, other_params, calib_params, rank_params = [], [], [], []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if id(p) in seen: continue
            seen.add(id(p))
            lname = name.lower()
            if "rank" in lname:
                rank_params.append(p) 
            elif "calibration" in lname:
                calib_params.append(p)
            elif "lora" in lname:
                lora_params.append(p)
            else:
                other_params.append(p)

        def _count(ps): return sum(p.numel() for p in ps)
        logging.info(
            f"Param groups -> LoRA: {len(lora_params)} ({_count(lora_params):,}), "
            f"Calibration: {len(calib_params)} ({_count(calib_params):,}) "
            f"Ranking: {len(rank_params)} ({_count(rank_params):,}), "
            f"Other: {len(other_params)} ({_count(other_params):,}), "
        )

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": base_lr, "weight_decay": wd, "name": "lora"})
        if calib_params:
            param_groups.append({"params": calib_params, "lr": base_lr * calib_lr_mult, "weight_decay": 0.0, "name": "calib"})
        if rank_params:
            param_groups.append({"params": rank_params, "lr": base_lr * rank_lr_mult, "weight_decay": 0.0, "name": "rank"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd, "name": "other"})

        if not param_groups:
            trainable = [p for p in self.parameters() if p.requires_grad]
            logging.warning("No param groups found; defaulting to all trainables at base LR.")
            param_groups = [{"params": trainable, "lr": base_lr, "weight_decay": wd, "name": "all"}]

        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999), fused=True)

        # --- Determine total steps & phase split
        if lr_total_steps is None:
            if self.trainer and hasattr(self.trainer, "estimated_stepping_batches"):
                lr_total_steps = int(self.trainer.estimated_stepping_batches)
                logging.info(f"Using estimated total steps for LR scheduler: {lr_total_steps}")
            else:
                lr_total_steps = lr_warmup_steps
        main_end = max(0, int(lr_total_steps) - max(0, lr_tail_steps))  # steps [0, main_end) = warmup+main

        # --- Lambda schedules

        def delayed_warmcos(step: int, warmup: int, total_main: int, delay: int) -> float:
            """Linear warmup -> cosine decay to 0 over [warmup, total_main)."""
            if total_main <= 0:
                return 1.0
            if step <= delay:
                return 0
            if step <= float(warmup + delay):
                return float(step - delay) / float(max(1, warmup))
            t = min(max(0, step - warmup), max(1, total_main - warmup))
            prog = t / float(max(1, total_main - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * prog))  # 1 -> 0 over main phase
        
        def higher_initial(step: int, delay: int, multiplier: float) -> float:
            if step <= delay:
                return multiplier
            else:
                return 1.0
            
        delay = burn_steps if burn_steps > 0 else 0

        def lambda_lora(step: int) -> float:
            if step < main_end:
                return delayed_warmcos(step, lr_warmup_steps, main_end, delay) * (1 - int(self.hparams.freeze_lora))
            # tail: constant multiplier (could be 0.0 to freeze)
            return lr_tail_lora_mult * (1 - int(self.hparams.freeze_lora))

        def lambda_other(step: int) -> float:
            # Same as LoRA for "other" backbone params
            return lambda_lora(step)

        def lambda_calib(step: int) -> float:
            if step < main_end:         # constant during main
                return higher_initial(step, delay, multiplier=5)
            return 1.0
        
        def lambda_tau(step: int) -> float:
            if step < main_end:         # constant during main
                return higher_initial(step, delay, multiplier=5)
            return 1.0

        lambdas = []
        for g in param_groups:
            name = g.get("name", "")
            if name == "lora":
                lambdas.append(lmul(lambda_lora))
            elif name == "calib":
                lambdas.append(lmul(lambda_calib))
            elif name == "rank":
                lambdas.append(lmul(lambda_tau))
            else:
                lambdas.append(lmul(lambda_other))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        self._pg_name_to_idx = {g["name"]: i for i, g in enumerate(param_groups) if "name" in g}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1, 
            },
        }
    
    def _get_lrs(self):
        lrs = {}
        opt = self.trainer.optimizers[0]
        for i, g in enumerate(opt.param_groups):
            name = g.get("name", f"group{i}")
            # Lightning prefers python scalars (not tensors)
            lrs.update({name: float(g["lr"])})
        return lrs
                     
    def _log_lrs(self):
        opt = self.trainer.optimizers[0]  # or: self.optimizers() in PL < 2.0
        for i, g in enumerate(opt.param_groups):
            name = g.get("name", f"group{i}")
            # Lightning prefers python scalars (not tensors)
            self.log(f"lr/{name}", float(g["lr"]),
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Saves only the trainable parameters (LoRA and/or Head) in the checkpoint's state_dict.
        """
        current_model_state_dict = self.model.state_dict()
        weights_to_save = {}

        # Identify trainable parameters based on requires_grad flag
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights_to_save[name] = current_model_state_dict[name]

        if not weights_to_save:
            logging.warning("on_save_checkpoint: No trainable parameters found to save. Checkpoint state_dict will be empty.")
        else:
             logging.info(f"Saving {len(weights_to_save)} trainable parameter tensors in checkpoint state_dict.")

        # Overwrite the state_dict ONLY with the trainable weights
        checkpoint['state_dict'] = weights_to_save

# === Data Loading Setup ===

def setup_dataloaders(args: argparse.Namespace, tokenizer: Any, structure_encoder: Any) -> Tuple[List[DataLoader], List[DataLoader], List[str], List[str]]:
    """Sets up training and validation dataloaders based on objective."""
    data_path_base = args.data_path
    # --- Load DataFrames ---
    try:
        df_train = pd.read_csv(os.path.join(data_path_base, args.train_data_file))
        # Add unique ID if missing (using 'code' and 'mut_type' if available)
        if 'uid' not in df_train.columns and 'code' in df_train.columns and 'mut_type' in df_train.columns:
            df_train['uid'] = df_train['code'].astype(str) + '_' + df_train['mut_type'].astype(str)
        if 'uid' in df_train.columns: df_train = df_train.drop_duplicates(subset=['uid'])
        logging.info(f"Loaded training data: {len(df_train)} entries from {args.train_data_file}")
    except FileNotFoundError:
        logging.error(f"Training data file not found: {os.path.join(data_path_base, args.train_data_file)}")
        raise
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

    if args.val_data_file:
        try:
            df_val = pd.read_csv(os.path.join(data_path_base, args.val_data_file))
            if 'uid' not in df_val.columns and 'code' in df_val.columns and 'mut_type' in df_val.columns:
                df_val['uid'] = df_val['code'].astype(str) + '_' + df_val['mut_type'].astype(str)
            if 'uid' in df_val.columns: df_val = df_val.drop_duplicates(subset=['uid'])
            logging.info(f"Loaded validation data: {len(df_val)} entries from {args.val_data_file}")
        except FileNotFoundError:
            logging.error(f"Validation data file not found: {os.path.join(data_path_base, args.val_data_file)}")
            raise
        except Exception as e:
            logging.error(f"Error loading validation data: {e}")
            raise
    else:
        df_val = df_train.copy() # Use train data for validation if no separate file
        logging.info("Using training data for validation.")


    # remove fake mutations
    orig_train = len(df_train)
    df_train = df_train.loc[~df_train['mut_type'].apply(utils.is_fake_mutation)]
    df_train = df_train.loc[~df_train['uid'].apply(utils.is_improper_mutation)]
    logging.warning(f"Removed {orig_train-len(df_train)} fake or improper mutations (wt == mut) from train dataframe")
    df_val = df_val.loc[~df_val['mut_type'].apply(utils.is_fake_mutation)]
    df_val = df_val.loc[~df_val['uid'].apply(utils.is_improper_mutation)]
    #df_val = df_val.loc[~df_val['uid'].apply(lambda x: x=='1SF0_V59K_delK59')]
    logging.warning(f"Removed {orig_train-len(df_val)} fake or improper mutations (wt == mut) from val dataframe")

    # --- Determine Train/Val Splits ---
    if args.split_file:
        try:
            splits = pd.read_csv(args.split_file, index_col=0)
            train_protein_list = eval(splits.loc[args.split_key, 'training'])
            val_protein_list = eval(splits.loc[args.split_key, 'validation'])
            logging.info(f"Using split file {args.split_file} with key '{args.split_key}'.")
        except FileNotFoundError: raise FileNotFoundError(f"Split file not found: {args.split_file}")
        except KeyError: raise KeyError(f"Split key '{args.split_key}' not found in {args.split_file}")
        except Exception as e: raise RuntimeError(f"Error reading split file {args.split_file}: {e}")
    else:
        train_protein_list = df_train['code'].unique().tolist() if 'code' in df_train.columns else []
        val_protein_list = [] #df_val['code'].unique().tolist() if 'code' in df_val.columns else []
        if not train_protein_list: logging.warning("Could not determine train protein list from 'code' column.")
        if not val_protein_list: logging.warning("Could not determine validation protein list from 'code' column.")
        overlap = set(train_protein_list) & set(val_protein_list)
        if overlap and args.val_data_file: logging.warning(f"Overlap detected between train/val protein codes: {len(overlap)} proteins.")
        logging.info("Using unique protein codes from data files for train/val split.")

    # Optional: Limit number of training proteins
    if args.max_train_proteins > 0:
        train_protein_list = train_protein_list[:args.max_train_proteins]
        logging.info(f"Limiting training to {len(train_protein_list)} proteins.")

    # --- Create Datasets and Individual DataLoaders ---

    train_dataloaders = []
    train_loader_names = []

    logging.info(f'Creating train datasets for {len(train_protein_list)} proteins...')
    for prot_code in tqdm(train_protein_list, desc="Creating train datasets"):
        df_prot_train = df_train[df_train['code'].str.contains(prot_code, regex=False)]
        if not prot_code.startswith('v2_'):
            df_prot_train = df_prot_train.loc[~df_prot_train['code'].str.startswith('v2_')]
        if df_prot_train.empty: 
            logging.warning(f"Skipping train protein '{prot_code}': Doesn't appear to be in the dataframe")
            assert False
        try:             
            dataset = ProteinStructureMutationEpistasisDatasetChainRule(
                dms_df=df_prot_train, tokenizer=tokenizer, structure_encoder=None if args.include_structure_encoder else structure_encoder,
                dms_name=prot_code, path=args.cache_path, score_name=args.score_column, enable_mutctx_masking=args.enable_mutctx_masking,
                generate=args.regenerate_cache, use_mut_structs=args.use_mut_structs, mut_structs_root=args.mut_structures_root,
                incl_destab=args.incl_destab, incl_reversions=args.incl_reversions, incl_mutctx=not (args.singles_only or args.effect_strategy=='direct_forward'), incl_doubles=args.incl_doubles)
            if len(dataset) == 0:
                 logging.warning(f"Train dataset for '{prot_code}' is empty. Skipping.")
                 continue
            loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_chainrule,
                                     num_workers=args.num_workers, pin_memory=True)
            train_dataloaders.append(loader)
            train_loader_names.append(prot_code)
        except (FileNotFoundError, KeyError, ValueError) as e: logging.warning(f"Skipping train protein '{prot_code}': {e}")
        except Exception as e: logging.error(f"Unexpected error creating train dataset for '{prot_code}': {e}", exc_info=True)


    val_dataloaders = []
    val_loader_names = []

    logging.info(f'Creating validation datasets for {len(val_protein_list)} proteins...')
    for prot_code in tqdm(val_protein_list, desc="Creating val datasets"):
        df_prot_val = df_val[df_val['code'].str.contains(prot_code, regex=False)]
        if not prot_code.startswith('v2_'):
            df_prot_val = df_prot_val.loc[~df_prot_val['code'].str.startswith('v2_')]
        if df_prot_val.empty: 
            logging.warning(f"Skipping val protein '{prot_code}': Doesn't appear to be in the dataframe")
            continue
        try:
            dataset = ProteinStructureMutationEpistasisDatasetChainRule(
                dms_df=df_prot_val, tokenizer=tokenizer, structure_encoder=None if args.include_structure_encoder else structure_encoder,
                dms_name=prot_code, path=args.cache_path, score_name=args.score_column, enable_mutctx_masking=args.enable_mutctx_masking,
                generate=args.regenerate_cache, use_mut_structs=False, incl_destab=args.incl_destab, 
                incl_reversions=False, incl_mutctx=False, incl_doubles=True)
            if len(dataset) == 0:
                 logging.warning(f"Validation dataset for '{prot_code}' is empty. Skipping.")
                 continue
            loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_chainrule,
                                     num_workers=0, shuffle=False, pin_memory=True)
            val_dataloaders.append(loader)
            val_loader_names.append(prot_code)
        except (FileNotFoundError, KeyError, ValueError) as e: logging.warning(f"Skipping val protein '{prot_code}': {e}")
        except Exception as e: logging.error(f"Unexpected error creating val dataset for '{prot_code}': {e}", exc_info=True)

    # --- Add Standard Benchmark Datasets (Optional) ---
    benchmark_datasets = {'ptmuld': 'ptmuld_mapped_new.csv', 's461': 's461_mapped_new.csv', 'ssym': 'ssym_mapped_new.csv'}
    for name, filename in benchmark_datasets.items():
        filepath = os.path.join(data_path_base, filename)
        if os.path.exists(filepath):
            try:
                df_bench = pd.read_csv(filepath)
                # Assume benchmark score is always 'ddG' and needs to be negated for ptmuld
                score_name_bench = 'ddG'
                if score_name_bench not in df_bench.columns:
                     logging.warning(f"Benchmark '{name}' missing '{score_name_bench}' column. Skipping.")
                     continue

                dataset = ProteinStructureMutationEpistasisDatasetChainRule(
                    dms_df=df_bench, tokenizer=tokenizer, structure_encoder=None if args.include_structure_encoder else structure_encoder,
                    dms_name=name, path=args.cache_path, score_name=score_name_bench, enable_mutctx_masking=False,
                    generate=args.regenerate_cache, use_mut_structs=False,
                    incl_destab=args.incl_destab, incl_reversions=False, incl_mutctx=False, incl_doubles=True)
                
                if len(dataset) == 0: continue
                loader = PooledDataLoader([DataLoader(dataset, batch_size=1, collate_fn=collate_fn_chainrule, num_workers=0, shuffle=False, pin_memory=True)], batch_size=4)
                #loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_chainrule, num_workers=0, shuffle=False, pin_memory=True)
                val_dataloaders.append(loader)
                val_loader_names.append(name)
                logging.info(f"Added benchmark validation dataset: {name}")
            except Exception as e: logging.warning(f"Could not load benchmark {name}: {e}", exc_info=True)
        else:
            logging.warning(f'{filepath} could not be found! Skipping this loader.')

    if not train_dataloaders: raise RuntimeError("No valid training dataloaders created.")
    if not val_dataloaders: logging.warning("No validation dataloaders created.")

    # --- Assemble Dataloaders ---
    
    if args.dataloading == 'cycle':
        if not args.use_subset_restrict:
            logging.info(f"Using ProteinCyclingDataloader")
            train_combined_loader = ProteinCyclingDataLoader(train_dataloaders, args.batch_size, train_loader_names, collate_fn_chainrule, #subset_weights=subset_weights,
                                                            strategy=args.loader_strategy, positional=args.positional)
        else:
            logging.info(f"Using SubsetRestrictedProteinCyclingDataLoader")
            train_combined_loader = SubsetRestrictedProteinCyclingDataLoader(train_dataloaders, args.batch_size, train_loader_names, collate_fn_chainrule,
                                                            strategy='all')
        train_dataloaders_final = [train_combined_loader]
        
    elif args.dataloading == 'pool':
        logging.info("Using PooledDataLoader.")
        train_combined_loader = PooledDataLoader(train_dataloaders, args.batch_size, train_loader_names, strategy=args.loader_strategy)
        train_dataloaders_final = [train_combined_loader]

    else: # Should not happen due to arg choices
        logging.warning(f"Unsupported loader_strategy '{args.loader_strategy}'. Using individual loaders.")
        train_dataloaders_final = train_dataloaders
        train_loader_names_final = train_loader_names

    train_loader_names_final = [f'{"cycled" if args.dataloading == "cycle" else "pooled"}_train']

    # Keep validation dataloaders separate
    val_dataloaders_final = val_dataloaders
    val_loader_names_final = val_loader_names

    logging.info(f"Setup complete. Final Training loaders: {len(train_dataloaders_final)} (Names: {train_loader_names_final}), Final Validation loaders: {len(val_dataloaders_final)} (Names: {val_loader_names_final})")
    return train_dataloaders_final, val_dataloaders_final, train_loader_names_final, val_loader_names_final


# === Argument Parsing ===

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments with decoupled model/objective."""
    parser = argparse.ArgumentParser(description="Train ESM3 Stability Model")

    # --- LoRA Configuration ---
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument('--lora_rank', type=int, default=4, help="Rank of LoRA matrices.")
    lora_group.add_argument('--lora_alpha', type=int, default=8, help="Alpha scaling factor for LoRA.")
    lora_group.add_argument('--lora_dropout', type=float, default=0.1, help="Dropout fraction for LoRA.")
    lora_group.add_argument('--include_structure_encoder', action=argparse.BooleanOptionalAction, default=False, help="Train LoRA for structure encoder as well as sequence transformer.")
    lora_group.add_argument('--use_dora', action=argparse.BooleanOptionalAction, default=False, help="Use DoRA instead of regular LoRA.")
    lora_group.add_argument('--use_qkv_lora', action=argparse.BooleanOptionalAction, default=False, help="Apply LoRA to QKV projections in attention.")
    lora_group.add_argument('--freeze_lora', action=argparse.BooleanOptionalAction, default=False, help="Freeze LoRA params e.g. for training regression head.")
    lora_group.add_argument('--calibration', type=str, default='linear', choices=['linear', 'nonlinear'], 
                            help="Whether to adapt log-likelihoods to ddGs via linear or non-linear (+linear residual) transformation.")
    
    # --- Loss Configuration ---
    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument('--rank_loss', type=str, default='listmle', help="Loss function for ranking.", choices=['listmle', 'listmle_enhanced'])
    loss_group.add_argument('--reg_loss', type=str, default='huber', help="Loss function for regression.", choices=['huber', 'mse', 'asymmetric'])
    loss_group.add_argument('--epi_loss', type=str, default='huber', help="Loss function for dddG / epistasis.", choices=['huber', 'mse'])
    loss_group.add_argument('--lambda_rank', type=float, default=1, help="ListMLE loss weight.")
    loss_group.add_argument('--lambda_reg', type=float, default=0.5, help="Regression loss weight.")
    loss_group.add_argument('--lambda_epi', type=float, default=0.0, help="Epistasis prediction regression loss weight.")
    loss_group.add_argument('--mut_ctx_weight', type=float, default=0.5, help="How much weight mutated context items should have.")
    loss_group.add_argument('--rank_tau_init', type=float, default=None, help="Temperature tau to prevent overfit on ranking.")
    loss_group.add_argument('--detach_regression', action='store_true', help="Whether to stop the regression gradient so it only applies to the calibration head.") 

    # --- Calibration Configuration ---
    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=None, help="Scale of shared calibration head. None = No shared scale")
    calibration_group.add_argument('--shared_bias_init', type=float, default=None, help="Bias of shared calibration head. None = No shared bias")
    calibration_group.add_argument('--single_scale_init', type=float, default=None, help="Scale of single calibration head. None = No single scale")
    calibration_group.add_argument('--single_bias_init', type=float, default=None, help="Bias of single calibration head. None = No single bias")
    calibration_group.add_argument('--mutctx_scale_init', type=float, default=None, help="Scale of mutctx calibration head. None = No mutctx scale")
    calibration_group.add_argument('--mutctx_bias_init', type=float, default=None, help="Bias of mutctx calibration head. None = No mutctx bias")
    calibration_group.add_argument('--reversion_scale_init', type=float, default=None, help="Scale of reversion calibration head. None = No reversion scale")
    calibration_group.add_argument('--reversion_bias_init', type=float, default=None, help="Bias of reversion calibration head. None = No reversion bias")

    # --- Ranking Configuration ---
    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--subset_size', type=int, default=24, help="Size of each subset for ListMLE.")
    rank_group.add_argument('--invert_list_loss', action=argparse.BooleanOptionalAction, default=False, help='Whether to treat destabilizing mutations as the most important for detection')

    # --- Scoring Configuration ---
    score_group = parser.add_argument_group("Mutation Effect Scoring Configuration")
    score_group.add_argument('--effect_strategy', type=str, default="chain_rule_avg", help="How to score double mutants, [direct_forward, heuristic].")
    score_group.add_argument('--additive_condition', type=str, default="wt", help="What to fill in the non-mutated residue with when making additive approximations for doubles [wt, mask, mut].")

    # --- Masking Configuration --- 
    mask_group = parser.add_argument_group("Masking Strategy")
    mask_group.add_argument('--mask_sequence_fraction', type=float, default=0.0, help="Fraction of sequence tokens to mask.")
    mask_group.add_argument('--mask_coords_fraction', type=float, default=0.0, help="Fraction of coords to mask.")
    mask_group.add_argument('--mask_structure_fraction', type=float, default=0.0, help="Fraction of structure tokens to mask.")
    mask_group.add_argument('--unmask_sequence_pos', action=argparse.BooleanOptionalAction, default=False, help="Whether to mask the sequence at the mutated position.")
    mask_group.add_argument('--mask_coords_pos', action=argparse.BooleanOptionalAction, default=False, help="Whether to mask the coordinates at the mutated position.")
    mask_group.add_argument('--mask_structure_pos', action=argparse.BooleanOptionalAction, default=False, help="Whether to mask the structure tokens at the mutated position.")
    mask_group.add_argument('--mask_sequence_flank', type=int, default=0, help="Mask N neighbors for the sequence at the mutated position.")
    mask_group.add_argument('--mask_coords_flank', type=int, default=0, help="Mask N neighbors for the coordinates at the mutated position.")
    mask_group.add_argument('--mask_structure_flank', type=int, default=0, help="Mask N neighbors for the structure tokens at the mutated position.")
    mask_group.add_argument('--enable_mutctx_masking', action=argparse.BooleanOptionalAction, default=False, help="Whether to mask the coordinates at the mutated position.")

    # --- Training Parameters ---
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=20, help="Training epochs.")
    train_group.add_argument('--learning_rate', type=float, default=3e-5, # Adjusted default LR
                              help="Peak learning rate for AdamW.")
    train_group.add_argument('--lr_warmup_steps', type=int, default=2000, # Adjusted default warmup
                              help="Number of linear warmup steps for LR scheduler.")
    train_group.add_argument('--lr_total_steps', type=int, default=None,
                              help="Total steps for cosine decay (estimated by Trainer if None).")
    train_group.add_argument('--batch_size', type=int, default=256, # Adjusted default batch size
                              help="Per-device batch size.")
    train_group.add_argument('--effective_batch_size', type=int, default=256, # Adjusted default effective batch size
                              help="Target effective batch size (achieved via gradient accumulation).")
    train_group.add_argument('--precision', type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed", "64"],
                              help="Training precision (e.g., '16-mixed', 'bf16-mixed', '32').")
    train_group.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use (0 for CPU).")
    train_group.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    train_group.add_argument('--offline_model', action=argparse.BooleanOptionalAction, default=False, help="Use ESM3 weights stored in ./data/weights (can symlink from .cache).")

    # --- Data & Caching ---
    data_group = parser.add_argument_group("Data Handling")
    data_group.add_argument('--data_path', type=str, required=True, help="Base directory containing data CSV files.")
    data_group.add_argument('--train_data_file', type=str, required=True, help="Filename of the training data CSV within data_path.")
    data_group.add_argument('--val_data_file', type=str, default=None, help="Filename of the validation data CSV (uses train data if None).")
    data_group.add_argument('--dataloading', type=str, default="cycle", required=True, choices=["pool", "cycle"], help="Whether to cycle one protein at a time or pool them together")
    data_group.add_argument('--loader_strategy', type=str, default='all', choices=['equal', 'min', 'all'], help="Whether to subsample (equal) or use all points (all)")
    data_group.add_argument('--use_subset_restrict', action=argparse.BooleanOptionalAction, default=False, help="Whether to construct batches while limiting the amount of mut_ctx examples")
    data_group.add_argument('--positional', action="store_true", help="Whether to group all mutations to a given amino acid(s) together")
    data_group.add_argument('--split_file', type=str, default=None, help="Optional CSV file defining train/val protein splits.")
    data_group.add_argument('--split_key', type=str, default='stability', help="Key (row index) in split_file to use.")
    data_group.add_argument('--score_column', type=str, default='ddG_ML', help="Column name for ground truth scores/ranks in CSVs.")
    data_group.add_argument('--cache_path', type=str, default='./data_cache', help="Directory to cache processed ProteinStructureMutationDataset objects.")
    data_group.add_argument('--regenerate_cache', action='store_true', help="Force regeneration of dataset cache files.")
    data_group.add_argument('--num_workers', type=int, default=-1, help="Number of worker processes for DataLoader (-1 for all).")
    data_group.add_argument('--max_train_proteins', type=int, default=-1, help="Limit training to the first N proteins (-1 for all).")
    data_group.add_argument('--incl_destab', action=argparse.BooleanOptionalAction, default=False, help="Whether to include destabilized backbones (missing structural information).")
    data_group.add_argument('--singles_only', action=argparse.BooleanOptionalAction, default=False, help="Whether to only train on single mutations.")
    #data_group.add_argument('--doubles_only', action=argparse.BooleanOptionalAction, default=False, help="Whether to only train on double mutations.")
    data_group.add_argument('--incl_doubles', action=argparse.BooleanOptionalAction, default=False, help="Whether to use include explicit doubles for training.")
    data_group.add_argument('--use_mut_structs', action=argparse.BooleanOptionalAction, default=False, help="Whether to use Rosetta-generated single mutant structures for context.") 
    data_group.add_argument('--mut_structures_root', type=str, default='/home/sareeves/PSLMs/data/lora/FINAL_results/', help="Whether to use Rosetta-generated single mutant structures for context.")
    data_group.add_argument('--incl_reversions', action=argparse.BooleanOptionalAction, default=False, help="Whether to use Rosetta-generated single mutant structures for reversions.")
    data_group.add_argument('--use_plddt', action=argparse.BooleanOptionalAction, default=False, help="Whether to use input per-residue PLDDT from AF2 structures.")

    # --- Checkpointing & Logging ---
    log_group = parser.add_argument_group("Checkpointing & Logging")
    log_group.add_argument('--experiment_name', type=str, required=True, help="Experiment name for logging.")
    log_group.add_argument('--version', type=str, default=None, help="Run version for logging (defaults to auto-increment).")
    log_group.add_argument('--checkpoint_path', type=str, default='./checkpoints', help="Base directory to save checkpoints.")
    log_group.add_argument('--checkpoint_filename', type=str, default='{epoch:02d}-{val_rho_avg:.3f}',
                             help="Checkpoint filename format (within experiment_name/version dir).")
    log_group.add_argument('--monitor_metric', type=str, default='val_rho_avg', help="Metric to monitor for saving best checkpoints.")
    log_group.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help="Monitor mode ('max' or 'min').")
    log_group.add_argument('--save_top_k', type=int, default=5, help="Save top K checkpoints based on monitor_metric.")
    log_group.add_argument('--load_lora_checkpoint', type=str, default=None,
                             help="Path to checkpoint file to load LoRA weights from (used by 'lora'/'both').")
    log_group.add_argument('--log_dir', type=str, default='./logs', help="Base directory for CSV and other logs.")
    log_group.add_argument('--comet_api_key', type=str, default=None, help="Comet ML API key (optional).")
    log_group.add_argument('--comet_project_name', type=str, default="esm-msr-final", help="Comet ML project name.")
    log_group.add_argument('--log_every_n_steps', type=int, default=50, help="Log training metrics every N steps.")
    log_group.add_argument('--check_val_every_n_epoch', type=int, default=1, help="Run validation every N epochs.")
    log_group.add_argument('--num_sanity_val_steps', type=int, default=2,
                             help="Number of validation batches to run before training starts (0 to disable).")
    log_group.add_argument('--offline', type=bool, default=False, action=argparse.BooleanOptionalAction, help="Save dashboard to local zip if running offline.")
    log_group.add_argument('--skip_val', type=bool, default=False, action=argparse.BooleanOptionalAction, help="Whether to skip initial (zero shot) evaluation.")
    log_group.add_argument('--resume_global_step', type=int, default=0, help="Resume schedule for partial run.")

    # Parse known args for main parser
    args, remaining_argv = parser.parse_known_args()
    args.mask_sequence_pos = not args.unmask_sequence_pos

    # Check if any arguments were truly unrecognized by any relevant parser
    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    # --- Argument Validation / Warnings ---
    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        num_workers_to_use = max(1, num_cpus - 1) # Example: Use all but one
        args.num_workers = num_workers_to_use

    return args


# === Main Execution Function ===

def main(args: argparse.Namespace):
    """Main training loop setup and execution."""
    logging.info("Starting training script...")
    logging.info(f"Full Arguments: {args}")
    pl.seed_everything(args.seed)

    if args.offline_model:
        os.environ['INFRA_PROVIDER'] = "1"

    # --- Setup Tokenizer/Structure Encoder ---
    # Load a temporary base model instance just to get tokenizer/encoder
    try:
        tokenizer = EsmSequenceTokenizer("cpu")
        structure_encoder = ESM3_structure_encoder_v0("cpu")
        logging.info("Successfully obtained tokenizer and structure encoder.")
    except Exception as e:
        logging.error(f"Failed to load base model for tokenizer/encoder: {e}. Exiting.")
        return

    # --- Setup DataLoaders ---
    try:
        train_loaders, val_loaders, train_names, val_names = setup_dataloaders(args, tokenizer, structure_encoder)
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        logging.error(f"Failed to setup dataloaders: {e}. Exiting.")
        return

    # --- Trainer Setup ---
    if torch.cuda.is_available() and args.gpus > 0:
         accelerator = "gpu"
         devices = args.gpus
         model_device = 'cuda:0'
         # Use DDP only if multiple GPUs are requested
         strategy = 'auto'
    else:
         accelerator = "cpu"
         devices = 1
         model_device = 'cpu'
         strategy = 'auto'
         logging.info("CUDA not available or gpus=0. Using CPU.")
    logging.info(f"Trainer using accelerator: {accelerator}, devices: {devices}, strategy: {strategy if isinstance(strategy, str) else type(strategy).__name__}")

    # --- Initialize Lightning Module ---
    try:
        lightning_model = ESM3EpistasisLightningModule(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            include_structure_encoder=args.include_structure_encoder,
            use_dora=args.use_dora,
            use_qkv_lora=args.use_qkv_lora,
            calibration=args.calibration,
            shared_scale=args.shared_scale_init,
            shared_bias=args.shared_bias_init,
            single_scale=args.single_scale_init,
            single_bias=args.single_bias_init,
            mutctx_scale=args.mutctx_scale_init,
            mutctx_bias=args.mutctx_bias_init,
            reversion_scale=args.reversion_scale_init,
            reversion_bias=args.reversion_bias_init,
            detach_regression=args.detach_regression,
            singles_only=args.singles_only,
            incl_doubles=args.incl_doubles, 
            positional=args.positional,
            subset_size=args.subset_size,            # Pass dict
            invert_list_loss=args.invert_list_loss,
            learning_rate=args.learning_rate,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_total_steps=args.lr_total_steps,
            batch_size=args.batch_size,
            effective_batch_size=args.effective_batch_size,
            val_dataloader_names=val_names,
            train_dataloader_names=train_names,
            precision=args.precision,
            seed=args.seed,
            freeze_lora=args.freeze_lora,
            rank_tau_init=args.rank_tau_init,
            rank_loss=args.rank_loss,
            reg_loss=args.reg_loss,
            epi_loss = args.epi_loss,  # 'huber' or 'mse' or 'listmle',
            lambda_rank = args.lambda_rank,
            lambda_reg = args.lambda_reg,
            lambda_epi = args.lambda_epi,
            mut_ctx_weight=args.mut_ctx_weight,
            tokenizer = tokenizer,
            use_plddt = args.use_plddt,
            use_subset_restrict = args.use_subset_restrict,
            mask_sequence_fraction=args.mask_sequence_fraction,
            mask_coords_fraction=args.mask_coords_fraction,
            mask_structure_fraction=args.mask_structure_fraction,
            mask_sequence_pos=args.mask_sequence_pos,
            mask_coords_pos=args.mask_coords_pos,
            mask_structure_pos=args.mask_structure_pos,
            flank_seq=args.mask_sequence_flank,
            flank_coords=args.mask_coords_flank,
            flank_struct=args.mask_structure_flank,           
            additive_condition=args.additive_condition,
            effect_strategy=args.effect_strategy,
            resume_global_step=args.resume_global_step,
            model_device=model_device
        )
    except (ValueError, AttributeError, KeyError) as e:
         logging.error(f"Failed to initialize Lightning Module: {e}. Check arguments and model/loss definitions.", exc_info=True)
         return

    # --- Load Pre-trained Weights (LoRA and/or Head) ---
    if args.load_lora_checkpoint:
        logging.info(f"Attempting to load LoRA weights from: {args.load_lora_checkpoint}")
        lightning_model.model = utils.load_ckpt_weights(lightning_model.model, args.load_lora_checkpoint, device=lightning_model.device)
        # No warning needed here, load_lora_weights handles checks internally

    # --- Logging & Callbacks ---
    loggers = []
    # CSV Logger
    if args.log_dir:
        try:
            # Ensure version is handled correctly (use PL's default if None)
            csv_logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name, version=args.version)
            loggers.append(csv_logger)
            # Access the determined log_dir AFTER initialization
            log_save_dir = csv_logger.log_dir if hasattr(csv_logger, 'log_dir') else os.path.join(args.log_dir, args.experiment_name, csv_logger.version)
            logging.info(f"Logging CSV to {log_save_dir}")
            # Use this determined directory for checkpointing too
            checkpoint_dir = log_save_dir # Checkpoints inside the log dir
        except Exception as e:
             logging.error(f"Failed to initialize CSVLogger: {e}. Check permissions for {args.log_dir}.")
             checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment_name, args.version or "default_version") # Fallback path
    else:
        logging.warning("No log_dir specified. CSV logs will not be saved.")
        checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment_name, args.version or "default_version")

    # Comet Logger (Optional)
    if args.comet_api_key:
        comet_logger = CometLogger(
            api_key=args.comet_api_key,
            project=args.comet_project_name,
            name=f"{args.experiment_name}-{args.version or 'run'}" # Combine name/version
        )
        loggers.append(comet_logger)
        logging.info(f"Logging to Comet ML project: {args.comet_project_name}")

    # Checkpoint Callback
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=args.checkpoint_filename, # Use format string directly
        save_top_k=args.save_top_k,
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        save_last=True, # Save the last epoch checkpoint
        save_weights_only=False # Save full lightning checkpoint (includes hparams etc.)
                               # Our on_save_checkpoint modifies the state_dict within it
    )
        
    callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=min(10, args.log_every_n_steps))]

    try:
        trainer = Trainer(
            max_epochs=args.num_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=args.precision,
            logger=loggers if loggers else False, # Pass list of loggers or False
            callbacks=callbacks,
            enable_checkpointing=True, # Redundant? Callback handles it.
            num_sanity_val_steps=args.num_sanity_val_steps,
            log_every_n_steps=args.log_every_n_steps,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
        )
    except Exception as e:
         logging.error(f"Failed to initialize PyTorch Lightning Trainer: {e}", exc_info=True)
         return

    # --- Training ---
    if not args.skip_val:
        logging.info("Running initial validation...")
        trainer.validate(lightning_model, dataloaders=val_loaders)
    logging.info("Starting trainer.fit()...")
    try:
        trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders)
        logging.info("Training finished.")
    except Exception as e:
         logging.error(f"Error during training: {e}", exc_info=True)
         return # Exit after error

    # --- Final Actions ---
    logging.info("Training loop completed.")
    # Save final model components explicitly (LoRA/Head) if needed
    # Note: The 'last.ckpt' saved by ModelCheckpoint contains the necessary state.
    # Explicit saving might be useful for creating smaller, component-specific files.
    final_lora_path = os.path.join(checkpoint_dir, "final_lora_weights.pt")

    # Check if LoRA weights exist in the final model state
    if any('lora' in name for name, _ in lightning_model.model.named_parameters()):
        try:
            lora_weights = {name: param for name, param in lightning_model.model.state_dict().items() if 'lora' in name}
            if lora_weights:
                 torch.save(lora_weights, final_lora_path)
                 logging.info(f"Saved final LoRA-only weights to {final_lora_path}")
        except Exception as e:
            logging.error(f"Failed to save final LoRA weights separately: {e}")

    logging.info("Script execution finished.")


# === Entry Point ===
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
