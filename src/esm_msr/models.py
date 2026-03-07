import logging
from typing import Dict, Any, Optional
import numpy as np
import os
import re
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C

from collections import defaultdict
from esm_msr import utils


class StabilityPredictorBase(nn.Module):
    """Base class for stability prediction models using ESM3."""
    def __init__(self,
                 esm_model: nn.Module, # Can be base or PEFT model
        ):
        super().__init__()
        self.model = esm_model # Store the passed model (base or PEFT)

        # Handle potential differences in tokenizer access
        if hasattr(esm_model, 'tokenizers') and hasattr(esm_model.tokenizers, 'sequence'):
             self.sequence_tokenizer = self.model.tokenizers.sequence
        else:
             raise AttributeError("Could not find sequence tokenizer in the provided ESM model.")
        
        # Handle potential differences in tokenizer access
        try:
             self.structure_encoder = self.model.get_structure_encoder()
        except AttributeError:
             raise AttributeError("Could not find structure encoder for the provided ESM model.")

        self.vocab = self.sequence_tokenizer.get_vocab()
        self.valid_canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.canonical_aa_token_ids = [self.vocab.get(wt_aa) for wt_aa in self.valid_canonical_aas]
        self.canonical_idx_tensor = torch.tensor(self.canonical_aa_token_ids, dtype=torch.long)

    def _get_esm3_outputs(self,
                          sequence_tokens: torch.Tensor,
                          structure_coords: Optional[torch.Tensor] = None,
                          structure_tokens: Optional[torch.Tensor] = None,
                          per_res_plddt: Optional[torch.Tensor] = None
                          ):
        """
        Internal helper to run the underlying ESM model's forward pass.
        Returns the output heads dictionary and the final embedding.
        Uses self.model which could be base or PEFT.
        """
        # Input preparation logic (same as before)
        def _prepare_input(tensor, expected_dims):
            if tensor is None or not torch.is_tensor(tensor): return None
            # Handle batch dimension if missing (e.g., for single sample inference)
            if tensor.dim() == expected_dims - 1:
                 tensor = tensor.unsqueeze(0)
            # Squeeze extra singleton dimensions (e.g., B, 1, L -> B, L)
            while tensor.dim() > expected_dims and tensor.shape[1] == 1:
                 tensor = tensor.squeeze(1)
            # Final check
            if tensor.dim() != expected_dims:
                 logging.warning(f"Input tensor shape {tensor.shape} doesn't match expected dims {expected_dims} after preparation.")
            return tensor

        # Prepare inputs based on expected dimensions (adjust if ESM3 expects different shapes)
        sequence_tokens = _prepare_input(sequence_tokens, 2) # Expected (B, L)
        structure_coords = _prepare_input(structure_coords, 4) # Expected (B, L, 37, 3)
        structure_tokens = _prepare_input(structure_tokens, 2) # Expected (B, L)
        per_res_plddt = _prepare_input(per_res_plddt, 2) # Expected (B, L)

        return  self.model.model(
                    sequence_tokens=sequence_tokens,
                    structure_coords=structure_coords,
                    structure_tokens=structure_tokens,
                    per_res_plddt=per_res_plddt
                )
    

class CalibrationHead(nn.Module):
    """
    y_cal = scale * y_raw + bias
    scale is constrained positive via softplus(raw_scale) + min_scale.
    If init_bias=None, no bias Parameter is registered; forward uses 0.
    """
    def __init__(
        self,
        init_scale: float | None = 1/3, # rough approximation, gets calibrated
        init_bias: float | None = 0.0,
        *,
        min_scale: float = 1e-4,
        beta: float = 1.0,
        max_scale: float | None = None,
        requires_grad: bool = True,
    ):
        super().__init__()

        # Initialize raw_scale s.t. softplus(raw)+min_scale ≈ init_scale
        self.use_scale = True
        if init_scale is None:
            self.use_scale = False
            init_scale = 1.0
        else:
            init_scale = float(init_scale)
        target = max(init_scale - min_scale, 1e-12)
        raw_init = self._inv_softplus(torch.tensor(target, dtype=torch.float32), beta=beta)
        self.raw_scale = nn.Parameter(raw_init, requires_grad=requires_grad)

        if init_bias is None:
            # No bias parameter at all → avoids accidental re-enabling
            self.register_parameter("bias", None)
        else:
            init_bias = float(init_bias)
            self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32),
                                     requires_grad=requires_grad)

        self.min_scale = float(min_scale)
        self.beta = float(beta)
        self.max_scale = float(max_scale) if max_scale is not None else None

    @staticmethod
    def _inv_softplus(y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        # x = (1/beta) * log(exp(beta*y) - 1), with a stable large-y branch
        by = beta * y
        out = torch.empty_like(y)
        large = by > 20.0
        out[large] = by[large]                               # ~ y*beta for large y
        out[~large] = torch.log(torch.expm1(by[~large]))
        return out / beta

    @property
    def scale(self) -> torch.Tensor:
        s = F.softplus(self.raw_scale, beta=self.beta) + self.min_scale
        if self.max_scale is not None:
            s = torch.clamp(s, max=self.max_scale)
        return s

    def forward(self, y_raw: torch.Tensor) -> torch.Tensor:
        s = 1.0 if not self.use_scale else self.scale
        b = 0.0 if self.bias is None else self.bias
        return y_raw * s + b
    

class ESM3LoRAModel(StabilityPredictorBase):
    """
    Uses a PEFT-modified (e.g., LoRA) ESM3 model to predict stability scores
    based on the difference in log-likelihoods between mutant and wild-type
    residues at mutation sites. Designed for ranking objectives (e.g., ListMLE).
    """
    def __init__(
            self, 
            peft_model: nn.Module, 
            freeze_lora: bool = False, 
            shared_scale: float | None = None,
            shared_bias: float | None = None,
            single_scale: float | None = None,
            single_bias: float | None = None,
            mutctx_scale: float | None = None,
            mutctx_bias: float | None = None,
            reversion_scale: float | None = None,
            reversion_bias: float | None = None,
            inference_mode: bool = False,
            log_likelihood: bool = False,
            use_plddt: bool = False,
            quaternary_mode: str = 'single_chain'
            ):
        super().__init__(esm_model=peft_model)

        self.quaternary_mode = quaternary_mode

        if freeze_lora:
            print('Freezing LoRA weights')
            for name, p in self.model.model.named_parameters():
                if 'lora' in name:
                    p.requires_grad = False
        logging.info(f"Initialized {'frozen ' if freeze_lora else ''}ESM3LoRAModel (uses PEFT model for likelihoods) with {'learnable' if not inference_mode else 'fixed'} calibration.")

        if single_scale is not None or single_bias is not None:
            self.calibration_head_single = CalibrationHead(init_scale=single_scale, init_bias=single_bias, requires_grad=not inference_mode)
        if mutctx_scale is not None or mutctx_bias is not None:
            self.calibration_head_mut_ctx = CalibrationHead(init_scale=mutctx_scale, init_bias=mutctx_bias, requires_grad=not inference_mode)
        if reversion_scale is not None or reversion_bias is not None:
            self.calibration_head_reversion = CalibrationHead(init_scale=reversion_scale, init_bias=reversion_bias, requires_grad=not inference_mode)
        if shared_scale is not None or shared_bias is not None:
            self.calibration_head_shared = CalibrationHead(init_scale=shared_scale, init_bias=shared_bias, requires_grad=not inference_mode)

        # RAM caches (process-local; not persisted)
        self._ram_cache = {
            "singles": {}  # key: (pdb_basename, chain) -> {"scheme": str, "wt_ids": List[int], "per_pos_logits": torch.Tensor[L,V]}
        }

        self.warned_st = False
        self.log_likelihood = log_likelihood
        self.use_plddt = use_plddt

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Check if this is a paired batch (WT + Mut)
        if 'sequence_tokens_mut' in batch and 'sequence_tokens_wt' in batch:
            # Use the Delta Mean LL approach
            return self.forward_delta_seq(batch)
        
        # Check if this is a standard/legacy batch (Single set of tokens)
        elif 'sequence_tokens' in batch and 'use_absolute' in batch:
            # If the batch has 'mutations', use the original single-point logit logic
            # If it doesn't, return the Mean LL of the single sequence provided
            return self.forward_PLL(batch)
        
        # Check if this is a standard/legacy batch (Single set of tokens)
        elif 'sequence_tokens' in batch and 'mutations' in batch:
            # If the batch has 'mutations', use the original single-point logit logic
            # If it doesn't, return the Mean LL of the single sequence provided
            return self.forward_delta_logits(batch)

        else:
            raise ValueError(f"Batch keys {batch.keys()} not recognized by Forward.")
    
    def forward_delta_logits(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Computes stability scores as the sum of log-likelihood differences
        (Mutant - WildType) at mutation sites using the PEFT model's logits.
        """
        sequence_tokens = batch['sequence_tokens']
        mutations = batch.get('mutations')
        if mutations is None:
            raise ValueError("Batch is missing 'mutations' key required for ESM3LoRAModel.")
        
        coords = batch['coords']
        structure_tokens = batch['structure_tokens']
        plddt = batch['plddt']

        if structure_tokens.sum() < 0:
            residue_index = batch['residue_index']
            coords_ = coords.clone().squeeze(1)
            coords_unpadded = coords_[:, 1:-1, :, :]
            _, structure_tokens = self.model._structure_encoder.encode(coords_unpadded, residue_index=residue_index)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
            structure_tokens[:, 0] = C.STRUCTURE_BOS_TOKEN
            structure_tokens[:, -1] = C.STRUCTURE_EOS_TOKEN

        device = sequence_tokens.device
        batch_size, seq_len = sequence_tokens.shape

        # --- Get Model Outputs (Logits) ---
        with autocast():
            # _get_esm3_outputs uses self.model, which is the peft_model here
            model_outputs = self._get_esm3_outputs(
                sequence_tokens=sequence_tokens,
                structure_coords=coords,
                structure_tokens=structure_tokens,
                per_res_plddt=plddt if self.use_plddt else None
            )

        # Extract sequence logits
        sequence_logits = model_outputs['sequence_logits'] if isinstance(model_outputs, dict) else \
            model_outputs.sequence_logits # Shape: (B, L, VocabSize)
        if sequence_logits is None:
             raise ValueError("ESM3 model output missing 'sequence_logits'.")

        # --- Calculate Scores ---
        batch_scores = torch.zeros(batch_size, device=device, dtype=sequence_logits.dtype)
        for i in range(batch_size):
            protein_mutations = mutations[i]
            if not protein_mutations: # Handle empty mutation list for a sample
                try:
                    logging.warning(f"Sample {i} from {batch[0]['pdb']} has no mutations.")
                except:
                    logging.warning(f"Sample {i} has no mutations.")
                continue

            for wt_aa, pos_idx, mut_aa in protein_mutations:
                if 0 <= pos_idx < seq_len:
                    try:
                        wt_id = self.vocab.get(wt_aa)
                        mut_id = self.vocab.get(mut_aa)
                        if wt_id is None or mut_id is None:
                            raise KeyError(f"AA '{wt_aa if wt_id is None else mut_aa}' not in vocab.")

                        pos_logits = sequence_logits[i, pos_idx, :].float()
                        
                        if self.log_likelihood:
                            # 1. Create a mask of negative infinity
                            masked_logits = torch.full_like(pos_logits, float('-inf'))
                            
                            # 2. Copy only the canonical logits into the mask
                            # (Requires self.canonical_idx_tensor to be on pos_logits.device)
                            idx = self.canonical_idx_tensor.to(pos_logits.device)
                            masked_logits[idx] = pos_logits[idx]
                            
                            # 3. Calculate log probabilities strictly over the 20 AAs
                            pos_logits = torch.nn.functional.log_softmax(masked_logits, dim=-1)
                            #print(pos_logits)

                        delta = pos_logits[mut_id] - pos_logits[wt_id]

                        batch_scores[i] += delta.to(sequence_logits.dtype)

                    except KeyError as e: logging.warning(f"Sample {i}, Pos {pos_idx}: {e}")
                    except IndexError: logging.warning(f"Sample {i}: Pos {pos_idx} out of bounds for seq len {seq_len}.")
                else: logging.warning(f"Sample {i}: Invalid mutation position {pos_idx} for seq len {seq_len}.")
        return batch_scores
    
    def forward_delta_seq(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Computes stability scores as the difference in Mean Log-Likelihood:
        Score = Mean_LL(Mutant) - Mean_LL(WildType)
        """

        structure_tokens = batch['structure_tokens_mut']
        if structure_tokens.sum() < 0:
            if not self.warned_st:
                print('Computing structure tokens')
                self.warned_st = True
            residue_index = batch['residue_index_mut']
            coords_ = batch['coords_mut'].clone().squeeze(1)
            coords_unpadded = coords_[:, 1:-1, :, :]
            _, structure_tokens = self.model._structure_encoder.encode(coords_unpadded, residue_index=residue_index)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=C.STRUCTURE_PAD_TOKEN)
            structure_tokens[:, 0] = C.STRUCTURE_BOS_TOKEN
            structure_tokens[:, -1] = C.STRUCTURE_EOS_TOKEN
            batch['structure_tokens_mut'] = structure_tokens
        
        # 1. Run Mutant Forward Pass
        mut_logits = self._get_esm3_outputs(
            sequence_tokens=batch['sequence_tokens_mut'],
            structure_coords=batch['coords_mut'],
            structure_tokens=batch['structure_tokens_mut']
        ).sequence_logits # (B, L_mut, V)

        # 2. Run Wild-Type Forward Pass 
        # (Note: In a production environment, you might cache WT logits 
        # if 90% of your data shares the same 3 WTs)
        wt_logits = self._get_esm3_outputs(
            sequence_tokens=batch['sequence_tokens_wt'],
            structure_coords=batch['coords_wt'],
            structure_tokens=batch['structure_tokens_wt']
        ).sequence_logits # (B, L_wt, V)

        # 3. Compute Mean Log-Likelihoods
        mut_mll = self._compute_mean_ll(
            logits=mut_logits, 
            tokens=batch['sequence_tokens_mut'], 
            ignore_index=C.SEQUENCE_PAD_TOKEN
        )
        
        wt_mll = self._compute_mean_ll(
            logits=wt_logits, 
            tokens=batch['sequence_tokens_wt'], 
            ignore_index=C.SEQUENCE_PAD_TOKEN
        )

        # 4. Final Delta Score
        # Higher score = Mutant is more plausible/stable than WT
        batch_scores = mut_mll - wt_mll
        
        return batch_scores
    
    def forward_PLL(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Computes stability scores as the difference in Mean Log-Likelihood:
        Score = Mean_LL(Mutant) - Mean_LL(WildType)
        """

        structure_tokens = batch['structure_tokens']
        if structure_tokens.sum() < 0:
            if not self.warned_st:
                print('Computing structure tokens')
                self.warned_st = True
            residue_index = batch['residue_index']
            coords_ = batch['coords'].clone().squeeze(1)
            coords_unpadded = coords_[:, 1:-1, :, :]
            _, structure_tokens = self.model._structure_encoder.encode(coords_unpadded, residue_index=residue_index)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=C.STRUCTURE_PAD_TOKEN)
            structure_tokens[:, 0] = C.STRUCTURE_BOS_TOKEN
            structure_tokens[:, -1] = C.STRUCTURE_EOS_TOKEN
            batch['structure_tokens'] = structure_tokens
        
        mask = torch.ones_like((batch['sequence_tokens']), dtype=torch.bool, device='cuda')
        sequence_tokens_masked = batch['sequence_tokens'].masked_fill(mask, C.SEQUENCE_MASK_TOKEN)
        sequence_tokens_masked[:, 0] = C.SEQUENCE_BOS_TOKEN
        sequence_tokens_masked[:, -1] = C.SEQUENCE_EOS_TOKEN
        
        logits = self._get_esm3_outputs(
            sequence_tokens=sequence_tokens_masked, #batch['sequence_tokens'],
            structure_coords=batch['coords'],
            structure_tokens=batch['structure_tokens']
        ).sequence_logits # (B, L_mut, V)
        
        pll = self._compute_mean_ll(
            logits=logits, 
            tokens=batch['sequence_tokens'], 
            ignore_index=C.SEQUENCE_PAD_TOKEN
        )
        
        return pll

    def _compute_mean_ll(self, logits: torch.Tensor, tokens: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """
        Calculates the average log-probability of the ground-truth tokens.
        """
        # logits: (B, L, V), tokens: (B, L)
        # Shift logits and tokens for next-token prediction if needed, 
        # but for ESM3 Masked LM style, we usually align directly.
        
        #log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather the log-probs of the actual tokens
        # (B, L, 1)
        token_log_probs = torch.gather(logits, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        mask = (tokens != ignore_index).float()
        masked_log_probs = token_log_probs * mask
        
        # Compute mean per sequence (sum / number of non-pad tokens)
        seq_sums = masked_log_probs.sum(dim=-1)
        seq_counts = mask.sum(dim=-1)
        
        return seq_sums / seq_counts
    
    def infer_seq(self, pdb_path, chain='A'):
        dev = next(self.parameters()).device
        seq_tokens, coords, plddt, structure_tokens = self.preprocess_structure(pdb_path, chain, dev=dev, backbone_mutation=None, assert_wt=True, mask_ctx=False)
        batch = {}
        batch['sequence_tokens'] = seq_tokens
        batch['coords'] = coords
        batch['structure_tokens'] = structure_tokens

        return self.forward_PLL(batch)
    
    # Common ESM forward wrapper
    def _forward_raw(self, seq_batch, coords_batch, struct_batch, plddt_batch=None):
        with autocast():
            outputs = self._get_esm3_outputs(
                sequence_tokens=seq_batch,
                structure_coords=coords_batch,
                structure_tokens=struct_batch,
                per_res_plddt=plddt_batch
            )
        seq_logits_local = outputs["sequence_logits"] if isinstance(outputs, dict) else outputs.sequence_logits
        if seq_logits_local is None:
            raise RuntimeError("ESM3 model output missing 'sequence_logits'")
        return seq_logits_local

    def _handle_mutated_backbone(self, protein_chain, coords, structure_tokens, backbone_mutation=None, assert_wt=False, assert_mut=False, mask_ctx=False):
        corrected_seq = protein_chain.sequence

        mutated_backbone_pos: Optional[int] = None
        if backbone_mutation:
            mutated_backbone_pos = int(backbone_mutation[1:-1])
            wt = backbone_mutation[0]
            mut = backbone_mutation[-1]
            if assert_wt:
                assert corrected_seq[mutated_backbone_pos-1] == wt
                assert mut in 'ACDEFGHIKLMNPQRSTVWY', (mut, backbone_mutation)
                corrected_seq = list(corrected_seq)
                corrected_seq[mutated_backbone_pos-1] = mut
                print(f"Backbone implies mutated residue at {mutated_backbone_pos}: {wt}->{mut}; preserved in corrected_seq.")
            elif assert_mut:
                assert corrected_seq[mutated_backbone_pos-1] == mut
                assert mut in 'ACDEFGHIKLMNPQRSTVWY', (mut, backbone_mutation)

        if mask_ctx:
            print(f"Also masking the coordinates and structure tokens at {mutated_backbone_pos}")
            structure_tokens[:, mutated_backbone_pos] = C.STRUCTURE_MASK_TOKEN
            coords[:, mutated_backbone_pos, :, :] = np.nan
            
        corrected_seq = ''.join(corrected_seq)

        return corrected_seq, coords, structure_tokens

    def preprocess_structure(self, pdb_path, chain, dev, backbone_mutation, assert_wt=False, assert_mut=False, mask_ctx=False):
        # ----------------------------
        # 1) Load structure + encode
        # ----------------------------

        protein_chain = ProteinChain.from_pdb(pdb_path, chain, is_predicted=True)

        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        coords = coords.to(dev)
        residue_index = residue_index.to(dev)
        
        _, structure_tokens = self.structure_encoder.encode(coords, residue_index=residue_index)

        # Pad BOS/EOS (structure side)
        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
        structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
        if structure_tokens.shape[1] > 0:
            structure_tokens[:, 0] = C.STRUCTURE_BOS_TOKEN
            structure_tokens[:, -1] = C.STRUCTURE_EOS_TOKEN

        corrected_seq, possibly_masked_coords, possibly_masked_structure_tokens = \
            self._handle_mutated_backbone(protein_chain, coords, structure_tokens, backbone_mutation=backbone_mutation, assert_wt=assert_wt, assert_mut=assert_mut, mask_ctx=mask_ctx)

        # Sequence tokens (these already include BOS/EOS in your tokenizer)
        corrected_seq_tokens = torch.tensor(
            np.array(self.sequence_tokenizer.encode(corrected_seq), dtype=np.int64)
        ).unsqueeze(0).to(dev)  # (1, L_seq+2)

        possibly_masked_coords = possibly_masked_coords.to(dev)
        possibly_masked_structure_tokens = possibly_masked_structure_tokens.to(dev)

        return corrected_seq_tokens, possibly_masked_coords, plddt, possibly_masked_structure_tokens
    
    def preprocess_multimer(self, pdb_path, dev, backbone_mutation, assert_wt=False, assert_mut=False, mask_ctx=False, quaternary_mode='heteromer', target_chain='A'):

        # 1. Figure out which chains are present (Pass 1)
        # We use PDBParser directly here just to get the IDs safely
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('scan', pdb_path)
        except Exception as e:
            raise ValueError(f"Failed to scan chains in {pdb_path}: {e}")

        # Assume Model 0 is the target. 
        # If the PDB is an ensemble, this takes the first model.
        model = structure[0]
        chain_ids = [chain.id for chain in model]
        print(chain_ids)

        all_chains = []
        for chain_id in chain_ids:
            all_chains.append(ProteinChain.from_pdb(pdb_path, chain_id=chain_id))
            
        if not all_chains:
            raise ValueError(f"No valid chains found in {pdb_path}")
        
        selected_chains = []

        if quaternary_mode == 'heteromer':
            # Keep all chains, preserving the order in the file
            selected_chains = all_chains
            
        elif quaternary_mode == 'homomer':
            # 1. Find the sequence of the target chain
            target_sequence = None
            
            # Search for the specific chain ID to establish the target sequence
            for chain in all_chains:
                if chain.chain_id == target_chain:
                    target_sequence = chain.sequence
                    break
            
            if target_sequence is None:
                available_ids = [c.chain_id for c in all_chains]
                raise ValueError(f"Target chain '{target_chain}' not found in PDB. Available chains: {available_ids}")

            # 2. Select ALL chains that share this exact sequence
            # We compare the sequence string directly
            for chain in all_chains:
                if chain.sequence == target_sequence:
                    selected_chains.append(chain)
            
            print(f"Homomer mode: Found {len(selected_chains)} chains matching sequence of chain {target_chain}")

        else:
            raise ValueError(f"Unknown quaternary_mode: {quaternary_mode}")

        # Concatenate selected chains
        protein_chain = ProteinChain.concat(selected_chains)
        print(protein_chain.sequence)

        # ----------------------------
        # 2) Encode structure
        # ----------------------------

        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        coords = coords.to(dev)
        residue_index = residue_index.to(dev)
        
        _, structure_tokens = self.structure_encoder.encode(coords, residue_index=residue_index)

        # Pad BOS/EOS (structure side)
        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
        structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
        
        if structure_tokens.shape[1] > 0:
            structure_tokens[:, 0] = C.STRUCTURE_BOS_TOKEN
            structure_tokens[:, -1] = C.STRUCTURE_EOS_TOKEN
            
            # Explicitly mark chain breaks in the structure tokens.
            # The encoder outputs arbitrary tokens for 'inf' coords, so we overwrite them.
            # The sequence index `i` corresponds to structure_token index `i + 1` (due to BOS).
            for i, char in enumerate(protein_chain.sequence):
                if char == "|":
                    structure_tokens[:, i + 1] = C.STRUCTURE_CHAINBREAK_TOKEN

        # ----------------------------
        # 3) Handle Mutations & Finalize
        # ----------------------------
        
        corrected_seq, possibly_masked_coords, possibly_masked_structure_tokens = \
            self._handle_mutated_backbone(protein_chain, coords, structure_tokens, backbone_mutation=backbone_mutation, assert_wt=assert_wt, assert_mut=assert_mut, mask_ctx=mask_ctx)

        # Sequence tokens (already include BOS/EOS in tokenizer)
        # The tokenizer automatically converts '|' characters to C.SEQUENCE_CHAINBREAK_TOKEN
        corrected_seq_tokens = torch.tensor(
            np.array(self.sequence_tokenizer.encode(corrected_seq), dtype=np.int64)
        ).unsqueeze(0).to(dev)  # (1, L_seq+2)

        possibly_masked_coords = possibly_masked_coords.to(dev)
        possibly_masked_structure_tokens = possibly_masked_structure_tokens.to(dev)

        return corrected_seq_tokens, possibly_masked_coords, plddt, possibly_masked_structure_tokens
    
    @torch.no_grad()
    def infer_mutants(
        self,
        df,
        strategy: str = "masked",          # {"parallel", "masked", "direct"}
        K_paths: int = 4,
        max_mutations: int = 10,
        device: str | torch.device | None = None,
        backbone_mutation: str | None = None,
        use_modeled_context_structs: bool = False,
        mut_structs_root: str | None = None,
        mem_scale: float = 1.0,
        quiet: bool = False
    ):
        if len(set(['wild_type', 'position', 'mutation']).intersection(set(df.columns))) == 3:
            df['mut_type'] = df['wild_type'] + df['position'].astype(int).astype(str) + df['mutation']
        elif 'mut_info_seq_pos' in df.columns:
            df['mut_type'] = df['mut_info_seq_pos']
        
        # 1. Generate common columns (wt1, pos1, mut1, ..., frX, posX, toX)
        df = utils.parse_multimutant_column(df, 'mut_type', max_mutations=max_mutations)

        if 'chain' not in df.columns:
            logging.warning(f"Chain not specified. Assuming it is always A")
            df['chain'] = 'A'

        all_preds = []

        # Group by PDB and Chain to handle multiple structures in one batch
        for (pdb, chain), data in df.groupby(['pdb_file', 'chain']):
            # Split data based on mutation count inferred from colons in 'mut_type'
            singles = data.loc[data['mut_type'].str.count(':') == 0]
            doubles = data.loc[data['mut_type'].str.count(':') == 1]
            multi = data.loc[data['mut_type'].str.count(':') >= 2]

            if len(singles) > 0:
                pred_singles = self.infer_single_mutants(
                    pdb, chain=chain, strategy=strategy, subset_df=singles, mem_scale=mem_scale,
                    backbone_mutation=backbone_mutation, quiet=quiet, device=device
                )
                all_preds.append(pred_singles)
            
            if len(doubles) > 0:
                pred_doubles = self.infer_double_mutants(
                    pdb, chain=chain, strategy=strategy, subset_df=doubles, mem_scale=mem_scale,
                    backbone_mutation=backbone_mutation, quiet=quiet, device=device,
                    use_modeled_context_structs=use_modeled_context_structs,
                    mut_structs_root=mut_structs_root,
                )
                all_preds.append(pred_doubles)
            
            if len(multi) > 0:
                if strategy == 'parallel':
                    return pd.DataFrame(columns=['mut_type'])
                pred_multi = self.infer_multimutants_sampled(
                    pdb, chain=chain, strategy=strategy, subset_df=multi,
                    K_paths=K_paths
                )
                all_preds.append(pred_multi)

        if all_preds:
            return pd.concat(all_preds)
        return pd.DataFrame(columns=['mut_type'])
    
    @torch.no_grad()
    def infer_single_mutants(
        self,
        pdb_path: str,
        *,
        chain: str | None = "A",
        strategy: str = "parallel",          # {"parallel", "masked", "direct"}
        device: str | torch.device | None = None,
        mem_scale: float = 1.0,              # ~positions per masked batch; linear VRAM knob
        subset_df: pd.DataFrame | None = None,  # columns: pos, to (1-based position and target AA)
        backbone_mutation: str | None = None,
        quiet: bool = False
    ):
        """
        Δlogits for all single mutants of a protein from a PDB, or a subset if specified.

        - parallel: 1 forward, no masking; read logits at token index 1..L (BOS/EOS at 0, L+1).
        - masked:   build batches of masked inputs where each row masks one site; do multiple forwards if needed.
                    Batch size ~= int(round(mem_scale)). Doubling mem_scale ~ doubles VRAM (linear in batch size).
        - direct:   Same as "masked" for single mutants (masks the mutation position).
        - subset_df: Optional DataFrame with columns ['position', 'mutation'] specifying which single mutants to compute.
                     If None, computes all possible single mutants.

        Returns:
            DataFrame with columns ["pdb","chain","wt1","pos1","mut1","delta_logit1","ddg_pred","scheme"]
        """
        dev = torch.device(device) if device is not None else next(self.parameters()).device

        if self.quaternary_mode == 'single_chain':
            seq_tokens, coords, plddt, structure_tokens = self.preprocess_structure(pdb_path, chain, dev=dev, backbone_mutation=backbone_mutation, assert_wt=True, mask_ctx=False)
        else:
            seq_tokens, coords, plddt, structure_tokens = self.preprocess_multimer(pdb_path, dev=dev, backbone_mutation=backbone_mutation, assert_wt=True, mask_ctx=False, quaternary_mode=self.quaternary_mode, target_chain=chain)
        L_seq = seq_tokens.size()[1] - 2

        # Canonical AAs and ids
        canonical = self.valid_canonical_aas
        can_ids = self.canonical_aa_token_ids

        wt_ids = seq_tokens[0].tolist()  # includes BOS/EOS

        mask_id = C.SEQUENCE_MASK_TOKEN

        # Build aa2id and id2aa mappings
        aa2id = {aa: tid for aa, tid in zip(canonical, can_ids)}
        id2aa = {tid: aa for aa, tid in zip(canonical, can_ids)}

        # ----------------------------
        # Handle subset_df logic
        # ----------------------------
        if subset_df is not None:
            dfq = subset_df.copy()
            required_cols = {'wt1', 'pos1', 'mut1'}
            missing = required_cols - set(dfq.columns)
            if missing:
                raise ValueError(f"subset_df missing columns: {missing}")
            
            # Validate and filter
            dfq = dfq.dropna(subset=['wt1', 'pos1', 'mut1'])
            dfq['pos1'] = dfq['pos1'].astype(int)
            
            def _is_can(x): return x in aa2id
            valid = (
                dfq['pos1'].between(1, L_seq) &
                dfq['mut1'].map(_is_can)
            )
            assert sum(valid.astype(int)) == len(dfq)
            
            # Get unique (pos, to) pairs needed
            singles_needed = sorted(set(zip(dfq['wt1'], dfq['pos1'], dfq['mut1'])))
            positions_needed = sorted(set(dfq['pos1']))
        else:
            dfq = None
            singles_needed = None
            positions_needed = list(range(1, L_seq + 1))

        # ----------------------------
        # Build inputs per strategy
        # ----------------------------
        scheme = strategy.lower()
        if scheme not in ("parallel", "masked", "direct"):
            raise ValueError("strategy must be one of {'parallel','masked','direct'}")

        pdb_id = os.path.basename(pdb_path)
        ch = chain or ""

        rows = []
        # We'll also build a compact per-position logits tensor for caching:
        # per_pos_logits: [L_seq, V], the logits to score single mutants at each site.
        per_pos_logits = None

        if scheme == "parallel":
            if not quiet:
                print(f'Inferring singles; strategy = parallel')
            # Single forward, no masking
            seq_batch = seq_tokens                  # (1, L_seq+2)
            coords_batch = coords                   # (1, L_seq+2, 37, 3)
            struct_batch = structure_tokens         # (1, L_seq+2)

            seq_logits = self._forward_raw(seq_batch, coords_batch, struct_batch)  # (1, L_seq+2, V)
            V = seq_logits.shape[-1]

            # Extract logits at token positions 1..L_seq
            per_pos_logits = seq_logits[0, 1:L_seq+1, :].float().cpu()       # (L_seq, V)

            # If subset_df provided, only compute for requested mutations
            if subset_df is not None:
                for (fr_aa, pos, to_aa) in singles_needed:
                    wt_id = wt_ids[pos]
                    # Map WT id back to an AA (skip if non-canonical)
                    wt_aa = id2aa.get(wt_id)
                    assert fr_aa == wt_aa
                    if wt_aa is None:
                        continue
                    
                    # Skip if trying to mutate to wild-type
                    mut_id = aa2id[to_aa]
                    if mut_id == wt_id:
                        continue

                    pos_logits = per_pos_logits[pos-1]  # (V,)
                    wt_logit = float(pos_logits[wt_id].item())
                    
                    try:
                        cal_pred = self.calibration_head_single(float(pos_logits[mut_id].item() - wt_logit)).cpu().numpy()
                    except AttributeError:
                        cal_pred = self.calibration_head_shared(float(pos_logits[mut_id].item() - wt_logit)).cpu().numpy()
                    
                    rows.append({
                        "pdb": pdb_id,
                        "chain_id": ch,
                        "mut_type": f"{wt_aa}{pos}{to_aa}",
                        "wt1": wt_aa,
                        "pos1": pos,          # 1-based sequence position
                        "mut1": to_aa,
                        "delta_logit1": float(pos_logits[mut_id].item() - wt_logit),
                        "ddg_pred": cal_pred,
                        "scheme": "parallel",
                    })
            else:
                # Full enumeration (original behavior)
                for pos in range(1, L_seq+1):
                    wt_id = wt_ids[pos]
                    # Map WT id back to an AA (skip if non-canonical)
                    wt_aa = None
                    for aa, aa_id in zip(canonical, can_ids):
                        if aa_id == wt_id:
                            wt_aa = aa
                            break
                    if wt_aa is None:
                        continue

                    pos_logits = per_pos_logits[pos-1]  # (V,)
                    wt_logit = float(pos_logits[wt_id].item())

                    for aa, aa_id in zip(canonical, can_ids):
                        if aa_id == wt_id:
                            continue
                        try:
                            cal_pred = self.calibration_head_single(float(pos_logits[aa_id].item() - wt_logit)).cpu().numpy()
                        except AttributeError:
                            cal_pred = self.calibration_head_shared(float(pos_logits[aa_id].item() - wt_logit)).cpu().numpy()
                        rows.append({
                            "pdb": pdb_id,
                            "chain_id": ch,
                            "mut_type": f"{wt_aa}{pos}{aa}",
                            "wt1": wt_aa,
                            "pos1": pos,          # 1-based sequence position
                            "mut1": aa,
                            "delta_logit1": float(pos_logits[aa_id].item() - wt_logit),
                            "ddg_pred": cal_pred,
                            "scheme": "parallel",
                        })

        else:
            # masked OR direct strategy with batching over positions
            # For singles, "direct" means masking the single position, which is identical to "masked"
            
            # Only compute for needed positions (subset or all)
            if subset_df is not None:
                pos_mask = positions_needed
            else:
                pos_mask = list(range(1, L_seq + 1))
            
            positions_per_batch = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
            V = None
            per_pos_logits_dict = {}  # accumulate {pos: logits_row (V,)}

            # Make broadcastable bases once
            coords_base = coords
            struct_base = structure_tokens
            seq_base = seq_tokens

            # Chunk positions into strides of positions_per_batch
            desc_str = f'Inferring singles; strategy = {scheme}'
            for start in tqdm(range(0, len(pos_mask), positions_per_batch), desc=desc_str, disable=quiet):
                chunk_positions = pos_mask[start:start + positions_per_batch]
                B = len(chunk_positions)

                # Build masked batch: B rows, each masks its own position (token index == pos)
                seq_batch = seq_base.repeat(B, 1).clone()              # (B, L_seq+2)
                for row_idx, pos in enumerate(chunk_positions):
                    seq_batch[row_idx, pos] = mask_id

                coords_batch = coords_base.repeat(B, 1, 1, 1)          # (B, L_seq+2, 14, 3)
                struct_batch = struct_base.repeat(B, 1)                # (B, L_seq+2)

                seq_logits_chunk = self._forward_raw(seq_batch, coords_batch, struct_batch)  # (B, L_seq+2, V)
                if V is None:
                    V = seq_logits_chunk.shape[-1]

                # For masked scheme, for a row that masked 'position', we read logits at [row_idx, pos, :]
                for row_idx, pos in enumerate(chunk_positions):
                    per_pos_logits_dict[pos] = seq_logits_chunk[row_idx, pos, :].float().cpu()  # (V,)

            # Rebuild full per_pos_logits tensor if needed for caching (fill with NaN for non-computed positions)
            per_pos_logits = torch.full((L_seq, V), float('nan'), dtype=torch.float32)
            for pos, row in per_pos_logits_dict.items():
                per_pos_logits[pos-1, :] = row

            # Compose rows
            if subset_df is not None:
                for (fr_aa, pos, to_aa) in singles_needed:
                    wt_id = wt_ids[pos]
                    wt_aa = id2aa.get(wt_id)
                    assert fr_aa == wt_aa
                    
                    mut_id = aa2id[to_aa]
                    if mut_id == wt_id:
                        continue

                    if pos not in per_pos_logits_dict:
                        continue
                    
                    pos_logits = per_pos_logits_dict[pos]  # (V,)
                    wt_logit = float(pos_logits[wt_id].item())
                    
                    #try:
                    #    cal_pred = self.calibration_head_single(float(pos_logits[mut_id].item() - wt_logit)).cpu().numpy()
                    #except AttributeError:
                    cal_pred = self.calibration_head_shared(float(pos_logits[mut_id].item() - wt_logit)).cpu().numpy()
                    
                    rows.append({
                        "pdb": pdb_id,
                        "chain_id": ch,
                        "mut_type": f"{wt_aa}{pos}{to_aa}",
                        "wt1": wt_aa,
                        "pos1": pos,
                        "mut1": to_aa,
                        "delta_logit1": float(pos_logits[mut_id].item() - wt_logit),
                        "ddg_pred": cal_pred,
                        "scheme": scheme,
                    })
            else:
                # Full enumeration
                for pos in range(1, L_seq+1):
                    wt_id = wt_ids[pos]
                    wt_aa = None
                    for aa, aa_id in zip(canonical, can_ids):
                        if aa_id == wt_id:
                            wt_aa = aa
                            break
                    if wt_aa is None:
                        continue

                    if pos not in per_pos_logits_dict:
                        continue

                    pos_logits = per_pos_logits_dict[pos]  # (V,)
                    wt_logit = float(pos_logits[wt_id].item())
                    for aa, aa_id in zip(canonical, can_ids):
                        if aa_id == wt_id:
                            continue
                        #try:
                        #    cal_pred = self.calibration_head_single(float(pos_logits[aa_id].item() - wt_logit)).cpu().numpy()
                        #except AttributeError:
                        cal_pred = self.calibration_head_shared(float(pos_logits[aa_id].item() - wt_logit)).cpu().numpy()
                        rows.append({
                            "pdb": pdb_id,
                            "chain_id": ch,
                            "mut_type": f"{wt_aa}{pos}{aa}",
                            "wt1": wt_aa,
                            "pos1": pos,
                            "mut1": aa,
                            "delta_logit1": float(pos_logits[aa_id].item() - wt_logit),
                            "ddg_pred": cal_pred,
                            "scheme": scheme,
                        })

        df = pd.DataFrame(rows)

        # --- RAM cache (not on disk) ---
        key = (os.path.basename(pdb_path), ch)
        self._ram_cache["singles"][key] = {
            "scheme": scheme,               # "parallel" or "masked" or "direct"
            "wt_ids": wt_ids,               # includes BOS/EOS
            "per_pos_logits": per_pos_logits.cpu().float(),  # [L_seq, V]
        }

        return df

    @torch.no_grad()
    def infer_double_mutants(
        self,
        pdb_path: str,
        *,
        chain: str | None = "A",
        strategy: str = "parallel",              # {"parallel","masked","direct"}
        device: str | torch.device | None = None,
        mem_scale: float = 1.0,                  # positions per masked batch (add your L-normalization outside)
        use_modeled_context_structs: bool = False,
        mut_structs_root: str | None = None,
        positions: list[int] | None = None,      # optional 1-based restriction for full enumeration
        subset_df: pd.DataFrame | None = None,   # columns: wt1,pos1,mut1, wt2,pos2,mut2 (1-based)
        reuse_singles: str = "recompute",        # {"auto","require_cache","recompute"}
        backbone_mutation: str | None = None,
        unmask_mut_ctx_id: bool = False,
        quiet: bool = False
    ):
        """
        Compute chain-rule-avg double mutant Δlogits for either:
        - ALL pairs (when subset_df is None), or
        - ONLY the pairs listed in subset_df (efficient path).
        
        If subset_df is provided with 'wt1', 'wt2' columns (wild-type amino acids),
        validates that these match the actual wild-type sequence at the specified positions.
        """

        dev = torch.device(device) if device is not None else next(self.parameters()).device
        scheme = strategy.lower()
        if scheme not in ("parallel", "masked", "direct"):
            raise ValueError("strategy must be one of {'parallel','masked','direct'}")

        # -------------------- Load WT structure/seq --------------------
        seq_tokens_wt, coords_wt, plddt_wt, struct_tokens_wt = self.preprocess_structure(pdb_path, chain, dev=dev, backbone_mutation=backbone_mutation, assert_wt=True, mask_ctx=False)
        L_seq = seq_tokens_wt.size()[1] - 2

        # Canonical maps
        canonical = self.valid_canonical_aas
        can_ids = self.canonical_aa_token_ids
        vocab = self.vocab
        aa2id = {aa: tid for aa, tid in zip(canonical, can_ids)}
        id2aa = {tid: aa for aa, tid in zip(canonical, can_ids)}

        # WT token ids per position (with BOS/EOS)
        wt_token_ids = seq_tokens_wt[0].tolist()
        pdb_id = os.path.basename(pdb_path)
        ch = chain or ""

        # Mask id (for masked scheme)
        if hasattr(self.sequence_tokenizer, "mask_token_id") and (self.sequence_tokenizer.mask_token_id is not None):
            mask_id = int(self.sequence_tokenizer.mask_token_id)
        else:
            mask_id = int(vocab["<mask>"])

        # -------------------- Build needed sets --------------------
        if subset_df is not None:
            dfq = subset_df.copy()
            required_cols = {'wt1','pos1','mut1','wt2','pos2','mut2'}
            missing = required_cols - set(dfq.columns)
            if missing:
                raise ValueError(f"subset_df missing columns: {missing}")
            dfq = dfq.dropna(subset='pos2')
            dfq['pos1'] = dfq['pos1'].astype(int)
            dfq['pos2'] = dfq['pos2'].astype(int)

            def _is_can(x): return x in aa2id
            valid = (
                dfq['pos1'].between(1, L_seq) &
                dfq['pos2'].between(1, L_seq) &
                dfq['mut1'].map(_is_can) &
                dfq['mut2'].map(_is_can) &
                (dfq['pos1'] != dfq['pos2'])
            )

            assert len(valid) == len(dfq)
            dfq = dfq.loc[valid].reset_index(drop=True)

            # -------------------- VALIDATE WILD-TYPE IDENTITIES --------------------
            has_wt1 = 'wt1' in dfq.columns and dfq['wt1'].notna().any()
            has_wt2 = 'wt2' in dfq.columns and dfq['wt2'].notna().any()
            
            if has_wt1 or has_wt2:
                wt_seq_map = {}
                for pos in range(1, L_seq + 1):
                    wt_id = wt_token_ids[pos]
                    wt_aa = id2aa.get(wt_id, None)
                    if wt_aa is not None:
                        wt_seq_map[pos] = wt_aa
                
                mismatches = []
                
                if has_wt1:
                    for idx, row in dfq.iterrows():
                        if pd.notna(row['wt1']):
                            pos = int(row['pos1'])
                            expected_wt = row['wt1']
                            actual_wt = wt_seq_map.get(pos, '?')
                            if expected_wt != actual_wt:
                                mismatches.append({'row': idx, 'position': pos, 'mut_position': 1, 'expected_wt': expected_wt, 'actual_wt': actual_wt, 'to_aa': row['mut1']})
                
                if has_wt2:
                    for idx, row in dfq.iterrows():
                        if pd.notna(row['wt2']):
                            pos = int(row['pos2'])
                            expected_wt = row['wt2']
                            actual_wt = wt_seq_map.get(pos, '?')
                            if expected_wt != actual_wt:
                                mismatches.append({'row': idx, 'position': pos, 'mut_position': 2, 'expected_wt': expected_wt, 'actual_wt': actual_wt, 'to_aa': row['mut2']})
                
                if mismatches:
                    error_lines = [f"ERROR: Wild-type sequence mismatch in subset_df for PDB {pdb_path} chain {ch}"]
                    for mm in mismatches[:5]:
                        error_lines.append(f"Row {mm['row']}: Pos {mm['position']} expected {mm['expected_wt']} got {mm['actual_wt']}")
                    raise ValueError('\n'.join(error_lines))

            contexts_needed = sorted(set(zip(dfq['pos1'], dfq['mut1'])) | set(zip(dfq['pos2'], dfq['mut2'])))
            singles_needed = contexts_needed.copy()

            ctx_to_partner_js = {}
            cdict = defaultdict(set)
            for p1, t1, p2 in zip(dfq['pos1'], dfq['mut1'], dfq['pos2']):
                cdict[(int(p1), t1)].add(int(p2))
            for p2, t2, p1 in zip(dfq['pos2'], dfq['mut2'], dfq['pos1']):
                cdict[(int(p2), t2)].add(int(p1))
            ctx_to_partner_js = {k: sorted(v) for k, v in cdict.items()}
        else:
            dfq = None
            positions_all = positions if positions is not None else list(range(1, L_seq+1))
            contexts_needed = None
            singles_needed = None
            ctx_to_partner_js = None
            aa_items = list(aa2id.items())

        # -------------------- WT singles logits (reuse RAM cache) --------------------
        pdb_key = (pdb_id, ch)
        wt_per_pos_logits = None
        if reuse_singles != "recompute" and hasattr(self, "_ram_cache") and "singles" in self._ram_cache:
            if pdb_key in self._ram_cache["singles"]:
                cached = self._ram_cache["singles"][pdb_key]
                if len(cached["wt_ids"]) == (L_seq + 2):
                    wt_per_pos_logits = cached["per_pos_logits"].float().cpu()

        if wt_per_pos_logits is None:
            if scheme == "parallel":
                wt_logits_full = self._forward_raw(seq_tokens_wt, coords_wt, struct_tokens_wt)  # (1, L+2, V)
                wt_per_pos_logits = wt_logits_full[0, 1:L_seq+1, :].float().cpu()
            else:
                positions_per_batch = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
                pos_mask = (sorted(set(dfq['pos1']) | set(dfq['pos2'])) if dfq is not None else list(range(1, L_seq+1)))
                rows_logits = {}
                for start in range(0, len(pos_mask), positions_per_batch):
                    chunk = pos_mask[start:start+positions_per_batch]
                    B = len(chunk)
                    seq_batch = seq_tokens_wt.repeat(B, 1).clone()
                    for row_idx, j in enumerate(chunk):
                        seq_batch[row_idx, j] = mask_id
                    coords_batch = coords_wt.repeat(B, 1, 1, 1)
                    struct_batch = struct_tokens_wt.repeat(B, 1)
                    sl = self._forward_raw(seq_batch, coords_batch, struct_batch)  # (B, L+2, V)
                    for row_idx, j in enumerate(chunk):
                        rows_logits[int(j)] = sl[row_idx, j, :].float().cpu()
                Vtmp = next(iter(rows_logits.values())).shape[-1]
                wt_per_pos_logits = torch.full((L_seq, Vtmp), float('nan'), dtype=torch.float32)
                for j, row in rows_logits.items():
                    wt_per_pos_logits[j-1, :] = row

            if hasattr(self, "_ram_cache"):
                self._ram_cache.setdefault("singles", {})
                self._ram_cache["singles"][pdb_key] = {
                    "scheme": scheme,
                    "wt_ids": wt_token_ids,
                    "per_pos_logits": wt_per_pos_logits,
                }

        V = wt_per_pos_logits.shape[-1]

        def _delta_from_logits_row(row_logits: torch.Tensor, wt_id: int, mut_id: int) -> float:
            return float(row_logits[mut_id].item() - row_logits[wt_id].item())
        
        singles_delta = {}
        if dfq is not None:
            for (p, to) in singles_needed:
                wt_id = wt_token_ids[p]
                mut_id = aa2id[to]
                row = wt_per_pos_logits[p-1]
                singles_delta[(p, to)] = _delta_from_logits_row(row, wt_id, mut_id)

        rows = []  # Universal row collection for all schemes
        wt_letters_by_pos = {p: id2aa.get(wt_token_ids[p], 'X') for p in range(1, L_seq+1)}

        # -------------------- "Direct" Strategy Block --------------------
        if scheme == "direct":
            jobs = []
            if dfq is not None:
                for idx, row in dfq.iterrows():
                    p1, t1 = int(row['pos1']), row['mut1']
                    p2, t2 = int(row['pos2']), row['mut2']
                    if p1 == p2: continue
                    jobs.append((p1, t1, p2, t2))
            else:
                for i_idx, p1 in enumerate(positions_all):
                     wt1 = wt_token_ids[p1]
                     for t1, t1_id in aa_items:
                         if t1_id == wt1: continue
                         for p2 in positions_all[i_idx+1:]:
                             wt2 = wt_token_ids[p2]
                             for t2, t2_id in aa_items:
                                 if t2_id == wt2: continue
                                 jobs.append((p1, t1, p2, t2))
            
            if dfq is None:
                for p in positions_all:
                    row = wt_per_pos_logits[p-1]
                    wt_id = wt_token_ids[p]
                    for aa, aa_id in aa_items:
                        if aa_id != wt_id:
                            singles_delta[(p, aa)] = _delta_from_logits_row(row, wt_id, aa_id)

            batch_size = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
            
            for start in tqdm(range(0, len(jobs), batch_size), desc="Direct doubles", disable=quiet):
                chunk = jobs[start:start+batch_size]
                B = len(chunk)
                
                seq_batch = seq_tokens_wt.repeat(B, 1).clone()
                coords_batch = coords_wt.repeat(B, 1, 1, 1)
                struct_batch = struct_tokens_wt.repeat(B, 1)
                
                for b_i, (p1, _, p2, _) in enumerate(chunk):
                    seq_batch[b_i, p1] = mask_id
                    seq_batch[b_i, p2] = mask_id
                    
                logits = self._forward_raw(seq_batch, coords_batch, struct_batch) # (B, L+2, V)
                
                for b_i, (p1, t1, p2, t2) in enumerate(chunk):
                    wt1_id = wt_token_ids[p1]; mut1_id = aa2id[t1]
                    wt2_id = wt_token_ids[p2]; mut2_id = aa2id[t2]
                    
                    d1 = float(logits[b_i, p1, mut1_id] - logits[b_i, p1, wt1_id])
                    d2 = float(logits[b_i, p2, mut2_id] - logits[b_i, p2, wt2_id])
                    direct_score = d1 + d2
                    
                    s1 = singles_delta.get((p1, t1), 0.0)
                    s2 = singles_delta.get((p2, t2), 0.0)
                    
                    wt1_letter = wt_letters_by_pos[p1]
                    wt2_letter = wt_letters_by_pos[p2]

                    rows.append({
                        "pdb": pdb_id, "chain_id": ch,
                        "mut_type": f"{wt1_letter}{p1}{t1}:{wt2_letter}{p2}{t2}",
                        "wt1": wt1_letter, "pos1": p1, "mut1": t1,
                        "wt2": wt2_letter, "pos2": p2, "mut2": t2,
                        "delta_logit1": s1, "delta_logit2": s2,
                        "direct_score": direct_score,  # Stored uncalibrated for bulk processing
                        "scheme": "direct"
                    })

        # -------------------- Parallel/Masked Context Logic --------------------
        else:
            ctx_geom_cache = {}
            def _get_context_geom(i_pos: int, mut_letter: str, exp_seq_toks = None):
                wt_letter = id2aa[seq_tokens_wt[0, i_pos].cpu().item()]
                key = (i_pos, mut_letter)
                if key in ctx_geom_cache:
                    return ctx_geom_cache[key]
                seq_tokens_ctx = seq_tokens_wt
                coords_ctx = coords_wt
                struct_ctx = struct_tokens_wt
                if use_modeled_context_structs and (mut_structs_root is not None):
                    base_code = os.path.basename(pdb_path).split('.')[0]
                    dir_lvl1 = f"{base_code}"
                    dir_lvl2 = "pdb_models"
                    fname = f"{chain}[{wt_letter}{i_pos}{mut_letter}].pdb"
                    pdb_ctx = os.path.join(mut_structs_root, dir_lvl1, dir_lvl2, fname)
                    if os.path.exists(pdb_ctx):
                        seq_tokens_ctx, coords_ctx, plddt_ctx, struct_ctx = self.preprocess_structure(pdb_ctx, chain, dev=dev, backbone_mutation=f'{wt_letter}{i_pos}{mut_letter}', assert_mut=True, mask_ctx=False)
                        if exp_seq_toks is not None:
                            assert torch.equal(seq_tokens_ctx, exp_seq_toks)
                    else:
                        print('Didn\'t find', pdb_ctx)

                ctx_geom_cache[key] = (seq_tokens_ctx, coords_ctx, struct_ctx)
                return seq_tokens_ctx, coords_ctx, struct_ctx

            per_context_partial: dict[tuple[int, str], dict[int, torch.Tensor]] = {}

            if dfq is not None:
                if scheme == "parallel":
                    ctx_batch_cap = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
                    contexts_ordered = [ctx for ctx in sorted(ctx_to_partner_js.keys()) if len(ctx_to_partner_js[ctx]) > 0]

                    for start in tqdm(range(0, len(contexts_ordered), ctx_batch_cap), desc="Context forwards (parallel)", disable=quiet):
                        ctx_chunk = contexts_ordered[start:start+ctx_batch_cap]
                        Bc = len(ctx_chunk)
                        seq_batch = seq_tokens_wt.repeat(Bc, 1)
                        coords_list, struct_list = [], []
                        for row_idx, (i_pos, mut_i) in enumerate(ctx_chunk):
                            seq_batch[row_idx, i_pos] = aa2id[mut_i]
                            sq_ctx, c_ctx, st_ctx = _get_context_geom(i_pos, mut_i, seq_batch[row_idx, :].unsqueeze(0))
                            coords_list.append(c_ctx)
                            struct_list.append(st_ctx)
                        coords_batch = torch.cat(coords_list, dim=0)
                        struct_batch = torch.cat(struct_list, dim=0)
                        sl = self._forward_raw(seq_batch, coords_batch, struct_batch)
                        per_pos_logits = sl[:, 1:L_seq+1, :].float().cpu()

                        for row_idx, (i_pos, mut_i) in enumerate(ctx_chunk):
                            partner_js = ctx_to_partner_js[(i_pos, mut_i)]
                            ctx_map = per_context_partial.setdefault((i_pos, mut_i), {})
                            for j in partner_js:
                                ctx_map[j] = per_pos_logits[row_idx, j-1, :]
                else:
                    row_jobs = []
                    for (i_pos, mut_i), js in ctx_to_partner_js.items():
                        row_jobs.extend([(i_pos, mut_i, j) for j in js])

                    row_batch_cap = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
                    for start in tqdm(range(0, len(row_jobs), row_batch_cap), desc="Context rows (masked)", disable=quiet):
                        chunk = row_jobs[start:start+row_batch_cap]
                        B = len(chunk)
                        seq_batch = seq_tokens_wt.repeat(B, 1)
                        coords_list, struct_list, js_idx = [], [], []

                        for row_idx, (i_pos, mut_i, j) in enumerate(chunk):
                            seq_batch[row_idx, i_pos] = aa2id[mut_i]
                            ctx_check = seq_batch[row_idx, :].clone().unsqueeze(0)
                            if not unmask_mut_ctx_id:
                                seq_batch[row_idx, j] = mask_id
                            sq_ctx, c_ctx, st_ctx = _get_context_geom(i_pos, mut_i, ctx_check)
                            coords_list.append(c_ctx)
                            struct_list.append(st_ctx)
                            js_idx.append(j)

                        coords_batch = torch.cat(coords_list, dim=0)
                        struct_batch = torch.cat(struct_list, dim=0)
                        sl = self._forward_raw(seq_batch, coords_batch, struct_batch)
                        for row_idx, (i_pos, mut_i, j) in enumerate(chunk):
                            ctx_map = per_context_partial.setdefault((i_pos, mut_i), {})
                            ctx_map[j] = sl[row_idx, j, :].float().cpu()

                for k in range(len(dfq)):
                    p1 = int(dfq['pos1'].iloc[k]); t1 = dfq['mut1'].iloc[k]
                    p2 = int(dfq['pos2'].iloc[k]); t2 = dfq['mut2'].iloc[k]
                    if p1 == p2: continue

                    if (p1, t1) not in singles_delta or (p2, t2) not in singles_delta:
                        continue
                    delta_i = singles_delta[(p1, t1)]
                    delta_j = singles_delta[(p2, t2)]

                    row_i = per_context_partial.get((p1, t1), {}).get(p2, None)
                    row_j = per_context_partial.get((p2, t2), {}).get(p1, None)
                    if row_i is None or row_j is None:
                        continue

                    wt_i_id = wt_token_ids[p1]; wt_j_id = wt_token_ids[p2]
                    mut_i_id = aa2id[t1];       mut_j_id = aa2id[t2]
                    dj_i = float(row_i[mut_j_id].item() - row_i[wt_j_id].item())
                    di_j = float(row_j[mut_i_id].item() - row_j[wt_i_id].item())
                    dbl = 0.5 * ((delta_i + dj_i) + (delta_j + di_j))
                    
                    rows.append({
                        "pdb": pdb_id, "chain_id": ch,
                        "mut_type": f"{wt_letters_by_pos[p1]}{p1}{t1}:{wt_letters_by_pos[p2]}{p2}{t2}",
                        "wt1": wt_letters_by_pos[p1], "pos1": p1, "mut1": t1,
                        "wt2": wt_letters_by_pos[p2], "pos2": p2, "mut2": t2,
                        "delta_logit1": float(delta_i),
                        "delta_logit2": float(delta_j),
                        "delta_logit2_given_1": float(dj_i),
                        "delta_logit1_given_2": float(di_j),
                        "double_delta_chainrule_avg": float(dbl),
                        "scheme": scheme,
                        "ctx_geom_1": bool(use_modeled_context_structs and (p1, t1) in ctx_geom_cache and ctx_geom_cache[(p1, t1)][0] is not coords_wt),
                        "ctx_geom_2": bool(use_modeled_context_structs and (p2, t2) in ctx_geom_cache and ctx_geom_cache[(p2, t2)][0] is not coords_wt),
                    })

            else:
                pos_list = positions_all
                for i_pos in pos_list:
                    wt_i_id = wt_token_ids[i_pos]
                    row = wt_per_pos_logits[i_pos-1]
                    for mut_i, mut_i_id in aa_items:
                        if mut_i_id == wt_i_id: continue
                        singles_delta[(i_pos, mut_i)] = _delta_from_logits_row(row, wt_i_id, mut_i_id)

                if scheme == "parallel":
                    ctx_jobs = []
                    for i_pos in pos_list:
                        wt_i_id = wt_token_ids[i_pos]
                        for mut_i, mut_i_id in aa_items:
                            if mut_i_id == wt_i_id: continue
                            ctx_jobs.append((i_pos, mut_i))
                    ctx_batch_cap = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
                    for start in tqdm(range(0, len(ctx_jobs), ctx_batch_cap), desc="All contexts (parallel)", disable=quiet):
                        chunk = ctx_jobs[start:start+ctx_batch_cap]
                        Bc = len(chunk)
                        seq_batch = seq_tokens_wt.repeat(Bc, 1)
                        coords_list, struct_list = [], []
                        for row_idx, (i_pos, mut_i) in enumerate(chunk):
                            seq_batch[row_idx, i_pos] = aa2id[mut_i]
                            sq_ctx, c_ctx, st_ctx = _get_context_geom(i_pos, mut_i, seq_batch[row_idx, :].unsqueeze(0))
                            coords_list.append(c_ctx); struct_list.append(st_ctx)
                        coords_batch = torch.cat(coords_list, dim=0)
                        struct_batch = torch.cat(struct_list, dim=0)
                        sl = self._forward_raw(seq_batch, coords_batch, struct_batch)
                        per_pos_logits = sl[:, 1:L_seq+1, :].float().cpu()
                        for row_idx, (i_pos, mut_i) in enumerate(chunk):
                            m = per_context_partial.setdefault((i_pos, mut_i), {})
                            for j in pos_list:
                                if j == i_pos: continue
                                m[j] = per_pos_logits[row_idx, j-1, :]
                else:
                    row_jobs = []
                    for i_pos in pos_list:
                        wt_i_id = wt_token_ids[i_pos]
                        for mut_i, mut_i_id in aa_items:
                            if mut_i_id == wt_i_id: continue
                            for j in pos_list:
                                if j == i_pos: continue
                                row_jobs.append((i_pos, mut_i, j))
                    row_batch_cap = max(1, int(round(mem_scale * (400**3) / (L_seq**3))))
                    for start in tqdm(range(0, len(row_jobs), row_batch_cap), desc="All context rows (masked)", disable=quiet):
                        chunk = row_jobs[start:start+row_batch_cap]
                        B = len(chunk)
                        seq_batch = seq_tokens_wt.repeat(B, 1)
                        coords_list, struct_list, js_idx = [], [], []
                        for row_idx, (i_pos, mut_i, j) in enumerate(chunk):
                            seq_batch[row_idx, i_pos] = aa2id[mut_i]
                            ctx_check = seq_batch[row_idx, :].clone().unsqueeze(0)
                            seq_batch[row_idx, j] = mask_id
                            sq_ctx, c_ctx, st_ctx = _get_context_geom(i_pos, mut_i, ctx_check)
                            coords_list.append(c_ctx); struct_list.append(st_ctx); js_idx.append((i_pos, mut_i, j))
                        coords_batch = torch.cat(coords_list, dim=0)
                        struct_batch = torch.cat(struct_list, dim=0)
                        sl = self._forward_raw(seq_batch, coords_batch, struct_batch)
                        for row_idx, (i_pos, mut_i, j) in enumerate(chunk):
                            m = per_context_partial.setdefault((i_pos, mut_i), {})
                            m[j] = sl[row_idx, j, :].float().cpu()

                for idx_i, i_pos in enumerate(pos_list):
                    wt_i_id = wt_token_ids[i_pos]
                    wt_i = id2aa.get(wt_i_id, 'X')
                    if wt_i not in aa2id: continue
                    for j_pos in pos_list[idx_i+1:]:
                        wt_j_id = wt_token_ids[j_pos]
                        wt_j = id2aa.get(wt_j_id, 'X')
                        if wt_j not in aa2id: continue

                        for mut_i, mut_i_id in aa_items:
                            if mut_i_id == wt_i_id: continue
                            for mut_j, mut_j_id in aa_items:
                                if mut_j_id == wt_j_id: continue

                                delta_i = singles_delta[(i_pos, mut_i)]
                                delta_j = singles_delta[(j_pos, mut_j)]

                                row_i = per_context_partial.get((i_pos, mut_i), {}).get(j_pos, None)
                                row_j = per_context_partial.get((j_pos, mut_j), {}).get(i_pos, None)
                                if row_i is None or row_j is None:
                                    continue

                                dj_i = float(row_i[mut_j_id].item() - row_i[wt_j_id].item())
                                di_j = float(row_j[mut_i_id].item() - row_j[wt_i_id].item())
                                dbl = 0.5 * ((delta_i + dj_i) + (delta_j + di_j))

                                rows.append({
                                    "pdb": pdb_id, "chain_id": ch,
                                    "mut_type": f"{wt_i}{i_pos}{mut_i}:{wt_j}{j_pos}{mut_j}",
                                    "wt1": wt_i, "pos1": i_pos, "mut1": mut_i,
                                    "wt2": wt_j, "pos2": j_pos, "mut2": mut_j,
                                    "delta_logit1": float(delta_i),
                                    "delta_logit2": float(delta_j),
                                    "delta_logit2_given_1": float(dj_i),
                                    "delta_logit1_given_2": float(di_j),
                                    "double_delta_chainrule_avg": float(dbl),
                                    "scheme": scheme,
                                    "ctx_geom_1": bool(use_modeled_context_structs and (i_pos, mut_i) in ctx_geom_cache and ctx_geom_cache[(i_pos, mut_i)][0] is not coords_wt),
                                    "ctx_geom_2": bool(use_modeled_context_structs and (j_pos, mut_j) in ctx_geom_cache and ctx_geom_cache[(j_pos, mut_j)][0] is not coords_wt),
                                })

        # -------------------- Unified Calibration Application --------------------
        df = pd.DataFrame(rows)
        
        # Shared setup: singles are present in all schemes
        uncal_mut1_delta = torch.tensor(df['delta_logit1'].to_numpy(copy=False)).to(dev)
        uncal_mut2_delta = torch.tensor(df['delta_logit2'].to_numpy(copy=False)).to(dev)
        
        cal_mut1_delta = self.calibration_head_shared(uncal_mut1_delta).cpu().numpy()
        cal_mut2_delta = self.calibration_head_shared(uncal_mut2_delta).cpu().numpy()

        if scheme == "direct":
            uncal_direct = torch.tensor(df['direct_score'].to_numpy(copy=False)).to(dev)
            cal_direct = self.calibration_head_shared(uncal_direct).cpu().numpy()
            
            df['ddg_pred'] = cal_direct
            df['ddg_pred_additive'] = cal_mut1_delta + cal_mut2_delta
            
            # Clean up the uncalibrated direct score column
            df = df.drop(columns=['direct_score'])
        else:
            uncal_mut1_2_delta = torch.tensor(df['delta_logit1_given_2'].to_numpy(copy=False)).to(dev)
            uncal_mut2_1_delta = torch.tensor(df['delta_logit2_given_1'].to_numpy(copy=False)).to(dev)
            
            cal_mut1_2_delta = self.calibration_head_shared(uncal_mut1_2_delta).cpu().numpy()
            cal_mut2_1_delta = self.calibration_head_shared(uncal_mut2_1_delta).cpu().numpy()
            
            df['cal_delta1'] = cal_mut1_delta
            df['cal_delta2'] = cal_mut2_delta
            df['cal_delta1_given_2'] = cal_mut1_2_delta
            df['cal_delta2_given_1'] = cal_mut2_1_delta
            
            df['ddg_pred_additive'] = cal_mut1_delta + cal_mut2_delta
            df['ddg_pred'] = 0.5 * ((cal_mut1_delta + cal_mut2_1_delta) + (cal_mut2_delta + cal_mut1_2_delta))

        df['dddg_pred'] = df['ddg_pred'] - df['ddg_pred_additive']
        
        return df

    @torch.no_grad()
    def infer_multimutants_sampled(
        self,
        pdb_path: str,
        *,
        subset_df,                          
        chain: str | None = "A",
        strategy: str = "masked",         
        device: str | torch.device | None = None,
        K_paths: int = 4,                   
        apply_calibration: bool = True,     
        return_path_summaries: bool = False,
        batch_size: int = 64               
    ):
        """
        Estimate Δ for multi-mutants by sampling K random masked single-mutation paths,
        or by direct simultaneous masking. Computes additive baseline first on
        unique mutations, and uses a flattened batch queue for accurate ETAs.
        """

        dev = torch.device(device) if device is not None else next(self.parameters()).device
        scheme = strategy.lower()
        if scheme not in ("masked", "direct"):
            raise ValueError("strategy must be one of {'masked','direct'}")

        def _canonical_mut_string(muts):
            return ':'.join([f"{fr}{pos}{to}" for (fr,pos,to) in muts])

        # ---------- 1) Load WT structure/seq ----------
        protein_chain = ProteinChain.from_pdb(pdb_path, chain, is_predicted=False)
        seq_wt = protein_chain.sequence
        L_seq = len(seq_wt)

        coords_wt, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        coords_wt = coords_wt.to(dev); residue_index = residue_index.to(dev)
        _, struct_tokens_wt = self.structure_encoder.encode(coords_wt, residue_index=residue_index)

        coords_wt = F.pad(coords_wt, (0,0,0,0,1,1), value=torch.inf)
        struct_tokens_wt = F.pad(struct_tokens_wt, (1,1), value=0)
        if struct_tokens_wt.shape[1] > 0:
            struct_tokens_wt[:, 0] = C.STRUCTURE_BOS_TOKEN
            struct_tokens_wt[:, -1] = C.STRUCTURE_EOS_TOKEN

        seq_tokens_wt = torch.tensor(
            np.array(self.sequence_tokenizer.encode(seq_wt), dtype=np.int64)
        ).unsqueeze(0).to(dev) 

        canonical = self.valid_canonical_aas
        can_ids = self.canonical_aa_token_ids
        aa2id = {a:i for a,i in zip(canonical, can_ids)}
        mask_id = C.SEQUENCE_MASK_TOKEN
        pdb_id = os.path.basename(pdb_path)
        ch = chain or ""

        # ---------- 2) Parse subset_df into per-row mutation lists ----------
        muts_per_row = []
        max_slots = 10
        cols = subset_df.columns

        for idx in range(len(subset_df)):
            muts = []
            for k in range(1, max_slots+1):
                fr_col, pos_col, to_col = f"wt{k}", f"pos{k}", f"mut{k}"
                if (fr_col in cols) and (pos_col in cols) and (to_col in cols):
                    fr = subset_df.iloc[idx][fr_col]
                    pos = subset_df.iloc[idx][pos_col]
                    to  = subset_df.iloc[idx][to_col]
                    if (isinstance(fr, str) and fr) and (pd.notna(pos)) and (isinstance(to, str) and to):
                        pos = int(pos)
                        muts.append((fr, pos, to))
                else:
                    break
            
            muts = [(fr,pos,to) for (fr,pos,to) in muts if 1 <= pos <= L_seq and fr in aa2id and to in aa2id and fr != to]
            seen = set()
            valid = True
            for fr, pos, to in muts:
                if pos in seen: valid = False; break
                seen.add(pos)
                
            if not valid or len(muts) == 0:
                muts_per_row.append([])
            else:
                muts_per_row.append(muts)

        valid_row_indices = [i for i, muts in enumerate(muts_per_row) if len(muts) > 0]
        if not valid_row_indices:
            raise AssertionError('No valid indices.')

        # ---------- 3) Compute Additive Baseline (Unique Singles Only) ----------
        unique_single_muts = set()
        for i in valid_row_indices:
            for fr, pos, to in muts_per_row[i]:
                wt_id = int(seq_tokens_wt[0, pos].item())
                mut_id = aa2id[to]
                unique_single_muts.add((pos, wt_id, mut_id))

        unique_single_muts = list(unique_single_muts)
        num_unique = len(unique_single_muts)
        single_mut_deltas = {}

        if num_unique > 0:
            additive_seqs = seq_tokens_wt.expand(num_unique, -1).clone()
            pos_tensor = torch.tensor([m[0] for m in unique_single_muts])
            additive_seqs[torch.arange(num_unique), pos_tensor] = mask_id
            
            deltas = []
            for i in tqdm(range(0, num_unique, batch_size), desc=f"Additive Baseline (N={num_unique} unique)"):
                chunk = additive_seqs[i:i+batch_size]
                b_sz = chunk.size(0)
                c_chunk = coords_wt.expand(b_sz, *coords_wt.shape[1:])
                s_chunk = struct_tokens_wt.expand(b_sz, *struct_tokens_wt.shape[1:])
                
                logits = self._forward_raw(chunk, c_chunk, s_chunk)
                
                chunk_pos = pos_tensor[i:i+batch_size]
                chunk_wt = torch.tensor([m[1] for m in unique_single_muts[i:i+batch_size]])
                chunk_mut = torch.tensor([m[2] for m in unique_single_muts[i:i+batch_size]])
                
                b_idx = torch.arange(b_sz)
                chunk_deltas = (logits[b_idx, chunk_pos, chunk_mut] - logits[b_idx, chunk_pos, chunk_wt]).float()
                
                if apply_calibration:
                    chunk_deltas = self.calibration_head_shared(chunk_deltas.unsqueeze(-1)).squeeze(-1)
                deltas.extend(chunk_deltas.cpu().tolist())

            for (pos, wt_id, mut_id), delta in zip(unique_single_muts, deltas):
                single_mut_deltas[(pos, wt_id, mut_id)] = delta

        additive_sums = {i: 0.0 for i in valid_row_indices}
        for i in valid_row_indices:
            for fr, pos, to in muts_per_row[i]:
                wt_id = int(seq_tokens_wt[0, pos].item())
                mut_id = aa2id[to]
                additive_sums[i] += single_mut_deltas.get((pos, wt_id, mut_id), 0.0)

        # ---------- 4) Epistatic / Strategy Processing (Batched, Lazy Eval) ----------
        direct_scores = {}
        path_results = {i: [] for i in valid_row_indices}

        if scheme == "direct":
            num_direct = len(valid_row_indices)
            for i in tqdm(range(0, num_direct, batch_size), desc="Direct Inference"):
                chunk_indices = valid_row_indices[i:i+batch_size]
                b_sz = len(chunk_indices)
                
                chunk_seqs = seq_tokens_wt.expand(b_sz, -1).clone()
                for j, row_i in enumerate(chunk_indices):
                    for _, pos, _ in muts_per_row[row_i]:
                        chunk_seqs[j, pos] = mask_id
                
                c_chunk = coords_wt.expand(b_sz, *coords_wt.shape[1:])
                s_chunk = struct_tokens_wt.expand(b_sz, *struct_tokens_wt.shape[1:])
                logits = self._forward_raw(chunk_seqs, c_chunk, s_chunk)
                
                chunk_scores = []
                for j, row_i in enumerate(chunk_indices):
                    score = 0.0
                    for _, pos, to in muts_per_row[row_i]:
                        wt_id = int(seq_tokens_wt[0, pos].item())
                        mut_id = aa2id[to]
                        score += float(logits[j, pos, mut_id] - logits[j, pos, wt_id])
                    chunk_scores.append(score)
                    
                chunk_scores_t = torch.tensor(chunk_scores, device=dev, dtype=torch.float32)
                if apply_calibration:
                    try:
                        chunk_scores_t = self.calibration_head_shared(chunk_scores_t.unsqueeze(-1)).squeeze(-1)
                    except AttributeError:
                        pass
                        
                for j, score in enumerate(chunk_scores_t.cpu().tolist()):
                    row_i = chunk_indices[j]
                    direct_scores[row_i] = score

        else:
            # Masked Strategy via flat dynamic queue
            path_states = deque()
            total_evals = 0
            
            for row_i in valid_row_indices:
                muts = muts_per_row[row_i]
                N = len(muts)
                total_evals += N * K_paths
                for k in range(K_paths):
                    order = list(range(N))
                    if N > 1: random.shuffle(order)
                    path_states.append({
                        "row_i": row_i, 
                        "seq": seq_tokens_wt[0].clone(),
                        "order": order, 
                        "cum_delta": 0.0, 
                        "step": 0, 
                        "N": N
                    })
                    
            total_batches = math.ceil(total_evals / batch_size)
            
            with tqdm(total=total_batches, desc="Masked Path Sampling") as pbar:
                while path_states:
                    chunk_paths = []
                    while path_states and len(chunk_paths) < batch_size:
                        chunk_paths.append(path_states.popleft())
                        
                    b_sz = len(chunk_paths)
                    
                    # Stack sequences for this specific chunk
                    chunk_seqs = torch.stack([p["seq"] for p in chunk_paths])
                    pos_list, wt_list, mut_list = [], [], []
                    
                    for j, p in enumerate(chunk_paths):
                        row_i = p["row_i"]
                        mut_idx = p["order"][p["step"]]
                        _, pos, to = muts_per_row[row_i][mut_idx]
                        
                        chunk_seqs[j, pos] = mask_id
                            
                        pos_list.append(pos)
                        wt_list.append(int(p["seq"][pos].item()))
                        mut_list.append(aa2id[to])
                    
                    c_chunk = coords_wt.expand(b_sz, *coords_wt.shape[1:])
                    s_chunk = struct_tokens_wt.expand(b_sz, *struct_tokens_wt.shape[1:])
                    logits = self._forward_raw(chunk_seqs, c_chunk, s_chunk)
                    
                    chunk_pos = torch.tensor(pos_list, device=dev)
                    chunk_wt = torch.tensor(wt_list, device=dev)
                    chunk_mut = torch.tensor(mut_list, device=dev)
                    b_idx = torch.arange(b_sz, device=dev)
                    
                    chunk_deltas = (logits[b_idx, chunk_pos, chunk_mut] - logits[b_idx, chunk_pos, chunk_wt]).float()
                    
                    if apply_calibration:
                        try:
                            calibrated = self.calibration_head_single(chunk_deltas.unsqueeze(-1)).squeeze(-1)
                        except AttributeError:
                            calibrated = self.calibration_head_shared(chunk_deltas.unsqueeze(-1)).squeeze(-1)
                        deltas = calibrated.cpu().tolist()
                    else:
                        deltas = chunk_deltas.cpu().tolist()
                        
                    for j, p in enumerate(chunk_paths):
                        p["cum_delta"] += deltas[j]
                        row_i = p["row_i"]
                        mut_idx = p["order"][p["step"]]
                        _, pos, to = muts_per_row[row_i][mut_idx]
                        
                        # Apply the mutation for the next step
                        p["seq"][pos] = aa2id[to] 
                        p["step"] += 1
                        
                        # Route the state back to the queue or into results
                        if p["step"] < p["N"]:
                            path_states.append(p)
                        else:
                            path_results[p["row_i"]].append(p["cum_delta"])
                            
                    pbar.update(1)

        # ---------- 5) Construct Output DataFrame ----------
        out_rows = []
        for row_i in valid_row_indices:
            muts = muts_per_row[row_i]
            rec = {
                "pdb": pdb_id,
                "chain_id": ch,
                "mut_type": _canonical_mut_string(muts),
                "n_muts": len(muts),
                "K_paths": int(K_paths),
                "ddg_pred_additive": float(additive_sums[row_i]),
            }
            
            if scheme == "direct":
                rec["ddg_pred"] = direct_scores[row_i]
            else:
                sums = path_results[row_i]
                rec["ddg_pred"] = float(np.mean(sums))
                rec[f"ddg_pred_std_{K_paths}"] = float(np.std(sums, ddof=1)) if len(sums) > 1 else 0.0
                if return_path_summaries:
                    rec["path_sums"] = sums
                    
            out_rows.append(rec)

        df = pd.DataFrame(out_rows)
        df['dddg_pred'] = df['ddg_pred'] - df['ddg_pred_additive']
        return df
    
    @torch.no_grad()
    def infer_full_seq(
        self,
        pdb_path: str,
        subset_df: pd.DataFrame,
        *,
        chain: str = "A",
        device: str | torch.device | None = None,
        backbone_mutation: str | None = None, # e.g., "V59K" or "V59-"
        apply_calibration: bool = True,
        quiet: bool = False
    ):
        """
        Unified inference for substitutions and indels using the Full Sequence paradigm.
        
        Logic:
        1. Establishes a 'Baseline' (Parent) by applying backbone_mutation to the PDB WT.
        2. For each row in subset_df, applies the mutations relative to that Baseline.
        3. Scores every variant as Delta Mean LL (Mutant vs Baseline).
        """
        dev = torch.device(device) if device is not None else next(self.parameters()).device
        pdb_id = os.path.basename(pdb_path)

        # 1. --- BASELINE PREPROCESSING (The "Template") ---
        # Load the absolute WT from PDB
        protein_chain = ProteinChain.from_pdb(pdb_path, chain, is_predicted=False)
        raw_wt_seq = protein_chain.sequence
        coords_ref, _, ri_ref = protein_chain.to_structure_encoder_inputs()
        
        # Standardize Reference Tensors (L+2 for BOS/EOS)
        coords_ref = F.pad(coords_ref.to(dev), (0, 0, 0, 0, 1, 1), value=float("inf"))
        _, struct_tokens_ref = self.structure_encoder.encode(coords_ref[:, 1:-1], residue_index=ri_ref.to(dev))
        struct_ref = F.pad(struct_tokens_ref, (1, 1), value=C.STRUCTURE_PAD_TOKEN)
        struct_ref[:, 0] = C.STRUCTURE_BOS_TOKEN
        struct_ref[:, -1] = C.STRUCTURE_EOS_TOKEN
        
        # Pad Residue Index for RoPE (BOS/EOS protection)
        ri_ref = ri_ref.to(dev)
        ri_ref = torch.cat([(ri_ref[0]-1).unsqueeze(0), ri_ref, (ri_ref[-1]+1).unsqueeze(0)], dim=0)

        # 2. --- ESTABLISH BACKBONE (The "Parent") ---
        # If a backbone mutation exists, the Baseline for the whole DF shifts.
        if backbone_mutation:
            # Internal helper parses "V59K" or "V59-"
            b_muts = self._parse_mutation_string(backbone_mutation) 
            b_seq, b_tokens, b_coords, b_struct, b_ri, _ = self._apply_surgery(
                raw_wt_seq, coords_ref, struct_ref, ri_ref, b_muts
            )
            b_tokens = b_tokens.to(dev)
        else:
            b_seq = raw_wt_seq
            b_tokens = torch.tensor(self.sequence_tokenizer.encode(b_seq)).to(dev)
            b_coords, b_struct, b_ri = coords_ref, struct_ref, ri_ref

        # Pre-compute Baseline Mean LL (The "Subtrahend")
        # This is done once per protein to save massive compute time
        batch_baseline = {
            'sequence_tokens': b_tokens.unsqueeze(0),
            'coords': b_coords,
            'structure_tokens': b_struct
        }
        baseline_mll = self._compute_mean_ll(
            self._get_esm3_outputs(
                sequence_tokens=batch_baseline['sequence_tokens'],
                structure_coords=batch_baseline['coords'],
                structure_tokens=batch_baseline['structure_tokens']
            ).sequence_logits,
            batch_baseline['sequence_tokens'],
            ignore_index=C.SEQUENCE_PAD_TOKEN
        ).item()

        # 3. --- VARIANT INFERENCE LOOP ---
        rows = []
        desc = f"Inference ({pdb_id})"
        for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=desc, disable=quiet):
            # Parse the mutation string (handles "A10C:B20-" etc.)
            mut_type_str = row.get('mut_type', row.get('uid', ''))
            muts = self._parse_mutation_string(mut_type_str)
            
            # Apply Surgery to the Baseline to get the Mutant
            m_seq, m_tokens, m_coords, m_struct, m_ri, is_ind = self._apply_surgery(
                b_seq, b_coords, b_struct, b_ri, muts
            )
            
            # Calculate Mutant Mean LL
            mut_logits = self._get_esm3_outputs(
                sequence_tokens=m_tokens.unsqueeze(0).to(dev),
                structure_coords=m_coords,
                structure_tokens=m_struct
            ).sequence_logits
            
            mut_mll = self._compute_mean_ll(
                mut_logits, m_tokens.unsqueeze(0).to(dev), ignore_index=C.SEQUENCE_PAD_TOKEN
            ).item()
            
            delta_ll = mut_mll - baseline_mll
            
            # Calibration
            ddg_pred = delta_ll
            if apply_calibration:
                ddg_pred = self.calibration_head_shared(torch.tensor([delta_ll]).to(dev)).item()

            rows.append({
                "uid": row.get('uid', mut_type_str),
                "mut_type": mut_type_str,
                "delta_ll": delta_ll,
                "ddg_pred": ddg_pred,
                "is_indel": is_ind,
                "seq_len": len(m_seq)
            })

        return pd.DataFrame(rows)

    def _apply_surgery(self, base_seq, base_coords, base_struct, base_ri, mutations):
        """
        The engine for Indels. Synchronizes sequence, coordinates, structure tokens, 
        and residue indices (RoPE).
        """
        seq_list = list(base_seq)
        coords, struct, ri = base_coords.clone(), base_struct.clone(), base_ri.clone()
        is_ind = False

        # Reverse order prevents index-shifting collisions
        for wt, pos, mt in sorted(mutations, key=lambda x: x[1], reverse=True):
            # Index pos (1-based) maps to Tensor index pos (due to BOS at 0)
            if mt == '-': # DELETION
                is_ind = True
                if pos <= len(seq_list):
                    seq_list.pop(pos-1)
                    coords = torch.cat([coords[:, :pos], coords[:, pos+1:]], dim=1)
                    struct = torch.cat([struct[:, :pos], struct[:, pos+1:]], dim=1)
                    ri = torch.cat([ri[:pos], ri[pos+1:]], dim=0)
            
            elif wt == '-': # INSERTION
                is_ind = True
                for aa in reversed(mt):
                    seq_list.insert(pos, aa)
                    # Add 'missing' coordinate slot
                    ins_c = torch.full((1, 1, 37, 3), float("inf"), device=coords.device)
                    coords = torch.cat([coords[:, :pos+1], ins_c, coords[:, pos+1:]], dim=1)
                    # Add structure mask
                    ins_s = torch.tensor([[C.STRUCTURE_MASK_TOKEN]], device=struct.device)
                    struct = torch.cat([struct[:, :pos+1], ins_s, struct[:, pos+1:]], dim=1)
                    # Duplicate Residue Index to prevent RoPE phase-shift
                    ins_ri = ri[pos:pos+1]
                    ri = torch.cat([ri[:pos+1], ins_ri, ri[pos+1:]], dim=0)
            
            else: # SUBSTITUTION
                if pos <= len(seq_list):
                    seq_list[pos-1] = mt
                    #if self.enable_mutctx_masking:
                    #    struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                    #    coords[:, pos, :, :] = np.nan

        final_seq = "".join(seq_list)
        final_tokens = self.sequence_tokenizer.encode(final_seq)
        
        # Length-locking safety check for ESM3 (prevents 68 vs 67 crashes)
        if len(final_tokens) != struct.shape[1]:
            # Handle tokenizer quirks (like unexpected special tokens)
            target_L = len(final_tokens)
            struct = struct[:, :target_L]
            coords = coords[:, :target_L]
            ri = ri[:target_L]

        return final_seq, torch.tensor(final_tokens), coords, struct, ri, is_ind

    def _parse_mutation_string(self, mut_str: str):
        """Parses A10C:B20-: -30GY into structured list."""
        muts = []
        for m in str(mut_str).split(':'):
            m = m.strip()
            if m.endswith('-'): # Del
                muts.append((m[0], int(m[1:-1]), '-'))
            elif m.startswith('-'): # Ins
                pos = int(re.search(r'\d+', m).group())
                res = re.sub(r'[-\d]+', '', m)
                muts.append(('-', pos, res))
            elif re.search(r'[A-Z]\d+[A-Z]', m): # Sub
                muts.append((m[0], int(m[1:-1]), m[-1]))
        return muts