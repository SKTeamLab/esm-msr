import logging
import time
import copy
from collections import defaultdict
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.constants import esm3 as C
from peft import LoraConfig, get_peft_model

if TYPE_CHECKING:  # the annotation on score_screening_batch is a string, but it should still resolve
    from esm_msr.auto_batch import AutoBatchSizer


class ESM3PredictorBase(nn.Module):
    """Base class for stability prediction models using ESM3."""
    def __init__(self, esm_model: nn.Module):
        super().__init__()
        self.model = esm_model 

        if hasattr(esm_model, 'tokenizers') and hasattr(esm_model.tokenizers, 'sequence'):
             self.sequence_tokenizer = self.model.tokenizers.sequence
        else:
             raise AttributeError("Could not find sequence tokenizer in the provided ESM model.")
        
        try:
             self.structure_encoder = self.model.get_structure_encoder()
        except AttributeError:
             raise AttributeError("Could not find structure encoder for the provided ESM model.")

        self.vocab = self.sequence_tokenizer.get_vocab()
        self.valid_canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.canonical_aa_token_ids = [self.vocab.get(wt_aa) for wt_aa in self.valid_canonical_aas]
        
        if None in self.canonical_aa_token_ids:
            raise AssertionError("Failed to map some canonical amino acids to tokenizer vocabulary.")
            
        self.register_buffer('canonical_idx_tensor', torch.tensor(self.canonical_aa_token_ids, dtype=torch.long))

    def _get_esm3_outputs(self, sequence_tokens: torch.Tensor, structure_coords: Optional[torch.Tensor] = None, structure_tokens: Optional[torch.Tensor] = None, per_res_plddt: Optional[torch.Tensor] = None, active_model: Optional[nn.Module] = None):
        """Internal helper to run the underlying ESM model's forward pass."""
        def _prepare_input(tensor, expected_dims):
            if tensor is None or not torch.is_tensor(tensor): return None
            if tensor.dim() == expected_dims - 1:
                 tensor = tensor.unsqueeze(0)
            while tensor.dim() > expected_dims and tensor.shape[1] == 1:
                 tensor = tensor.squeeze(1)
            if tensor.dim() != expected_dims:
                 logging.warning(f"Input tensor shape {tensor.shape} doesn't match expected dims {expected_dims} after preparation.")
            return tensor

        target_model = active_model if active_model is not None else self.model
        return target_model.model(
            sequence_tokens=_prepare_input(sequence_tokens, 2),
            structure_coords=_prepare_input(structure_coords, 4),
            structure_tokens=_prepare_input(structure_tokens, 2),
        )


class _ScaleSource:
    """Plain (non-Module) holder for a shared calibration scale.

    ``mt.scale = wt.scale`` cannot be expressed by assigning the WT head to an attribute of the MT head: nn.Module
    would register it as a submodule, duplicating it in the state dict and in ``parameters()``. This holder is a plain
    object, so the reference stays a Python pointer and ``CalibrationHead.scale`` reads the WT head's live scale.
    """
    def __init__(self, head):
        self.head = head

    @property
    def scale(self):
        return self.head.scale


MASK_STRATEGIES = ('marginal', 'independent')


def pass_mask_strategy(hp, pass_type: str) -> Optional[str]:
    """
    The sequence-masking strategy of one pass, ``--mask_strategy_<pass_type>`` ('none' or unset = unmasked). ``hp`` is anything with
    ``.get``. The motivating pairing is ``marginal`` on the WT pass and ``independent`` on the MT pass: the WT pass then masks both sites
    of a double (it knows WHICH pair is mutated but not to what: pair and line effects, by construction no interaction) and the MT pass
    reads each site masked with the partner's mutant residue visible, the conditional masked marginal of ddG(A|B). The same strategy on
    both passes cannot do this: ``marginal`` on the MT pass hides the partner, so the model cannot represent an interaction at all
    (``cell_rank`` is exactly 0).
    """
    v = hp.get(f'mask_strategy_{pass_type}', None)
    return None if v in (None, 'none') else v


def _per_pass(mask_strategy, pass_type: str) -> Optional[str]:
    """``mask_strategy`` as ``forward_batch`` / ``forward_cycle`` take it: one strategy for both passes, or a dict {'wt': ..., 'mt': ...}."""
    return mask_strategy.get(pass_type) if isinstance(mask_strategy, dict) else mask_strategy


class CalibrationHead(nn.Module):
    """
    Scales and biases raw log-likelihood ratios: ``y_cal = scale * y_raw + bias``.

    With ``per_mutation``, the bias is ``bias * n_mut`` instead: one bias per mutated position rather than one per
    item. The LLR sum the head calibrates is itself a sum over the item's positions, so a constant offset per
    position is the form that keeps a double's calibration consistent with its two singles - a single scalar bias
    makes a double's prediction sit one bias BELOW the sum of its singles (measured at 0.1065 kcal/mol on
    devel-revision: L_AB - L_A - L_B = -b).
    """
    def __init__(self, init_scale: float | None = 1/3, init_bias: float | None = 0.0, *, min_scale: float = 1e-4, beta: float = 1.0, max_scale: float | None = None, requires_grad: bool = True, per_mutation: bool = False):
        super().__init__()
        self.use_scale = True
        self.scale_ref = None          # set by MSRModel when --tie_mt_scale: a _ScaleSource on the WT head
        if init_scale is None:
            self.use_scale = False
            init_scale = 1.0
        else:
            init_scale = float(init_scale)
            
        target = max(init_scale - min_scale, 1e-12)
        raw_init = self._inv_softplus(torch.tensor(target, dtype=torch.float32), beta=beta)
        self.raw_scale = nn.Parameter(raw_init, requires_grad=requires_grad)

        if init_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32), requires_grad=requires_grad)

        self.min_scale, self.beta, self.max_scale = float(min_scale), float(beta), float(max_scale) if max_scale is not None else None
        self.per_mutation = bool(per_mutation)

    @staticmethod
    def _inv_softplus(y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        by = beta * y
        out = torch.empty_like(y)
        large = by > 20.0
        out[large] = by[large]                               
        out[~large] = torch.log(torch.expm1(by[~large]))
        return out / beta

    @property
    def scale(self) -> torch.Tensor:
        if self.scale_ref is not None:
            return self.scale_ref.scale
        s = F.softplus(self.raw_scale, beta=self.beta) + self.min_scale
        return torch.clamp(s, max=self.max_scale) if self.max_scale is not None else s

    def bias_for(self, n_mut: torch.Tensor | None = None):
        """
        The bias this head adds to an item, which is ``bias * n_mut`` for a per-mutation head.

        Anything that INVERTS the calibration has to go through this too - training's teacher-forcing label is
        de-calibrated as ``(label - bias_for(n_mut)) / scale`` - which is why it is a method and not inlined.
        """
        if self.bias is None:
            return 0.0
        if not self.per_mutation:
            return self.bias
        if n_mut is None:
            raise AssertionError("per_mutation calibration needs n_mut (the item's mutated-position count); "
                                 "calling it without one would silently apply a single mutation's bias to a double.")
        return self.bias * n_mut.to(self.bias.dtype)

    def forward(self, y_raw: torch.Tensor, n_mut: torch.Tensor | None = None) -> torch.Tensor:
        s = 1.0 if not self.use_scale else self.scale
        return y_raw * s + self.bias_for(n_mut)
    

def native_cond_mt_batch(batch: dict) -> dict:
    """Return ``batch`` with the MT pass's sequence replaced by the WT pass's on native_cond rows.

    A mutant-background library's ``mt_sequence_tokens`` carries the background mutation AND the scored mutation, so
    it is two steps from the true wild type. The MT head's task on those items is the conditional effect from the
    LIBRARY'S starting sequence, so it must see the WT pass's input (the background), not the fully mutated one; the
    remaining sequence/structure mismatch is exactly the MT head's domain. Rows of any other subset, and the input
    dict, are untouched.
    """
    st = batch.get('subset_type')
    mt_seq, wt_seq = batch.get('mt_sequence_tokens'), batch.get('wt_sequence_tokens')
    if st is None or mt_seq is None or wt_seq is None:
        return batch
    nc = torch.as_tensor([t == 'native_cond' for t in st], device=mt_seq.device)
    if not bool(nc.any()):
        return batch
    out = dict(batch)
    seq = mt_seq.clone()
    seq[nc] = wt_seq[nc]
    out['mt_sequence_tokens'] = seq
    return out


def combine_predictions(wt, mt):
    """
    How the WT (additive) and MT (mutant-context) heads become the reported prediction: ``0.5 * wt + 0.5 * mt``.

    Trained against the measured ddG, this makes MT learn ``2 * ddG - wt``: each head produces a TOTAL ddG-like prediction, which
    is what the per-position LLR sum it computes actually is, and the ensemble averages the two. (Two other modes existed and are
    gone: ``sum`` asked MT for ``ddG - wt``, a pure difference the LLR sum cannot produce; ``route`` reported WT for singles and MT
    for the rest, and never ran.) Every site that forms a combined prediction - training, validation and screening - goes through
    here, or an ablation trains one way and is scored another.
    """
    return 0.5 * wt + 0.5 * mt


class MSRModel(ESM3PredictorBase):
    """ 
    Mutational Stability Regression (MSR) Model.
    
    Wraps an ESM3 model with Parameter-Efficient Fine-Tuning (PEFT) adapters to predict 
    protein stability scores (e.g., ddG). Supports dual (WT + MT) and fused (one shared) adapters.
    """
    def __init__(
            self, lora_config: dict, shared_scale_init: float | None = None,
            shared_bias_init: float | None = None, inference_mode: bool = False, log_likelihood: bool = False,
            use_plddt: bool = False, quaternary_mode: str = 'single_chain', model_dtype: torch.dtype = torch.bfloat16, 
            adapter_mode: str = 'dual', lora_mode: str = 'ensemble', strict_loading: bool = True,
            combined_mode: str = 'mean', per_mutation_bias: bool = True, tie_mt_scale: bool = False,
            native_cond_mt_input: bool = False
        ):
        if (combined_mode or 'mean') != 'mean':
            raise AssertionError(f"combined_mode {combined_mode!r} was removed: the reported prediction is the mean of the two heads.")
        if lora_mode != 'ensemble':
            raise AssertionError(f"lora_mode {lora_mode!r} was removed: the MT head is always an ensemble partner of the WT head.")
        # --native_cond_for_mt: the MT pass reads the WT pass's sequence on native_cond rows (native_cond_mt_batch). A
        # property of the trained model, not of the label alone: a run that only excludes those items keeps the
        # ordinary MT input.
        self.native_cond_mt_input = bool(native_cond_mt_input)
        logging.info("Initializing ESM3 Base Model...")
        base_esm3 = ESM3_sm_open_v0()
        base_esm3.to(model_dtype)
        super().__init__(esm_model=base_esm3)
        
        self.quaternary_mode, self.log_likelihood, self.use_plddt, self.dtype = quaternary_mode, log_likelihood, use_plddt, model_dtype 
        self.adapter_mode, self.lora_mode = adapter_mode, lora_mode
        self.strict_loading = strict_loading
        
        # 1. Initialize Calibration
        if shared_scale_init is not None or shared_bias_init is not None:
            cal_kw = dict(init_scale=shared_scale_init, init_bias=shared_bias_init,
                          requires_grad=not inference_mode, per_mutation=per_mutation_bias)
            if self.adapter_mode == 'fused':
                self.calibration_head_fused = CalibrationHead(**cal_kw)
            else:
                self.calibration_head_wt = CalibrationHead(**cal_kw)
                self.calibration_head_mt = CalibrationHead(**cal_kw)
                if tie_mt_scale:
                    # The MT head shares the WT's SCALE (its own is unused and frozen) but keeps its own bias.
                    self.calibration_head_mt.scale_ref = _ScaleSource(self.calibration_head_wt)
                    self.calibration_head_mt.raw_scale.requires_grad_(False)
        
        # 2. Add LoRAs using config
        logging.info(f"Injecting Adapters (Mode: {self.adapter_mode.upper()} | Strategy: {self.lora_mode.upper()})...")
        self.lora_config = lora_config
        self.add_loras_to_esm3(**self.lora_config)
        
        # 3. Handle structure encoder dtype constraints
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, '_structure_encoder'):
            self.model.base_model._structure_encoder.to(torch.float32)
        elif hasattr(self.model, '_structure_encoder'):
            self.model._structure_encoder.to(torch.float32)

        # 3b. Re-share the structure encoder across adapters (dual mode only).
        # The MT copy is dead weight — it is never used for scoring — and
        # wastes VRAM. The structure encoder is a child module (not a
        # parameter), so it is NOT covered by the parameter re-share in
        # add_loras_to_esm3. Pointing the MT reference at the WT module
        # makes it a single shared copy. Verified numerically lossless
        # (max |Δ pred| == 0.0) in tmp/autocast_fix_test.py.
        if self.adapter_mode == 'dual':
            wt_se = getattr(getattr(self.model, 'base_model', None), '_structure_encoder', None)
            mt_base = getattr(self.peft_mt, 'base_model', None)
            if wt_se is not None and mt_base is not None:
                mt_base._structure_encoder = wt_se

        # 4. Optional freezing for strict inference
        if inference_mode:
            for name, p in self.named_parameters():
                if 'lora' in name or 'peft' in name: p.requires_grad = False

        # 5. Log final model statistics
        self._log_trainable_parameters()

    def _create_lora_config(self, kwargs_dict: dict) -> LoraConfig:
        """Helper to dynamically construct a LoraConfig dictionary and object."""
        lora_rank = kwargs_dict.get('lora_rank', 6)
        lora_alpha = kwargs_dict.get('lora_alpha', 12)
        lora_dropout = kwargs_dict.get('lora_dropout', 0.15)
        target_mode = kwargs_dict.get('target_mode', 'expanded')
        last_n_layers = kwargs_dict.get('last_n_layers', 0)
        incl_structure_encoder = kwargs_dict.get('incl_structure_encoder', False)
        incl_sequence_head = kwargs_dict.get('incl_sequence_head', False)

        TOTAL_BLOCKS = 48
        targets = []
        if target_mode == "baseline": targets.append(r"(?:attn|geom_attn)\.layernorm_qkv\.1")
        elif target_mode == "qkv_outproj": targets.extend([r"(?:attn|geom_attn)\.layernorm_qkv\.1", r"(?:attn|geom_attn)\.out_proj"])
        elif target_mode == "ffn": targets.extend([r"ffn\.1", r"ffn\.3"])
        elif target_mode == "ffn_outproj": targets.extend([r"(?:attn|geom_attn)\.out_proj", r"ffn\.1", r"ffn\.3"])
        elif target_mode == "expanded": targets.extend([r"(?:attn|geom_attn)\.layernorm_qkv\.1", r"ffn\.1", r"ffn\.3"])
        elif target_mode == "all": targets.extend([r"(?:attn|geom_attn)\.layernorm_qkv\.1", r"(?:attn|geom_attn)\.out_proj", r"ffn\.1", r"ffn\.3"])
        else: raise ValueError(f"Unknown target_mode: {target_mode}")

        target_pattern = "|".join(targets)
        block_pattern = r"\d+" if last_n_layers <= 0 or last_n_layers >= TOTAL_BLOCKS else f"({'|'.join([str(i) for i in range(TOTAL_BLOCKS - last_n_layers, TOTAL_BLOCKS)])})"
        base_regex = f".*(?:transformer|structure_encoder)\\.(?:blocks|layers)\\.{block_pattern}\\.({target_pattern})$" if incl_structure_encoder else f"^(?!.*structure_encoder).*transformer\\.blocks\\.{block_pattern}\\.({target_pattern})$"
        target_modules_regex = f"(?:{base_regex})|(?:.*output_heads\\.sequence_head\\.(?:0|3))$" if incl_sequence_head else base_regex

        config_dict = {
            "target_modules": target_modules_regex, 
            "lora_dropout": lora_dropout, 
            "lora_alpha": lora_alpha, 
            "r": lora_rank, 
            "use_rslora": True, 
            "bias": "none"
        }
        return LoraConfig(**config_dict)

    def add_loras_to_esm3(self, **kwargs):
        """
        Instantiates PEFT LoRA adapters for the ESM3 model. 
        Supports independent instantiation of WT and MT adapters using nested 
        'wt_config' and 'mt_config' dictionaries inside the primary config.
        """
        seed, dtype = kwargs.get('seed', None), kwargs.get('dtype', self.dtype)
        for param in self.model.parameters(): param.requires_grad = False
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        if self.adapter_mode == 'dual':
            wt_kwargs = kwargs.get('wt_config', kwargs)
            mt_kwargs = kwargs.get('mt_config', kwargs)
            
            wt_config = self._create_lora_config(wt_kwargs)
            mt_config = self._create_lora_config(mt_kwargs)
            
            logging.info("--- Dual Adapter Configuration ---")
            logging.info(f" WT Adapter | Rank: {wt_config.r:2} | Alpha: {wt_config.lora_alpha:2} | Dropout: {wt_config.lora_dropout}")
            logging.info(f" MT Adapter | Rank: {mt_config.r:2} | Alpha: {mt_config.lora_alpha:2} | Dropout: {mt_config.lora_dropout}")
            logging.debug(f" Target Regex (WT): {wt_config.target_modules}")
            
            base_mt = copy.deepcopy(self.model)
            
            self.peft_wt = get_peft_model(self.model, wt_config, adapter_name="wt_adapter").to(dtype)
            self.peft_mt = get_peft_model(base_mt, mt_config, adapter_name="mt_adapter").to(dtype)
            
            # Re-share the exact underlying memory footprint AFTER .to(dtype) casting!
            def _clean_peft_name(n: str) -> str:
                return n.replace("base_model.model.", "").replace(".base_layer.", ".").replace(".original_module.", ".")

            wt_base_params = {_clean_peft_name(n): p for n, p in self.peft_wt.named_parameters() if "lora_" not in n and "dora_" not in n}
            for name_mt, p_mt in self.peft_mt.named_parameters():
                if "lora_" not in name_mt and "dora_" not in name_mt:
                    clean_name = _clean_peft_name(name_mt)
                    if clean_name in wt_base_params:
                        p_mt.data = wt_base_params[clean_name].data
            
            self.wt_adapter_name = "wt_adapter"
            self.mt_adapter_name = "mt_adapter"
            
            peft_models = [self.peft_wt, self.peft_mt]
            
            # Maintain self.model pointing to peft_wt for backwards compatibility
            self.model = self.peft_wt

        elif self.adapter_mode == 'fused':
            mt_kwargs = kwargs.get('mt_config', kwargs)

            config = self._create_lora_config(mt_kwargs)

            logging.info("--- Fused Adapter Configuration ---")
            logging.info(f" Fused Adapter | Rank: {config.r:2} | Alpha: {config.lora_alpha:2} | Dropout: {config.lora_dropout}")
            logging.debug(f" Target Regex: {config.target_modules}")
            
            self.peft_fused = get_peft_model(self.model, config).to(dtype)
            self.wt_adapter_name = self.mt_adapter_name = list(self.peft_fused.peft_config.keys())[0]
            
            peft_models = [self.peft_fused]
            self.model = self.peft_fused

        # Apply requires_grad correctly via independent PEFT wrappers
        for pm in peft_models:
            for name, param in pm.named_parameters():
                if "lora" in name.lower() or "dora" in name.lower(): param.requires_grad = True

    def _log_trainable_parameters(self):
        """Helper to print total and trainable parameter counts grouped by module."""
        trainable_params, all_param = 0, 0
        group_counts = defaultdict(lambda: {"trainable": 0, "all": 0})
        seen_ids = set()
        
        for name, param in self.named_parameters():
            ptr = param.data.data_ptr()
            if ptr in seen_ids: continue
            seen_ids.add(ptr)

            num_params = param.numel()
            all_param += num_params
            
            # Categorize the parameter robustly using injected PEFT adapter names
            if "wt_adapter" in name: group = "WT Adapter"
            elif "mt_adapter" in name: group = "MT Adapter"
            elif "default" in name and ("lora" in name or "dora" in name): group = "Fused Adapter"
            elif "calibration" in name: group = "Calibration Heads"
            elif "layernorm" in name.lower() or "norm" in name.lower(): group = "LayerNorms"
            else: group = "Base Model (ESM3)"

            group_counts[group]["all"] += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                group_counts[group]["trainable"] += num_params

        logging.info("\n--- Parameter Count Summary ---")
        for group, counts in group_counts.items():
            if counts["trainable"] > 0 or group.endswith("Adapter") or group == "Calibration Heads":
                logging.info(f" {group:<25} | Trainable: {counts['trainable']:>12,d} | Total: {counts['all']:>12,d}")
        
        pct = 100 * trainable_params / all_param if all_param > 0 else 0
        logging.info("-" * 65)
        logging.info(f" TOTAL                     | Trainable: {trainable_params:>12,d} | Total: {all_param:>12,d} ({pct:.4f}%)")
        logging.info("-------------------------------\n")

    def load_lora_weights(self, checkpoint_path: Union[str, Dict[str, str]], load_wt_only: bool = False):
        """
        Intelligently loads LoRA weights, supporting dynamic remapping for independent configurations.
        """
        import os
        
        logging.info("\n=== Loading LoRA Checkpoint(s) ===")
        if isinstance(checkpoint_path, dict):
            logging.info(f"Multi-Checkpoint Mode Triggered. Targets: {list(checkpoint_path.keys())}")
        else:
            logging.info(f"Single Checkpoint Path: {checkpoint_path}")

        my_keys = dict(self.named_parameters())
        new_state_dict = {}
        
        # Track if we encounter MT weights while in load_wt_only mode
        mt_weights_skipped = 0 

        def extract_core(k: str):
            """
            Strips out all nested Lightning and PEFT wrappers to return 
            the clean architectural path + adapter type.
            """
            # 1. Identify adapter type
            adapter = 'unknown'
            if 'wt_adapter' in k or 'calibration_head_wt' in k: adapter = 'wt'
            elif 'mt_adapter' in k or 'calibration_head_mt' in k: adapter = 'mt'
            elif 'default' in k or 'calibration_head_fused' in k: adapter = 'default'
                
            # 2. Normalize adapter names
            k = k.replace('wt_adapter', '<ADAPT>').replace('mt_adapter', '<ADAPT>').replace('default', '<ADAPT>')
            k = k.replace('calibration_head_wt', 'calibration_head_<ADAPT>')
            k = k.replace('calibration_head_mt', 'calibration_head_<ADAPT>')
            k = k.replace('calibration_head_fused', 'calibration_head_<ADAPT>')
            
            # 3. Strip PEFT namespace injections
            k = k.replace('peft_wt.', '').replace('peft_mt.', '').replace('peft_fused.', '')
            
            # 4. Strip nested model/base_model wrappers (handles arbitrary Lightning/PEFT nesting)
            k = re.sub(r'^(model\.|base_model\.)+', '', k)
            
            # 5. Clean up PEFT's internal module renaming
            k = k.replace('.base_layer.', '.').replace('.original_module.', '.')
            
            # 6. Final safety strip in case PEFT exposed another base_model prefix
            k = re.sub(r'^(model\.|base_model\.)+', '', k)
            
            return k, adapter

        # Build a reverse-lookup map of our target model parameters
        my_core_map = defaultdict(list)
        for my_key in my_keys.keys():
            core, adapter = extract_core(my_key)
            my_core_map[(core, adapter)].append(my_key)

        def process_state_dict(state_dict, override_target=None):
            nonlocal mt_weights_skipped
            
            for ckpt_key, tensor in state_dict.items():
                # Allow auxiliary components and unfrozen LayerNorms through
                if not any(x in ckpt_key.lower() for x in ['lora', 'dora', 'calibration_head', 'norm']): 
                    continue

                core, ckpt_adapter = extract_core(ckpt_key)
                
                # Apply forced overrides for dictionary-based checkpoint loading
                if override_target:
                    if override_target in ['shared', 'base']: ckpt_adapter = 'unknown'
                    elif override_target == 'wt_adapter': ckpt_adapter = 'wt'
                    elif override_target == 'mt_adapter': ckpt_adapter = 'mt'

                # Intercept MT adapters if we are only loading WT
                if load_wt_only and ckpt_adapter == 'mt':
                    mt_weights_skipped += 1
                    continue

                # Determine which internal adapters should receive this checkpoint weight
                targets = []
                if ckpt_adapter == 'default':
                    if self.adapter_mode == 'dual': targets.extend(['wt', 'mt'])
                    else: targets.append('default')
                elif ckpt_adapter == 'wt':
                    targets.append('wt' if self.adapter_mode == 'dual' else 'default')
                elif ckpt_adapter == 'mt':
                    if self.adapter_mode == 'dual': targets.append('mt')
                else:
                    targets.append('unknown')

                # Map the weights
                mapped = False
                for target in targets:
                    for my_target_key in my_core_map.get((core, target), []):
                        new_state_dict[my_target_key] = tensor.clone()
                        mapped = True
                
                if not mapped:
                    new_state_dict[ckpt_key] = tensor # fallback for PyTorch to gracefully drop/warn

        def _load_single_checkpoint(path: str) -> dict:
            """Handles seamless loading for both .safetensors and .ckpt formats."""
            if not os.path.exists(path):
                raise FileNotFoundError(f"LoRA checkpoint file does not exist at: {path}")

            if path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    return load_file(path, device='cpu')
                except ImportError:
                    raise NotImplementedError(
                        "The 'safetensors' library is required to load this checkpoint. "
                        "Please install it via 'pip install safetensors'."
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to parse safetensors file. Ensure the file is not corrupted. Error: {e}")
            else:
                try:
                    # weights_only=True blocks execution of malicious pickle payloads.
                    ckpt = torch.load(path, map_location='cpu', weights_only=True)
                    return ckpt.get('state_dict', ckpt)
                except Exception as e:
                    raise RuntimeError(f"Failed to load PyTorch .ckpt file. Error: {e}")

        # Process the checkpoints
        if isinstance(checkpoint_path, str):
            state_dict = _load_single_checkpoint(checkpoint_path)
            process_state_dict(state_dict)
        elif isinstance(checkpoint_path, dict):
            for adapter_target, path in checkpoint_path.items():
                state_dict = _load_single_checkpoint(path)
                process_state_dict(state_dict, override_target=adapter_target)

        if load_wt_only and mt_weights_skipped > 0:
            logging.warning("\n--- Partial Load Triggered ---")
            logging.warning(f"Detected {mt_weights_skipped} mutant (MT) adapter weights in the checkpoint.")
            logging.warning("Because 'load_wt_only=True' was specified, these MT weights have been intentionally discarded.")

        # Execute Load & Track Outcomes
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        # Cross-reference with all keys that *should* be trained (ignoring freeze locks)
        trainable_names = {k for k, p in self.named_parameters() if p.requires_grad or 'lora' in k.lower() or 'dora' in k.lower() or 'calibration' in k.lower()}
        
        loaded_trainable = set(new_state_dict.keys()).intersection(trainable_names)
        missing_trainable = set(missing).intersection(trainable_names)
        
        logging.info(f"Checkpoint Keys Found: {len(new_state_dict) + len(unexpected)}")
        logging.info(f"Keys Mapped to Model: {len(new_state_dict)}")
        logging.info(f"Missing Parameters: {len(missing_trainable)}")
        
        if missing_trainable:
            logging.error("\n--- Missing Modules (Not Loaded) ---")
            logging.error("These parameters were not found in the checkpoint and retain their random initialization:")
            
            missing_groups = defaultdict(int)
            for k in missing_trainable:
                if 'wt_adapter' in k or 'peft_wt' in k: missing_groups['WT Adapter'] += 1
                elif 'mt_adapter' in k or 'peft_mt' in k: missing_groups['MT Adapter'] += 1
                elif 'peft_fused' in k or 'default' in k: missing_groups['Fused Adapter'] += 1
                elif 'calibration_head' in k: missing_groups['Calibration Head'] += 1
                else: missing_groups['Other'] += 1
                
            for group, count in missing_groups.items():
                logging.error(f"  > {group:<25}: {count} tensors")
                
            logging.debug("Detailed missing keys:")
            for k in sorted(missing_trainable)[:10]: logging.debug(f"  - {k}")
            if len(missing_trainable) > 10: logging.debug(f"  ... and {len(missing_trainable)-10} more.")
            logging.error("---------------------------------------------")
            if self.strict_loading:
                raise KeyError('Expected parameters were missing from the LoRA, suggesting that the wrong configuration was used.')

        if unexpected:
            logging.error(f"Unexpected Tensors in Checkpoint (Ignored): {len(unexpected)}")
            print(unexpected)

        logging.info("=== Checkpoint Loading Complete ===\n")

    def forward_batch(self, batch_in: Dict[str, Any], cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, skip_reverse: bool = False, mask_strategy: Union[str, Dict[str, Optional[str]], None] = None) -> Dict[str, torch.Tensor]:
        if self.training: raise AssertionError("forward_batch is for inference only. Use forward_partitioned for training.")

        wt_out = self.forward_partitioned(batch_in, pass_type='wt', cached_wt_esm3=cached_wt_esm3, mask_strategy=_per_pass(mask_strategy, 'wt'))
        if not skip_reverse:
            mt_out = self.forward_partitioned(batch_in, pass_type='mt', mask_strategy=_per_pass(mask_strategy, 'mt'))
        else:
            mt_out = wt_out
        
        wt_pred_cal, mt_pred_cal = wt_out['pred_calibrated'], mt_out['pred_calibrated']
        wt_pred_raw, mt_pred_raw = wt_out['pred_raw'], mt_out['pred_raw']

        combined_pred = combine_predictions(wt_pred_cal, mt_pred_cal)

        # The predicted epistasis is whatever the combined prediction adds beyond the WT head: 0.5(mt-wt) under
        # 'mean', mt-wt under 'route' (doubles). Mode-agnostic definition.
        epi_pred = combined_pred - wt_pred_cal

        if skip_reverse:
            mt_pred_raw, mt_pred_cal, combined_pred, epi_pred = float('nan'), float('nan'), float('nan'), float('nan')

        return {'wt_lora_pred': wt_pred_cal, 'mt_lora_pred': mt_pred_cal, 'wt_lora_raw': wt_pred_raw, 'mt_lora_raw': mt_pred_raw, 'combined_pred': combined_pred, 'epi_pred': epi_pred}

    def forward_cycle(self, batch_in: Dict[str, Any], mask_strategy: Union[str, Dict[str, Optional[str]], None] = None,
                      include_mt: bool = True) -> Dict[str, torch.Tensor]:
        """
        The two passes forward_batch does not run, both returned as a FORWARD (wild type -> mutant) ddG estimate:

        * wt_rev: the WT adapter on the MUTATED sequence, asked about the reverse mutation (mutant -> wild type): the adapter's native readout of
          the reverse change, calibrated as usual and negated.
        * mt_fwd: the MT adapter on the WILD-TYPE sequence, scoring the forward mutation (an input regime it never trains on: a probe).

        With forward_batch's WT pass (wt_fwd) and MT pass (mt_rev: mutated sequence in; its readout is the forward one up to sign) they make
        the 2 x 2 of adapters and directions behind the symmetrised heads of the epistasis hierarchy, HEAD_MUT = (X + ~X)/2.

        ``include_mt=False`` returns only ``wt_rev``, for a run whose MT head is not trained (a WT-only sweep): the ctx control is
        the WT adapter read in reverse and does not need the MT probe, which would only be a forward through an untrained adapter.
        """
        if self.training:
            raise AssertionError("forward_cycle is for inference only.")
        rev = dict(batch_in)
        rev['wt_sequence_tokens'] = batch_in['mt_sequence_tokens']
        rev['wt_id'], rev['mt_id'] = batch_in['mt_id'], batch_in['wt_id']
        wt_rev = self.forward_partitioned(rev, pass_type='wt', mask_strategy=_per_pass(mask_strategy, 'wt'))['pred_calibrated']
        if not include_mt:
            return {'wt_rev': -wt_rev}
        fwd = dict(batch_in)
        fwd['mt_sequence_tokens'] = batch_in['wt_sequence_tokens']
        mt_fwd = self.forward_partitioned(fwd, pass_type='mt', mask_strategy=_per_pass(mask_strategy, 'mt'))['pred_calibrated']
        return {'wt_rev': -wt_rev, 'mt_fwd': mt_fwd}

    def _process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.log_likelihood: return logits
        idx = self.canonical_idx_tensor
        log_probs_canonical = torch.nn.functional.log_softmax(logits[:, :, idx], dim=-1)
        full_log_probs = torch.full_like(logits, float('-inf'))
        full_log_probs[:, :, idx] = log_probs_canonical
        return full_log_probs

    def _mt_pass_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """The MT pass's input: native_cond rows get the WT pass's sequence only in a --native_cond_for_mt model."""
        return native_cond_mt_batch(batch) if self.native_cond_mt_input else batch

    def forward_partitioned(self, batch: Dict[str, Any], pass_type: str, cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, mask_strategy: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if pass_type not in ['wt', 'mt']: raise AssertionError(f"pass_type must be 'wt' or 'mt'. Received: {pass_type}")

        if pass_type == 'mt':
            batch = self._mt_pass_batch(batch)

        seq, mut_pos = batch.get(f'{pass_type}_sequence_tokens'), batch.get('mut_pos')
        wt_id, mt_id, mut_mask = batch.get('wt_id'), batch.get('mt_id'), batch.get('mut_mask')
        coords, struct_tokens, plddt = batch.get('coords'), batch.get('structure_tokens'), batch.get('plddt')

        B, max_muts = seq.shape[0], mut_pos.shape[1]
        safe_pos = mut_pos.clone(); safe_pos[~mut_mask] = 0
        safe_wt_id = wt_id.clone(); safe_wt_id[~mut_mask] = 0
        safe_mt_id = mt_id.clone(); safe_mt_id[~mut_mask] = 0
        b_idx = torch.arange(B, device=seq.device).unsqueeze(1).expand(-1, max_muts)

        pdb_list = batch.get('pdb')
        if B <= 1:
            same_struct = True
        elif isinstance(pdb_list, (list, tuple)) and len(pdb_list) == B:
            same_struct = (len(set(pdb_list)) == 1)
        else:
            same_struct = ((coords is None or bool(torch.equal(coords[0], coords[-1]))) and
                           (struct_tokens is None or bool(torch.equal(struct_tokens[0], struct_tokens[-1]))))

        if mask_strategy is not None:
            if mask_strategy == 'chain':
                mask_strategy = 'independent'  # legacy alias for old hparams.yaml
            if mask_strategy not in ['independent', 'marginal']:
                raise AssertionError(f"Invalid mask_strategy: '{mask_strategy}'. Expected None, 'independent', 'chain', or 'marginal'.")
            if cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used with mask_strategy='{mask_strategy}'. Each masking pass fundamentally alters the model sequence state.")
            
            mask_token_id = C.SEQUENCE_MASK_TOKEN
            
            if getattr(self, 'adapter_mode', 'dual') == 'dual':
                active_model = self.peft_wt if pass_type == 'wt' else self.peft_mt
            else:
                active_model = self.peft_fused
                
            unsummed_llr = torch.zeros((B, max_muts), dtype=torch.float32, device=seq.device)
            lib_cache = getattr(self, '_masked_logit_cache', None)
            can_dedup = same_struct and not getattr(active_model, 'training', False)

            def _expand_b(t: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
                return t[0:1].expand(n, *([-1] * (t.ndim - 1))) if t is not None else None

            def _eval_masked_seqs(seqs_in: torch.Tensor, row_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
                n_rows = seqs_in.shape[0]
                if same_struct:
                    c_in = _expand_b(coords, n_rows)
                    s_in = _expand_b(struct_tokens, n_rows)
                    p_in = _expand_b(plddt, n_rows)
                else:
                    c_in = coords[row_idx] if (coords is not None and row_idx is not None) else coords
                    s_in = struct_tokens[row_idx] if (struct_tokens is not None and row_idx is not None) else struct_tokens
                    p_in = plddt[row_idx] if (plddt is not None and row_idx is not None) else plddt

                has_pdb_id = isinstance(pdb_list, (list, tuple)) and len(pdb_list) == B and len(set(pdb_list)) == 1
                if same_struct and has_pdb_id and lib_cache is not None and not torch.is_grad_enabled():
                    pdb_id = pdb_list[0]
                    seqs_np = seqs_in.detach().cpu().numpy()
                    keys = [(pdb_id, pass_type, seqs_np[r].tobytes()) for r in range(n_rows)]
                    miss_indices = [r for r, k in enumerate(keys) if k not in lib_cache]
                    if miss_indices:
                        miss_t = torch.tensor(miss_indices, dtype=torch.long, device=seqs_in.device)
                        miss_seqs = seqs_in[miss_t]
                        chunk_sz = max(1, B)
                        for st in range(0, len(miss_indices), chunk_sz):
                            en = min(st + chunk_sz, len(miss_indices))
                            sub_n = en - st
                            mc = _expand_b(coords, sub_n)
                            ms = _expand_b(struct_tokens, sub_n)
                            mp = _expand_b(plddt, sub_n)
                            out_m = self._get_esm3_outputs(miss_seqs[st:en], mc, ms, mp, active_model=active_model)
                            logits_m = self._process_logits(out_m.sequence_logits.float())
                            for j_local in range(sub_n):
                                r_orig = miss_indices[st + j_local]
                                lib_cache[keys[r_orig]] = logits_m[j_local]
                    return torch.stack([lib_cache[k] for k in keys], dim=0)

                out = self._get_esm3_outputs(seqs_in, c_in, s_in, p_in, active_model=active_model)
                return self._process_logits(out.sequence_logits.float())
            
            if mask_strategy == 'independent':
                b_list, i_list, pos_list, seq_list = [], [], [], []
                for i in range(max_muts):
                    curr_mask = mut_mask[:, i]
                    if not curr_mask.any():
                        continue
                    b_valid = torch.where(curr_mask)[0]
                    p_valid = safe_pos[b_valid, i]
                    m_seq = seq[b_valid].clone()
                    m_seq[torch.arange(b_valid.shape[0], device=seq.device), p_valid] = mask_token_id
                    b_list.append(b_valid)
                    i_list.append(torch.full_like(b_valid, i))
                    pos_list.append(p_valid)
                    seq_list.append(m_seq)

                if b_list:
                    b_all = torch.cat(b_list, dim=0)
                    i_all = torch.cat(i_list, dim=0)
                    pos_all = torch.cat(pos_list, dim=0)
                    seqs_all = torch.cat(seq_list, dim=0)

                    if same_struct:
                        unique_seqs, inv_idx = torch.unique(seqs_all, dim=0, return_inverse=True)
                        step_logits_U = _eval_masked_seqs(unique_seqs)
                        mt_logits = step_logits_U[inv_idx, pos_all, safe_mt_id[b_all, i_all]]
                        wt_logits = step_logits_U[inv_idx, pos_all, safe_wt_id[b_all, i_all]]
                        unsummed_llr[b_all, i_all] = mt_logits - wt_logits
                    else:
                        step_logits_M = _eval_masked_seqs(seqs_all, row_idx=b_all)
                        m_idx = torch.arange(seqs_all.shape[0], device=seq.device)
                        mt_logits = step_logits_M[m_idx, pos_all, safe_mt_id[b_all, i_all]]
                        wt_logits = step_logits_M[m_idx, pos_all, safe_wt_id[b_all, i_all]]
                        unsummed_llr[b_all, i_all] = mt_logits - wt_logits

            elif mask_strategy == 'marginal':
                masked_seq = seq.clone()
                
                # Apply all masks simultaneously
                for i in range(max_muts):
                    curr_mask = mut_mask[:, i]
                    batch_idx_valid = torch.where(curr_mask)[0]
                    pos_valid = safe_pos[curr_mask, i]
                    masked_seq[batch_idx_valid, pos_valid] = mask_token_id
                
                if same_struct:
                    unique_seqs, inv_idx = torch.unique(masked_seq, dim=0, return_inverse=True)
                    step_logits_U = _eval_masked_seqs(unique_seqs)
                    step_logits_B = step_logits_U[inv_idx]
                else:
                    step_logits_B = _eval_masked_seqs(masked_seq)
                
                mt_logits, wt_logits = step_logits_B[b_idx, safe_pos, safe_mt_id], step_logits_B[b_idx, safe_pos, safe_wt_id]
                unsummed_llr = torch.where(mut_mask, mt_logits - wt_logits, torch.zeros_like(mt_logits))

        else:
            if pass_type == 'wt' and cached_wt_esm3 is not None:
                ref_seq = cached_wt_esm3['seq']
                # Fast path: when the cached 'seq' is the same storage (e.g. an
                # expand view of the chunk sequence), skip the torch.equal
                # device sync; otherwise fall back to the value check.
                if seq.shape != ref_seq.shape or (
                        seq.data_ptr() != ref_seq.data_ptr()
                        and not torch.equal(seq[0], ref_seq[0])):
                    logging.info(seq[0])
                    logging.info(ref_seq[0])
                    raise AssertionError("Sequence mismatch in cache.")
                logits_B = cached_wt_esm3['logits'].expand(B, -1, -1)
            else:
                if getattr(self, 'adapter_mode', 'dual') == 'dual':
                    active_model = self.peft_wt if pass_type == 'wt' else self.peft_mt
                else:
                    active_model = self.peft_fused

                if pass_type == 'wt' and B > 1 and same_struct and not getattr(active_model, 'training', False) and bool((seq == seq[0:1]).all()):
                    lib_cache = getattr(self, '_masked_logit_cache', None)
                    has_pdb_id = isinstance(pdb_list, (list, tuple)) and len(pdb_list) == B and len(set(pdb_list)) == 1
                    cache_key = (pdb_list[0], 'wt_unmasked', seq[0].detach().cpu().numpy().tobytes()) if (has_pdb_id and lib_cache is not None and not torch.is_grad_enabled()) else None
                    if cache_key is not None and cache_key in lib_cache:
                        logits_1 = lib_cache[cache_key]
                    else:
                        c_1 = coords[0:1] if coords is not None else None
                        s_1 = struct_tokens[0:1] if struct_tokens is not None else None
                        p_1 = plddt[0:1] if plddt is not None else None
                        out = self._get_esm3_outputs(seq[0:1], c_1, s_1, p_1, active_model=active_model)
                        logits_1 = self._process_logits(out.sequence_logits.float())
                        if cache_key is not None:
                            lib_cache[cache_key] = logits_1
                    logits_B = logits_1.expand(B, -1, -1)
                else:
                    out = self._get_esm3_outputs(seq, coords, struct_tokens, plddt, active_model=active_model)
                    logits_B = self._process_logits(out.sequence_logits.float())

            if pass_type == 'wt':
                mt_logits, wt_logits = logits_B[b_idx, safe_pos, safe_mt_id], logits_B[b_idx, safe_pos, safe_wt_id]
                unsummed_llr = torch.where(mut_mask, mt_logits - wt_logits, torch.zeros_like(mt_logits))
            elif pass_type == 'mt':
                wt_logits, mt_logits = logits_B[b_idx, safe_pos, safe_wt_id], logits_B[b_idx, safe_pos, safe_mt_id]
                unsummed_llr = -torch.where(mut_mask, wt_logits - mt_logits, torch.zeros_like(wt_logits))

        # --- Shared Post-Processing & Calibration ---
        llr_sum_raw = unsummed_llr.sum(dim=1)
        # Both the sum above and a per-mutation bias are per-position quantities, so the count comes from the
        # same mask that selected the positions.
        n_mut = mut_mask.sum(dim=1)

        if pass_type == 'wt':
            cal_wt = getattr(self, 'calibration_head_fused', None) if self.adapter_mode == 'fused' else getattr(self, 'calibration_head_wt', None)
            llr_sum_cal = cal_wt(llr_sum_raw, n_mut) if cal_wt is not None else llr_sum_raw
        elif pass_type == 'mt':
            cal_mt = getattr(self, 'calibration_head_fused', None) if self.adapter_mode == 'fused' else getattr(self, 'calibration_head_mt', None)
            llr_sum_cal = cal_mt(llr_sum_raw, n_mut) if cal_mt is not None else llr_sum_raw

        output_dict = {'pred_calibrated': llr_sum_cal, 'pred_raw': llr_sum_raw, 'unsummed_llr': unsummed_llr}
        return output_dict

    @torch.no_grad()
    def score_screening_batch(self, wt_sequence_tokens: torch.Tensor, mut_pos: torch.Tensor, wt_id: torch.Tensor, mt_id: torch.Tensor, mut_mask: torch.Tensor, coords: Optional[torch.Tensor] = None, structure_tokens: Optional[torch.Tensor] = None, plddt: Optional[torch.Tensor] = None, mask_strategy: Union[str, Dict[str, Optional[str]], None] = None, batch_size: int = 32, skip_reverse: bool = False, cached_wt_esm3: Optional[Dict] = None, quiet: bool = False, auto_batch: Optional["AutoBatchSizer"] = None) -> Dict[str, torch.Tensor]:
        """
        A unified, sparse-input scoring engine. 
        Routes dynamically between dense chunking (for unmasked) and state deduplication (for masked)
        to prevent VRAM explosions while maintaining maximum throughput.
        """
        if self.training:
            raise AssertionError("score_screening_batch is strictly for inference.")
        if wt_sequence_tokens.shape[0] != 1:
            raise AssertionError(f"Memory Guard: score_screening_batch requires exactly ONE wild-type sequence of shape [1, L]. Received shape {wt_sequence_tokens.shape}. Sparse indices must be used to define the batch.")

        B, max_muts = mut_pos.shape
        device = wt_sequence_tokens.device
        _hb = (auto_batch.log if (auto_batch is not None and getattr(auto_batch, 'log', None)) else (lambda m: print(m, flush=True)))
        
        wt_lora_pred = torch.zeros(B, dtype=torch.float32, device=device)
        mt_lora_pred = torch.zeros(B, dtype=torch.float32, device=device)
        combined_pred = torch.zeros(B, dtype=torch.float32, device=device)

        mask_wt = _per_pass(mask_strategy, 'wt')
        mask_mt = _per_pass(mask_strategy, 'mt')
        if mask_wt == 'chain': mask_wt = 'independent'
        if mask_mt == 'chain': mask_mt = 'independent'
        for m_s in (mask_wt, mask_mt):
            if m_s not in (None, 'independent', 'marginal'):
                raise AssertionError(f"Invalid mask_strategy: '{m_s}'. Expected 'independent', 'chain', 'marginal', or None.")

        if mask_wt is None and (mask_mt is None or skip_reverse):
            # ROUTE 1: Dense Chunking for Unmasked (Maximum GPU Saturation, No Hashing Overhead)
            def _dense_chunk(start_idx, end_idx):
                curr_B = end_idx - start_idx

                if skip_reverse and cached_wt_esm3 is not None:
                    # WT-only cached path: the WT pass is served entirely from
                    # the cached logits, so neither dense sequence tensor is
                    # ever read. Skip the O(B) clones and the per-row
                    # torch.where/setitem reconstruction of chunk_mt_seq.
                    chunk_wt_seq = wt_sequence_tokens.expand(curr_B, -1)
                    chunk_mt_seq = chunk_wt_seq
                else:
                    chunk_wt_seq = wt_sequence_tokens.expand(curr_B, -1).clone()
                    chunk_mt_seq = chunk_wt_seq.clone()

                    # Reconstruct dense mutant sequences just-in-time
                    for i in range(curr_B):
                        b = start_idx + i
                        valid_idx = torch.where(mut_mask[b])[0]
                        chunk_mt_seq[i, mut_pos[b, valid_idx]] = mt_id[b, valid_idx]

                chunk_batch = {
                    'wt_sequence_tokens': chunk_wt_seq,
                    'mt_sequence_tokens': chunk_mt_seq,
                    'mut_pos': mut_pos[start_idx:end_idx],
                    'wt_id': wt_id[start_idx:end_idx],
                    'mt_id': mt_id[start_idx:end_idx],
                    'mut_mask': mut_mask[start_idx:end_idx],
                    'coords': coords.expand(curr_B, -1, -1, -1) if coords is not None else None,
                    'structure_tokens': structure_tokens.expand(curr_B, -1) if structure_tokens is not None else None,
                    'plddt': plddt.expand(curr_B, -1) if plddt is not None else None,
                }

                chunk_cache = None
                if cached_wt_esm3 is not None:
                     chunk_cache = {
                         'seq': chunk_wt_seq,
                         'logits': cached_wt_esm3['logits'].expand(curr_B, -1, -1),
                         'embeddings': cached_wt_esm3['embeddings'].expand(curr_B, -1, -1) if cached_wt_esm3.get('embeddings') is not None else None
                     }

                out = self.forward_batch(chunk_batch, cached_wt_esm3=chunk_cache, skip_reverse=skip_reverse, mask_strategy=None)

                wt_lora_pred[start_idx:end_idx] = out['wt_lora_pred']
                mt_lora_pred[start_idx:end_idx] = out['mt_lora_pred']
                combined_pred[start_idx:end_idx] = out['combined_pred']

            if auto_batch is None:
                for start_idx in tqdm(range(0, B, batch_size), desc='Computing dense unmasked mutants', disable=quiet):
                    _dense_chunk(start_idx, min(start_idx + batch_size, B))
            else:
                auto_batch.run(B, _dense_chunk, label=f'dense-B{B}-L{wt_sequence_tokens.shape[1]}')

        else:
            # ROUTE 2: State Deduplication for Masked (Supports symmetric or per-pass asymmetric masking)
            if mask_wt is not None and cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used when WT mask_strategy='{mask_wt}'.")

            mask_token_id = int(C.SEQUENCE_MASK_TOKEN)
            wt_state_reqs = defaultdict(set)
            mt_state_reqs = defaultdict(set)
            state_map = defaultdict(dict)

            base_wt = wt_sequence_tokens.squeeze(0)
            mut_pos_cpu = mut_pos.detach().cpu().tolist()
            mt_id_cpu = mt_id.detach().cpu().tolist()
            mut_mask_cpu = mut_mask.detach().cpu().tolist()

            _tA = time.time()
            _hb(f"    [PHASE-A] start B={B} L={wt_sequence_tokens.shape[1]} (wt={mask_wt}, mt={mask_mt})")
            for b in tqdm(range(B), desc='Constructing efficient masked batches to evaluate', disable=quiet):
                if b and b % 100_000 == 0:
                    _hb(f"    [PHASE-A] {b}/{B} ({time.time()-_tA:.0f}s)")
                valid_indices = [i for i, m in enumerate(mut_mask_cpu[b]) if m]
                if not valid_indices:
                    continue

                if mask_wt == 'independent':
                    wt_keys = {i: ((mut_pos_cpu[b][i], mask_token_id),) for i in valid_indices}
                elif mask_wt == 'marginal':
                    wk = tuple(sorted((mut_pos_cpu[b][i], mask_token_id) for i in valid_indices))
                    wt_keys = {i: wk for i in valid_indices}
                else:
                    wt_keys = {i: None for i in valid_indices}

                if not skip_reverse and mask_mt == 'independent':
                    mt_keys = {}
                    for i in valid_indices:
                        ov = {}
                        for j in valid_indices:
                            ov[mut_pos_cpu[b][j]] = int(mt_id_cpu[b][j])
                        ov[mut_pos_cpu[b][i]] = mask_token_id
                        mt_keys[i] = tuple(sorted(ov.items()))
                elif not skip_reverse and mask_mt == 'marginal':
                    mk = tuple(sorted((mut_pos_cpu[b][i], mask_token_id) for i in valid_indices))
                    mt_keys = {i: mk for i in valid_indices}
                else:
                    mt_keys = {i: None for i in valid_indices}

                for i in valid_indices:
                    pos = mut_pos_cpu[b][i]
                    wk, mk = wt_keys[i], mt_keys[i]
                    if wk is not None:
                        wt_state_reqs[wk].add(pos)
                    if mk is not None:
                        mt_state_reqs[mk].add(pos)
                    state_map[b][i] = (wk, mk)

            _hb(f"    [PHASE-A] done in {time.time()-_tA:.0f}s: {len(wt_state_reqs)} unique wt states, {len(mt_state_reqs)} unique mt states")

            def compute_cache(states_reqs, active_model, label):
                cache = defaultdict(dict)
                if not states_reqs:
                    return cache
                states_list = list(states_reqs.keys())
                done, t0, next_hb = 0, time.time(), 100_000

                def _state_chunk(start_idx, end_idx):
                    nonlocal done, next_hb
                    batch_keys = states_list[start_idx:end_idx]
                    curr_b = len(batch_keys)
                    batch_tensor = base_wt.unsqueeze(0).expand(curr_b, -1).clone()
                    for j, state_key in enumerate(batch_keys):
                        for p_ov, tok_ov in state_key:
                            batch_tensor[j, p_ov] = tok_ov

                    b_coords = coords.expand(curr_b, -1, -1, -1) if coords is not None else None
                    b_struct = structure_tokens.expand(curr_b, -1) if structure_tokens is not None else None
                    b_plddt = plddt.expand(curr_b, -1) if plddt is not None else None

                    out = self._get_esm3_outputs(batch_tensor, b_coords, b_struct, b_plddt, active_model=active_model)
                    logits = self._process_logits(out.sequence_logits.float())

                    for j, state_key in enumerate(batch_keys):
                        for p in states_reqs[state_key]:
                            cache[state_key][p] = logits[j, p, :].clone()
                    done += curr_b
                    if done >= next_hb:
                        _hb(f"    [CACHE {label}] {done}/{len(states_list)} states ({time.time()-t0:.0f}s)")
                        next_hb += 100_000

                if auto_batch is None:
                    for start_idx in tqdm(range(0, len(states_list), batch_size), desc='Computing cache', disable=quiet):
                        _state_chunk(start_idx, min(start_idx + batch_size, len(states_list)))
                else:
                    auto_batch.run(len(states_list), _state_chunk, label=label)
                return cache

            is_dual = getattr(self, 'adapter_mode', 'dual') == 'dual'
            _L = wt_sequence_tokens.shape[1]
            active_wt = self.peft_wt if is_dual else self.peft_fused
            active_mt = self.peft_mt if is_dual else self.peft_fused

            unsummed_llr_wt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)
            unsummed_llr_mt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)

            safe_pos = mut_pos.clone(); safe_pos[~mut_mask] = 0
            safe_wt = wt_id.clone(); safe_wt[~mut_mask] = 0
            safe_mt = mt_id.clone(); safe_mt[~mut_mask] = 0

            if mask_wt is None:
                if cached_wt_esm3 is not None:
                    wt_logits_1L = cached_wt_esm3['logits']
                else:
                    out_wt = self._get_esm3_outputs(wt_sequence_tokens, coords, structure_tokens, plddt, active_model=active_wt)
                    wt_logits_1L = self._process_logits(out_wt.sequence_logits.float())
                wt_pass_mt = wt_logits_1L[0, safe_pos, safe_mt]
                wt_pass_wt = wt_logits_1L[0, safe_pos, safe_wt]
                unsummed_llr_wt = torch.where(mut_mask, wt_pass_mt - wt_pass_wt, torch.zeros_like(wt_pass_mt))
                wt_cache = {}
            else:
                wt_cache = compute_cache(wt_state_reqs, active_wt, label=f'wt-L{_L}')

            if not skip_reverse:
                if mask_mt is None:
                    def _dense_mt_chunk(start_idx, end_idx):
                        curr_B = end_idx - start_idx
                        chunk_mt_seq = wt_sequence_tokens.expand(curr_B, -1).clone()
                        for i_loc in range(curr_B):
                            b_abs = start_idx + i_loc
                            v_idx = torch.where(mut_mask[b_abs])[0]
                            chunk_mt_seq[i_loc, mut_pos[b_abs, v_idx]] = mt_id[b_abs, v_idx]
                        chunk_batch = {
                            'wt_sequence_tokens': wt_sequence_tokens.expand(curr_B, -1),
                            'mt_sequence_tokens': chunk_mt_seq,
                            'mut_pos': mut_pos[start_idx:end_idx],
                            'wt_id': wt_id[start_idx:end_idx],
                            'mt_id': mt_id[start_idx:end_idx],
                            'mut_mask': mut_mask[start_idx:end_idx],
                            'coords': coords.expand(curr_B, -1, -1, -1) if coords is not None else None,
                            'structure_tokens': structure_tokens.expand(curr_B, -1) if structure_tokens is not None else None,
                            'plddt': plddt.expand(curr_B, -1) if plddt is not None else None,
                        }
                        mt_out = self.forward_partitioned(chunk_batch, pass_type='mt', mask_strategy=None)
                        unsummed_llr_mt[start_idx:end_idx] = mt_out['unsummed_llr']

                    if auto_batch is None:
                        for start_idx in tqdm(range(0, B, batch_size), desc='Computing dense MT mutants', disable=quiet):
                            _dense_mt_chunk(start_idx, min(start_idx + batch_size, B))
                    else:
                        auto_batch.run(B, _dense_mt_chunk, label=f'dense-mt-B{B}-L{_L}')
                    mt_cache = {}
                elif is_dual or mask_wt is None:
                    mt_cache = compute_cache(mt_state_reqs, active_mt, label=f'mt-L{_L}')
                else:
                    all_reqs = defaultdict(set)
                    for tup, poses in wt_state_reqs.items(): all_reqs[tup].update(poses)
                    for tup, poses in mt_state_reqs.items(): all_reqs[tup].update(poses)
                    mt_cache = wt_cache = compute_cache(all_reqs, self.peft_fused, label=f'fused-L{_L}')
            else:
                mt_cache = {}

            _tC = time.time()
            _hb(f"    [COLLATE] start B={B}")
            wt_id_cpu = wt_id.detach().cpu().tolist()
            if mask_wt is not None or (not skip_reverse and mask_mt is not None):
                for b in tqdm(range(B), desc='Collating results', disable=quiet):
                    if b and b % 100_000 == 0:
                        _hb(f"    [COLLATE] {b}/{B} ({time.time()-_tC:.0f}s)")
                    valid_indices = [i for i, m in enumerate(mut_mask_cpu[b]) if m]
                    for i in valid_indices:
                        wt_tup, mt_tup = state_map[b][i]
                        pos = mut_pos_cpu[b][i]
                        w_id, m_id = wt_id_cpu[b][i], mt_id_cpu[b][i]

                        if wt_tup is not None:
                            wt_pass_mt_logit = wt_cache[wt_tup][pos][m_id]
                            wt_pass_wt_logit = wt_cache[wt_tup][pos][w_id]
                            unsummed_llr_wt[b, i] = wt_pass_mt_logit - wt_pass_wt_logit

                        if not skip_reverse and mt_tup is not None:
                            mt_pass_wt_logit = mt_cache[mt_tup][pos][w_id]
                            mt_pass_mt_logit = mt_cache[mt_tup][pos][m_id]
                            unsummed_llr_mt[b, i] = mt_pass_mt_logit - mt_pass_wt_logit
            if skip_reverse:
                unsummed_llr_mt = unsummed_llr_wt.clone()

            _hb(f"    [COLLATE] done in {time.time()-_tC:.0f}s")

            wt_llr_sum = unsummed_llr_wt.sum(dim=1)
            mt_llr_sum = unsummed_llr_mt.sum(dim=1)

            _n_mut = mut_mask.sum(dim=1)
            if self.adapter_mode == 'fused':
                cal_f = getattr(self, 'calibration_head_fused', None)
                wt_lora_pred = cal_f(wt_llr_sum, _n_mut) if cal_f is not None else wt_llr_sum
                mt_lora_pred = cal_f(mt_llr_sum, _n_mut) if cal_f is not None else mt_llr_sum
            else:
                cal_wt = getattr(self, 'calibration_head_wt', None)
                cal_mt = getattr(self, 'calibration_head_mt', None)
                wt_lora_pred = cal_wt(wt_llr_sum, _n_mut) if cal_wt is not None else wt_llr_sum
                mt_lora_pred = cal_mt(mt_llr_sum, _n_mut) if cal_mt is not None else mt_llr_sum

            if skip_reverse:
                mt_lora_pred = torch.full_like(wt_lora_pred, float('nan'))
                combined_pred = torch.full_like(wt_lora_pred, float('nan'))
            else:
                combined_pred = combine_predictions(wt_lora_pred, mt_lora_pred)

        return {'wt_lora_pred': wt_lora_pred, 'mt_lora_pred': mt_lora_pred, 'combined_pred': combined_pred}