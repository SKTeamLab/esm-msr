import logging
import copy
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.constants import esm3 as C
from peft import LoraConfig, get_peft_model


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


class CalibrationHead(nn.Module):
    """ Scales and biases raw log-likelihood ratios: y_cal = scale * y_raw + bias """
    def __init__(self, init_scale: float | None = 1/3, init_bias: float | None = 0.0, *, min_scale: float = 1e-4, beta: float = 1.0, max_scale: float | None = None, requires_grad: bool = True):
        super().__init__()
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
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32), requires_grad=requires_grad)

        self.min_scale, self.beta, self.max_scale = float(min_scale), float(beta), float(max_scale) if max_scale is not None else None

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
        s = F.softplus(self.raw_scale, beta=self.beta) + self.min_scale
        return torch.clamp(s, max=self.max_scale) if self.max_scale is not None else s

    def forward(self, y_raw: torch.Tensor) -> torch.Tensor:
        s = 1.0 if not self.use_scale else self.scale
        b = 0.0 if self.bias is None else self.bias
        return y_raw * s + b
    

class MSRModel(ESM3PredictorBase):
    """ 
    Mutational Stability Regression (MSR) Model.
    
    Wraps an ESM3 model with Parameter-Efficient Fine-Tuning (PEFT) adapters to predict 
    protein stability scores (e.g., ddG). Supports dual/fused adapters and multiple 
    strategies (ensemble, corrector).
    """
    def __init__(
            self, lora_config: dict, shared_scale_init: float | None = None,
            shared_bias_init: float | None = None, inference_mode: bool = False, log_likelihood: bool = False,
            use_plddt: bool = False, quaternary_mode: str = 'single_chain', model_dtype: torch.dtype = torch.bfloat16, 
            adapter_mode: str = 'dual', lora_mode: str = 'ensemble', strict_loading: bool = True
        ):
        logging.info("Initializing ESM3 Base Model...")
        base_esm3 = ESM3_sm_open_v0()
        base_esm3.to(model_dtype)
        super().__init__(esm_model=base_esm3)
        
        if lora_mode not in ['ensemble', 'corrector']:
            raise AssertionError(f"Unknown lora_mode '{lora_mode}'. Must be 'ensemble' or 'corrector'.")
            
        self.quaternary_mode, self.log_likelihood, self.use_plddt, self.dtype = quaternary_mode, log_likelihood, use_plddt, model_dtype 
        self.adapter_mode, self.lora_mode = adapter_mode, lora_mode
        self.strict_loading = strict_loading
        
        # 1. Initialize Calibration
        if shared_scale_init is not None or shared_bias_init is not None:
            if self.adapter_mode == 'fused':
                self.calibration_head_fused = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
            else:
                self.calibration_head_wt = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
                self.calibration_head_mt = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
        
        # 2. Add LoRAs using config
        logging.info(f"Injecting Adapters (Mode: {self.adapter_mode.upper()} | Strategy: {self.lora_mode.upper()})...")
        self.lora_config = lora_config
        self.add_loras_to_esm3(**self.lora_config)
        
        # 3. Handle structure encoder dtype constraints
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, '_structure_encoder'):
            self.model.base_model._structure_encoder.to(torch.float32)
        elif hasattr(self.model, '_structure_encoder'):
            self.model._structure_encoder.to(torch.float32)

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
        use_dora = kwargs_dict.get('use_dora', False)
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
        if use_dora: config_dict['use_dora'] = True

        return LoraConfig(**config_dict)

    def add_loras_to_esm3(self, **kwargs):
        """
        Instantiates PEFT LoRA adapters for the ESM3 model. 
        Supports independent instantiation of WT and MT adapters using nested 
        'wt_config' and 'mt_config' dictionaries inside the primary config.
        """
        seed, dtype = kwargs.get('seed', None), kwargs.get('dtype', self.dtype)
        TOTAL_BLOCKS = 48
            
        for param in self.model.parameters(): param.requires_grad = False
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        active_configs = []

        if self.adapter_mode == 'dual':
            wt_kwargs = kwargs.get('wt_config', kwargs)
            mt_kwargs = kwargs.get('mt_config', kwargs)
            
            wt_config = self._create_lora_config(wt_kwargs)
            mt_config = self._create_lora_config(mt_kwargs)
            
            logging.info("--- Dual Adapter Configuration ---")
            logging.info(f" WT Adapter | Rank: {wt_config.r:2} | Alpha: {wt_config.lora_alpha:2} | Dropout: {wt_config.lora_dropout} | DoRA: {getattr(wt_config, 'use_dora', False)}")
            logging.info(f" MT Adapter | Rank: {mt_config.r:2} | Alpha: {mt_config.lora_alpha:2} | Dropout: {mt_config.lora_dropout} | DoRA: {getattr(mt_config, 'use_dora', False)}")
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
            
            active_configs.extend([wt_kwargs, mt_kwargs])
            peft_models = [self.peft_wt, self.peft_mt]
            
            # Maintain self.model pointing to peft_wt for backwards compatibility
            self.model = self.peft_wt

        elif self.adapter_mode == 'fused':
            mt_kwargs = kwargs.get('mt_config', kwargs)

            config = self._create_lora_config(mt_kwargs)

            logging.info("--- Fused Adapter Configuration ---")
            logging.info(f" Fused Adapter | Rank: {config.r:2} | Alpha: {config.lora_alpha:2} | Dropout: {config.lora_dropout} | DoRA: {getattr(config, 'use_dora', False)}")
            logging.debug(f" Target Regex: {config.target_modules}")
            
            self.peft_fused = get_peft_model(self.model, config).to(dtype)
            self.wt_adapter_name = self.mt_adapter_name = list(self.peft_fused.peft_config.keys())[0]
            
            active_configs.append(mt_kwargs)
            peft_models = [self.peft_fused]
            self.model = self.peft_fused

        # Apply requires_grad correctly via independent PEFT wrappers
        for pm, cfg in zip(peft_models, active_configs):
            for name, param in pm.named_parameters():
                if "lora" in name.lower() or "dora" in name.lower(): param.requires_grad = True

            if cfg.get('unfreeze_layernorms', False):
                ln_last_n = cfg.get('last_n_layers', 0)
                for name, param in pm.named_parameters():
                    if any(ln_name in name for ln_name in ["layernorm_qkv.0", "q_ln", "k_ln", "s_norm", "transformer.norm"]):
                        if ln_last_n <= 0 or any(f"blocks.{b}." in name for b in [str(i) for i in range(TOTAL_BLOCKS - ln_last_n, TOTAL_BLOCKS)]):
                             param.requires_grad = True

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

        # Process the checkpoints
        if isinstance(checkpoint_path, str):
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            process_state_dict(ckpt.get('state_dict', ckpt))
        elif isinstance(checkpoint_path, dict):
            for adapter_target, path in checkpoint_path.items():
                ckpt = torch.load(path, map_location='cpu')
                process_state_dict(ckpt.get('state_dict', ckpt), override_target=adapter_target)

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
        #logging.info(f"Loaded Parameters: {len(loaded_trainable)}")
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
                raise KeyError('Expected parameters were missing from the LoRA, suggesing that the wrong configuration was used.')

        if unexpected:
            logging.error(f"Unexpected Tensors in Checkpoint (Ignored): {len(unexpected)}")
            print(unexpected)

        logging.info("=== Checkpoint Loading Complete ===\n")

    def forward_batch(self, batch_in: Dict[str, Any], cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, skip_reverse: bool = False, mask_strategy: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if self.training: raise AssertionError("forward_batch is for inference only. Use forward_partitioned for training.")

        wt_out = self.forward_partitioned(batch_in, pass_type='wt', cached_wt_esm3=cached_wt_esm3, mask_strategy=mask_strategy)
        if not skip_reverse:
            mt_out = self.forward_partitioned(batch_in, pass_type='mt', mask_strategy=mask_strategy)
        else:
            mt_out = wt_out
        
        wt_pred_cal, mt_pred_cal = wt_out['pred_calibrated'], mt_out['pred_calibrated']
        wt_pred_raw, mt_pred_raw = wt_out['pred_raw'], mt_out['pred_raw']

        if self.adapter_mode == 'fused':
            combined_pred = 0.5 * self.calibration_head_fused(wt_pred_raw) + 0.5 * self.calibration_head_fused(mt_pred_raw)
        else:
            combined_pred = 0.5 * wt_pred_cal + 0.5 * mt_pred_cal

        epi_pred = 0.5 * mt_pred_cal - 0.5 * wt_pred_cal

        if skip_reverse:
            mt_pred_raw, mt_pred_cal, combined_pred, epi_pred = float('nan'), float('nan'), float('nan'), float('nan')

        return {'wt_lora_pred': wt_pred_cal, 'mt_lora_pred': mt_pred_cal, 'wt_lora_raw': wt_pred_raw, 'mt_lora_raw': mt_pred_raw, 'combined_pred': combined_pred, 'epi_pred': epi_pred}

    def _process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.log_likelihood: return logits
        idx = self.canonical_idx_tensor
        log_probs_canonical = torch.nn.functional.log_softmax(logits[:, :, idx], dim=-1)
        full_log_probs = torch.full_like(logits, float('-inf'))
        full_log_probs[:, :, idx] = log_probs_canonical
        return full_log_probs

    def forward_partitioned(self, batch: Dict[str, Any], pass_type: str, cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, mask_strategy: Optional[str] = None, detach_calibration: bool = False) -> Dict[str, torch.Tensor]:
        if pass_type not in ['wt', 'mt']: raise AssertionError(f"pass_type must be 'wt' or 'mt'. Received: {pass_type}")

        seq, mut_pos = batch.get(f'{pass_type}_sequence_tokens'), batch.get('mut_pos')
        wt_id, mt_id, mut_mask = batch.get('wt_id'), batch.get('mt_id'), batch.get('mut_mask')
        coords, struct_tokens, plddt = batch.get('coords'), batch.get('structure_tokens'), batch.get('plddt')

        B, max_muts = seq.shape[0], mut_pos.shape[1]
        safe_pos = mut_pos.clone(); safe_pos[~mut_mask] = 0
        safe_wt_id = wt_id.clone(); safe_wt_id[~mut_mask] = 0
        safe_mt_id = mt_id.clone(); safe_mt_id[~mut_mask] = 0
        b_idx = torch.arange(B, device=seq.device).unsqueeze(1).expand(-1, max_muts)

        if mask_strategy is not None:
            if mask_strategy not in ['chain', 'marginal']:
                raise AssertionError(f"Invalid mask_strategy: '{mask_strategy}'. Expected None, 'chain', or 'marginal'.")
            if cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used with mask_strategy='{mask_strategy}'. Each masking pass fundamentally alters the model sequence state.")
            
            mask_token_id = C.SEQUENCE_MASK_TOKEN
            
            if getattr(self, 'adapter_mode', 'dual') == 'dual':
                active_model = self.peft_wt if pass_type == 'wt' else self.peft_mt
            else:
                active_model = self.peft_fused
                
            unsummed_llr = torch.zeros((B, max_muts), dtype=torch.float32, device=seq.device)
            
            if mask_strategy == 'chain':
                for i in range(max_muts):
                    curr_mask = mut_mask[:, i]
                    if not curr_mask.any():
                        continue
                    
                    masked_seq = seq.clone()
                    batch_idx_valid = torch.where(curr_mask)[0]
                    pos_valid = safe_pos[curr_mask, i]
                    
                    # Apply the mask locally
                    masked_seq[batch_idx_valid, pos_valid] = mask_token_id
                    
                    out = self._get_esm3_outputs(masked_seq, coords, struct_tokens, plddt, active_model=active_model)
                    step_logits_B = self._process_logits(out.sequence_logits.float())
                    
                    # Regardless of WT or MT pass, logic calculates MT - WT 
                    # (Equivalent to -(WT - MT) used originally for MT pass)
                    mt_logits = step_logits_B[batch_idx_valid, pos_valid, safe_mt_id[curr_mask, i]]
                    wt_logits = step_logits_B[batch_idx_valid, pos_valid, safe_wt_id[curr_mask, i]]
                    unsummed_llr[batch_idx_valid, i] = mt_logits - wt_logits

            elif mask_strategy == 'marginal':
                masked_seq = seq.clone()
                
                # Apply all masks simultaneously
                for i in range(max_muts):
                    curr_mask = mut_mask[:, i]
                    batch_idx_valid = torch.where(curr_mask)[0]
                    pos_valid = safe_pos[curr_mask, i]
                    masked_seq[batch_idx_valid, pos_valid] = mask_token_id
                
                out = self._get_esm3_outputs(masked_seq, coords, struct_tokens, plddt, active_model=active_model)
                step_logits_B = self._process_logits(out.sequence_logits.float())
                
                mt_logits, wt_logits = step_logits_B[b_idx, safe_pos, safe_mt_id], step_logits_B[b_idx, safe_pos, safe_wt_id]
                unsummed_llr = torch.where(mut_mask, mt_logits - wt_logits, torch.zeros_like(mt_logits))

        else:
            if pass_type == 'wt' and cached_wt_esm3 is not None:
                if seq.shape != cached_wt_esm3['seq'].shape or not torch.equal(seq[0], cached_wt_esm3['seq'][0]):
                    logging.info(seq[0])
                    logging.info(cached_wt_esm3['seq'][0])
                    raise AssertionError("Sequence mismatch in cache.")
                logits_B = cached_wt_esm3['logits'].expand(B, -1, -1)
            else:
                if getattr(self, 'adapter_mode', 'dual') == 'dual':
                    active_model = self.peft_wt if pass_type == 'wt' else self.peft_mt
                else:
                    active_model = self.peft_fused

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
        llr_sum_for_cal = llr_sum_raw.detach() if detach_calibration else llr_sum_raw
        
        if pass_type == 'wt':
            llr_sum_cal = self.calibration_head_fused(llr_sum_for_cal) if self.adapter_mode == 'fused' and hasattr(self, 'calibration_head_fused') else (self.calibration_head_wt(llr_sum_for_cal) if hasattr(self, 'calibration_head_wt') else llr_sum_raw)
        elif pass_type == 'mt':
            llr_sum_cal = self.calibration_head_fused(llr_sum_for_cal) if self.adapter_mode == 'fused' and hasattr(self, 'calibration_head_fused') else (self.calibration_head_mt(llr_sum_for_cal) if hasattr(self, 'calibration_head_mt') else llr_sum_raw)

        output_dict = {'pred_calibrated': llr_sum_cal, 'pred_raw': llr_sum_raw, 'unsummed_llr': unsummed_llr}
        return output_dict

    @torch.no_grad()
    def score_screening_batch(self, wt_sequence_tokens: torch.Tensor, mut_pos: torch.Tensor, wt_id: torch.Tensor, mt_id: torch.Tensor, mut_mask: torch.Tensor, coords: Optional[torch.Tensor] = None, structure_tokens: Optional[torch.Tensor] = None, plddt: Optional[torch.Tensor] = None, mask_strategy: Optional[str] = None, batch_size: int = 32, skip_reverse: bool = False, cached_wt_esm3: Optional[Dict] = None, quiet: bool = False) -> Dict[str, torch.Tensor]:
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
        
        wt_lora_pred = torch.zeros(B, dtype=torch.float32, device=device)
        mt_lora_pred = torch.zeros(B, dtype=torch.float32, device=device)
        combined_pred = torch.zeros(B, dtype=torch.float32, device=device)

        if mask_strategy is None:
            # ROUTE 1: Dense Chunking for Unmasked (Maximum GPU Saturation, No Hashing Overhead)
            for start_idx in tqdm(range(0, B, batch_size), desc='Computing dense unmasked mutants', disable=quiet):
                end_idx = min(start_idx + batch_size, B)
                curr_B = end_idx - start_idx

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

        else:
            # ROUTE 2: State Deduplication for Masked (Resolves combinatorial explosion)
            if mask_strategy not in ['chain', 'marginal']:
                raise AssertionError(f"Invalid mask_strategy: '{mask_strategy}'. Expected 'chain', 'marginal', or None.")
            if cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used with mask_strategy='{mask_strategy}'.")
            
            mask_token_id = C.SEQUENCE_MASK_TOKEN
            wt_state_reqs = defaultdict(set)
            mt_state_reqs = defaultdict(set)
            state_map = defaultdict(dict)

            base_wt = wt_sequence_tokens.squeeze(0)

            for b in tqdm(range(B), desc='Constructing efficient masked batches to evaluate', disable=quiet):
                valid_indices = torch.where(mut_mask[b])[0].tolist()
                if not valid_indices: continue
                
                # Construct base MT purely from sparse indices
                base_mt = base_wt.clone()
                for i in valid_indices:
                    base_mt[mut_pos[b, i]] = mt_id[b, i]
                
                if mask_strategy == 'chain':
                    for i in valid_indices:
                        pos = mut_pos[b, i].item()
                        
                        wt_state_t = base_wt.clone()
                        wt_state_t[pos] = mask_token_id
                        wt_tup = tuple(wt_state_t.tolist())
                        wt_state_reqs[wt_tup].add(pos)
                        
                        mt_state_t = base_mt.clone()
                        mt_state_t[pos] = mask_token_id
                        mt_tup = tuple(mt_state_t.tolist())
                        mt_state_reqs[mt_tup].add(pos)
                        
                        state_map[b][i] = (wt_tup, mt_tup)

                elif mask_strategy == 'marginal':
                    all_positions = mut_pos[b, valid_indices]
                    
                    wt_state_t = base_wt.clone()
                    wt_state_t[all_positions] = mask_token_id
                    wt_tup = tuple(wt_state_t.tolist())
                    
                    mt_state_t = base_mt.clone()
                    mt_state_t[all_positions] = mask_token_id
                    mt_tup = tuple(mt_state_t.tolist())
                    
                    for i in valid_indices:
                        pos = mut_pos[b, i].item()
                        wt_state_reqs[wt_tup].add(pos)
                        mt_state_reqs[mt_tup].add(pos)
                        state_map[b][i] = (wt_tup, mt_tup)

            def compute_cache(states_reqs, active_model):
                cache = defaultdict(dict)
                states_list = list(states_reqs.keys())
                for start_idx in tqdm(range(0, len(states_list), batch_size), desc='Computing cache', disable=quiet):
                    batch_tuples = states_list[start_idx:start_idx+batch_size]
                    batch_tensor = torch.tensor(batch_tuples, dtype=torch.long, device=device)
                    
                    curr_b = batch_tensor.shape[0]
                    b_coords = coords.expand(curr_b, -1, -1, -1) if coords is not None else None
                    b_struct = structure_tokens.expand(curr_b, -1) if structure_tokens is not None else None
                    b_plddt = plddt.expand(curr_b, -1) if plddt is not None else None

                    out = self._get_esm3_outputs(batch_tensor, b_coords, b_struct, b_plddt, active_model=active_model)
                    logits = self._process_logits(out.sequence_logits.float())

                    for j, state_tuple in enumerate(batch_tuples):
                        for p in states_reqs[state_tuple]:
                            cache[state_tuple][p] = logits[j, p, :].clone()
                return cache

            is_dual = getattr(self, 'adapter_mode', 'dual') == 'dual'
            wt_cache = compute_cache(wt_state_reqs, self.peft_wt if is_dual else self.peft_fused)
            
            if not skip_reverse:
                if is_dual:
                    mt_cache = compute_cache(mt_state_reqs, self.peft_mt)
                else:
                    all_reqs = defaultdict(set)
                    for tup, poses in wt_state_reqs.items(): all_reqs[tup].update(poses)
                    for tup, poses in mt_state_reqs.items(): all_reqs[tup].update(poses)
                    mt_cache = wt_cache = compute_cache(all_reqs, self.peft_fused)

            unsummed_llr_wt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)
            unsummed_llr_mt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)

            for b in tqdm(range(B), desc='Collating results', disable=quiet):
                valid_indices = torch.where(mut_mask[b])[0].tolist()
                for i in valid_indices:
                    wt_tup, mt_tup = state_map[b][i]
                    pos = mut_pos[b, i].item()
                    w_id, m_id = wt_id[b, i], mt_id[b, i]

                    wt_pass_mt_logit = wt_cache[wt_tup][pos][m_id]
                    wt_pass_wt_logit = wt_cache[wt_tup][pos][w_id]
                    unsummed_llr_wt[b, i] = wt_pass_mt_logit - wt_pass_wt_logit

                    if not skip_reverse:
                        mt_pass_wt_logit = mt_cache[mt_tup][pos][w_id]
                        mt_pass_mt_logit = mt_cache[mt_tup][pos][m_id]
                        unsummed_llr_mt[b, i] = mt_pass_mt_logit - mt_pass_wt_logit
                    else:
                        unsummed_llr_mt[b, i] = unsummed_llr_wt[b, i]

            wt_llr_sum = unsummed_llr_wt.sum(dim=1)
            mt_llr_sum = unsummed_llr_mt.sum(dim=1)

            if self.adapter_mode == 'fused':
                wt_lora_pred = self.calibration_head_fused(wt_llr_sum)
                mt_lora_pred = self.calibration_head_fused(mt_llr_sum)
            else:
                wt_lora_pred = self.calibration_head_wt(wt_llr_sum) if hasattr(self, 'calibration_head_wt') else wt_llr_sum
                mt_lora_pred = self.calibration_head_mt(mt_llr_sum) if hasattr(self, 'calibration_head_mt') else mt_llr_sum

            combined_pred = 0.5 * wt_lora_pred + 0.5 * mt_lora_pred if self.adapter_mode != 'fused' else 0.5 * self.calibration_head_fused(wt_llr_sum) + 0.5 * self.calibration_head_fused(mt_llr_sum)

        return {'wt_lora_pred': wt_lora_pred, 'mt_lora_pred': mt_lora_pred, 'combined_pred': combined_pred}