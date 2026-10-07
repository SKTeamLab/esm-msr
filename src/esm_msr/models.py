import logging
import time
import copy
import types
from collections import defaultdict
from typing import Dict, Any, Optional, Union
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from esm.pretrained import ESM3_sm_open_v0
from esm.models.esm3 import ESMOutput, OutputHeads
from esm.utils.constants import esm3 as C
from peft import LoraConfig, get_peft_model

from esm_msr import routing


def _sequence_only_output_heads(heads: OutputHeads, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
    """
    Drop-in replacement for ``OutputHeads.forward`` that evaluates only the sequence head.

    Stability scoring reads ``sequence_logits`` exclusively. The stock forward also
    runs the structure (4096), function (8x260), residue (1478), SS8 and SASA heads:
    ~2% of the FLOPs of a forward pass, but B x L x ~7.7k extra logits plus their
    autograd buffers that sit in memory until the output object is released.
    """
    return ESMOutput(
        sequence_logits=heads.sequence_head(x),
        structure_logits=None, secondary_structure_logits=None, sasa_logits=None,
        function_logits=None, residue_logits=None, embeddings=embed,
    )


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

    Wraps ESM3 with LoRA adapters and scores a mutation set as the summed log-likelihood
    ratio ``sum_i [logit(to_i) - logit(from_i)]`` at the mutated positions, mapped to ddG by
    a scalar ``CalibrationHead``.

    adapter_mode='dual' keeps two adapters over one shared, frozen backbone:

    * WT adapter (``peft_wt``) reads ``wt_sequence_tokens``: the real wild-type sequence
      on its real structure. Its domain is single mutations.
    * MT adapter (``peft_mt``) reads ``mt_sequence_tokens``: a sequence that already
      carries mutations, on a structure that is *not* that sequence's structure (the WT
      structure, possibly masked at mutated sites). Its domain is conditional effects
      ddG(i | other mutations present).

    See ``esm_msr.routing`` for which data subsets train which adapter and why
    0.5 * WT + 0.5 * MT is the thermodynamically correct estimate for multi-mutants.

    The backbone runs once per unique (sequence, structure) row of a batch and the logits of the
    duplicates are gathered (``dedup_backbone``, always on). All single and double items of one
    protein share their WT input, so the WT pass costs one forward per batch instead of one per
    mutation. Exact in eval; in training, rows that share an input also share one LoRA-dropout
    sample.

    Args (beyond the LoRA configuration):
        sequence_head_only: Skip ESM3's unused structure/function/residue/SS8/SASA heads.
        mask_structure: Blank the structure at every position the MT-pass sequence mutates
            relative to the structure (``struct_mut_pos``): coordinates to NaN and structure
            tokens to the mask token. The WT pass is never masked, because its sequence and
            its structure agree. Both channels must be blanked - ESM3 builds its affine
            frames from the coordinates and reads the tokens separately, so masking one
            leaves the other informative. This setting must match between training and
            inference, so it is recorded in hparams.yaml and re-applied from there.
    """
    dedup_backbone = True

    def __init__(
            self, lora_config: dict, shared_scale_init: float | None = None,
            shared_bias_init: float | None = None, inference_mode: bool = False, log_likelihood: bool = False,
            quaternary_mode: str = 'single_chain', model_dtype: torch.dtype = torch.bfloat16,
            adapter_mode: str = 'dual', strict_loading: bool = True,
            sequence_head_only: bool = True,
            mask_structure: bool = False,
        ):
        logging.info("Initializing ESM3 Base Model...")
        base_esm3 = ESM3_sm_open_v0()
        base_esm3.to(model_dtype)
        super().__init__(esm_model=base_esm3)
        
        self.quaternary_mode, self.log_likelihood, self.dtype = quaternary_mode, log_likelihood, model_dtype 
        self.adapter_mode = adapter_mode
        self.strict_loading = strict_loading
        self.mask_structure = mask_structure
        
        # 1. Initialize Calibration
        if shared_scale_init is not None or shared_bias_init is not None:
            if self.adapter_mode == 'fused':
                self.calibration_head_fused = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
            else:
                self.calibration_head_wt = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
                self.calibration_head_mt = CalibrationHead(init_scale=shared_scale_init, init_bias=shared_bias_init, requires_grad=not inference_mode)
        
        # 2. Add LoRAs using config
        logging.info(f"Injecting Adapters (Mode: {self.adapter_mode.upper()})...")
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
            mt_base = getattr(getattr(self, 'peft_mt', None), 'base_model', None)
            if wt_se is not None and mt_base is not None:
                mt_base._structure_encoder = wt_se

        # 3c. Only the sequence head is ever read; skip the other output heads.
        if sequence_head_only:
            for pm in (getattr(self, 'peft_wt', None), getattr(self, 'peft_mt', None), getattr(self, 'peft_fused', None)):
                if pm is None:
                    continue
                for m in pm.modules():
                    if isinstance(m, OutputHeads):
                        m.forward = types.MethodType(_sequence_only_output_heads, m)

        # 4. Optional freezing for strict inference
        if inference_mode:
            for name, p in self.named_parameters():
                if 'lora' in name or 'peft' in name: p.requires_grad = False

        # 5. Log final model statistics
        self._log_trainable_parameters()

    @property
    def peft_wt(self):
        """The WT-adapter wrapper. An alias for ``self.model``, not a second submodule."""
        return self.model

    @property
    def peft_fused(self):
        """The single wrapper in fused mode. An alias for ``self.model``."""
        return self.model

    def _create_lora_config(self, kwargs_dict: dict) -> LoraConfig:
        """Helper to dynamically construct a LoraConfig dictionary and object."""
        lora_rank = kwargs_dict.get('lora_rank', 6)
        lora_alpha = kwargs_dict.get('lora_alpha', 12)
        lora_dropout = kwargs_dict.get('lora_dropout', 0.15)
        target_mode = kwargs_dict.get('target_mode', 'expanded')
        last_n_layers = kwargs_dict.get('last_n_layers', 0)
        use_dora = kwargs_dict.get('use_dora', False)
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
        # The structure encoder is deliberately excluded. ESM3.forward never calls it -- it
        # consumes precomputed structure tokens -- so adapters placed there could never
        # receive gradient. Adapting it would require moving encoding into the forward pass.
        base_regex = f"^(?!.*structure_encoder).*transformer\\.blocks\\.{block_pattern}\\.({target_pattern})$"
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
            
            # Registered as self.model only. Assigning the same module to a second attribute
            # (self.peft_wt) would register it twice, so every WT parameter appeared twice in
            # named_parameters and in saved checkpoints. `peft_wt` is a read-only alias below.
            self.model = get_peft_model(self.model, wt_config, adapter_name="wt_adapter").to(dtype)
            self.peft_mt = get_peft_model(base_mt, mt_config, adapter_name="mt_adapter").to(dtype)
            
            # Re-share the exact underlying memory footprint AFTER .to(dtype) casting!
            def _clean_peft_name(n: str) -> str:
                return n.replace("base_model.model.", "").replace(".base_layer.", ".").replace(".original_module.", ".")

            wt_base_params = {_clean_peft_name(n): p for n, p in self.model.named_parameters() if "lora_" not in n and "dora_" not in n}
            for name_mt, p_mt in self.peft_mt.named_parameters():
                if "lora_" not in name_mt and "dora_" not in name_mt:
                    clean_name = _clean_peft_name(name_mt)
                    if clean_name in wt_base_params:
                        p_mt.data = wt_base_params[clean_name].data
            
            self.wt_adapter_name = "wt_adapter"
            self.mt_adapter_name = "mt_adapter"
            
            active_configs.extend([wt_kwargs, mt_kwargs])
            peft_models = [self.model, self.peft_mt]

        elif self.adapter_mode == 'fused':
            mt_kwargs = kwargs.get('mt_config', kwargs)

            config = self._create_lora_config(mt_kwargs)

            logging.info("--- Fused Adapter Configuration ---")
            logging.info(f" Fused Adapter | Rank: {config.r:2} | Alpha: {config.lora_alpha:2} | Dropout: {config.lora_dropout} | DoRA: {getattr(config, 'use_dora', False)}")
            logging.debug(f" Target Regex: {config.target_modules}")
            
            self.model = get_peft_model(self.model, config).to(dtype)
            self.wt_adapter_name = self.mt_adapter_name = list(self.model.peft_config.keys())[0]

            active_configs.append(mt_kwargs)
            peft_models = [self.model]

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

    def forward_batch(self, batch_in: Dict[str, Any], cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, skip_reverse: bool = False, mask_strategy: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Inference forward: runs the WT pass and (unless ``skip_reverse``) the MT pass and
        reports ``combined_pred`` as 0.5*WT + 0.5*MT for every item.

        ``epi_pred = 0.5*(MT - WT)`` is the implied pairwise epistasis for multi-mutants
        (meaningless for single-mutation items).
        """
        if self.training: raise AssertionError("forward_batch is for inference only. Use forward_partitioned for training.")

        wt_out = self.forward_partitioned(batch_in, pass_type='wt', cached_wt_esm3=cached_wt_esm3, mask_strategy=mask_strategy)
        if not skip_reverse:
            mt_out = self.forward_partitioned(batch_in, pass_type='mt', mask_strategy=mask_strategy)
        else:
            mt_out = wt_out

        wt_pred_cal, mt_pred_cal = wt_out['pred_calibrated'], mt_out['pred_calibrated']
        wt_pred_raw, mt_pred_raw = wt_out['pred_raw'], mt_out['pred_raw']

        if self.adapter_mode == 'fused':
            wt_cal = self.calibration_head_fused(wt_pred_raw)
            mt_cal = self.calibration_head_fused(mt_pred_raw)
        else:
            wt_cal, mt_cal = wt_pred_cal, mt_pred_cal

        # Always the two-path average, for every item type. This is the quantity the
        # architecture is built around (0.5*WT + 0.5*MT is exact for a double; see
        # esm_msr.routing), so it stays the reported prediction even where a single head
        # owns the item in training and would be used alone in practice.
        combined_pred = 0.5 * wt_cal + 0.5 * mt_cal

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

    @staticmethod
    def _blank_structure(coords, struct_tokens, pos, pos_mask):
        """
        Remove structural information at ``pos`` (1-based, padded, valid where ``pos_mask``).

        Coordinates go to NaN, which is how ESM3 marks an absent backbone frame, and structure
        tokens go to the mask token. Both are required: ESM3 builds its affine frames from the
        coordinates and reads the tokens through a separate embedding, so blanking one leaves
        the other fully informative.

        Returns new tensors and leaves the inputs untouched, because the same batch also feeds
        the unmasked WT pass. Collation emits coordinates as [B, L, A, 3] or [B, 1, L, A, 3]
        and tokens as [B, L] or [B, 1, L], so leading singleton axes are squeezed to put the
        residue axis second; ``_get_esm3_outputs`` accepts either form.
        """
        if pos is None or pos_mask is None or not bool(pos_mask.any()):
            return coords, struct_tokens

        def _residue_axis_second(t, ndim):
            while t.dim() > ndim and t.shape[1] == 1:
                t = t.squeeze(1)
            return t

        B = pos.shape[0]
        rows, cols = torch.where(pos_mask)
        p = pos[rows, cols]

        if torch.is_tensor(coords) and coords.dim() >= 3 and coords.shape[0] == B:
            coords = _residue_axis_second(coords, 4).clone()
            if int(p.max()) < coords.shape[1]:
                coords[rows, p] = float('nan')
            else:
                raise AssertionError(
                    f"struct_mut_pos max {int(p.max())} exceeds the coordinate residue axis "
                    f"({coords.shape[1]}); positions must be 1-based indices into the padded sequence.")
        if torch.is_tensor(struct_tokens) and struct_tokens.dim() >= 2 and struct_tokens.shape[0] == B:
            struct_tokens = _residue_axis_second(struct_tokens, 2).clone()
            if int(p.max()) < struct_tokens.shape[1]:
                struct_tokens[rows, p] = C.STRUCTURE_MASK_TOKEN
        return coords, struct_tokens

    def _active_model(self, pass_type: str) -> nn.Module:
        if getattr(self, 'adapter_mode', 'dual') == 'dual':
            return self.peft_wt if pass_type == 'wt' else self.peft_mt
        return self.peft_fused

    @staticmethod
    def _unique_rows(seq: torch.Tensor, coords: Optional[torch.Tensor], struct_tokens: Optional[torch.Tensor]):
        """
        Group batch rows with identical backbone inputs.

        Returns ``(first, inverse)``: ``first[u]`` is a representative row of unique input
        ``u`` and ``inverse[b]`` maps row ``b`` to its unique input. Sequence, structure
        tokens and coordinates (NaN/inf-safe) all enter the key.
        """
        B = seq.shape[0]
        parts = [seq.reshape(B, -1).float()]
        if torch.is_tensor(struct_tokens) and struct_tokens.dim() > 0 and struct_tokens.shape[0] == B:
            parts.append(struct_tokens.reshape(B, -1).float())
        if torch.is_tensor(coords) and coords.dim() > 0 and coords.shape[0] == B:
            parts.append(torch.nan_to_num(coords.reshape(B, -1).float(), nan=-7.7e7, posinf=8.8e7, neginf=-9.9e7))
        key = torch.cat(parts, dim=1)
        _, inverse = torch.unique(key, dim=0, return_inverse=True)
        n_unique = int(inverse.max().item()) + 1
        first = torch.full((n_unique,), B, dtype=torch.long, device=seq.device)
        first = first.scatter_reduce(0, inverse, torch.arange(B, device=seq.device), reduce='amin')
        return first, inverse

    def _backbone_logits(self, seq: torch.Tensor, coords, struct_tokens, plddt, active_model: nn.Module):
        """
        Sequence logits for every row of ``seq``, running the backbone once per unique
        input row when ``self.dedup_backbone`` is set.

        Returns ``(logits [U, L, V], row_index [B])``; logits for row ``b`` are
        ``logits[row_index[b]]``.
        """
        B = seq.shape[0]
        if self.dedup_backbone and B > 1:
            first, row_index = self._unique_rows(seq, coords, struct_tokens)
            if first.numel() < B:
                def _take(t):
                    return t[first] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
                out = self._get_esm3_outputs(seq[first], _take(coords), _take(struct_tokens), _take(plddt), active_model=active_model)
                return self._process_logits(out.sequence_logits.float()), row_index
        out = self._get_esm3_outputs(seq, coords, struct_tokens, plddt, active_model=active_model)
        return self._process_logits(out.sequence_logits.float()), torch.arange(B, device=seq.device)

    def forward_partitioned(self, batch: Dict[str, Any], pass_type: str, cached_wt_esm3: Optional[Dict[str, torch.Tensor]] = None, mask_strategy: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        One adapter pass over a batch.

        pass_type='wt' reads ``wt_sequence_tokens`` with the WT adapter; pass_type='mt' reads
        ``mt_sequence_tokens`` with the MT adapter. Both return per-item
        ``sum_i [logit(mt_id_i) - logit(wt_id_i)]`` at ``mut_pos`` (``pred_raw``), its
        calibrated value (``pred_calibrated``), and the per-mutation terms (``unsummed_llr``).
        ``wt_id``/``mt_id`` are the item's from/to residues, so for reversion-style items
        (e.g. reversion) "mt_id" is the wild-type residue.
        """
        if pass_type not in ['wt', 'mt']: raise AssertionError(f"pass_type must be 'wt' or 'mt'. Received: {pass_type}")

        seq, mut_pos = batch.get(f'{pass_type}_sequence_tokens'), batch.get('mut_pos')
        wt_id, mt_id, mut_mask = batch.get('wt_id'), batch.get('mt_id'), batch.get('mut_mask')
        coords, struct_tokens, plddt = batch.get('coords'), batch.get('structure_tokens'), batch.get('plddt')

        if pass_type == 'mt' and batch.get('mt_coords') is not None:
            # hidden BEFORE the structure encoder (data.premask_structure): the item carries the MT pass's own coordinates and re-encoded tokens
            coords, struct_tokens = batch['mt_coords'], batch['mt_structure_tokens']
        elif pass_type == 'mt' and getattr(self, 'mask_structure', False):
            smp = batch.get('struct_mut_pos')
            smm = batch.get('struct_mut_mask')
            if smp is None:
                smp, smm = mut_pos, mut_mask   # caches without the field: mask what we know
            coords, struct_tokens = self._blank_structure(coords, struct_tokens, smp, smm)

        B, max_muts = seq.shape[0], mut_pos.shape[1]
        safe_pos = mut_pos.masked_fill(~mut_mask, 0)
        safe_wt_id = wt_id.masked_fill(~mut_mask, 0)
        safe_mt_id = mt_id.masked_fill(~mut_mask, 0)
        b_idx = torch.arange(B, device=seq.device).unsqueeze(1).expand(-1, max_muts)

        if mask_strategy is not None:
            if mask_strategy == 'chain':
                mask_strategy = 'independent'  # legacy alias for old hparams.yaml
            if mask_strategy not in ['independent', 'marginal']:
                raise AssertionError(f"Invalid mask_strategy: '{mask_strategy}'. Expected None, 'independent', 'chain', or 'marginal'.")
            if cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used with mask_strategy='{mask_strategy}'. Each masking pass fundamentally alters the model sequence state.")

            mask_token_id = C.SEQUENCE_MASK_TOKEN
            active_model = self._active_model(pass_type)
            unsummed_llr = torch.zeros((B, max_muts), dtype=torch.float32, device=seq.device)

            if mask_strategy == 'independent':
                # One masked forward per mutation slot, restricted to the rows that have
                # a mutation in that slot.
                for i in range(max_muts):
                    rows = torch.where(mut_mask[:, i])[0]
                    if rows.numel() == 0:
                        continue
                    pos_valid = safe_pos[rows, i]
                    masked_seq = seq[rows].clone()
                    masked_seq[torch.arange(rows.numel(), device=seq.device), pos_valid] = mask_token_id

                    def _rows(t):
                        return t[rows] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
                    logits, row_index = self._backbone_logits(masked_seq, _rows(coords), _rows(struct_tokens), _rows(plddt), active_model)
                    mt_logits = logits[row_index, pos_valid, safe_mt_id[rows, i]]
                    wt_logits = logits[row_index, pos_valid, safe_wt_id[rows, i]]
                    unsummed_llr[rows, i] = mt_logits - wt_logits

            elif mask_strategy == 'marginal':
                masked_seq = seq.clone()
                masked_seq[b_idx[mut_mask], safe_pos[mut_mask]] = mask_token_id
                logits, row_index = self._backbone_logits(masked_seq, coords, struct_tokens, plddt, active_model)
                r = row_index[b_idx]
                mt_logits, wt_logits = logits[r, safe_pos, safe_mt_id], logits[r, safe_pos, safe_wt_id]
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
                logits, row_index = cached_wt_esm3['logits'].expand(B, -1, -1), torch.arange(B, device=seq.device)
            else:
                logits, row_index = self._backbone_logits(seq, coords, struct_tokens, plddt, self._active_model(pass_type))

            # Both passes compute logit(to) - logit(from); they differ only in which
            # sequence (and adapter) provides the context.
            r = row_index[b_idx]
            mt_logits, wt_logits = logits[r, safe_pos, safe_mt_id], logits[r, safe_pos, safe_wt_id]
            unsummed_llr = torch.where(mut_mask, mt_logits - wt_logits, torch.zeros_like(mt_logits))

        # --- Shared Post-Processing & Calibration ---
        llr_sum_raw = unsummed_llr.sum(dim=1)

        if self.adapter_mode == 'fused':
            head = getattr(self, 'calibration_head_fused', None)
        else:
            head = getattr(self, f'calibration_head_{pass_type}', None)
        llr_sum_cal = head(llr_sum_raw) if head is not None else llr_sum_raw

        output_dict = {'pred_calibrated': llr_sum_cal, 'pred_raw': llr_sum_raw, 'unsummed_llr': unsummed_llr}
        return output_dict

    @torch.no_grad()
    def score_screening_batch(self, wt_sequence_tokens: torch.Tensor, mut_pos: torch.Tensor, wt_id: torch.Tensor, mt_id: torch.Tensor, mut_mask: torch.Tensor, coords: Optional[torch.Tensor] = None, structure_tokens: Optional[torch.Tensor] = None, plddt: Optional[torch.Tensor] = None, mask_strategy: Optional[str] = None, batch_size: int = 32, skip_reverse: bool = False, cached_wt_esm3: Optional[Dict] = None, quiet: bool = False, auto_batch: Optional["AutoBatchSizer"] = None) -> Dict[str, torch.Tensor]:
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

        if mask_strategy is None:
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
                    # screening scores against the WT structure, so every mutated position is
                    # a structure mismatch
                    'struct_mut_pos': mut_pos[start_idx:end_idx],
                    'struct_mut_mask': mut_mask[start_idx:end_idx],
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
            # ROUTE 2: State Deduplication for Masked (Resolves combinatorial explosion)
            if mask_strategy == 'chain':
                mask_strategy = 'independent'  # legacy alias for old hparams.yaml
            if mask_strategy not in ['independent', 'marginal']:
                raise AssertionError(f"Invalid mask_strategy: '{mask_strategy}'. Expected 'independent', 'chain', 'marginal', or None.")
            if cached_wt_esm3 is not None:
                raise NotImplementedError(f"cached_wt_esm3 cannot be used with mask_strategy='{mask_strategy}'.")
            
            mask_token_id = C.SEQUENCE_MASK_TOKEN
            wt_state_reqs = defaultdict(set)
            mt_state_reqs = defaultdict(set)
            state_map = defaultdict(dict)

            base_wt = wt_sequence_tokens.squeeze(0)

            _tA = time.time()
            _hb(f"    [PHASE-A] start B={B} L={wt_sequence_tokens.shape[1]}")
            for b in tqdm(range(B), desc='Constructing efficient masked batches to evaluate', disable=quiet):
                if b and b % 100_000 == 0:
                    _hb(f"    [PHASE-A] {b}/{B} ({time.time()-_tA:.0f}s)")
                valid_indices = torch.where(mut_mask[b])[0].tolist()
                if not valid_indices: continue
                
                # Construct base MT purely from sparse indices
                base_mt = base_wt.clone()
                for i in valid_indices:
                    base_mt[mut_pos[b, i]] = mt_id[b, i]
                
                if mask_strategy == 'independent':
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

            _hb(f"    [PHASE-A] done in {time.time()-_tA:.0f}s: {len(wt_state_reqs)} unique wt states, {len(mt_state_reqs)} unique mt states")

            def compute_cache(states_reqs, active_model, label):
                cache = defaultdict(dict)
                states_list = list(states_reqs.keys())
                done, t0, next_hb = 0, time.time(), 100_000

                def _state_chunk(start_idx, end_idx):
                    nonlocal done, next_hb
                    batch_tuples = states_list[start_idx:end_idx]
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
                    done += len(batch_tuples)
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
            wt_cache = compute_cache(wt_state_reqs, self.peft_wt if is_dual else self.peft_fused, label=f'wt-L{_L}')

            if not skip_reverse:
                if is_dual:
                    mt_cache = compute_cache(mt_state_reqs, self.peft_mt, label=f'mt-L{_L}')
                else:
                    all_reqs = defaultdict(set)
                    for tup, poses in wt_state_reqs.items(): all_reqs[tup].update(poses)
                    for tup, poses in mt_state_reqs.items(): all_reqs[tup].update(poses)
                    mt_cache = wt_cache = compute_cache(all_reqs, self.peft_fused, label=f'fused-L{_L}')

            unsummed_llr_wt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)
            unsummed_llr_mt = torch.zeros((B, max_muts), dtype=torch.float32, device=device)

            _tC = time.time()
            _hb(f"    [COLLATE] start B={B}")
            for b in tqdm(range(B), desc='Collating results', disable=quiet):
                if b and b % 100_000 == 0:
                    _hb(f"    [COLLATE] {b}/{B} ({time.time()-_tC:.0f}s)")
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

            _hb(f"    [COLLATE] done in {time.time()-_tC:.0f}s")

            wt_llr_sum = unsummed_llr_wt.sum(dim=1)
            mt_llr_sum = unsummed_llr_mt.sum(dim=1)

            if self.adapter_mode == 'fused':
                wt_lora_pred = self.calibration_head_fused(wt_llr_sum)
                mt_lora_pred = self.calibration_head_fused(mt_llr_sum)
            else:
                wt_lora_pred = self.calibration_head_wt(wt_llr_sum) if hasattr(self, 'calibration_head_wt') else wt_llr_sum
                mt_lora_pred = self.calibration_head_mt(mt_llr_sum) if hasattr(self, 'calibration_head_mt') else mt_llr_sum

            combined_pred = 0.5 * wt_lora_pred + 0.5 * mt_lora_pred

        return {'wt_lora_pred': wt_lora_pred, 'mt_lora_pred': mt_lora_pred, 'combined_pred': combined_pred}