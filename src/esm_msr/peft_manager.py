import logging

class PEFTStateManager:
    """
    Abstracts PEFT requires_grad state subversion to prevent it from cluttering the Lightning module.
    """
    def __init__(self, model):
        self.model = model
        self.baseline_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        self.has_transitioned = False
        self.wt_path_is_frozen = False
        self.mt_path_is_frozen = False

    def apply_baseline_requires_grad(self):
        """Forces all parameters to match the registered baseline requires_grad states."""
        for name, p in self.model.named_parameters():
            if name in self.baseline_requires_grad:
                p.requires_grad = self.baseline_requires_grad[name]

    def freeze_wt_components(self):
        """Freezes the wild-type/additive predictor components."""
        wt_name = getattr(self.model, 'wt_adapter_name', 'wt_adapter')
        for name, _ in self.model.named_parameters():
            if wt_name in name or 'calibration_head_wt' in name or 'calibration_head_fused' in name or 'default' in name:
                self.baseline_requires_grad[name] = False
        self.apply_baseline_requires_grad()
        print('WT components frozen')
        self.wt_path_is_frozen = True
        self.model._log_trainable_parameters()

    def freeze_mt_components(self):
        """Freezes the mutant/corrector predictor components."""
        mt_name = getattr(self.model, 'mt_adapter_name', 'mt_adapter')
        for name, _ in self.model.named_parameters():
            if mt_name in name or 'calibration_head_mt' in name or 'calibration_head_fused' in name or 'default' in name:
                self.baseline_requires_grad[name] = False
        self.apply_baseline_requires_grad()
        print('MT components frozen')
        self.mt_path_is_frozen = True
        self.model._log_trainable_parameters()

    def unfreeze_mt_components(self):
        """Unfreezes the mutant/corrector predictor components."""
        mt_name = getattr(self.model, 'mt_adapter_name', 'mt_adapter')
        for name, _ in self.model.named_parameters():
            if mt_name in name or 'calibration_head_mt' in name or 'calibration_head_fused' in name or 'default' in name:
                self.baseline_requires_grad[name] = True
        self.apply_baseline_requires_grad()
        print('MT components thawed')
        self.mt_path_is_frozen = False
        self.model._log_trainable_parameters()

    def enforce_freezing(self, optimizers, zero_lrs=True):
        """Zeroes out gradients and learning rates dynamically for cleanly frozen adapters."""
        frozen_groups = set()
        wt_name = getattr(self.model, 'wt_adapter_name', 'wt_adapter')
        mt_name = getattr(self.model, 'mt_adapter_name', 'mt_adapter')
        
        for name, p in self.model.named_parameters():
            if not self.baseline_requires_grad.get(name, True):
                p.requires_grad = False
                if p.grad is not None: p.grad = None
                
                if 'calibration_head_wt' in name: frozen_groups.add('calib_wt')
                elif 'calibration_head_mt' in name: frozen_groups.add('calib_mt')
                elif wt_name in name: frozen_groups.add('lora_wt')
                elif mt_name in name: frozen_groups.add('lora_mt')
                elif 'default' in name: frozen_groups.add('lora_wt')
                else: logging.warning(f"Parameter {name} is frozen but does not match expected adapter or calibration head naming conventions.")
                    
        if zero_lrs and frozen_groups and optimizers:
            opts = optimizers if isinstance(optimizers, list) else [optimizers]
            for opt in opts:
                for g in getattr(opt, 'optimizer', opt).param_groups:
                    if g.get("name") in frozen_groups:
                        g["lr"] = 0.0
                        g["weight_decay"] = 0.0