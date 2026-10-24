import logging

class PEFTStateManager:
    """
    Abstracts PEFT requires_grad state subversion to prevent it from cluttering the Lightning module.
    """
    def __init__(self, model):
        self.model = model
        self.baseline_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        # Which parameters arrived trainable. The pretrained base is frozen before this manager ever sees it, so a
        # frozen base weight is the normal state and says nothing; only a parameter that was trainable and is now
        # frozen is something this manager did, and only that is worth naming when it matches no known component.
        self.initially_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.has_transitioned = False
        self.wt_path_is_frozen = False
        self.mt_path_is_frozen = False
        self._warned_unmatched = False

    def apply_baseline_requires_grad(self):
        """Forces all parameters to match the registered baseline requires_grad states."""
        for name, p in self.model.named_parameters():
            if name in self.baseline_requires_grad:
                p.requires_grad = self.baseline_requires_grad[name]

    def wt_parameter_names(self):
        """Names of the parameters that make up the WT (additive-predictor) path: its adapter, its calibration head,
        and the shared 'default' parameters. Single source of truth so freezing a head and loading a pre-trained
        one can never disagree about what 'the WT path' is."""
        wt_name = getattr(self.model, 'wt_adapter_name', 'wt_adapter')
        return [name for name, _ in self.model.named_parameters()
                if wt_name in name or 'calibration_head_wt' in name or 'calibration_head_fused' in name or 'default' in name]

    def freeze_wt_components(self):
        """Freezes the wild-type/additive predictor components."""
        for name in self.wt_parameter_names():
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
        
        unmatched = []
        for name, p in self.model.named_parameters():
            if not self.baseline_requires_grad.get(name, True):
                p.requires_grad = False
                if p.grad is not None: p.grad = None
                
                if 'calibration_head_wt' in name or 'calibration_head_fused' in name: frozen_groups.add('calib_wt')
                elif 'calibration_head_mt' in name: frozen_groups.add('calib_mt')
                elif wt_name in name: frozen_groups.add('lora_wt')
                elif mt_name in name: frozen_groups.add('lora_mt')
                elif 'default' in name: frozen_groups.add('lora_wt')
                elif name in self.initially_trainable: unmatched.append(name)

        # Once per run, and only for parameters this manager actually froze. This is called on every optimizer
        # step while anything is frozen, and it used to warn per tensor per step about the frozen pretrained base,
        # which is not a finding: a run that froze the WT path at epoch 4 wrote 1140 warnings per step, 528960 of
        # them before anyone noticed. A parameter that arrived trainable and matches no known component still
        # means its optimizer group cannot be identified, so its learning rate will not be zeroed - that is worth
        # exactly one line.
        if unmatched and not self._warned_unmatched:
            self._warned_unmatched = True
            shown = ', '.join(unmatched[:5]) + (f", ... (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else "")
            logging.warning(f"{len(unmatched)} parameter(s) were frozen but match no known adapter or calibration "
                            f"head naming convention, so their optimizer groups cannot be identified and their "
                            f"learning rates will not be zeroed: {shown}")

        if zero_lrs and frozen_groups and optimizers:
            opts = optimizers if isinstance(optimizers, list) else [optimizers]
            for opt in opts:
                for g in getattr(opt, 'optimizer', opt).param_groups:
                    if g.get("name") in frozen_groups:
                        g["lr"] = 0.0
                        g["weight_decay"] = 0.0