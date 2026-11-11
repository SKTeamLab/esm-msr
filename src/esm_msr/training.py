import json
import math
import types
import os
import logging
import warnings
from collections import defaultdict
import gc

import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import CometLogger, CSVLogger

from esm_msr.models import MSRModel, combine_predictions
from esm_msr import blocks, censoring, config, epi_metrics, stats, utils, val_stats
from esm_msr.losses import ListMLELoss, tobit_posterior_mean, tobit_regression_loss, TOBIT_SIGMA_Y, TOBIT_MU_F, TOBIT_SIGMA_F, TF_PRIOR_SD
from esm_msr.preprocess_megascale import setup_dataloaders
from esm_msr.config import parse_arguments, monitor_metric, convergence_metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high')

# The LoRA spec that is not a flag (config.LORA_TARGET_MODE and the rest). inference.parse_hparams_to_lora_config uses the same values.
LORA_CONSTANTS = dict(target_mode=config.LORA_TARGET_MODE, last_n_layers=0, incl_structure_encoder=False, use_dora=False, unfreeze_layernorms=False)
SHARED_SCALE_INIT, SHARED_BIAS_INIT = 0.3, 0.0       # the calibration heads' starting scale and bias
SAVE_TOP_K = 3
# command-line values that say where things are, not what the run is: kept out of hparams.yaml (and so out of Comet)
UNLOGGED_ARGS = ('log_dir', 'cache_path', 'comet', 'comet_project_name', 'comet_offline', 'comet_experiment_key')


def validation_regimes(hp) -> list:
    """The regimes validation reads, ``[(prefix, {'wt': ..., 'mt': ...}), ...]``, the main one first with prefix ''. The main regime masks every mutated site on its own for both adapters
    (the partner's mutant residue visible: the readout block training's masked passes train), or simultaneously on the MT pass under ``--block_marginal_mt``. A run in which either head
    reads its site visible with some probability (``--block_mask_wt`` / ``--block_mask_mt`` below 1) also scores the unmasked regime (neither adapter masks), logged under ``val_unmasked_*``."""
    mt_masked = 'marginal' if bool(hp.get('block_marginal_mt', False)) else 'independent'
    regimes = [('', {'wt': 'independent', 'mt': mt_masked})]
    if float(hp.get('block_mask_wt', 1.0)) < 1.0 or float(hp.get('block_mask_mt', 1.0)) < 1.0:
        regimes.append(('unmasked_', {'wt': None, 'mt': None}))
    return regimes


def observed_scale_link(hp):
    """The only link there is: the expected reading under the noisy-floor Tobit observation model (``val_stats.tobit_link``), applied to the predictions when validation scores them on
    the assay's observed scale (RMSE and the observed-scale epistasis). Reporting only; it takes no part in training."""
    return val_stats.tobit_link(float(hp.get('tobit_sigma_y', TOBIT_SIGMA_Y)), float(hp.get('tobit_mu_f', TOBIT_MU_F)), float(hp.get('tobit_sigma_f', TOBIT_SIGMA_F)))


# What validation sends to the logger lives in val_stats (shared verbatim with main-baseline). These names stay importable from here.
EXTERNAL_SUMMARY_METRICS = val_stats.EXTERNAL_LOGGED


def epi_log_names(res: dict, diag: dict, prefix: str = '') -> dict:
    """The ``val_*`` epistasis scalars one validation regime sends to the logger (val_stats.epi_log_names): the kept names only, finite values only."""
    return val_stats.epi_log_names(res, diag, prefix)


def summarise_external_benchmarks(per_loader: dict, external: set, prefix: str = '') -> dict:
    """OOD validation loaders summarised as ``val_<metric>_external`` (val_stats.summarise_external): rho_combined, rho_wt, rmse_combined, rmse_singles_wt.

    ``per_loader`` maps loader name -> {metric: value}. These libraries are deliberately kept out of every aggregate
    (``val_*_avg``, the pooled global ranking, the epistasis levels): they are out of the training distribution, so
    including them makes the aggregate incomparable across runs and inconsistent with what the model was trained on.
    """
    return val_stats.summarise_external(per_loader, external, prefix)


_AA20 = 'ACDEFGHIKLMNPQRSTVWY'
_AA_TO_IDX = {a: i for i, a in enumerate(_AA20)}


def classify_megascale_library(name: str) -> str:
    """Classify an in-distribution MegaScale library name into ``'natural_wt'``, ``'designed'``, or ``'mutant_bg'`` (val_stats.library_group: nat / des / mutbg)."""
    return {'nat': 'natural_wt', 'des': 'designed', 'mutbg': 'mutant_bg'}[val_stats.library_group(name)]


def _parse_single_aa_pair(mut):
    """Extract ``(wt_aa, mt_aa)`` from a single-mutant identifier (``'A12G'`` or ``[('A', 12, 'G')]``); ``None`` otherwise."""
    if isinstance(mut, str):
        if ':' in mut or len(mut) < 3:
            return None
        w, m = mut[0], mut[-1]
        return (w, m) if (w in _AA_TO_IDX and m in _AA_TO_IDX) else None
    if isinstance(mut, (list, tuple)):
        item = mut[0] if (len(mut) == 1 and isinstance(mut[0], (list, tuple))) else (mut if len(mut) == 3 and not isinstance(mut[0], (list, tuple)) else None)
        if item is not None and len(item) >= 3:
            w, m = str(item[0]), str(item[2])
            return (w, m) if (w in _AA_TO_IDX and m in _AA_TO_IDX) else None
    return None


def compute_nat_debiased_rho(nat_single_records: list) -> float:
    """Mean per-library Spearman on natural WT validation singles after projecting out additive ``(aa_wt, aa_mut)`` effects.

    ``nat_single_records`` is a list of ``(lib_name, wt_scores, gt_scores, aa_pairs)`` over canonical singles with finite
    ground truth and valid canonical amino-acid pairs. Fits one pooled linear projection ``lib_intercept + aa_wt + aa_mut``
    across all natural validation singles (avoiding per-domain small-sample overfitting) and averages within-library
    Spearman of residualised ``wt_scores`` against residualised ``gt_scores``.
    """
    valid_libs = [(lib, np.asarray(w, float), np.asarray(g, float), pairs)
                  for lib, w, g, pairs in nat_single_records if len(w) >= 5]
    if not valid_libs:
        return float('nan')
    total_n = sum(len(w) for _, w, _, _ in valid_libs)
    n_libs = len(valid_libs)
    # Design matrix: n_libs library indicators + 19 wt_aa indicators + 19 mt_aa indicators
    X = np.zeros((total_n, n_libs + 38), dtype=np.float64)
    y_w = np.zeros(total_n, dtype=np.float64)
    y_g = np.zeros(total_n, dtype=np.float64)
    slices = []
    pos = 0
    for li, (lib, w, g, pairs) in enumerate(valid_libs):
        m = len(w)
        X[pos:pos + m, li] = 1.0
        for r_idx, (wa, ma) in enumerate(pairs):
            wi, mi = _AA_TO_IDX[wa], _AA_TO_IDX[ma]
            if wi < 19:
                X[pos + r_idx, n_libs + wi] = 1.0
            if mi < 19:
                X[pos + r_idx, n_libs + 19 + mi] = 1.0
        y_w[pos:pos + m] = w
        y_g[pos:pos + m] = g
        slices.append((pos, pos + m))
        pos += m
    bw, _, _, _ = np.linalg.lstsq(X, y_w, rcond=1e-6)
    bg, _, _, _ = np.linalg.lstsq(X, y_g, rcond=1e-6)
    rw = y_w - X @ bw
    rg = y_g - X @ bg
    rhos = [stats.safe_spearman(rw[a:b], rg[a:b]) for a, b in slices]
    rhos = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(rhos)) if rhos else float('nan')


def tf_single_label(y, cens, cens_src, wt_lat, dG_wt, post=None):
    """The latent ddG label of one measured single under --tf_label posterior, detached; ``y`` and ``wt_lat`` (the WT head's read) on the ddG scale, ``dG_wt`` finite.

    Defined everywhere (no reading: the prior; a range tag: the survival term, its ``y`` being the tag's bound). A floor-censored number (--rank_censor_threshold) is a number: only
    the assay's own tags count as bounds, as in the Tobit regression. ``post`` carries the posterior's constants (``s`` the prior width, ``sigma_y``, ``mu_f``, ``sigma_f``;
    --tf_prior_sd and the --tobit_* flags); None is the defaults of ``losses``.
    """
    tag = torch.where(cens_src == censoring.SRC_RANGE, cens, torch.zeros_like(cens))
    return tobit_posterior_mean(dG_wt + y, dG_wt + wt_lat, tag, **(post or {})) - dG_wt


class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(self, val_dataloader_names=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['val_dataloader_names'])
        hp = self.hparams

        lora = lambda side: {**LORA_CONSTANTS, "lora_rank": hp[f'lora_rank_{side}'], "lora_alpha": hp[f'lora_alpha_{side}'], "lora_dropout": hp[f'lora_dropout_{side}'],
                             "target_mode": hp.get(f'target_mode_{side}', config.LORA_TARGET_MODE), "seed": hp.seed, "incl_sequence_head": bool(hp[f'incl_sequence_head_{side}'])}
        lora_config = {'wt_config': lora('wt'), 'mt_config': lora('mt'), 'seed': hp.seed}

        self.model = MSRModel(lora_config=lora_config, shared_scale_init=SHARED_SCALE_INIT, shared_bias_init=SHARED_BIAS_INIT, model_dtype=torch.float32)

        # on_save_checkpoint deliberately keeps only the lora / calibration entries, so a
        # checkpoint is missing every frozen base weight. Lightning's default strict load would then refuse
        # to resume from our own checkpoints; the base comes from the pretrained weights either way.
        self.strict_loading = False
        # Which parameters arrived trainable (the pretrained base is frozen before this point); validation re-applies it, and configure_optimizers builds its groups from it.
        self._baseline_requires_grad = {name: p.requires_grad for name, p in self.model.named_parameters()}
        self._initially_trainable = {name for name, p in self.model.named_parameters() if p.requires_grad}

        self.crit_rank_wt = ListMLELoss() if hp.lambda_rank_wt > 0 else None
        self.crit_rank_mt = ListMLELoss() if hp.lambda_rank_mt > 0 else None

        # Regression is the noisy-floor Tobit likelihood on the latent absolute dG. The noise model's constants (--tobit_sigma_y / _mu_f / _sigma_f) and the width of the teacher-forcing
        # prior around the WT head's read (--tf_prior_sd); the defaults are the constants of losses.py.
        self.tobit_params = dict(sigma_y=float(hp.get('tobit_sigma_y', TOBIT_SIGMA_Y)), mu_f=float(hp.get('tobit_mu_f', TOBIT_MU_F)),
                                 sigma_f=float(hp.get('tobit_sigma_f', TOBIT_SIGMA_F)))
        self.tf_post = dict(s=float(hp.get('tf_prior_sd', TF_PRIOR_SD)), **self.tobit_params)
        logging.info(f"Regression: noisy-floor Tobit NLL {self.tobit_params} on latent dG"
                     + (f"; teacher-forcing prior width {self.tf_post['s']}" if hp.get('tf_label', 'measured') == 'posterior' else ''))
        # The one link: the Tobit expected reading, applied when validation scores the observed scale (RMSE, naive_rho_comb). Reporting only.
        self.obs_link = observed_scale_link(hp)

        # Validation reads the passes in the regimes of validation_regimes(): the main one masks every mutated site on its own for both adapters, and a run that trained either head
        # on visible sites also scores the unmasked regime under val_unmasked_*.
        self.val_regimes = validation_regimes(hp)
        self.val_mask_strategy = self.val_regimes[0][1]
        self.monitor_metric = monitor_metric(hp)

        patience = int(hp.get('plateau_patience', 0) or 0)
        warmup = int(hp.get('lr_warmup_steps', 0) or 0)
        self._decay_start_step = {'wt': None if patience > 0 else warmup, 'mt': None if patience > 0 else warmup}
        self._wt_plateau_best, self._wt_plateau_bad = -float('inf'), 0
        self._mt_plateau_best, self._mt_plateau_bad = -float('inf'), 0

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = list(val_dataloader_names) if val_dataloader_names else ['val']
        self.external_val_sets = set(config.EXTERNAL_VAL_SETS)
        logging.info(f"OOD validation sets held OUT of every aggregate (diagnostics only): {sorted(self.external_val_sets)}; summarised as val_rho_combined_external and val_rmse_combined_external")

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
        peft_wt = self.model.peft_wt
        try:
            if 'bf16' not in str(self.trainer.precision):       # the CPU smoke runs in fp32 (ESM_MSR_PRECISION=32); production is bf16-mixed
                return
        except RuntimeError:
            pass
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
        peft_wt, peft_mt = self.model.peft_wt, self.model.peft_mt
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

        This has to happen before the first forward, not just before training. ``main()`` calls
        ``trainer.validate`` before ``trainer.fit``, so the hook this used to live in (``on_train_start``) fired too
        late: that first validation ran with the base weights still duplicated between the two adapters AND still in
        fp32, which is roughly 11 GB of parameters instead of 3. That sets the CUDA caching allocator's high-water
        mark at its maximum, and the reshape that follows frees those fp32 storages into the allocator's CACHE
        rather than back to the driver - so training then allocates its activations inside a large, badly
        fragmented reservation. The symptom is a monotone slowdown (2.2 s/step rising past 12 s/step by step 260)
        with the card pinned near 100%, which looks like a leak and is really fragmentation.

        Do NOT call empty_cache here. An earlier version did, on the theory that the freed fp32 storages should go
        back to the driver. Measured at micro_batch_size 64, with the GPU exclusive, it cost 3.7 GiB of DEVICE
        memory while changing torch's own accounting not at all:

            main-baseline            torch peak 23.10  reserved 23.61  nvidia-smi 27.38 GiB
            here, with empty_cache   torch peak 23.10  reserved 23.63  nvidia-smi 31.07 GiB
            here, without it         torch peak 23.10  reserved 23.61  nvidia-smi 27.38 GiB

        Under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (which every launcher here sets), empty_cache
        releases the expandable segment's pages and the next allocation re-expands into fresh mappings; the WSL2
        driver keeps the released ones, so the process footprint grows by roughly what was freed. The allocator
        reuses those blocks perfectly well on its own - reserved is 23.61 either way - so the call bought nothing
        and spent 3.7 GiB of the 31.84 GiB card, which is the difference between 4.4 GiB of headroom at
        micro_batch_size 64 and 0.8.

        The reshape ordering above is still the point of this method and still necessary.

        Called from every stage hook, and deliberately NOT behind a once-only flag. The flag this used to carry
        made the re-share fire exactly once per process, which is wrong whenever a stage boundary moves the
        module: ``main()`` validates before it fits, so a fresh run's first hook is ``on_validation_start``;
        Lightning returns the module to the CPU when that stage tears down; and the move back for ``fit`` gives
        every parameter its own storage again - the breakage this exists to undo, one stage later, with the flag
        blocking the repair. Measured, dc8_s2 against its own resume (which passes --skip_val and so re-shares
        at ``on_train_start`` instead): mem/allocated_gb 5.706 GB against 2.972, i.e. a second copy of the
        frozen bf16 base carried for the whole run, and the same 2.73 GB again on reserved and on peak. Only
        runs that skip the zero-shot validation were ever correct. Re-sharing is a dict build and ~570 pointer
        assignments, so running it once per stage costs nothing worth guarding.
        """
        self._reshare_base_params_on_device()

    def on_validation_start(self):
        self._prepare_device_weights()

    def on_train_start(self):
        self._prepare_device_weights()
        sch = self.lr_schedulers() if getattr(self, '_trainer', None) is not None else None
        if sch is not None and hasattr(sch, 'lr_lambdas') and hasattr(sch, 'base_lrs'):
            step = int(getattr(sch, 'last_epoch', 0))
            for g, base_lr, fn in zip(sch.optimizer.param_groups, sch.base_lrs, sch.lr_lambdas):
                if callable(fn):
                    g['lr'] = float(base_lr) * float(fn(step))

    def _compute_rank_loss(self, pred, targets, mask, cens, list_size, crit_fn, norm_to=None):
        """
        ListMLE over ``pred`` cut into lists of ``list_size`` (a block hands it exactly one list), censored members tied at the bottom.

        ListMLE sums its terms over the list, so a list's loss grows with its length. ``norm_to`` (``--rank_list_size``) rescales every list to that many members' worth (loss x norm_to /
        its valid members), which keeps the rank:regression balance of ``--lambda_rank_*`` / ``--lambda_reg_*`` the same whatever the list is made of. A list with fewer than two valid
        members contributes 0 and still counts.
        """
        if mask.dtype != torch.bool:
            raise AssertionError(f"rank mask must be bool, got {mask.dtype}.")
        valid_len = (pred.shape[0] // list_size) * list_size
        if valid_len <= 0:
            return None, 0.0, 0
        view = lambda t: t[:valid_len].view(-1, list_size)
        pred_rank, targ_rank, mask_rank, cens_rank = view(pred), view(targets), view(mask), view(cens)
        num_lists = valid_len // list_size
        # Per list, so that neither the scaling nor forward_censored's own mean can mix lists. A starved list returns a gradient-attached zero, which is what lets it
        # count in the mean without contributing.
        per_list = torch.stack([crit_fn.forward_censored(
            pred_rank[i:i + 1], targ_rank[i:i + 1], mask_rank[i:i + 1], cens_rank[i:i + 1]
        ) for i in range(num_lists)])
        n_valid = mask_rank.float().sum(dim=-1).clamp(min=1.0)
        ref = float(norm_to) if norm_to else float(list_size)
        scaled_loss = (per_list * (ref / n_valid)).mean()
        return scaled_loss, per_list.detach().mean().item(), num_lists


    def _reg_loss(self, pred, target, bound, cens, mask, denom, dgwt):
        """
        The Tobit regression over ``mask``: the noisy-floor mixture on the LATENT absolute dG (``pred`` is the latent ddG, so each item is shifted by its library's ``dgwt``), with the
        assay's own '<-1' / '>5' tags scored by the survival term against ``bound``.

        Returns ``(loss_term, summed_value, count)``: the sum of the per-item losses over ``denom``, the same sum on its own, and the number of items.
        """
        if mask.dtype != torch.bool:
            raise AssertionError(f"regression mask must be bool, got {mask.dtype}: an integer mask indexes rows "
                                 f"instead of selecting them, which fails silently.")
        off = dgwt[mask]
        L = tobit_regression_loss(off + pred[mask], off + target[mask], off + bound[mask], cens[mask], **self.tobit_params)
        return L.sum() / denom, float(L.sum().item()), float(L.numel())


    # ------------------------------------------------------------------------------------------------------------ block training
    # Block training (esm_msr.blocks): a step is a singles block (B positions, a [B, 20] tensor from B masked-site passes, ONE ListMLE list of
    # B x 20 cells) or a pair block (2 WT + 38/40 MT passes, the cells of a position pair, lists of B rows or columns). The objective: the WT head on the
    # singles, the MT head through 0.5 * (forced WT half + MT) with --tf_mix, the wild-type identity cell in every ListMLE list, and (--lambda_cond_mt)
    # the MT head's conditional legs trained directly.

    def _block_cal(self, side):
        return getattr(self.model, f'calibration_head_{side}')

    def _block_cal_sg(self, side, raw, n):
        """The calibrated prediction with the calibration head's scale and bias held constant (stop-gradient), as the audit's ListMLE scores it."""
        head = self._block_cal(side)
        s = head.scale.detach() if head.use_scale else 1.0
        b = head.bias_for(n)
        return raw * s + (b.detach() if torch.is_tensor(b) else b)

    def _block_s_wt(self):
        head = self._block_cal('wt')
        return head.scale.detach() if head.use_scale else 1.0

    def _block_llr(self, b, seqs, read_pos, wt_at, side, grad):
        chunk = int(self.hparams.get('block_row_chunk', 0) or 0)
        R = int(seqs.shape[0])
        if grad and chunk > 0 and R > chunk:
            # --block_row_chunk: the passes run in checkpointed chunks. Only a chunk's activations are alive at a time (the rest are recomputed
            # inside backward, with the dropout draw restored), so a 40-row pair block fits a 20 GB card; the gradient is exact.
            from torch.utils.checkpoint import checkpoint
            return torch.cat([checkpoint(self.model.forward_site_llr, seqs[k:k + chunk], read_pos[k:k + chunk], wt_at[k:k + chunk], side,
                                         b['coords'], b['structure_tokens'], use_reentrant=False) for k in range(0, R, chunk)])
        with torch.set_grad_enabled(grad):
            llr = self.model.forward_site_llr(seqs, read_pos, wt_at, side, b['coords'], b['structure_tokens'])
        if not grad:
            llr = llr.detach()
            if torch.cuda.is_available():
                torch.clear_autocast_cache()        # a weight cast under no_grad must not be reused by a later grad forward
        return llr

    def _block_singles_llr(self, b, seqs, pos, wt_ids, side, grad):
        """
        The ``[B, 20]`` raw LLR of a singles block for one head. ``--block_mask_wt/_mt p`` is the probability that a position is read with its site
        MASKED (1, the default: every position, one pass reading the whole alphabet); a position drawn unmasked is read with its residue visible.
        The WT head then reads the plain wild-type sequence at that position (identical rows are one forward, so this is the cheap case); the
        MT head reads each of the 19 substitutions as its own sequence, ``logit[x] - logit[wt]`` at the visible mutation, which costs 19 passes
        for that position instead of 1 (use ``--block_row_chunk``). The wild-type column is exactly 0 either way.
        """
        p_mask = float(self.hparams.get('block_mask_' + side, 1.0))
        P = int(pos.shape[0])
        if p_mask >= 1.0:
            return self._block_llr(b, seqs, pos, wt_ids, side, grad)
        keep = torch.rand(P, device=pos.device) >= p_mask                                 # True: this position is read visible
        if side == 'wt':
            return self._block_llr(b, blocks.unmasked_rows(seqs, b['seq'], keep), pos, wt_ids, side, grad)
        masked, vis = (~keep).nonzero().squeeze(1), keep.nonzero().squeeze(1)
        llr = torch.zeros(P, 20, device=pos.device)
        if masked.numel():
            llr = llr.index_copy(0, masked, self._block_llr(b, seqs[masked], pos[masked], wt_ids[masked], side, grad))
        if vis.numel():
            vseq, vpos, vwt, vrow, vcol = blocks.mutant_single_sequences(b['seq'], pos[vis], wt_ids[vis])
            out = self._block_llr(b, vseq, vpos, vwt, side, grad)
            llr = llr.index_put((vis[vrow], vcol), out[torch.arange(int(vcol.numel()), device=pos.device), vcol])
        return llr

    def _block_ctx(self, lab, rank_groups, reg_exclude=None, rank_exclude=None):
        """
        The labels of one block on the item's own scales, and the lists the rank loss is drawn over (index tensors into the flat block).

        ``reg_exclude`` takes cells out of the regression only (the identity cells, whose prediction is the constant 0); ``rank_exclude`` out of
        the lists (a conditional leg whose partner is unusable).
        """
        y = lab['y']
        dg_ok = torch.isfinite(lab['dG_wt'])
        obs_rank = torch.isfinite(y)
        if rank_exclude is not None:
            obs_rank = obs_rank & ~rank_exclude
        obs_reg = obs_rank & dg_ok
        if reg_exclude is not None:
            obs_reg = obs_reg & ~reg_exclude
        # An unobserved cell has no dG_wt: its input is 0 so nothing is NaN, and dg_ok / obs_* say which cells count.
        dG_wt = torch.where(dg_ok, lab['dG_wt'], torch.zeros_like(lab['dG_wt']))
        # The Tobit likelihood is on the latent absolute dG and takes the tag's own bound; only the assay's '<-1' / '>5' range tags are censored for the regression, a
        # floor-censored cell (--rank_censor_threshold) has a number the mixture scores and only ties its rank.
        cens_reg = torch.where(lab['cens_src'] == censoring.SRC_RANGE, lab['cens'], torch.zeros_like(lab['cens']))
        return types.SimpleNamespace(
            y=y, y_safe=torch.nan_to_num(y, nan=0.0), cens=lab['cens'], cens_reg=cens_reg, dG_wt=dG_wt,
            dg_ok=dg_ok, obs_rank=obs_rank, obs_reg=obs_reg, groups=rank_groups, bound=lab['bound'])

    def _block_reg_rank(self, c, pred_cal, rank_pred, lam_reg, lam_rank, crit_rank, name):
        """Regression over the observed cells + ListMLE of ``rank_pred`` over each list; returns the weighted terms and logs them."""
        terms, flat = [], (lambda t: t.reshape(-1))
        if lam_reg > 0 and bool(c.obs_reg.any()):
            denom = float(max(1, int(c.obs_reg.sum())))
            term, val, w = self._reg_loss(flat(pred_cal), flat(c.y_safe), flat(c.bound), flat(c.cens_reg), flat(c.obs_reg), denom, flat(c.dG_wt))
            terms.append(lam_reg * term)
            self._block_sums[f'reg_{name}'] += val; self._block_cnts[f'reg_{name}'] += w
        if lam_rank > 0 and crit_rank is not None:
            ls, vals = [], []
            norm_to = int(self.hparams.rank_list_size)
            for g in c.groups:
                L, val, _ = self._compute_rank_loss(flat(rank_pred)[g], flat(c.y_safe)[g], flat(c.obs_rank)[g], flat(c.cens)[g], int(g.numel()), crit_rank,
                                                    norm_to=norm_to)
                if L is not None:
                    ls.append(L); vals.append(val)
            if ls:
                terms.append(lam_rank * torch.stack(ls).mean())
                self._block_sums[f'rank_{name}'] += float(np.mean(vals)); self._block_cnts[f'rank_{name}'] += 1
        return terms

    def _block_backward(self, terms, what):
        if not terms:
            return
        total = sum(terms)
        if not torch.isfinite(total): raise AssertionError(f"{what} loss evaluated to NaN/Inf.")
        if not total.requires_grad: raise AssertionError(f"{what} loss detached from the graph; cannot call backward.")
        self.manual_backward(total)

    def _block_forced(self, wt_cal, label_lat, has_label):
        """The WT half of the combined prediction: tf_mix of the label where one is usable, else the WT head (detached: the MT head's loss never trains the WT head)."""
        tf_mix = float(self.hparams.tf_mix)
        base = wt_cal.detach()
        return torch.where(has_label, tf_mix * label_lat + (1.0 - tf_mix) * base, base)

    def _block_step(self, b):
        self._block_sums, self._block_cnts = defaultdict(float), defaultdict(float)
        (self._block_singles if b['block'] == 'single' else self._block_pair)(b)
        logs = {f'L_{k}': self._block_sums[k] / max(1, self._block_cnts[k])
                for k in ('reg_wt', 'rank_wt', 'reg_combined', 'rank_combined', 'reg_cond', 'rank_cond') if self._block_cnts[k] > 0}
        self._cens_diag = (int((b['cens'] < 0).sum().item()), int((b['cens'] > 0).sum().item()))
        return logs

    def _block_singles(self, b):
        hp = self.hparams
        pos, wt_ids = b['positions'], b['wt_ids']
        dev = b['y'].device
        seqs = blocks.single_sequences(b['seq'], pos)
        lab = {k: b[k].float() if k not in ('cens', 'cens_src') else b[k] for k in ('y', 'cens', 'bound', 'dG_meas', 'dG_wt', 'cens_src')}
        n = torch.ones(b['y'].shape, dtype=torch.long, device=dev)                        # one mutation per cell
        # The wild-type residue is a cell of the list too: it reads exactly 0 (no mutation, so no bias either) and its label is 0.
        ar = torch.arange(pos.shape[0], device=dev)
        wi = torch.as_tensor([blocks.TOKEN_TO_AA[int(t)] for t in wt_ids], device=dev)
        lab['y'], lab['cens'] = lab['y'].clone(), lab['cens'].clone()
        lab['y'][ar, wi], lab['cens'][ar, wi] = 0.0, 0
        n[ar, wi] = 0
        ident_cells = torch.zeros(b['y'].shape, dtype=torch.bool, device=dev)
        ident_cells[ar, wi] = True
        n_all = int(b['y'].numel())
        rows_per_list = int(hp.get('block_list_rows', 0) or 0)
        if rows_per_list > 0:                                                              # --block_list_rows R: lists of R positions (R = 1: one position each)
            P = int(pos.shape[0])
            cells = torch.arange(n_all, device=dev).view(P, 20)[torch.randperm(P, device=dev)]
            groups = [g.reshape(-1) for g in torch.tensor_split(cells, max(1, -(-P // rows_per_list)))]
        else:
            groups = [torch.arange(n_all, device=dev)]                                     # one list: the whole [B, 20] block
        c = self._block_ctx(lab, groups, reg_exclude=ident_cells)
        wt_train = (hp.lambda_reg_wt > 0 or hp.lambda_rank_wt > 0) and not ESM3EpistasisLightningModule._head_is_frozen(self, 'wt')
        mt_on = (hp.lambda_reg_mt > 0 or hp.lambda_rank_mt > 0) and not ESM3EpistasisLightningModule._head_is_frozen(self, 'mt')
        llr_wt = self._block_singles_llr(b, seqs, pos, wt_ids, 'wt', grad=wt_train)
        wt_cal = self._block_cal('wt')(llr_wt, n)
        s_wt = self._block_s_wt()
        # The lists are scored on the calibrated scale (stop-gradient), where "no mutation" is 0 and a mutation carries its bias.
        if wt_train:
            wt_rank = self._block_cal_sg('wt', llr_wt, n) / s_wt
            self._block_backward(self._block_reg_rank(c, wt_cal, wt_rank, hp.lambda_reg_wt, hp.lambda_rank_wt, self.crit_rank_wt, 'wt'), 'WT')      # freed before the MT pass builds its graph
        if not mt_on:
            return
        llr_mt = self._block_singles_llr(b, seqs, pos, wt_ids, 'mt', grad=True)       # with --block_mask_* 1 a masked single is the same input for both adapters
        mt_cal = self._block_cal('mt')(llr_mt, n)
        if hp.get('tf_label', 'measured') == 'measured':
            lat = c.y.detach()
            has = torch.isfinite(lat) & c.dg_ok & c.obs_rank
            t_single, tf_cut = hp.get('rank_censor_threshold'), hp.get('tf_cut')
            if tf_cut is not None:                               # a single read at or below the cut gives no label
                has = has & ~((c.dG_wt + c.y) <= float(tf_cut))
            elif t_single is not None:
                has = has & (c.cens == 0)
        else:
            lat = tf_single_label(c.y, c.cens, lab['cens_src'], wt_cal.detach(), c.dG_wt, post=self.tf_post)
            lat = torch.where(n == 0, torch.zeros_like(lat), lat)      # the identity cell: the wild type has no effect to estimate
            has = torch.isfinite(lat) & c.dg_ok & c.obs_rank
        comb_cal = combine_predictions(self._block_forced(wt_cal, lat, has), mt_cal)
        comb_rank = combine_predictions(self._block_forced(self._block_cal_sg('wt', llr_wt, n), lat, has), self._block_cal_sg('mt', llr_mt, n)) / s_wt
        self._block_backward(self._block_reg_rank(c, comb_cal, comb_rank, hp.lambda_reg_mt, hp.lambda_rank_mt, self.crit_rank_mt, 'combined'), 'MT')


    def _block_cond_legs(self, b, llr_m, Xa, Ya, wi, wj, full, ident):
        """
        The MT head's conditional legs trained directly (--lambda_cond_mt, the audit's two-conditional supervision on the block's passes).

        Row pass x (the substitution x at i, j masked) reads ddG(Y | X) for every y at j in one forward, column pass y the other way. Each leg is
        scored as the partner single's reading plus the calibrated leg, ``single_x + cal(Row[x][y])``, against the DOUBLE's own label (so a
        saturated reading is never differenced), over the cells whose partner single is usable: finite and its absolute dG above
        ``--cond_partner_min_dg`` (an unfolded partner leaves nothing to condition on). The
        lists also hold the identity cell of each pass: no second mutation, so the prediction is the partner single itself.
        """
        hp = self.hparams
        lam = float(hp.get('lambda_cond_mt', 0.0) or 0.0)
        dev = llr_m.device
        nx = len(Xa)
        Row, Col = llr_m[:nx], llr_m[nx:]
        dG = torch.nanmean(full['dG_wt'])
        cutoff = float(hp.get('cond_partner_min_dg', 1.0))
        rank_w = lam * float(hp.lambda_rank_mt) / max(float(hp.lambda_reg_mt), 1e-8)
        B = max(1, int(hp.get('block_list_rows', 0) or 0) or int(hp.get('block_positions', 8)))
        s_wt = self._block_s_wt()
        terms = []
        for passes, raw_all, partner, wt_self, wt_read, axis_idx, lab_rows in (
                ('row', Row, b['si_y'], wi, wj, Xa, lambda k: {f: full[f][k] for f in full}),                       # partner single at i
                ('col', Col, b['sj_y'], wj, wi, Ya, lambda k: {f: full[f].t()[k] for f in full})):                   # partner single at j
            real = [r for r, a in enumerate(axis_idx) if a != wt_self]
            if not real:
                continue
            ridx = torch.as_tensor(real, device=dev)
            aa = torch.as_tensor([axis_idx[r] for r in real], device=dev)
            raw = raw_all[ridx]                                                                      # [P, 20] raw LLR over the residue read
            n_leg = torch.ones(raw.shape, dtype=torch.long, device=dev)
            n_leg[:, wt_read] = 0                                                                    # reading the wild type: no second mutation
            leg = self._block_cal('mt')(raw, n_leg)
            s = partner[aa]                                                                          # the partner single's ddG
            ok = torch.isfinite(s) & ((dG + s) > cutoff)
            lat_p = torch.nan_to_num(torch.where(ok, s, torch.zeros_like(s)).detach(), nan=0.0)
            lab = lab_rows(aa)
            ident_cols = torch.zeros(raw.shape, dtype=torch.bool, device=dev)
            ident_cols[:, wt_read] = True
            # the cells: the double (or, on the identity column, the partner single itself) of each pass, with the partner usable
            c = self._block_ctx(lab, [], reg_exclude=ident_cols | ~ok.unsqueeze(1), rank_exclude=~ok.unsqueeze(1))
            total = lat_p.unsqueeze(1) + leg
            P = raw.shape[0]
            order = torch.randperm(P, device=dev)
            c.groups = [g.reshape(-1) for g in torch.tensor_split(torch.arange(P * 20, device=dev).view(P, 20)[order], max(1, -(-P // B)))]
            rank_pred = (lat_p.unsqueeze(1) + self._block_cal_sg('mt', raw, n_leg)) / s_wt
            terms += self._block_reg_rank(c, total, rank_pred, 0.5 * lam, 0.5 * rank_w, self.crit_rank_mt, 'cond')
        return terms

    def _block_pair(self, b):
        hp = self.hparams
        dev = b['y'].device
        i, j = int(b['positions'][0]), int(b['positions'][1])
        wt_i, wt_j = int(b['wt_ids'][0]), int(b['wt_ids'][1])
        wi, wj = blocks.TOKEN_TO_AA[wt_i], blocks.TOKEN_TO_AA[wt_j]
        wt_rows, row_seqs, col_seqs, Xa, Ya = blocks.pair_sequences(b['seq'], i, j, wt_i, wt_j)
        lab, ident_cells, full = blocks.pair_labels(b, Xa, Ya, wi, wj)
        if not bool((torch.isfinite(lab['y']) & ~ident_cells).any()):
            return
        nx, ny = len(Xa), len(Ya)
        xi, yj = torch.as_tensor(Xa, device=dev), torch.as_tensor(Ya, device=dev)
        n = (xi != wi).long().view(-1, 1) + (yj != wj).long().view(1, -1)                   # mutations per cell: 2 for a double, 1 for a single, 0 for the wild type
        # lists of B rows or B columns of the cell matrix (--block_list_rows overrides B), the unit chosen at random for each block
        B = max(1, int(hp.get('block_list_rows', 0) or 0) or int(hp.get('block_positions', 8)))
        flat_idx = torch.arange(nx * ny, device=dev).view(nx, ny)
        units = flat_idx if bool(torch.rand(()) < 0.5) else flat_idx.t()
        units = units[torch.randperm(units.shape[0], device=dev)]
        groups = [g.reshape(-1) for g in torch.tensor_split(units, max(1, -(-units.shape[0] // B)))]
        c = self._block_ctx(lab, groups, reg_exclude=ident_cells)
        lam_cond = float(hp.get('lambda_cond_mt', 0.0) or 0.0)
        if not (hp.lambda_reg_mt > 0 or hp.lambda_rank_mt > 0 or lam_cond > 0) or ESM3EpistasisLightningModule._head_is_frozen(self, 'mt'):
            return
        wt_pos = torch.as_tensor([i, j], device=dev)
        p_wt, p_mt = float(hp.get('block_mask_wt', 1.0)), float(hp.get('block_mask_mt', 1.0))
        if p_wt < 1.0:                                                                         # --block_mask_wt: each WT read sees its site visible with prob 1 - p
            wt_rows = blocks.unmasked_rows(wt_rows, b['seq'], torch.rand(2, device=dev) >= p_wt)
        llr_w = self._block_llr(b, wt_rows, wt_pos, b['wt_ids'], 'wt', grad=False)           # 2 wild-type passes (one forward when both are visible); the MT loss never trains the WT head
        llr_m = None
        if p_mt < 1.0 and bool(torch.rand(()) >= p_mt):
            # --block_mask_mt: this block's mutant passes read every site VISIBLE. One sequence per cell (x at i, y at j), read at both sites: the
            # per-item unmasked double. 20 x 20 sequences instead of 40, so it needs --block_row_chunk, and it has no row / column passes to condition on.
            useqs, uread = blocks.pair_unmasked_sequences(b['seq'], i, j, Xa, Ya)
            out = self._block_llr(b, useqs, uread, torch.as_tensor([wt_i, wt_j], device=dev).repeat(nx * ny), 'mt', grad=True)
            cell = torch.arange(nx * ny, device=dev)
            mt_raw = (out[0::2][cell, xi.repeat_interleave(ny)] + out[1::2][cell, yj.repeat(nx)]).view(nx, ny)
            wt_raw = llr_w[0][xi].unsqueeze(1) + llr_w[1][yj].unsqueeze(0)
        elif bool(hp.get('block_marginal_mt', False)):
            # --block_marginal_mt: mask all mutated residues simultaneously on the MT pass (both i and j masked on doubles; single site masked on the identity row / column).
            mseqs = blocks.pair_marginal_sequences(b['seq'], i, j, wt_rows)
            mread = torch.as_tensor([i, j, i, j], device=dev)
            mwt = torch.as_tensor([wt_i, wt_j, wt_i, wt_j], device=dev)
            out = self._block_llr(b, mseqs, mread, mwt, 'mt', grad=True)
            wt_raw, mt_raw = blocks.assemble_pair_marginal(llr_w[0], llr_w[1], out, Xa, Ya, wi, wj)
        else:
            mt_seqs = torch.cat([row_seqs, col_seqs])                                          # 40 mutant passes
            read = torch.as_tensor([j] * nx + [i] * ny, device=dev)
            wt_at = torch.as_tensor([wt_j] * nx + [wt_i] * ny, device=dev)
            llr_m = self._block_llr(b, mt_seqs, read, wt_at, 'mt', grad=True)
            wt_raw, mt_raw = blocks.assemble_pair(llr_w[0], llr_w[1], llr_m[:nx], llr_m[nx:], Xa, Ya)
        wt_cal, mt_cal = self._block_cal('wt')(wt_raw, n), self._block_cal('mt')(mt_raw, n)
        # The teacher-forcing label of a cell: its two singles, one at a time and summed, each from the block's own singles (a double with a tagged or missing single has no ddG_A / ddG_B).
        # A single without a usable measurement is read from the WT head, so it never costs its partner's measurement: one measurement and one guess beat two guesses.
        t_single, tf_cut = hp.get('rank_censor_threshold'), hp.get('tf_cut')
        dG0 = c.dG_wt.reshape(-1)[0]
        cut = float(tf_cut) if tf_cut is not None else float(t_single) if t_single is not None else config.ASSAY_FLOOR
        cal_w = self._block_cal('wt')

        def one(aa, wt_aa, llr, tag):
            """The labels of one position's singles, and which of them carry a measurement; the wild type has no effect to estimate."""
            w = cal_w(llr[aa], (aa != wt_aa).long()).detach()
            y, cens, src = b[tag + 'y'][aa].float(), b[tag + 'cens'][aa], b[tag + 'cens_src'][aa]
            if hp.get('tf_label', 'measured') == 'measured':
                z = y.detach()
                ok = torch.isfinite(z) & ((dG0 + y) > cut) & ~((cens != 0) & (src == censoring.SRC_RANGE))
            else:
                z = tf_single_label(y, cens, src, w, dG0, post=self.tf_post)
                ok = torch.isfinite(z)
            wild = aa == wt_aa
            return torch.where(wild, torch.zeros_like(z), torch.where(ok, z, w)), ok & ~wild
        zA, measA = one(xi, wi, llr_w[0], 'si_')
        zB, measB = one(yj, wj, llr_w[1], 'sj_')
        lat = zA.unsqueeze(1) + zB.unsqueeze(0)
        measA, measB = measA.unsqueeze(1), measB.unsqueeze(0)
        has = (measA | measB | (n == 0)) & c.dg_ok                           # with neither single measured the WT half stays the WT head's own
        comb_cal = combine_predictions(self._block_forced(wt_cal, lat, has), mt_cal)
        s_wt = self._block_s_wt()
        comb_rank = combine_predictions(self._block_forced(self._block_cal_sg('wt', wt_raw, n), lat, has), self._block_cal_sg('mt', mt_raw, n)) / s_wt
        terms = self._block_reg_rank(c, comb_cal, comb_rank, hp.lambda_reg_mt, hp.lambda_rank_mt, self.crit_rank_mt, 'combined')
        if lam_cond > 0 and llr_m is not None:                                             # the conditional legs are the masked row / column passes
            terms += self._block_cond_legs(b, llr_m, Xa, Ya, wi, wj, full, ident_cells)
        if 'n_lib_pairs' in b:                                                             # --block_domain_steps: this pair's share of its library's doubles
            w = float(hp.block_pair_weight) / int(b['n_lib_pairs'])
            terms = [w * t for t in terms]
        self._block_backward(terms, 'MT')

    def _log_lrs(self):
        opts = self.trainer.optimizers
        for opt in opts:
            for i, g in enumerate(opt.param_groups):
                name = g.get("name", f"group{i}")
                self.log(f"lr/{name}", float(g["lr"]), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def log_calibration_head(self, on_step=True, on_epoch=False, prog_bar=False, logger=True):
        for suffix in ("wt", "mt"):
            head = getattr(self.model, f"calibration_head_{suffix}", None)
            if head is None:
                continue
            for param in ("scale", "bias"):
                val = getattr(head, param, None)
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

    def on_train_epoch_start(self) -> None:
        """Synchronize the block sampler with trainer's current_epoch for resumable dataloading."""
        trainer = getattr(self, 'trainer', None)
        if trainer is None:
            return
        epoch = getattr(trainer, 'current_epoch', 0)
        train_loader = getattr(trainer, 'train_dataloader', None)
        if train_loader is not None:
            sampler = getattr(train_loader, 'sampler', None)                 # batch_size=None: the sampler yields the blocks
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

    def training_step(self, batch: dict, batch_idx: int):
        self.model.train()
        batch = utils._normalize_batch(batch)                  # Lightning hands a single loader's batch over as a one-element list
        logs = self._block_step(batch)

        optim = self.optimizers()

        # --grad_accum_blocks K: the K blocks' gradients add up in .grad and the optimizer steps once, after the K-th. Everything below (norms, clip, step,
        # scheduler, logs) therefore runs once per K blocks, on the mean gradient and the mean of the K blocks' losses.
        # --block_domain_steps: the same, closed by the sampler at the end of a library, on the plain sum (the pair blocks carry their own weight); the
        # singles' and the doubles' losses are logged apart.
        accum = max(1, int(self.hparams.get('grad_accum_blocks', 1) or 1))
        domain = bool(self.hparams.get('block_domain_steps', False))
        if accum > 1 or domain:
            self._accum_n = getattr(self, '_accum_n', 0) + 1
            if domain and batch.get('block') == 'pair':
                logs = {f'{k}_pair': v for k, v in logs.items()}
            acc, cnt = getattr(self, '_accum_logs', None) or {}, getattr(self, '_accum_cnts', None) or {}
            for k, v in logs.items():
                acc[k], cnt[k] = acc.get(k, 0.0) + float(v), cnt.get(k, 0) + 1
            self._accum_logs, self._accum_cnts = acc, cnt
            if not (batch['step_end'] if domain else self._accum_n >= accum):
                return torch.tensor(0.0, device=self.device)
            logs = {k: v / (cnt[k] if domain else accum) for k, v in acc.items()}
            self._accum_n, self._accum_logs, self._accum_cnts = 0, {}, {}
            seen_g = set()
            for g in optim.param_groups:
                for p in g["params"]:
                    if accum > 1 and p.grad is not None and id(p) not in seen_g:
                        seen_g.add(id(p))
                        p.grad.div_(accum)
            self._opt_steps = getattr(self, '_opt_steps', -1) + 1
            batch_idx = self._opt_steps                         # the logging cadence below counts optimizer steps

        log_every = int(getattr(self.trainer, 'log_every_n_steps', 25) or 25) if getattr(self, '_trainer', None) is not None else 1
        if batch_idx == 0 or (batch_idx + 1) % log_every == 0:
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

        # The gradient clip, per head: the WT parameters (adapter and calibration head) and the MT parameters are clipped separately, so the MT head's doubles do not shrink the WT head's step.
        seen_p = set()
        clip_sets = {}
        for g in optim.param_groups:
            name = str(g.get("name", ""))
            key = 'wt' if name.endswith('_wt') else 'mt' if name.endswith('_mt') else 'other'
            for p in g["params"]:
                if p.grad is not None and id(p) not in seen_p:
                    seen_p.add(id(p))
                    clip_sets.setdefault(key, []).append(p)
        for key, trainable_params in clip_sets.items():
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.GRAD_CLIP_NORM)
            if batch_idx == 0 or (batch_idx + 1) % log_every == 0:
                self.log(f"norm_grad/pre_clip_{key}", float(pre_clip_norm), on_step=True, on_epoch=False, logger=True, sync_dist=True)

        optim.step()
        optim.zero_grad(set_to_none=True)
        self.lr_schedulers().step()   # one config, so Lightning hands back the scheduler itself

        self._log_lrs()
        self.log_calibration_head(on_step=True)

        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)

        if getattr(self, '_cens_diag', None) is not None:
            self.log("train/cens_lower_items", float(self._cens_diag[0]), on_step=True)
            self.log("train/cens_upper_items", float(self._cens_diag[1]), on_step=True)

        if torch.cuda.is_available():
            gb = 1.0 / (1024 ** 3)
            self.log("mem/allocated_gb", torch.cuda.memory_allocated() * gb, on_step=True)
            self.log("mem/reserved_gb", torch.cuda.memory_reserved() * gb, on_step=True, prog_bar=True)
            self.log("mem/peak_allocated_gb", torch.cuda.max_memory_allocated() * gb, on_step=True)

        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        if getattr(self, '_val_cache_dl_idx', None) != dataloader_idx:
            self._val_cache_dl_idx = dataloader_idx
            self.model._masked_logit_cache = {}
        elif getattr(self.model, '_masked_logit_cache', None) is None:
            self.model._masked_logit_cache = {}

        with torch.inference_mode():
            out_dict = self.model.forward_batch(batch, mask_strategy=self.val_mask_strategy)

        B = int(out_dict['wt_lora_pred'].shape[0])
        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        nan_col = lambda: np.full(B, np.nan, dtype=np.float32)
        get = lambda k: batch[k].detach().cpu().float().numpy() if torch.is_tensor(batch.get(k)) else nan_col()

        # The rank metrics read the latent scores (a monotone link leaves them alone). The observed scale is carried separately, for the rmse metrics:
        # observed ddG = h(dG_wt + latent) - dG_wt with h the Tobit expected reading.
        def observed_scale(out):
            res = {}
            if torch.is_tensor(batch.get('dG_wt')):
                dgw = batch['dG_wt'].detach().cpu().double().numpy()
                ok = np.isfinite(dgw)
                z = np.where(ok, dgw, 0.0)
                for key, src in (('wt', out['wt_lora_pred']), ('comb', out['combined_pred'])):
                    v = src.detach().cpu().double().numpy()
                    res[key] = np.where(ok, self.obs_link(z + v) - z, v)
            return res
        obs = observed_scale(out_dict)

        rec = {
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
            'pdb': list(batch.get('pdb', [])),
            'mutations': list(batch.get('mutations', [])),
        }
        if obs:
            rec['obs'] = obs
        if len(self.val_regimes) > 1:
            # the same items read in the other regime (unmasked), scored beside the main one under a prefix
            rec['regimes'] = {}
            for prefix, strategy in self.val_regimes[1:]:
                with torch.inference_mode():
                    r = self.model.forward_batch(batch, mask_strategy=strategy)
                e = {'wt_scores': r['wt_lora_pred'].detach().cpu().float().numpy(), 'mt_scores': r['mt_lora_pred'].detach().cpu().float().numpy(),
                     'comb_scores': r['combined_pred'].detach().cpu().float().numpy()}
                ex_obs = observed_scale(r)
                if ex_obs:
                    e['obs'] = ex_obs
                rec['regimes'][prefix] = e
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
        for prefix, _ in self.val_regimes:
            self._score_regime(prefix, avg_metrics)

        if not self.trainer.sanity_checking and self.global_step > 0:
            self._convergence_step('wt', avg_metrics)
            self._convergence_step('mt', avg_metrics)
            self._stop_if_nothing_is_training()

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _head_is_frozen(self, head: str) -> bool:
        """True once ``head`` ('wt' or 'mt') has started and finished its post-convergence cosine decay (all its optimizer groups at lr <= 0)."""
        if getattr(self, '_trainer', None) is None or not getattr(self.trainer, 'optimizers', None):
            return False
        if getattr(self, '_decay_start_step', {}).get(head) is None:
            return False
        names = (f'lora_{head}', f'calib_{head}')
        groups = [g for opt in self.trainer.optimizers for g in getattr(opt, 'optimizer', opt).param_groups if g.get('name') in names]
        return bool(groups) and all(float(g.get('lr', 0.0)) <= 0.0 for g in groups)

    def _stop_if_nothing_is_training(self) -> None:
        """End the run once every optimizer group that holds trainable parameters has reached lr 0."""
        if getattr(self, '_trainer', None) is None or not getattr(self.trainer, 'optimizers', None):
            return
        for opt in self.trainer.optimizers:
            for g in getattr(opt, 'optimizer', opt).param_groups:
                if float(g.get('lr', 0.0)) > 0.0 and any(p.requires_grad for p in g['params']):
                    return
        if not self.trainer.should_stop:
            logging.info(f"Every trainable group is at lr 0 after epoch {self.trainer.current_epoch}; stopping.")
            self.trainer.should_stop = True

    def _convergence_step(self, head: str, avg_metrics: dict) -> None:
        """Start ``head``'s ('wt' or 'mt') cosine decay once its convergence metric fails to improve by > 1e-4 for ``--plateau_patience`` validations."""
        patience = int(self.hparams.get('plateau_patience', 0) or 0)
        warmup = int(self.hparams.get('lr_warmup_steps', 0) or 0)
        if not hasattr(self, '_decay_start_step'):
            self._decay_start_step = {'wt': None if patience > 0 else warmup, 'mt': None if patience > 0 else warmup}
        if patience <= 0 or self._decay_start_step.get(head) is not None:
            return

        raw_key = convergence_metric(self.hparams, head)
        stripped = raw_key.removeprefix('val_').removesuffix('_avg')
        key = None
        if not raw_key.startswith('val_') and not stripped.startswith('unmasked_') and float(self.hparams.get(f'block_mask_{head}', 1.0)) <= 0.5:
            if avg_metrics.get(f'unmasked_{stripped}'):
                key = f'unmasked_{stripped}'
        if key is None:
            if avg_metrics.get(stripped):
                key = stripped
            elif avg_metrics.get(raw_key):
                key = raw_key
        if key is None:
            if not getattr(self, f'_warned_no_{head}_metric', False):
                setattr(self, f'_warned_no_{head}_metric', True)
                logging.warning(f"--plateau_patience is set but '{raw_key}' is not among the validation metrics {sorted(avg_metrics)}, "
                                f"so the {head.upper()} cosine decay will never start.")
            return

        current = float(np.nanmean(avg_metrics[key]))
        if not np.isfinite(current):
            return

        best = getattr(self, f'_{head}_plateau_best', -float('inf'))
        if current > best + 1e-4:
            setattr(self, f'_{head}_plateau_best', current)
            setattr(self, f'_{head}_plateau_bad', 0)
            return

        bad = getattr(self, f'_{head}_plateau_bad', 0) + 1
        setattr(self, f'_{head}_plateau_bad', bad)
        if bad < patience:
            return

        sch = self.lr_schedulers() if getattr(self, '_trainer', None) is not None else None
        step_now = int(getattr(sch, 'last_epoch', None) if sch is not None else getattr(self, 'global_step', 0))
        start_step = max(warmup, step_now)
        self._decay_start_step[head] = start_step
        logging.info(f"{head.upper()} convergence on {raw_key} (best={best:.4f}, now={current:.4f}) at step {step_now}: starting cosine decay from step {start_step}")

    def _score_regime(self, prefix: str, avg_metrics: dict) -> None:
        """
        Scores and logs one validation regime: the main one (``prefix`` '') or the unmasked copy ('unmasked_'). What is logged is
        decided in ``esm_msr.val_stats``: five per-library ddG metrics, their means and the group means, the external summary and the pooled epistasis. ``avg_metrics`` collects every
        value under ``prefix + name``, logged or not.
        """
        groups = {g: [] for g in val_stats.GROUPS}
        pooled = defaultdict(list)
        ext_per_loader = {}          # OOD loaders: kept out of avg_metrics/pooled, survived by the _external summaries
        extra_prefixes = [p for p, _ in self.val_regimes[1:]]

        for dataloader_idx, valid_outputs in self.validation_step_outputs.items():
            name = (self.val_dataloader_names[dataloader_idx] if dataloader_idx < len(self.val_dataloader_names)
                    else f"unknown_dl_{dataloader_idx}")
            if prefix:
                if not all(prefix in o.get('regimes', {}) for o in valid_outputs):
                    continue
                valid_outputs = [{**o, **o['regimes'][prefix]} for o in valid_outputs]

            cat = lambda key: (np.concatenate([np.atleast_1d(np.asarray(o[key])) for o in valid_outputs])
                               if valid_outputs else np.array([]))
            wt, mt, comb = cat('wt_scores'), cat('mt_scores'), cat('comb_scores')
            gt, cens, dddG = cat('ground_truths'), cat('cens'), cat('dddG_truths')
            cens_src = cat('cens_src')
            # Score every run on the released labelling: a mutant-background library's measurements are singles to both heads. A cache that still carries the 'native_cond'
            # label (a run that predates --exclude_mutbg) is read as singles too, so the yardstick never depends on how a run routed them.
            subset_types = ['single' if t == 'native_cond' else t for o in valid_outputs for t in o['subset_types']]
            obs = None
            if valid_outputs and 'obs' in valid_outputs[0]:
                obs = {k: np.concatenate([o['obs'][k] for o in valid_outputs]) for k in valid_outputs[0]['obs']}

            if not len(gt):
                continue

            # Score the CANONICAL population: every item the assay actually fitted, censored or not. Floor-censored items are scored like any other, because censoring belongs
            # in the loss and excluding them makes the numbers incomparable to a run that censors nothing. The rows the assay only bounded (--censor_range) are held back
            # instead: their label IS their bound and is a library constant, so scoring them would compare predictions against a flat line. They are still dumped in full.
            canon = cens_src != censoring.SRC_RANGE
            sub = lambda a: np.asarray(a)[canon]
            metrics = stats.compute_metrics(
                sub(wt), sub(mt), sub(comb), sub(gt), [t for t, k in zip(subset_types, canon) if k],
                obs=None if obs is None else {k: sub(v) for k, v in obs.items()})
            is_external = name in self.external_val_sets
            for metric, val in metrics.items():
                if val is None or not np.isfinite(val):
                    continue
                if metric in val_stats.PER_LIBRARY_LOGGED:
                    self.log(f"val_{prefix}{metric}/{name}", float(val), on_epoch=True, sync_dist=True)
                if not is_external:
                    avg_metrics[prefix + metric].append(float(val))
            if is_external:
                # OOD benchmark: scored and logged above, but out of every aggregate - avg_metrics, the groups, the pooled epistasis - and dropped before the pooling below.
                # Only its own summary survives (val_stats.summarise_external).
                ext_per_loader[name] = {k: metrics[k] for k in val_stats.EXTERNAL_LOGGED if k in metrics}
                continue
            if np.isfinite(metrics.get('rho_combined', np.nan)):
                groups[val_stats.library_group(name)].append(float(metrics['rho_combined']))

            # Pool for the epistasis hierarchy, which needs every library at once.
            keys = [k for o in valid_outputs for k in self._mut_keys(o['pdb'], o['mutations'])]
            pooled['mut_key'].extend(keys)
            pooled['subset_type'].extend(subset_types)
            pooled['loader'].extend([name] * len(keys))
            pooled['n_mut'].extend([len(k) if k else 0 for k in keys])
            for field, arr in (('wt_scores', wt), ('mt_scores', mt), ('comb_scores', comb),
                               ('ground_truths', gt), ('dddG', dddG), ('cens', cens), ('cens_src', cens_src),
                               ('dG_wt', cat('dG_wt'))):
                pooled[field].extend(np.asarray(arr).tolist())
            if not prefix:       # the dump carries the other regimes' scores too, so any of them can be rescored offline
                for p in extra_prefixes:
                    if valid_outputs and all(p in o.get('regimes', {}) for o in self.validation_step_outputs[dataloader_idx]):
                        for field in ('wt_scores', 'mt_scores', 'comb_scores'):
                            pooled[f'{field}@{p}'].extend(np.concatenate([np.atleast_1d(np.asarray(o['regimes'][p][field]))
                                                                          for o in self.validation_step_outputs[dataloader_idx]]).tolist())

        for key in val_stats.AVG_LOGGED:
            values = avg_metrics.get(prefix + key)
            if values:
                self.log(f'val_{prefix}{key}_avg', float(np.nanmean(values)), on_epoch=True,
                         prog_bar=(not prefix) and key in self._VAL_PROGRESS_BAR, sync_dist=True)

        # natural WT vs de novo designed vs mutant-background MegaScale libraries: the mean of rho_combined over each group
        for g, vals in groups.items():
            if vals:
                v_mean = float(np.nanmean(vals))
                if np.isfinite(v_mean):
                    self.log(f'val_{prefix}{g}_rho_combined_avg', v_mean, on_epoch=True, sync_dist=True)
                    avg_metrics[f'{prefix}{g}_rho_combined'].append(v_mean)

        # OOD benchmarks, as their own section: mean over the external loaders, never inside an aggregate.
        for _k, _v in summarise_external_benchmarks(ext_per_loader, self.external_val_sets, prefix).items():
            self.log(_k, _v, on_epoch=True, sync_dist=True)

        # The epistasis scalars are pooled over every library at once, so they are not per-library values and never entered avg_metrics. Merge them in as single-element
        # lists AFTER the val_<key>_avg logging above, so nothing is logged twice.
        if pooled['ground_truths']:
            for _k, _v in (self._log_epistasis(pooled, prefix=prefix, dump=not prefix) or {}).items():
                if np.isfinite(_v):
                    avg_metrics[prefix + _k].append(float(_v))

    def _log_epistasis(self, pooled, prefix: str = '', dump: bool = True):
        """
        Logs ``val_epi_<metric>_<head>`` (``esm_msr.epi_metrics``: the full epistasis hierarchy for epi and ctx,
        naive_rho also for comb and add, and global calibration/scale for epi), pooled over every validation library,
        and writes ``val_dump_<tag>.npz`` for offline rescoring.

        Heads: epi (~MT, evaluated relative to the WT additive baseline), comb ((WT + ~MT)/2, reported prediction, naive_rho only), add (WT, additive by construction - the control for
        what saturation alone can earn). The predictions are read through the Tobit expected reading (``obs_link``), which only the observed-scale metrics see (of the logged epistasis
        statistics, naive_rho_comb alone): every residual, pair, line and cell metric reads the latent scores, which is what makes them blind to saturation. Never raises.
        """
        try:
            keys = pooled['mut_key']
            if not keys or all(k is None for k in keys):
                return {}
            f = lambda name: np.asarray(pooled[name], dtype=float)
            n_items = len(keys)
            for field in ('dG_wt', 'ground_truths', 'dddG', 'wt_scores', 'mt_scores', 'comb_scores',
                          'cens', 'cens_src'):
                got = len(np.atleast_1d(np.asarray(pooled.get(field, []))))
                if got != n_items:    # a field on_validation_epoch_end failed to pool: fail loudly, never mis-score quietly
                    raise ValueError(f"pooled '{field}' has {got} values for {n_items} items")
            items = {'mut_key': keys, 'subset_type': list(pooled['subset_type']), 'cens': f('cens').astype(int),
                     'cens_src': f('cens_src').astype(int),
                     'ddG': f('ground_truths'), 'dddG': f('dddG'), 'dG_wt': f('dG_wt'),
                     'wt': f('wt_scores'), 'mt': f('mt_scores'), 'comb': f('comb_scores')}

            log_dir = getattr(self.logger, 'log_dir', None) or getattr(self.logger, 'save_dir', None)
            if dump and log_dir:
                tag = 'zs' if self.trainer.sanity_checking or self.global_step == 0 else f'e{self.current_epoch}'
                os.makedirs(log_dir, exist_ok=True)
                extra = {'link_summary': np.array(json.dumps(val_stats.tobit_link_summary(**self.tobit_params)))}
                for p in [p for p, _ in self.val_regimes[1:]]:
                    if len(pooled.get(f'comb_scores@{p}', [])) == len(keys):
                        extra.update({f'{h}_{p.rstrip("_")}': f(f'{h}_scores@{p}') for h in ('wt', 'mt', 'comb')})
                np.savez_compressed(
                    os.path.join(log_dir, f'val_dump_{tag}.npz'),
                    wt=items['wt'], mt=items['mt'], comb=items['comb'], dddG=items['dddG'], ground_truth=items['ddG'],
                    dG_wt=items['dG_wt'], cens=items['cens'],
                    cens_src=items['cens_src'],
                    n_mut=np.asarray(pooled['n_mut']), subset_type=np.asarray(items['subset_type'], dtype=str),
                    loader=np.asarray(pooled['loader']),
                    mut_key=np.asarray([json.dumps(k) for k in keys]), **extra)

            # The dump above carries every mutant, so any of this can be rescored offline. The metrics are
            # computed on the canonical population only - the same one the per-library metrics use: the rows the assay only bounded are held back.
            canon = items['cens_src'] != censoring.SRC_RANGE
            scored = {k: ([v[i] for i in np.flatnonzero(canon)] if isinstance(v, list) else np.asarray(v)[canon])
                      for k, v in items.items()}
            # The dump holds the latent scores and link_summary, so any scale can be rescored offline. Of the logged epistasis metrics only naive_rho_comb looks at the
            # observed scale (a Tobit run's Tobit expected reading: see docs/FEATURES.md §4.5 for what it does to its seed-to-seed spread).
            tbl = epi_metrics.Table(scored, self.obs_link)
            res = epi_metrics.compute(tbl)
            diag = epi_metrics.compute_clean_diagnostics(tbl)
            for name, val in epi_log_names(res, diag, prefix).items():
                self.log(name, val, on_epoch=True, sync_dist=True,
                         prog_bar=name == f'val_{prefix}epi_cell_rank_comb')
            cr = res.get('epi_cell_rank_comb', res.get('epi_cell_rank_epi', float('nan')))
            bl = res.get('epi_beyond_lib_comb', res.get('epi_beyond_lib_epi', float('nan')))
            if np.isfinite(cr) and np.isfinite(bl):
                comp = 0.5 * (float(cr) + float(bl))
                self.log(f"val_{prefix}epi_composite", comp, on_epoch=True, sync_dist=True)       # val_unmasked_epi_composite for the unmasked regime
                res['epi_composite'] = comp
            return res                  # the caller merges these into avg_metrics
        except Exception as e:
            logging.warning(f"epistasis metrics skipped: {e}", exc_info=True)
        return {}


    def on_validation_end(self):
        self.model._masked_logit_cache = None
        self._val_cache_dl_idx = None
        for name, p in self.model.named_parameters():                 # validation toggles requires_grad: restore what arrived
            if name in self._baseline_requires_grad:
                p.requires_grad = self._baseline_requires_grad[name]

    def configure_optimizers(self):
        base_lr = float(self.hparams.learning_rate)
        wd = 0.0                      # no weight decay, ever

        lora_wt_params, lora_mt_params, other_params, calib_wt_params, calib_mt_params = [], [], [], [], []

        # Grab the robust strings injected by PEFT
        wt_name = getattr(self.model, 'wt_adapter_name', 'wt_adapter').lower()
        mt_name = getattr(self.model, 'mt_adapter_name', 'mt_adapter').lower()

        seen = set()
        for name, p in self.model.named_parameters():
            if id(p) in seen: continue
            # Keyed on what arrived trainable, not on the current requires_grad (validation toggles it).
            if name not in self._initially_trainable:
                continue

            seen.add(id(p))
            lname = name.lower()

            if "calibration" in lname:
                if "mt" in lname: calib_mt_params.append(p)
                else: calib_wt_params.append(p)
            elif mt_name in lname: lora_mt_params.append(p)
            elif wt_name in lname or "default" in lname: lora_wt_params.append(p)
            else: other_params.append(p)

        calib_mult = config.CALIB_LR_MULT
        main_groups = []
        if lora_wt_params: main_groups.append({"params": lora_wt_params, "lr": base_lr, "weight_decay": wd, "name": "lora_wt"})
        if lora_mt_params: main_groups.append({"params": lora_mt_params, "lr": base_lr, "weight_decay": wd, "name": "lora_mt"})
        if calib_wt_params: main_groups.append({"params": calib_wt_params, "lr": base_lr * calib_mult, "weight_decay": 0.0, "name": "calib_wt"})
        if calib_mt_params: main_groups.append({"params": calib_mt_params, "lr": base_lr * calib_mult, "weight_decay": 0.0, "name": "calib_mt"})
        if other_params: main_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd, "name": "other"})

        if not main_groups:
            logging.warning("No param groups found; defaulting to all trainables.")
            main_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd, "name": "all"}]

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=torch.cuda.is_available(), weight_decay=wd)

        # Linear warmup, then hold at peak until each head's convergence metric plateaus (or immediately if --plateau_patience 0), then cosine decay to config.COSINE_MIN_LR_RATIO.
        warmup = int(self.hparams.lr_warmup_steps)
        patience = int(self.hparams.get('plateau_patience', 0) or 0)
        if not hasattr(self, '_decay_start_step'):
            self._decay_start_step = {'wt': None if patience > 0 else warmup, 'mt': None if patience > 0 else warmup}
        else:
            for h in ('wt', 'mt'):
                if h not in self._decay_start_step:
                    self._decay_start_step[h] = None if patience > 0 else warmup
        min_ratio = config.COSINE_MIN_LR_RATIO
        num_epochs = max(1, int(getattr(getattr(self, '_trainer', None), 'max_epochs', None) or self.hparams.get('num_epochs', 20) or 20))
        est_steps = getattr(self.trainer, 'estimated_stepping_batches', None) if getattr(self, '_trainer', None) is not None else None
        total_steps = int(est_steps) if (est_steps is not None and math.isfinite(est_steps) and est_steps > 0) else max(warmup + 1, num_epochs * 900)
        accum = max(1, int(self.hparams.get('grad_accum_blocks', 1) or 1))
        if accum > 1:                                       # the scheduler counts optimizer steps, one per --grad_accum_blocks blocks
            total_steps = max(warmup + 1, -(-total_steps // accum))
        steps_per_epoch = max(1, total_steps // num_epochs)
        if bool(self.hparams.get('block_domain_steps', False)) and getattr(self, '_trainer', None) is not None:
            loader = self.trainer.train_dataloader              # fit is handed a one-element list
            sampler = (loader[0] if isinstance(loader, (list, tuple)) else loader).sampler      # it knows how many blocks make a step; the estimate above counts blocks
            steps_per_epoch = max(1, int(sampler.n_steps()))
            total_steps = max(warmup + 1, -(-total_steps * steps_per_epoch // len(sampler)))
        decay_epochs = max(1, int(self.hparams.get('cosine_decay_epochs', 5) or 5))
        decay_steps = max(1, decay_epochs * steps_per_epoch)

        def _make_cos_fn(head, warmup=warmup, total_steps=total_steps, decay_steps=decay_steps, patience=patience, min_ratio=min_ratio):
            def _cos_fn(step):
                if warmup > 0 and step < warmup:
                    return max(1e-4, step / float(warmup))
                start = self._decay_start_step.get(head)
                if start is None or step <= start:
                    return 1.0
                if patience <= 0:
                    prog = min(1.0, max(0.0, (step - start) / float(max(1, total_steps - start))))
                    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))
                elapsed = step - start
                if elapsed >= decay_steps:
                    return 0.0
                prog = elapsed / float(max(1, decay_steps - 1)) if decay_steps > 1 else 1.0
                return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))
            return _cos_fn

        lambdas_main = [_make_cos_fn('wt' if g.get('name', '').endswith('_wt') else 'mt') for g in main_groups]
        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        return [opt_main], [{"scheduler": warmup_main, "interval": "step", "frequency": 1}]



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
        trainer = getattr(self, 'trainer', None)
        checkpoint['dataloader_state'] = {
            'epoch': getattr(trainer, 'current_epoch', 0) if trainer else 0,
            'global_step': getattr(trainer, 'global_step', 0) if trainer else 0,
        }
        checkpoint['schedule_state'] = {
            'decay_start_step': dict(getattr(self, '_decay_start_step', {})),
            'wt_plateau_best': getattr(self, '_wt_plateau_best', -float('inf')),
            'wt_plateau_bad': getattr(self, '_wt_plateau_bad', 0),
            'mt_plateau_best': getattr(self, '_mt_plateau_best', -float('inf')),
            'mt_plateau_bad': getattr(self, '_mt_plateau_bad', 0),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        dl_state = checkpoint.get('dataloader_state', {})
        if 'global_step' in dl_state:
            self._opt_steps = int(dl_state['global_step']) - 1
        sched_state = checkpoint.get('schedule_state', {})
        if not hasattr(self, '_decay_start_step'):
            self._decay_start_step = {}
        if 'decay_start_step' in sched_state:
            self._decay_start_step.update(sched_state['decay_start_step'])
        for head in ('wt', 'mt'):
            if f'{head}_plateau_best' in sched_state:
                setattr(self, f'_{head}_plateau_best', sched_state[f'{head}_plateau_best'])
            if f'{head}_plateau_bad' in sched_state:
                setattr(self, f'_{head}_plateau_bad', sched_state[f'{head}_plateau_bad'])


class OfflineCometLogger(CometLogger):
    """CometLogger whose experiment is written to disk (COMET_OFFLINE_DIRECTORY) for `comet upload` later.

    Lightning's own ``online=False`` is not that: it sets ``disabled=True`` and the logger silently discards everything.
    """
    def _create_experiment(self) -> None:
        import comet_ml
        self._experiment = comet_ml.start(api_key=self._api_key, workspace=self._workspace, project=self._project_name,
                                          online=False, experiment_config=comet_ml.ExperimentConfig(**self._kwargs))
        self._experiment_key = self._experiment.get_key()
        self._project_name = self._experiment.project_name


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


def _comet_api_key():
    """The Comet key from the environment, else ~/.comet.config; None if neither has one. Never from the command line."""
    key = os.environ.get('COMET_API_KEY')
    if key:
        return key
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(os.path.expanduser('~/.comet.config'))
        return cfg.get('comet', 'api_key', fallback=None)
    except Exception:
        return None


def main():
    args = parse_arguments()
    pl.seed_everything(args.seed)

    train_loaders, val_loaders, bench_loaders, train_names, val_names, bench_names = setup_dataloaders(args, add_benchmarks_to_val=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # hparams.yaml (and Comet) record what the run IS, not where things are on this machine or how to reach Comet
    hparams = {k: v for k, v in vars(args).items() if k not in UNLOGGED_ARGS}
    try:
        lightning_model = ESM3EpistasisLightningModule(val_dataloader_names=val_names, **hparams)
    except Exception as e:
        logging.error(f"Failed to initialize Lightning Module: {e}.", exc_info=True)
        return

    csv_logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name, version=args.version)
    loggers = [csv_logger]
    checkpoint_dir = csv_logger.log_dir
    if args.resume_from_checkpoint:
        import csv as _csv
        metrics_path = os.path.join(checkpoint_dir, 'metrics.csv')
        if os.path.isfile(metrics_path):
            with open(metrics_path, newline='') as f:
                reader = _csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            exp = csv_logger.experiment                      # triggers _check_log_dir_exists (which deletes metrics.csv) once
            if fields and rows:
                exp.metrics_keys = fields
                with open(metrics_path, 'w', newline='') as f:
                    writer = _csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

    if args.comet or args.comet_offline:
        key = _comet_api_key()
        if key:
            comet_kwargs = dict(api_key=key, project=args.comet_project_name, name=f"{args.experiment_name}-{args.version or 'run'}")
            if args.comet_experiment_key:
                # Comet has no notion of resuming on its own: without this a resumed run opens a SECOND
                # experiment, and the run's curves end up split across two of them.
                comet_kwargs.update(experiment_key=args.comet_experiment_key, mode='get_or_create')
            if args.comet_offline:
                offline_dir = os.path.abspath(os.path.join(args.log_dir or '.', 'comet_offline'))
                os.makedirs(offline_dir, exist_ok=True)           # comet_ml does not create it, and refuses to start without it
                os.environ['COMET_OFFLINE_DIRECTORY'] = offline_dir
                comet_logger = OfflineCometLogger(**comet_kwargs)
            else:
                comet_logger = CometLogger(**comet_kwargs)
            loggers.append(comet_logger)
        else:
            logging.warning("No Comet key (COMET_API_KEY or ~/.comet.config): this run logs to CSV only.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    monitor = monitor_metric(args)
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename=f"{{epoch:02d}}-{{{monitor}:.4f}}", save_top_k=SAVE_TOP_K,
                        monitor=monitor, mode='max', save_last=True),
        TQDMProgressBar(refresh_rate=10)
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=monitor, patience=args.early_stopping_patience, mode='max', verbose=True))

    try:
        trainer = Trainer(max_epochs=args.num_epochs, accelerator=accelerator, devices=1, logger=loggers, callbacks=callbacks,
                          enable_checkpointing=True, num_sanity_val_steps=0, log_every_n_steps=25,
                          precision=os.environ.get('ESM_MSR_PRECISION', 'bf16-mixed'))   # bf16-mixed always; fp32 only for the CPU smoke test
    except Exception as e:
        logging.error(f"Trainer init failed: {e}", exc_info=True)
        return

    ckpt = args.resume_from_checkpoint
    if ckpt:
        logging.info(f"Resuming from {ckpt}")
    else:
        # verbose=False: Lightning's own table repeats every epoch-level metric under every dataloader (one per protein), so the
        # zero-shot validation is printed once here instead
        results = trainer.validate(lightning_model, dataloaders=val_loaders, verbose=False)
        if trainer.is_global_zero:
            logging.info("Zero-shot validation:\n" + format_validation_results(results))
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders, ckpt_path=ckpt)
    if args.comet_offline and len(loggers) > 1 and trainer.is_global_zero:
        loggers[1].experiment.end()  # writes the archive to upload with `comet upload`

if __name__ == "__main__":
    main()