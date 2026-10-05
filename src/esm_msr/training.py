import os
import logging
import warnings
from collections import Counter, defaultdict
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
from esm_msr.losses import ListMLELoss
from esm_msr import censoring, link as link_mod
from esm_msr.flipkeys import split_flip_key
from esm_msr.preprocess_megascale import setup_dataloaders
from esm_msr.peft_manager import PEFTStateManager
from esm_msr.config import parse_arguments
from esm_msr import stats
from esm_msr import routing

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high') 

# A complete block of a position-pair matrix for --lambda_int_mt: rows are scored substitutions, columns are partner residues. Double-centring
# leaves nothing to learn from fewer than two of either; four rows keep the row means from being dominated by a single cell.
INT_MIN_ROWS, INT_MIN_COLS = 4, 2

class GroupPlateau:
    """ReduceLROnPlateau (mode max, relative threshold 1e-4) for the named parameter groups of one optimizer only. A group whose rate is
    already 0 (a frozen head) is left at 0 rather than lifted to ``min_lr``."""
    def __init__(self, optimizer, names, factor=0.1, patience=1, min_lr=1e-7, threshold=1e-4):
        self.groups = [g for g in optimizer.param_groups if g.get('name') in names]
        self.factor, self.patience, self.min_lr, self.threshold = factor, patience, min_lr, threshold
        self.best, self.bad = -float('inf'), 0

    def step(self, value: float):
        if value > self.best * (1 + self.threshold) if self.best > 0 else value > self.best + self.threshold:
            self.best, self.bad = value, 0
            return
        self.bad += 1
        if self.bad > self.patience:
            for g in self.groups:
                if g['lr'] > 0:
                    g['lr'] = max(g['lr'] * self.factor, self.min_lr)
            self.bad = 0


class ESM3EpistasisLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters(ignore=['tokenizer'])
        
        mt_lora_config = {
            "lora_rank": self.hparams.lora_rank_mt, "lora_alpha": self.hparams.lora_alpha_mt, "lora_dropout": self.hparams.lora_dropout_mt,
            "target_mode": self.hparams.target_mode_mt, "use_dora": self.hparams.use_dora_mt, "seed": self.hparams.seed,
            "last_n_layers": self.hparams.last_n_layers_mt,
            "incl_sequence_head": self.hparams.incl_sequence_head_mt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_mt,
        }

        wt_lora_config = {
            "lora_rank": self.hparams.lora_rank_wt, "lora_alpha": self.hparams.lora_alpha_wt, "lora_dropout": self.hparams.lora_dropout_wt,
            "target_mode": self.hparams.target_mode_wt, "use_dora": self.hparams.use_dora_wt, "seed": self.hparams.seed,
            "last_n_layers": self.hparams.last_n_layers_wt,
            "incl_sequence_head": self.hparams.incl_sequence_head_wt, "unfreeze_layernorms": self.hparams.unfreeze_layernorms_wt,
        }

        lora_config = {'wt_config': wt_lora_config, 'mt_config': mt_lora_config, 'seed': self.hparams.seed}

        self.model = MSRModel(
            lora_config=lora_config, shared_scale_init=self.hparams.shared_scale_init, shared_bias_init=self.hparams.shared_bias_init, adapter_mode=self.hparams.adapter_mode,
            model_dtype=torch.float32,
            mask_structure=self.hparams.get('mask_structure', False),
        )
        # Everything trainable at construction (adapters, calibration heads, unfrozen
        # layernorms); used to keep checkpoints adapter-only.
        self._trainable_param_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        self._warned_unrouted = False

        self.peft_manager = PEFTStateManager(self.model)

        if self.hparams.freeze_wt_adapter: self.peft_manager.freeze_wt_components()
        if self.hparams.freeze_mt_adapter: self.peft_manager.freeze_mt_components()

        # ListMLE, which is also the censored Plackett-Luce likelihood when items carry censoring (esm_msr.censoring)
        self.crit_rank_wt = ListMLELoss() if self.hparams.lambda_rank_wt > 0 else None
        self.crit_rank_mt = ListMLELoss() if self.hparams.lambda_rank_mt > 0 else None
        self.crit_reg = nn.MSELoss(reduction='none')

        # Monotone saturating link (esm_msr.link): latent stability -> the assay's observed dG. Lives on the Lightning module
        # (not the backbone wrapper) so inference and checkpoint loading of the adapters are untouched.
        self.link_head = None
        if self.hparams.get('link', 'softclamp') != 'none':
            self.link_head = link_mod.MonotoneLink(
                lo=self.hparams.link_lo, hi=self.hparams.link_hi, tau_lo=self.hparams.link_tau, tau_hi=self.hparams.link_tau,
                learn_bounds=bool(self.hparams.link_learn_bounds))

        self.automatic_optimization = False
        self.validation_step_outputs = defaultdict(list)
        self.val_dataloader_names = self.hparams.get('val_dataloader_names', ['val'])

    def on_train_start(self):
        self._reshare_base_params_on_device()
        if self.peft_manager.has_transitioned or self.hparams.freeze_wt_adapter or self.hparams.freeze_mt_adapter:
            self.peft_manager.enforce_freezing(self.optimizers(), zero_lrs=True)

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

    def _compute_rank_loss(self, pred, targets, mask, list_size, crit_fn, cens=None):
        """Listwise rank loss over consecutive blocks of ``list_size``. ``cens`` (-1/0/+1 per row, optional) makes the lists
        two-sided censored (``ListMLELoss.forward_censored``); it is used only when some row of the blocks is censored, so
        without censored rows the call is exactly the uncensored one."""
        valid_len = (pred.shape[0] // list_size) * list_size
        if valid_len > 0:
            pred_rank = pred[:valid_len].view(-1, list_size)
            targ_rank = targets[:valid_len].view(-1, list_size)
            mask_rank = mask[:valid_len].view(-1, list_size)
            if cens is not None and bool(((cens[:valid_len] != 0) & mask[:valid_len]).any()):
                L_raw = crit_fn.forward_censored(pred_rank, targ_rank, mask_rank, cens[:valid_len].view(-1, list_size))
            else:
                L_raw = crit_fn(pred_rank, targ_rank, mask=mask_rank)
            avg_len = mask_rank.float().sum(dim=-1).mean()
            scaled_loss = L_raw * (list_size / avg_len.clamp(min=1.0))
            num_lists = valid_len // list_size
            return scaled_loss, L_raw.detach(), num_lists
        return None, 0.0, 0

    def _compute_flip_loss(self, pred, targets, valid, flip_keys, crit_fn, min_len, cens=None):
        """Within-column rank loss on the MT pass - the flip-signature objective.

        A *flip column* is one scored position with one fixed partner identity, over the
        substitutions available at that position. Inside a column the conditional target
        ddG(A|B) differs from the double's ddG_AB only by the constant ddG_B, so ordering by
        either is the same ordering; imposing it is therefore equivalent to reproducing the
        measured ordering of the double-mutant phenotypes.

        Why this and not regression: an ordering within a column is invariant to any monotone
        function of the underlying stability, so the assay's saturating response and its
        dynamic-range floor cannot be fitted by it at all. A regression on ddG can, and does.

        One honest caveat, because it decides how to read the metric. Under strict additivity
        every column shares ONE ordering, so a purely additive model already satisfies much of
        this loss; what it cannot satisfy is the per-column *deviation* from that consensus,
        which is the identity-dependent part. So this loss is artifact-immune and includes the
        interaction term, but is not exclusively about it. The matching validation metric
        (``val_rho_flip``) double-centres away the consensus and therefore IS exclusive - use
        the loss to train and the metric to judge.

        Columns are variable length, so they are padded into a [G, L] block with a mask
        rather than reshaped.

        ``cens`` (optional -1/0/+1 per row, see ``esm_msr.censoring``) marks items that only bound the true value:
        lower-censored ones (dead, or pinned at the assay floor) rank below every uncensored member and upper-censored
        ones (hyperstable) above it, with their order among themselves not scored (two-sided censored Plackett-Luce,
        ``ListMLELoss.forward_censored``). A column counts toward ``min_len`` by total membership but needs >= 1
        uncensored member to carry any information.
        """
        groups = {}
        for i, k in enumerate(flip_keys):
            if k and bool(valid[i]):
                groups.setdefault(k, []).append(i)
        groups = [g for g in groups.values() if len(g) >= min_len
                  and (cens is None or any(int(cens[i]) == 0 for i in g))]
        if not groups:
            return None, 0.0, 0
        L = max(len(g) for g in groups)
        G = len(groups)
        dev = pred.device
        p = torch.zeros(G, L, device=dev, dtype=pred.dtype)
        t = torch.zeros(G, L, device=dev, dtype=pred.dtype)
        m = torch.zeros(G, L, device=dev, dtype=torch.bool)
        cb = torch.zeros(G, L, device=dev, dtype=torch.long)
        for gi, g in enumerate(groups):
            idx = torch.as_tensor(g, device=dev, dtype=torch.long)
            p[gi, :len(g)] = pred[idx]
            t[gi, :len(g)] = targets[idx]
            m[gi, :len(g)] = True
            if cens is not None:
                cb[gi, :len(g)] = cens[idx]
        if cens is not None and bool(((cb != 0) & m).any()):
            L_raw = crit_fn.forward_censored(p, t, m, cb)
        else:
            L_raw = crit_fn(p, t, mask=m)
        avg_len = m.float().sum(dim=-1).mean()
        scaled = L_raw * (L / avg_len.clamp(min=1.0))
        n_unc = int((m & (cb == 0)).sum())
        self._flip_diag = (G, float(avg_len), int(sum(len(g) for g in groups)), n_unc)
        return scaled, L_raw.detach(), G

    def _compute_int_loss(self, pred, target, valid, flip_keys, row_ids, cens, min_rows, min_cols):
        """Interaction-only loss on the MT pass: squared error between DOUBLE-CENTRED predictions and measurements.

        Rows sharing a position pair form a matrix (rows = scored substitutions, columns = partner residues). Each matrix is trimmed to a
        complete block (the most-missing row or column is dropped until no cell is missing), then predictions and targets are double-centred
        separately: subtract each row's mean and each column's mean and add back the grand mean. What survives depends only on the specific
        combination of the two residues. The position-pair offset, each substitution's own effect and each partner's own effect are exactly
        annihilated on both sides, so this loss neither teaches nor unteaches them; they are learned through the ordinary regression.
        Censored items are left out. Returns ``(sum of squared differences, its detached value, n_matrices, n_cells, sum of squared double-centred targets)``, with the sum so
        the caller can normalise by the whole batch; ``(None, 0.0, 0, 0, 0.0)`` when no usable matrix is present.
        """
        groups = {}
        for i, k in enumerate(flip_keys):
            if not k or not bool(valid[i]) or (cens is not None and int(cens[i]) != 0):
                continue
            pair, res = split_flip_key(k)
            if res is None:
                continue
            groups.setdefault(pair, {}).setdefault(res, {})[int(row_ids[i])] = i
        total, n_cells, n_mat, ss_y = None, 0, 0, 0.0
        for pair, cols in groups.items():
            names = sorted(cols)
            rows = sorted({r for c in cols.values() for r in c})
            M = [[cols[c].get(r, -1) for c in names] for r in rows]
            while M and M[0]:
                miss_r = [sum(1 for v in row if v < 0) for row in M]
                miss_c = [sum(1 for row in M if row[j] < 0) for j in range(len(M[0]))]
                if max(miss_r) == 0 and max(miss_c) == 0:
                    break
                if max(miss_r) / max(1, len(M[0])) >= max(miss_c) / max(1, len(M)):
                    M.pop(miss_r.index(max(miss_r)))
                else:
                    j = miss_c.index(max(miss_c))
                    M = [row[:j] + row[j + 1:] for row in M]
            if len(M) < min_rows or not M or len(M[0]) < min_cols:
                continue
            idx = torch.as_tensor(M, device=pred.device, dtype=torch.long)
            P, Y = pred[idx], target[idx]
            dc = lambda X: X - X.mean(dim=1, keepdim=True) - X.mean(dim=0, keepdim=True) + X.mean()
            part = ((dc(P) - dc(Y)) ** 2).sum()
            ss_y += float((dc(Y) ** 2).sum())
            total = part if total is None else total + part
            n_cells += idx.numel()
            n_mat += 1
        if total is None:
            return None, 0.0, 0, 0, 0.0
        return total, float(total.detach()), n_mat, n_cells, ss_y

    def _subset_weights(self, subset_types, device) -> torch.Tensor:
        """Per-item loss weight from its subset type (1.0 for singles and unknown types)."""
        hp = self.hparams
        weight_by_subset = {
            'cond': hp.cond_weight,
            'native_cond': hp.native_cond_weight,
        }
        return torch.tensor([float(weight_by_subset.get(routing.canonical_subset(s), 1.0)) for s in subset_types],
                            dtype=torch.float32, device=device)

    def _plan_units(self, batch: dict, wt_rows, mt_rows, mb: int):
        """
        Group a batch into work units, each of which runs exactly ONE adapter.

        A unit is ``(kind, rows)`` with kind 'wt' or 'mt', so no unit alternates between the adapters in the inner loop. Singles can
        appear in both a 'wt' unit (wild-type-context regression) and an 'mt' unit (the single anchor): the same two forward passes
        as before, just split so each runs alone.

        WT units are sized by how many *distinct* backbone inputs they contain rather than by micro_batch_size: every item of a
        library shares one wild-type sequence, so the whole WT block is a single forward (the backbone runs once per unique input)
        and splitting it would only repeat that forward.

        MT units are cut at position-pair boundaries whenever a flip loss is on, so no pair matrix (and no flip column) is split
        across two micro-batches; otherwise every ``mb`` rows.
        """
        units = []
        if len(wt_rows):
            size = mb
            if getattr(self.model, 'dedup_backbone', False):
                first, _ = self.model._unique_rows(batch['wt_sequence_tokens'][wt_rows],
                                                   batch['coords'][wt_rows],
                                                   batch['structure_tokens'][wt_rows])
                if first.numel() <= mb:
                    size = int(len(wt_rows))
            units += [('wt', wt_rows[s:s + size]) for s in range(0, len(wt_rows), size)]
        if len(mt_rows):
            if self.hparams.lambda_rank_mt > 0 or self.hparams.get('lambda_int_mt', 0.0) > 0:
                units += [('mt', mt_rows[c]) for c in self._aligned_chunks(mt_rows, mb)]
            else:
                units += [('mt', mt_rows[s:s + mb]) for s in range(0, len(mt_rows), mb)]
        return units

    def _aligned_chunks(self, mt_rows, mb):
        """
        Index chunks of at most ``mb`` MT rows that never cut a position-pair group.

        Rows are first ordered so each pair's columns are contiguous (rows with no flip key, such as anchored singles, last), then
        whole pairs are packed greedily into chunks; a pair larger than ``mb`` is cut at column boundaries when it must be. Returns
        long index tensors into ``mt_rows``.
        """
        keys = getattr(self, '_last_flip_keys', None) or []
        info = []
        for pos in range(len(mt_rows)):
            fk = keys[int(mt_rows[pos])] if int(mt_rows[pos]) < len(keys) else ''
            pair, res = split_flip_key(fk)
            info.append((fk == '', pair or '', fk, pos))
        info.sort()
        groups, cur_key = [], object()
        for no_key, pair, fk, pos in info:
            gk = (no_key, pair) if not no_key else (True, pos)          # rows without a key are free-standing
            if gk != cur_key:
                groups.append([])
                cur_key = gk
            groups[-1].append((fk, pos))
        chunks, cur = [], []
        for g in groups:
            if len(g) > mb:                                            # a pair larger than a micro-batch: cut it at column boundaries
                cols, last_fk = [], None
                for fk, pos in g:
                    if fk != last_fk:
                        cols.append([])
                        last_fk = fk
                    cols[-1].append(pos)
                for col in cols:
                    while len(col) > mb:                               # a single column longer than a micro-batch
                        if cur:
                            chunks.append(cur); cur = []
                        chunks.append(col[:mb]); col = col[mb:]
                    if len(cur) + len(col) > mb:
                        chunks.append(cur); cur = []
                    cur.extend(col)
                continue
            if len(cur) + len(g) > mb:
                chunks.append(cur); cur = []
            cur.extend(p for _, p in g)
        if cur:
            chunks.append(cur)
        return [torch.as_tensor(c, dtype=torch.long, device=mt_rows.device) for c in chunks if c]

    def _compose_losses_streaming_and_backward(self, batch: dict) -> dict:
        """
        Computes every loss for one batch and back-propagates each work unit immediately, so
        at most one unit's activations are alive at a time.

        Targets and masks are derived once for the whole batch, then units are planned by
        :meth:`_plan_units` so that each one drives a single adapter. Routing comes from
        ``esm_msr.routing``:

        * WT head - ``single`` items, on their measured ddG.
        * MT head - ``cond`` and ``native_cond`` on their own ddG, plus, when
          ``mt_single_anchor_weight > 0``, ordinary singles (the zero-background case of the
          MT task, with clean measured labels).
        * Any other subset has no head in training and is skipped (doubles reach the MT head
          as their two ``cond`` items; ``--subset_caps`` rejects ``double`` and ``reversion``).
        """
        hp = self.hparams
        device, B = batch['ddG'].device, int(batch['ddG'].shape[0])
        list_size = max(1, int(hp.wt_list_size))
        mb = min(B, int(hp.get('micro_batch_size', 32)))

        st_all = list(batch.get('subset_type', ['single'] * B))
        flip_keys = list(batch.get('flip_key', [''] * B))
        if len(flip_keys) != B:
            flip_keys = [''] * B
        w_all = self._subset_weights(st_all, device)
        global_w_sum = w_all.sum().clamp_min(1e-9)
        global_num_lists = max(1, B // list_size)
        # How many flip columns the whole batch offers, so each micro-batch's contribution
        # is weighted by its share (mirrors global_num_lists for the ListMLE terms).
        self._last_flip_keys = flip_keys
        # Censoring (esm_msr.censoring): -1 lower-bounded, +1 upper-bounded, 0 ordinary. All zeros without
        # --include_out_of_range / --censor_floor, in which case every path below reduces to the uncensored one.
        cens_all = batch['cens'].to(device) if torch.is_tensor(batch.get('cens')) else torch.zeros(B, dtype=torch.long, device=device)
        cens_bound = batch['cens_bound'].float().to(device) if torch.is_tensor(batch.get('cens_bound')) else torch.full((B,), float('nan'), device=device)
        cens_src = batch['cens_src'].to(device) if torch.is_tensor(batch.get('cens_src')) else torch.zeros(B, dtype=torch.long, device=device)
        use_cens = bool((cens_all != 0).any())
        _dbg = os.environ.get('MSR_CENS_DEBUG', '')       # bisecting a flag-dependent crash: 'ignore' = data only, 'force' = loss-code path only
        if _dbg == 'ignore':
            cens_all, use_cens = torch.zeros_like(cens_all), False
        elif _dbg == 'force':
            use_cens = True
        self._cens_diag = (int((cens_all < 0).sum()), int((cens_all > 0).sum())) if use_cens else None
        # Regression sees a censored item as a bound (hinge) when it has no usable value (out of range), or if asked for floor items.
        hinge_w = float(hp.get('censor_reg_weight', 1.0))
        hinge_src = (cens_src == censoring.SRC_RANGE) | bool(hp.get('censor_floor_hinge', False))
        reg_cens = torch.where((cens_all != 0) & hinge_src, cens_all, torch.zeros_like(cens_all))
        # Monotone link (esm_msr.link): regression on the observed scale, observed = h(dG_wt + background + latent) - dG_wt.
        link = self.link_head
        use_link = link is not None
        dGwt_all = batch['dG_wt'].float().to(device) if torch.is_tensor(batch.get('dG_wt')) else torch.full((B,), float('nan'), device=device)
        bg_all = batch['bg_offset'].float().to(device) if torch.is_tensor(batch.get('bg_offset')) else torch.zeros(B, device=device)
        link_ok = torch.isfinite(dGwt_all) if use_link else torch.ones(B, dtype=torch.bool, device=device)
        global_flip_items = max(1, sum(1 for i, k in enumerate(flip_keys) if k and not (use_cens and int(cens_all[i]) != 0)))
        _fk = Counter(k for i, k in enumerate(flip_keys) if k)
        _unc = Counter(k for i, k in enumerate(flip_keys) if k and not (use_cens and int(cens_all[i]) != 0))
        global_num_flip = max(1, sum(1 for k, c in _fk.items() if c >= int(hp.flip_list_min) and _unc[k] > 0))

        anchor_w = float(hp.get('mt_single_anchor_weight', 0.0) or 0.0)
        if anchor_w > 0 and hp.lambda_reg_mt <= 0:
            raise AssertionError('mt_single_anchor_weight > 0 requires lambda_reg_mt > 0.')
        wt_frozen, mt_frozen = self.peft_manager.wt_path_is_frozen, self.peft_manager.mt_path_is_frozen

        # ---------------- per-item targets and masks, derived once ----------------
        ddG = batch['ddG'].float()

        is_wt_subset = routing.subset_mask(st_all, routing.WT_HEAD_SUBSETS, device)
        is_mt_subset = routing.subset_mask(st_all, routing.MT_HEAD_SUBSETS, device)

        if not self._warned_unrouted:
            unrouted = sorted({s for s in st_all if routing.canonical_subset(s) not in routing.WT_HEAD_SUBSETS | routing.MT_HEAD_SUBSETS})
            if unrouted:
                logging.warning(f"Subsets {unrouted} have no head in training and are excluded from all losses (see esm_msr.routing).")
                self._warned_unrouted = True

        # WT head: the measured ddG of single mutations; a censored single is only a bound, so it is withheld from the ordinary regression.
        wt_ok = is_wt_subset & torch.isfinite(ddG)
        wt_cens = torch.where(is_wt_subset, cens_all, torch.zeros_like(cens_all))
        wt_reg_cens = torch.where(is_wt_subset, reg_cens, torch.zeros_like(reg_cens))

        # MT head: its own subsets, plus anchored singles. Anchored singles are the bulk of
        # the MT pass's backbone rows, so `mt_single_anchor_frac` subsamples them per step;
        # weights are scaled by 1/frac to leave the anchor's expected contribution unchanged.
        mt_w = torch.where(is_mt_subset, w_all, torch.zeros_like(w_all))
        if anchor_w > 0:
            anchor_rows = is_wt_subset
            frac = float(hp.get('mt_single_anchor_frac', 1.0) or 1.0)
            if not 0.0 < frac <= 1.0:
                raise AssertionError(f'mt_single_anchor_frac must be in (0, 1]; got {frac}.')
            if frac < 1.0 and bool(anchor_rows.any()):
                gen = torch.Generator(device='cpu').manual_seed(int(self.global_step) * 7919 + 13)
                keep = torch.rand(int(anchor_rows.sum()), generator=gen).to(device) < frac
                sampled = torch.zeros_like(anchor_rows)
                sampled[anchor_rows.nonzero(as_tuple=True)[0][keep]] = True
                anchor_rows = sampled
            mt_w = torch.where(anchor_rows, w_all * (anchor_w / frac), mt_w)
        mt_ok = (mt_w > 0) & torch.isfinite(ddG)

        # Items whose absolute target is trustworthy enough for the REGRESSION terms (without the link). The dataset marks sub-floor
        # double-derived items reg_ok=False (see MutationStabilityDataset._drop_unreachable): the assay reports them without any flag,
        # roughly half are unidentifiable fits, and a third are genuine compensation. Their ordering is informative, their value is
        # not, so they stay in the rank losses and are withheld here. With the link the saturation is h's job and reg_ok is ignored.
        reg_keep = batch.get('reg_ok')
        reg_keep = (reg_keep.to(device) if torch.is_tensor(reg_keep)
                    else torch.ones(B, dtype=torch.bool, device=device))

        # ---------------- plan ----------------
        idx = torch.arange(B, device=device)
        train_wt = (not wt_frozen) and (hp.lambda_reg_wt > 0 or hp.lambda_rank_wt > 0)
        train_mt = (not mt_frozen) and (hp.lambda_reg_mt > 0 or hp.lambda_rank_mt > 0 or hp.get('lambda_int_mt', 0.0) > 0)

        wt_rows = idx[wt_ok] if train_wt else idx[:0]
        mt_rows = idx[mt_ok] if train_mt else idx[:0]
        units = self._plan_units(batch, wt_rows, mt_rows, mb)

        zero = torch.zeros((), device=device)
        sums, cnts = defaultdict(lambda: zero), defaultdict(lambda: zero)

        for kind, rows in units:
            # A unit with no loss term runs no backward, so its activation graph stays alive through these references; drop
            # them before this unit's forward allocates a second copy (this doubled peak memory and caused OOM / driver errors).
            wt_pred_cal = wt_pred_raw = mt_pred_cal = mt_pred_raw = None
            p_wt = p_mt = p_int = None
            L = Lh = L_flip = L_int = L_rank = total = None
            losses_wt, losses_mt = [], []
            micro, w_mb = utils.slice_batch_by_index(batch, rows), w_all[rows]
            if os.environ.get('MSR_MEM_DEBUG') and torch.cuda.is_available():
                # one line per work unit: the PREVIOUS unit's high-water mark, then this unit's kind / rows / token shape
                logging.info(f"UNIT prev_peak={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB kind={kind} rows={len(rows)} "
                             f"tokens={tuple(micro['wt_sequence_tokens'].shape)} alloc={torch.cuda.memory_allocated() / 2 ** 30:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
            m_wt_ok, m_mt_ok = wt_ok[rows], mt_ok[rows]

            # ---- WT unit ----
            if kind == 'wt':
                wt_out = self.model.forward_partitioned(micro, pass_type='wt', mask_strategy=hp.mask_strategy)
                wt_pred_cal, wt_pred_raw = wt_out['pred_calibrated'].float(), wt_out['pred_raw'].float()
                del wt_out

                if m_wt_ok.any():
                    t = ddG[rows]
                    if hp.lambda_reg_wt > 0:
                        c_reg = wt_reg_cens[rows]
                        if use_link:
                            # scored on the observed scale; a single has no background
                            reg_rows = m_wt_ok & link_ok[rows]
                            p_wt = link.obs_ddG(wt_pred_cal, dGwt_all[rows])
                        else:
                            reg_rows, p_wt = m_wt_ok, wt_pred_cal
                        ord_wt = reg_rows & (c_reg == 0)
                        if ord_wt.any():
                            L = self.crit_reg(p_wt[ord_wt], t[ord_wt]) * w_mb[ord_wt]
                            losses_wt.append(hp.lambda_reg_wt * L.sum() / global_w_sum)
                            sums['reg_wt'] = sums['reg_wt'] + L.sum().detach()
                            cnts['reg_wt'] = cnts['reg_wt'] + w_mb[ord_wt].sum()
                        cen_wt = reg_rows & (c_reg != 0) & torch.isfinite(cens_bound[rows])
                        if hinge_w > 0 and cen_wt.any():
                            Lh = censoring.censored_regression_loss(self.crit_reg, p_wt[cen_wt], cens_bound[rows][cen_wt],
                                                                    c_reg[cen_wt]) * w_mb[cen_wt] * hinge_w
                            losses_wt.append(hp.lambda_reg_wt * Lh.sum() / global_w_sum)
                            sums['reg_wt_cens'] = sums['reg_wt_cens'] + Lh.sum().detach()
                            cnts['reg_wt_cens'] = cnts['reg_wt_cens'] + w_mb[cen_wt].sum()
                    if hp.lambda_rank_wt > 0 and self.crit_rank_wt is not None:
                        L_rank, val, n_list = self._compute_rank_loss(
                            wt_pred_raw, torch.nan_to_num(t), m_wt_ok, list_size, self.crit_rank_wt,
                            cens=wt_cens[rows] if use_cens else None)
                        if L_rank is not None:
                            losses_wt.append(hp.lambda_rank_wt * L_rank * (n_list / global_num_lists))
                            sums['rank_wt'] = sums['rank_wt'] + val * n_list
                            cnts['rank_wt'] = cnts['rank_wt'] + n_list

                if losses_wt:
                    total = sum(losses_wt)
                    if not torch.isfinite(total): raise AssertionError("WT Loss evaluated to NaN/Inf.")
                    if not total.requires_grad: raise AssertionError("WT Loss detached from PyTorch Graph! Cannot call backward.")
                    self.manual_backward(total)
                continue

            # ---- MT unit ----
            mt_out = self.model.forward_partitioned(micro, pass_type='mt', mask_strategy=hp.mask_strategy)
            mt_pred_cal, mt_pred_raw = mt_out['pred_calibrated'].float(), mt_out['pred_raw'].float()
            del mt_out

            if m_mt_ok.any() and hp.lambda_reg_mt > 0:
                w = mt_w[rows]
                if use_link:
                    # the saturation is h's job, so items below the floor need no special handling (reg_ok is ignored); a
                    # conditional item is scored as the double it came from: observed ddG_AB = h(dG_wt + ddG_B + ddG(A|B)) - dG_wt
                    reg_ok = m_mt_ok & link_ok[rows]
                    p_mt = link.obs_ddG(mt_pred_cal, dGwt_all[rows], bg_all[rows])
                    t_mt, b_mt = ddG[rows] + bg_all[rows], cens_bound[rows] + bg_all[rows]
                else:
                    reg_ok = m_mt_ok & reg_keep[rows] if hp.subfloor_rank_only else m_mt_ok
                    p_mt, t_mt, b_mt = mt_pred_cal, ddG[rows], cens_bound[rows]
                c_reg = reg_cens[rows]
                reg_ord = reg_ok & (c_reg == 0)
                if reg_ord.any():
                    L = self.crit_reg(p_mt[reg_ord], t_mt[reg_ord]) * w[reg_ord]
                    losses_mt.append(hp.lambda_reg_mt * L.sum() / global_w_sum)
                    sums['reg_mt'] = sums['reg_mt'] + L.sum().detach()
                    cnts['reg_mt'] = cnts['reg_mt'] + w[reg_ord].sum()
                cen_mt = m_mt_ok & link_ok[rows] & (c_reg != 0) & torch.isfinite(b_mt)
                if hinge_w > 0 and cen_mt.any():
                    Lh = censoring.censored_regression_loss(self.crit_reg, p_mt[cen_mt], b_mt[cen_mt],
                                                            c_reg[cen_mt]) * w[cen_mt] * hinge_w
                    losses_mt.append(hp.lambda_reg_mt * Lh.sum() / global_w_sum)
                    sums['reg_mt_cens'] = sums['reg_mt_cens'] + Lh.sum().detach()
                    cnts['reg_mt_cens'] = cnts['reg_mt_cens'] + w[cen_mt].sum()
            if hp.lambda_rank_mt > 0 and self.crit_rank_mt is not None:
                fk_rows = [flip_keys[int(r)] for r in rows]
                L_flip, val, n_grp = self._compute_flip_loss(
                    mt_pred_raw, ddG[rows], m_mt_ok, fk_rows,
                    self.crit_rank_mt, hp.flip_list_min,
                    cens=cens_all[rows] if use_cens else None)
                if L_flip is not None:
                    losses_mt.append(hp.lambda_rank_mt * L_flip * (n_grp / max(global_num_flip, 1)))
                    sums['rank_mt'] = sums['rank_mt'] + val * n_grp
                    cnts['rank_mt'] = cnts['rank_mt'] + n_grp
            if hp.get('lambda_int_mt', 0.0) > 0:
                if use_link:
                    p_int = link.obs_ddG(mt_pred_cal, dGwt_all[rows], bg_all[rows])
                    t_int = ddG[rows] + bg_all[rows]
                else:
                    p_int, t_int = mt_pred_cal, ddG[rows]
                fk_int = [flip_keys[int(r)] for r in rows]
                mt_id_rows = batch['mt_id'][rows][:, 0].detach().cpu().numpy()
                L_int, val_int, n_mat, n_cells, ss_tgt = self._compute_int_loss(
                    p_int, t_int, m_mt_ok & link_ok[rows], fk_int, mt_id_rows,
                    cens_all[rows] if use_cens else None, INT_MIN_ROWS, INT_MIN_COLS)
                if L_int is not None:
                    losses_mt.append(hp.lambda_int_mt * L_int / global_flip_items)
                    sums['int_mt'] = sums['int_mt'] + val_int
                    cnts['int_mt'] = cnts['int_mt'] + n_cells
                    sums['int_tgt'] = sums['int_tgt'] + ss_tgt      # with L_int_mt: the share of interaction variance left unexplained
                    cnts['int_tgt'] = cnts['int_tgt'] + n_cells

            if losses_mt:
                total = sum(losses_mt)
                if not torch.isfinite(total): raise AssertionError("MT Loss evaluated to NaN/Inf.")
                if not total.requires_grad: raise AssertionError("MT Loss detached from PyTorch Graph! Cannot call backward.")
                self.manual_backward(total)

        # One host sync for all logged values instead of one per unit and loss term.
        keys = [k for k in ('reg_wt', 'rank_wt', 'reg_wt_cens', 'reg_mt', 'reg_mt_cens', 'rank_mt', 'int_mt', 'int_tgt') if k in cnts]
        if not keys:
            return {}
        vals = torch.stack([torch.stack([torch.as_tensor(sums[k], device=device, dtype=torch.float32),
                                         torch.as_tensor(cnts[k], device=device, dtype=torch.float32)]) for k in keys]).tolist()
        return {f'L_{k}': s_ / c_ for k, (s_, c_) in zip(keys, vals) if c_ > 0}

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
        
        # Norm diagnostics only at the log cadence (log_every_n_steps): the
        # param->name dict build plus three norm reductions over every parameter
        # group ran every step and forced host syncs that dominated step time.
        if self.global_step % self.trainer.log_every_n_steps == 0:
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
        
        sch_warmup = self.lr_schedulers()
        total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
        if sch_warmup.last_epoch < total_warmup_steps:
            sch_warmup.step()

        self._log_lrs()
        self.log_calibration_head(on_step=True)
        if self.link_head is not None:
            for k, v in self.link_head.summary().items():
                if k in ('floor', 'ceiling'):
                    continue            # ceiling == hi exactly and floor == lo to within ~1e-3: the same curves under two names
                self.log(f"link/{k}", v, on_step=True)

        if torch.cuda.is_available() and self.global_step % max(int(self.trainer.log_every_n_steps), 1) == 0:
            # live tensors vs what the caching allocator holds, and the high-water mark since the last log: tells an
            # activation-bound run (peak tracks the micro-batch) from an allocator-held one (reserved >> peak)
            gb = 1.0 / 2 ** 30
            self.log("mem/allocated_gb", torch.cuda.memory_allocated() * gb, on_step=True)
            self.log("mem/peak_allocated_gb", torch.cuda.max_memory_allocated() * gb, on_step=True, prog_bar=True)
            self.log("mem/reserved_gb", torch.cuda.memory_reserved() * gb, on_step=True, prog_bar=True)
            torch.cuda.reset_peak_memory_stats()
        for k, v in logs.items():
            if v > 0.0: self.log(f"train/{k}", v, on_step=True)
        if getattr(self, '_cens_diag', None) is not None:
            self.log("train/cens_lower_items", float(self._cens_diag[0]), on_step=True)
            self.log("train/cens_upper_items", float(self._cens_diag[1]), on_step=True)
            self._cens_diag = None
        if getattr(self, '_flip_diag', None) is not None:
            g, alen, nitems, n_unc = self._flip_diag
            self.log("train/flip_cols", float(g), on_step=True)
            self.log("train/flip_len", float(alen), on_step=True)
            self.log("train/flip_items", float(nitems), on_step=True)
            self.log("train/flip_uncensored", float(n_unc), on_step=True)
            self._flip_diag = None
            
        if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
            self.log("amp_scale", self.trainer.precision_plugin.scaler.get_scale(), on_step=True)

        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        with torch.inference_mode():
            out_dict = self.model.forward_batch(batch, mask_strategy=self.hparams.mask_strategy)

        ddG = utils._get_label(batch, 'ddG', device=batch['ddG'].device)
        dddG = batch.get('dddG')
        n_items = int(out_dict['wt_lora_pred'].shape[0])

        def _np(t):
            return t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.full(n_items, float(t))

        mt_id = batch.get('mt_id')
        row_id = (mt_id[:, 0].detach().cpu().numpy() if torch.is_tensor(mt_id) and mt_id.ndim == 2
                  else np.full(n_items, -1))
        extra = {}
        if self.link_head is not None and torch.is_tensor(batch.get('dG_wt')):
            # observed-scale predictions, h(dG_wt + latent) - dG_wt, for the absolute-error and ddG-ddG epistasis metrics
            dgw = batch['dG_wt'].float().to(out_dict['wt_lora_pred'].device)
            with torch.no_grad():
                for k, key in (('wt', 'wt_lora_pred'), ('mt', 'mt_lora_pred'), ('comb', 'combined_pred')):
                    v = out_dict[key]
                    extra[f'{k}_obs'] = _np(self.link_head.obs_ddG(v.float(), dgw)) if torch.is_tensor(v) else np.full(n_items, float('nan'))
        self.validation_step_outputs[dataloader_idx].append({
            **extra,
            'wt_scores': _np(out_dict['wt_lora_pred']),
            'mt_scores': _np(out_dict['mt_lora_pred']),
            'comb_scores': _np(out_dict['combined_pred']),
            'ground_truths': _np(ddG) if ddG is not None else np.full(n_items, np.nan),
            'dddG': _np(dddG) if dddG is not None else np.full(n_items, np.nan),
            'subset_type': list(batch.get('subset_type', ['single'] * n_items)),
            'cens': (batch['cens'].detach().cpu().numpy() if torch.is_tensor(batch.get('cens')) else np.zeros(n_items, dtype=int)),
            # For val_rho_flip: the column key, and the substitution identity that indexes
            # the row within that column.
            'flip_key': list(batch.get('flip_key', [''] * n_items)),
            'row_id': row_id,
            # Hashable per-item mutation tuple, so rho_epi_full can pair each double with its
            # two singles (comb_AB - comb_A - comb_B).
            'mut_key': [tuple(tuple(m) for m in muts) for muts in batch.get('mutations', [()] * n_items)],
        })

    # What validation logs. Per protein only these three (a library's rank and error, plus its interaction score if it has doubles).
    _VAL_PER_PROTEIN = ('rho_combined', 'rmse_combined', 'rho_flip_pair')
    # Library-equal means. rho_combined and rho_wt_valid are read by the checkpoint name, the plateau scheduler and the convergence logic.
    _VAL_AVG = ('rho_combined', 'rmse_combined', 'rho_wt_valid', 'rho_mt_valid',
                'rho_epi_full', 'rho_colrank', 'rho_colrank_wt', 'rho_flip', 'rho_flip_pair', 'auc_dead_wt', 'auc_dead_mt')
    # Pooled over every item of every library: the pair-level components need pooling to have enough pairs.
    _VAL_POOLED = ('rho_combined', 'rmse_combined', 'rho_epi_full', 'rho_pair_offset', 'rho_row_effect', 'rho_col_effect',
                   'rho_colrank', 'rho_colrank_wt', 'rho_flip', 'rho_flip_pair', 'auc_dead_wt', 'auc_dead_mt')
    _VAL_PROGRESS_BAR = ('rho_combined', 'rmse_combined', 'rho_wt_valid', 'rho_flip_pair')

    def on_validation_epoch_start(self):
        self.validation_step_outputs = defaultdict(list)

    def on_validation_epoch_end(self):
        """
        Logs, per dataloader, the three head metrics and the calibration RMSE from
        ``stats.compute_metrics``, their means across dataloaders (``*_avg``, which the
        checkpoint monitor and the plateau scheduler read), and the same metrics pooled
        over every item of every dataloader (``*_pooled``), which weights proteins by
        their size instead of equally.
        """
        per_loader, pooled = {}, defaultdict(list)
        n_flip_matrices = 0

        for dataloader_idx, outputs in self.validation_step_outputs.items():
            name = (self.val_dataloader_names[dataloader_idx]
                    if dataloader_idx < len(self.val_dataloader_names) else f"unknown_dl_{dataloader_idx}")
            if not outputs:
                continue

            cols = {k: np.concatenate([np.asarray(o[k]).reshape(-1) for o in outputs])
                    for k in ('wt_scores', 'mt_scores', 'comb_scores', 'ground_truths', 'dddG')}
            subset_types = [s for o in outputs for s in o['subset_type']]
            cens_val = np.concatenate([np.asarray(o['cens']).reshape(-1) for o in outputs])
            mut_keys = [k for o in outputs for k in o.get('mut_key', [])]
            if len(mut_keys) != len(subset_types):
                mut_keys = None

            obs_val = ({k: np.concatenate([np.asarray(o[f'{k}_obs']).reshape(-1) for o in outputs]) for k in ('wt', 'mt', 'comb')}
                       if all('comb_obs' in o for o in outputs) else None)
            per_loader[name] = stats.compute_metrics(
                cols['wt_scores'], cols['mt_scores'], cols['comb_scores'],
                cols['ground_truths'], subset_types, dddG=cols['dddG'], mut_keys=mut_keys, cens=cens_val, obs=obs_val)

            # Identity-dependent interaction, scored on the MT pass. This is the only
            # validation number that is specific to what the MT adapter exists for: it is
            # exactly zero for an additive readout and unaffected by the assay's monotone
            # response, so unlike rho_combined it cannot be satisfied by learning saturation.
            fk = [k for o in outputs for k in o.get('flip_key', [])]
            rid = np.concatenate([np.asarray(o['row_id']).reshape(-1) for o in outputs]) \
                if all('row_id' in o for o in outputs) else np.array([])
            if len(fk) == len(cols['mt_scores']) and len(rid) == len(fk):
                rho_flip, n_pairs, n_cells = stats.flip_signature_rho(
                    cols['mt_scores'], np.where(cens_val == 0, cols['ground_truths'], np.nan), fk, rid,
                    min_len=int(self.hparams.flip_list_min))
                per_loader[name]['rho_flip'] = rho_flip
                # The same statistic with one matrix per position pair: nothing that ignores the partner residue can score.
                rho_pair, n_pp, _ = stats.flip_signature_rho(
                    cols['mt_scores'], np.where(cens_val == 0, cols['ground_truths'], np.nan), fk, rid,
                    min_len=int(self.hparams.flip_list_min), by_partner_position=True)
                per_loader[name]['rho_flip_pair'] = rho_pair
                # Within-column rank agreement, nothing centred: the quantity the rank loss optimises, between rho_epi and the flip metrics.
                per_loader[name]['rho_colrank'], _ = stats.colrank_rho(
                    cols['mt_scores'], np.where(cens_val == 0, cols['ground_truths'], np.nan), fk, min_len=int(self.hparams.flip_list_min))
                # The same on the WT head's score of the item's own mutation in the wild-type context: it cannot see the partner, so this is
                # the partner-ignorant baseline for rho_colrank (the part of within-column ordering that needs no knowledge of the other mutation).
                per_loader[name]['rho_colrank_wt'], _ = stats.colrank_rho(
                    cols['wt_scores'], np.where(cens_val == 0, cols['ground_truths'], np.nan), fk, min_len=int(self.hparams.flip_list_min))
                n_flip_matrices += n_pp
                pooled['flip_key'].extend(fk)
                pooled['row_id'].append(rid)
            else:
                per_loader[name]['rho_flip'] = float('nan')
                per_loader[name]['rho_flip_pair'] = float('nan')
                per_loader[name]['rho_colrank'] = float('nan')
                per_loader[name]['rho_colrank_wt'] = float('nan')

            for k, v in cols.items():
                pooled[k].append(v)
            pooled['subset_type'].extend(subset_types)
            pooled['cens'].append(cens_val)
            if self.link_head is not None:
                # a loader whose batches carry no dG_wt has no observed-scale outputs: pad with NaN so the pooled arrays stay aligned
                # (those items then drop out of the pooled observed-scale metrics)
                for k in ('wt', 'mt', 'comb'):
                    pooled[f'{k}_obs'].append(obs_val[k] if obs_val is not None else np.full(len(subset_types), np.nan))
            # Mutations are numbered per library, so tag them with the loader to keep pooled
            # singles from pairing with another protein's doubles.
            pooled['mut_key'].extend(
                [tuple((name,) + m for m in k) for k in mut_keys] if mut_keys is not None
                else [None] * len(subset_types))

        # Per protein: only the three that answer "is this library fine" (rho_flip_pair exists only for libraries with doubles).
        for name, metrics in per_loader.items():
            for metric in self._VAL_PER_PROTEIN:
                val = metrics.get(metric, float('nan'))
                if not np.isnan(val):
                    self.log(f"val_{metric}/{name}", val, on_epoch=True, sync_dist=True)

        avg_metrics = {}
        for metric in self._VAL_AVG:
            vals = [m[metric] for m in per_loader.values() if metric in m and not np.isnan(m[metric])]
            if vals:
                avg_metrics[metric] = float(np.mean(vals))
                self.log(f"val_{metric}_avg", avg_metrics[metric], on_epoch=True, prog_bar=metric in self._VAL_PROGRESS_BAR, sync_dist=True)

        if pooled['subset_type']:
            pooled_metrics = stats.compute_metrics(
                np.concatenate(pooled['wt_scores']), np.concatenate(pooled['mt_scores']),
                np.concatenate(pooled['comb_scores']), np.concatenate(pooled['ground_truths']),
                pooled['subset_type'], dddG=np.concatenate(pooled['dddG']),
                mut_keys=None if any(k is None for k in pooled['mut_key']) else pooled['mut_key'],
                cens=np.concatenate(pooled['cens']),
                obs=({k: np.concatenate(pooled[f'{k}_obs']) for k in ('wt', 'mt', 'comb')} if pooled.get('comb_obs') else None))
            pooled_all = dict(pooled_metrics)
            if pooled['row_id'] and len(pooled['flip_key']) == len(np.concatenate(pooled['mt_scores'])):
                # The flip family over every library at once (flip keys carry the library code, so matrices never mix libraries).
                mt_all, rid_all = np.concatenate(pooled['mt_scores']), np.concatenate([np.asarray(r).reshape(-1) for r in pooled['row_id']])
                tgt_all = np.where(np.concatenate(pooled['cens']) == 0, np.concatenate(pooled['ground_truths']), np.nan)
                min_len = int(self.hparams.flip_list_min)
                if len(rid_all) == len(mt_all):
                    pooled_all['rho_flip'] = stats.flip_signature_rho(mt_all, tgt_all, pooled['flip_key'], rid_all, min_len=min_len)[0]
                    pooled_all['rho_flip_pair'] = stats.flip_signature_rho(mt_all, tgt_all, pooled['flip_key'], rid_all, min_len=min_len,
                                                                           by_partner_position=True)[0]
                    pooled_all['rho_colrank'] = stats.colrank_rho(mt_all, tgt_all, pooled['flip_key'], min_len=min_len)[0]
                    pooled_all['rho_colrank_wt'] = stats.colrank_rho(np.concatenate(pooled['wt_scores']), tgt_all, pooled['flip_key'], min_len=min_len)[0]
            for metric in self._VAL_POOLED:
                val = pooled_all.get(metric, float('nan'))
                if not np.isnan(val):
                    self.log(f"val_{metric}_pooled", val, on_epoch=True, sync_dist=True)
            self.log("val_n_flip_pair_matrices", float(n_flip_matrices), on_epoch=True, sync_dist=True)

        if (not self.trainer.sanity_checking and self.hparams.get('wt_early_stop_patience', 0) > 0
                and not self.peft_manager.has_transitioned and 'rho_wt_valid' in avg_metrics):
            self._wt_early_stop(float(avg_metrics['rho_wt_valid']))

        if not self.trainer.sanity_checking and getattr(self, '_plateaus', None):
            total_warmup_steps = self.hparams.lr_warmup_steps + max(int(getattr(self.hparams, "calib_delay_steps", 0)), int(getattr(self.hparams, "mt_lora_delay_steps", 500)))
            if self.trainer.global_step >= total_warmup_steps:
                for metric, plateau in self._plateaus:
                    if metric in avg_metrics:
                        plateau.step(avg_metrics[metric])

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _wt_param_names(self):
        """Trainable parameters of the WT head: its adapter and its calibration head."""
        return [n for n, _ in self.model.named_parameters()
                if n in self._trainable_param_names
                and any(x in n for x in ('wt_adapter', 'calibration_head_wt', 'default', 'calibration_head_fused'))]

    def _wt_early_stop(self, value: float):
        """
        Early stopping of the WT head alone, on its own validation Spearman (``val_rho_wt_valid_avg``).

        Every validation that beats the best by 1e-4 snapshots the WT adapter and calibration head (a few MB, on the CPU). After
        ``--wt_early_stop_patience`` consecutive validations without a new best the snapshot is put back, the WT head is frozen at
        that best state, and the MT head carries on training. Freezing at the best state (not at the state patience epochs later) is
        the point: the head that goes on to serve as the MT head's fixed baseline is the best one seen.
        """
        pm = self.peft_manager
        if value > getattr(pm, 'wt_best_metric', -float('inf')) + 1e-4:
            pm.wt_best_metric, pm.wt_patience_counter = value, 0
            params = dict(self.model.named_parameters())
            self._wt_best_state = {n: params[n].detach().to('cpu', copy=True) for n in self._wt_param_names()}
            return
        pm.wt_patience_counter = getattr(pm, 'wt_patience_counter', 0) + 1
        if pm.wt_patience_counter < int(self.hparams.wt_early_stop_patience):
            return
        best = getattr(self, '_wt_best_state', None)
        if best:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, v in best.items():
                    params[n].copy_(v.to(params[n].device))
        logging.info(f"WT head early-stopped: no val_rho_wt_valid_avg gain over {pm.wt_best_metric:.4f} for {pm.wt_patience_counter} "
                     f"validations; WT adapter {'restored to its best state and ' if best else ''}frozen, MT head continues.")
        self._save_converged_wt_weights()
        pm.freeze_wt_components()
        pm.unfreeze_mt_components()
        pm.has_transitioned = True
        pm.enforce_freezing(self.optimizers(), zero_lrs=True)

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

            # Never hand the frozen backbone to the optimizer (with the WT early stop the baseline
            # check below is bypassed, which used to put all ~1.4B frozen base weights into the
            # 'other' group and its per-step norm logging).
            if name not in self._trainable_param_names:
                continue
            # Rely on the manager's baseline configuration to determine valid groups
            if not self.peft_manager.baseline_requires_grad.get(name, True) and not self.hparams.get('wt_early_stop_patience', 0):
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
        if self.link_head is not None:
            link_params = [p for p in self.link_head.parameters() if p.requires_grad]
            main_groups.append({"params": link_params, "lr": float(self.hparams.link_lr), "weight_decay": 0.0, "name": "link"})

        if not main_groups:
            logging.warning("No param groups found; defaulting to all trainables.")
            main_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd, "name": "all"}]

        opt_main = torch.optim.AdamW(main_groups, lr=base_lr, betas=(0.9, 0.999), fused=True, weight_decay=wd)

        lambdas_main = []
        for g in main_groups:
            if "calib" in g["name"] or g["name"] == "link": lambdas_main.append(lambda step: 1.0)
            elif g["name"] == "lora_mt":
                lambdas_main.append(lambda step, delay=self.hparams.mt_lora_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))
            else:
                lambdas_main.append(lambda step, delay=self.hparams.calib_delay_steps, warmup=self.hparams.lr_warmup_steps: 0.0 if step < delay else min(1.0, max(1e-4, (step - delay) / max(1, warmup))))

        warmup_main = torch.optim.lr_scheduler.LambdaLR(opt_main, lr_lambda=lambdas_main)
        # Each head cuts its own learning rate on its own validation metric (10x after two validations without a gain): the WT adapter and its
        # calibration head on val_rho_wt_valid_avg, the MT adapter and its calibration head on val_rho_flip_pair_avg. The link is never cut.
        self._plateaus = [('rho_wt_valid', GroupPlateau(opt_main, ('lora_wt', 'calib_wt'))),
                          ('rho_flip_pair', GroupPlateau(opt_main, ('lora_mt', 'calib_mt')))]

        return [opt_main], [{"scheduler": warmup_main, "interval": "step", "frequency": 1}]

    def _save_converged_wt_weights(self):
        """Extracts and saves only the WT adapter and calibration head at the moment of convergence."""
        
        logging.info("Saving converged WT adapter weights before transition...")
        wt_state_dict = {}
        
        for name, param in self.model.named_parameters():
            # WT adapter + its calibration head only. (Matching on 'peft_wt' used to grab every
            # frozen backbone weight in the WT PEFT wrapper as well: a ~5.6 GB file.)
            if name not in self._trainable_param_names:
                continue
            if any(x in name for x in ['wt_adapter', 'calibration_head_wt', 'default', 'calibration_head_fused']):
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
        
        trainable_keys = {f"model.{n}" for n in self._trainable_param_names}
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if 'lora' in k.lower() or 'calibration' in k.lower() or k in trainable_keys or k.startswith('link_head.')
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
    if os.environ.get('MSR_MEM_FRACTION') and torch.cuda.is_available():
        # hard cap on this process's CUDA memory: the allocator must free cached blocks instead of growing into the shared
        # (system) memory that WDDM/WSL spills to when the card is full
        torch.cuda.set_per_process_memory_fraction(float(os.environ['MSR_MEM_FRACTION']))
        logging.info(f"CUDA memory capped at {float(os.environ['MSR_MEM_FRACTION']):.2f} of the device")

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
        comet_kwargs = {
            "api_key": args.comet_api_key,
            "project": args.comet_project_name,
            "name": f"{args.experiment_name}-{args.version or 'run'}",
        }
        if getattr(args, "comet_experiment_key", None):
            comet_kwargs["experiment_key"] = args.comet_experiment_key
        loggers.append(CometLogger(**comet_kwargs))

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename=args.checkpoint_filename, save_top_k=args.save_top_k, monitor=args.monitor_metric, mode=args.monitor_mode, save_last=True),
        TQDMProgressBar(refresh_rate=min(10, args.log_every_n_steps))
    ]

    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=args.early_stopping_metric, patience=args.early_stopping_patience, mode=args.monitor_mode, verbose=True))

    trainer_kwargs = {
        "max_epochs": args.num_epochs, "min_epochs": args.min_epochs, "accelerator": accelerator, "devices": devices, "strategy": strategy,
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

    if not args.skip_val: trainer.validate(lightning_model, dataloaders=val_loaders, ckpt_path=args.ckpt_path)
    trainer.fit(lightning_model, train_dataloaders=train_loaders, val_dataloaders=val_loaders, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()