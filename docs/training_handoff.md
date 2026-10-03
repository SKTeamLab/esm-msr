# Training handoff

Operational brief for whoever runs the long training jobs. Read
`docs/dual_adapter_design.md` first for *why* the settings below are what they are; this
document is *what to run and what to watch*.

Environment and paths are as of 2026-10-02 on the single-RTX-5090 workstation.

## 1. What changed, in one paragraph

The two LoRA adapters are now trained on strictly separate tasks: the WT adapter sees only
real wild-type sequences on their real structures (single mutations), and the MT adapter
sees only already-mutated sequences and predicts conditional effects. Subsets were
collapsed and renamed (`mut_ctx`/`mut_ctx_rev` -> `cond`, `native_mut_ctx` ->
`native_cond`); the data is filtered for a measurement artifact that was manufacturing
fake epistasis; 2D density capping is gone; the structure cache is rebuilt unmasked; and
validation reports four metrics instead of thirty. Old flag names still resolve where it
was cheap to support them, and `esm_msr/routing.py` is the single source of truth for which
subset trains which adapter.

## 2. Preconditions

| thing | value | check |
|---|---|---|
| cache | `cache_v4/` — 404 libraries, 31 GB, unmasked structures | `ls cache_v4 \| wc -l` -> 404 |
| raw table | `/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | exists |
| structures | `.../AlphaFold_model_PDBs` | exists |
| split | `/home/sareeves/software/esm-msr/data/hyperopt_splits.pkl` — 120 train / 26 val / 34 test | loads |
| benchmarks | `repo/data/preprocessed` — `ptmuld_mapped.csv`, `s461_mapped.csv`, `ssym_mapped.csv` | all three present |
| venv | `/home/sareeves/miniconda3/envs/msr_venv/bin/python` | never write into it |
| env | `PYTHONPATH=<repo>/src`, `HF_HUB_OFFLINE=1` | ESM3 base is in the HF cache |

`cache_mcr` and `cache_cond` were deleted and are obsolete. If `cache_v4` is ever lost it
rebuilds from the raw CSV and PDBs in about **4 minutes** (0.4-0.8 s/library), so never
treat a missing cache as a blocker.

**The cache does not need rebuilding to change masking.** It stores unmasked structures and
masking is applied at run time.

## 3. Canonical command

```bash
cd /home/sareeves/playground/esm-msr-devel
export PYTHONPATH=repo/src HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python

"$PY" repo/src/esm_msr/training.py \
  --experiment_name RUN_NAME --version 0 \
  --raw_data_file '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv' \
  --af_model_folder '/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs' \
  --split_file '/home/sareeves/software/esm-msr/data/hyperopt_splits.pkl' \
  --cache_path cache_v4 \
  --benchmark_data_path repo/data/preprocessed \
  --checkpoint_path training_checkpoints --log_dir training_logs \
  --num_epochs 8 --seed 1 \
  --dataloading cycle --loader_strategy all \
  --subset_caps single=None cond=None native_cond=None \
  --min_additive_dG -1.0 \
  --lora_rank_wt 2  --lora_alpha_wt 4  --lora_dropout_wt 0.1 --target_mode_wt expanded \
  --lora_rank_mt 16 --lora_alpha_mt 16 --lora_dropout_mt 0.1 --target_mode_mt expanded \
  --incl_sequence_head_wt --incl_sequence_head_mt \
  --adapter_mode dual --lora_mode ensemble \
  --lambda_reg_wt 1.0 --lambda_rank_wt 1.0 --lambda_reg_mt 1.0 \
  --mt_single_anchor_weight 0.5 --cond_weight 0.5 --native_cond_weight 1.0 \
  --reg_loss mse --precision bf16-mixed \
  --batch_size 256 --micro_batch_size 64 --subset_size 16 \
  --learning_rate 2e-4 --lr_warmup_steps 500 \
  --shared_scale_init 0.3 --shared_bias_init 0 --detach_ensemble_input \
  --num_workers 4 --log_every_n_steps 25 --save_top_k 3 \
  --monitor_metric val_rho_combined_avg --monitor_mode max
```

### Why the step count is lower than the item count suggests

Work units are planned so each drives one adapter (`training._plan_units`). All items of a
library share one wild-type sequence, so the entire WT block of a batch is a single backbone
forward; only the MT units scale with `micro_batch_size`. With the anchor on, singles are
visited twice — once in a WT unit, once in an MT unit — which is the same two passes as
before, just not interleaved.

### Expected cost

Counted from `cache_v4` against the 120-protein train split:

```
train: 127 libraries, 233,574 trainable items -> ~217,000 after min_additive_dG
       ~847 steps/epoch at batch 256
val:    26 libraries,  ~40,500 items, ~158 batches, plus 3 benchmark loaders
```

The smoke run did batch 64 at ~1.08 it/s. **Verify the real rate after the first 50 steps
and recompute** — do not trust an extrapolation from a 2-library smoke. Budget roughly
45-70 min/epoch including validation and treat that as unverified until measured.

VRAM: the project's own earlier note records peak ~16.5 GB at `micro_batch_size 64` on this
box. I did not capture peak myself. Watch the first few hundred steps; if it approaches 30
GB, halve `micro_batch_size` (loss values are micro-batch invariant, so this changes speed
and memory only).

## 4. Overnight parameter sets

Run sequentially — one GPU. Each uses the canonical command with the listed overrides and a
distinct `--experiment_name`. At ~1 h/epoch, **3 epochs each fits four configs in ~12 h**;
raise `--num_epochs` only if you drop configs.

| # | name | overrides | question it answers |
|---|---|---|---|
| **A** | `v4_baseline` | *(none)* | Does separated-head training beat the old combined-loss model? This is the reference. |
| **B** | `v4_maskstruct` | `--mask_structure` | Is it better to tell the MT adapter "geometry unknown here" than to show it wild-type geometry at mutated sites? |
| **C** | `v4_strictfilter` | `--min_additive_dG 0.0` | Is the residual dynamic-range bias still hurting at -1? Costs ~28% more doubles. |
| **D** | `v4_noanchor` | `--mt_single_anchor_weight 0.0` | How much does the MT adapter rely on clean single-mutant labels to calibrate its readout? |

Optional fifth if time allows, as a bridge to the released model:

| **E** | `v4_legacy_combined` | `--lambda_reg_mt 0 --mt_single_anchor_weight 0 --lambda_reg_combined 1.0 --lambda_rank_combined 1.0 --no-detach_ensemble_input` | Reproduces the old teacher-forced ensemble objective on the new data. Note `combine_rule` auto-switches to `average` for this. |

Set `--num_epochs 3` for A-D in a first pass. If one config is clearly ahead, give it a
longer run afterwards rather than extending all of them.

## 5. What to watch

Six metrics. Per dataloader, plus `_avg` (mean over libraries, what the checkpoint monitor
reads) and `_pooled`.

| metric | meaning | rough expectation |
|---|---|---|
| `val_rho_wt_valid` | WT adapter on plain single mutations — its own domain | should rise first and highest |
| `val_rho_wt_all` | WT adapter on everything, indiscriminately | lower; the gap shows off-domain degradation |
| `val_rho_mt_valid` | MT adapter on conditional targets only | expect below WT: harder task, noisier labels |
| `val_rho_mt_all` | MT adapter on everything | always defined, so use it to compare across loaders |
| `val_rho_combined` | the reported 0.5*WT + 0.5*MT average, measured items | the headline; monitored metric |
| `val_rmse_combined` | calibration in kcal/mol | should fall; rank metrics cannot see scale error |

A `_valid` metric going missing from a loader is expected, not a failure: it is NaN wherever
the subset is absent. A library with no double mutants has no `cond` items so no
`rho_mt_valid`; a mutant-background library (code like `1SF0_V59K`) has no plain singles so
no `rho_wt_valid`; the external benchmarks load without derived items so they never have
`rho_mt_valid`. The `_all` forms are always defined — use those when comparing loaders.

Reference points, so you know when to stop rather than chasing noise:

* The released `esm-msr-small` checkpoint reached `val_rho_combined_avg` ~0.816 under the
  **old** metric definition, which pooled derived items. The new definition restricts the
  headline metrics to measured items, so **the numbers are not directly comparable** —
  compare configs against each other within this batch of runs.
* `val_rho_mt_avg` is computed on conditional targets whose label reliability is ~0.86
  (attenuation ceiling ~0.93). Do not expect it to approach `val_rho_wt_avg`.
* Label noise is not the binding constraint: the single-mutant ceiling is ~0.97. Headroom
  above 0.82 is real model headroom.

Also log-watch: `calibration_heads/*_scale` and `*_bias` (should settle, not drift
monotonically), and `norm_grad/lora_mt` vs `norm_grad/lora_wt` (if the MT group's gradient
norm is orders of magnitude larger, lower `cond_weight`).

## 6. Failure modes seen in this codebase

| symptom | cause | action |
|---|---|---|
| `AssertionError: Heterogeneous sequence lengths ... batch sampler is violating the homogeneous microbatch assumption` | a batch mixed two proteins | the cycling sampler batches per library; do not switch to a plain shuffled DataLoader |
| `AssertionError: WT Loss detached from PyTorch Graph` | every item in the slice was excluded from the WT loss | usually means `--subset_caps` left no `single` items; check the subset mix logged per library |
| `mt_single_anchor_weight > 0 requires lambda_reg_mt > 0` | the anchor is a weight on the MT regression, not its own loss | set `--lambda_reg_mt 1.0` |
| CUDA OOM in the first few steps | `micro_batch_size` too high for the longest library | halve it; losses are micro-batch invariant |
| `min_additive_dG set but WT dG unknown` warnings | a library has no `wt` rows in the raw table | harmless; that library keeps all its doubles |
| validation metrics all NaN for a library | it has no measured items after filtering | harmless; `_avg` skips NaNs |

## 7. Do not change these without reading the design doc

* **`--subset_caps`** must leave at least one family unrestricted (`None`) or the baseline
  size drops to zero and the sampler raises.
* **`--incl_reversions`** stays off. Reversion items have no head by design; the MT single
  anchor covers the same physics with an inference-consistent structure.
* **`--mask_structure` must match between training and inference.** It is written into
  `hparams.yaml` and re-read by the inference path. The testing script's
  `--mask_structure_pos` forces it on and warns if that contradicts the checkpoint.
* **`--mask_strategy`** (sequence masking) stays off. Unmasked scored best on singles and
  tied on conditionals, and it costs one forward per variant instead of one per mutation.
* **Density capping is gone.** For family balance use `--subset_caps`; for emphasis use
  `--cond_weight` / `--native_cond_weight`. Do not reintroduce resampling conditioned on
  the target — it distorts a correlation that is substantially real physics.
* **`--incl_structure_encoder_*` was removed.** `ESM3.forward` never calls the structure
  encoder, so adapters there could never receive gradient.

## 8. Open questions worth a run if time permits

1. **Matched-backbone distance control** (CPU only, no GPU). Three domains (2K5H, 1H8K,
   1OPS) were scanned both in wild-type form and in two mutant backgrounds, giving 5,512
   single mutations measured in *both* contexts — directly measured conditional effects with
   a real distance axis. This would settle whether the MT adapter's structural conditioning
   earns its capacity, better than the designed-contact doubles can.
2. **k >= 3 generalisation.** The dataset contains only single and double mutants — zero
   triples — so the MT adapter trains at k in {1, 2} and is applied at k up to 44 on
   ProteinGym. Nothing in these runs tests that extrapolation.
3. **Structure-encoder adaptation.** Now possible in principle (the unmasked cache means
   encoding could move into the forward pass), but it would require re-encoding per unique
   input and would undo the shared-encoder VRAM saving.
4. **Re-encode masking.** Current masking blanks the masked position's own coordinates and
   token; neighbouring tokens still carry its geometry. A re-encode from masked coordinates
   would remove that leak.

## 9. Reporting back

For each config, report: final and best `val_rho_wt_avg`, `val_rho_combined_avg`,
`val_rho_mt_avg`, `val_rmse_combined_avg`; the epoch they peaked; wall-clock per epoch;
peak VRAM; and any assertion or NaN event with the surrounding 20 log lines. Checkpoints
land in `training_checkpoints/<experiment_name>/` and CSV metrics in
`training_logs/<experiment_name>/<version>/metrics.csv`.
