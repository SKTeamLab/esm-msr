# Agent Checklist & Mandatory Analysis Conventions (`analysis/`)

Every agent or contributor editing evaluation scripts, scoring tables, statistical tests, or figures in `analysis/` **must verify all rules below** before writing code or reporting numbers.

---

## 1. Singles-Trained vs. Doubles-Trained Model Views
- **Single-mutant datasets / assays**: always score the **singles-trained** version of every two-version model:
  - `ESM-MSR`: **WT-view** (`wt_lora_pred`)
  - `MutateEverything`: **Stage 1** (`me1`)
  - `ThermoMPNN`: **ThermoMPNN** (single-additive, `tmpnn`)
  - `SPURS`: **SPURS** (single-additive, `spurs`)
- **Multi-mutant datasets / assays** (doubles & higher-order, e.g., MegaScale doubles, `PTMUL`, `PTMUL-D`, ProteinGym multi-mutant assays, MGnify/Rocklin multi-mutants): always score the **doubles-trained / multi-mutant** version:
  - `ESM-MSR`: **Dual-view** (`combined_pred`)
  - `MutateEverything`: **Stage 2** (`me2`)
  - `ThermoMPNN`: **ThermoMPNN-D** (`tmpnnd`, applied to doubles; additive on higher-order)
  - `SPURS`: **SPURS-multi** (`spursm`)
- Never mix a singles-only model onto a multi-mutant benchmark (except in explicitly labelled additive/epistasis baselines) or a Dual-view/Stage-2 model onto a single-mutant-only benchmark (such as Domainome or single-mutant external $\Delta\Delta G$ sets).

## 2. ESM3 Zero-Shot Convention (`σ = 0`, `per_res_plddt = 1.0`)
- `ESM3-small-open` zero-shot is evaluated as **ESM-MSR at $\sigma = 0$** (`lora_alpha = 0`) with **`per_res_plddt = 1.0` wherever structure coordinates are finite** (`0.0` at BOS/EOS/missing coordinates), matching official `ESM3.logits()` conditioning (`src/esm_msr/models.py:_get_esm3_outputs`).
- Never leave `per_res_plddt=None` in `ESM3.forward()`, which silently defaults `per_res_plddt` to all zeros and degrades zero-shot ESM3 accuracy.
- When matching ProteinGym's published `ESM3` column, use single-site masked marginal summed additively over mutated positions (`--mask_strategy independent --skip_reverse --sigma 0`), which reproduces public ProteinGym `ESM3` to $r > 0.9999$.

## 3. Rosetta Raw-Output Energy Scaling (`1 / 2.94`)
- **Rosetta Cartesian $\Delta\Delta G$ predictions must always be scaled by `1 / 2.94`** (REU $\to$ kcal/mol) whenever scored from raw Rosetta outputs (critical for RMSE/MAE, calibration, and any scale-sensitive metric; verify sign convention too so stabilizing/destabilizing direction matches the target).

## 4. External Dataset Holdout Filter (Four-Part Sequence + Structure Rule)
- Whenever evaluating held-out external datasets (**ProteinGym**, **Domainome**, **MGnify**, **Rocklin *de novo* designs**), **always apply the exact four-part sequence + structure homology filter** against the 172 capped MegaScale training libraries (`analysis/reference/tsuboyama_rule_flags.csv`, loaded via `msr_analysis.heldout.excluded(dataset)` / `keep_ids(dataset)`):
  1. **MMseqs2 E-value** $\le 10^{-3}$
  2. **BLOSUM62 shorter-sequence local identity** $> 0.40$
  3. **Foldseek E-value** $\le 10^{-3}$
  4. **Foldseek $\text{TM}_{\max}$** (shorter-chain normalized) $> 0.70$
- Exclude any protein/assay meeting **any** of the four criteria. Never fall back to a sequence-only filter, an uncalibrated aligned-column `fident` filter, or a "drop Tsuboyama by name" heuristic.
- **No artificial `PG_DROP` subsetting on ProteinGym**: evaluate on all **176** non-homologous ProteinGym assays (`217 - 41` homologous). If an external baseline has missing/`NaN` assays on $>1{,}800$-aa proteins, average over its scored assays and report exact `n_scored` rather than dropping those assays from models that scored them.

## 5. Canonical Model Ordering & Inclusion
Always arrange models in this exact group order (and use the immutable colors/markers in `analysis/msr_analysis/models.py`):
1. **Zero-shot** *(ending with ESM3)*:
   - `ProSST`, `ProteinMPNN`, `ESM3-small-open`
2. **Transfer-learned / retrained on Homology-Restricted MegaScale** *(starting with ESM-MSR)*:
   - `ESM-MSR` *(followed by `ESM-MSR σ=0.5` when LoRA scaling is shown)*, `MutateEverything`, `ThermoMPNN`, `SPURS`
3. **Externally supervised (not trained on MegaScale, where relevant)**:
   - `GeoDDG-3D`, `Rosetta`, `ESM3ΔG`
4. **Supplement-only (unsplit MegaScale-trained, only when explicitly requested)**:
   - `ESMtherm*`, `IFUM*`, `DDGemb`, `DDMut`
   - **Do NOT include `SPURS published*`** — it is retired and not relevant.

## 6. Mandatory Tobit Observation Link Whenever $\Delta G_{\text{wt}}$ Is Available
- **ESM-MSR must always use the Tobit observation link** $h(\Delta G_{\text{wt}} + \widehat{\Delta\Delta G}) - \Delta G_{\text{wt}}$ (`msr_analysis/tobit.py`, `esm_msr.val_stats.tobit_link(**TOBIT)`) whenever wild-type stability $\Delta G_{\text{wt}}$ is available — specifically on **all MegaScale splits** (Homology-Restricted `capped` and `mega_splits`), **MGnify**, and **Rocklin *de novo* designs**, all measured by the same lab with the same cDNA proteolysis assay.
- Never report unlinked latent ESM-MSR as the primary row on MegaScale, MGnify, or Rocklin (especially for RMSE/MAE, Pearson $r$, or pooled $\rho$ across scaffolds, where unlinked latent predictions incur large floor-saturation error against censored proteolysis measurements).
- In supplementary tables reporting RMSE on these cDNA proteolysis benchmarks, you may additionally include an **`ESM-MSR (unlinked)`** row directly below `ESM-MSR` that is blank (`---`) in every column except RMSE, showing the raw latent RMSE for transparency.

## 7. Additional Consistent Conventions Across the Repository
- **Triplicate seed averaging & joint `[cluster/assay × seed]` bootstrap**: average across seeds 1–3 for all retrained models (`ESM-MSR`, `MutateEverything`, `ThermoMPNN`, `SPURS`) and our 3-run `ProteinMPNN`. Always resample replicate seeds **jointly** with proteins/assays in bootstrap CIs and paired $p$-values (see `analysis/docs/SEED_AUDIT.md`).
- **Censoring & wild-type backgrounds on MegaScale**: filter out censored rows (`cens == 0`) for numeric $\Delta\Delta G$ evaluation and distinguish wild-type-backbone libraries from stabilizing mutant-background libraries per `docs/BENCHMARKS_OCT2026.md`.
- **Per-mutation calibration bias**: always enforce `tie_mt_scale=False`, `per_mutation_bias=True` (bias applied once per mutated position, not once per variant).
- **Atomic cache & file writes**: always write caches and outputs via `msr_analysis.paths.atomic_to_csv` / `atomic_pickle_dump` / `atomic_write_text` so worktree symlinks are never mutated in place.
- **Git worktree discipline**: perform edits, commits, and tests in a dedicated `git worktree` on a feature branch rather than editing the primary checkout directly unless explicitly requested.
