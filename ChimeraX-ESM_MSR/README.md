# ChimeraX-ESM-MSR

Interactive UCSF ChimeraX plugin (**v0.5.0**, bundle `ChimeraX-ESM-MSR`, package `chimerax.ESM_MSR`) for running live **ESM-MSR** protein stability and epistasis inference, two-stage fast double-mutant discovery, and publication-grade 3D/2D mutational visualization directly inside ChimeraX (GUI and headless CLI).

---

## Table of Contents

1. [Overview & Key Capabilities](#overview--key-capabilities)
2. [Requirements & Compatibility](#requirements--compatibility)
3. [Installation & Setup](#installation--setup)
4. [Quickstart Workflows](#quickstart-workflows)
5. [Two-Stage Fast Double Mutant Selection](#two-stage-fast-double-mutant-selection)
6. [Categorical Pair Visualization (`Top-M Ranked Pairs`)](#categorical-pair-visualization-top-m-ranked-pairs)
7. [Complete UI Reference](#complete-ui-reference)
   - [Tab 1: `Execution / IO`](#tab-1-execution--io)
   - [Tab 2: `Visualization`](#tab-2-visualization)
   - [Global Action Bar & Status Footer](#global-action-bar--status-footer)
8. [Native ChimeraX CLI Commands Reference](#native-chimerax-cli-commands-reference)
9. [CSV Input Schema & Scientific Conventions](#csv-input-schema--scientific-conventions)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Development & Automated Headless Testing](#development--automated-headless-testing)

---

## Overview & Key Capabilities

**ESM-MSR Predictor & Visualizer** bridges the ESM-MSR stability predictor (`src/esm_msr/inference.py`) with UCSF ChimeraX in a two-tab docked tool (`Tools -> Stability -> ESM_MSR`) plus six scriptable native ChimeraX CLI commands (`esm_msr ...`):

- **Live Subprocess Inference & One-Click Env Discovery**: Export any open PDB/mmCIF structure (or specific model in an NMR ensemble) and launch `inference.py` or two-stage `fast_doubles.py` in your local Python/conda/venv environment. Use **`Auto-Detect`** to discover local conda/mamba/venv environments on disk (~25 ms, prioritizing `[esm_msr, esm, torch]`), monitor live `tqdm` progress and stage banners in the footer and collapsible in-panel log console, and terminate safely with **`STOP`**.
- **Two-Stage Fast Double Mutant Selection (`4. Fast Top-N Doubles (2-Stage)`)**: Discover top stabilizing double mutants across the entire protein in ~1 minute on GPU without scoring all $O(L^2)$ pairs — Stage 1 runs a single cached WT-context pass, an exact $O(N \log S)$ monotone-tree heap selects the Top-$N$ additive pairs (scaling $N = 1\text{–}1{,}000{,}000$ in `< 1 s`), and Stage 2 rescores those Top-$N$ pairs with the full dual-view WT+MT model and re-ranks them into a unified CSV.
- **Categorical Pair Visualization (`Top-M Ranked Pairs`)**: Publication-grade 3D visualization purpose-built for distant/non-proximal top double mutants using colorblind-friendly **Okabe–Ito** categorical pair coloring on a neutral matte cartoon backdrop, with **`Disjoint residues`** deduplication, optional subtle dashed connectors, color-matched 3D labels, stacked 2D on-canvas legend, and **`Isolate Pair`** / **`Show All`** spotlighting.
- **Interactive 3D Stability, Epistasis & Results Table**:
  - **Singles**, **WT epistasis**, **MT epistasis**, and **Top-M Ranked Pairs** display modes.
  - **Color Scale & Export**: Five verified zero-network palettes (classic `Blue–White–Red`, colorblind-safe `PRGn`, `BrBG`, `Cividis`, and perceptually uniform `Viridis`), auto or manual color bounds, and headless-safe **2D Mutational Landscape Heatmap** (`Export 2D Heatmap...`) and **Top-M Pair Legend Card** (`Export Pair Card...`) exports (PNG/PDF/SVG).
  - **Interactive Ranked Results Table**: Search/filter by mutation or residue, sort numerically across metrics, click any row to focus & select residues in 3D, and spotlight individual pairs.
- **Native ChimeraX CLI Commands**: Six registered commands (`esm_msr open`, `esm_msr load`, `esm_msr visualize`, `esm_msr pairs`, `esm_msr heatmap`, `esm_msr reset`) supporting both interactive GUI synchronization and headless `--offscreen` `.cxc` figure-generation scripts.
- **Non-Destructive ChimeraX Integration**: Epistasis edges and ranked-pair connectors live in dedicated `"ESM-MSR Epistasis"` and `"ESM-MSR Ranked Pairs"` `PseudobondGroup`s, 2D legend labels use tracked `esm_msr_leg_*` models, residue scores use the ChimeraX Python attribute API (`r:esm_msr_score`), hydrogen hiding is scoped strictly to active ESM-MSR models, user atom selections are preserved during contact queries, and **`Reset / Clear Viz`** (or `esm_msr reset`) restores the target model without touching unrelated models or user pseudobonds.

---

## Requirements & Compatibility

### ChimeraX Environment
- **UCSF ChimeraX**: `>= 1.3` (declared in `bundle_info.xml`).
- **Verified Platform**: UCSF ChimeraX **1.11.1** (bundled Python **3.11**, **PyQt6 6.9**, **matplotlib 3.10**, and **pandas 2.x & 3.x**, including pandas `3.0.1`).
- **Bundle Dependencies**: `ChimeraX-Core >= 1.3`, `pandas >= 2.0` (automatically installed by ChimeraX during `devel install` / `toolshed install`). `matplotlib` is bundled with ChimeraX and used via its headless object-oriented Agg canvas for 2D heatmap and pair card exports.

### ESM-MSR Inference Environment (for live prediction)
Visualization of pre-computed CSV files, 2D exports, and native CLI rendering commands work directly inside ChimeraX with no PyTorch installation required. To run live inference from the **Execution / IO** tab (or standalone `fast_doubles.py`), you also need a separate Python/conda/venv environment containing:
- Python `3.11–3.13`
- `esm-msr` installed in editable mode (`pip install -e .` from the repository root)
- PyTorch (CUDA 12.8 recommended for GPU acceleration; Apple Silicon `mps` and `cpu` are also supported)
- ESM3-small-open weights (either auto-downloaded via `esm`/HuggingFace or cloned locally from [`biohub/esm3-sm-open-v1`](https://huggingface.co/biohub/esm3-sm-open-v1))

See the [main repository README](../README.md#recommended-installation) for step-by-step Python environment and CUDA setup instructions.

> **WSL2 Note:** If you installed `esm-msr` inside Windows Subsystem for Linux (WSL), install ChimeraX on Windows rather than inside WSL for interactive GUI use. For Linux/WSL automated testing or scripted `.cxc` rendering, headless `chimerax --offscreen` is supported.

---

## Installation & Setup

### Option 1: Direct Source Install (Recommended)

1. Open UCSF ChimeraX and enable the Command Line Interface (`Tools -> Command Line Interface` if not already visible).
2. Run (replacing `/path/to/esm-msr` with your local repository path):
   ```
   devel install /path/to/esm-msr/ChimeraX-ESM_MSR
   ```
   To install in editable mode while developing the plugin:
   ```
   devel install /path/to/esm-msr/ChimeraX-ESM_MSR editable true
   ```
   You can also run the install from a terminal without opening the GUI:
   ```bash
   chimerax --nogui --cmd "devel install /path/to/esm-msr/ChimeraX-ESM_MSR user true; exit"
   ```

### Option 2: Build a Distributable Wheel & Install

Build the wheel (`dist/chimerax_esm_msr-0.5.0-py3-none-any.whl`) and install it via ChimeraX Toolshed:

```
devel build /path/to/esm-msr/ChimeraX-ESM_MSR
toolshed install /path/to/esm-msr/ChimeraX-ESM_MSR/dist/chimerax_esm_msr-0.5.0-py3-none-any.whl
```

Headless CLI equivalent:
```bash
chimerax --nogui --cmd "devel build /path/to/esm-msr/ChimeraX-ESM_MSR; exit"
chimerax --nogui --cmd "toolshed install /path/to/esm-msr/ChimeraX-ESM_MSR/dist/chimerax_esm_msr-0.5.0-py3-none-any.whl; exit"
```

### Upgrading, Uninstalling, and Cleaning Build Artifacts

- **Upgrade**: Re-run `devel install /path/to/esm-msr/ChimeraX-ESM_MSR` (or uninstall first, then reinstall).
- **Uninstall**:
  ```
  toolshed uninstall ChimeraX-ESM-MSR
  ```
- **Clean local build directories (`build/`, `dist/`)**:
  ```
  devel clean /path/to/esm-msr/ChimeraX-ESM_MSR
  ```
- **Offline / dependency note**: `devel install` installs `pandas >= 2.0` into ChimeraX's user site-packages automatically. If you are offline and `pandas >= 2.0` is already present in ChimeraX's Python, append `noDeps true` to `devel install`.

### Launching the Tool

Open the plugin from the ChimeraX menu bar or Command Line:

> **`Tools -> Stability -> ESM_MSR`** &nbsp;&nbsp;*(or run `esm_msr open`)*

The docked panel opens titled **ESM-MSR Predictor & Visualizer**. Your path fields (`Base Repo Dir`, `Python Env`, `Output CSV`, `Checkpoint`, `LoRA Config`, `ESM3 Weights`) are persisted automatically between sessions via `QSettings("ESM_MSR_Tools", "ESM_MSR")`.

---

## Quickstart Workflows

### Workflow A: Visualize an Existing Prediction CSV (Demo: `1UFM`)

No GPU or PyTorch setup is needed to test visualization and 2D export with the bundled example CSV (`data/example/1ufm_example.csv`):

1. Load the demo structure in ChimeraX:
   ```
   open 1ufm
   ```
2. Open **`Tools -> Stability -> ESM_MSR`** (or run `esm_msr open`) and switch to the **`Visualization`** tab.
3. Confirm **`Target Model:`** is `#1 1ufm` and **`Chain:`** is `A`.
4. Click **`Load CSV`** in the global action bar and select `data/example/1ufm_example.csv` from your `esm-msr` checkout.
5. Inspect the top summary banner and the interactive **`Ranked Results`** table — click any row to select and focus the camera on that mutation.
6. Refine the view under **`Color Scale & Export`** and **`Global Thresholds and Networks`**:
   - Choose a **`Palette:`** (e.g., `Purple–White–Green (PRGn, Colorblind-Safe)` or `Blue–White–Red (Classic)`).
   - Set **`Pos Threshold:`** to `0.5` and **`Neg Threshold:`** to `-10.0` to show only stabilizing mutations (ΔΔG ≥ +0.5 kcal/mol), check **`No Mut Cys`**, and check **`Color Backbone by Highest Additive ΔΔG`**.
   - Click **`Update Visualization`** to re-render, or click **`Export 2D Heatmap...`** to save a publication-ready 20 × $L$ landscape figure.
7. Click **`Reset / Clear Viz`** (or run `esm_msr reset`) to remove generated clones, pseudobond groups, and 2D legend labels and restore a clean white ribbon.

### Workflow B: Fast Top-N Double Mutant Discovery & Categorical Pair Visualization

1. Load a structure in ChimeraX (e.g., `open 1ufm` or drag-and-drop a `.pdb`/`.cif` file).
2. Open **`Tools -> Stability -> ESM_MSR`** on the **`Execution / IO`** tab.
3. Set **`Base Repo Dir:`** via **`Browse...`** (automatically populating `Output CSV` and the default `Checkpoint` `LoRA_models/esm-msr-small/lora_seed1.safetensors` if blank).
4. Click **`Auto-Detect`** next to **`Python Env:`** to scan local conda/mamba/venv environments on disk and auto-select your `[esm_msr, esm, torch]` environment.
5. Under **`Compute Environment`**, select **`Compute Device:`** (`cuda`), **`Model Precision (dtype):`** (`bfloat16` on Ampere+ CUDA GPUs), and **`Batch Size:`** (`16` or `32`).
6. Under **`Screening Config -> Mutation Scope`**, select **`4. Fast Top-N Doubles (2-Stage):`**:
   - Set **`Top N:`** to `1000` and **`Visualize Top M:`** to `5`.
   - Optionally check **`Filter Cys/Pro before Stage 2`** (recommended when avoiding disulfide artifacts) and/or **`Dist <=`** if you want to restrict to proximal pairs (leave unchecked to discover distant stabilizing pairs).
7. Click **`Run Prediction Script`**. Expand **`▸ Show Log Console`** if you want to inspect live `[Stage 1/2]`, `[Selection]`, and `[Stage 2/2]` output.
8. When complete, click **`Load Last Output`** — the plugin jumps to Tab 2, switches **`Mode:`** to **`Top-M Ranked Pairs`**, renders the Top-$M$ disjoint pairs with Okabe–Ito categorical colors, 3D labels, subtle dashed connectors, and a 2D legend, and populates **`Ranked Results`**.
9. Click any pair row in **`Ranked Results`** and click **`Isolate Pair`** to spotlight that pair (click **`Show All`** to restore all pairs), or click **`Export Pair Card...`** / **`Export 2D Heatmap...`** to save companion figures.

---

## Two-Stage Fast Double Mutant Selection

Exhaustive dual-view double-mutant screening scales as $O(L^2)$ forward passes — on a 101-residue protein like CcdB (`CCDB_ECOLI`), there are ~1.84 million raw double mutants, and on a 300-residue protein over 16 million. **Two-Stage Fast Double Mutant Selection** (`ChimeraX-ESM_MSR/src/fast_doubles.py`) makes whole-protein double-mutant discovery practical in ~1 minute on a single GPU:

### How It Works

1. **Stage 1 — Fast Single WT Pass (`--mode singles --skip_reverse`)**:
   Runs one cached wild-type forward pass to score all $19L$ single mutants (`wt_lora_pred`) in seconds.
2. **Exact $O(N \log S)$ Monotone-Tree Heap Top-$N$ Additive Selection**:
   Sorts the $S$ valid single mutants descending by `wt_lora_pred` and enumerates upper-triangular pairs $(i, j)$ ($0 \le i < j < S$, distinct positions) using a duplicate-free, visited-set-free monotone tree best-first max-heap (root `(0, 1)`, right child `(i, j + 1)`, down-diagonal child `(i + 1, i + 2)` only when `j == i + 1`). Every pair has a unique parent with sum $\ge$ its child, guaranteeing **exact descending order** of `stage1_additive_pred` $= \text{score}_i + \text{score}_j$ in $O(N \log S)$ time and $O(S)$ heap memory:
   - Scales smoothly across `top_n` from **`1` to `1,000,000`** in under 1 second (verified on 2,014 singles in ChimeraX Python 3.11: `top_n=1,000` in **0.006 s**, `top_n=100,000` in **0.094 s**, `top_n=1,000,000` in **0.922 s**).
   - Supports pre-Stage-2 amino-acid exclusions (`--exclude_wt_cys/pro`, `--exclude_mut_cys/pro`), minimum sequence separation (`--min_seq_sep`), and optional pure-NumPy WT heavy-atom distance filtering (`--distance_threshold`).
3. **Stage 2 — Dual-View WT+MT Rescoring (`--skip_additive --calculate_distances --skip_fix_noncanonical`)**:
   Passes the Top-$N$ candidates to `src/esm_msr/inference.py --input_csv` to obtain full dual-view `combined_pred`, mutant-context `mt_lora_pred`, authoritative 1× ensemble epistasis `epi_pred`, and heavy-atom inter-residue distance `dist`.
4. **Re-Ranking & Unified Output CSV**:
   Merges Stage-1 singles and Stage-2 doubles into a single unified `--output_csv`, re-ranking the double mutants descending by `combined_pred` and adding:
   - `stage1_additive_pred`: Stage-1 additive sum used for selection
   - `additive_rank`: 1-based rank before Stage-2 rescoring (`1..N`)
   - `full_rank`: 1-based rank after Stage-2 dual-view rescoring (`1..N`)
   - `rank_change`: `additive_rank - full_rank` (positive = promoted by epistasis/MT context)
   - `dist`: minimum WT heavy-atom distance in Ångströms

   On `CCDB_ECOLI.pdb` (Chain A, 101 aa, RTX 5090, `bfloat16`, `batch_size=32`), a full Top-1,000 run completes in **65.9 s** total (29.7 s Stage 1 + 0.01 s selection from 1.84M raw pairs + 36.1 s Stage 2).

### GUI Controls (Tab 1 Mode 4)

Select **`4. Fast Top-N Doubles (2-Stage):`** under **Screening Config -> Mutation Scope**:
- **`Top N:`** (`1–1,000,000`, default `1000`) — number of top additive pairs forwarded to Stage-2 dual-view rescoring.
- **`Visualize Top M:`** (`1–12`, default `5`) — bidirectionally synced with Tab 2's **`Top M Pairs:`** spinbox; automatically applied when clicking **`Load Last Output`**.
- **`Filter Cys/Pro before Stage 2`** *(unchecked by default)* — passes `--exclude_wt_cys --exclude_wt_pro --exclude_mut_cys --exclude_mut_pro` before pairing (any individual Tab 2 **`No WT/Mut Cys/Pro`** checkboxes are also forwarded).
- **`Dist <=`** (`0.0–100.0` Å, default `10.0 Å`, *unchecked by default*) — optionally limits pairing to residue pairs within the specified WT heavy-atom distance. Leave unchecked to discover non-proximal stabilizing pairs.
- **`Positions:`** / **`Grab Selection`** / **`Invert (Except)`** from Mode 1 also apply to the Mode 4 Stage-1 scan. **`Skip MT LoRA Pass (Additive Approximation)`** is automatically disabled in Mode 4 because Stage 2 always runs the full dual-view pass.

### Standalone CLI Usage (`fast_doubles.py`)

You can run the two-stage pipeline directly from the command line without opening ChimeraX:

```bash
python ChimeraX-ESM_MSR/src/fast_doubles.py \
  --repo_dir /path/to/esm-msr \
  --pdb_file /path/to/protein.pdb \
  --code MYPROT \
  --chain A \
  --output_csv ./myprot_top1000_doubles.csv \
  --top_n 1000 \
  --checkpoint_path /path/to/esm-msr/LoRA_models/esm-msr-small/lora_seed1.safetensors \
  --device cuda \
  --model_dtype bfloat16 \
  --batch_size 32
```

#### `fast_doubles.py` CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--repo_dir` | *(required)* | Path to the `esm-msr` repository root (must contain `src/esm_msr/inference.py`). |
| `--pdb_file` | *(required)* | Path to input PDB structure. |
| `--output_csv` | *(required)* | Destination path for the unified output CSV (Stage-1 singles + Stage-2 re-ranked doubles). |
| `--checkpoint_path` | *(required)* | Path to LoRA checkpoint (`.safetensors`, `.pt`, or `.ckpt`). |
| `--code` | `protein` | Protein code identifier written into CSV rows. |
| `--chain` | `A` | Target PDB chain ID. |
| `--top_n` | `1000` | Number of top additive double mutants to rescore (`1` to `1,000,000`). |
| `--lora_config` | `None` | Optional path to `lora_config.json`. |
| `--hparams_path` | `None` | Optional path to `hparams.yaml` (auto-detected beside checkpoint if omitted). |
| `--base_model_loc` | `None` | Optional local directory containing ESM3 weights. |
| `--device` | `cuda` | Torch device (`cuda`, `cuda:0`, `mps`, `cpu`). |
| `--model_dtype` | `bfloat16` | Precision: `float32`, `bfloat16`, or `float16`. |
| `--batch_size` | `16` | Inference batch size. |
| `--sigma` | `1.0` | LoRA α multiplier (`1.0` = fine-tuned model; `0.0` = zero-shot ESM3 control). |
| `--mask_strategy` | `None` | Optional override: `marginal` or `independent`. |
| `--screen_residues` | `None` | Comma-separated PDB residue positions for Stage-1 screening (mutually exclusive with `--screen_residues_except`). |
| `--screen_residues_except` | `None` | Comma-separated PDB residue positions to exclude from Stage-1 screening. |
| `--skip_fix_noncanonical` | `False` | Skip PDBFixer non-canonical residue replacement in Stage 1. |
| `--distance_threshold` | `None` | Optional max WT heavy-atom distance (Å) for double-mutant pairing. |
| `--exclude_wt_cys` / `--exclude_wt_pro` | `False` | Exclude wild-type Cysteine / Proline positions before Stage-2 pairing. |
| `--exclude_mut_cys` / `--exclude_mut_pro` | `False` | Exclude mutations to Cysteine / Proline before Stage-2 pairing. |
| `--min_seq_sep` | `1` | Minimum sequence index separation `|res_i - res_j|` between paired residues. |
| `--keep_intermediates` | `False` | Retain intermediate Stage-1/Stage-2 CSVs in `<output_stem>_fast_doubles_work/`. |

---

## Categorical Pair Visualization (`Top-M Ranked Pairs`)

### Why Categorical Pair Identity Coloring Instead of Energy Coloring?

When screening double mutants without a tight distance cutoff, the highest-scoring pairs are **predominantly non-proximal** across the 3D fold (for example, on `CCDB_ECOLI`, 84.2% of Top-1,000 pairs are $\ge 8$ Å apart with a median distance of `15.4 Å`) and all sit within a narrow stabilizing ΔΔG band. Coloring top pairs on a continuous stability ramp makes every mutation look nearly the same shade — you cannot tell at a glance which two distant residues belong to the same double mutant.

**`Top-M Ranked Pairs`** solves this by dedicating color to **pair identity**:

- **Okabe–Ito + Tol Colorblind-Safe Categorical Palette**: Ranks `#1`–`#6` use the classic Okabe–Ito palette (`#0072B2` Blue, `#D55E00` Vermilion, `#009E73` Bluish green, `#CC79A7` Reddish purple, `#E69F00` Orange, `#56B4E9` Sky blue), extended to 12 pairs with high-contrast Tol hues (`#332288`, `#882255`, `#44AA99`, `#AA4499`, `#117733`, `#999933`).
- **Neutral Matte Backdrop & Publication Lighting**: Renders the target chain as a neutral light-grey ribbon (`#D8DCE2` at 35% transparency) and, with **`Pub Lighting`** enabled, applies a clean white background, `lighting soft`, and subtle outlines (`graphics silhouettes true width 1.5`).
- **Per-Pair Clones & Anchor Highlights**: Each ranked pair gets its own ribbon-hidden mutant clone (`*_pair_<rank>`) with `swapaa` sidechain sticks (`stickRadius 0.25`; Glycine `CA` spheres at `0.45`), pair-colored carbons + `byhetero` heteroatoms (N/O/S remain chemically legible), and opaque pair-colored anchor segments on the WT cartoon.
- **`Disjoint residues` Deduplication**: Enabled by default so a single dominant "hub" mutation (e.g., `E79C` appearing in 9 of the top 10 raw CcdB doubles) cannot monopolize all Top-$M$ slots; every displayed pair occupies two distinct residue positions.
- **Subtle Connectors, 3D Labels & On-Canvas 2D Legend**:
  - **`Connectors:`** draws optional Cα–Cα pseudobonds in a dedicated `"ESM-MSR Ranked Pairs"` `PseudobondGroup` (`Dashed (Subtle)` default, `None (Labels Only)`, or `Solid`).
  - **`3D Labels`** places concise color-matched `#<rank> <mut>` labels on each mutated residue.
  - **`2D Legend`** places a stacked top-left `2dlabels` summary (`esm_msr_leg_hdr`, `esm_msr_leg_1..M`) reporting rank, pair notation, combined ΔΔG, optional epistasis, and heavy-atom distance.
- **Single-Pair Spotlighting**: Select any pair in the **`Ranked Results`** table and click **`Isolate Pair`** (or run `esm_msr pairs ... isolate <rank>`) to hide all other pair clones, connectors, and legend rows, highlight only that pair's WT anchor segments, and smoothly zoom the camera (`view ... pad 0.35`). Click **`Show All`** to restore all rendered pairs.

---

## Complete UI Reference

The plugin window (**ESM-MSR Predictor & Visualizer**) is organized into two vertically scrollable tabs (`QScrollArea`) above a shared global action bar and status footer.

### Tab 1: `Execution / IO`

#### Local Path Configuration

| Control | Description & Behavior |
|---|---|
| **`Base Repo Dir:`** (`Browse...`) | Root directory of your `esm-msr` checkout (must contain `src/esm_msr/inference.py`). All relative user paths in Tab 1 resolve against this directory, and `<Base Repo Dir>/src` is prepended to `PYTHONPATH` for the subprocess. |
| **`Python Env:`** (`Auto-Detect`, `Browse...`) | Python interpreter used to launch inference. Accepts: **(1)** direct path to a `python` / `python.exe` binary (checked first), **(2)** path to a `venv` or conda environment directory (checks `python.exe`, `Scripts/python.exe`, `bin/python`, `bin/python3`), **(3)** a conda environment name (invoked via `conda run --no-capture-output -n <name> python -u`), or **(4)** blank to use `python`/`python3` from `PATH`. Always preserves `-u` and `PYTHONUNBUFFERED=1`. |
| **`Auto-Detect`** & **`Detected:`** dropdown | Disk-only environment scanner (zero subprocesses, ~25 ms) checking `CONDA_PREFIX`, `VIRTUAL_ENV`, `~/.conda/environments.txt`, standard conda/mamba/miniforge/micromamba roots, and repo-local virtualenvs (`.venv`, `venv`, `env`). Tags environments with `[esm_msr, esm, torch]`, ranks complete ESM-MSR environments first, auto-selects the top match, and reveals the **`Detected:`** dropdown to switch between discovered environments. |
| **`Output CSV:`** (`Browse...`) | Destination file passed via `--output_csv`. Defaults to `<Base Repo Dir>/example_inference.csv`. Its parent directory must already exist. |
| **`ESM3 Weights:`** (`Browse...`) | Optional local folder containing `data/weights/esm3_sm_open_v1.pth` (`--base_model_loc`). Leave blank to let the `esm` package download/cache weights from HuggingFace automatically. |

#### Compute Environment

| Control | Options / Range | CLI Flag & Notes |
|---|---|---|
| **`Compute Device:`** | `cuda`, `cuda:0`, `cuda:1`, `mps`, `cpu` | `--device`. Use `cpu` if no CUDA/Apple GPU is configured. |
| **`Model Precision (dtype):`** | `float32` *(default)*, `bfloat16`, `float16` | `--model_dtype`. `float32` works on CPU & GPU; `bfloat16` is faster and uses less VRAM on Ampere+ CUDA GPUs; `float16` suits older GPUs. |
| **`Batch Size:`** | `1–512` *(default `16`)* | `--batch_size`. Reduce if you hit CUDA out-of-memory errors. |

#### Model Configuration Files

| Control | Description & Behavior |
|---|---|
| **`Checkpoint:`** (`Browse...`) | Path to a LoRA checkpoint (`.safetensors` or `.ckpt`, passed via `--checkpoint_path`). Defaults to `LoRA_models/esm-msr-small/lora_seed1.safetensors`. The repository ships three training seeds (`lora_seed1.safetensors`, `lora_seed2.safetensors`, `lora_seed3.safetensors`) alongside a shared `hparams.yaml` under `LoRA_models/esm-msr-small/`. Full-rank models are available on [HuggingFace (`sareeves96/esm-msr`)](https://huggingface.co/sareeves96/esm-msr). Editing or browsing a new checkpoint clears `LoRA Config` so the sibling `hparams.yaml` is picked up cleanly. |
| **`LoRA Config:`** (`Browse...`) | Optional `.json` (`--lora_config`) or `.yaml`/`.yml` (`--hparams_path`). Leave blank to auto-detect `hparams.yaml` in the checkpoint's directory. An inline red warning banner alerts you if either file is missing. |

> **Zero-Shot ESM3 Backbone Control (`--sigma 0.0`)**:
> - **Recommended**: Keep a valid checkpoint selected (e.g. `LoRA_models/esm-msr-small/lora_seed1.safetensors`) and set **`Sigma (σ):`** to `0.0`.
> - **Alternative (checkpoint blank, Modes 1–3)**: Leave **`Checkpoint:`** blank (the plugin omits `--checkpoint_path`), provide a **`LoRA Config:`** (e.g., `LoRA_models/esm-msr-small/hparams.yaml`), and set **`Sigma (σ):`** to `0.0`. *(Mode 4 always requires a checkpoint path; use `Sigma (σ) = 0.0` with a checkpoint for zero-shot two-stage runs.)*

#### Screening Config

- **`Target Selection`** *(synced bidirectionally with Tab 2)*:
  - **`Target Model:`** + **`Refresh`** — lists open ChimeraX atomic `Structure` models (automatically excluding ESM-MSR generated mutant clones). For NMR ensembles, choose the specific submodel you want to score.
  - **`Chain:`** — editable dropdown auto-populated from the selected model's chains.
- **`Mutation Scope (Mutually Exclusive)`** — selecting one radio disables the controls for the other three:
  1. **`1. Full Screen:`**
     - Mode dropdown: `singles` (site-saturation mutagenesis) or `singles+doubles` (singles + double-mutant epistasis) (`--mode`).
     - **`Invert (Except)`**: switches `--screen_residues` to `--screen_residues_except` (shared with Mode 4).
     - **`Positions:`** + **`Grab Selection`**: comma-separated PDB residue positions (supports negative indices and insertion codes such as `-2,15A`; leave blank to screen all residues; shared with Mode 4 Stage-1). **`Grab Selection`** reads currently selected atoms restricted strictly to the chosen `Target Model` and `Chain`, preserving insertion codes.
     - **`Filter doubles by WT heavy-atom distance (Å):`** + spinbox (`0.0–100.0`, default `6.0`): enabled only when `singles+doubles` is selected. When checked, passes `--distance_threshold <val> --calculate_distances` (otherwise `--distance_threshold -1`).
  2. **`2. Input CSV:`** + path + **`Browse...`**: passes `--input_csv <file>` (must contain a single unique `pdb_file` + `chain` combination and a mutation column `mut_type`, `mut_type_renumbered`, or `mut_type_pdb`). Skips temporary PDB export.
  3. **`3. Direct Mutations:`** + text field: passes `--mutations <list>` for comma-separated PDB-numbered mutations (e.g., `A12C,A12C:D15E`).
  4. **`4. Fast Top-N Doubles (2-Stage):`**
     - **`Top N:`** (`1–1,000,000`, default `1000`) & **`Visualize Top M:`** (`1–12`, default `5`, synced with Tab 2).
     - **`Filter Cys/Pro before Stage 2`** & optional **`Dist <=`** (`0.0–100.0` Å, default `10.0 Å`, off by default).
     - Invokes `ChimeraX-ESM_MSR/src/fast_doubles.py`, streams stage progress into the status bar and log console, and auto-switches to **`Top-M Ranked Pairs`** upon **`Load Last Output`**.
- **`Screening Parameters`**:
  - **`Mask Strategy:`** — `Checkpoint default` (omits `--mask_strategy` so inference uses the checkpoint's `hparams.yaml` — unmasked for released models), `marginal`, or `independent`.
  - **`Sigma (σ):`** (`0.0–100.0`, step `0.1`, default `1.0`) — passes `--sigma`, scaling the LoRA α parameters (`1.0` = trained ESM-MSR model; `0.0` = zero-shot ESM3 control).
  - **`Replace non-canonical residues`** *(checked by default)* — runs PDBFixer replacement (e.g. `MSE -> MET`) prior to inference. Uncheck to pass `--skip_fix_noncanonical`.
  - **`Skip MT LoRA Pass (Additive Approximation)`** *(unchecked by default; disabled in Mode 4)* — passes `--skip_reverse`. Runs only the fast cached WT-context pass (~10× faster); outputs `wt_lora_pred` only and omits `mt_lora_pred`, `combined_pred`, and epistasis.

#### Collapsible In-Panel Log Console

- **`▸ Show Log Console`** / **`▾ Hide Log Console`**: collapsed by default to keep Tab 1 compact. Clicking reveals a dark monospace log console (2,000-line ring buffer, auto-scrolling) displaying the run header, working directory, full shell-quoted launch command, `[Stage 1/2]` / `[Selection]` / `[Stage 2/2]` banners, and live subprocess stdout/stderr.
- **`Clear`** & **`Copy`**: visible when the console is expanded to clear the buffer or copy the full log to your system clipboard.

---

### Tab 2: `Visualization`

- **CSV Summary Banner** *(top)*: Displays the loaded CSV filename, active chain (or available chains), number of unique single mutations and positions, number of double mutations (if present), `2-stage` badge for unified fast-doubles CSVs, single-mutant score range (`[min, max]` kcal/mol), and any auto-fallback or chain-mismatch notice.
- **`Target Configuration (Synced with Screening)`**: **`Target Model:`**, **`Refresh`**, and **`Chain:`** (kept in sync with Tab 1).
- **`Display Mode and Priority`**:
  - **`Mode:`**
    - `Singles`: Builds one clone (`*_singles_viz`) with the highest-scoring passing single mutant (`swapaa`) at each position, colored using the active **`Palette:`** and range.
    - `WT epistasis`: Filters double mutants to Alanine/Glycine truncations (`X -> A` for non-Ala WT; `A -> G` for Ala WT), maps additive truncation scores onto native WT sidechains (disabling the `WT:` ghost styling row while honoring the `Mutant:` styling row), and renders epistasis pseudobonds.
    - `MT epistasis`: Packs selected double-mutant pairs into minimal non-conflicting clone layers (`*_epi_layer_0`, `*_epi_layer_1`, ...) so overlapping mutations at the same residue do not collide, and connects interacting sidechains with epistasis pseudobonds.
    - `Top-M Ranked Pairs`: Renders the Top-$M$ double-mutant pairs with Okabe–Ito categorical colors, neutral matte backdrop, optional connectors, 3D labels, and a stacked 2D legend.
    - *`WT epistasis` and `MT epistasis` are automatically disabled if the CSV lacks double mutants or epistasis columns; `Top-M Ranked Pairs` is enabled whenever double mutants are present.*
  - **`Select Pairs By:`** *(epistasis modes)*: `Epistasis` or `Stability` (`combined_pred`).
  - **`Display Priority:`** *(epistasis and `Top-M Ranked Pairs` modes)*: `High score`, `Low score`, or `Magnitude` (`|score|`).
  - **`Epistasis Metric:`** *(epistasis and `Top-M Ranked Pairs` modes)*: `epi_pred (1× ensemble)` *(default when present)* or `combined_dddg_pred`. Unavailable metrics are disabled automatically.
- **`Top-M Ranked Pairs Settings`** *(visible only when `Mode:` is `Top-M Ranked Pairs`)*:
  - **`Top M Pairs:`** (`1–12`, default `5`, bidirectionally synced with Tab 1 `Visualize Top M:`).
  - **`Rank By:`** — `Combined ΔΔG` (`combined_pred`), `Epistasis` (`epi_score`), `Stage-1 Additive` (`stage1_additive_pred`), or `Rank Gain (ΔRank)` (`rank_change`).
  - **`Connectors:`** — `Dashed (Subtle)` *(default)*, `None (Labels Only)`, or `Solid`.
  - Checkboxes: **`Disjoint residues`** *(default on)*, **`3D Labels`** *(default on)*, **`2D Legend`** *(default on)*, **`Pub Lighting`** *(default on)*.
- **`Color Scale & Export`**:
  - **`Palette:`** — five verified zero-network color scales applied consistently to 3D mutant sidechains, backbone ribbons (`color byattribute r:esm_msr_score`), epistasis pseudobonds, the ChimeraX `key` colorbar, and 2D heatmap exports:

    | Palette Label | CLI Key | Type | ChimeraX Stops (Destabilizing → Stabilizing) | Matplotlib Colormap |
    |---|---|---|---|---|
    | `Blue–White–Red (Classic)` *(default)* | `blue_white_red` | Diverging | `blue:white:red` | `RdBu_r` |
    | `Purple–White–Green (PRGn, Colorblind-Safe)` | `purple_white_green` | Diverging | `#762a83:#f7f7f7:#1b7837` | `PRGn` |
    | `Brown–White–Teal (BrBG, Colorblind-Safe)` | `brown_white_teal` | Diverging | `#8c510a:#f5f5f5:#01665e` | `BrBG` |
    | `Viridis (Perceptually Uniform)` | `viridis` | Sequential | `#440154:#3b528b:#21918c:#5ec962:#fde725` | `viridis` |
    | `Cividis (Colorblind-Safe Sequential)` | `cividis` | Sequential | `#00204c:#414d6b:#7c7b78:#bcaf6f:#ffea46` | `cividis` |

  - **`Auto Range`** *(checked by default)* — automatically sets symmetric 98th-percentile bounds `[-b, +b]` for diverging palettes or `[p2, p98]` for sequential palettes. Uncheck to unlock manual **`Min:`** and **`Max:`** spinboxes (`-20.0` to `+20.0`, step `0.5`, defaults `-3.0` / `+3.0`).
  - **`Export 2D Heatmap...`** — saves a 20 × $L$ single-mutant ΔΔG landscape heatmap (`.png`, `.pdf`, or `.svg`, 200 DPI) grouped by amino-acid physicochemical class (`Charged: D E K R H | Polar: N Q S T | Aliphatic: A V I L M | Aromatic: F Y W | Special: C G P`), with wild-type diamond markers (`◆`), gray missing/WT cells, and the currently selected palette and range.
  - **`Export Pair Card...`** — saves a publication-ready legend card (`.png`, `.pdf`, or `.svg`, 200 DPI) for the Top-$M$ double-mutant pairs showing each pair's categorical swatch, `#rank`, `mut1 + mut2`, `Combined ΔΔG`, and (when present in the CSV) `Stage-1 Add.`, `Epistasis`, `ΔRank`, and `Dist (Å)`.
- **`Ranked Results`** *(interactive table)*:
  - **`Filter mutation / residue...`** — instant substring filter on mutation strings or residue numbers (e.g. `A12V`, `E79`, `45`).
  - **`View:`** — `Auto` (follows active `Mode:`), `Ranked Pairs`, `All Singles`, or `All Doubles`.
  - **`Isolate Pair`** & **`Show All`** — spotlight the currently selected Top-$M$ pair row in 3D or restore all rendered pairs.
  - **Sortable Table** (up to 200 visible rows with row count indicator):
    - *Ranked Pairs / All Doubles columns*: `Rank` (with Okabe–Ito color swatch icon in `Ranked Pairs`), `Mutation`, `ΔΔG`, `Epistasis`, `Stage-1`, `ΔRank`, `Dist (Å)`.
    - *All Singles columns*: `Rank`, `Mutation`, `Pos`, `ΔΔG`, `WT`, `MT`.
    - All numeric columns sort numerically (missing values sorted last). Clicking any row selects the residue(s) on the target model and focuses the ChimeraX camera (`view #<id>/<chain>:<pos> pad 0.35`).
- **`Global Filters`**:
  - **`No WT Cys`**, **`No WT Pro`**, **`No Mut Cys`**, **`No Mut Pro`** — exclude mutations with wild-type or mutant Cysteine (`C`) or Proline (`P`). Checking `No Mut Cys` helps filter out false positives associated with protease-assay disulfide artifacts.
- **`Score Selection (Additive Base)`**:
  - **`Dual-view predictions (recommended; required for epistasis)`** (`combined_pred`)
  - **`WT LoRA predictions (additive)`** (`wt_lora_pred`)
  - **`MT LoRA predictions (not recommended)`** (`mt_lora_pred`)
  - When a CSV generated with `--skip_reverse` is loaded, `combined_pred` and `mt_lora_pred` are automatically disabled and `WT LoRA predictions (additive)` is auto-selected. On unified two-stage fast-doubles CSVs, Stage-1 singles automatically use `wt_lora_pred` while Stage-2 doubles use `combined_pred`.
- **`Global Thresholds and Networks (kcal/mol, positive = stabilizing)`**:
  - **`Pos Threshold:`** (`0.0–1000.0`, default `0.0`) & **`Neg Threshold:`** (`-1000.0–0.0`, default `0.0`) — displays mutations/pairs satisfying `score >= Pos Threshold` **or** `score <= Neg Threshold`. (To view only stabilizing variants, set `Neg Threshold` to a large negative number such as `-10.0`; to view only destabilizing variants, set `Pos Threshold` to `10.0`.)
  - **`Non-Target Chain Transp %:`** (`0–100`, default `90`) & **`Hide Fully`** — dims or hides chains other than `Chain:` in the active model.
  - **`Max Interactions / Pos:`** (`1–20`, default `1`, epistasis modes only) — caps the number of epistasis pseudobonds incident on any single residue position; undirected residue pairs are deduplicated.
  - **`Color Backbone by Highest Additive ΔΔG`** *(enabled in `Singles` and `WT epistasis`)* — colors the target chain ribbon by each residue's best single-mutant ΔΔG using `color byattribute r:esm_msr_score` with the active palette.
  - **`Nearby atoms within:`** + spinbox (`0.0–50.0` Å, default `3.0` Å) + **`Exclude backbone`** *(checked by default)* — highlights wild-type contacting atoms and residues within the specified radius of displayed mutant sidechains (automatically including `CA` for Glycine mutants) without modifying your ChimeraX selection.
- **`Rendering and Styling`**:
  Six-column table (`Element`, `Color (name/#hex)`, color-picker swatch button, `Style`, `Transp %`, `Hide`) for four structural elements:

  | Element | Default Color | Default Style | Default Transp % | Default Hide | Notes |
  |---|---|---|---|---|---|
  | **`WT:`** | `white` | `stick` | `100` | Checked | Native "ghost" WT sidechains for geometric comparison (disabled in `WT epistasis` and `Top-M Ranked Pairs`). |
  | **`Mutant:`** | *(blank = score)* | `stick` | `0` | Unchecked | Leave blank to color carbons by additive ΔΔG score using the selected `Palette:`; enter a color name / `#hex` or click the swatch button to override. |
  | **`Near Atom:`** | `purple` | `ball` | `0` | — | Contacting WT atoms within the distance cutoff (enabled when `Nearby atoms within:` is checked). |
  | **`Near Res:`** | `grey` | `stick` | `80` | Unchecked | Full residues containing contacting WT atoms (enabled when `Nearby atoms within:` is checked). |

  Non-carbon atoms (`~C`) are automatically colored by element (`byelement`). Epistasis pseudobonds in `"ESM-MSR Epistasis"` also follow the active `Palette:`, with radius (`0.05–0.40` Å) and opacity scaling continuously with score magnitude above the threshold alongside a ChimeraX `key` colorbar in the lower-left corner.

---

### Global Action Bar & Status Footer

- **Row 1**:
  - **`Run Prediction Script`** — validates all inputs before creating temporary files, exports the active target model if needed, and starts `inference.py` or `fast_doubles.py`. Switches to a red **`STOP`** button while running.
  - **`Load CSV`** — opens a file dialog (defaulting to the last completed output directory when available) to load and visualize any ESM-MSR prediction CSV.
  - **`Load Last Output`** — one-click shortcut (enabled whenever a finished run output or the configured `Output CSV` exists on disk) that switches to Tab 2 (auto-selecting `Top-M Ranked Pairs` after a Mode 4 run) and loads the file immediately.
- **Row 2**:
  - **`Update Visualization`** — re-parses the currently loaded CSV and refreshes the 3D scene and `Ranked Results` table using the current Tab 2 settings and selected `Chain`.
  - **`Reset / Clear Viz`** — closes all ESM-MSR generated mutant clones, deletes only the `"ESM-MSR Epistasis"` and `"ESM-MSR Ranked Pairs"` pseudobond groups and `esm_msr_leg_*` 2D legend labels, and resets the target model to a clean white ribbon without disturbing other open models or user pseudobonds.
- **Footer**:
  - Compact **`QProgressBar`** (indeterminate during model loading, determinate `0–100%` on `tqdm` output across Stage 1 and Stage 2).
  - Color-coded **`Status:`** label (`info`, green `success`, amber `warning`, red `error`).
  - **`Predicted output file:`** label showing the most recent run output filename.

---

## Native ChimeraX CLI Commands Reference

Six native ChimeraX commands are registered in `bundle_info.xml` and `src/commands.py`. All commands work identically in an interactive ChimeraX session (staying in sync with the GUI panel if open) **and** in headless scripted sessions (`chimerax --offscreen --script ...` or `.cxc` command scripts).

### Command Summary

#### 1. `esm_msr open`
Open or raise the **ESM-MSR Predictor & Visualizer** GUI panel.
```
esm_msr open
```

#### 2. `esm_msr load`
Load an ESM-MSR prediction CSV and visualize it on the target structure.
```
esm_msr load <csv_path> [model <models>] [chain <str>] [mode <str>]
```
- `csv_path` *(required, file path)*: Path to an ESM-MSR prediction CSV.
- `model` *(optional, ChimeraX model spec, e.g. `#1`)*: Target atomic structure (defaults to first non-clone atomic structure).
- `chain` *(optional, string)*: Target chain ID (e.g. `A`).
- `mode` *(optional, string)*: `Singles` *(default)*, `WT epistasis`, `MT epistasis`, or `Top-M Ranked Pairs` (also accepts short aliases `singles`, `epistasis`, `pairs`).

#### 3. `esm_msr visualize`
Re-render the currently loaded CSV with a custom mode, color palette, score bounds, or thresholds.
```
esm_msr visualize [mode <str>] [palette <str>] [minScore <float>] [maxScore <float>] [posThreshold <float>] [negThreshold <float>]
```
- `mode`: `Singles`, `WT epistasis`, `MT epistasis`, or `Top-M Ranked Pairs`.
- `palette`: `blue_white_red`, `purple_white_green` (or `PRGn`), `brown_white_teal` (or `BrBG`), `viridis`, or `cividis`.
- `minScore`, `maxScore`: Manual colormap bounds (omit both for automatic 98th-percentile scaling).
- `posThreshold`, `negThreshold`: Score display thresholds in kcal/mol (default `0.0`).

#### 4. `esm_msr pairs`
Render publication-grade **Top-M Ranked Pairs** with Okabe–Ito categorical pair coloring, subtle connectors, 3D labels, and 2D legend.
```
esm_msr pairs <csv_path> [model <models>] [chain <str>] [topM <int>] [sortBy <str>] [diverse <bool>] [connectors <str>] [labels <bool>] [legend <bool>] [pubLighting <bool>] [isolate <int>]
```
- `csv_path` *(required, file path)*: Path to a two-stage fast-doubles or `singles+doubles` CSV.
- `model` *(optional)*: Target atomic model (e.g. `#1`).
- `chain` *(optional)*: Target chain ID.
- `topM` *(int, default `5`, range `1–12`)*: Number of ranked pairs to display.
- `sortBy` *(string, default `combined_pred`)*: Ranking metric — `combined_pred` (`combined`, `stability`, `ddg`), `epi_score` (`epi`, `epi_pred`, `epistasis`), `stage1_additive_pred` (`stage1`, `additive`), or `rank_change` (`rank_gain`, `drank`).
- `diverse` *(bool, default `true`)*: Enforce disjoint residue positions across displayed pairs.
- `connectors` *(string, default `dashed`)*: `dashed`, `none`, or `solid`.
- `labels` *(bool, default `true`)*: Draw color-matched 3D `#<rank> <mut>` residue labels.
- `legend` *(bool, default `true`)*: Draw stacked top-left 2D legend (`esm_msr_leg_*`).
- `pubLighting` *(bool, default `true`)*: Apply white background, soft lighting, and silhouettes.
- `isolate` *(optional int)*: Spotlight only pair `#<rank>` (1-based) and zoom the camera to its residues.

#### 5. `esm_msr heatmap`
Export a publication-quality 20 × $L$ single-mutant landscape heatmap image directly from ChimeraX.
```
esm_msr heatmap <csv_path> <output_path> [chain <str>] [palette <str>] [minScore <float>] [maxScore <float>] [title <str>]
```
- `csv_path` *(required, file path)*: ESM-MSR CSV containing single-mutant predictions.
- `output_path` *(required, save path)*: Output `.png`, `.pdf`, or `.svg` file.
- `chain` *(optional)*: Chain ID to plot (defaults to first chain present).
- `palette` *(optional, default `blue_white_red`)*: Any of the 5 palette keys or matplotlib names (`PRGn`, `BrBG`, `viridis`, `cividis`).
- `minScore`, `maxScore` *(optional floats)*: Manual colorbar limits.
- `title` *(optional string)*: Custom figure title.

#### 6. `esm_msr reset`
Close all ESM-MSR mutant clones, delete `"ESM-MSR Epistasis"` and `"ESM-MSR Ranked Pairs"` pseudobond groups and `esm_msr_leg_*` 2D legend labels, and restore the target structure to a clean white ribbon.
```
esm_msr reset
```

### Complete Reproducible `.cxc` Figure-Generation Script Example

Save the following as `make_esm_msr_figures.cxc` and run it interactively (`open make_esm_msr_figures.cxc`) or headlessly (`chimerax --offscreen --exit make_esm_msr_figures.cxc`):

```chimerax
# 1. Load structure and set clean publication window size
open /path/to/CCDB_ECOLI.pdb
windowsize 1600 1200
view #1

# 2. Panel A: Top-5 disjoint double mutants ranked by Combined ΔΔG (Okabe-Ito categorical colors)
esm_msr pairs /path/to/ccdb_fast_doubles_top1000.csv model #1 chain A topM 5 sortBy combined_pred diverse true connectors dashed labels true legend true pubLighting true
save figures/fig_top5_pairs_overview.png width 2400 height 1800 supersample 3

# 3. Panel B: Spotlight Pair #1 close-up
esm_msr pairs /path/to/ccdb_fast_doubles_top1000.csv model #1 chain A topM 5 sortBy combined_pred diverse true connectors dashed labels true legend true pubLighting true isolate 1
save figures/fig_pair1_spotlight.png width 2400 height 1800 supersample 3

# 4. Panel C: Export companion 2D single-mutant landscape heatmap (colorblind-safe PRGn)
esm_msr heatmap /path/to/ccdb_fast_doubles_top1000.csv figures/fig_singles_heatmap_prgn.png chain A palette purple_white_green title "CcdB Single-Mutant Stability Landscape (ESM-MSR)"

# 5. Panel D: Single-mutant 3D stability view on PRGn palette (stabilizing >= +0.5 kcal/mol)
esm_msr reset
esm_msr load /path/to/ccdb_fast_doubles_top1000.csv model #1 chain A mode Singles
esm_msr visualize mode Singles palette purple_white_green minScore -3.0 maxScore 3.0 posThreshold 0.5 negThreshold -10.0
view #1
save figures/fig_singles_3d_prgn.png width 2400 height 1800 supersample 3

# 6. Clean up all ESM-MSR overlays
esm_msr reset
```

---

## CSV Input Schema & Scientific Conventions

### Sign Conventions
- **Stability (ΔΔG, kcal/mol)**: **Positive (`> 0`) = stabilizing**, **Negative (`< 0`) = destabilizing**.
- **Epistasis (ΔΔΔG, kcal/mol)**: **Positive (`> 0`) = synergistic**, **Negative (`< 0`) = antagonistic**.

### Expected Columns

| Column | Required? | Description |
|---|---|---|
| `chain` | **Required** | PDB chain ID (filtered against the selected `Chain:` in the GUI/CLI). |
| `mut_type_pdb` | **Required** | Colon-separated PDB-numbered mutation string(s). |
| `combined_pred` | One of the 3 score cols required | Dual-view WT + MT ensemble ΔΔG prediction (recommended; required for epistasis modes). |
| `wt_lora_pred` | One of the 3 score cols required | WT-context LoRA ΔΔG prediction (always present, including `--skip_reverse` and two-stage Stage-1 singles). |
| `mt_lora_pred` | One of the 3 score cols required | MT-context LoRA ΔΔG prediction (absent when `--skip_reverse` is used). |
| `epi_pred` | Required for epistasis (preferred) | Authoritative 1× ensemble epistasis: $\text{comb}(AB) - \text{WT}(AB) = \lambda\,(\text{MT}(AB) - \text{WT}(AB))$ (default $\lambda = 0.5$; `0.0` on singles). |
| `combined_dddg_pred` | Alternate epistasis col | Additive-subtracted epistasis: $\text{comb}(AB) - (\text{comb}(A) + \text{comb}(B))$ (equals `epi_pred` if run with `--skip_additive`). |
| `stage1_additive_pred`, `additive_rank`, `full_rank`, `rank_change`, `dist` | Optional *(written by `fast_doubles.py`)* | Two-stage additive baseline, before/after ranks, rank promotion (`additive_rank - full_rank`), and WT heavy-atom distance (Å). |
| `pdb_file`, `code`, `mut_type_renumbered`, `*_additive`, `mt_lora_dddg_pred` | Optional | Standard metadata and auxiliary columns written by `src/esm_msr/inference.py`. |

### Parsing & Validation Rules
- **Single-mutant requirement**: Single-mutant rows (`k = 1`) are required in all visualization modes (`Singles`, `WT epistasis`, `MT epistasis`, and `Top-M Ranked Pairs` / 2D heatmap). Both `--mode singles+doubles` and two-stage `fast_doubles.py` emit unified CSVs containing singles and doubles automatically.
- **Two-stage unified CSV handling**: When a unified CSV produced by `fast_doubles.py` is loaded (Stage-1 singles carry `wt_lora_pred` while Stage-2 doubles carry full `combined_pred` and `epi_pred`), the parser automatically uses Stage-1 `wt_lora_pred` for single-mutant heatmaps/sidechain coloring and Stage-2 `combined_pred`/`epi_pred` for double-mutant ranking and visualization.
- **`mut_type_pdb` format**: Each `:`-delimited token must match `^([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$`:
  - Standard single: `G289A`
  - Standard double: `A12C:D15E`
  - Negative PDB residue number: `G-2A`
  - PDB insertion code: `A15AC` (wild-type `A` at position `15A` mutated to `C`)
  - If *any* token in a multi-mutant string is malformed, the row is skipped with a warning rather than silently downgraded to a single mutant.
- **Numeric robustness**: Rows with `NaN` or `±Inf` scores are ignored with a warning; exact `0.0` scores (common in zero-shot `--sigma 0.0` runs) are preserved. Duplicate single-mutant entries for the same `(chain, position, mutant_aa)` are averaged.
- **Case-duplicate columns**: If an input CSV contains columns differing only in case (e.g., `CHAIN` and `chain`), the exact lowercase column is kept (or the first occurrence otherwise), ensuring compatibility with both pandas 2.x and pandas 3.x.
- **Multi-chain filtering**: Rows are filtered strictly to the selected `Chain:` in **Target Configuration**.

---

## Troubleshooting & FAQ

### 1. `PermissionError` or `temp_fixed.pdb` errors during inference
The plugin runs `inference.py` and `fast_doubles.py` inside a fresh writable temporary `work/` directory (cleaned up automatically when the run finishes, is stopped, or the tool closes) while resolving all relative user paths against `Base Repo Dir`. Thus ChimeraX's launch directory is never polluted and read-only launch directories never cause `temp_fixed.pdb` write errors. If you invoke `src/esm_msr/inference.py` manually from the command line, run it from a writable directory or pass `--skip_fix_noncanonical`.

### 2. Loading a CSV created with `--skip_reverse`
When `--skip_reverse` (`Skip MT LoRA Pass`) is enabled, `inference.py` omits all-NaN columns (`mt_lora_pred` and `combined_pred`). The visualizer automatically detects this on load, selects `WT LoRA predictions (additive)` (`wt_lora_pred`), displays a notice in the top summary banner, and disables `Dual-view`, `MT LoRA`, and epistasis display modes.

### 3. "Selected chain 'X' has no matching rows in CSV"
The CSV summary banner lists the chain ID(s) present in the file and retains the loaded CSV path. Switch **`Chain:`** under **Target Configuration** to one of the available chains and click **`Update Visualization`** — no need to browse for the CSV again.

### 4. CUDA Out-of-Memory (OOM) errors during inference
- Switch **`Model Precision (dtype):`** from `float32` to **`bfloat16`** (Ampere+ GPUs) or **`float16`**.
- Lower **`Batch Size:`** (e.g., from `16` to `4` or `1`).
- For whole-protein double-mutant discovery, use **`4. Fast Top-N Doubles (2-Stage):`** instead of exhaustive `singles+doubles`.
- For exhaustive `singles+doubles` screens, enable **`Filter doubles by WT heavy-atom distance (Å):`** (e.g. `6.0` Å) or target specific residues with **`Grab Selection`**.
- For single-mutant screens on large proteins, enable **`Skip MT LoRA Pass (Additive Approximation)`** (runs a single cached WT forward at batch size 1).

### 5. `Python Env` resolution fails / `conda` not found on PATH
GUI launches of ChimeraX on macOS, Linux desktops, or Windows sometimes do not inherit your shell's `PATH`. Click **`Auto-Detect`** next to **`Python Env:`** to locate your conda/mamba/venv Python binary directly on disk (no `conda` executable on `PATH` required), or point **`Python Env:`** directly to the environment folder or its `python` / `python.exe` executable.

### 6. Will visualization overwrite my other ChimeraX pseudobonds, labels, or models?
No:
- Epistasis edges and ranked-pair connectors are stored in dedicated `"ESM-MSR Epistasis"` and `"ESM-MSR Ranked Pairs"` `PseudobondGroup`s; updating or resetting deletes **only** those groups and leaves hydrogen bonds, metal coordination bonds, and distance monitors untouched.
- 2D legend labels are tracked under the `esm_msr_leg_*` prefix and closed cleanly without affecting any user-created `2dlabels`.
- Hydrogen hiding (`hide ... & H`) applies only to the target model and ESM-MSR mutant clones.
- Generated mutant clones (`*_singles_viz`, `*_epi_layer_*`, `*_pair_*`) are tracked internally and filtered out of the `Target Model` dropdowns. Click **`Reset / Clear Viz`** (or run `esm_msr reset`) to close them and restore the target structure.

### 7. Running zero-shot ESM3 predictions (`--sigma 0.0`)
Keep a valid checkpoint selected (such as `LoRA_models/esm-msr-small/lora_seed1.safetensors`) and set **`Sigma (σ):`** to `0.0`. In Modes 1–3, you can also leave **`Checkpoint:`** empty, specify `LoRA_models/esm-msr-small/hparams.yaml` in **`LoRA Config:`**, and set **`Sigma (σ):`** to `0.0`.

### 8. Where can I inspect the exact inference command or traceback?
Expand **`▸ Show Log Console`** at the bottom of Tab 1 (with **`Clear`** and **`Copy`** buttons) or open the ChimeraX Log (`Tools -> General -> Log`). Every run logs its working directory, full shell-quoted command line, stage banners, and all stdout/stderr lines.

---

## Development & Automated Headless Testing

### Bundle Layout

```
ChimeraX-ESM_MSR/
├── bundle_info.xml          # ChimeraX bundle manifest (ChimeraX-ESM-MSR v0.5.0, tool + 6 CLI commands)
├── README.md                # This documentation
├── src/
│   ├── __init__.py          # BundleAPI entrypoint (get_class, start_tool, register_command)
│   ├── parsing.py           # CSV parsing, palettes, Top-M pair selection & disk-only env discovery
│   ├── fast_doubles.py      # Two-stage Fast Double Mutant Selection engine (library + standalone CLI)
│   ├── heatmap.py           # Headless matplotlib 2D mutational heatmap & Top-M pair legend card exporters
│   ├── commands.py          # Shared headless-safe 3D renderer & 6 native ChimeraX CLI commands
│   └── tool.py              # PyQt6 GUI, QProcess runner, Ranked Results table & ChimeraX visualization
└── tests/                   # Automated unit, offscreen PyQt6 & ChimeraX integration tests
```

### Building & Verifying the Wheel

```bash
chimerax --nogui --cmd "devel build /path/to/esm-msr/ChimeraX-ESM_MSR; exit"
```

### Running Automated Headless Tests

The automated test suite under `ChimeraX-ESM_MSR/tests/` covers `parsing.py`, `fast_doubles.py`, `heatmap.py`, headless PyQt6 widget/layout and subprocess-argv tests, and real ChimeraX `--offscreen` 3D rendering and native CLI integration tests (including Top-M categorical pairs, pair isolation, 2D legend cleanup, negative residue numbers, insertion codes, Glycine contacts, dedicated pseudobond groups, and `Reset / Clear Viz`).

Run the complete test suite with ChimeraX's bundled Python (unset `DISPLAY` and `WAYLAND_DISPLAY` on Linux/WSL so Qt runs purely offscreen):

```bash
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \
  /usr/lib/ucsf-chimerax/bin/python3.11 -m pytest ChimeraX-ESM_MSR/tests/ -v
```

Or run a test module directly inside a headless ChimeraX session:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \
  chimerax --offscreen --exit --script ChimeraX-ESM_MSR/tests/test_chimerax_esm_msr.py
```

*(Adjust the ChimeraX binary and bundled Python paths for your OS installation as needed.)*
