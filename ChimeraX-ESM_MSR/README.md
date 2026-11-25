# ChimeraX-ESM-MSR

Interactive UCSF ChimeraX plugin (**v0.4.0**, bundle `ChimeraX-ESM-MSR`, package `chimerax.ESM_MSR`) for running live **ESM-MSR** protein stability and epistasis inference and visualizing single- and double-mutant effects directly on 3D structures.

---

## Table of Contents

1. [Overview & Key Capabilities](#overview--key-capabilities)
2. [Requirements & Compatibility](#requirements--compatibility)
3. [Installation & Setup](#installation--setup)
4. [Quickstart Workflows](#quickstart-workflows)
5. [Complete UI Reference](#complete-ui-reference)
   - [Tab 1: `Execution / IO`](#tab-1-execution--io)
   - [Tab 2: `Visualization`](#tab-2-visualization)
   - [Global Action Bar & Status Footer](#global-action-bar--status-footer)
6. [CSV Input Schema & Scientific Conventions](#csv-input-schema--scientific-conventions)
7. [Troubleshooting & FAQ](#troubleshooting--faq)
8. [Development & Automated Headless Testing](#development--automated-headless-testing)

---

## Overview & Key Capabilities

**ESM-MSR Predictor & Visualizer** bridges the ESM-MSR stability predictor (`src/esm_msr/inference.py`) with UCSF ChimeraX in a single two-tab docked tool (`Tools -> Stability -> ESM_MSR`):

- **Live Subprocess Inference**: Export any open PDB/mmCIF structure (or specific model in an NMR ensemble) and launch `src/esm_msr/inference.py` in your local Python/conda/venv environment with unbuffered live `tqdm` progress reporting, full command logging for reproducibility, and safe process-tree termination (`STOP`).
- **Flexible Screening Modes**: Run exhaustive single-mutant (`singles`) or epistasis (`singles+doubles`) scans on all residues, a ChimeraX selection (`Grab Selection`), an inverted exclusion set, an existing input CSV, or explicit comma-separated multi-mutants — with optional WT heavy-atom distance filtering and zero-shot ESM3 control (`Sigma (σ) = 0.0`).
- **Interactive 3D Stability & Epistasis Visualization**:
  - **Singles**: Displays top passing single-mutant sidechains on a cloned mutant structure using ChimeraX `swapaa`, colored on a dynamic Pink–White–Green ΔΔG scale with optional ribbon backbone coloring and nearby contacting atom/residue highlights.
  - **WT Epistasis**: Evaluates Alanine/Glycine double-mutant scans mapped onto native wild-type sidechains with score-scaled epistasis pseudobonds.
  - **MT Epistasis**: Automatically packs double-mutant sidechains into minimal non-conflicting clone layers (`*_epi_layer_0`, `*_epi_layer_1`, ...) and draws Orange–White–Blue epistasis network edges.
- **Non-Destructive ChimeraX Integration**: All epistasis edges live in a dedicated `"ESM-MSR Epistasis"` `PseudobondGroup`, hydrogen hiding is scoped to active ESM-MSR models, user atom selections are preserved during contact queries, residue scores use the ChimeraX Python attribute API (`residue_score_viz_score`), and **Reset / Clear Viz** cleanly restores the target model without touching unrelated models or user pseudobonds.
- **Smart CSV Handling**: Supports both `epi_pred` (authoritative 1× ensemble epistasis) and `combined_dddg_pred`, automatically detects `--skip_reverse` outputs (auto-selecting `wt_lora_pred` and disabling unavailable controls), handles negative PDB residue numbers and insertion codes (e.g. `G-2A`, `A15AC`), deduplicates case-variant columns safely, and filters by target chain.

---

## Requirements & Compatibility

### ChimeraX Environment
- **UCSF ChimeraX**: `>= 1.3` (declared in `bundle_info.xml`).
- **Verified Platform**: UCSF ChimeraX **1.11.1** (bundled Python **3.11**, **PyQt6 6.9**, and **pandas 2.x & 3.x**, including pandas `3.0.1`).
- **Bundle Dependencies**: `ChimeraX-Core >= 1.3`, `pandas >= 2.0` (automatically installed by ChimeraX during `devel install` / `toolshed install`).

### ESM-MSR Inference Environment (for live prediction)
Visualization of pre-computed CSV files works directly inside ChimeraX with no PyTorch installation required. To run live inference from the **Execution / IO** tab, you also need a separate Python/conda/venv environment containing:
- Python `3.11–3.13`
- `esm-msr` installed in editable mode (`pip install -e .` from the repository root)
- PyTorch (CUDA 12.8 recommended for GPU acceleration; Apple Silicon `mps` and `cpu` are also supported)
- ESM3-small-open weights (either auto-downloaded via `esm`/HuggingFace or cloned locally from [`biohub/esm3-sm-open-v1`](https://huggingface.co/biohub/esm3-sm-open-v1))

See the [main repository README](../README.md#recommended-installation) for step-by-step Python environment and CUDA setup instructions.

> **WSL2 Note:** If you installed `esm-msr` inside Windows Subsystem for Linux (WSL), install ChimeraX on Windows rather than inside WSL for interactive GUI use. For Linux/WSL automated testing or scripted rendering, headless `chimerax --offscreen` is supported.

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

Build the wheel (`dist/chimerax_esm_msr-0.4.0-py3-none-any.whl`) and install it via ChimeraX Toolshed:

```
devel build /path/to/esm-msr/ChimeraX-ESM_MSR
toolshed install /path/to/esm-msr/ChimeraX-ESM_MSR/dist/chimerax_esm_msr-0.4.0-py3-none-any.whl
```

Headless CLI equivalent:
```bash
chimerax --nogui --cmd "devel build /path/to/esm-msr/ChimeraX-ESM_MSR; exit"
chimerax --nogui --cmd "toolshed install /path/to/esm-msr/ChimeraX-ESM_MSR/dist/chimerax_esm_msr-0.4.0-py3-none-any.whl; exit"
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

Open the plugin from the ChimeraX menu bar:

> **`Tools -> Stability -> ESM_MSR`**

The docked panel opens titled **ESM-MSR Predictor & Visualizer**. Your path fields (`Base Repo Dir`, `Python Env`, `Output CSV`, `Checkpoint`, `LoRA Config`, `ESM3 Weights`) are persisted automatically between sessions via `QSettings("ESM_MSR_Tools", "ESM_MSR")`.

---

## Quickstart Workflows

### Workflow A: Visualize an Existing Prediction CSV (Demo: `1UFM`)

No GPU or PyTorch setup is needed to test visualization with the bundled example CSV (`data/example/1ufm_example.csv`):

1. Load the demo structure in ChimeraX:
   ```
   open 1ufm
   ```
2. Open **`Tools -> Stability -> ESM_MSR`** and switch to the **`Visualization`** tab.
3. Confirm **`Target Model:`** is `#1 1ufm` and **`Chain:`** is `A`.
4. Click **`Load CSV`** in the global action bar and select `data/example/1ufm_example.csv` from your `esm-msr` checkout.
5. Inspect the top summary banner (filename, chain, number of singles/positions, and score range in kcal/mol).
6. Refine the view as desired — for example:
   - Set **`Pos Threshold:`** to `0.5` and **`Neg Threshold:`** to `-10.0` to display only stabilizing mutations (ΔΔG ≥ +0.5 kcal/mol).
   - Check **`No Mut Cys`** under **Global Filters** to omit mutations to Cysteine.
   - Check **`Color Backbone by Highest Additive ΔΔG`** and/or **`Nearby atoms within:`** (`3.0` Å).
   - Click **`Update Visualization`** to re-render.
7. Click **`Reset / Clear Viz`** whenever you want to remove generated mutant clones and epistasis lines and restore the structure to a clean white ribbon.

### Workflow B: Run Live Inference from a Loaded PDB/mmCIF Structure

1. Load a structure in ChimeraX (e.g., `open 1ufm` or drag-and-drop a `.pdb`/`.cif` file).
2. Open **`Tools -> Stability -> ESM_MSR`** on the **`Execution / IO`** tab.
3. Click **`Browse...`** next to **`Base Repo Dir:`** and select your `esm-msr` repository root.
   - Setting `Base Repo Dir` automatically fills `Python Env` (`<repo>/msr_venv`), `Output CSV` (`<repo>/example_inference.csv`), and `Checkpoint` (`LoRA_models/esm-msr-small/lora_seed1.safetensors`) if those fields are empty.
4. Confirm **`Python Env:`** points to your inference environment (a conda environment name such as `msr_venv`, a venv/conda folder path, or the direct `python`/`python.exe` binary path).
5. Under **`Compute Environment`**, pick your **`Compute Device:`** (`cuda` / `mps` / `cpu`), **`Model Precision (dtype):`** (`bfloat16` recommended on Ampere+ CUDA GPUs; `float32` for CPU), and **`Batch Size:`** (default `16`).
6. Under **`Screening Config`**:
   - Confirm **`Target Model:`** and **`Chain:`**.
   - Select **`1. Full Screen:`** (`singles` for site-saturation mutagenesis, or `singles+doubles` with **`Filter doubles by WT heavy-atom distance (Å):`** checked at `6.0` Å for tractable epistasis scans).
   - Optionally select residues in the 3D viewport and click **`Grab Selection`** to populate **`Positions:`**.
   - For a fast (<1 minute) single-mutant screen, check **`Skip MT LoRA Pass (Additive Approximation)`**.
7. Click **`Run Prediction Script`**.
   - The button switches to a red **`STOP`** button and the progress bar activates (indeterminate during model initialization, then determinate 0–100% as `tqdm` progresses).
   - The exact launched CLI command is recorded in the ChimeraX Log.
8. When the status reports completion, click **`Load Last Output`** to jump directly to the **`Visualization`** tab and render the new CSV.

---

## Complete UI Reference

The plugin window (**ESM-MSR Predictor & Visualizer**) is organized into two vertically scrollable tabs (`QScrollArea`) above a shared global action bar and status footer.

### Tab 1: `Execution / IO`

#### Local Path Configuration

| Control | Description & Behavior |
|---|---|
| **`Base Repo Dir:`** (`Browse...`) | Root directory of your `esm-msr` checkout (must contain `src/esm_msr/inference.py`). All relative user paths in Tab 1 resolve against this directory, and `<Base Repo Dir>/src` is prepended to `PYTHONPATH` for the subprocess. |
| **`Python Env:`** (`Browse...`) | Python interpreter used to launch inference. Accepts: **(1)** direct path to a `python` / `python.exe` binary (checked first), **(2)** path to a `venv` or conda environment directory (checks `python.exe`, `Scripts/python.exe`, `bin/python`, `bin/python3`), **(3)** a conda environment name (invoked via `conda run --no-capture-output -n <name> python -u`), or **(4)** blank to use `python`/`python3` from `PATH`. Always preserves `-u` and `PYTHONUNBUFFERED=1`. |
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
> - **Alternative (checkpoint blank)**: Leave **`Checkpoint:`** blank (the plugin omits `--checkpoint_path`), provide a **`LoRA Config:`** (e.g., `LoRA_models/esm-msr-small/hparams.yaml`), and set **`Sigma (σ):`** to `0.0`.

#### Screening Config

- **`Target Selection`** *(synced bidirectionally with Tab 2)*:
  - **`Target Model:`** + **`Refresh`** — lists open ChimeraX atomic `Structure` models (automatically excluding ESM-MSR generated mutant clones). For NMR ensembles, choose the specific submodel you want to score.
  - **`Chain:`** — editable dropdown auto-populated from the selected model's chains.
- **`Mutation Scope (Mutually Exclusive)`** — selecting one radio disables the controls for the other two:
  1. **`1. Full Screen:`**
     - Mode dropdown: `singles` (site-saturation mutagenesis) or `singles+doubles` (singles + double-mutant epistasis) (`--mode`).
     - **`Invert (Except)`**: switches `--screen_residues` to `--screen_residues_except`.
     - **`Positions:`** + **`Grab Selection`**: comma-separated PDB residue positions (supports negative indices and insertion codes such as `-2,15A`; leave blank to screen all residues). **`Grab Selection`** reads currently selected atoms restricted strictly to the chosen `Target Model` and `Chain`, preserving insertion codes.
     - **`Filter doubles by WT heavy-atom distance (Å):`** + spinbox (`0.0–100.0`, default `6.0`): enabled only when `singles+doubles` is selected. When checked, passes `--distance_threshold <val> --calculate_distances` (otherwise `--distance_threshold -1`). Recommended for proteins >100 residues to keep double-mutant runtimes tractable.
  2. **`2. Input CSV:`** + path + **`Browse...`**: passes `--input_csv <file>` (must contain a single unique `pdb_file` + `chain` combination and a mutation column `mut_type`, `mut_type_renumbered`, or `mut_type_pdb`). Skips temporary PDB export.
  3. **`3. Direct Mutations:`** + text field: passes `--mutations <list>` for comma-separated PDB-numbered mutations (e.g., `A12C,A12C:D15E`).
- **`Screening Parameters`**:
  - **`Mask Strategy:`** — `Checkpoint default` (omits `--mask_strategy` so inference uses the checkpoint's `hparams.yaml` — unmasked for released models), `marginal`, or `independent`.
  - **`Sigma (σ):`** (`0.0–100.0`, step `0.1`, default `1.0`) — passes `--sigma`, scaling the LoRA α parameters (`1.0` = trained ESM-MSR model; `0.0` = zero-shot ESM3 control).
  - **`Replace non-canonical residues`** *(checked by default)* — runs PDBFixer replacement (e.g. `MSE -> MET`) prior to inference. Uncheck to pass `--skip_fix_noncanonical`.
  - **`Skip MT LoRA Pass (Additive Approximation)`** *(unchecked by default)* — passes `--skip_reverse`. Runs only the fast cached WT-context pass (~10× faster); outputs `wt_lora_pred` only and omits `mt_lora_pred`, `combined_pred`, and epistasis.

---

### Tab 2: `Visualization`

- **CSV Summary Banner** *(top)*: Displays the loaded CSV filename, active chain (or available chains), number of unique single mutations and positions, number of double mutations (if present), single-mutant score range (`[min, max]` kcal/mol), and any auto-fallback or chain-mismatch notice.
- **`Target Configuration (Synced with Screening)`**: **`Target Model:`**, **`Refresh`**, and **`Chain:`** (kept in sync with Tab 1).
- **`Display Mode and Priority`**:
  - **`Mode:`**
    - `Singles`: Builds one clone (`*_singles_viz`) with the highest-scoring passing single mutant (`swapaa`) at each position.
    - `WT epistasis`: Filters double mutants to Alanine/Glycine truncations (`X -> A` for non-Ala WT; `A -> G` for Ala WT), maps additive truncation scores onto native WT sidechains (disabling the `WT:` ghost styling row while honoring the `Mutant:` styling row), and renders epistasis pseudobonds.
    - `MT epistasis`: Packs selected double-mutant pairs into minimal non-conflicting clone layers (`*_epi_layer_0`, `*_epi_layer_1`, ...) so overlapping mutations at the same residue do not collide, and connects interacting sidechains with epistasis pseudobonds.
    - *Both epistasis modes are automatically disabled if the loaded CSV lacks double-mutant rows or epistasis columns.*
  - **`Select Pairs By:`** *(epistasis modes)*: `Epistasis` (filter and rank pairs by the chosen epistasis metric) or `Stability` (filter and rank pairs by total double-mutant stability `combined_pred`).
  - **`Display Priority:`** *(epistasis modes)*: `High score`, `Low score`, or `Magnitude` (`|score|`), controlling which pairs are kept up to **`Max Interactions / Pos:`**.
  - **`Epistasis Metric:`** *(epistasis modes)*: `epi_pred (1× ensemble)` *(default when present)* or `combined_dddg_pred`. Unavailable metrics are disabled automatically.
- **`Global Filters`**:
  - **`No WT Cys`**, **`No WT Pro`**, **`No Mut Cys`**, **`No Mut Pro`** — exclude mutations with wild-type or mutant Cysteine (`C`) or Proline (`P`). Checking `No Mut Cys` helps filter out false positives associated with protease-assay disulfide artifacts.
- **`Score Selection (Additive Base)`**:
  - **`Dual-view predictions (recommended; required for epistasis)`** (`combined_pred`)
  - **`WT LoRA predictions (additive)`** (`wt_lora_pred`)
  - **`MT LoRA predictions (not recommended)`** (`mt_lora_pred`)
  - When a CSV generated with `--skip_reverse` is loaded, `combined_pred` and `mt_lora_pred` are automatically disabled and `WT LoRA predictions (additive)` is auto-selected.
- **`Global Thresholds and Networks (kcal/mol, positive = stabilizing)`**:
  - **`Pos Threshold:`** (`0.0–1000.0`, default `0.0`) & **`Neg Threshold:`** (`-1000.0–0.0`, default `0.0`) — displays mutations/pairs satisfying `score >= Pos Threshold` **or** `score <= Neg Threshold`. (To view only stabilizing variants, set `Neg Threshold` to a large negative number such as `-10.0`; to view only destabilizing variants, set `Pos Threshold` to `10.0`.)
  - **`Non-Target Chain Transp %:`** (`0–100`, default `90`) & **`Hide Fully`** — dims or hides chains other than `Chain:` in the active model (unrelated open models are hidden automatically).
  - **`Max Interactions / Pos:`** (`1–20`, default `1`, epistasis modes only) — caps the number of epistasis pseudobonds incident on any single residue position to avoid visual clutter; undirected residue pairs are deduplicated.
  - **`Color Backbone by Highest Additive ΔΔG`** *(enabled in `Singles` and `WT epistasis`)* — colors the target chain ribbon by each residue's best single-mutant ΔΔG using `color byattribute residue_score_viz_score`.
  - **`Nearby atoms within:`** + spinbox (`0.0–50.0` Å, default `3.0` Å) + **`Exclude backbone`** *(checked by default)* — highlights wild-type contacting atoms and residues within the specified radius of displayed mutant sidechains (automatically including `CA` for Glycine mutants) without modifying your ChimeraX selection.
- **`Rendering and Styling`**:
  Six-column table (`Element`, `Color (name/#hex)`, color-picker swatch button, `Style`, `Transp %`, `Hide`) for four structural elements:

  | Element | Default Color | Default Style | Default Transp % | Default Hide | Notes |
  |---|---|---|---|---|---|
  | **`WT:`** | `white` | `stick` | `100` | Checked | Native "ghost" WT sidechains for geometric comparison (disabled in `WT epistasis`). |
  | **`Mutant:`** | *(blank = score)* | `stick` | `0` | Unchecked | Leave blank to color carbons by additive ΔΔG score (gradient swatch); enter a color name / `#hex` or click the swatch button to override. |
  | **`Near Atom:`** | `purple` | `ball` | `0` | — | Contacting WT atoms within the distance cutoff (enabled when `Nearby atoms within:` is checked). |
  | **`Near Res:`** | `grey` | `stick` | `80` | Unchecked | Full residues containing contacting WT atoms (enabled when `Nearby atoms within:` is checked). |

  Non-carbon atoms (`~C`) are automatically colored by element (`byelement`).

#### Color Mapping Guide
- **Additive ΔΔG (Mutant Sidechains & Backbone Ribbon)**:
  - **Green (`> 0`)**: Stabilizing mutation (favorable ΔΔG).
  - **White (`0`)**: Neutral.
  - **Pink (`#FF1493`, `< 0`)**: Destabilizing mutation (unfavorable ΔΔG).
- **Epistasis Pseudobonds (`"ESM-MSR Epistasis"` group)**:
  - **Blue (`> 0`)**: Positive / synergistic epistasis.
  - **Orange (`< 0`)**: Negative / antagonistic epistasis.
  - Pseudobond radius (`0.05–0.40` Å) and opacity scale continuously with score magnitude above the threshold, accompanied by a ChimeraX `key` colorbar in the lower-left corner.

---

### Global Action Bar & Status Footer

- **Row 1**:
  - **`Run Prediction Script`** — validates all inputs before creating temporary files, exports the active target model if needed, and starts `src/esm_msr/inference.py`. Switches to a red **`STOP`** button while running.
  - **`Load CSV`** — opens a file dialog (defaulting to the last completed output directory when available) to load and visualize any ESM-MSR prediction CSV.
  - **`Load Last Output`** — one-click shortcut (enabled whenever a finished run output or the configured `Output CSV` exists on disk) that switches to Tab 2 and loads the file immediately.
- **Row 2**:
  - **`Update Visualization`** — re-parses the currently loaded CSV and refreshes the 3D scene using the current Tab 2 settings and selected `Chain`.
  - **`Reset / Clear Viz`** — closes all ESM-MSR generated mutant clones, deletes only the `"ESM-MSR Epistasis"` pseudobond group, and resets the target model to a clean white ribbon without disturbing other open models or user pseudobonds.
- **Footer**:
  - Compact **`QProgressBar`** (indeterminate during model loading, determinate `0–100%` on `tqdm` output).
  - Color-coded **`Status:`** label (`info`, green `success`, amber `warning`, red `error`).
  - **`Predicted output file:`** label showing the most recent run output filename.

---

## CSV Input Schema & Scientific Conventions

### Sign Conventions
- **Stability (ΔΔG, kcal/mol)**: **Positive (`> 0`) = stabilizing**, **Negative (`< 0`) = destabilizing**.
- **Epistasis (ΔΔΔG, kcal/mol)**: **Positive (`> 0`) = synergistic**, **Negative (`< 0`) = antagonistic**.

### Expected Columns

| Column | Required? | Description |
|---|---|---|
| `chain` | **Required** | PDB chain ID (filtered against the selected `Chain:` in the GUI). |
| `mut_type_pdb` | **Required** | Colon-separated PDB-numbered mutation string(s). |
| `combined_pred` | One of the 3 score cols required | Dual-view WT + MT ensemble ΔΔG prediction (recommended; required for epistasis modes). |
| `wt_lora_pred` | One of the 3 score cols required | WT-context LoRA ΔΔG prediction (always present, including `--skip_reverse` outputs). |
| `mt_lora_pred` | One of the 3 score cols required | MT-context LoRA ΔΔG prediction (absent when `--skip_reverse` is used). |
| `epi_pred` | Required for epistasis (preferred) | Authoritative 1× ensemble epistasis: $\text{comb}(AB) - \text{WT}(AB) = \lambda\,(\text{MT}(AB) - \text{WT}(AB))$ (default $\lambda = 0.5$; `0.0` on singles). |
| `combined_dddg_pred` | Alternate epistasis col | Additive-subtracted epistasis: $\text{comb}(AB) - (\text{comb}(A) + \text{comb}(B))$ (equals `epi_pred` if run with `--skip_additive`). |
| `pdb_file`, `code`, `mut_type_renumbered`, `dist`, `*_additive`, `mt_lora_dddg_pred` | Optional | Standard metadata and auxiliary columns written by `src/esm_msr/inference.py`. |

### Parsing & Validation Rules
- **Single-mutant requirement**: Single-mutant rows (`k = 1`) are required in **all** visualization modes (`Singles`, `WT epistasis`, and `MT epistasis`) so individual mutated sidechains can be colored by their single-mutant additive ΔΔG. When running epistasis scans, always use `--mode singles+doubles`.
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
In v0.4.0, the plugin runs `inference.py` inside a fresh writable temporary `work/` directory (cleaned up automatically when the run finishes, is stopped, or the tool closes) while resolving all relative user paths against `Base Repo Dir`. Thus ChimeraX's launch directory is never polluted and read-only launch directories never cause `temp_fixed.pdb` write errors. If you invoke `src/esm_msr/inference.py` manually from the command line, run it from a writable directory or pass `--skip_fix_noncanonical`.

### 2. Loading a CSV created with `--skip_reverse`
When `--skip_reverse` (`Skip MT LoRA Pass`) is enabled, `inference.py` omits all-NaN columns (`mt_lora_pred` and `combined_pred`). The visualizer automatically detects this on load, selects `WT LoRA predictions (additive)` (`wt_lora_pred`), displays a notice in the top summary banner, and disables `Dual-view`, `MT LoRA`, and epistasis display modes.

### 3. "Selected chain 'X' has no matching rows in CSV"
The CSV summary banner lists the chain ID(s) present in the file and retains the loaded CSV path. Switch **`Chain:`** under **Target Configuration** to one of the available chains and click **`Update Visualization`** — no need to browse for the CSV again.

### 4. CUDA Out-of-Memory (OOM) errors during inference
- Switch **`Model Precision (dtype):`** from `float32` to **`bfloat16`** (Ampere+ GPUs) or **`float16`**.
- Lower **`Batch Size:`** (e.g., from `16` to `4` or `1`).
- For `singles+doubles` screens, enable **`Filter doubles by WT heavy-atom distance (Å):`** (e.g. `6.0` Å) or target specific residues with **`Grab Selection`**.
- For single-mutant screens on large proteins, enable **`Skip MT LoRA Pass (Additive Approximation)`** (runs a single cached WT forward at batch size 1).

### 5. `Python Env` resolution fails / `conda` not found on PATH
GUI launches of ChimeraX on macOS, Linux desktops, or Windows sometimes do not inherit your shell's `PATH` (so `conda` cannot be found when entering a bare conda environment name like `msr_venv`). Point **`Python Env:`** directly to the environment folder (e.g. `/home/user/miniconda3/envs/msr_venv` or `C:\Users\user\miniconda3\envs\msr_venv`) or directly to its `python` / `python.exe` executable.

### 6. Will visualization overwrite my other ChimeraX pseudobonds or models?
No. In v0.4.0:
- Epistasis edges are stored in a dedicated `"ESM-MSR Epistasis"` `PseudobondGroup`; updating or resetting deletes **only** that group and leaves hydrogen bonds, metal coordination bonds, and distance monitors untouched.
- Hydrogen hiding (`hide ... & H`) applies only to the target model and ESM-MSR mutant clones.
- Generated mutant clones (`*_singles_viz`, `*_epi_layer_*`) are tracked internally and filtered out of the `Target Model` dropdowns. Click **`Reset / Clear Viz`** to close them and restore the target structure.

### 7. Running zero-shot ESM3 predictions (`--sigma 0.0`)
Keep a valid checkpoint selected (such as `LoRA_models/esm-msr-small/lora_seed1.safetensors`) and set **`Sigma (σ):`** to `0.0`. Alternatively, leave **`Checkpoint:`** empty, specify `LoRA_models/esm-msr-small/hparams.yaml` in **`LoRA Config:`**, and set **`Sigma (σ):`** to `0.0`.

### 8. Where can I inspect the exact inference command or traceback?
Open the ChimeraX Log (`Tools -> General -> Log`). Every run logs its working directory, full shell-quoted command line, and all non-progress stdout/stderr lines from `inference.py`.

---

## Development & Automated Headless Testing

### Bundle Layout

```
ChimeraX-ESM_MSR/
├── bundle_info.xml          # ChimeraX bundle manifest (ChimeraX-ESM-MSR v0.4.0)
├── README.md                # This documentation
├── src/
│   ├── __init__.py          # BundleAPI entrypoint (get_class, start_tool)
│   ├── parsing.py           # Pure-Python/pandas CSV, mutation, path & env parsing
│   └── tool.py              # PyQt6 GUI, QProcess runner & ChimeraX visualization
└── tests/                   # Automated unit, offscreen PyQt6 & ChimeraX tests
```

### Building & Verifying the Wheel

```bash
chimerax --nogui --cmd "devel build /path/to/esm-msr/ChimeraX-ESM_MSR; exit"
```

### Running Automated Headless Tests

The automated test suite under `ChimeraX-ESM_MSR/tests/` covers pure `parsing.py` unit tests, headless PyQt6 widget and subprocess-argv tests, and real ChimeraX `--offscreen` 3D rendering and integration tests (including negative residue numbers, insertion codes, Glycine contacts, dedicated `"ESM-MSR Epistasis"` pseudobonds, and `Reset / Clear Viz`).

Run the complete test suite (unit + headless PyQt6 + ChimeraX `--offscreen` integration) with ChimeraX's bundled Python (unset `DISPLAY` and `WAYLAND_DISPLAY` on Linux/WSL so Qt runs purely offscreen):

```bash
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \
  /usr/lib/ucsf-chimerax/bin/python3.11 -m pytest ChimeraX-ESM_MSR/tests/test_chimerax_esm_msr.py -v
```

Or run the ChimeraX-session integration tests directly inside a headless ChimeraX session:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \
  chimerax --offscreen --exit --script ChimeraX-ESM_MSR/tests/test_chimerax_esm_msr.py
```

*(Adjust the ChimeraX binary and bundled Python paths for your OS installation as needed.)*
