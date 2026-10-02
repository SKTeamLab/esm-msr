# ESM-MSR

This mutant stability predictor was created by parameter-efficient fine-tuning of ESM3-small-open (https://www.science.org/doi/10.1126/science.ads0018) on protease susceptibility assays from Tsuboyama et al. (https://www.nature.com/articles/s41586-023-06328-6). It generates state-of-the-art predictions on numerous benchmark datasets including S461 and PTMUL-D. This repository is designed to enable fast inference using this approach and facilitate reproducing results from our paper, currently in pre-print. We also created an interface for inference and visualization in ChimeraX. You can read the accompanying preprint here: https://www.biorxiv.org/content/10.64898/2026.06.04.730231v1.

![Alt text](_assets/diagram_epistasis.png)

## Requirements

Python 3.11-3.13 [Download Python](https://www.python.org/downloads/windows/)

CUDA 12.8 if using GPU-accelerated inference

NVIDIA GPU with 24+ GB VRAM for training, 8GB is likely sufficient for low batch size inference. Inference can also be done on a CPU (very slowly).

ChimeraX if intending to use the graphical user inference (GUI) and visualization tool: [Download ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html).

Tested extensively on Python 3.12, CUDA 12.8, ChimeraX 1.10.

Installation time: 10 minutes to clone repository and setup virtual environment. An additional 10 minutes is required to install ChimeraX and the ESM-MSR plugin. Downloading additional LoRAs from HuggingFace can be done concurrently.

## Demo Information

All steps below are required to complete the demo, except "Basic Usage - Command Line Interface (Skip if using ChimeraX GUI)". The end of the README, starting from "Using the ChimeraX GUI", is the demo; make sure to load the 1UFM structure if trying to check demo files (`open 1ufm`). The expected output is visually indicated at the end of the file (equivalent to Figure 6 in the manuscript). You can also compare your output CSV to the one saved at `data/example/1ufm_example.csv`.

## Recommended Installation

## Windows Set-up from Zero:

1. Download and install Python version 3.11-3.13: [Download Python](https://www.python.org/downloads/windows/). Be sure to check the "Add python.exe to PATH" box before clicking "Install Now".

2. Download the .zip version of this repository from the top left of this GitHub page (or clone it if you have installed Git for Windows).

3. Extract the repository where you would like the program to be installed.

4. Inside the esm-msr folder in File Explorer (you should see pyproject.toml), click on the address bar and type "cmd" to open a command prompt in the repository location.

5. Enter the following commands (you can cut and paste):

```
python -m venv msr_venv python=3.12
.\Scripts\activate.bat
pip install torch
pip install -e .
```

Your Python environment is now setup, but you still need to obtain the ESM3 base model and install the tool in ChimeraX to use the GUI. See below.

## Linux Command Line Setup with CUDA GPU acceleration:

Clone the repo, create a conda environment, and install in editable mode:

```
git clone https://github.com/SKTeamLab/esm-msr.git
cd esm-msr
conda create -n msr_venv python=3.12
conda activate msr_venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Obtaining ESM3-small-open weights

Weights are available from [HuggingFace](https://huggingface.co/biohub/esm3-sm-open-v1).

From here you have two options:

1. Allow the environment package `esm` to handle the model download and possible updates. No action is required unless ESM/Biohub changes this access mechanism.
2. Alternatively: download all files in the `data/weights` HuggingFace repo to your machine. If you have git installed, the easiest way is to use `git clone https://huggingface.co/biohub/esm3-sm-open-v1`. You will need to enter this location into the CLI or GUI.

## Basic Usage - Command Line Interface (Skip if using ChimeraX GUI)

You should now have everything you need to make your first predictions from the command line, except for a pdb/mmcif structure of interest, which you should download if following this step.

We include a small version of our LoRA model in this repository for convenience, which is used in the example below. If you want to use other models, please see our [HuggingFace page](https://huggingface.co/sareeves96/esm-msr), download the models to the LoRA_models folder, and change the path specified in the command.

Inference strategies, performance and compute time are discussed in the paper. The below command is a fast approximation of single mutant saturation mutagenesis that should take less than a minute even on a CPU, apart from loading the model weights which is very hardware dependent.

`python src/esm_msr/inference.py --checkpoint_path LoRA_models/esm-msr-small/epoch\=03-val_rho_combined_avg\=0.816.ckpt --pdb_file path_to_your_structure_file --mode singles --skip_reverse --output_csv ./example_output.csv`

Or, if you want to directly use a PDB structure:

`python src/esm_msr/inference.py --checkpoint_path LoRA_models/esm-msr-small/epoch\=03-val_rho_combined_avg\=0.816.ckpt --code 1A0F --chain A --mode singles --skip_reverse --output_csv ./example_1A0F.csv`

Remove the `--skip_reverse` flag for a much slower, slightly higher accuracy screen (proportional to total possible mutations). Change the screening `--mode` to `singles+doubles` to screen all double and single mutants (the singles are comparatively almost free and useful for visualization later). It is not recommended to use `--skip_reverse` for `--mode doubles` because this ignores epistasis. A full double mutant screen on a protein greater than 200 residues is very compute expensive. It is therefore recommended to screen only double mutants where the wild-type residues are within 6 Angstrom heavy atom distance. This is controlled with the `--distance_threshold` parameter. Multi-mutants must be screened individually, either by generating an `--input_csv` with columns `pdb_file, code, chain, mut_type` and mutations (`mut_type`) specified like A2C:D3E, or individually passing in comma separated mutations via `--mutations`. An example can be seen in `data/preprocessed/ptmul_mapped.csv`; extra columns are allowed.

The visualizer generates predictions using this script. You can read the GUI section to understand how `inference.py` can be used from the command line.

## Reproducing Benchmarks

If you want to reproduce the benchmarks without running preprocessing, you can download the relevant data from Zenodo (https://doi.org/10.5281/zenodo.21539277; `preprocessed_data.tar.gz`), place the contents in the data folder, and then run this command:

`python inference_scripts/esm_msr_testing.py --checkpoint esm-msr-small/epoch=03-val_rho_combined_avg=0.816.ckpt --split hyperopt_splits --local_path_to_structures path_to_structures_from_zenodo`

Note that the model selected here is included in the repo due to its small size and will have very similar performance to the one used in the paper, but the exact model(s) must be downloaded from HuggingFace and the `--checkpoint` argument must be updated accordingly. Benchmarking will take at least an hour even on a powerful GPU.

## ProteinGym Benchmarking

ESM-MSR can be scored against the full [ProteinGym](https://proteingym.org/) deep mutational scanning (DMS) benchmark (v1.3 release: 217 DMS, over 2.4 million mutants, [Zenodo 15293562](https://zenodo.org/records/15293562)). Two scripts are involved: `preprocessing/pgym_preprocess.py` (one-time setup per machine) and the `--protein_gym` mode of `inference_scripts/esm_msr_testing.py` (the actual scoring, which is resumable).

### 1. Download the official ProteinGym data (~62 MB)

`wget https://zenodo.org/records/15293562/files/DMS_substitutions.csv`
`wget https://zenodo.org/records/15293562/files/DMS_ProteinGym_substitutions.zip`
`wget https://zenodo.org/records/15293562/files/ProteinGym_AF2_structures.zip`

`mkdir -p ProteinGym && mv DMS_substitutions.csv ProteinGym/`
`unzip DMS_ProteinGym_substitutions.zip -d ProteinGym`
`unzip ProteinGym_AF2_structures.zip -d ProteinGym`

Each zip extracts to a folder of the same name, so you should end up with 217 per-DMS CSVs under `ProteinGym/DMS_ProteinGym_substitutions/` and 199 AlphaFold2 structures under `ProteinGym/ProteinGym_AF2_structures/` (one PDB per protein, shared by the DMS on that protein).

### 2. Preprocess (one-time, a few minutes)

`python preprocessing/pgym_preprocess.py --proteingym_dir ProteinGym --out pgym_inputs`

This aligns each DMS's mutations to the corresponding AlphaFold2 structure (re-numbering positions when the DMS and structure use different numbering schemes) and writes one ready-to-score CSV per DMS plus a `manifest.csv` into `pgym_inputs/`. On the official release you should see `Runnable DMS: 217`, `Total mapped: 2465767`, `Skipped DMS: 0`, and `unmapped total: 0`. A nonzero `wt_mismatch` count (57 on the official release) is expected and harmless. Note that `pgym_inputs/` is machine-specific: the manifest records the absolute path to your `ProteinGym` folder, so if you move the data (or copy `pgym_inputs` to another machine), just re-run the preprocessor.

### 3. Run the benchmark (resumable, ~1 day on a 32 GB GPU)

`python inference_scripts/esm_msr_testing.py --checkpoint esm-msr/lora_seed1.safetensors --protein_gym --pgym_dir pgym_inputs --pgym_out pgym_results --auto_batch_size --dtype bf16 --lora_epsilon 1.0`

* `--checkpoint` is relative to the `LoRA_models/` folder. The full model above is not included in the repo (only the small demo model is); download it once from our [HuggingFace page](https://huggingface.co/sareeves96/esm-msr): `huggingface-cli download sareeves96/esm-msr --local-dir LoRA_models/esm-msr`.
* `--auto_batch_size` sizes each batch from measured GPU memory, so no manual tuning is needed. If a DMS simply does not fit in GPU memory even at batch size 1, it is logged as a failure and the run continues with the next DMS (on WSL2, very large DMS may instead spill into system RAM and complete very slowly — see [docs/known_issues.md](docs/known_issues.md)).
* `--dtype bf16` is required for accuracy and speed on modern GPUs.
* `--lora_epsilon 1.0` runs the trained ESM-MSR model; set it to `0.0` to score with the untrained (zero-shot) ESM3 backbone as a control.
* `--skip_reverse` scores with the WT pass only (the additive approximation): each DMS runs a single cached WT-context forward, and every mutant is then scored by gathering the per-position local-substitution log-likelihood ratios from it. No mutant-context forward is run, so `mt_lora_pred`/`combined_pred` are NaN (omitted from the CSVs) and the per-DMS correlation is computed on `wt_lora_pred` instead. It is ~10× faster than the full two-pass mode (~17 min vs ~1 day for all 217 DMS on a 32 GB card), and the batch size no longer affects VRAM, since the one forward is always run at batch size 1.
* The run is resumable: any DMS that already has an output CSV in `--pgym_out` is skipped, so you can interrupt and restart freely. Use `--pgym_dms DMS1,DMS2` to score only a subset (defaults to all).
* Outputs: one CSV per DMS with the predictions (`combined_pred`) alongside the experimental `DMS_score`, plus a `summary.csv` with the per-DMS Spearman correlation (`spearman_combined`), timing, and status. A DMS that crashes mid-run is recorded in `summary.csv` with `status=fail` and an error message; fix the cause (usually memory) and re-run — only that DMS will be re-attempted.

### Reference run and expected results

Our reference runs score **all 217 of 217 DMS** on a single RTX 5090 (32 GB): the full model at σ=1.0 for all three training seeds (`esm-msr/lora_seed1.safetensors`, `lora_seed2.safetensors`, `lora_seed3.safetensors`) — each also scored in WT-only mode (`--skip_reverse`, the additive approximation, scored on `wt_lora_pred`) — and the small demo model (`esm-msr-small/epoch=03-...ckpt`) at σ=1.0, 0.5, and 0.0 (zero-shot ESM3 control), and the legacy chain model (`esm-msr-chain/epoch=04-...ckpt`) at σ=1.0:

| Model | σ | mean Spearman | median | DMS positive | best DMS (ρ) | worst DMS (ρ) |
|---|---|---|---|---|---|---|
| esm-msr seed1 | 1.0 | **0.555** | 0.497 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.972) | `TADBP_HUMAN_Bolognesi_2019` (−0.259) |
| esm-msr seed2 | 1.0 | 0.560 | 0.505 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.979) | `TADBP_HUMAN_Bolognesi_2019` (−0.254) |
| esm-msr seed3 | 1.0 | 0.554 | 0.501 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.970) | `TADBP_HUMAN_Bolognesi_2019` (−0.239) |
| esm-msr seed1, WT-only | 1.0 | 0.550 | 0.509 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.950) | `TADBP_HUMAN_Bolognesi_2019` (−0.276) |
| esm-msr seed2, WT-only | 1.0 | 0.553 | 0.507 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.958) | `TADBP_HUMAN_Bolognesi_2019` (−0.244) |
| esm-msr seed3, WT-only | 1.0 | 0.550 | 0.514 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.953) | `TADBP_HUMAN_Bolognesi_2019` (−0.244) |
| esm-msr-small | 1.0 | 0.552 | 0.496 | 214 / 3 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.971) | `TADBP_HUMAN_Bolognesi_2019` (−0.262) |
| esm-msr-small | 0.5 | 0.542 | 0.517 | 216 / 1 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.923) | `TADBP_HUMAN_Bolognesi_2019` (−0.031) |
| esm-msr-small | 0.0 | 0.457 | 0.465 | 216 / 1 | `PR40A_HUMAN_Tsuboyama_2023_1UZC` (0.849) | `SYUA_HUMAN_Newberry_2020` (−0.014) |
| esm-msr-chain | 1.0 | 0.556 | 0.518 | 215 / 2 | `NUSA_ECOLI_Tsuboyama_2023_1WCL` (0.974) | `B2L11_HUMAN_Dutta_2010_binding-Mcl-1` (−0.125) |

Four of the 217 are size extremes that need special handling on a 32 GB card: `BRCA2_HUMAN_Erwood_2022_HEK293T` (2,832 residues) saturates the GPU at batch size 1 and completes via CPU spill into system RAM (~13 min; needs ~54 GB of free RAM and PyTorch's default allocator), `SCN5A_HUMAN_Glazer_2019` (2,016 residues), `POLG_CXB3N_Mattenberger_2021` (2,185) and `BRCA1_HUMAN_Findlay_2018` (1,863) also need the default allocator (they fit on-card). If any of them fails on a first attempt, retry just that DMS with `PYTORCH_CUDA_ALLOC_CONF=` (empty) set in the environment — see [docs/known_issues.md](docs/known_issues.md). [docs/vram_and_batch_sizes.md](docs/vram_and_batch_sizes.md) tabulates the maximum structure length that fits at each batch size (e.g. ~1,740 residues at batch 1, ~500 at batch 12, ~215 at batch 64 on a 32 GB card in bf16), measured throughput, and the CPU-spill behavior for larger structures.

## Adding the Visualizer to ChimeraX

*Note: if using Windows Subsystem for Linux (WSL), it is recommended to install ChimeraX on Windows, not WSL. Everything should work even if you installed ESM-MSR into WSL.*

1. Download, install, and open [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html) (free for non-commercial use):
1. Go to Tools -> Command Line Interface (check box if not checked)
2. In the command line interface at the bottom, type (replacing the `/path/to/repo`): `devel install /path/to/repo/ChimeraX-ESM_MSR`

## Using the ChimeraX GUI

Load a valid protein structure (PDB, mmCIF) into ChimeraX. You can click and drag structure files into the window, or directly open a PDB structure (e.g. for the demo, use `open 1ufm`) via the ChimeraX command line. Open the GUI, located in the Tools tab under Stability->ESM_MSR. The GUI workflow is split into three main tabs: **Execution / IO**, **Screening Config**, and **Visualization**. The tool automatically remembers your most recent paths and configuration settings between sessions. You must fill out the first two boxes before hitting **"Run Prediction Script"**, and you must have a valid output (esp. by running the script) before you can complete the third tab and visualize the predictions with the **"Load CSV + Visualize Scores"** button. To complete the demo, you can leave all settings at their default values in the Execution I/O tab.

### 1. Execution & IO (Environment & Models)

**Environment & Paths:**
* **Base Repo Dir:** Browse to the root of your cloned `esm-msr` folder. *Note: Setting this automatically populates the Python Env and Output CSV fields if they are currently empty. If you followed the instructions, the paths should be correct.*
* **Python Env:** The environment used to run inference. *WARNING: if you used a conda environment, replace the path with just the name e.g. `msr_venv`.*
* **Output CSV:** Where the resulting predictions will be saved.
* **ESM3 Weights Location:** If you want to use locally stored weights, enter the location here.

**Compute Environment & Model Configuration Files:**
* **Compute Device & Batch Size:** Select your hardware (`cuda`, `mps`, `cpu`) and batch size. Lower the batch size if you encounter CUDA Out-Of-Memory (OOM) errors. Use `cpu` unless you configured CUDA during setup and have an Nvidia GPU.
* **Checkpoint (.ckpt/.safetensors):** Select the trained LoRA checkpoint, for example the one stored in LoRA_models/esm-msr-small in this repo, or any of the checkpoints available from our HuggingFace page.
* **LoRA Config (JSON/YAML):**  When you use the "Browse" button to select a Checkpoint, the GUI will automatically assume the configuration file is named `hparams.yaml` and is located in the same parent directory. It will warn you in red text if either file is missing or contains a dangling path reference. Architectural parameters (adapter mode, lora mode, rank, alpha) are automatically parsed from this file during inference.

### 2. Screening Config

This section defines exactly which mutations will be evaluated on your structure.

**Target Selection:**
Select which open ChimeraX model and specific chain you want to predict on. For NMR ensembles, changing the target model allows you to chose a different structural model from the ensemble.

**Mutation Scope (Mutually Exclusive):**
Select **one** of three methods to define the mutation space. Selecting one method will automatically disable the inputs for the others.
1. **1. Full Screen:** Exhaustively scores mutations. Choose `singles`, `singles+doubles`.
   * *Positions:* Leave empty for all residues, or type indices manually (e.g., `11,12`). You can also select residues in ChimeraX (ctrl + click + drag) and click **Grab Selection**.
   * *Filter doubles by distance (Å):* If you are screening `singles+doubles`, you can check this box to strictly evaluate pairs of residues that are within a certain 3D spatial proximity based on minimum side-chain heavy atom distance.
2. **2. Specify mutations in CSV:** Upload a predefined CSV list of mutations to score.
3. **3. Input Mutations Directly:** Manually type a comma-separated list of precise mutations (e.g., `A12C,A12C:D15E`).

**Screening Parameters:**
* **Mask Strategy:** Choose between `Default (unmasked)`, `marginal`, or `independent`. `unmasked` tends to perform best, but there are compute savings especially if only specifying a few positions using `independent`. `marginal` is not recommended.
* **Skip MT pass (Use Additive Approximation):** Fast approximation, especially suitable for single mutants. *Warning: Skips generating `mt_lora` predictions.*

**Running Inference:**
Click **Run Prediction Script** at the bottom of the window. A red **STOP** button will appear, which allows you to forcefully terminate the process tree if you accidentally launch a massive screening run. After preliminary setup (~1 minute or possibly much longer if downloading ESM3 weights for the first time), you can track the screening progress at the bottom of the GUI, stop, and modify the run parameters if it will take too long. *Note: Singles screening or using the additive approximation takes <1 minute on modern GPUs. Doubles screening on a 300AA protein can take hours (`independent` masking) or days (`unmasked`).*

### 3. Visualization

Once inference completes (or if you load an existing output CSV), navigate to the **Visualization** tab to map the stability and epistatic scores onto your 3D structure.

**Note on Requirements:** *Single-mutation data is required for all visualization modes* to properly map additive stability scores onto sidechains.

**Target Selection:**

Confirm that you will apply the visualization to the correct chain entity. If you are visualizing a newly created result, this should be auto-populated to the correct value.

#### Core Configuration
The central paradigm is to visualize either singles, doubles, or interactions that are outside of the thresholds defined in the next section.
* **Display Mode:** Choose the primary visualization strategy:
  * **Singles:** Visualizes independent single mutations.
  * **WT Epistasis:** Visualizes epistatic interactions between wild-type residues by asssessing truncation to Alanine (or Glycine). Mapped directly onto the native WT geometry; residue colors indicate effects of mutation to alanine.
  * **MT Epistasis:** Visualizes epistatic interactions between mutant pairs, utilizing dynamically generated structural layers to resolve overlapping geometry.
* **Select Pairs By:** Determines the metric used to filter and sort interactions (either the raw Epistasis ΔΔΔG score, or the total Double Mutant Stability score).
* **Display Priority:** Determines which interactions to keep when the "Max Interactions" cap is hit. You can prioritize by High score (highest predicted stability change), Low score, or Magnitude (absolute value).
* **Global Filters:** Quickly exclude specific mutations from the visualization to clean up the display. Note that No MT Cys is especially useful to mitigate false positives caused by assay bias.
* **Score Selection (Additive Base):** Select which raw additive score to use as the base metric (Dual-view, WT LoRA, or MT LoRA). *Note: Dual-view is required for epistasis. If you skipped the MT pass during inference, only WT LoRA predictions will be available.*

#### Global Thresholds and Networks
* **Pos/Neg Thresholds:** Only mutations or epistatic pairs with scores strictly greater than the Positive Threshold or less than the Negative Threshold are visualized. Setting the negative threshold to -10 will effectively filter out al destabilizing mutants.
* **Non-Target Chain Transp %:** Adjusts the opacity of opposing chains in the complex to reduce visual clutter.
* **Max Interactions per Position:** (Epistasis modes only). Caps the number of epistatic network edges that can originate from a single residue to prevent visual overload (the "hairball" effect).
* **Visualize Contacts:** Shows surrounding wild-type contextual residues within a specified Angstrom radius of the visualized mutant sidechains. 
  * Contacting atoms are explicitly highlighted based on your styling preferences.
* **Color Backbone by Highest Additive ΔΔG:** (Singles & WT Epistasis mode only). Colors the wild-type ribbon backbone on a Pink-to-White-to-Green gradient based on the highest-scoring candidate at each position.

#### Rendering and Styling
Customize the color, geometry style (stick, ball, sphere, wire), and transparency of the structural components.
* **WT Style:** Applies to the "ghost" wild-type residues left behind for structural context, so you can assess "is the mutation really a better/worse fit?".
* **Mut Color / Style:** Applies to the mutated sidechains. **LEAVE BLANK FOR ADDITIVE SCORE** to automatically color the mutant carbons based on their individual stability score (Pink-to-White-to-Green gradient).
* **Contact Style:** Applies explicitly to the surrounding context residues.

#### Color Mapping Guide
When using the default styling (leaving the Mut Color blank), the visualizer generates dynamic colorbars mapped to your data:
* 🟩 **Green (Atoms/Backbone):** Favorable additive single-mutant stability (score > 0).
* 🟥 **Pink (Atoms/Backbone):** Unfavorable additive single-mutant stability (score < 0).
* 🟦 **Blue (Pseudobonds):** Positive epistasis / synergistic interaction (score > 0).
* 🟧 **Orange (Pseudobonds):** Negative epistasis / antagonistic interaction (score < 0).

### Expected Output

An example is shown below after loading the CSV under the indicated settings for the structure `1UFM`, corresponding to a default single mutant screen. The expected runtime is 1 minute for inference on an NVIDIA GPU at a batch size of 16 or 5 minutes on a CPU in addition to up to two minutes to load visual elements in ChimeraX. Note that because the example uses a smaller model and crystal rather than predicted structure, the selected mutations are slightly different than those from the paper.

![Alt text](_assets/tool_github.png)
