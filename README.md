# ESM-MSR

This mutant stability predictor was created by parameter-efficient fine-tuning of ESM3-small-open (https://www.science.org/doi/10.1126/science.ads0018) on protease susceptibility assays from Tsuboyama et al. (https://www.nature.com/articles/s41586-023-06328-6). It generates state-of-the-art predictions on numerous benchmark datasets including S461 and PTMUL-D. This repository is designed to enable fast inference using this approach and facilitate reproducing results from our paper, currently in pre-print. We also created an interface for inference and visualization in ChimeraX. **Built with ESM**.

![Alt text](_assets/diagram_epistasis.png)

## Requirements

Python 3.11-3.13 [Download Python](https://www.python.org/downloads/windows/)

CUDA 12.8 if using GPU-accelerated inference

NVIDIA GPU with 24+ GB VRAM for training, 8GB is likely sufficient for low batch size inference

ChimeraX if intending to use the graphical user inference (GUI) and visualization tool: [Download ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html).

## Recommended Installation

## Windows Set-up from Zero:

1. Download and install Python version 3.11-3.13: [Download Python](https://www.python.org/downloads/windows/). Be sure to check the "Add python.exe to PATH" box before clicking "Install Now".

2. Download the .zip version of this repository from the top left of this GitHub page (or clone it if you have installed Git for Windows).

3. Extract the repository where you would like the program to be installed.

4. Inside the esm-msr folder in File Explorer (you should see pyproject.toml), click on the address bar and type "cmd" to open a command prompt in the repository location.

5. Enter the following commands (you can cut and paste):

```
python -m virtualenv msr_venv python=3.12
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

Weights are available from [HuggingFace](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/). You must create a HuggingFace account and agree to the Cambrian Non-Commercial License Agreement in the linked page to get access. Note that the same license applies to our method and can be found in the `LICENSE.md` file. 

From here you have two options:

1. Create an access token by going to the user profile icon (top right), selecting Access Tokens, and creating a new Read token. You will be able to pass this token into our CLI or GUI to obtain the weights automatically. It is recommended that you login once in the command line, then you shouldn't need to paste in the token each time: `huggingface-cli login`. Paste in your username and read token when prompted.
2. Download all files in this folder into a new folder e.g. `esm-msr/data/weights`. You will need to paste this location into the CLI or GUI.

## Basic Usage - Command Line Interface (Skip if using ChimeraX GUI)

You should now have everything you need to make your first predictions from the command line, except for a pdb/mmcif structure of interest, which you should download if following this step.

We include a small version of our LoRA model in this repository for convenience, which is used in the example below. If you want to use other models, please see our HuggingFace page, download the models to the LoRA_models folder, and change the specified location.

Inference strategies, performance and compute time are discussed in the paper. The below command is a fast approximation of single mutant saturation mutagenesis that should take less than a minute even on a CPU, apart from loading the model weights which is very hardware dependent.

`python src/esm_msr/inference.py --checkpoint_path LoRA_models/esm-msr-small/epoch\=03-val_rho_combined_avg\=0.816.ckpt --pdb_file path_to_your_structure_file --mode singles --skip_reverse --output_csv ./example_output.csv`

Remove the `--skip_reverse` flag for a much slower, slightly higher accuracy screen (proportional to total possible mutations). Change the screening `--mode` to `doubles` (or to `both`) to screen all double mutants (or both singles and doubles). It is not recommended to use `--skip_reverse` for `--mode doubles` because this ignores epistasis. A full double mutant screen on a protein greater than 200 residues is very compute expensive. It is therefore recommended to screen only double mutants where the wild-type residues are within 6 Angstrom heavy atom distance. This is controlled with the `--distance_threshold` parameter. Multi-mutants must be screened individually, either by generating an `--input_csv` with columns `pdb_file, code, chain, mut_type` and mutations (`mut_type`) specified like A2C:D3E, or individually passing in comma separated mutations via `--mutations`. An example can be seen in `data/preprocessed/ptmul_mapped.csv`; extra columns are allowed.

The visualizer generates predictions using this script. You can read the GUI section to understand how `inference.py` can be used from the command line.

## Adding the Visualizer to ChimeraX

*Note: if using Windows Subsystem for Linux (WSL), it is recommended to install ChimeraX on Windows, not WSL. Everything should work even if you installed ESM-MSR into WSL.*

1. Download, install, and open [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html) (free for non-commercial use):
1. Go to Tools -> Command Line Interface (check box if not checked)
2. In the command line interface at the bottom, type (replacing the `/path/to/repo`): `devel install /path/to/repo/ChimeraX-ResidueScoreVisualizer`

## Using the ChimeraX GUI

Load a valid protein structure (PDB, mmCIF) into ChimeraX. You can click and drag structure files into the window, or directly open a PDB structure (e.g., `open 1enh`) via the ChimeraX command line. Open the GUI, located in the Tools tab under Visualization (ESM_MSR_Visualizer). The GUI workflow is split into three main tabs: **Execution / IO**, **Screening Config**, and **Visualization**. The tool automatically remembers your most recent paths and configuration settings between sessions. You must fill out the first two tabs before hitting **"Run Prediction Script"**, and you must have a valid output (esp. by running the script) before you can complete the third tab and visualize the predictions with the **"Load CSV + Visualize Scores"** buttom.

### 1. Execution & IO (Environment & Models)

**Environment & Paths:**
* **Base Repo Dir:** Browse to the root of your cloned `esm-msr` folder. *Note: Setting this automatically populates the Python Env and Output CSV fields if they are currently empty. If you followed the instructions, the paths should be correct.*
* **Python Env:** The environment used to run inference. *Note: if you used a conda environment, replace the path with just the name e.g. `msr_venv`.*
* **Output CSV:** Where the resulting predictions will be saved.

**Model Weights Source:**
Choose between downloading weights directly via a **HuggingFace Token** (requires internet) or using a local offline folder via **Base Model Location**, which you should have setup previously. If you logged in via `huggingface-cli`, you can leave both boxes blank.

**Compute Environment & Model Configuration Files:**
* **Compute Device & Batch Size:** Select your hardware (`cuda`, `mps`, `cpu`) and batch size. Lower the batch size if you encounter CUDA Out-Of-Memory (OOM) errors. Use `cpu` unless you configured CUDA during setup and have an Nvidia GPU.
* **Checkpoint (.ckpt):** Select the trained LoRA checkpoint, for example the one stored in LoRA_models/esm-msr-small in this repo.
* **LoRA Config (JSON/YAML):** *Auto-populating.* When you use the "Browse" button to select a Checkpoint, the GUI will automatically assume the configuration file is named `hparams.yaml` and is located in the same parent directory. It will warn you in red text if either file is missing or contains a dangling path reference. Architectural parameters (adapter mode, lora mode, rank, alpha) are automatically parsed from this file during inference.

### 2. Screening Config

This tab defines exactly which mutations will be evaluated on your structure.

**Target Selection:**
Select which open ChimeraX model and specific chain you want to predict on.

**Mutation Scope (Mutually Exclusive):**
Select **one** of three methods to define the mutation space. Selecting one method will automatically disable the inputs for the others.
1. **1. Full Screen:** Exhaustively scores mutations. Choose `singles`, `doubles`, or `both`.
   * *Positions:* Leave empty for all residues, or type indices manually (e.g., `11,12`). You can also select residues in ChimeraX (ctrl + click + drag) and click **Grab Selection**.
   * *Filter doubles by distance (Å):* If you are screening `doubles` or `both`, you can check this box to strictly evaluate pairs of residues that are within a certain 3D spatial proximity based on minimum side-chain heavy atom distance.
2. **2. Specify mutations in CSV:** Upload a predefined CSV list of mutations to score.
3. **3. Input Mutations Directly:** Manually type a comma-separated list of precise mutations (e.g., `A12C,A12C:D15E`).

**Screening Parameters:**
* **Mask Strategy:** Choose between `Default (unmasked)`, `marginal`, or `chain`. `unmasked` tends to perform best, but there are compute savings especially if only specifying a few positions using `chain`. `marginal` is not recommended.
* **Artificial Background Mutation:** Applies a universal baseline mutation before evaluating specific targets (e.g., `A15G`). Generally leave this blank.
* **Execution Approximations:**
    * *Approximate Epistasis (Not Recommended):* Skips rigorous additive sub-calculations for multi-mutants, instead computing their scores as 0.5 * MT_LoRA_prediction - 0.5 * WT_LoRA_prediction.
    * *Skip MT pass (Additive Approximation):* Fast approximation, especially suitable for single mutants. *Warning: Skips generating `mt_lora` predictions.*
* **Protein Complex Mode (Experimental):** Allows scoring within the context of multiple chains vs a `single_chain`. In our limited testing, this does not appear to be more useful for binding prediction than using the single chain of interest.

**Running Inference:**
Click **Run Prediction Script** at the bottom of the window. A red **STOP** button will appear, which allows you to forcefully terminate the process tree if you accidentally launch a massive screening run. *Note: Singles screening takes <1 minute on modern GPUs. Doubles screening on a 300AA protein can take hours (`chain`) or days (`unmasked`).*

### 3. Visualization

Once inference completes (or if you load an existing output CSV), navigate to the **Visualization** tab to map the stability scores onto your 3D structure.

**Score Selection:**
If you ran standard inference, choose between visualizing **WT LoRA**, **MT LoRA**, or the recommended **Dual-view predictions**. *Note: If you checked "Skip MT pass" during inference, only WT LoRA predictions will be available in the CSV.*

**Standard Single-Mutant Mode:**
*(Requires a CSV generated using `singles` or `both` mode. Ensure "Epistasis Mode" is unchecked.)*
* **Score Threshold:** Only mutations with a predicted score above this value (positive = stabilizing) will be visualized.
* **Color Backbone by Highest ΔΔG:** Colors the wild-type backbone on a Red-to-White-to-Green gradient based on the highest-scoring candidate at each position, essentially giving a 'mutability' visualization.
* **Show Highest-Scoring Mutations:** Physically mutates the residue to the highest-scoring candidate and renders it as sticks. The original wild-type sticks remain visible for structural comparison. Adjust transparency settings to your liking.
* **Visualize Contacts:** Shows surrounding contextual residues within a specified Angstrom radius of the highest-scoring mutant sidechains. 
   * The structural context (surrounding non-interacting atoms) is rendered as ghosted thin lines.
   * Specific contacting atoms are highlighted as balls.

**Epistasis Mode (Double Mutants):**
*(Requires a CSV generated using `doubles` mode. Check the "Epistasis Mode" box.)*
* **Epistasis Threshold:** Filters out weak epistatic interactions based on absolute magnitude.
* **What it shows:** Turns the active model transparent and replaces residues involved in significant epistatic interactions with their mutated counterparts. Scaled pseudobonds are drawn between interacting pairs:
    * 🟩 **Green Line:** Positive epistasis (score > 0).
    * 🟥 **Red Line:** Negative epistasis (score < 0).
    * Line thickness and brightness are mathematically scaled based on the magnitude of the predicted interaction.
