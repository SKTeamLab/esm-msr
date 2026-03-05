# ESM-MSR

This mutant stability predictor was created by parameter-efficient fine-tuning of ESM3-small-open (https://www.science.org/doi/10.1126/science.ads0018) on protease susceptibility assays from Tsuboyama et al (https://www.nature.com/articles/s41586-023-06328-6). It generates state-of-the-art predictions on numerous benchmark datasets including PTMUL and Ssym. This repository is designed to enable inference using this approach and facilitate reproducing results from our paper: link. We also created an interface for inference and visualization in ChimeraX. **Built with ESM**.

![Alt text](_assets/diagram_epistasis.png)

## Requirements

Python 3.12
CUDA 12.8
NVIDIA GPU with 24+ GB VRAM

## Recommended Installation

Clone the repo, create a conda environment, and install in editable mode:

`git clone https://github.com/SKTeamLab/esm-msr.git`
`cd esm-msr`
`conda create -n msr_venv python=3.12`
`conda activate msr_venv`
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
`pip install -e .`

## Obtaining ESM3-small-open weights

We recommend downloading the weights from HuggingFace directly into `data/weights`. Alternatively, you can provide a HuggingFace token if you have ESM3 api access.

Weights are available from https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/tree/main/data/weights. Download all files in this folder into `esm-msr/data/weights`, overwriting the placeholder files. You must agree to the Cambrian Non-Commercial License Agreement to get access. Note that the same license applies to our method and can be found in the `LICENSE.md` file. 

## Basic Usage

You must choose a LoRA adapter from `LoRA_models`. 'Unmasked' versions are suggested for using the faster strategy, 'parallel'. Performance losses are negligible relative to using 'singles_only' models with the 'masked' strategy, which performed best in our paper. The 'direct' (masked marginal) strategy is much faster for multimutants but does not correctly capture epistasis and is not recommended. All provided LoRAs have a rank of 6 and a default alpha of 12. This alpha can be lowered to obtain predictions more similar to the base ESM3 model. Choose between mode 'singles', 'doubles' or 'multi'. By default, singles or doubles modes will score all possible mutations of that type for the provided protein. Multi is designed for use with a 'subset_df', which is a csv file with columns 'wt1','pos1','mut1','wt2','pos2','mut2'...'wtn','posn','mutn' indicating which mutations of interest should be scored. This option also works with singles and doubles and is useful for benchmarking. Indices refer to the position within the provided structure file.

`python src/esm_msr/inference.py --checkpoint LoRA_models/msr_chain_unmasked/seed1_epoch\=07-val_rho_avg\=0.754.ckpt --lora_alpha 12 --input_structure path_to_structure_file --strategy parallel --mode singles`

The visualizer generates predictions using this script. You can read the GUI section to understand how `inference.py` can be used from the command line.

## Adding the visualizer to ChimeraX

Download and install ChimeraX (tested version: 1.11) from https://www.cgl.ucsf.edu/chimerax/download.html (free for non-commercial use)

1. Go to Tools -> Command Line Interface (check box)
2. In the command line interface at the bottom, type (replacing the `/path/to/repo`):

`devel install /path/to/repo/ChimeraX-ResidueScoreVisualizer`

## Using the ChimeraX GUI


The GUI workflow has two primary steps: **1. Make predictions** using the model via a background process, and **2. Visualize** the outputs directly on your 3D structure.

### 1. Running an External Prediction

Load a valid protein structure (PDB, mmCIF) into ChimeraX. You can click and drag structure files into the window, or directly download and open a PDB structure (e.g., `open 1enh`) via the ChimeraX command line.

**Repository and Environment Configuration:**
* **Base Repo Directory:** Browse to the root of your cloned `esm-msr` folder. 
* **Environment Path or Conda Name:** Specify the Python environment to run the inference script. If you followed the recommended installation above, simply type `msr_venv` in this box. If that fails, paste the absolute path to your conda environment (find it using `conda env list` in your terminal).
* **HF Token:** Optional. If you haven't downloaded the ESM3 weights locally, enter your HuggingFace token here to fetch them at runtime.
* **Prediction Output File (CSV):** Specify the filepath where the model's output predictions will be saved.

**Model Parameters:**
* **LoRA Path:** Select the trained `.ckpt` file you wish to use from the `LoRA_models` directory.
* **Alpha:** Determines the weight/shift of the LoRA adapter. Default is 12 (optimized for stability). Lowering this (e.g., to 8) shifts the output closer to the base ESM3 model's preferences.
* **Rank:** Must match the rank of your selected LoRA checkpoint (default is 6).

**Inference Strategy Options:**
* **Model & Chain ID:** Select which open ChimeraX model and specific chain you want to predict on. 
* **Strategy:** * `masked` (Default): Masks the target position(s) before prediction. Most accurate for single mutations.
    * `parallel`: Uses unmasked sequences for inference. Provides linear scaling speedups for multi-mutants (use with 'unmasked' LoRAs).
    * `direct`: Predicts the sequence without masking. Fast, but poorly captures epistasis.
* **Multi Paths:** Applies only to multi-mutant predictions. Sets `K_paths` to determine the number of sampling trajectories explored.
* **Disallow Cysteine:** Excludes cysteine (C) from being evaluated as a possible mutation.

**Mutations to Score (Choose ONLY ONE method):**
1.  **Screening Mode:** Exhaustively scores mutations. Choose `singles` to score all possible single amino acid substitutions, or `doubles` to score all KxK pairs. 
    * *Selected Residues:* You can restrict the screen to a specific set of residues. Highlight residues in ChimeraX and click "Grab Selection", or type indices manually (e.g., `11,12,15`). Use the "Invert" checkbox to screen everything *except* those residues.
2.  **Subset CSV:** Upload a predefined CSV list of mutations to score. The script automatically infers the mode based on the columns.
3.  **Specific Mutations:** Manually type a comma-separated list of precise mutations to score (e.g., `A12C,A12C:D15E`).

**Execution:**
* **Compute Device:** Select `gpu` or `cpu`.
* Click **Run Prediction Script**. The tool saves a temporary PDB of your selected chain and runs inference in the background. 
    * *Note:* Singles screening takes <1 minute on modern GPUs. Doubles screening on a 300AA protein takes ~1 hour, scaling with $L(L-1)/2$.

---

### 2. Loading and Visualizing Scores

Once inference completes (or if you already have an existing output CSV), you can map the scores onto your structure. 

**General Visualization Controls:**
* **Score Threshold:** Only mutations with a predicted score above this value will be visualized.
* **Non-Target Chain Transparency %:** Dims the chains you aren't currently analyzing to reduce visual clutter. 

**Standard Single-Mutant Mode:**
*(Requires a CSV generated using the `singles` mode. Ensure "Epistasis Mode" is unchecked.)*
* **Color Backbone by Highest ΔΔG:** Colors the wild-type backbone on a Red-to-Green gradient based on the highest-scoring possible mutation at each position (Red = highly deleterious, Green = highly beneficial).
* **Show Sticks for High-Scoring Mutations:** Creates a ghost model that physically mutates the residue to the highest-scoring candidate and renders it as opaque sticks. More green = more stabilizing. Since the original mutation is superimposed, you can hover over a mutation to see whether it comes from a mutant model or the original.
* **Visualize Contacts:** Generates dotted lines indicating non-covalent contacts for the mutated sticks to help rationalise the predicted stability shift.

**Epistasis Mode (Double Mutants):**
*(Requires a CSV generated using the `doubles` mode. Check the "Epistasis Mode" box.)*
* **Epistasis dddg_pred Threshold:** Filters out weak epistatic interactions based on absolute magnitude.
* **What it shows:** Turns the active model transparent and replaces residues involved in significant epistatic interactions with their mutated counterparts as solid sticks. Dotted lines are drawn between interacting pairs:
    * 🟩 **Green Line:** Positive epistasis (score > 0).
    * 🟥 **Red Line:** Negative epistasis (score < 0).
    * Line thickness and brightness is based on the magnitude of the predicted interaction.

Click **Load CSV + Visualize Scores** to apply your selected settings.
