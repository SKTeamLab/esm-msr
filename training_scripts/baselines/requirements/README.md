# Baseline Retraining Environments

Most users of ESM-MSR only need the main repository environment (`pyproject.toml` / `msr_venv`). The files in this folder are needed **only** if you want to retrain the three external baselines (`ThermoMPNN` / `ThermoMPNN-D`, `SPURS` / `SPURS-multi`, and `MutateEverything` single / double) from scratch, including on modern CUDA 12.8 / Blackwell (`sm_120`, e.g. RTX 5090) GPUs.

Prerequisite: clone the pinned upstream repositories via `bash inference_scripts/baselines/setup_third_party.sh` (by default into `THIRD_PARTY_DIR=~/software`, or `~/software/msr_baselines`).

---

## 1. ThermoMPNN & ThermoMPNN-D (`ThermoMPNN-D`)

```bash
conda create -y -n ThermoMPNN-D python=3.11
conda run -n ThermoMPNN-D pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n ThermoMPNN-D pip install -r training_scripts/baselines/requirements/requirements_thermompnn.txt
```

**Compatibility notes (PyTorch >= 2.6):**
- `inference_scripts/baselines/thermompnn_inference.py` and `slurm_thermompnn_train.sh` set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because Rosetta single-mutant feature files (`FINAL_results/*/pt_models/*.pt`) store NumPy arrays inside `.pt` dictionaries, which PyTorch >= 2.6 blocks under `weights_only=True`.
- `thermompnn_inference.py` shims `torch.optim.lr_scheduler.ReduceLROnPlateau` to ignore `verbose=True` (removed in PyTorch 2.7), keeping the pinned `ThermoMPNN-D` checkout (`df9a75a`) unmodified.

---

## 2. SPURS & SPURS-multi (`spurs`)

```bash
conda create -y -n spurs python=3.10
conda run -n spurs pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n spurs pip install -r training_scripts/baselines/requirements/requirements_spurs.txt
```

**Compatibility notes:**
- SPURS's training pipeline (`spurs/spurs_retrain.patch`) targets `pytorch-lightning==1.7.7`, `torchmetrics==0.11.4`, `numpy==1.26.4`, and `biopython<1.84` (`Bio.pairwise2`).
- `slurm_spurs_train.sh` prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (avoiding `CXXABI_1.3.15` mismatches against system `libstdc++.so.6` when `sqlite3`/ICU loads), sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch >= 2.6 checkpoint loads, and sets `PYTHONPATH=$SPURS_TRAIN_DIR` so the patched training tree takes precedence over any editable `spurs` install.
- On first run, ESM-2 (`esm2_t33_650M_UR50D.pt`) must be present in `~/.cache/torch/hub/checkpoints/` (or set `HF_HUB_OFFLINE=0` once to let `fair-esm` download it).

---

## 3. MutateEverything Stage 1 & Stage 2 (`mutate_everything_cu128`)

Upstream MutateEverything (`034dd91`) originally targeted Python 3.7 + PyTorch 1.12 (`cu113`), which lacks kernels for Ada/Hopper/Blackwell (`sm_89`/`sm_90`/`sm_120`) GPUs. Create `mutate_everything_cu128` (Python 3.10 + PyTorch `cu128` + CUDA Toolkit 12.8 `nvcc`) as follows:

```bash
conda create -y -n mutate_everything_cu128 python=3.10
conda run -n mutate_everything_cu128 pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n mutate_everything_cu128 pip install \
  -r training_scripts/baselines/requirements/requirements_mutate_everything.txt
PYTHON=$(conda run -n mutate_everything_cu128 which python) \
  CUDA_HOME=/usr/local/cuda-12.8 \
  bash training_scripts/baselines/requirements/install_openfold_cu128.sh
```

**Compatibility notes (documented deviations from the legacy `cu113` env):**
- `biopython==1.79` is pinned because OpenFold `1.0.1` imports `Bio.Data.SCOPData` (removed in Biopython 1.82); `numpy==1.26.4` is pinned for NumPy 1.x compatibility.
- `deepspeed` is intentionally omitted: OpenFold's `checkpointing.py` falls back cleanly to `torch.utils.checkpoint.checkpoint` when `deepspeed` is not installed.
- `install_openfold_cu128.sh` copies `$MUTATE_EVERYTHING_DIR/openfold` to a temporary directory, updates `-std=c++14` to `-std=c++17` (required by PyTorch >= 2.1 headers), compiles `attn_core_inplace_cuda` for `sm_80;sm_86;sm_89;sm_90;sm_120` (removing deprecated `sm_37`/`sm_52` rejected by `nvcc` 12.8), and leaves the `MutateEverything-oct06` checkout clean.
- `slurm_mutate_everything_train.sh` automatically prefers `mutate_everything_cu128` when present and falls back to `mutate_everything`.
