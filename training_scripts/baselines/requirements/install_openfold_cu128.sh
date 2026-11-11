#!/bin/bash
# Build and install OpenFold's `attn_core_inplace_cuda` extension from the pinned
# MutateEverything-oct06 checkout into a modern PyTorch 2.x + CUDA 12.x environment
# without modifying the tracked files in $MUTATE_EVERYTHING_DIR.
#
# Usage:
#   PYTHON=/path/to/mutate_everything_cu128/bin/python \
#   CUDA_HOME=/usr/local/cuda-12.8 \
#   MUTATE_EVERYTHING_DIR=~/software/MutateEverything-oct06 \
#     bash training_scripts/baselines/requirements/install_openfold_cu128.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
if [ -z "${CUDA_HOME:-}" ]; then
  for cand in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [ -x "$cand/bin/nvcc" ]; then
      CUDA_HOME=$cand
      break
    fi
  done
fi
: "${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit directory containing bin/nvcc (e.g. /usr/local/cuda-12.8)}"

if [ -z "${MUTATE_EVERYTHING_DIR:-}" ]; then
  for cand in "${THIRD_PARTY_DIR:-$HOME/software}/MutateEverything-oct06" "$HOME/software/msr_baselines/MutateEverything-oct06"; do
    if [ -d "$cand/openfold" ]; then
      MUTATE_EVERYTHING_DIR=$cand
      break
    fi
  done
fi
: "${MUTATE_EVERYTHING_DIR:?set MUTATE_EVERYTHING_DIR to the MutateEverything-oct06 checkout}"

TMP_BUILD=$(mktemp -d "${TMPDIR:-/tmp}/openfold_build_XXXXXX")
trap 'rm -rf "$TMP_BUILD"' EXIT
cp -r "$MUTATE_EVERYTHING_DIR/openfold/." "$TMP_BUILD/"

# Patch the temporary OpenFold build tree before installing:
# 1. PyTorch >= 2.1 C++/CUDA headers require C++17 (-std=c++17 instead of -std=c++14).
# 2. CUDA 12.8 removed sm_37 / sm_52 and WSL2/headless nodes may not expose libcuda.so to
#    scripts.utils.get_nvidia_cc(); compile fatbins for sm_80..sm_120 (or $OPENFOLD_CUDA_ARCHS).
# 3. NumPy >= 1.24 removed deprecated scalar/type aliases (np.object, np.int, np.float,
#    np.bool, np.str) used in openfold/data/templates.py and openfold/np/relax/utils.py.
"$PYTHON" - "$TMP_BUILD" <<'PY'
from pathlib import Path
import os
import re
import sys

root = Path(sys.argv[1])
p = root / "setup.py"
src = p.read_text()
src = src.replace("'-std=c++14'", "'-std=c++17'")
src = src.replace("'cxx': ['-O3'] + version_dependent_macros,", "'cxx': ['-O3', '-std=c++17'] + version_dependent_macros,")

old_cc = """compute_capabilities = set([
    (3, 7), # K80, e.g.
    (5, 2), # Titan X
    (6, 1), # GeForce 1000-series
])

compute_capabilities.add((7, 0))
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    compute_capabilities.add((8, 0))

compute_capability, _ = get_nvidia_cc()
if compute_capability is not None:
    compute_capabilities = set([compute_capability])"""

new_cc = """_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
arch_env = os.environ.get("OPENFOLD_CUDA_ARCHS", "8.0;8.6;8.9;9.0;12.0" if int(bare_metal_major) >= 12 else "7.0;8.0;8.6;8.9;9.0")
compute_capabilities = sorted({tuple(int(x) for x in item.strip().split(".")) for item in arch_env.split(";") if item.strip()})"""

assert old_cc in src, "Unexpected OpenFold setup.py layout"
src = src.replace(old_cc, new_cc, 1)
p.write_text(src)

np_alias_re = re.compile(r"(?<!openfold\.)\bnp\.(object|int|float|bool|str)\b")
patched_files = 0
for py_file in sorted(root.rglob("*.py")):
    text = py_file.read_text(encoding="utf-8")
    new_text, count = np_alias_re.subn(r"\1", text)
    if count:
        py_file.write_text(new_text, encoding="utf-8")
        patched_files += 1
print(f"Patched deprecated NumPy aliases in {patched_files} OpenFold file(s).")
PY

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
"$PYTHON" -m pip install --no-build-isolation --no-deps "$TMP_BUILD"
