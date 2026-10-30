#!/bin/bash
# ESM3-small-open (ESM-MSR checkpoint at sigma = 0, so the LoRA adapters are switched off and only the calibration head remains)
# scored with canonical masked-marginal or independent masking on the gamut: Tsuboyama val/test, the 12 external sets, the 7 DMS, Domainome.
# No ProteinGym, no context runs. Submit one job per strategy through the fixed gpu4 slot (run from the repository root on the cluster):
#
#   STRATEGY=marginal    OUT=outputs/esm3_masked/marginal    inference_scripts/sbatch_fixed_slot.sh ~/msr_masked_runs/esm3_masked_gamut.sh
#   STRATEGY=independent OUT=outputs/esm3_masked/independent inference_scripts/sbatch_fixed_slot.sh ~/msr_masked_runs/esm3_masked_gamut.sh
#
#   marginal     every mutated position masked at once, in the wild-type sequence: Meier et al.'s masked marginal. The mutant-side pass sees the
#                same masked input, so it is skipped (--skip_reverse): wt_lora_pred is the score.
#   independent  each mutated position masked on its own with the other mutations visible (the mutant pass) or not (the WT pass):
#                mt_lora_pred is the pseudo-likelihood-style score, wt_lora_pred the additive reading. Domainome reads the WT pass only.
# Outputs: $OUT/analysis/{predictions,stats}/... with the strategy in the file tag (_marginal, _independent), $OUT/resources/.
#SBATCH --job-name=esm3_mask
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
STRATEGY=${STRATEGY:?set STRATEGY (marginal|independent)}
OUT=${OUT:?set OUT}
CKPT=${CKPT:-esm-msr-baseline-w2a4-m16a16-capped/epoch=07-val_rho_combined_avg=0.839.ckpt}   # under LoRA_models/: the released recipe retrained on the capped split
SPLIT=${SPLIT:-splits_oct06_capped}
REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CONDA_SH=${CONDA_SH:-/usr/local/anaconda3/etc/profile.d/conda.sh}
[ -f "$CONDA_SH" ] || CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
case $STRATEGY in
  marginal)    EXTRA_STRAT="--skip_reverse";;
  independent) EXTRA_STRAT="--skip_reverse_domainome";;
  *) echo "unknown STRATEGY=$STRATEGY"; exit 1;;
esac
hostname
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" || { echo "unhealthy GPU in allocation (${CUDA_VISIBLE_DEVICES:-?})"; exit 1; }
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
set +u; source "$CONDA_SH"; conda activate "${MSR_ENV:-msr_venv}"; set -u
STRUCT=${MSR_STRUCTURES_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/esm-msr/data/structures}
echo "esm3 sigma0 strategy=$STRATEGY ckpt=$CKPT out=$OUT split=$SPLIT slot: cpus=${MSR_FIXED_CPUS:-?} ram=${MSR_FIXED_RAM_GB:-?}G"
cd "$REPO/inference_scripts"
python -u esm_msr_testing.py --checkpoint "$CKPT" --split "$SPLIT" --local_cluster --local_path_to_structures "$STRUCT" \
    --analysis_root "$OUT/analysis" --resource_dir "$OUT/resources" --batch_mode "${BATCH_MODE:-auto}" \
    --sigma 0 --mask_strategy "$STRATEGY" --skip_ctx $EXTRA_STRAT ${EXTRA:-}
echo "done: $OUT"
