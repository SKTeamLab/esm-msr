#!/bin/bash
# Inference of one model on EVERY benchmark with resource tracking (Tsuboyama val/test, the 12 external sets, the 7 DMS,
# Domainome, ProteinGym), as a Slurm job. Submit it through ../sbatch_fixed_slot.sh so every model gets the same slot:
#
#   MODEL=esm_msr          CKPT=release-baseline/s1/epoch=04-val_rho_combined_avg=0.855.ckpt OUT=runs/esm_msr    ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=thermompnn       SINGLE=.../single.ckpt EPISTATIC=.../epistatic.ckpt            OUT=runs/thermompnn ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=mutate_everything CKPT=logs/af_double_..._seed1/checkpoint-99.pth               OUT=runs/me         ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=spurs            OUT=runs/spurs                                                 ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=proteinmpnn      OUT=runs/proteinmpnn                                           ../sbatch_fixed_slot.sh slurm_inference.sh
#
# Optional, any model: MUTANTS=single|multi|both (which rows; default both), SPLIT (Tsuboyama split, default
# splits_oct06_capped), EXTRA="..." (appended to the model's command line). Model specific: ThermoMPNN accepts SINGLE only,
# EPISTATIC only, or both; SPURS MODELS=single|multi|both, SPURS_SINGLE / SPURS_MULTI checkpoints; ME SETS (default all).
# Outputs land in $OUT (predictions per benchmark, ProteinGym summaries, resources/<run_id>/ with the tracker's files).
#
# The Python environments are in envs/ (conda env names msr_venv, spurs_higher, mutate_everything; a venv for ThermoMPNN-D);
# upstream checkouts: setup_third_party.sh. CONDA_SH points at the conda profile script.
#SBATCH --job-name=bench_inf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
MODEL=${MODEL:?set MODEL (esm_msr|thermompnn|mutate_everything|spurs|proteinmpnn)}
OUT=${OUT:?set OUT}
REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}          # sbatch runs a copy of this script, so the repo cannot be found from $0
B=$REPO/inference_scripts/baselines
SPLIT=${SPLIT:-splits_oct06_capped}
MUTANTS=${MUTANTS:-both}
EXTRA=${EXTRA:-}
CONDA_SH=${CONDA_SH:-/usr/local/anaconda3/etc/profile.d/conda.sh}
hostname
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" || { echo "unhealthy GPU in allocation (${CUDA_VISIBLE_DEVICES:-?})"; exit 1; }
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
activate() { set +u; source "$CONDA_SH"; conda activate "$1"; set -u; }
PGYM=${MSR_PGYM_INPUTS:-$REPO/pgym_inputs}
STRUCT=${MSR_STRUCTURES_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/esm-msr/data/structures}
echo "model=$MODEL out=$OUT mutants=$MUTANTS split=$SPLIT slot: cpus=${MSR_FIXED_CPUS:-?} ram=${MSR_FIXED_RAM_GB:-?}G"
case $MODEL in
  esm_msr)
    CKPT=${CKPT:?set CKPT (a checkpoint under LoRA_models/)}
    activate msr_venv
    cd "$REPO/inference_scripts"
    python -u esm_msr_testing.py --checkpoint "$CKPT" --split "$SPLIT" --local_cluster --local_path_to_structures "$STRUCT" \
        --resource_dir "$OUT/resources" $EXTRA
    python -u esm_msr_testing.py --checkpoint "$CKPT" --split "$SPLIT" --local_cluster --local_path_to_structures "$STRUCT" \
        --protein_gym --pgym_dir "$PGYM" --pgym_out "$OUT/proteingym" --resource_dir "$OUT/resources" $EXTRA;;
  thermompnn)
    PY=${THERMOMPNN_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/ThermoMPNN-D}/.venv/bin/python
    [ -n "${SINGLE:-}${EPISTATIC:-}" ] || { echo "set SINGLE and/or EPISTATIC (ThermoMPNN-D checkpoints)"; exit 1; }
    "$PY" -u "$B/thermompnn.py" all ${SINGLE:+--single_ckpt "$SINGLE"} ${EPISTATIC:+--epistatic_ckpt "$EPISTATIC"} \
        --out "$OUT" --mutants "$MUTANTS" --pgym_dir "$PGYM" $EXTRA;;
  mutate_everything)
    CKPT=${CKPT:?set CKPT (a MutateEverything checkpoint, relative to the ME checkout or absolute)}
    module purge 2>/dev/null || true
    activate mutate_everything
    python -u "$B/mutate_everything.py" --resume "$CKPT" --output_dir "$OUT" --sets "${SETS:-all}" --mutants "$MUTANTS" \
        --skip_existing --continue_on_error $EXTRA;;
  spurs)
    activate spurs_higher
    python -u "$B/spurs.py" --out "$OUT" --models "${MODELS:-both}" --mutants "$MUTANTS" --split "$SPLIT" \
        --checkpoint_single "${SPURS_SINGLE:-hf}" --checkpoint_multi "${SPURS_MULTI:-hf}" --pgym_dir "$PGYM" $EXTRA;;
  proteinmpnn)
    activate msr_venv
    python -u "$B/proteinmpnn.py" --out "$OUT" --mutants "$MUTANTS" --split "$SPLIT" --pgym_dir "$PGYM" $EXTRA;;
  *) echo "unknown MODEL=$MODEL"; exit 1;;
esac
echo "done: $OUT (resource report: python analysis_scripts/resource_report.py $OUT/resources)"
