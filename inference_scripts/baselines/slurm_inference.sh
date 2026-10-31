#!/bin/bash
# Inference of one model on EVERY benchmark with resource tracking (Tsuboyama val/test, the 12 external sets, the 7 DMS,
# Domainome, ProteinGym), as a Slurm job. Submit it through ../sbatch_fixed_slot.sh so every model gets the same slot:
#
#   MODEL=esm_msr          CKPT=release-baseline/s1/epoch=04-val_rho_combined_avg=0.855.ckpt OUT=runs/esm_msr    ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=thermompnn       SINGLE=.../single.ckpt EPISTATIC=.../epistatic.ckpt MSR_BASELINE_DATA=<prepared dir> OUT=runs/thermompnn ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=mutate_everything CKPT=logs/af_double_..._seed1/checkpoint-99.pth SPLIT_NAME=<data/ dir> OUT=runs/me ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=spurs            OUT=runs/spurs                                                 ../sbatch_fixed_slot.sh slurm_inference.sh
#   MODEL=proteinmpnn      OUT=runs/proteinmpnn                                           ../sbatch_fixed_slot.sh slurm_inference.sh
#
# Every line also needs SPLIT, the Tsuboyama split (a name under data/, e.g. SPLIT=splits_oct06_capped). ESM-MSR, SPURS and
# ProteinMPNN read its pickle; ThermoMPNN and ME read tables prepared from it (prepare/prepare_baseline_data.py: MSR_BASELINE_DATA is
# that output directory; build_mutate_everything_data.py: SPLIT_NAME is data/<name> in the ME checkout) and stop if those were
# prepared from another split.
#
# Batch size (every method but MutateEverything, which has no variant batching): BATCH_MODE=table|auto|fixed, default table = one
# size per model and dataset from batch_sizes_gpu4.json (ProteinGym dynamic); BATCH_SIZE overrides the table (fixed: items per
# call; auto: a cap). CALIBRATE=1 instead of benchmarking runs the sizer on each dataset's worst-case proteins and writes
# $OUT/batch_sizes_$MODEL.json (calibration.py merge builds the table); do it on the GPU the table is for.
# Optional, any model:
# MUTANTS=single|multi|both (which rows; default both), PGYM_DMS=id,id (ProteinGym subset; default all), EXTRA="..." (appended to the model's command line;
# EXTRA="--max_proteins 2" is the smoke subset). Model specific: ThermoMPNN accepts SINGLE only,
# EPISTATIC only, or both; SPURS MODELS=single|multi|both, SPURS_SINGLE / SPURS_MULTI checkpoints; ME SETS (default all).
# ESM-MSR GAMUT_ONLY=1 stops after the 22-dataset gamut (no ProteinGym phase, which alone takes ~33 h).
# Outputs land in $OUT (predictions per benchmark, ProteinGym summaries, resources/<run_id>/ with the tracker's files).
#
# The Python environments are in envs/ (conda env names msr_venv, spurs_higher, mutate_everything; a venv for ThermoMPNN-D);
# upstream checkouts: setup_third_party.sh. CONDA_SH points at the conda profile script; another machine names its environments
# differently: MSR_ENV (ESM-MSR, ProteinMPNN), SPURS_ENV, ME_ENV, THERMOMPNN_PY (the python of the ThermoMPNN-D environment).
# The script also runs outside Slurm (`bash slurm_inference.sh`); without MSR_FIXED_* the tracker checks no slot.
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
SPLIT=${SPLIT:?set SPLIT (the Tsuboyama split under data/, e.g. splits_oct06_capped)}
MUTANTS=${MUTANTS:-both}
EXTRA=${EXTRA:-}
BATCH=(--batch_mode "${BATCH_MODE:-table}" ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"})     # not for MutateEverything: one protein per forward
CAL=(${CALIBRATE:+--calibrate_batch_sizes "$OUT/batch_sizes_$MODEL.json"})
CONDA_SH=${CONDA_SH:-/usr/local/anaconda3/etc/profile.d/conda.sh}
[ -f "$CONDA_SH" ] || CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
hostname
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" || { echo "unhealthy GPU in allocation (${CUDA_VISIBLE_DEVICES:-?})"; exit 1; }
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export HF_HUB_OFFLINE=1
# expandable segments help the large-variant-count models; MutateEverything's older torch rejects the option
[ "$MODEL" = mutate_everything ] || export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
activate() { set +u; source "$CONDA_SH"; conda activate "$1"; set -u; }
PGYM=${MSR_PGYM_INPUTS:-$REPO/pgym_inputs}
STRUCT=${MSR_STRUCTURES_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/esm-msr/data/structures}
echo "model=$MODEL out=$OUT mutants=$MUTANTS split=$SPLIT slot: cpus=${MSR_FIXED_CPUS:-?} ram=${MSR_FIXED_RAM_GB:-?}G"
case $MODEL in
  esm_msr)
    CKPT=${CKPT:?set CKPT (a checkpoint under LoRA_models/)}
    activate "${MSR_ENV:-msr_venv}"
    cd "$REPO/inference_scripts"
    python -u esm_msr_testing.py --checkpoint "$CKPT" --split "$SPLIT" --local_cluster --local_path_to_structures "$STRUCT" \
        --analysis_root "$OUT/analysis" --resource_dir "$OUT/resources" "${BATCH[@]}" ${CAL[@]+"${CAL[@]}"} $EXTRA
    [ -z "${CALIBRATE:-}" ] || { echo "calibration done: $OUT/batch_sizes_$MODEL.json"; exit 0; }
    [ -z "${GAMUT_ONLY:-}" ] || { echo "done: $OUT (GAMUT_ONLY: no ProteinGym phase)"; exit 0; }
    python -u esm_msr_testing.py --checkpoint "$CKPT" --split "$SPLIT" --local_cluster --local_path_to_structures "$STRUCT" \
        --protein_gym --pgym_dir "$PGYM" --pgym_dms "${PGYM_DMS:-all}" --pgym_out "$OUT/proteingym" --analysis_root "$OUT/analysis" --resource_dir "$OUT/resources" "${BATCH[@]}" $EXTRA;;
  thermompnn)
    PY=${THERMOMPNN_PY:-${THERMOMPNN_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/ThermoMPNN-D}/.venv/bin/python}
    [ -n "${SINGLE:-}${EPISTATIC:-}" ] || { echo "set SINGLE and/or EPISTATIC (ThermoMPNN-D checkpoints)"; exit 1; }
    "$PY" -u "$B/thermompnn_inference.py" all ${SINGLE:+--single_ckpt "$SINGLE"} ${EPISTATIC:+--epistatic_ckpt "$EPISTATIC"} \
        --tsuboyama_split "$SPLIT" --out "$OUT" --mutants "$MUTANTS" --pgym_dir "$PGYM" --pgym_dms "${PGYM_DMS:-all}" "${BATCH[@]}" ${CAL[@]+"${CAL[@]}"} $EXTRA;;
  mutate_everything)
    CKPT=${CKPT:?set CKPT (a MutateEverything checkpoint, relative to the ME checkout or absolute)}
    module purge 2>/dev/null || true
    activate "${ME_ENV:-mutate_everything}"
    python -u "$B/mutate_everything_inference.py" --resume "$CKPT" --output_dir "$OUT" --split_name "${SPLIT_NAME:?set SPLIT_NAME (the ME data/ directory of the split)}" --tsuboyama_split "$SPLIT" --sets "${SETS:-all}" --mutants "$MUTANTS" \
        --skip_existing --continue_on_error $EXTRA;;
  spurs)
    activate "${SPURS_ENV:-spurs_higher}"
    python -u "$B/spurs_inference.py" --out "$OUT" --models "${MODELS:-both}" --mutants "$MUTANTS" --split "$SPLIT" \
        --checkpoint_single "${SPURS_SINGLE:-hf}" --checkpoint_multi "${SPURS_MULTI:-hf}" --pgym_dir "$PGYM" --pgym_dms "${PGYM_DMS:-all}" "${BATCH[@]}" ${CAL[@]+"${CAL[@]}"} $EXTRA;;
  proteinmpnn)
    activate "${MSR_ENV:-msr_venv}"
    python -u "$B/proteinmpnn_inference.py" --out "$OUT" --mutants "$MUTANTS" --split "$SPLIT" --pgym_dir "$PGYM" --pgym_dms "${PGYM_DMS:-all}" "${BATCH[@]}" ${CAL[@]+"${CAL[@]}"} $EXTRA;;
  *) echo "unknown MODEL=$MODEL"; exit 1;;
esac
echo "done: $OUT (resource report: python analysis_scripts/resource_report.py $OUT/resources)"
