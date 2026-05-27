#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit_stage2_regression_eval.sh {smoke|10k|50k}

Environment overrides:
  REPO_DIR          Deployed Stage 2 checkout (default: /u/msalunkhe/DiffusionFusion)
  VENV_PATH         Python environment (default: /u/msalunkhe/.venv)
  TRAIN_OUTPUT_DIR  Legacy 80-epoch checkpoint directory
  OUTPUT_BASE       Parent directory for evaluation outputs
  GEN_BSZ           Generation batch size per GPU (default: 64)
  DRY_RUN=1         Print the guarded sbatch command without submitting
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  smoke)
    NUM_IMAGES=256
    ;;
  10k)
    NUM_IMAGES=10000
    ;;
  50k)
    NUM_IMAGES=50000
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

RUNG="$1"
REPO_DIR="${REPO_DIR:-/u/msalunkhe/DiffusionFusion}"
VENV_PATH="${VENV_PATH:-/u/msalunkhe/.venv}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep}"
OUTPUT_BASE="${OUTPUT_BASE:-${TRAIN_OUTPUT_DIR}/eval_stage2_check}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUNG}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"
FID_STATS_PATH="${FID_STATS_PATH:-${REPO_DIR}/JiT/fid_stats/jit_in256_stats.npz}"
SBATCH_SCRIPT="${REPO_DIR}/sbatch/jit_eval.sbatch"

for required_file in \
  "${REPO_DIR}/JiT/model_jit.py" \
  "${REPO_DIR}/JiT/denoiser.py" \
  "${SBATCH_SCRIPT}" \
  "${TRAIN_OUTPUT_DIR}/checkpoint-last.pth" \
  "${DECODER_CHECKPOINT}" \
  "${FID_STATS_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

echo "Checking Stage 2 deployment under ${REPO_DIR}"
if git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Deployed branch: $(git -C "${REPO_DIR}" branch --show-current)"
  echo "Deployed revision: $(git -C "${REPO_DIR}" rev-parse HEAD)"
  git -C "${REPO_DIR}" status --short --branch
else
  echo "No git metadata found under ${REPO_DIR}; source-symbol checks still apply."
fi
if ! grep -nE "JiTMultiStream|remap_dual_to_multistream" "${REPO_DIR}/JiT/model_jit.py"; then
  echo "Stage 2 model/remap symbols are absent from deployed model_jit.py." >&2
  exit 1
fi
if ! grep -n "streams: Dict\\[str, torch.Tensor\\]" "${REPO_DIR}/JiT/denoiser.py"; then
  echo "Stage 2 dict-keyed Denoiser signature is absent from deployed denoiser.py." >&2
  exit 1
fi

echo "Submitting ${RUNG}: images=${NUM_IMAGES}, checkpoint=${TRAIN_OUTPUT_DIR}/checkpoint-last.pth"
echo "Baseline settings: model_ema1 via evaluator, Heun-50, CFG 2.9, interval [0.1, 1.0]"

SBATCH_EXPORTS="ALL"
SBATCH_EXPORTS+=",REPO_DIR=${REPO_DIR},VENV_PATH=${VENV_PATH}"
SBATCH_EXPORTS+=",TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR},RESUME_DIR=${TRAIN_OUTPUT_DIR}"
SBATCH_EXPORTS+=",OUTPUT_DIR=${OUTPUT_DIR},DECODER_CHECKPOINT=${DECODER_CHECKPOINT}"
SBATCH_EXPORTS+=",FID_STATS_PATH=${FID_STATS_PATH},NUM_IMAGES=${NUM_IMAGES}"
SBATCH_EXPORTS+=",GEN_BSZ=${GEN_BSZ:-64},CFG=2.9,INTERVAL_MIN=0.1,INTERVAL_MAX=1.0"
SBATCH_EXPORTS+=",SAMPLING_METHOD=heun,NUM_SAMPLING_STEPS=50,DINO_TIME_SHIFT=0.0"
SBATCH_EXPORTS+=",WANDB_MODE=offline"
submit_command=(sbatch "--export=${SBATCH_EXPORTS}" "${SBATCH_SCRIPT}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "DRY RUN:"
  printf " %q" "${submit_command[@]}"
  printf "\n"
  exit 0
fi

"${submit_command[@]}"
