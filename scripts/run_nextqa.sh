#!/bin/bash
# =============================================================================
# DynaTokens on NExT-QA (domain-incremental, 8 tasks: TP CW DC TC DL DO TN CH)
#
# Usage (from the repository root):
#   bash scripts/run_nextqa.sh
#
# Every setting below can be overridden from the environment, e.g.
#   NGPUS=1 BATCH_SIZE=2 ACCUM_ITER=32 bash scripts/run_nextqa.sh
#   LLAMA_PATH=/path/to/Llama-2-7b DATA_ROOT=/path/to/data bash scripts/run_nextqa.sh
#
# PBS users: the script is also a valid PBS job file
#   qsub scripts/run_nextqa.sh
# (adjust the #PBS lines and the CONDA_SH / REPO_DIR variables first).
# =============================================================================
#PBS -N dynatokens_nextqa
#PBS -l select=1:ncpus=6:ngpus=2:mem=128gb
#PBS -l walltime=12:00:00

set -euo pipefail

# ── environment ──────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONDA_ENV="${CONDA_ENV:-dynatokens}"
CONDA_SH="${CONDA_SH:-}"            # e.g. /opt/anaconda3/etc/profile.d/conda.sh (leave empty if conda is already active)

cd "${REPO_DIR}"
if [[ -n "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

# NCCL settings that were required on our cluster; remove if not needed
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ── paths ────────────────────────────────────────────────────────────────────
LLAMA_PATH="${LLAMA_PATH:-./checkpoints/Llama-2-7b}"   # params.json, tokenizer.model, consolidated.00.pth
DATA_ROOT="${DATA_ROOT:-./data}"                       # data/nextqa/{split_data, clipvitl14.pth}
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/nextqa}"

# ── hardware ─────────────────────────────────────────────────────────────────
NGPUS="${NGPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"        # per GPU
ACCUM_ITER="${ACCUM_ITER:-16}"       # effective batch = BATCH_SIZE * ACCUM_ITER * NGPUS = 128

# ── schedule ─────────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
START_MERGE_EPOCH="${START_MERGE_EPOCH:-3}"
BLR="${BLR:-1e-2}"

# ── per-component learning rates ─────────────────────────────────────────────
TOKEN_GEN_LR="${TOKEN_GEN_LR:-1e-3}"
Z_INV_LR="${Z_INV_LR:-1e-3}"
Z_SP_LR="${Z_SP_LR:-1e-3}"

# ── LookAhead regularisation (Eq. 16) ────────────────────────────────────────
ALPHA="${ALPHA:-0.25}"               # lambda_LA
LOOKAHEAD_STEPS="${LOOKAHEAD_STEPS:-2}"
INNER_LR_SCALE="${INNER_LR_SCALE:-0.5}"
MASK_TOPK_PERCENT="${MASK_TOPK_PERCENT:-0.0}"

# ── z_inv regularisation / warm-start / keys ─────────────────────────────────
LAMBDA_INV="${LAMBDA_INV:-0.1}"
RHO_Z="${RHO_Z:-0.1}"
KEY_MODE="${KEY_MODE:-question}"     # question | raw
D_K="${D_K:-256}"
KEY_EMA_BETA="${KEY_EMA_BETA:-0.99}"

# ── auxiliary losses ─────────────────────────────────────────────────────────
WEIGHT_VID="${WEIGHT_VID:-0.75}"     # weight of the qav loss

# ── inference: task-agnostic routing (false) or oracle task id (true) ────────
USE_TASK_ID="${USE_TASK_ID:-false}"

# ── logging ──────────────────────────────────────────────────────────────────
USE_WANDB="${USE_WANDB:-false}"
SAVE_CKPT="${SAVE_CKPT:-false}"      # save best/last checkpoints (trainable params + task bank)

# ── output dir ───────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-${KEY_MODE}_taskid${USE_TASK_ID}_blr${BLR}_tglr${TOKEN_GEN_LR}_alpha${ALPHA}_look${LOOKAHEAD_STEPS}_vaq_qav}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS=()
[[ "${USE_TASK_ID}" == "true" ]] && EXTRA_FLAGS+=(--use_task_id)
[[ "${USE_WANDB}" == "true" ]] && EXTRA_FLAGS+=(--use_wandb --wandb_name "nextqa_${EXP_NAME}")
[[ "${SAVE_CKPT}" == "true" ]] && EXTRA_FLAGS+=(--save_best --save_last --ckpt_name "dynatokens_nextqa")

echo "== DynaTokens / NExT-QA =="
echo "repo: ${REPO_DIR}   gpus: ${NGPUS}   output: ${OUTPUT_DIR}"
nvidia-smi || true

# ── launch ───────────────────────────────────────────────────────────────────
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
torchrun --standalone --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" \
  train.py \
  --dataset nextqa \
  --data_root "${DATA_ROOT}" \
  --llama_model_path "${LLAMA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --start_merge_epoch "${START_MERGE_EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_iter "${ACCUM_ITER}" \
  --blr "${BLR}" \
  --token_gen_lr "${TOKEN_GEN_LR}" \
  --z_inv_lr "${Z_INV_LR}" \
  --z_sp_lr "${Z_SP_LR}" \
  --alpha "${ALPHA}" \
  --looking_head_steps "${LOOKAHEAD_STEPS}" \
  --inner_lr_scale "${INNER_LR_SCALE}" \
  --mask_topk_percent "${MASK_TOPK_PERCENT}" \
  --lambda_inv "${LAMBDA_INV}" \
  --rho_z "${RHO_Z}" \
  --key_mode "${KEY_MODE}" \
  --d_k "${D_K}" \
  --key_ema_beta "${KEY_EMA_BETA}" \
  --weight_aux_vid_loss "${WEIGHT_VID}" \
  --vaq --qav \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
