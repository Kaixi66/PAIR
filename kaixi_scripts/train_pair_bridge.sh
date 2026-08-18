#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

############################
# Training settings
############################

DATASET_NAME="${DATASET_NAME:-libero_10_no_noops}"
TFDS_DATASET_NAME="${TFDS_DATASET_NAME:-${DATASET_NAME}}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${PAIR_DATA_DIR}/libero}"
PAIR_ACTION_AE_ENCODER_PATH="${PAIR_ACTION_AE_ENCODER_PATH:-}"

GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}" # constant | multistep | cosine
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LORA_RANK="${LORA_RANK:-64}"
MAX_STEPS="${MAX_STEPS:-30000}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-${MAX_STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-${MAX_STEPS}}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-True}"
PAIR_ALIGN_WEIGHT="${PAIR_ALIGN_WEIGHT:-0.1}"
IMAGE_AUG="${IMAGE_AUG:-True}"

USE_VAL_SET="${USE_VAL_SET:-False}"
VAL_FREQ="${VAL_FREQ:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-50}"
VAL_SEED="${VAL_SEED:-7}"
VAL_SHUFFLE_BUFFER_SIZE="${VAL_SHUFFLE_BUFFER_SIZE:-10000}"
VAL_SPLIT_PERCENT="${VAL_SPLIT_PERCENT:-5}"
SAVE_BEST_VAL_CHECKPOINT="${SAVE_BEST_VAL_CHECKPOINT:-False}"
BEST_VAL_METRIC="${BEST_VAL_METRIC:-action_l1_loss}"
EPISODE_SPLIT_FILE="${EPISODE_SPLIT_FILE:-}"
DATA_CONTRACT_FILE="${DATA_CONTRACT_FILE:-}"

EXP_NAME="${EXP_NAME:-PAIR_${DATASET_NAME}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PAIR_CHECKPOINTS_DIR}}"
LOG_DIR="${LOG_DIR:-${PAIR_LOGS_DIR}/bridge}"
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"

VLM_PATH="${VLM_PATH:-${PAIR_PRETRAINED_MODELS_DIR}/prism-qwen25-extra-dinosiglip-224px-0_5b}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-${PAIR_PRETRAINED_MODELS_DIR}/configs}"
DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

############################
# Validation and wiring
############################

[[ -n "${PAIR_ACTION_AE_ENCODER_PATH}" ]] || {
    echo "PAIR_ACTION_AE_ENCODER_PATH is required." >&2
    exit 2
}

gpu_list="${GPUS// /}"
IFS=',' read -r -a gpu_array <<< "${gpu_list}"
if [[ "${NPROC_PER_NODE}" == auto ]]; then
    NPROC_PER_NODE="${#gpu_array[@]}"
fi

if [[ "${DRY_RUN}" != true ]]; then
    [[ -d "${DATA_ROOT_DIR}/${TFDS_DATASET_NAME}" ]] || {
        echo "Missing RLDS dataset: ${DATA_ROOT_DIR}/${TFDS_DATASET_NAME}" >&2
        exit 1
    }
    [[ -f "${PAIR_ACTION_AE_ENCODER_PATH}" ]] || {
        echo "Missing Action-AE encoder: ${PAIR_ACTION_AE_ENCODER_PATH}" >&2
        exit 1
    }
    [[ -z "${EPISODE_SPLIT_FILE}" || -f "${EPISODE_SPLIT_FILE}" ]] || {
        echo "Missing episode split: ${EPISODE_SPLIT_FILE}" >&2
        exit 1
    }
    [[ -z "${DATA_CONTRACT_FILE}" || -f "${DATA_CONTRACT_FILE}" ]] || {
        echo "Missing data contract: ${DATA_CONTRACT_FILE}" >&2
        exit 1
    }
fi

mkdir -p "${RUN_ROOT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cmd=(
    "${PAIR_PYTHON}" -m torch.distributed.run
    --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}"
    vla-scripts/finetune.py
    --vlm_path "${VLM_PATH}"
    --config_file_path "${CONFIG_FILE_PATH}"
    --data_root_dir "${DATA_ROOT_DIR}"
    --dataset_name "${DATASET_NAME}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --use_film True
    --num_images_in_input 2
    --use_proprio True
    --use_lora True
    --use_fz False
    --use_minivlm True
    --use_pro_version False
    --use_pair_bridge True
    --image_aug "${IMAGE_AUG}"
    --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
    --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_warmup_ratio "${LR_WARMUP_RATIO}"
    --lr_warmup_steps "${LR_WARMUP_STEPS}"
    --max_steps "${MAX_STEPS}"
    --save_freq "${SAVE_FREQ}"
    --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
    --batch_size "${BATCH_SIZE}"
    --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lora_rank "${LORA_RANK}"
    --pair_action_ae_encoder_path "${PAIR_ACTION_AE_ENCODER_PATH}"
    --pair_align_weight "${PAIR_ALIGN_WEIGHT}"
    --use_val_set "${USE_VAL_SET}"
    --val_freq "${VAL_FREQ}"
    --val_num_batches "${VAL_NUM_BATCHES}"
    --val_seed "${VAL_SEED}"
    --val_shuffle_buffer_size "${VAL_SHUFFLE_BUFFER_SIZE}"
    --val_split_percent "${VAL_SPLIT_PERCENT}"
    --save_best_val_checkpoint "${SAVE_BEST_VAL_CHECKPOINT}"
    --best_val_metric "${BEST_VAL_METRIC}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_log_freq "${WANDB_LOG_FREQ}"
    --run_id_override "${EXP_NAME}"
)

[[ -z "${EPISODE_SPLIT_FILE}" ]] || cmd+=(--episode_split_file "${EPISODE_SPLIT_FILE}")
[[ -z "${DATA_CONTRACT_FILE}" ]] || cmd+=(--data_contract_file "${DATA_CONTRACT_FILE}")

cd "${PAIR_REPO_DIR}"
cat <<EOF
[PAIR] dataset=${DATASET_NAME} data=${DATA_ROOT_DIR}
[PAIR] gpus=${GPUS} effective_batch=$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
[PAIR] lr=${LEARNING_RATE} scheduler=${LR_SCHEDULER} max_steps=${MAX_STEPS}
[PAIR] action_ae_encoder=${PAIR_ACTION_AE_ENCODER_PATH}
[PAIR] architecture=base-v1 (FiLM, two images, proprio, start injection, cosine alignment)
[PAIR] output=${RUN_ROOT_DIR} log=${LOG_FILE}
EOF
printf '[PAIR] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi
if [[ "${BACKGROUND}" == true ]]; then
    nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
    echo "[PAIR] started pid=$! log=${LOG_FILE}"
else
    "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
