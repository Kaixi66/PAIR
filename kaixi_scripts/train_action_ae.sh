#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

############################
# Training settings
############################

DATASET_NAME="${DATASET_NAME:-libero_10_no_noops}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${PAIR_DATA_DIR}/libero}"
CONFIG_PATH="${CONFIG_PATH:-${PAIR_WORKSPACE_DIR}/action_ae/configs/libero_all_v2_perception.yaml}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-${MAX_STEPS}}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SEED="${SEED:-7}"
EPISODE_SPLIT_FILE="${EPISODE_SPLIT_FILE:-}"
DATA_CONTRACT_FILE="${DATA_CONTRACT_FILE:-}"

EXP_NAME="${EXP_NAME:-ae_${DATASET_NAME}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PAIR_RUN_ROOT_DIR}/action_ae}"
LOG_DIR="${LOG_DIR:-${PAIR_LOGS_DIR}/action_ae}"
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

gpu_list="${GPUS// /}"
IFS=',' read -r -a gpu_array <<< "${gpu_list}"
if [[ "${NPROC_PER_NODE}" == auto ]]; then
    NPROC_PER_NODE="${#gpu_array[@]}"
fi

mkdir -p "${RUN_ROOT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${PAIR_WORKSPACE_DIR}/action_ae:${PAIR_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
    --config "${CONFIG_PATH}"
    --ae_version v2
    --data_root_dir "${DATA_ROOT_DIR}"
    --mixture "${DATASET_NAME}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --run_name "${EXP_NAME}"
    --batch_size "${BATCH_SIZE}"
    --max_steps "${MAX_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --log_every "${LOG_EVERY}"
    --eval_every "${EVAL_EVERY}"
    --save_every "${SAVE_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --seed "${SEED}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
    --latent_dim 16
    --encoder_layers 1
    --perception_layers 1
    --decoder_layers 1
    --mask_mode random
    --mask_prob 0.5
    --mask_count 4
    --noise_std 0.05
    --vlm_path "${PAIR_PRETRAINED_MODELS_DIR}/prism-qwen25-extra-dinosiglip-224px-0_5b"
    --vla_config_file_path "${PAIR_PRETRAINED_MODELS_DIR}/configs"
    --num_images_in_input 2
)

[[ -z "${EPISODE_SPLIT_FILE}" ]] || args+=(--episode_split_file "${EPISODE_SPLIT_FILE}")
[[ -z "${DATA_CONTRACT_FILE}" ]] || args+=(--data_contract_file "${DATA_CONTRACT_FILE}")

cmd=(
    "${PAIR_PYTHON}" -m torch.distributed.run
    --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}"
    --module pair_action_ae.train
    "${args[@]}"
)

cat <<EOF
[Action-AE] dataset=${DATASET_NAME} data=${DATA_ROOT_DIR}
[Action-AE] gpus=${GPUS} effective_batch=$((NPROC_PER_NODE * BATCH_SIZE))
[Action-AE] lr=${LEARNING_RATE} max_steps=${MAX_STEPS}
[Action-AE] architecture=base-v1 (v2, latent=16, layers=1/1/1, two images)
[Action-AE] output=${RUN_ROOT_DIR} log=${LOG_FILE}
EOF
printf '[Action-AE] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == true ]]; then
    exit 0
fi
if [[ "${BACKGROUND}" == true ]]; then
    nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
    echo "[Action-AE] started pid=$! log=${LOG_FILE}"
else
    "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi
