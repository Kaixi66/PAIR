#!/usr/bin/env bash
set -euo pipefail

# PAIR Action AE training launcher. Override any value when launching:
#   MAX_STEPS=2 BATCH_SIZE=4 WANDB_MODE=disabled ./train_action_ae.sh

#########################
# User-facing settings
#########################

GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_STEPS="${MAX_STEPS:-100000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-50000}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SEED="${SEED:-7}"

WANDB_ENTITY="${WANDB_ENTITY:-kaixi-university-of-maryland}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
export WANDB_MODE="${WANDB_MODE:-online}"

EXP_NAME="${EXP_NAME:-ae_libero_1}"
DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

#################
# Internal wiring
#################

CONDA_ENV="${CONDA_ENV:-vla-adapter}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"

PAIR_ROOT="${PAIR_ROOT:-/data/kaixi/PAIR}"
ENV_SH="${ENV_SH:-${PAIR_ROOT}/kaixi_scripts/env.sh}"
ACTION_AE_DIR="${ACTION_AE_DIR:-${PAIR_ROOT}/action_ae}"
VLA_ADAPTER_DIR="${VLA_ADAPTER_DIR:-${PAIR_ROOT}/VLA-Adapter}"
CONFIG_PATH="${CONFIG_PATH:-${ACTION_AE_DIR}/configs/libero_all.yaml}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data/kaixi/dataset/libero}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/umd-datapool/kaixi/PAIR/action_ae_runs}"
LOG_DIR="${LOG_DIR:-${ACTION_AE_DIR}/logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

current_time="${CURRENT_TIME:-$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="action_ae_libero_all_${current_time}"
fi
log_file="${LOG_FILE:-${LOG_DIR}/ActionAE--${EXP_NAME}.log}"

############################
# Environment
############################

source "${ENV_SH}"

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS// /}"
export PYTHONPATH="${ACTION_AE_DIR}:${VLA_ADAPTER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python_bin="${PAIR_PYTHON:-python}"

cmd=(
    "${python_bin}"
    -m pair_action_ae.train
    --config "${CONFIG_PATH}"
    --data_root_dir "${DATA_ROOT_DIR}"
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
)

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${EXTRA_ARGS})
    cmd+=("${extra_args_array[@]}")
fi

cat <<EOF
[train_action_ae] config: ${CONFIG_PATH}
[train_action_ae] data_root: ${DATA_ROOT_DIR}
[train_action_ae] gpus: ${CUDA_VISIBLE_DEVICES}
[train_action_ae] batch_size: ${BATCH_SIZE}
[train_action_ae] max_steps: ${MAX_STEPS}
[train_action_ae] learning_rate: ${LEARNING_RATE}
[train_action_ae] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})
[train_action_ae] exp_name: ${EXP_NAME}
[train_action_ae] run_root: ${RUN_ROOT_DIR}
[train_action_ae] log_file: ${log_file}
EOF

printf '[train_action_ae] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    pid="$!"
    echo "[train_action_ae] started in background: pid=${pid}"
    echo "[train_action_ae] tail log: tail -f ${log_file}"
else
    "${cmd[@]}" 2>&1 | tee "${log_file}"
fi
