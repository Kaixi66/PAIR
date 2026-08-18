#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# press-buttons | place-corn-in-bowl | put-cube-in-cup | stack-cups | all
REAL_TASK="${REAL_TASK:-}"
STAGE="${STAGE:-all}" # action_ae | pair_bridge | all
REALWORLD_DATA_DIR="${REALWORLD_DATA_DIR:-${PAIR_DATA_DIR}/realworld}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
SEED="${SEED:-7}"
DRY_RUN="${DRY_RUN:-false}"
VALIDATE_DATA="${VALIDATE_DATA:-true}"

AE_BATCH_SIZE="${AE_BATCH_SIZE:-8}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-3e-4}"
AE_MAX_STEPS="${AE_MAX_STEPS:-50000}"
AE_WEIGHT_DECAY="${AE_WEIGHT_DECAY:-1e-4}"

PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-8}"
PAIR_GRAD_ACCUMULATION_STEPS="${PAIR_GRAD_ACCUMULATION_STEPS:-1}"
PAIR_LEARNING_RATE="${PAIR_LEARNING_RATE:-2e-4}"
PAIR_LR_SCHEDULER="${PAIR_LR_SCHEDULER:-cosine}"
PAIR_LORA_RANK="${PAIR_LORA_RANK:-64}"
PAIR_MAX_STEPS="${PAIR_MAX_STEPS:-30000}"
PAIR_ALIGN_WEIGHT="${PAIR_ALIGN_WEIGHT:-0.1}"
PAIR_SAVE_FREQ="${PAIR_SAVE_FREQ:-${PAIR_MAX_STEPS}}"
PAIR_MERGE_LORA_DURING_TRAINING="${PAIR_MERGE_LORA_DURING_TRAINING:-True}"

WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
WANDB_MODE="${WANDB_MODE:-online}"

TFDS_DATASET_NAME="utokyo_xarm_pick_and_place_converted_externally_to_rlds"
DATASET_NAME="uf850_vr_teleop_rlds"
SPLIT_ROOT="${SPLIT_ROOT:-${PAIR_RUN_ROOT_DIR}/uf850_splits}"
CONTRACT_ROOT="${CONTRACT_ROOT:-${PAIR_RUN_ROOT_DIR}/uf850_contracts}"
ACTION_AE_RUN_ROOT="${ACTION_AE_RUN_ROOT:-${PAIR_RUN_ROOT_DIR}/action_ae}"

usage() {
    echo "Usage: REAL_TASK=press-buttons|place-corn-in-bowl|put-cube-in-cup|stack-cups|all [STAGE=action_ae|pair_bridge|all] $0" >&2
}

[[ -n "${REAL_TASK}" ]] || { usage; exit 2; }
case "${REAL_TASK}" in press-buttons|place-corn-in-bowl|put-cube-in-cup|stack-cups|all) ;; *) usage; exit 2 ;; esac
case "${STAGE}" in action_ae|pair_bridge|all) ;; *) usage; exit 2 ;; esac

task_data_root() {
    printf '%s/uf850-vr-teleop-%s-rlds-noop-filtered' "${REALWORLD_DATA_DIR}" "$1"
}

run_task() {
    local task="$1"
    local tag="${task//-/_}"
    local data_root split_file contract_file ae_name pair_name encoder_path
    data_root="$(task_data_root "${task}")"
    split_file="${SPLIT_ROOT}/${task}.json"
    contract_file="${CONTRACT_ROOT}/${task}.json"
    ae_name="ae_uf850_${tag}"
    pair_name="PAIR_uf850_${tag}"
    encoder_path="${ACTION_AE_RUN_ROOT}/${ae_name}/encoder_best.pt"

    if [[ "${DRY_RUN}" != true ]]; then
        [[ -d "${data_root}/${TFDS_DATASET_NAME}/1.0.1" ]] || {
            echo "Missing UF850 RLDS dataset: ${data_root}" >&2
            return 1
        }
        if [[ "${VALIDATE_DATA}" == true || ! -f "${split_file}" || ! -f "${contract_file}" ]]; then
            VLA_ROBOT_PLATFORM=UF850 "${PAIR_PYTHON}" "${SCRIPT_DIR}/validate_uf850_pair_data.py" \
                --dataset-root "${data_root}" --task "${task}" \
                --split-output "${split_file}" --contract-output "${contract_file}" --seed "${SEED}"
        fi
    fi

    echo "[UF850] task=${task} stage=${STAGE} data=${data_root}"
    echo "[UF850] proprio=[6] action=[8,7] cameras=primary+wrist"

    if [[ "${STAGE}" == action_ae || "${STAGE}" == all ]]; then
        VLA_ROBOT_PLATFORM=UF850 \
        DATASET_NAME="${DATASET_NAME}" DATA_ROOT_DIR="${data_root}" \
        CONFIG_PATH="${PAIR_WORKSPACE_DIR}/action_ae/configs/uf850_v2_perception.yaml" \
        RUN_ROOT_DIR="${ACTION_AE_RUN_ROOT}" EXP_NAME="${ae_name}" \
        GPUS="${GPUS}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
        BATCH_SIZE="${AE_BATCH_SIZE}" LEARNING_RATE="${AE_LEARNING_RATE}" \
        WEIGHT_DECAY="${AE_WEIGHT_DECAY}" MAX_STEPS="${AE_MAX_STEPS}" \
        EPISODE_SPLIT_FILE="${split_file}" DATA_CONTRACT_FILE="${contract_file}" \
        SEED="${SEED}" WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_MODE="${WANDB_MODE}" DRY_RUN="${DRY_RUN}" \
        bash "${SCRIPT_DIR}/train_action_ae.sh"
    fi

    if [[ "${STAGE}" == pair_bridge || "${STAGE}" == all ]]; then
        if [[ "${DRY_RUN}" != true && ! -f "${encoder_path}" ]]; then
            echo "Missing Action-AE encoder: ${encoder_path}" >&2
            return 1
        fi
        VLA_ROBOT_PLATFORM=UF850 \
        DATASET_NAME="${DATASET_NAME}" TFDS_DATASET_NAME="${TFDS_DATASET_NAME}" DATA_ROOT_DIR="${data_root}" \
        PAIR_ACTION_AE_ENCODER_PATH="${encoder_path}" EXP_NAME="${pair_name}" \
        GPUS="${GPUS}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
        BATCH_SIZE="${PAIR_BATCH_SIZE}" GRAD_ACCUMULATION_STEPS="${PAIR_GRAD_ACCUMULATION_STEPS}" \
        LEARNING_RATE="${PAIR_LEARNING_RATE}" LR_SCHEDULER="${PAIR_LR_SCHEDULER}" \
        LORA_RANK="${PAIR_LORA_RANK}" MAX_STEPS="${PAIR_MAX_STEPS}" SAVE_FREQ="${PAIR_SAVE_FREQ}" \
        PAIR_ALIGN_WEIGHT="${PAIR_ALIGN_WEIGHT}" \
        MERGE_LORA_DURING_TRAINING="${PAIR_MERGE_LORA_DURING_TRAINING}" \
        EPISODE_SPLIT_FILE="${split_file}" DATA_CONTRACT_FILE="${contract_file}" \
        WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" WANDB_MODE="${WANDB_MODE}" \
        DRY_RUN="${DRY_RUN}" \
        bash "${SCRIPT_DIR}/train_pair_bridge.sh"
    fi
}

if [[ "${REAL_TASK}" == all ]]; then
    for task in press-buttons place-corn-in-bowl put-cube-in-cup stack-cups; do
        run_task "${task}"
    done
else
    run_task "${REAL_TASK}"
fi
