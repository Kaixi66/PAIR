#!/usr/bin/env bash

# Source this file:
#   source /data/kaixi/PAIR/kaixi_scripts/env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please source this file instead of executing it:"
    echo "  source ${0}"
    exit 1
fi

_pair_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_pair_workspace_dir="$(cd "${_pair_env_dir}/.." && pwd)"
_pair_repo_dir="${_pair_workspace_dir}/VLA-Adapter"

export PAIR_ENV_DIR="${PAIR_ENV_DIR:-${_pair_env_dir}}"
export PAIR_WORKSPACE_DIR="${PAIR_WORKSPACE_DIR:-${_pair_workspace_dir}}"
export PAIR_REPO_DIR="${PAIR_REPO_DIR:-${_pair_repo_dir}}"

export PAIR_RUN_ROOT_DIR="${PAIR_RUN_ROOT_DIR:-/umd-datapool/kaixi/PAIR}"
export PAIR_CACHE_DIR="${PAIR_CACHE_DIR:-${PAIR_WORKSPACE_DIR}/.cache}"
export PAIR_CHECKPOINTS_DIR="${PAIR_CHECKPOINTS_DIR:-${PAIR_RUN_ROOT_DIR}/checkpoints}"
export PAIR_CKPT_DIR="${PAIR_CKPT_DIR:-${PAIR_CHECKPOINTS_DIR}}"
export PAIR_DATA_DIR="${PAIR_DATA_DIR:-/data/kaixi/dataset}"
export PAIR_OUTPUTS_DIR="${PAIR_OUTPUTS_DIR:-${PAIR_REPO_DIR}/outputs}"
export PAIR_LOGS_DIR="${PAIR_LOGS_DIR:-${PAIR_REPO_DIR}/logs}"
export PAIR_PRETRAINED_MODELS_DIR="${PAIR_PRETRAINED_MODELS_DIR:-${PAIR_REPO_DIR}/pretrained_models}"
export PAIR_LIBERO_DIR="${PAIR_LIBERO_DIR:-${PAIR_REPO_DIR}/LIBERO}"
export PAIR_CALVIN_ROOT="${PAIR_CALVIN_ROOT:-${PAIR_REPO_DIR}/calvin}"
export CALVIN_ROOT="${CALVIN_ROOT:-${PAIR_CALVIN_ROOT}}"

_pair_default_python="/data/miniconda3/envs/vla-adapter/bin/python"
if [[ -z "${PAIR_PYTHON:-}" && -x "${_pair_default_python}" ]]; then
    export PAIR_PYTHON="${_pair_default_python}"
else
    export PAIR_PYTHON="${PAIR_PYTHON:-python}"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PAIR_CACHE_DIR}/xdg}"
export HF_HOME="${HF_HOME:-${PAIR_CACHE_DIR}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PAIR_CACHE_DIR}/torch}"
export WANDB_DIR="${WANDB_DIR:-${PAIR_CACHE_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PAIR_CACHE_DIR}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PAIR_CACHE_DIR}/wandb-config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PAIR_CACHE_DIR}/triton}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PAIR_CACHE_DIR}/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PAIR_CACHE_DIR}/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PAIR_CACHE_DIR}/pip}"
export TMPDIR="${TMPDIR:-${PAIR_CACHE_DIR}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"

export PYTHONPATH="${PAIR_REPO_DIR}:${PAIR_LIBERO_DIR}:${PAIR_CALVIN_ROOT}/calvin_env:${PAIR_CALVIN_ROOT}/calvin_models${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PAIR_CACHE_DIR}"
mkdir -p "${PAIR_RUN_ROOT_DIR}"
mkdir -p "${PAIR_CHECKPOINTS_DIR}"
mkdir -p "${PAIR_DATA_DIR}"
mkdir -p "${PAIR_OUTPUTS_DIR}"
mkdir -p "${PAIR_LOGS_DIR}"
mkdir -p "${PAIR_PRETRAINED_MODELS_DIR}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${HF_HOME}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${WANDB_CONFIG_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${NUMBA_CACHE_DIR}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${TMPDIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

pair_env_summary() {
    cat <<EOF
PAIR_ENV_DIR=${PAIR_ENV_DIR}
PAIR_REPO_DIR=${PAIR_REPO_DIR}
PAIR_RUN_ROOT_DIR=${PAIR_RUN_ROOT_DIR}
PAIR_CACHE_DIR=${PAIR_CACHE_DIR}
PAIR_CHECKPOINTS_DIR=${PAIR_CHECKPOINTS_DIR}
PAIR_CKPT_DIR=${PAIR_CKPT_DIR}
PAIR_DATA_DIR=${PAIR_DATA_DIR}
PAIR_OUTPUTS_DIR=${PAIR_OUTPUTS_DIR}
PAIR_LOGS_DIR=${PAIR_LOGS_DIR}
PAIR_PRETRAINED_MODELS_DIR=${PAIR_PRETRAINED_MODELS_DIR}
PAIR_LIBERO_DIR=${PAIR_LIBERO_DIR}
PAIR_CALVIN_ROOT=${PAIR_CALVIN_ROOT}
CALVIN_ROOT=${CALVIN_ROOT}
PAIR_PYTHON=${PAIR_PYTHON}
HF_HOME=${HF_HOME}
TORCH_HOME=${TORCH_HOME}
NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}
WANDB_DIR=${WANDB_DIR}
EOF
}

echo "[pair env] initialized from ${PAIR_ENV_DIR}"
pair_env_summary

unset _pair_env_dir
unset _pair_workspace_dir
unset _pair_repo_dir
unset _pair_default_python
