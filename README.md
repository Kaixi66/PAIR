# PAIR

PAIR learns a perception-conditioned bridge from VLM features to the Action
Expert latent space. The repository contains the Action Autoencoder teacher,
the PAIR bridge, VLA integration, training launchers, and LIBERO evaluation
code.

The implementation is built on [VLA-Adapter](VLA-Adapter/README.md).

## Canonical architecture

The public code exposes one model definition rather than the architecture
switches used for internal ablations:

- perception-conditioned Action-AE with an 8-step, 7D action chunk;
- 1,024-dimensional PAIR bridge with one cross-attention block and one
  self-attention block;
- separate alignment and Action Expert delta projections;
- input-dependent per-step gate, initialized to zero and activated with `tanh`;
- one PAIR injection after the Action Expert stem and before its first block;
- cosine latent alignment;
- MiniVLM, two RGB inputs, proprioception, FiLM, and LoRA fine-tuning.

Model architecture choices are fixed in code. Launchers only expose data,
compute, optimization, checkpointing, validation, and logging settings.

## Layout

- `action_ae/`: Action-AE model, configs, and tests.
- `VLA-Adapter/prismatic/models/pair_bridge.py`: canonical PAIR bridge.
- `VLA-Adapter/prismatic/models/action_heads.py`: Action Expert integration.
- `VLA-Adapter/vla-scripts/finetune.py`: PAIR training loop.
- `kaixi_scripts/`: environment, training, and evaluation launchers.

## Environment

Create a Python environment with the dependencies documented under
`VLA-Adapter/`, then set paths only when they differ from the repository-local
defaults:

```bash
export PAIR_PYTHON=/path/to/python
export PAIR_DATA_DIR=/path/to/datasets
export PAIR_RUN_ROOT_DIR=/path/to/runs
```

Pretrained MiniVLM files are expected under
`VLA-Adapter/pretrained_models/` unless `VLM_PATH` and `CONFIG_FILE_PATH` are
provided to the launcher.

## Train Action-AE

```bash
DATASET_NAME=libero_10_no_noops \
DATA_ROOT_DIR=/path/to/rlds \
GPUS=0,1,2,3 \
./kaixi_scripts/train_action_ae.sh
```

## Train PAIR

```bash
PAIR_ACTION_AE_ENCODER_PATH=/path/to/encoder_best.pt \
DATASET_NAME=libero_10_no_noops \
DATA_ROOT_DIR=/path/to/rlds \
GPUS=0,1,2,3 \
MAX_STEPS=30000 \
./kaixi_scripts/train_pair_bridge.sh
```

Use `DRY_RUN=true` to print the resolved command without launching training.

For the four UF850 tasks, the two-stage launcher validates the 6D proprio,
8x7 action, dual-camera, language, and episode-split contract before training:

```bash
REAL_TASK=press-buttons \
REALWORLD_DATA_DIR=/path/to/realworld \
GPUS=0,1,2,3 \
./kaixi_scripts/train_pair_realworld.sh
```

## Checkpoints

PAIR checkpoints store shape metadata plus the fixed base model weights. The
released base checkpoints remain compatible because the canonical start gate
keeps the original `gate_norm` and `gate_proj` parameter names.

Large model weights, datasets, runs, logs, and caches are intentionally not
tracked by Git.
