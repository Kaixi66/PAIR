# PAIR

PAIR contains Action AE and PAIR bridge training/evaluation code built on top of VLA-Adapter.

## Layout

- `action_ae/`: standalone Action Autoencoder teacher code.
- `VLA-Adapter/`: VLA-Adapter code with integrated PAIR bridge support in `prismatic/models/pair_bridge.py`.
- `kaixi_scripts/`: training and evaluation launch scripts.
- `PAIR (1).pdf`: project plan/introduction document.

## Notes

Large runtime artifacts are intentionally not tracked:

- pretrained model weights
- training checkpoints
- datasets
- logs, rollouts, and cache directories

Default scripts expect local paths such as:

- `/data/kaixi/dataset/libero`
- `/umd-datapool/kaixi/PAIR/checkpoints`
- `/umd-datapool/kaixi/PAIR/action_ae_runs`

Adjust paths in `kaixi_scripts/*.sh` before running on a different machine.

## Common Commands

Train Action AE:

```bash
MAX_STEPS=2 BATCH_SIZE=4 WANDB_MODE=disabled ./kaixi_scripts/train_action_ae.sh
```

Train PAIR bridge:

```bash
MAX_STEPS=2 BATCH_SIZE=1 GPUS=0 WANDB_MODE=disabled ./kaixi_scripts/train_pair_bridge.sh
```

Evaluate LIBERO checkpoint:

```bash
PRETRAINED_CHECKPOINT=/path/to/checkpoint TASK_SUITES=libero_spatial ./kaixi_scripts/eval_libero_vla_adapter.sh
```
