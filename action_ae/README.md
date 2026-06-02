# PAIR Action AE

This directory contains the standalone Action Autoencoder teacher for PAIR.
It trains only on normalized action chunks and does not modify VLA-Adapter code.

## Layout

```text
action_ae/
  configs/libero_all.yaml
  pair_action_ae/
    data.py
    model.py
    train.py
    checkpoint.py
  tests/
```

## Model

`ActionTransformerAE` reconstructs normalized LIBERO action chunks:

```text
[B, 8, 7] -> encoder -> [B, 8, 16] -> decoder -> [B, 8, 7]
```

The saved `encoder.pt` is the Stage 2 teacher artifact. It contains an
`ActionEncoder` state dict and the model config needed to instantiate it.

## Training

Use the workspace launcher from the PAIR root:

```bash
MAX_STEPS=2 BATCH_SIZE=4 WANDB_MODE=disabled /data/kaixi/PAIR/kaixi_scripts/train_action_ae.sh
```

Default full training uses all four LIBERO suites through
`libero_4_task_suites_no_noops`, `BOUNDS_Q99` action normalization, and online
WandB logging to `kaixi-university-of-maryland/PAIR`.

Outputs are written under:

```text
/umd-datapool/kaixi/PAIR/action_ae_runs/<run_name>/
```
