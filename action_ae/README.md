# PAIR Action AE

This directory contains the standalone Action Autoencoder teachers for PAIR.
The v1 teacher is action-only; the v2 teacher is perception-conditioned and
uses a frozen VLA backbone to extract `[V0; T0]` tokens during AE training.

## Layout

```text
action_ae/
  configs/libero_all.yaml
  configs/libero_all_v2_perception.yaml
  pair_action_ae/
    data.py
    model.py
    perception.py
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

`ActionPerceptionTransformerAE` reconstructs clean actions from corrupted
actions cross-attended to frozen VLA perception tokens:

```text
actions [B, 8, 7] + perception [B, N, D] -> encoder -> [B, 8, 8] -> decoder -> [B, 8, 7]
```

Its `encoder.pt` metadata sets `requires_perception=true`, so PAIR Stage 2 can
call the teacher with `(actions, perception_tokens, perception_mask)`.

## Training

Use the workspace launcher from the PAIR root:

```bash
MAX_STEPS=2 BATCH_SIZE=4 WANDB_MODE=disabled /data/kaixi/PAIR/kaixi_scripts/train_action_ae.sh
```

For the perception-conditioned v2 teacher:

```bash
AE_VERSION=v2 MAX_STEPS=2 BATCH_SIZE=1 EVAL_BATCHES=1 WANDB_MODE=disabled /data/kaixi/PAIR/kaixi_scripts/train_action_ae.sh
```

Default full training uses all four LIBERO suites through
`libero_4_task_suites_no_noops`, `BOUNDS_Q99` action normalization, and online
WandB logging to `kaixi-university-of-maryland/PAIR`.

Outputs are written under:

```text
/umd-datapool/kaixi/PAIR/action_ae_runs/<run_name>/
```
