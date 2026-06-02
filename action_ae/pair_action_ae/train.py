"""Train the PAIR Action Autoencoder teacher."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .checkpoint import (
    append_jsonl,
    save_encoder_checkpoint,
    save_json,
    save_training_checkpoint,
)
from .data import ActionDataConfig, make_action_iterables
from .model import ActionAEConfig, ActionTransformerAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PAIR ActionTransformerAE.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--run_root_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}")
    return config


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if args.data_root_dir is not None:
        updates.setdefault("data", {})["data_root_dir"] = args.data_root_dir
    if args.run_root_dir is not None:
        updates.setdefault("run", {})["run_root_dir"] = str(args.run_root_dir)
    if args.run_name is not None:
        updates.setdefault("run", {})["run_name"] = args.run_name
    if args.batch_size is not None:
        updates.setdefault("training", {})["batch_size"] = args.batch_size
    if args.max_steps is not None:
        updates.setdefault("training", {})["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        updates.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        updates.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.log_every is not None:
        updates.setdefault("training", {})["log_every"] = args.log_every
    if args.eval_every is not None:
        updates.setdefault("training", {})["eval_every"] = args.eval_every
    if args.save_every is not None:
        updates.setdefault("training", {})["save_every"] = args.save_every
    if args.eval_batches is not None:
        updates.setdefault("training", {})["eval_batches"] = args.eval_batches
    if args.seed is not None:
        updates.setdefault("training", {})["seed"] = args.seed
    if args.device is not None:
        updates.setdefault("training", {})["device"] = args.device
    if args.wandb_entity is not None:
        updates.setdefault("wandb", {})["entity"] = args.wandb_entity
    if args.wandb_project is not None:
        updates.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_mode is not None:
        updates.setdefault("wandb", {})["mode"] = args.wandb_mode
    return deep_update(config, updates)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(actions: torch.Tensor, recon: torch.Tensor, latents: torch.Tensor, prefix: str) -> Dict[str, float]:
    error = (recon - actions).abs().float()
    metrics: Dict[str, float] = {
        f"{prefix}/l1": error.mean().item(),
        f"{prefix}/current_l1": error[:, 0].mean().item(),
        f"{prefix}/future_l1": error[:, 1:].mean().item(),
        f"{prefix}/latent_mean": latents.float().mean().item(),
        f"{prefix}/latent_std": latents.float().std(unbiased=False).item(),
        f"{prefix}/latent_norm": latents.float().norm(dim=-1).mean().item(),
    }
    per_dim_l1 = error.mean(dim=(0, 1))
    for dim, value in enumerate(per_dim_l1):
        metrics[f"{prefix}/dim_{dim}_l1"] = value.item()
    return metrics


@torch.no_grad()
def evaluate(
    *,
    model: ActionTransformerAE,
    eval_iterable: Iterable[Dict[str, Any]],
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0

    for batch in eval_iterable:
        actions = batch["actions"].to(device, non_blocking=True)
        recon, latents = model(actions)
        batch_metrics = compute_metrics(actions, recon, latents, "eval")
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if count >= max_batches:
            break

    if count == 0:
        raise RuntimeError("Evaluation iterable produced no batches.")

    return {key: value / count for key, value in totals.items()}


def init_wandb(config: Dict[str, Any], run_name: str):
    wandb_config = config.get("wandb", {})
    mode = os.environ.get("WANDB_MODE", wandb_config.get("mode", "online"))
    os.environ["WANDB_MODE"] = mode

    try:
        import wandb
    except ImportError:
        if mode == "disabled":
            return None
        raise

    return wandb.init(
        entity=wandb_config.get("entity", "kaixi-university-of-maryland"),
        project=wandb_config.get("project", "PAIR"),
        name=run_name,
        mode=mode,
        config=config,
    )


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    training_cfg = config.setdefault("training", {})
    run_cfg = config.setdefault("run", {})
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})

    seed = int(training_cfg.get("seed", 7))
    set_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_cfg.get("run_name") or f"action_ae_libero_all_{timestamp}"
    run_root_dir = Path(run_cfg.get("run_root_dir", "/umd-datapool/kaixi/PAIR/action_ae_runs"))
    run_dir = run_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    wandb_run = init_wandb(config, run_name)

    action_ae_config = ActionAEConfig.from_dict(model_cfg)
    model = ActionTransformerAE(action_ae_config)

    device_name = training_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)

    data_config = ActionDataConfig(
        data_root_dir=data_cfg.get("data_root_dir", "/data/kaixi/dataset/libero"),
        mixture=data_cfg.get("mixture", "libero_4_task_suites_no_noops"),
        train_split=data_cfg.get("train_split", "train[:95%]"),
        eval_split=data_cfg.get("eval_split", "train[95%:]"),
        batch_size=int(training_cfg.get("batch_size", data_cfg.get("batch_size", 1024))),
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 256_000)),
        traj_transform_threads=data_cfg.get("traj_transform_threads", 4),
        traj_read_threads=data_cfg.get("traj_read_threads", 4),
        balance_weights=bool(data_cfg.get("balance_weights", True)),
        include_dataset_name=bool(data_cfg.get("include_dataset_name", True)),
    )
    train_iterable, eval_iterable, dataset_statistics = make_action_iterables(data_config)
    save_json(run_dir / "dataset_statistics.json", dataset_statistics)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 100_000))
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        max_steps=max_steps,
    )

    log_every = int(training_cfg.get("log_every", 100))
    eval_every = int(training_cfg.get("eval_every", 5000))
    save_every = int(training_cfg.get("save_every", 10000))
    eval_batches = int(training_cfg.get("eval_batches", 20))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))

    best_eval_l1 = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    train_iterator = iter(train_iterable)
    start_time = time.time()

    print(f"[action_ae] run_dir: {run_dir}")
    print(f"[action_ae] device: {device}")
    print(f"[action_ae] max_steps: {max_steps}")
    print(f"[action_ae] batch_size: {data_config.batch_size}")

    for step in range(1, max_steps + 1):
        model.train()
        batch = next(train_iterator)
        actions = batch["actions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon, latents = model(actions)
        loss = F.l1_loss(recon, actions)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            metrics = compute_metrics(actions.detach(), recon.detach(), latents.detach(), "train")
            metrics.update(
                {
                    "step": step,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "time/elapsed_sec": time.time() - start_time,
                }
            )
            append_jsonl(metrics_path, metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            print(f"[action_ae] step={step} loss={loss.item():.6f} lr={lr:.3e}")

        if step % eval_every == 0 or step == max_steps:
            eval_metrics = evaluate(
                model=model,
                eval_iterable=eval_iterable,
                device=device,
                max_batches=eval_batches,
            )
            eval_metrics["step"] = step
            append_jsonl(metrics_path, eval_metrics)
            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=step)
            eval_l1 = eval_metrics["eval/l1"]
            print(f"[action_ae] eval step={step} l1={eval_l1:.6f}")

            if eval_l1 < best_eval_l1:
                best_eval_l1 = eval_l1
                save_training_checkpoint(
                    path=run_dir / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval_l1=best_eval_l1,
                    config=config,
                )
                save_encoder_checkpoint(
                    path=run_dir / "encoder.pt",
                    encoder=model.encoder,
                    config=action_ae_config,
                    metadata={"step": step, "eval_l1": best_eval_l1, "run_name": run_name},
                )

        if step % save_every == 0 or step == max_steps:
            save_training_checkpoint(
                path=run_dir / "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_eval_l1=best_eval_l1,
                config=config,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
