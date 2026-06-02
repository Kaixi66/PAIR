"""Checkpoint helpers for Action AE training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .model import ActionAEConfig, ActionEncoder


def to_jsonable(value: Any) -> Any:
    """Convert nested numpy/torch values into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def save_training_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_eval_l1: float,
    config: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_eval_l1": best_eval_l1,
            "config": to_jsonable(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def save_encoder_checkpoint(
    *,
    path: str | Path,
    encoder: ActionEncoder,
    config: ActionAEConfig,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "ActionEncoder",
            "model_config": config.to_dict(),
            "state_dict": encoder.state_dict(),
            "metadata": to_jsonable(metadata or {}),
        },
        path,
    )


def load_encoder_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ActionEncoder:
    payload = torch.load(path, map_location=map_location)
    config = ActionAEConfig.from_dict(payload["model_config"])
    encoder = ActionEncoder(config)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder
