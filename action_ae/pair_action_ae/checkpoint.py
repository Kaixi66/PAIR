"""Checkpoint helpers for Action AE training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .model import ActionAEConfig, ActionEncoder, ActionPerceptionAEConfig, ActionPerceptionEncoder


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


def _encoder_config_to_dict(config: Any) -> Dict[str, object]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported encoder config type: {type(config).__name__}")


def save_encoder_checkpoint(
    *,
    path: str | Path,
    encoder: torch.nn.Module,
    config: Any,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = "ActionPerceptionEncoder" if getattr(encoder, "requires_perception", False) else "ActionEncoder"
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("requires_perception", bool(getattr(encoder, "requires_perception", False)))
    payload_metadata.setdefault("latent_dim", int(getattr(encoder, "latent_dim", getattr(config, "latent_dim", 0))))
    torch.save(
        {
            "model_type": model_type,
            "model_config": _encoder_config_to_dict(config),
            "state_dict": encoder.state_dict(),
            "metadata": to_jsonable(payload_metadata),
        },
        path,
    )


def _migrate_legacy_perception_encoder_state(
    config_dict: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Map the original single-cross-attn v2 encoder keys to cross_blocks.0."""
    if not any(key.startswith("cross_attn.") for key in state_dict):
        return config_dict, state_dict

    migrated_config = dict(config_dict)
    migrated_config.setdefault("perception_layers", 1)
    prefix_map = {
        "cross_attn_norm.": "cross_blocks.0.query_norm.",
        "perception_norm.": "cross_blocks.0.memory_norm.",
        "cross_attn.": "cross_blocks.0.cross_attn.",
        "fuse_norm.": "cross_blocks.0.mlp_norm.",
        "fuse_mlp.": "cross_blocks.0.mlp.",
    }
    migrated_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        migrated_state[new_key] = value
    return migrated_config, migrated_state


def load_encoder_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ActionEncoder:
    payload = torch.load(path, map_location=map_location)
    model_type = payload.get("model_type", "ActionEncoder")
    if model_type == "ActionPerceptionEncoder":
        config_dict, state_dict = _migrate_legacy_perception_encoder_state(
            dict(payload["model_config"]),
            payload["state_dict"],
        )
        config = ActionPerceptionAEConfig.from_dict(config_dict)
        encoder = ActionPerceptionEncoder(config)
    elif model_type == "ActionEncoder":
        config = ActionAEConfig.from_dict(payload["model_config"])
        encoder = ActionEncoder(config)
        state_dict = payload["state_dict"]
    else:
        raise ValueError(f"Unsupported action encoder checkpoint model_type={model_type!r}")
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder
