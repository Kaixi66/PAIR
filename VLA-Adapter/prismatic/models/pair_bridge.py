"""PAIR bridge modules integrated into VLA-Adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class PairBridgeConfig:
    llm_dim: int = 4096
    bridge_dim: int = 512
    latent_dim: int = 16
    horizon: int = 8
    action_dim: int = 7
    num_heads: int = 8
    dropout: float = 0.0
    init_alpha: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PairBridgeConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class PairBridgeOutput:
    action_init: Tensor
    bridge_tokens: Tensor
    z_align: Tensor
    action_init_delta: Tensor
    init_gate: Tensor


class PairBridge(nn.Module):
    """Cross-attention bridge from initial perception tokens to action hidden states."""

    def __init__(self, config: PairBridgeConfig | None = None) -> None:
        super().__init__()
        self.config = config or PairBridgeConfig()

        self.down_proj = nn.Linear(self.config.llm_dim, self.config.bridge_dim, bias=True)
        self.bridge_queries = nn.Parameter(torch.zeros(self.config.horizon, self.config.bridge_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.config.bridge_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.align_proj = nn.Linear(self.config.bridge_dim, self.config.latent_dim, bias=True)
        self.init_proj = nn.Linear(self.config.bridge_dim, self.config.llm_dim, bias=True)

        self.slot_scale = nn.Parameter(torch.ones(self.config.action_dim, self.config.llm_dim))
        self.slot_bias = nn.Parameter(torch.zeros(self.config.action_dim, self.config.llm_dim))
        self.init_gate = nn.Parameter(torch.zeros(()))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bridge_queries, mean=0.0, std=0.02)

    def forward(
        self,
        perception_tokens: Tensor,
        base_action_init: Tensor,
        perception_mask: Optional[Tensor] = None,
    ) -> PairBridgeOutput:
        if perception_tokens.ndim != 3:
            raise ValueError(f"Expected perception_tokens [B,N,D], got {tuple(perception_tokens.shape)}")
        if base_action_init.ndim != 3:
            raise ValueError(f"Expected base_action_init [B,H*A,D], got {tuple(base_action_init.shape)}")

        batch_size, _, llm_dim = perception_tokens.shape
        expected_slots = self.config.horizon * self.config.action_dim
        if llm_dim != self.config.llm_dim:
            raise ValueError(f"Expected perception dim {self.config.llm_dim}, got {llm_dim}")
        if base_action_init.shape != (batch_size, expected_slots, self.config.llm_dim):
            raise ValueError(
                "Expected base_action_init shape "
                f"[{batch_size},{expected_slots},{self.config.llm_dim}], got {tuple(base_action_init.shape)}"
            )
        if perception_mask is not None and perception_mask.shape != perception_tokens.shape[:2]:
            raise ValueError(
                f"Expected perception_mask shape {tuple(perception_tokens.shape[:2])}, "
                f"got {tuple(perception_mask.shape)}"
            )

        source_tokens = self.down_proj(perception_tokens)
        queries = self.bridge_queries.to(dtype=source_tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None if perception_mask is None else ~perception_mask.bool()

        bridge_tokens, _ = self.cross_attn(
            query=queries,
            key=source_tokens,
            value=source_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        z_align = self.align_proj(bridge_tokens)
        z_init = self.init_proj(bridge_tokens)
        per_dim_init = (
            z_init.unsqueeze(2) * self.slot_scale.to(dtype=z_init.dtype).unsqueeze(0).unsqueeze(0)
            + self.slot_bias.to(dtype=z_init.dtype).unsqueeze(0).unsqueeze(0)
        )
        action_init_delta = per_dim_init.reshape(batch_size, expected_slots, self.config.llm_dim)
        gate = torch.tanh(self.init_gate).to(dtype=base_action_init.dtype)
        action_init = base_action_init + float(self.config.init_alpha) * gate * action_init_delta.to(
            dtype=base_action_init.dtype
        )

        return PairBridgeOutput(
            action_init=action_init,
            bridge_tokens=bridge_tokens,
            z_align=z_align,
            action_init_delta=action_init_delta,
            init_gate=gate,
        )


def cosine_alignment_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Alignment tensors must have same shape, got {predicted.shape} and {target.shape}")
    cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1)
    return (1.0 - cosine).mean()


def linear_warmup_weight(step: int, *, max_weight: float, max_steps: int, warmup_ratio: float) -> float:
    if max_weight <= 0:
        return 0.0
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    return float(max_weight) * min(1.0, max(0.0, float(step) / float(warmup_steps)))


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def save_pair_bridge_checkpoint(
    *,
    path: str | Path,
    pair_bridge: nn.Module,
    config: PairBridgeConfig,
    action_ae_encoder_path: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = _unwrap(pair_bridge)
    torch.save(
        {
            "model_type": "PairBridge",
            "model_config": config.to_dict(),
            "action_ae_encoder_path": action_ae_encoder_path,
            "state_dict": module.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_pair_bridge_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> PairBridge:
    payload = torch.load(path, map_location=map_location)
    config = PairBridgeConfig.from_dict(payload["model_config"])
    model = PairBridge(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_frozen_action_encoder(path: str | Path, *, device: torch.device | int | str) -> nn.Module:
    from pair_action_ae.checkpoint import load_encoder_checkpoint

    encoder = load_encoder_checkpoint(path, map_location="cpu")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.to(device)
