"""PAIR bridge modules integrated into VLA-Adapter."""

from __future__ import annotations

import math
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
    bridge_mlp_dim: int = 1024
    init_from_latent: bool = True
    init_gate_mode: str = "learnable"
    init_gate_value: float = 0.05
    init_gate_granularity: str = "per_step"
    input_dependent_gate: bool = True
    gate_activation: str = "sigmoid"
    init_gate_value_is_actual: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PairBridgeConfig":
        is_legacy_gate_config = "gate_activation" not in data
        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: value for key, value in data.items() if key in allowed}
        if "bridge_mlp_dim" not in values:
            values["bridge_mlp_dim"] = 0
        if "init_from_latent" not in values:
            values["init_from_latent"] = False
        if "init_gate_granularity" not in values:
            values["init_gate_granularity"] = "scalar"
        if "input_dependent_gate" not in values:
            values["input_dependent_gate"] = False
        if "gate_activation" not in values:
            values["gate_activation"] = "tanh"
        if "init_gate_value_is_actual" not in values:
            values["init_gate_value_is_actual"] = not is_legacy_gate_config
        return cls(**values)


@dataclass(frozen=True)
class PairBridgeOutput:
    action_init: Tensor
    bridge_tokens: Tensor
    z_align: Tensor
    action_init_delta: Tensor
    init_gate: Tensor
    init_gate_raw: Tensor


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
        if self.config.bridge_mlp_dim > 0:
            self.bridge_mlp_norm = nn.LayerNorm(self.config.bridge_dim)
            self.bridge_mlp = nn.Sequential(
                nn.Linear(self.config.bridge_dim, self.config.bridge_mlp_dim, bias=True),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.bridge_mlp_dim, self.config.bridge_dim, bias=True),
            )
        else:
            self.bridge_mlp_norm = None
            self.bridge_mlp = None
        if self.config.init_from_latent:
            self.latent_proj = nn.Linear(self.config.bridge_dim, self.config.latent_dim, bias=True)
            self.latent_to_step_init = nn.Sequential(
                nn.LayerNorm(self.config.latent_dim),
                nn.Linear(self.config.latent_dim, self.config.bridge_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.config.bridge_dim, self.config.llm_dim, bias=True),
            )
            self.align_proj = None
            self.init_proj = None
        else:
            self.latent_proj = None
            self.latent_to_step_init = None
            self.align_proj = nn.Linear(self.config.bridge_dim, self.config.latent_dim, bias=True)
            self.init_proj = nn.Linear(self.config.bridge_dim, self.config.llm_dim, bias=True)

        self.slot_scale = nn.Parameter(torch.ones(self.config.action_dim, self.config.llm_dim))
        self.uses_input_dependent_gate = (
            self.config.input_dependent_gate
            and self.config.init_gate_mode == "learnable"
            and self.config.init_gate_granularity == "per_step"
        )
        if self.uses_input_dependent_gate:
            self.gate_norm = nn.LayerNorm(self.config.bridge_dim)
            self.gate_proj = nn.Linear(self.config.bridge_dim, 1, bias=True)
        else:
            self.gate_norm = None
            self.gate_proj = None
            init_gate_value = (
                float(self.config.init_gate_value)
                if self.config.init_gate_mode == "fixed"
                else self._initial_gate_raw_value()
            )
            if self.config.init_gate_granularity == "scalar":
                init_gate = torch.full((), init_gate_value)
            elif self.config.init_gate_granularity == "per_step":
                init_gate = torch.full((self.config.horizon,), init_gate_value)
            else:
                raise ValueError(
                    "Unsupported init_gate_granularity="
                    f"{self.config.init_gate_granularity!r}; expected 'scalar' or 'per_step'."
                )
            if self.config.init_gate_mode == "learnable":
                self.init_gate = nn.Parameter(init_gate)
            elif self.config.init_gate_mode == "fixed":
                self.register_buffer("init_gate", init_gate)
            else:
                raise ValueError(
                    f"Unsupported init_gate_mode={self.config.init_gate_mode!r}; expected 'learnable' or 'fixed'."
                )

        self.reset_parameters()

    def _initial_gate_raw_value(self) -> float:
        value = float(self.config.init_gate_value)
        activation = self.config.gate_activation
        if not self.config.init_gate_value_is_actual:
            return value
        if activation == "sigmoid":
            if not 0.0 < value < 1.0:
                raise ValueError("Sigmoid gate actual init value must be in (0, 1).")
            return math.log(value / (1.0 - value))
        if activation == "tanh":
            if not -1.0 < value < 1.0:
                raise ValueError("Tanh gate actual init value must be in (-1, 1).")
            return math.atanh(value)
        raise ValueError(f"Unsupported gate_activation={activation!r}; expected 'sigmoid' or 'tanh'.")

    def _activate_gate(self, gate_raw: Tensor) -> Tensor:
        if self.config.gate_activation == "sigmoid":
            return torch.sigmoid(gate_raw)
        if self.config.gate_activation == "tanh":
            return torch.tanh(gate_raw)
        raise ValueError(
            f"Unsupported gate_activation={self.config.gate_activation!r}; expected 'sigmoid' or 'tanh'."
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bridge_queries, mean=0.0, std=0.02)
        if self.uses_input_dependent_gate:
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, self._initial_gate_raw_value())

    def keep_high_precision_params(self) -> None:
        """Keep small scale/gate parameters in fp32 after bulk bf16 conversion."""
        self.slot_scale = nn.Parameter(
            self.slot_scale.detach().float(),
            requires_grad=self.slot_scale.requires_grad,
        )
        if self.uses_input_dependent_gate:
            self.gate_norm.float()
            self.gate_proj.float()
        elif isinstance(self.init_gate, nn.Parameter):
            self.init_gate = nn.Parameter(self.init_gate.detach().float(), requires_grad=self.init_gate.requires_grad)
        else:
            self.init_gate = self.init_gate.detach().float()

    def keep_init_gate_fp32(self) -> None:
        self.keep_high_precision_params()

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
        if self.bridge_mlp is not None:
            bridge_tokens = bridge_tokens + self.bridge_mlp(self.bridge_mlp_norm(bridge_tokens))

        if self.config.init_from_latent:
            z_align = self.latent_proj(bridge_tokens)
            step_init = self.latent_to_step_init(z_align)
        else:
            z_align = self.align_proj(bridge_tokens)
            step_init = self.init_proj(bridge_tokens)
        per_dim_init = step_init.unsqueeze(2) * self.slot_scale.to(dtype=step_init.dtype).unsqueeze(0).unsqueeze(0)
        action_init_delta = per_dim_init.reshape(batch_size, expected_slots, self.config.llm_dim)
        if self.uses_input_dependent_gate:
            gate_raw = self.gate_proj(self.gate_norm(bridge_tokens.float())).squeeze(-1)
            gate = self._activate_gate(gate_raw).to(dtype=base_action_init.dtype)
            gated_delta = (
                gate.view(batch_size, self.config.horizon, 1, 1)
                * per_dim_init.to(dtype=base_action_init.dtype)
            ).reshape(batch_size, expected_slots, self.config.llm_dim)
        else:
            gate_raw = self.init_gate
            if self.config.init_gate_mode == "fixed" and self.config.init_gate_value_is_actual:
                gate = self.init_gate.to(dtype=base_action_init.dtype)
            else:
                gate = self._activate_gate(self.init_gate).to(dtype=base_action_init.dtype)
            if gate.ndim == 0:
                gated_delta = gate * action_init_delta.to(dtype=base_action_init.dtype)
            else:
                gated_delta = (
                    gate.view(1, self.config.horizon, 1, 1)
                    * per_dim_init.to(dtype=base_action_init.dtype)
                ).reshape(batch_size, expected_slots, self.config.llm_dim)
        action_init = base_action_init + gated_delta

        return PairBridgeOutput(
            action_init=action_init,
            bridge_tokens=bridge_tokens,
            z_align=z_align,
            action_init_delta=action_init_delta,
            init_gate=gate,
            init_gate_raw=gate_raw,
        )


def cosine_alignment_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Alignment tensors must have same shape, got {predicted.shape} and {target.shape}")
    cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1)
    return (1.0 - cosine).mean()


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
    state_dict = dict(payload["state_dict"])
    state_dict.pop("slot_bias", None)
    state_dict.pop("module.slot_bias", None)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_frozen_action_encoder(path: str | Path, *, device: torch.device | int | str) -> nn.Module:
    from pair_action_ae.checkpoint import load_encoder_checkpoint

    encoder = load_encoder_checkpoint(path, map_location="cpu")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.to(device)
