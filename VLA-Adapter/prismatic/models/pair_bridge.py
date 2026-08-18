"""PAIR bridge modules integrated into VLA-Adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import IGNORE_INDEX


@dataclass(frozen=True)
class PairBridgeConfig:
    """Shape metadata for the fixed PAIR base architecture.

    Architecture choices are intentionally class constants rather than constructor
    options.  The public implementation has one canonical model definition.
    """

    llm_dim: int = 4096
    latent_dim: int = 16
    horizon: int = 8
    action_dim: int = 7

    bridge_dim: ClassVar[int] = 1024
    num_heads: ClassVar[int] = 8
    dropout: ClassVar[float] = 0.0
    bridge_mlp_dim: ClassVar[int] = 4096
    init_mlp_dim: ClassVar[int] = 4096

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PairBridgeConfig":
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)


@dataclass(frozen=True)
class PairBridgeOutput:
    action_init: Tensor
    bridge_tokens: Tensor
    z_align: Tensor
    action_init_delta: Tensor
    init_gate: Tensor
    init_gate_raw: Tensor


def build_pair_perception_tokens(
    *,
    hidden_state: Tensor,
    num_patches: int,
    attention_mask: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    num_prompt_tokens: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Build `[V0;T0]` perception tokens with one train/eval masking rule.

    Training can pass `labels`, letting the helper drop action-token states and padding.
    Inference has no labels, so it passes the known prompt length and masks prompt
    padding from `attention_mask` when available.
    """
    if hidden_state.ndim != 3:
        raise ValueError(f"Expected hidden_state [B,S,D], got {tuple(hidden_state.shape)}")
    if num_patches < 0 or num_patches > hidden_state.shape[1]:
        raise ValueError(f"Invalid num_patches={num_patches} for hidden_state length {hidden_state.shape[1]}")

    vision_tokens = hidden_state[:, :num_patches, :]
    vision_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.bool, device=hidden_state.device)

    if labels is not None:
        if attention_mask is None:
            raise ValueError("attention_mask is required when labels are provided.")
        text_tokens = hidden_state[:, num_patches:-1, :]
        text_labels = labels[:, 1:].to(hidden_state.device)
        text_attention_mask = attention_mask[:, 1:].to(hidden_state.device)
        min_len = min(text_tokens.shape[1], text_labels.shape[1], text_attention_mask.shape[1])
        text_tokens = text_tokens[:, :min_len, :]
        text_labels = text_labels[:, :min_len]
        text_attention_mask = text_attention_mask[:, :min_len]

        action_mask = get_current_action_mask(text_labels) | get_next_actions_mask(text_labels)
        prompt_mask = text_attention_mask.bool() & (text_labels == IGNORE_INDEX) & ~action_mask
    else:
        if num_prompt_tokens is None:
            raise ValueError("num_prompt_tokens is required when labels are not provided.")
        prompt_end = num_patches + int(num_prompt_tokens)
        text_tokens = hidden_state[:, num_patches:prompt_end, :]
        if attention_mask is None:
            prompt_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=hidden_state.device)
        else:
            prompt_attention_mask = attention_mask[:, : int(num_prompt_tokens)].to(hidden_state.device)
            min_len = min(text_tokens.shape[1], prompt_attention_mask.shape[1])
            text_tokens = text_tokens[:, :min_len, :]
            prompt_mask = prompt_attention_mask[:, :min_len].bool()

    perception_tokens = torch.cat([vision_tokens, text_tokens], dim=1)
    perception_mask = torch.cat([vision_mask, prompt_mask], dim=1)
    return perception_tokens, perception_mask


class BridgeCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block matching the perception-conditioned AE style."""

    def __init__(self, *, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim, bias=True),
        )

    def forward(self, x: Tensor, memory: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        memory = self.memory_norm(memory)
        attended, _ = self.cross_attn(
            query=self.query_norm(x),
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class BridgeSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block over the 8 action-step bridge tokens."""

    def __init__(self, *, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.self_attn_norm(x)
        attended, _ = self.self_attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class PairBridge(nn.Module):
    """Cross-attention bridge from initial perception tokens to action hidden states."""

    def __init__(self, config: PairBridgeConfig | None = None) -> None:
        super().__init__()
        self.config = config or PairBridgeConfig()

        self.down_proj = nn.Linear(self.config.llm_dim, self.config.bridge_dim, bias=True)
        self.bridge_queries = nn.Parameter(torch.zeros(self.config.horizon, self.config.bridge_dim))
        self.bridge_pos_embed = nn.Parameter(torch.zeros(1, self.config.horizon, self.config.bridge_dim))
        self.cross_block = BridgeCrossAttentionBlock(
            dim=self.config.bridge_dim,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.bridge_mlp_dim,
            dropout=self.config.dropout,
        )
        self.self_block = BridgeSelfAttentionBlock(
            dim=self.config.bridge_dim,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.bridge_mlp_dim,
            dropout=self.config.dropout,
        )
        self.align_proj = nn.Sequential(
            nn.LayerNorm(self.config.bridge_dim),
            nn.Linear(self.config.bridge_dim, self.config.bridge_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.config.bridge_dim, self.config.latent_dim, bias=True),
        )
        self.init_proj = nn.Sequential(
            nn.LayerNorm(self.config.bridge_dim),
            nn.Linear(self.config.bridge_dim, self.config.init_mlp_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.config.init_mlp_dim, self.config.llm_dim, bias=True),
        )

        # Keep the established start-gate parameter names so released base
        # checkpoints remain directly loadable.
        self.gate_norm = nn.LayerNorm(self.config.bridge_dim)
        self.gate_proj = nn.Linear(self.config.bridge_dim, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bridge_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.bridge_pos_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def keep_high_precision_params(self) -> None:
        """Keep small gate parameters in fp32 after bulk bf16 conversion."""
        self.gate_norm.float()
        self.gate_proj.float()

    def keep_init_gate_fp32(self) -> None:
        self.keep_high_precision_params()

    def forward(
        self,
        perception_tokens: Tensor,
        base_action_init: Optional[Tensor] = None,
        perception_mask: Optional[Tensor] = None,
    ) -> PairBridgeOutput:
        if perception_tokens.ndim != 3:
            raise ValueError(f"Expected perception_tokens [B,N,D], got {tuple(perception_tokens.shape)}")
        if base_action_init is not None and base_action_init.ndim != 3:
            raise ValueError(f"Expected base_action_init [B,H,D], got {tuple(base_action_init.shape)}")

        batch_size, _, llm_dim = perception_tokens.shape
        if llm_dim != self.config.llm_dim:
            raise ValueError(f"Expected perception dim {self.config.llm_dim}, got {llm_dim}")
        if base_action_init is not None and base_action_init.shape != (
            batch_size,
            self.config.horizon,
            self.config.llm_dim,
        ):
            raise ValueError(
                "Expected base_action_init shape "
                f"[{batch_size},{self.config.horizon},{self.config.llm_dim}], got {tuple(base_action_init.shape)}"
            )
        if perception_mask is not None and perception_mask.shape != perception_tokens.shape[:2]:
            raise ValueError(
                f"Expected perception_mask shape {tuple(perception_tokens.shape[:2])}, "
                f"got {tuple(perception_mask.shape)}"
            )

        source_tokens = self.down_proj(perception_tokens)
        queries = self.bridge_queries.to(dtype=source_tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.bridge_pos_embed.to(dtype=source_tokens.dtype)
        key_padding_mask = None if perception_mask is None else ~perception_mask.bool()

        bridge_tokens = self.cross_block(queries, source_tokens, key_padding_mask)
        bridge_tokens = self.self_block(bridge_tokens)

        z_align = self.align_proj(bridge_tokens)
        action_init_delta = self.init_proj(bridge_tokens)
        gate_raw = self.gate_proj(self.gate_norm(bridge_tokens.float())).squeeze(-1)
        gate = torch.tanh(gate_raw).to(dtype=action_init_delta.dtype)
        gated_delta = gate.unsqueeze(-1) * action_init_delta

        if base_action_init is None:
            action_init = gated_delta
        else:
            action_init = base_action_init.to(dtype=gated_delta.dtype) + gated_delta

        return PairBridgeOutput(
            action_init=action_init,
            bridge_tokens=bridge_tokens,
            z_align=z_align,
            action_init_delta=action_init_delta,
            init_gate=gate,
            init_gate_raw=gate_raw,
        )


def alignment_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Alignment tensors must have same shape, got {predicted.shape} and {target.shape}")
    predicted_float = predicted.float()
    target_float = target.float()
    cosine = F.cosine_similarity(predicted_float, target_float, dim=-1)
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
    model_config = dict(payload["model_config"])
    state_dict = payload["state_dict"]
    config = PairBridgeConfig.from_dict(model_config)
    model = PairBridge(config)
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
