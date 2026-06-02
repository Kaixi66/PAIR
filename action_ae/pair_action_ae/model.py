"""Transformer action autoencoder used as the PAIR action-side teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionAEConfig:
    """Architecture settings for the action autoencoder."""

    horizon: int = 8
    action_dim: int = 7
    hidden_dim: int = 64
    latent_dim: int = 16
    encoder_layers: int = 4
    decoder_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ActionAEConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _make_transformer_stack(
    *,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class ActionEncoder(nn.Module):
    """Maps normalized action chunks `[B, H, action_dim]` to `[B, H, latent_dim]`."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.blocks = _make_transformer_stack(
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.latent_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, H, A], got {tuple(actions.shape)}")
        if actions.shape[1:] != (self.config.horizon, self.config.action_dim):
            raise ValueError(
                "Expected actions with trailing shape "
                f"[{self.config.horizon}, {self.config.action_dim}], got {tuple(actions.shape[1:])}"
            )

        x = self.input_proj(actions)
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.blocks(x)
        return self.latent_proj(x)


class ActionDecoder(nn.Module):
    """Reconstructs normalized action chunks from action latents."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.blocks = _make_transformer_stack(
            hidden_dim=config.hidden_dim,
            num_layers=config.decoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.output_proj = nn.Linear(config.hidden_dim, config.action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 3:
            raise ValueError(f"Expected latents with shape [B, H, Z], got {tuple(latents.shape)}")
        if latents.shape[1:] != (self.config.horizon, self.config.latent_dim):
            raise ValueError(
                "Expected latents with trailing shape "
                f"[{self.config.horizon}, {self.config.latent_dim}], got {tuple(latents.shape[1:])}"
            )

        x = self.input_proj(latents)
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.blocks(x)
        return self.output_proj(x)


class ActionTransformerAE(nn.Module):
    """Small action autoencoder for producing frozen action-side latent teachers."""

    def __init__(self, config: ActionAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionAEConfig()
        self.encoder = ActionEncoder(self.config)
        self.decoder = ActionDecoder(self.config)

    def encode(self, actions: Tensor) -> Tensor:
        return self.encoder(actions)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, actions: Tensor) -> Tuple[Tensor, Tensor]:
        latents = self.encode(actions)
        recon_actions = self.decode(latents)
        return recon_actions, latents
