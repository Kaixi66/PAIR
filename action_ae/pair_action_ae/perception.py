"""Perception token helpers shared by PAIR Action AE v2 training."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import IGNORE_INDEX


def build_perception_tokens(
    *,
    hidden_state: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    num_patches: int,
) -> Tuple[Tensor, Tensor]:
    """Build `[V0; T0]` from initial multimodal hidden states without action-token states."""
    vision_tokens = hidden_state[:, :num_patches, :]
    text_tokens = hidden_state[:, num_patches:-1, :]

    text_labels = labels[:, 1:].to(hidden_state.device)
    text_attention_mask = attention_mask[:, 1:].to(hidden_state.device)
    min_len = min(text_tokens.shape[1], text_labels.shape[1], text_attention_mask.shape[1])
    text_tokens = text_tokens[:, :min_len, :]
    text_labels = text_labels[:, :min_len]
    text_attention_mask = text_attention_mask[:, :min_len]

    action_mask = get_current_action_mask(text_labels) | get_next_actions_mask(text_labels)
    prompt_mask = text_attention_mask.bool() & (text_labels == IGNORE_INDEX) & ~action_mask

    perception_tokens = torch.cat([vision_tokens, text_tokens], dim=1)
    vision_mask = torch.ones(vision_tokens.shape[:2], dtype=torch.bool, device=hidden_state.device)
    perception_mask = torch.cat([vision_mask, prompt_mask], dim=1)
    return perception_tokens, perception_mask
