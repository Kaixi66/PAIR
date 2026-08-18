"""Perception token helpers shared by PAIR Action AE v2 training."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor

from prismatic.models.pair_bridge import build_pair_perception_tokens as _build_pair_perception_tokens


def build_perception_tokens(
    *,
    hidden_state: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    num_patches: int,
) -> Tuple[Tensor, Tensor]:
    """Build `[V0; T0]` from initial multimodal hidden states without action-token states."""
    return _build_pair_perception_tokens(
        hidden_state=hidden_state,
        labels=labels,
        attention_mask=attention_mask,
        num_patches=num_patches,
    )
