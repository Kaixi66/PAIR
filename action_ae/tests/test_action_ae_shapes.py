from pathlib import Path

import torch

from pair_action_ae.checkpoint import load_encoder_checkpoint, save_encoder_checkpoint
from pair_action_ae.model import (
    ActionAEConfig,
    ActionPerceptionAEConfig,
    ActionPerceptionTransformerAE,
    ActionTransformerAE,
    corrupt_actions,
)


def test_action_ae_forward_shapes():
    config = ActionAEConfig()
    model = ActionTransformerAE(config)
    actions = torch.randn(2, 8, 7)

    recon, latent = model(actions)

    assert recon.shape == (2, 8, 7)
    assert latent.shape == (2, 8, 16)


def test_encoder_checkpoint_roundtrip(tmp_path: Path):
    config = ActionAEConfig()
    model = ActionTransformerAE(config)
    path = tmp_path / "encoder.pt"

    save_encoder_checkpoint(path=path, encoder=model.encoder, config=config)
    encoder = load_encoder_checkpoint(path)
    actions = torch.randn(2, 8, 7)

    latent = encoder(actions)

    assert latent.shape == (2, 8, 16)


def test_action_perception_ae_forward_shapes():
    config = ActionPerceptionAEConfig(perception_dim=32)
    model = ActionPerceptionTransformerAE(config)
    actions = torch.randn(2, 8, 7)
    perception_tokens = torch.randn(2, 12, 32)
    perception_mask = torch.ones(2, 12, dtype=torch.bool)

    output = model(actions, perception_tokens, perception_mask)

    assert output.recon_actions.shape == (2, 8, 7)
    assert output.latents.shape == (2, 8, 8)
    assert output.corrupted_actions.shape == (2, 8, 7)
    assert output.action_mask.shape == (2, 8)


def test_action_corruption_masks_full_steps():
    actions = torch.ones(2, 8, 7)

    corrupted, mask = corrupt_actions(actions, mask_prob=1.0, noise_std=0.0, training=True)

    assert mask.shape == (2, 8)
    assert mask.all()
    assert torch.count_nonzero(corrupted) == 0


def test_perception_encoder_checkpoint_roundtrip(tmp_path: Path):
    config = ActionPerceptionAEConfig(perception_dim=32)
    model = ActionPerceptionTransformerAE(config)
    path = tmp_path / "encoder_v2.pt"

    save_encoder_checkpoint(path=path, encoder=model.encoder, config=config)
    encoder = load_encoder_checkpoint(path)
    actions = torch.randn(2, 8, 7)
    perception_tokens = torch.randn(2, 12, 32)
    perception_mask = torch.ones(2, 12, dtype=torch.bool)

    latent = encoder(actions, perception_tokens, perception_mask)

    assert getattr(encoder, "requires_perception")
    assert getattr(encoder, "latent_dim") == 8
    assert latent.shape == (2, 8, 8)
