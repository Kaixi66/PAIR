from pathlib import Path

import torch

from pair_action_ae.checkpoint import load_encoder_checkpoint, save_encoder_checkpoint
from pair_action_ae.model import ActionAEConfig, ActionTransformerAE


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
