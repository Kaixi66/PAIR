from pathlib import Path

import pytest
import torch

from prismatic.models.pair_bridge import load_frozen_action_encoder


ENCODER_PATH = Path("/umd-datapool/kaixi/PAIR/action_ae_runs/ae_libero_1/encoder.pt")


@pytest.mark.skipif(not ENCODER_PATH.exists(), reason="Action AE encoder checkpoint is not available")
def test_frozen_action_ae_encoder_shape():
    encoder = load_frozen_action_encoder(ENCODER_PATH, device="cpu")
    latents = encoder(torch.zeros(2, 8, 7))
    assert latents.shape == (2, 8, 16)
    assert all(not param.requires_grad for param in encoder.parameters())
