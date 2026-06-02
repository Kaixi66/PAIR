from pathlib import Path

import torch

from prismatic.models.pair_bridge import (
    PairBridge,
    PairBridgeConfig,
    load_pair_bridge_checkpoint,
    save_pair_bridge_checkpoint,
)


def test_pair_bridge_shapes_and_gate_init(tmp_path: Path):
    config = PairBridgeConfig(llm_dim=4096, bridge_dim=512, latent_dim=16, horizon=8, action_dim=7)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 20, 4096)
    base_init = torch.randn(2, 56, 4096)

    output = bridge(perception_tokens, base_init)

    assert output.action_init.shape == (2, 56, 4096)
    assert output.z_align.shape == (2, 8, 16)
    assert torch.allclose(output.init_gate, torch.tanh(torch.tensor(0.05)))
    assert not torch.allclose(output.action_init, base_init)
    assert "slot_bias" not in dict(bridge.named_parameters())

    ckpt = tmp_path / "pair_bridge.pt"
    save_pair_bridge_checkpoint(
        path=ckpt,
        pair_bridge=bridge,
        config=config,
        action_ae_encoder_path="/tmp/encoder.pt",
        metadata={"step": 0},
    )
    loaded = load_pair_bridge_checkpoint(ckpt)
    loaded_output = loaded(perception_tokens, base_init)

    assert loaded_output.action_init.shape == (2, 56, 4096)
    assert loaded_output.z_align.shape == (2, 8, 16)


def test_pair_bridge_fixed_gate_mode():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_mode="fixed",
        init_gate_value=0.1,
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert torch.allclose(output.init_gate, torch.tanh(torch.tensor(0.1)))
    assert "init_gate" not in dict(bridge.named_parameters())
    assert "init_gate" in dict(bridge.named_buffers())


def test_pair_bridge_keeps_init_gate_fp32_after_bf16_cast():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config).to(torch.bfloat16)

    bridge.keep_init_gate_fp32()

    assert bridge.down_proj.weight.dtype == torch.bfloat16
    assert bridge.init_gate.dtype == torch.float32
    assert dict(bridge.named_parameters())["init_gate"].dtype == torch.float32


def test_pair_bridge_perception_mask():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)
    perception_mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, False]])

    output = bridge(perception_tokens, base_init, perception_mask)

    assert output.action_init.shape == (2, 56, 64)
    assert output.z_align.shape == (2, 8, 8)
