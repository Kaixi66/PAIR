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
    assert output.init_gate.shape == (2, 8)
    assert output.init_gate_raw.shape == (2, 8)
    assert torch.allclose(output.init_gate, torch.full((2, 8), config.init_gate_value))
    assert not torch.allclose(output.action_init, base_init)
    assert "slot_bias" not in dict(bridge.named_parameters())
    assert bridge.config.init_gate_granularity == "per_step"
    assert bridge.config.input_dependent_gate
    assert "init_gate" not in dict(bridge.named_parameters())
    assert dict(bridge.named_parameters())["gate_proj.weight"].shape == (1, 512)
    assert torch.count_nonzero(bridge.gate_proj.weight) == 0
    expected_bias = torch.logit(torch.tensor(config.init_gate_value))
    assert torch.allclose(bridge.gate_proj.bias, expected_bias.reshape(1))
    assert bridge.config.bridge_mlp_dim == 1024
    assert bridge.bridge_mlp is not None
    assert torch.count_nonzero(bridge.bridge_mlp[-1].weight) > 0

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
    assert loaded_output.init_gate.shape == (2, 8)


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
        init_gate_granularity="scalar",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert torch.allclose(output.init_gate, torch.tensor(0.1))
    assert "init_gate" not in dict(bridge.named_parameters())
    assert "init_gate" in dict(bridge.named_buffers())


def test_pair_bridge_fixed_gate_mode_accepts_exact_one_with_tanh():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_mode="fixed",
        init_gate_value=1.0,
        init_gate_granularity="per_step",
        gate_activation="tanh",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.randn(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (8,)
    assert torch.allclose(output.init_gate, torch.ones(8))
    assert torch.allclose(
        output.action_init,
        base_init + output.action_init_delta,
    )


def test_pair_bridge_keeps_scale_and_gate_fp32_after_bf16_cast():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config).to(torch.bfloat16)

    bridge.keep_high_precision_params()

    assert bridge.down_proj.weight.dtype == torch.bfloat16
    assert bridge.bridge_mlp[0].weight.dtype == torch.bfloat16
    assert bridge.slot_scale.dtype == torch.float32
    assert bridge.gate_norm.weight.dtype == torch.float32
    assert bridge.gate_proj.weight.dtype == torch.float32
    assert dict(bridge.named_parameters())["slot_scale"].dtype == torch.float32
    assert dict(bridge.named_parameters())["gate_proj.weight"].dtype == torch.float32


def test_pair_bridge_bf16_forward_with_fp32_gate_on_cuda():
    if not torch.cuda.is_available():
        return

    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config).to(torch.bfloat16).cuda()
    bridge.keep_high_precision_params()

    perception_tokens = torch.randn(2, 6, 64, device="cuda", dtype=torch.bfloat16)
    base_init = torch.zeros(2, 56, 64, device="cuda", dtype=torch.bfloat16)
    perception_mask = torch.ones(2, 6, device="cuda", dtype=torch.bool)

    output = bridge(perception_tokens, base_init, perception_mask)

    assert output.action_init.shape == (2, 56, 64)
    assert output.init_gate.shape == (2, 8)
    assert output.action_init.dtype == torch.bfloat16
    assert output.init_gate.dtype == torch.bfloat16


def test_pair_bridge_per_step_gate_broadcasts_across_action_dims():
    config = PairBridgeConfig(
        llm_dim=64,
        bridge_dim=32,
        latent_dim=8,
        horizon=8,
        action_dim=7,
        num_heads=4,
        init_gate_value=0.2,
        init_gate_granularity="per_step",
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.randn(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (2, 8)
    delta_by_step = output.action_init_delta.reshape(2, 8, 7, 64)
    expected = base_init + (output.init_gate.view(2, 8, 1, 1) * delta_by_step).reshape(2, 56, 64)
    assert torch.allclose(output.action_init, expected)


def test_pair_bridge_input_dependent_gate_changes_with_tokens():
    torch.manual_seed(23)
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    with torch.no_grad():
        bridge.gate_proj.weight[:, 0] = 0.25
        bridge.gate_proj.bias.zero_()
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert output.init_gate.shape == (2, 8)
    assert output.init_gate.std(unbiased=False) > 0


def test_pair_bridge_legacy_config_disables_mlp():
    config = PairBridgeConfig.from_dict(
        {
            "llm_dim": 64,
            "bridge_dim": 32,
            "latent_dim": 8,
            "horizon": 8,
            "action_dim": 7,
            "num_heads": 4,
        }
    )
    bridge = PairBridge(config)

    assert bridge.config.bridge_mlp_dim == 0
    assert bridge.config.init_gate_granularity == "scalar"
    assert not bridge.config.input_dependent_gate
    assert bridge.config.gate_activation == "tanh"
    assert not bridge.config.init_gate_value_is_actual
    assert bridge.bridge_mlp is None
    assert "init_gate" in dict(bridge.named_parameters())


def test_pair_bridge_legacy_gate_value_is_raw():
    config = PairBridgeConfig.from_dict(
        {
            "llm_dim": 64,
            "bridge_dim": 32,
            "latent_dim": 8,
            "horizon": 8,
            "action_dim": 7,
            "num_heads": 4,
            "init_gate_value": 0.2,
        }
    )
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert torch.allclose(output.init_gate, torch.tanh(torch.tensor(0.2)))


def test_pair_bridge_perception_mask():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)
    perception_mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, False]])

    output = bridge(perception_tokens, base_init, perception_mask)

    assert output.action_init.shape == (2, 56, 64)
    assert output.z_align.shape == (2, 8, 8)


def test_pair_bridge_respects_configured_latent_dim():
    config = PairBridgeConfig(llm_dim=64, bridge_dim=32, latent_dim=8, horizon=8, action_dim=7, num_heads=4)
    bridge = PairBridge(config)
    perception_tokens = torch.randn(2, 6, 64)
    base_init = torch.zeros(2, 56, 64)

    output = bridge(perception_tokens, base_init)

    assert output.z_align.shape == (2, 8, 8)
