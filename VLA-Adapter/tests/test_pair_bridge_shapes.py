from pathlib import Path

import pytest
import torch

from prismatic.models.pair_bridge import (
    PairBridge,
    PairBridgeConfig,
    alignment_loss,
    build_pair_perception_tokens,
    load_pair_bridge_checkpoint,
    save_pair_bridge_checkpoint,
)


def make_config(**overrides):
    values = {"llm_dim": 64, "latent_dim": 8, "horizon": 8, "action_dim": 7}
    values.update(overrides)
    return PairBridgeConfig(**values)


def test_public_pair_architecture_is_fixed():
    config = make_config()
    assert config.bridge_dim == 1024
    assert config.num_heads == 8
    assert config.bridge_mlp_dim == 4096
    assert config.init_mlp_dim == 4096
    with pytest.raises(TypeError):
        PairBridgeConfig(llm_dim=64, bridge_dim=512)


def test_pair_bridge_shapes_and_zero_initialized_tanh_gate():
    bridge = PairBridge(make_config())
    output = bridge(torch.randn(2, 6, 64))

    assert output.bridge_tokens.shape == (2, 8, 1024)
    assert output.z_align.shape == (2, 8, 8)
    assert output.action_init_delta.shape == (2, 8, 64)
    assert output.init_gate.shape == (2, 8)
    assert output.init_gate_raw.shape == (2, 8)
    assert torch.count_nonzero(output.init_gate) == 0
    assert torch.count_nonzero(bridge.gate_proj.weight) == 0
    assert torch.count_nonzero(bridge.gate_proj.bias) == 0
    assert torch.count_nonzero(output.action_init) == 0


def test_pair_gate_is_input_dependent_per_step_tanh():
    bridge = PairBridge(make_config())
    with torch.no_grad():
        bridge.gate_proj.weight[:, 0] = 0.25
        bridge.gate_proj.bias.fill_(0.1)

    first = bridge(torch.randn(2, 6, 64))
    second = bridge(torch.randn(2, 6, 64))
    assert first.init_gate.shape == (2, 8)
    assert torch.all(first.init_gate.abs() <= 1)
    assert not torch.allclose(first.init_gate, second.init_gate)


def test_pair_bridge_and_gate_receive_gradients():
    bridge = PairBridge(make_config())
    with torch.no_grad():
        bridge.gate_proj.bias.fill_(0.2)
    output = bridge(torch.randn(2, 6, 64))
    loss = output.action_init.square().mean() + output.z_align.square().mean()
    loss.backward()

    assert bridge.gate_proj.weight.grad is not None
    assert bridge.gate_proj.weight.grad.abs().sum() > 0
    assert bridge.init_proj[-1].weight.grad is not None
    assert bridge.init_proj[-1].weight.grad.abs().sum() > 0
    assert bridge.align_proj[-1].weight.grad is not None
    assert bridge.align_proj[-1].weight.grad.abs().sum() > 0


def test_pair_bridge_checkpoint_round_trip_and_old_base_config(tmp_path: Path):
    bridge = PairBridge(make_config())
    path = tmp_path / "pair.pt"
    save_pair_bridge_checkpoint(
        path=path,
        pair_bridge=bridge,
        config=bridge.config,
        action_ae_encoder_path="encoder.pt",
    )
    payload = torch.load(path, map_location="cpu")
    payload["model_config"].update(
        {
            "bridge_dim": 1024,
            "gate_num_layers": 1,
            "init_gate_mode": "learnable",
            "init_gate_value": 0.0,
            "init_gate_granularity": "per_step",
            "gate_activation": "tanh",
            "injection_positions": ["start"],
        }
    )
    torch.save(payload, path)

    loaded = load_pair_bridge_checkpoint(path)
    assert loaded.config == bridge.config
    for key, value in bridge.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_pair_bridge_keeps_gate_fp32_after_bf16_cast():
    bridge = PairBridge(make_config()).to(torch.bfloat16)
    bridge.keep_high_precision_params()
    assert bridge.gate_norm.weight.dtype == torch.float32
    assert bridge.gate_proj.weight.dtype == torch.float32


def test_pair_perception_helper_training_masks_action_tokens():
    hidden = torch.randn(1, 7, 64)
    attention = torch.ones(1, 6, dtype=torch.long)
    labels = torch.full((1, 6), -100, dtype=torch.long)
    tokens, mask = build_pair_perception_tokens(
        hidden_state=hidden,
        num_patches=2,
        attention_mask=attention,
        labels=labels,
    )
    assert tokens.shape[:2] == mask.shape
    assert mask.dtype == torch.bool


def test_alignment_loss_is_cosine_only():
    predicted = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert torch.allclose(alignment_loss(predicted, target), torch.tensor(0.5))
    with pytest.raises(TypeError):
        alignment_loss(predicted, target, loss_type="l2")
    with pytest.raises(ValueError):
        alignment_loss(predicted, target[:, :1])
