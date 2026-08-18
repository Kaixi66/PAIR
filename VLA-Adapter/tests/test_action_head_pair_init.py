import pytest
import torch
import torch.nn as nn

from prismatic.models.action_heads import L1RegressionActionHead, MLPResNet
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


class _DoubleBlock(nn.Module):
    def forward(self, x, h_t=None, h_a=None, p=None):
        return 2.0 * x


def boundary_model():
    model = MLPResNet(num_blocks=24, input_dim=1, hidden_dim=1, output_dim=1)
    model.layer_norm1 = nn.Identity()
    model.fc1 = nn.Identity()
    model.relu = nn.Identity()
    model.mlp_resnet_blocks = nn.ModuleList([_DoubleBlock() for _ in range(24)])
    model.layer_norm2 = nn.Identity()
    model.fc2 = nn.Identity()
    return model


def test_pair_is_injected_after_stem_before_first_expert_block():
    model = boundary_model()
    x = torch.zeros(1, NUM_ACTIONS_CHUNK, 1)
    delta = torch.ones_like(x)
    gate = torch.ones(1, NUM_ACTIONS_CHUNK)
    expert_hidden = torch.zeros(1, 25, 1)

    output = model(x, h_a=expert_hidden, h_t=expert_hidden, pair_init=delta, pair_gate=gate)
    assert torch.equal(output, torch.full_like(output, 2.0**24))


def test_zero_pair_gate_matches_no_pair():
    model = boundary_model()
    x = torch.randn(2, NUM_ACTIONS_CHUNK, 1)
    delta = torch.randn_like(x)
    expert_hidden = torch.zeros(2, 25, 1)

    baseline = model(x, h_a=expert_hidden, h_t=expert_hidden)
    zero_gate = model(
        x,
        h_a=expert_hidden,
        h_t=expert_hidden,
        pair_init=delta,
        pair_gate=torch.zeros(2, NUM_ACTIONS_CHUNK),
    )
    assert torch.equal(baseline, zero_gate)


def test_pair_delta_and_gate_receive_gradients():
    model = boundary_model()
    x = torch.zeros(1, NUM_ACTIONS_CHUNK, 1)
    delta = torch.ones_like(x, requires_grad=True)
    gate = torch.full((1, NUM_ACTIONS_CHUNK), 0.5, requires_grad=True)
    expert_hidden = torch.zeros(1, 25, 1)

    model(x, h_a=expert_hidden, h_t=expert_hidden, pair_init=delta, pair_gate=gate).sum().backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    assert gate.grad is not None and gate.grad.abs().sum() > 0


def test_public_pair_gate_requires_batched_per_step_shape():
    model = boundary_model()
    x = torch.zeros(1, NUM_ACTIONS_CHUNK, 1)
    delta = torch.ones_like(x)
    expert_hidden = torch.zeros(1, 25, 1)

    with pytest.raises(ValueError, match="PAIR gate"):
        model(x, h_a=expert_hidden, h_t=expert_hidden, pair_init=delta, pair_gate=torch.tensor(1.0))


def test_action_head_accepts_canonical_pair_tensors():
    torch.manual_seed(19)
    hidden_dim = 16
    batch_size = 2
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=2,
        use_pro_version=False,
    )
    hidden_states = torch.randn(batch_size, 25, 2 + NUM_TOKENS, hidden_dim)
    pair_delta = torch.randn(batch_size, NUM_ACTIONS_CHUNK, hidden_dim)
    pair_gate = torch.tanh(torch.randn(batch_size, NUM_ACTIONS_CHUNK))

    actions = action_head.predict_action(
        hidden_states,
        phase="Inference",
        initial_action_states=pair_delta,
        initial_action_gate=pair_gate,
    )
    assert actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
