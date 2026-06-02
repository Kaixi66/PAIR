import torch
from torch import nn

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


def test_action_head_initial_action_states_preserve_zero_default():
    torch.manual_seed(7)
    hidden_dim = 16
    batch_size = 2
    num_task_tokens = 2
    num_layers = 25

    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        num_task_tokens=num_task_tokens,
        use_pro_version=False,
    )
    proprio_projector = nn.Linear(8, hidden_dim).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, num_layers, num_task_tokens + NUM_TOKENS, hidden_dim)
    proprio = torch.randn(batch_size, 8)
    zero_init = torch.zeros(batch_size, 7 * NUM_ACTIONS_CHUNK, hidden_dim)

    default_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
    )
    explicit_zero_actions = action_head.predict_action(
        hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        phase="Inference",
        initial_action_states=zero_init,
    )

    assert default_actions.shape == (batch_size, NUM_ACTIONS_CHUNK, 7)
    assert torch.allclose(default_actions, explicit_zero_actions)
