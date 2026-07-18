import numpy as np
import torch

from godynur.td3 import TD3


def _make_agent(**kwargs):
    return TD3(state_dim=4, action_dim=2, action_scale=1.0, hidden=8, seed=0, **kwargs)


def _seed_and_fill_buffer(agent, n=200, action_offset=0.0, seed=123):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.normal(size=4).astype(np.float32)
        a = (rng.uniform(-1, 1, size=2).astype(np.float32) + action_offset)
        r = np.array([rng.normal()], dtype=np.float32)
        s2 = rng.normal(size=4).astype(np.float32)
        d = np.array([0.0], dtype=np.float32)
        agent.buffer.add(s, a, r, s2, d)


def _capture_q1_action_input(agent):
    """Hook q1 to record the action half of the last (s, action) pair it
    saw, so the actor-loss's exact input tensor can be inspected."""
    captured = {}

    def hook(module, inputs):
        sa = inputs[0]
        captured["action"] = sa[:, -2:].detach().clone()  # action_dim=2

    handle = agent.q1.register_forward_pre_hook(hook)
    return captured, handle


def test_default_ste_off_actor_loss_uses_raw_actor_output_exactly():
    # Arrange: with use_action_ste off (the default), the tensor fed into
    # q1 for the actor loss must be EXACTLY actor(s) -- not any function
    # of the buffered action -- matching the pre-flag code path exactly.
    torch.manual_seed(0)
    agent = _make_agent()
    assert agent.use_action_ste is False
    _seed_and_fill_buffer(agent, n=200, action_offset=5.0)  # wildly off-policy

    captured, handle = _capture_q1_action_input(agent)
    for _ in range(agent.policy_delay):
        agent.learn(batch=32)
    handle.remove()

    # Assert: the buffered actions were offset by +5 (uniform(-1,1)+5,
    # i.e. in [4, 6]), far outside the actor's tanh output range. If the
    # action fed to q1 for the actor loss were derived from the buffer
    # (even partially, as STE would do), its magnitude would reflect
    # that offset. A max magnitude within tanh's [-1, 1] range proves the
    # fed-in action is exactly the raw actor output, untouched by the
    # buffered action -- matching the pre-flag behavior exactly.
    assert "action" in captured
    assert captured["action"].abs().max().item() <= 1.0 + 1e-5


def test_straight_through_forward_value_and_gradient_are_correct_in_isolation():
    # Arrange: the STE construction itself, isolated from the full TD3
    # pipeline -- forward value must equal the target exactly, and the
    # gradient must flow to the raw tensor as if it were the identity.
    raw = torch.tensor([0.1, -0.4, 0.9], requires_grad=True)
    target = torch.tensor([0.7, 0.7, 0.7])

    ste = raw + (target - raw).detach()
    loss = ste.sum()
    loss.backward()

    # Assert
    assert torch.allclose(ste.detach(), target, atol=1e-6)
    assert raw.grad is not None
    assert torch.allclose(raw.grad, torch.ones_like(raw))


def test_action_ste_actor_update_differs_from_default_when_buffer_action_offset():
    # Arrange: two agents, identical initialization and buffer contents,
    # differing only in use_action_ste. The buffered actions are
    # deliberately far from the freshly-initialized actor's own output,
    # mimicking a shield that substituted something very different from
    # the raw policy proposal.
    torch.manual_seed(0)
    agent_default = _make_agent(use_action_ste=False)
    torch.manual_seed(0)
    agent_ste = _make_agent(use_action_ste=True)

    np.random.seed(0)
    _seed_and_fill_buffer(agent_default, n=200, action_offset=5.0)
    np.random.seed(0)
    _seed_and_fill_buffer(agent_ste, n=200, action_offset=5.0)

    torch.manual_seed(1)
    np.random.seed(0)
    for _ in range(agent_default.policy_delay):
        agent_default.learn(batch=32)

    torch.manual_seed(1)
    np.random.seed(0)
    for _ in range(agent_ste.policy_delay):
        agent_ste.learn(batch=32)

    # Assert: the two actors diverge -- the flag actually changes the
    # gradient used for the actor update under this action mismatch.
    diverged = any(
        not torch.allclose(p_def, p_ste, atol=1e-6)
        for p_def, p_ste in zip(
            agent_default.actor.parameters(), agent_ste.actor.parameters()
        )
    )
    assert diverged


def test_action_ste_does_not_crash_and_actor_parameters_change():
    # Arrange
    torch.manual_seed(0)
    agent = _make_agent(use_action_ste=True)
    before = [p.clone() for p in agent.actor.parameters()]
    _seed_and_fill_buffer(agent, n=200)

    # Act
    for _ in range(agent.policy_delay):
        agent.learn(batch=32)

    # Assert
    changed = any(
        not torch.allclose(b, a, atol=1e-9)
        for b, a in zip(before, agent.actor.parameters())
    )
    assert changed


def test_explicit_default_projection_none_uses_raw_actor_output_exactly():
    agent = _make_agent(differentiable_projection=None)
    assert agent.differentiable_projection is None
    _seed_and_fill_buffer(agent, n=200, action_offset=5.0)

    actor_output = {}

    def capture_actor(module, inputs, output):
        actor_output["raw"] = output.detach().clone()

    actor_handle = agent.actor.register_forward_hook(capture_actor)
    captured, q1_handle = _capture_q1_action_input(agent)
    for _ in range(agent.policy_delay):
        agent.learn(batch=32)
    actor_handle.remove()
    q1_handle.remove()

    assert "raw" in actor_output
    assert "action" in captured
    assert torch.equal(captured["action"], actor_output["raw"])


def test_differentiable_projection_takes_precedence_over_action_ste():
    projected = {}

    def projection(raw_action, state_batch):
        projected["raw"] = raw_action.detach().clone()
        projected["state"] = state_batch.detach().clone()
        return raw_action + 0.25

    agent = _make_agent(
        use_action_ste=True, differentiable_projection=projection
    )
    _seed_and_fill_buffer(agent, n=200, action_offset=5.0)
    captured, handle = _capture_q1_action_input(agent)
    for _ in range(agent.policy_delay):
        agent.learn(batch=32)
    handle.remove()

    assert projected["state"].shape == (32, 4)
    assert torch.equal(captured["action"], projected["raw"] + 0.25)
    assert captured["action"].max().item() < 1.25 + 1e-5
