"""SuperMimic smoke tests without Isaac Gym, including real optimizer updates.

Run directly with ``--device cpu``, ``--device cuda:0`` or ``--device all``.
The default selects CUDA when available; pytest runs the CPU cases.
"""
import argparse
import math

import torch

from supermimic import DQStateAdapter, ReplayBuffer, EnsembleWorldModel, SuperMimicAgent


class MockEnv:
    """Match tensor shapes and the xyzw quaternion convention in B1Z1PickMulti."""

    def __init__(self, device="cpu", num_envs=4):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = 100
        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._dof_pos = torch.randn(num_envs, 20, device=device) * 0.1
        self._dof_vel = torch.randn(num_envs, 20, device=device) * 0.1
        self.ee_pos = torch.zeros(num_envs, 3, device=device)
        self.ee_orn = torch.zeros(num_envs, 4, device=device)
        self.ee_orn[:, 3] = 1.0
        self._cube_root_states = torch.zeros(num_envs, 13, device=device)
        self._cube_root_states[:, 2] = 0.2
        self._cube_root_states[:, 6] = 1.0
        self.dof_pos_gripper = torch.zeros(num_envs, 2, device=device)
        self.dof_vel_gripper = torch.zeros(num_envs, 2, device=device)
        self.rew_buf = torch.zeros(num_envs, device=device)


def assert_finite_info(info):
    for name, value in info.items():
        if isinstance(value, (float, int)):
            assert math.isfinite(value), f"Non-finite {name}: {value}"


def snapshot(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def assert_parameters_changed(before, module):
    assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters())), (
        f"{type(module).__name__} optimizer did not change any parameters"
    )
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())


def test_state_adapter(device="cpu"):
    env = MockEnv(device)
    adapter = DQStateAdapter(env, device=device)
    state = adapter.capture_state()
    assert state.shape == (4, 51)
    assert torch.isfinite(state).all()
    assert set(adapter.get_state_dict()) == {
        "robot_dof_pos", "robot_dof_vel", "ee_pose", "object_pose",
        "object_vel", "gripper_state", "context",
    }
    assert torch.allclose(state[:, 47], torch.ones(4, device=device), atol=1e-5)

    # Identity and 90 degrees around X: local +Z must rotate to world -Y.
    half_angle = math.sqrt(0.5)
    quats = state.new_tensor([[0, 0, 0, 1], [half_angle, 0, 0, half_angle]])
    rotated = adapter._quat_rotate_vec(quats, state.new_tensor([0, 0, 1]))
    expected = state.new_tensor([[0, 0, 1], [0, -1, 0]])
    assert torch.allclose(rotated, expected, atol=1e-6)

    # Check both the captured context and the differentiable reward use xyzw.
    env._cube_root_states[:, :3] = state.new_tensor([0, -0.2, 0])
    unaligned = adapter.capture_state()
    env.ee_orn[:] = quats[1]
    aligned = adapter.capture_state()
    assert torch.allclose(aligned[:, 47], torch.ones(4, device=device), atol=1e-5)
    reward_unaligned = adapter.compute_geometric_proxy_reward(unaligned, unaligned)
    reward_aligned = adapter.compute_geometric_proxy_reward(aligned, aligned)
    assert torch.allclose(reward_aligned - reward_unaligned,
                          torch.full((4,), 0.5, device=device), atol=1e-5)
    next_state = aligned.detach().clone().requires_grad_(True)
    reward = adapter.compute_geometric_proxy_reward(aligned, next_state)
    assert reward.shape == (4,)
    reward.sum().backward()
    assert torch.isfinite(next_state.grad).all()
    assert next_state.grad.abs().sum() > 0


def test_replay_buffer(device="cpu"):
    buffer = ReplayBuffer(capacity=128, state_dim=51, action_dim=9, device=device)
    states = torch.randn(4, 51, device=device)
    actions = torch.randn(4, 9, device=device)
    rewards = torch.randn(4, 1, device=device)
    next_states = states + 0.01
    # The native environment/wrapper returns integer column-shaped done flags.
    dones = torch.tensor([[0], [1], [0], [0]], dtype=torch.long, device=device)
    native_rewards = rewards + 1
    buffer.add(states, actions, rewards, next_states, dones, native_rewards)
    assert len(buffer) == 4
    assert buffer.dones.dtype == torch.bool
    assert torch.equal(buffer.dones[:4], dones[:, 0].bool())
    assert torch.equal(buffer.rewards[:4], rewards[:, 0])
    assert torch.equal(buffer.native_rewards[:4], native_rewards[:, 0])
    batch = buffer.sample(4)
    assert batch["states"].shape == (4, 51)
    assert batch["dones"].shape == (4,)

    # A single-environment batch must not disappear through squeeze().
    single_buffer = ReplayBuffer(capacity=8, state_dim=51, action_dim=9, device=device)
    single_buffer.add(states[:1], actions[:1], rewards[:1], next_states[:1], dones[:1])
    assert len(single_buffer) == 1
    assert single_buffer.rewards[0] == rewards[0, 0]


def test_world_model(device="cpu"):
    model = EnsembleWorldModel(state_dim=51, action_dim=9, num_ensemble=2,
                               hidden_dims=[32, 32], device=device)
    states = torch.randn(8, 51, device=device)
    actions = torch.randn(8, 9, device=device)
    assert model(states, actions).shape == (8, 51)
    mean, std = model.predict_with_uncertainty(states, actions)
    assert mean.shape == std.shape == (8, 51)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    rollout = model.rollout(states, torch.randn(8, 4, 9, device=device))
    assert rollout.shape == (8, 5, 51)
    loss, info = model.compute_loss(states, actions, states + 0.01)
    assert torch.isfinite(loss)
    assert_finite_info(info)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_supermimic_agent(device="cpu"):
    torch.manual_seed(7)
    env = MockEnv(device)
    adapter = DQStateAdapter(env, device=device)
    agent = SuperMimicAgent(
        state_dim=51, action_dim=9, device=device,
        num_ensemble=2, world_model_hidden=[32, 32], policy_hidden=[32, 32],
        imagination_horizon=3, world_model_batch_size=8,
        world_model_train_steps=2, imagination_batch_size=8,
        policy_update_steps=2, replay_capacity=128, min_buffer_size=32,
    )
    agent.attach_state_adapter(adapter)
    agent.warmup_iterations = 1

    state = adapter.capture_state()
    for step in range(12):
        actions = agent.select_action(state, exploration_noise=0.1)
        assert actions.shape == (env.num_envs, 9)
        assert actions.abs().max() <= 1.0
        env.ee_pos[:, 2] += 0.005
        env.progress_buf += 1
        next_state = adapter.capture_state()
        rewards = adapter.compute_geometric_proxy_reward(state, next_state).unsqueeze(-1)
        dones = torch.zeros(env.num_envs, 1, dtype=torch.long, device=device)
        if step == 11:
            dones[0] = 1
        agent.observe_transition(state, actions, rewards, next_state, dones, rewards + 1)
        state = next_state
    assert len(agent.replay_buffer) == 48

    world_before = snapshot(agent.world_model)
    warmup = agent.update()
    assert warmup["phase"] == "warmup"
    assert_finite_info(warmup)
    assert_parameters_changed(world_before, agent.world_model)
    assert agent.world_model_trainer.train_steps == 2
    assert agent.policy_trainer.train_steps == 0

    # Multiple joint updates expose reused autograd graphs and optimizer mutation errors.
    for _ in range(2):
        policy_before = snapshot(agent.policy)
        critic_before = snapshot(agent.critic)
        info = agent.update()
        assert info["phase"] == "joint_training"
        assert {"world_model_loss", "policy_loss", "critic_loss"}.issubset(info)
        assert_finite_info(info)
        assert_parameters_changed(policy_before, agent.policy)
        assert_parameters_changed(critic_before, agent.critic)
    assert agent.world_model_trainer.train_steps == 6
    assert agent.policy_trainer.train_steps == 4
    assert agent.get_stats()["iteration"] == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, all, cpu, or a CUDA device")
    args = parser.parse_args()
    if args.device == "all":
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    elif args.device == "auto":
        devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]
    else:
        devices = [args.device]
    # Small smoke-test networks run faster with a single CPU thread.
    torch.set_num_threads(1)
    for device in devices:
        print(f"SuperMimic integration tests on {device}", flush=True)
        for test in (test_state_adapter, test_replay_buffer, test_world_model, test_supermimic_agent):
            test(device)
            print(f"  PASS {test.__name__}", flush=True)
    print("All SuperMimic integration tests passed.")


if __name__ == "__main__":
    main()
