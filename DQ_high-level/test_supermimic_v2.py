"""SuperMimic v2 component smoke test with real warmup and joint optimization.

No simulator is required. Run with --device cpu, --device cuda:0, or --device all.
Small networks and replay thresholds below apply only to this smoke test.
"""
import argparse
import math
from types import SimpleNamespace

import torch

from supermimic_v2 import DQRobotMDP, StateCapture, Context, SuperMimicAgentV2


class MockEnv:
    """Use the physical field names and xyzw orientations of B1Z1PickMulti."""

    def __init__(self, device):
        self.num_envs = 4
        self.num_actions = 9
        self.num_obs = 50
        self.device = device
        self.max_episode_length = 100
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._robot_root_states = torch.zeros(self.num_envs, 13, device=device)
        self._robot_root_states[:, 0] = torch.linspace(-1.0, -0.7, self.num_envs, device=device)
        self._robot_root_states[:, 2] = .55
        self._robot_root_states[:, 6] = 1.
        self._cube_root_states = torch.zeros(self.num_envs, 13, device=device)
        self._cube_root_states[:, 0] = .1
        self._cube_root_states[:, 2] = .65
        self._cube_root_states[:, 6] = 1.
        self.dof_pos_gripper = torch.full((self.num_envs, 2), .3, device=device)
        self.dof_vel_gripper = torch.zeros(self.num_envs, 2, device=device)
        self.num_gripper_joints = 2
        self.dof_limits_lower = torch.full((20,), -.5, device=device)
        self.dof_limits_upper = torch.full((20,), .5, device=device)
        self.table_heights = torch.full((self.num_envs,), .5, device=device)
        self.init_height = torch.full((self.num_envs,), .05, device=device)
        self.lifted_success_threshold = .2
        self.object_mass = torch.full((self.num_envs, 1), .5, device=device)
        self.sim_params = SimpleNamespace(gravity=SimpleNamespace(x=0., y=0., z=-9.81))
        self.ee_pos = torch.tensor([.15, 0., .7], device=device).expand(self.num_envs, -1).clone()
        self.ee_orn = torch.zeros(self.num_envs, 4, device=device)
        self.ee_orn[:, 3] = 1.
        self.obs_buf = torch.randn(self.num_envs, self.num_obs, device=device) * .1

    def observation(self, action):
        """Return an environment observation independently of MDP reconstruction."""
        self.obs_buf[:, :3] = self._robot_root_states[:, :3]
        self.obs_buf[:, 3:6] = self._cube_root_states[:, :3]
        self.obs_buf[:, 6:9] = self.ee_pos
        self.obs_buf[:, 9:13] = self.ee_orn
        self.obs_buf[:, -self.num_actions:] = action
        return self.obs_buf.clone()


def parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def assert_changed(before, module):
    assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters())), (
        f"{type(module).__name__} parameters did not change"
    )
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())


def assert_finite(metrics):
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f"Non-finite {name}: {value}"


def run_component_checks(device):
    torch.manual_seed(17)
    env = MockEnv(device)
    capture = StateCapture(env, device=device)
    context = Context(env, device=device)
    mdp = DQRobotMDP(env, device=device)
    physical = capture.capture()
    action = torch.zeros(4, env.num_actions, device=device)
    obs = env.observation(action)
    state = mdp.capture_state(physical, obs)
    ctx = context.get_context()
    assert state.shape == (4, mdp.state_dim)
    assert mdp.state_dim == 41 + env.num_obs
    torch.testing.assert_close(state[:, :34], physical)
    torch.testing.assert_close(state[:, 34:37], env.ee_pos)
    torch.testing.assert_close(state[:, 37:41], env.ee_orn)
    torch.testing.assert_close(state[:, 41:], obs)
    assert ctx.shape == (4, mdp.context_dim)
    torch.testing.assert_close(state[:, :7], env._robot_root_states[:, :7])
    torch.testing.assert_close(state[:, 22], env.dof_pos_gripper.mean(dim=-1))
    torch.testing.assert_close(ctx[:, 11], env.table_heights + env.init_height + env.lifted_success_threshold)
    torch.testing.assert_close(state[:, 10:16], state.new_tensor([[1, 0, 0, 0, 1, 0]]).expand(4, -1))
    print("  PASS physical state and reset context", flush=True)

    torch.testing.assert_close(mdp.observe(state, ctx, action), obs)
    assert mdp.inputs(state, action, ctx).shape == (4, mdp.wm_input_dim)
    delta = mdp.targets(state, state + .001)
    assert delta.shape == (4, mdp.delta_dim)
    torch.testing.assert_close(mdp.state_error(delta, delta), torch.zeros_like(delta))
    assert mdp.critic_features(state, ctx, obs).shape == (4, mdp.critic_dim)
    reward, done = mdp.reward(state, action, state, ctx)
    assert reward.shape == done.shape == (4,)
    assert torch.isfinite(reward).all() and torch.isfinite(done).all()
    print("  PASS MDP input, target, critic and reward contracts", flush=True)

    agent = SuperMimicAgentV2(
        mdp=mdp, device=device,
        wm_hidden=(32, 32), policy_hidden=(32, 32), critic_hidden=(32, 32),
        wm_ensemble_size=2, wm_window=3, horizon=3,
        wm_batch_size=8, policy_batch_size=8,
        critic_epochs=2, critic_minibatches=2,
        warmup_iterations=1, replay_capacity=128, min_buffer_size=32,
    )
    # The old script inserted 200 transitions against a default threshold of
    # 10,000, so update() skipped every optimizer and still reported success.
    skipped = agent.update()
    assert skipped["skipped"] and agent.world_model_updates == 0
    for step in range(12):
        action = agent.select_action(obs, deterministic=False)
        assert action.shape == (4, 9) and torch.isfinite(action).all()
        capture.update_action(action)
        env._cube_root_states[:, 0] += .001
        env.ee_pos[:, 0] += .0005
        env.progress_buf += 1
        capture.increment_timestep()
        next_obs = env.observation(action)
        next_state = mdp.capture_state(capture.capture(), next_obs)
        torch.testing.assert_close(mdp.observe(next_state, ctx, action), next_obs)
        reward, _ = mdp.reward(state, action, next_state, ctx)
        done = torch.zeros(4, 1, dtype=torch.long, device=device)
        if step == 11:
            done[0] = 1
        agent.observe_transition(state, action, reward[:, None], next_state,
                                 done, ctx, obs, next_obs)
        state, obs = next_state, next_obs
    assert agent.buffer_size == 48 >= agent.min_buffer_size
    print("  PASS replay collection above the explicit training threshold", flush=True)

    wm_before = parameters(agent.wm)
    policy_before = parameters(agent.policy)
    warmup = agent.update()
    assert warmup["phase"] == "warmup" and not warmup["skipped"]
    assert_finite(warmup)
    assert_changed(wm_before, agent.wm)
    assert agent.world_model_updates == 1 and agent.policy_updates == 0
    assert all(torch.equal(old, new) for old, new in zip(policy_before, agent.policy.parameters()))
    print("  PASS actual world-model warmup optimizer step", flush=True)

    for _ in range(2):
        wm_before = parameters(agent.wm)
        policy_before = parameters(agent.policy)
        critic_before = parameters(agent.critic)
        metrics = agent.update()
        assert metrics["phase"] == "joint" and not metrics["skipped"]
        assert {"world_model_loss", "policy_loss", "critic_loss"}.issubset(metrics)
        assert_finite(metrics)
        assert metrics["policy_grad_norm"] > 0
        assert_changed(wm_before, agent.wm)
        assert_changed(policy_before, agent.policy)
        assert_changed(critic_before, agent.critic)
    assert agent.world_model_updates == 3 and agent.policy_updates == 2
    assert agent.get_stats()["iteration"] == 4
    print("  PASS two joint updates with world-model, policy and critic parameter changes", flush=True)


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
    torch.set_num_threads(1)
    for device in devices:
        print(f"SuperMimic v2 component smoke test on {device}", flush=True)
        run_component_checks(device)
    print("All SuperMimic v2 component smoke tests passed.")


if __name__ == "__main__":
    main()
