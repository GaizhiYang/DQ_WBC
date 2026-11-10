"""Physical geometry and observation-contract regressions for the v2 MDP."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from supermimic_v2.mdp import DQRobotMDP


def setup(device="cpu", batch=2, obs_dim=50):
    env = SimpleNamespace(
        num_envs=batch, num_actions=9, num_obs=obs_dim, num_gripper_joints=1,
        dof_limits_lower=torch.tensor([-1., -math.pi / 2], device=device),
        dof_limits_upper=torch.tensor([1., 0.], device=device),
        ee_pos=torch.tensor([[.4, .1, .6]], device=device).repeat(batch, 1),
        ee_orn=torch.tensor([[0., 0., 0., 1.]], device=device).repeat(batch, 1),
    )
    mdp = DQRobotMDP(env, device=device)
    physical = torch.zeros(batch, 34, device=device)
    physical[:, 6] = 1
    physical[:, 7:10] = torch.tensor([.5, .1, .65], device=device)
    physical[:, 10] = physical[:, 14] = 1
    physical[:, 22] = -math.pi / 4  # Halfway between actual open and closed limits.
    obs = torch.arange(batch * obs_dim, device=device, dtype=torch.float32).reshape(batch, obs_dim) / 100
    context = torch.zeros(batch, 16, device=device)
    context[:, 10] = .2
    context[:, 11] = .8
    return env, mdp, physical, obs, context


class MDPContractTests(unittest.TestCase):
    def check_capture(self, device):
        env, mdp, physical, obs, context = setup(device, obs_dim=1284)
        physical.requires_grad_()
        obs.requires_grad_()
        state = mdp.capture_state(physical, obs)
        expected = torch.cat((physical, env.ee_pos, env.ee_orn, obs), dim=-1).detach()
        self.assertEqual(state.shape, (2, 1325))
        self.assertEqual(mdp.critic_dim, 63)
        self.assertFalse(state.requires_grad)
        torch.testing.assert_close(state, expected)
        torch.testing.assert_close(mdp.observe(state, context, torch.zeros(2, 9, device=device)), obs)
        with torch.no_grad():
            physical.fill_(100)
            obs.fill_(200)
            env.ee_pos.fill_(300)
            env.ee_orn.fill_(400)
        torch.testing.assert_close(state, expected)

    def test_full_observation_capture_is_detached_and_owns_storage_cpu(self):
        self.check_capture("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_full_observation_capture_is_detached_and_owns_storage_cuda(self):
        self.check_capture("cuda:0")

    def test_capture_rejects_missing_or_incompatible_measurements(self):
        env, mdp, physical, obs, _ = setup()
        for bad_physical, bad_obs in ((physical[:, :33], obs), (physical, obs[:, :-1])):
            with self.assertRaises(ValueError):
                mdp.capture_state(bad_physical, bad_obs)
        env.ee_orn = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "XYZW"):
            mdp.capture_state(physical, obs)
        del env.ee_orn
        with self.assertRaisesRegex(AttributeError, "measured"):
            mdp.capture_state(physical, obs)

    def test_joint_limits_are_required_and_not_assumed_from_action_range(self):
        env, _, _, _, _ = setup()
        del env.dof_limits_lower
        with self.assertRaisesRegex(AttributeError, "actual gripper joint limits"):
            DQRobotMDP(env, "cpu")

    def test_state_contract_and_critic_dimension_do_not_duplicate_observation(self):
        for obs_dim in (1, 50, 1284):
            with self.subTest(obs_dim=obs_dim):
                _, mdp, physical, obs, context = setup(obs_dim=obs_dim)
                state = mdp.capture_state(physical, obs)
                action = torch.zeros(2, 9)
                self.assertEqual(mdp.inputs(state, action, context).shape, (2, 41 + obs_dim + 9 + 16))
                self.assertEqual(mdp.critic_features(state, context, obs).shape, (2, 63))
                metadata = mdp.contract()
                self.assertEqual(metadata["quaternion_order"], "xyzw")
                self.assertEqual(metadata["observation_slice"], [41, 41 + obs_dim])
                self.assertIn("proxy", metadata["reward"])

    def test_predicted_observation_reaches_next_policy_with_gradient(self):
        _, mdp, physical, obs, context = setup(obs_dim=3)
        next_state = mdp.capture_state(physical, obs).requires_grad_()
        next_state.retain_grad()
        action = torch.zeros(2, 9)
        predicted_obs = mdp.advance_obs(torch.full_like(obs, -999), next_state, action, context)
        policy = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            policy.weight.copy_(torch.tensor([[1., 2., 3.]]))
        policy(predicted_obs).sum().backward()
        torch.testing.assert_close(next_state.grad[:, 41:], torch.tensor([[1., 2., 3.]]).repeat(2, 1))
        self.assertEqual(next_state.grad[:, :41].abs().sum().item(), 0)
        torch.testing.assert_close(predicted_obs, obs)


class MDPGeometryTests(unittest.TestCase):
    def setUp(self):
        self.env, self.mdp, physical, self.obs, self.context = setup()
        self.state = self.mdp.capture_state(physical, self.obs)
        self.action = torch.zeros(2, 9)

    def reward(self, state, context=None):
        return self.mdp.reward(self.state, self.action, state,
                               self.context if context is None else context)

    def test_reward_uses_measured_ee_not_base_quaternion(self):
        reference, _ = self.reward(self.state)
        changed_base = self.state.clone()
        changed_base[:, 3:7] = torch.tensor([.5, -.5, .5, -.5])
        torch.testing.assert_close(self.reward(changed_base)[0], reference)
        changed_ee = self.state.clone()
        changed_ee[:, 34] -= 1
        self.assertTrue((self.reward(changed_ee)[0] < reference).all())

    def test_gripper_forward_axis_uses_xyzw_and_native_local_x(self):
        identity = self.state.clone()
        torch.testing.assert_close(self.mdp._get_ee_forward_axis(identity), torch.tensor([[1., 0., 0.]]).repeat(2, 1))
        rotated = identity.clone()
        rotated[:, 37:41] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        torch.testing.assert_close(self.mdp._get_ee_forward_axis(rotated), torch.tensor([[0., 1., 0.]]).repeat(2, 1), atol=1e-6, rtol=0)
        negated = rotated.clone()
        negated[:, 37:41] *= -2
        torch.testing.assert_close(self.mdp._get_ee_forward_axis(negated), self.mdp._get_ee_forward_axis(rotated))

    def test_closure_uses_measured_joint_limits_and_position_not_velocity(self):
        opened, closed = self.state.clone(), self.state.clone()
        opened[:, 22] = -math.pi / 2
        closed[:, 22] = 0
        torch.testing.assert_close(self.mdp._gripper_closure(opened), torch.zeros(2))
        torch.testing.assert_close(self.mdp._gripper_closure(self.state), torch.full((2,), .5))
        torch.testing.assert_close(self.mdp._gripper_closure(closed), torch.ones(2))
        self.assertTrue((self.reward(closed)[0] > self.reward(opened)[0]).all())
        altered_velocity = closed.clone()
        altered_velocity[:, 23] = 100
        torch.testing.assert_close(self.reward(altered_velocity)[0], self.reward(closed)[0])

    def test_each_episodes_target_height_controls_lift_reward_and_proxy_done(self):
        state = self.state.clone()
        state[:, 22] = 0
        context = self.context.clone()
        context[:, 11] = torch.tensor([.6, .8])
        reward, done = self.reward(state, context)
        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertEqual(done.tolist(), [1., 0.])

    def test_world_translation_with_context_preserves_reward_and_done(self):
        shift = torch.tensor([4., -3., 2.])
        current, following, context = self.state.clone(), self.state.clone(), self.context.clone()
        for state in (current, following):
            state[:, :3] += shift
            state[:, 7:10] += shift
            state[:, 34:37] += shift
        context[:, :3] += shift
        context[:, 7:10] += shift
        context[:, 11] += shift[2]
        reference = self.reward(self.state)
        translated = self.mdp.reward(current, self.action, following, context)
        for actual, expected in zip(translated, reference):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0)

    def test_proxy_reward_gradients_reach_ee_orientation_gripper_and_height(self):
        following = self.state.clone()
        following[:, 37:41] = torch.tensor([0., 0., math.sin(.15), math.cos(.15)])
        following.requires_grad_()
        reward, _ = self.reward(following)
        reward.sum().backward()
        self.assertTrue(torch.isfinite(following.grad).all())
        for columns in (slice(34, 37), slice(37, 41), slice(22, 23), slice(9, 10)):
            self.assertGreater(following.grad[:, columns].abs().sum().item(), 0)
        self.assertEqual(following.grad[:, 3:7].abs().sum().item(), 0)

    def test_normalized_episode_clock_terminates_proxy_rollout(self):
        state = self.state.clone()
        state[:, 33] = torch.tensor([.99, 1.])
        self.assertEqual(self.reward(state)[1].tolist(), [0., 1.])


if __name__ == "__main__":
    unittest.main()
