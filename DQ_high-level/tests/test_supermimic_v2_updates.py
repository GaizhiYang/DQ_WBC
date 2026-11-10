"""Run actual v2 optimization, including the transition from warmup to joint."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from supermimic_v2 import DQRobotMDP, SuperMimicAgentV2
from supermimic_v2.networks_v2 import EmpiricalNormalization, WorldModel


def agent_for(device="cpu", num_envs=4, **overrides):
    settings = dict(
        device=device, wm_hidden=(16,), policy_hidden=(16,), critic_hidden=(16,),
        wm_ensemble_size=2, wm_window=3, horizon=3, wm_batch_size=4,
        policy_batch_size=4, replay_capacity=64, min_buffer_size=12,
        warmup_iterations=1, critic_epochs=2, critic_minibatches=3,
    )
    settings.update(overrides)
    ee_pos = torch.tensor([.05, 0., .55], device=device).expand(num_envs, -1).clone()
    ee_orn = torch.tensor([0., 0., 0., 1.], device=device).expand(num_envs, -1).clone()
    env = SimpleNamespace(
        num_envs=num_envs, num_actions=9, num_obs=50, ee_pos=ee_pos, ee_orn=ee_orn,
        num_gripper_joints=2,
        dof_limits_lower=torch.full((20,), -.5, device=device),
        dof_limits_upper=torch.full((20,), .5, device=device),
    )
    mdp = DQRobotMDP(env, device=device)
    return SuperMimicAgentV2(mdp=mdp, **settings)


def fill(agent, steps=6, num_envs=4, terminal=False, with_grad=False):
    for _ in range(steps):
        physical = torch.randn(num_envs, 34, device=agent.device) * .1
        physical[:, 3:7] = physical.new_tensor([0., 0., 0., 1.])
        physical[:, 10:16] = physical.new_tensor([1., 0., 0., 0., 1., 0.])
        physical[:, 9] = .5
        obs = torch.randn(num_envs, agent.mdp.obs_dim, device=agent.device) * .1
        next_obs = obs + torch.randn_like(obs) * .01
        state = agent.mdp.capture_state(physical, obs)
        next_state = agent.mdp.capture_state(physical + torch.randn_like(physical) * .01, next_obs)
        state.requires_grad_(with_grad)
        agent.observe_transition(
            state, torch.randn(num_envs, 9, device=agent.device) * .1,
            torch.ones(num_envs, 1, dtype=torch.long, device=agent.device),
            next_state,
            torch.full((num_envs, 1), int(terminal), dtype=torch.long, device=agent.device),
            torch.randn(num_envs, 16, device=agent.device) * .1,
            obs, next_obs,
        )


def parameters(module):
    return [value.detach().clone() for value in module.parameters()]


class SuperMimicV2UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(12)

    def assert_finite_metrics(self, metrics):
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value), (name, value))

    def assert_changed(self, before, module):
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, module.parameters())))

    def check_actual_updates(self, device):
        agent = agent_for(device)
        fill(agent, with_grad=True)
        for value in agent.replay_buffer.values():
            self.assertFalse(value.requires_grad)
        wm, policy, critic = parameters(agent.wm), parameters(agent.policy), parameters(agent.critic)
        first = agent.update()
        self.assertEqual(first["phase"], "warmup")
        self.assertFalse(first["skipped"])
        self.assert_finite_metrics(first)
        self.assert_changed(wm, agent.wm)
        for old, new in zip(policy, agent.policy.parameters()):
            torch.testing.assert_close(old, new)
        for _ in range(3):
            result = agent.update()
            self.assertEqual(result["phase"], "joint")
            self.assertFalse(result["skipped"])
            self.assert_finite_metrics(result)
            self.assertGreater(result["policy_grad_norm"], 0)
        self.assert_changed(policy, agent.policy)
        self.assert_changed(critic, agent.critic)
        self.assertEqual(agent.world_model_updates, 4)
        self.assertEqual(agent.policy_updates, 3)
        self.assertEqual(agent.iteration, 4)
        self.assertGreater(agent.policy.obs_norm.count.item(), 0)
        self.assertGreater(agent.critic.norm.count.item(), 0)

    def test_actual_warmup_then_multiple_joint_updates_cpu(self):
        self.check_actual_updates("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_actual_warmup_then_multiple_joint_updates_cuda(self):
        self.check_actual_updates("cuda:0")

    def test_actor_bptt_does_not_change_world_model_or_critic_parameters(self):
        agent = agent_for()
        fill(agent)
        agent.update_world_model()
        agent.wm_optimizer.zero_grad(set_to_none=True)
        agent.critic_optimizer.zero_grad(set_to_none=True)
        wm, critic, policy = parameters(agent.wm), parameters(agent.critic), parameters(agent.policy)
        with mock.patch.object(agent, "_update_critic", return_value=0.0):
            metrics = agent.update_policy()
        self.assertGreater(metrics["policy_grad_norm"], 0)
        self.assert_changed(policy, agent.policy)
        for before, module in ((wm, agent.wm), (critic, agent.critic)):
            for old, new in zip(before, module.parameters()):
                torch.testing.assert_close(old, new)
                self.assertIsNone(new.grad)
                self.assertTrue(new.requires_grad)

    def test_skipped_updates_do_not_consume_world_model_warmup(self):
        agent = agent_for()
        for _ in range(4):
            self.assertTrue(agent.update()["skipped"])
            self.assertEqual(agent.phase, "warmup")
        fill(agent)
        self.assertEqual(agent.update()["phase"], "warmup")
        self.assertEqual(agent.update()["phase"], "joint")

    def test_single_sample_and_ensemble_with_more_minibatches_than_samples(self):
        agent = agent_for(num_envs=1, wm_ensemble_size=1, min_buffer_size=1, wm_batch_size=1,
                          policy_batch_size=1, horizon=1, critic_minibatches=4)
        fill(agent, steps=1, num_envs=1)
        for _ in range(3):
            self.assert_finite_metrics(agent.update())
        self.assertEqual(agent.policy_updates, 2)

    def test_terminal_transition_is_a_valid_world_model_target(self):
        agent = agent_for()
        fill(agent, terminal=True)
        windows = agent.sample_windows(16)
        torch.testing.assert_close(windows["mask"][:, 0], torch.ones(16))
        self.assertTrue((windows["mask"][:, 1:] == 0).all())
        before = parameters(agent.wm)
        self.assertFalse(agent.update_world_model()["skipped"])
        self.assert_changed(before, agent.wm)

    def test_windows_follow_same_environment_across_ring_wrap(self):
        agent = agent_for(replay_capacity=17, min_buffer_size=1)
        for step in range(9):
            state = torch.zeros(4, agent.mdp.state_dim)
            state[:, 0] = torch.arange(4)
            state[:, 1] = step
            next_state = state.clone()
            next_state[:, 1] += 1
            agent.observe_transition(state, torch.zeros(4, 9), torch.zeros(4, 1), next_state,
                                     torch.zeros(4, 1, dtype=torch.long), torch.zeros(4, 16),
                                     torch.zeros(4, 50), torch.zeros(4, 50))
        windows = agent.sample_windows(100)
        self.assertTrue((windows["mask"] == 1).all())
        self.assertTrue((windows["s"][:, :, 0] == windows["s"][:, :1, 0]).all())
        torch.testing.assert_close(windows["s"][:, 1:, 1] - windows["s"][:, :-1, 1], torch.ones(100, 2))
        torch.testing.assert_close(windows["s"][:, 1:], windows["s1"][:, :-1])

    def test_short_windows_sample_all_environment_ids_and_mask_missing_future(self):
        agent = agent_for(wm_window=8, min_buffer_size=1)
        state = torch.zeros(4, agent.mdp.state_dim)
        state[:, 0] = torch.arange(4)
        next_state = state.clone()
        next_state[:, :34] += 1
        agent.observe_transition(state, torch.zeros(4, 9), torch.zeros(4), next_state,
                                 torch.zeros(4), torch.zeros(4, 16), torch.zeros(4, 50), torch.zeros(4, 50))
        windows = agent.sample_windows(100)
        self.assertEqual(set(windows["s"][:, 0, 0].tolist()), {0., 1., 2., 3.})
        self.assertTrue((windows["mask"][:, 0] == 1).all())
        self.assertTrue((windows["mask"][:, 1:] == 0).all())
        self.assert_finite_metrics(agent.update_world_model())

    def test_one_insertion_larger_than_capacity_preserves_newest_rows(self):
        agent = agent_for(replay_capacity=3, min_buffer_size=1)
        state = torch.zeros(4, agent.mdp.state_dim)
        state[:, 0] = torch.arange(4)
        agent.observe_transition(state, torch.zeros(4, 9), torch.zeros(4), state,
                                 torch.zeros(4), torch.zeros(4, 16), torch.zeros(4, 50), torch.zeros(4, 50))
        self.assertEqual(agent.buffer_size, 3)
        self.assertEqual(agent.total_transitions, 4)
        self.assertEqual(set(agent.replay_buffer["state"][:, 0].tolist()), {1., 2., 3.})

    def test_lambda_returns_bootstrap_and_stop_at_terminal(self):
        agent = agent_for(gamma=.5, lambda_return=.25)
        rewards = torch.tensor([[1., 1.], [2., 2.]])
        dones = torch.tensor([[0., 1.], [0., 0.]])
        values = torch.tensor([[100., 100.], [10., 10.], [20., 20.]])
        # G1=2+.5*20=12; G0=1+.5*(.75*10+.25*12)=6.25.
        torch.testing.assert_close(agent._compute_lambda_returns(rewards, dones, values),
                                   torch.tensor([[6.25, 1.], [12., 12.]]))

    def test_world_model_denormalizes_outputs_and_normalizer_matches_batch_statistics(self):
        norm = EmpiricalNormalization(2)
        batches = [torch.tensor([[1., 8.], [2., 4.]]), torch.tensor([[6., 9.]])]
        for batch in batches:
            norm.update(batch)
        combined = torch.cat(batches)
        torch.testing.assert_close(norm._mean, combined.mean(0))
        torch.testing.assert_close(norm._var, combined.var(0, unbiased=False))
        model = WorldModel(3, 2, hidden=(4,))
        model.out_norm.update(torch.tensor([[2., 4.], [4., 8.]]))
        for parameter in model.net.parameters():
            parameter.data.zero_()
        torch.testing.assert_close(model(torch.zeros(1, 3)), torch.tensor([[3., 6.]]))

    def test_mdp_critic_shape_matches_network(self):
        agent = agent_for()
        features = agent.mdp.critic_features(torch.zeros(4, agent.mdp.state_dim),
                                           torch.zeros(4, 16), torch.zeros(4, agent.mdp.obs_dim))
        self.assertEqual(features.shape, (4, agent.mdp.critic_dim))
        self.assertEqual(agent.critic(features).shape, (4, 1))

    def test_checkpoint_restores_normalization_counters_and_continues_updates(self):
        agent = agent_for()
        fill(agent)
        agent.update()
        agent.update()
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "model")
            agent.save(prefix)
            restored = agent_for()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                restored.load(prefix)
            self.assertEqual(restored.iteration, agent.iteration)
            self.assertEqual(restored.world_model_updates, agent.world_model_updates)
            self.assertEqual(restored.policy_updates, agent.policy_updates)
            torch.testing.assert_close(restored.policy.obs_norm._mean, agent.policy.obs_norm._mean)
            fill(restored)
            self.assertEqual(restored.update()["phase"], "joint")


if __name__ == "__main__":
    unittest.main()
