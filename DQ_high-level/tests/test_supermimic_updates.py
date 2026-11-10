"""Exercise optimizer boundaries that a forward-only integration test misses."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from supermimic.agent import SuperMimicAgent


class ProxyReward:
    def compute_geometric_proxy_reward(self, state, next_state):
        return -(next_state[:, :3] - 0.5).square().sum(dim=-1)


def make_agent(device="cpu", **overrides):
    settings = dict(
        state_dim=5, action_dim=2, device=device, num_ensemble=2,
        world_model_hidden=[16], policy_hidden=[16], imagination_horizon=3,
        world_model_batch_size=4, world_model_train_steps=2,
        imagination_batch_size=4, policy_update_steps=2,
        replay_capacity=64, min_buffer_size=16,
    )
    settings.update(overrides)
    agent = SuperMimicAgent(**settings)
    agent.attach_state_adapter(ProxyReward())
    return agent


def fill_replay(agent, steps=8, num_envs=4, terminal=False):
    for _ in range(steps):
        state = torch.randn(num_envs, agent.state_dim, device=agent.device)
        action = torch.randn(num_envs, agent.action_dim, device=agent.device)
        agent.observe_transition(
            state, action, torch.ones(num_envs, 1, device=agent.device), state + 0.1,
            torch.full((num_envs, 1), int(terminal), dtype=torch.long, device=agent.device),
            torch.full((num_envs,), 2.0, device=agent.device),
        )


class SuperMimicUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)

    def assert_finite(self, info):
        for key, value in info.items():
            if isinstance(value, (float, int)):
                self.assertTrue(math.isfinite(value), (key, value))

    def check_actor_gradient_isolation(self, device):
        agent = make_agent(device)
        fill_replay(agent)
        actor_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
        fixed_before = [parameter.detach().clone()
                        for module in (agent.world_model, agent.critic) for parameter in module.parameters()]
        # Inspect the actor update alone; value optimization is checked below.
        with mock.patch.object(agent.policy_trainer, "update_critic", return_value={}):
            info = agent.update_policy_via_imagination()
        self.assert_finite(info)
        self.assertGreater(info["policy_grad_norm"], 0.0)
        self.assertEqual(agent.policy_trainer.train_steps, 2)
        self.assertTrue(any(not torch.equal(old, parameter)
                            for old, parameter in zip(actor_before, agent.policy.parameters())))
        fixed_after = [parameter for module in (agent.world_model, agent.critic)
                       for parameter in module.parameters()]
        for old, parameter in zip(fixed_before, fixed_after):
            torch.testing.assert_close(old, parameter)
            self.assertIsNone(parameter.grad)
            self.assertTrue(parameter.requires_grad)

    def test_actor_bptt_only_updates_actor_cpu(self):
        self.check_actor_gradient_isolation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_actor_bptt_only_updates_actor_cuda(self):
        self.check_actor_gradient_isolation("cuda:0")

    def test_critic_does_not_backpropagate_into_inputs_or_targets(self):
        agent = make_agent()
        source = torch.randn(4, 5, requires_grad=True)
        targets = source.square().sum(dim=-1)
        critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
        self.assert_finite(agent.policy_trainer.update_critic(source * 2, targets))
        self.assertIsNone(source.grad)
        self.assertTrue(any(not torch.equal(old, parameter)
                            for old, parameter in zip(critic_before, agent.critic.parameters())))

    def test_world_model_updates_before_eight_environment_timesteps(self):
        agent = make_agent()
        fill_replay(agent, steps=4)
        self.assertTrue(agent.replay_buffer.is_ready(agent.min_buffer_size))
        self.assertFalse(agent.replay_buffer.can_sample_sequences(8))
        self.assert_finite(agent.update_world_model())
        self.assertEqual(agent.world_model_trainer.train_steps, 2)

    def test_world_model_updates_even_when_every_transition_terminates(self):
        agent = make_agent()
        fill_replay(agent, steps=10, terminal=True)
        self.assertFalse(agent.replay_buffer.can_sample_sequences(8))
        self.assert_finite(agent.update_world_model())
        self.assertEqual(agent.world_model_trainer.train_steps, 2)

    def test_single_sample_horizon_and_ensemble_have_finite_metrics(self):
        agent = make_agent(num_ensemble=1, min_buffer_size=1, world_model_batch_size=1,
                           imagination_batch_size=1, imagination_horizon=1)
        fill_replay(agent, steps=1, num_envs=1)
        agent.warmup_iterations = 0
        for _ in range(3):
            self.assert_finite(agent.update())
        state = torch.zeros(1, 5)
        _, uncertainty = agent.world_model.predict_with_uncertainty(state, torch.zeros(1, 2))
        torch.testing.assert_close(uncertainty, torch.zeros_like(uncertainty))

    def test_checkpoint_restores_ring_metadata_then_training_continues(self):
        agent = make_agent()
        fill_replay(agent, steps=20)
        agent.warmup_iterations = 0
        agent.update()
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "agent")
            agent.save(prefix)
            restored = make_agent()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                restored.load(prefix)
            original_buffer, restored_buffer = agent.replay_buffer, restored.replay_buffer
            for name in ("num_envs", "total_added", "ptr", "size"):
                self.assertEqual(getattr(original_buffer, name), getattr(restored_buffer, name))
            torch.testing.assert_close(original_buffer.states, restored_buffer.states)
            torch.testing.assert_close(original_buffer.native_rewards, restored_buffer.native_rewards)
            self.assertTrue(restored_buffer.can_sample_sequences(8))
            self.assertEqual(restored.policy_trainer.train_steps, agent.policy_trainer.train_steps)
            restored.warmup_iterations = 0
            self.assert_finite(restored.update())

    def test_legacy_checkpoint_discards_replay_without_environment_metadata(self):
        agent = make_agent()
        fill_replay(agent)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "agent")
            agent.save(prefix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                payload = torch.load(prefix + "_buffer.pt")
                del payload["buffer_num_envs"]
                del payload["buffer_total_added"]
                torch.save(payload, prefix + "_buffer.pt")
                with self.assertWarnsRegex(UserWarning, "discarded legacy replay"):
                    agent.load(prefix)
            self.assertEqual(len(agent.replay_buffer), 0)
            self.assertIsNone(agent.replay_buffer.num_envs)


if __name__ == "__main__":
    unittest.main()
