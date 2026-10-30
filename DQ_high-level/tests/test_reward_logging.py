"""Weighted reward accounting through the production reward and logging paths.

Only reward inputs and the simulator are replaced; aggregation, trainer cadence,
final flushing, and TensorBoard serialization execute repository code on CPU.
"""

from collections import Counter
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from test_sgs_env import source_method
from test_success_logging import COUNT, RATE, LoggingAgent, SequentialTrainer


TOTAL = "Reward / Instantaneous reward (mean)"
FAMILY_RATE = "SuccessRate / near_goal"


def component_tag(name):
    return "Reward components / {} (mean)".format(name)


class RewardLoggingEnv:
    """Two different environments expose weighting, clipping, and averaging errors."""

    _prepare_reward_function = source_method("b1z1_base.py", "B1Z1Base", "_prepare_reward_function")
    compute_reward = source_method("b1z1_base.py", "B1Z1Base", "compute_reward")
    num_envs = 2
    num_agents = 1
    device = "cpu"

    def __init__(self, clipping=False, dtype=torch.float32, termination_scale=-3.):
        self.cfg = {"reward": {"only_positive_rewards": clipping}}
        self.rew_buf = torch.zeros(self.num_envs, dtype=dtype)
        self.reward_scales = {"vector": 2., "negative": -1., "scalar": .5,
                              "disabled": 0., "termination": termination_scale}
        self.extras = {}
        self.calls = Counter()
        self.step_number = 0
        self._prepare_reward_function()

    def _reward_vector(self):
        self.calls["vector"] += 1
        return self.rew_buf.new_tensor([1., 3.]) * self.step_number, 101.

    def _reward_negative(self):
        self.calls["negative"] += 1
        return self.rew_buf.new_tensor([6., 2.]) * self.step_number, 202.

    def _reward_scalar(self):
        self.calls["scalar"] += 1
        return 2., 303.

    def _reward_disabled(self):
        raise AssertionError("a zero-weight reward must not be evaluated for logging")

    def _reward_termination(self):
        self.calls["termination"] += 1
        return self.rew_buf.new_tensor([1., 0.]), 404.

    def reset(self):
        return torch.zeros(self.num_envs, 1), {}

    def step(self, actions):
        self.step_number += 1
        self.compute_reward()
        # Deliberately change snapshots on every step: the final snapshot and
        # the reward-window average must not be confused with one another.
        rate = self.step_number / 1000.
        self.extras["success_rate"] = {RATE: rate, COUNT: self.step_number}
        self.extras["scalar_snapshots"] = {FAMILY_RATE: rate / 2.}
        flags = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        return (torch.zeros(self.num_envs, 1), self.rew_buf[:, None],
                flags, flags.clone(), self.extras)


def make_reward_trainer(timesteps, clipping=False, write_interval=240):
    env = RewardLoggingEnv(clipping=clipping)
    agent = LoggingAgent({}, device="cpu", cfg={"experiment": {
        "write_interval": write_interval, "checkpoint_interval": 0,
    }})
    trainer = SequentialTrainer(env=env, agents=agent, cfg={
        "timesteps": timesteps, "headless": True, "disable_progressbar": True,
        "close_environment_at_exit": False,
    })
    return trainer, agent, env


def expected_components(mean_step, clipping):
    # Across environments, the two weighted vectors cancel before clipping.
    # The terminal penalty remains negative because it is added after clipping.
    values = {"vector": 4. * mean_step, "negative": -4. * mean_step,
              "scalar": 1., "disabled": 0., "termination": -1.5}
    if clipping:
        values["positive_clipping_adjustment"] = 2. * mean_step - .5
    return values


class RewardLoggingTests(unittest.TestCase):
    def test_all_weighted_terms_account_for_reward_without_repeating_functions(self):
        for dtype in (torch.float32, torch.float64):
            for clipping in (False, True):
                with self.subTest(dtype=dtype, clipping=clipping):
                    env = RewardLoggingEnv(clipping=clipping, dtype=dtype)
                    self.assertIn("disabled", env.reward_component_names)
                    self.assertNotIn("disabled", env.reward_names)
                    for step in (1, 2):
                        env.step(torch.zeros(2, 1))
                        components = env.extras["reward_components"]
                        expected = expected_components(step, clipping)
                        self.assertEqual(set(components), set(expected))
                        for name, value in components.items():
                            self.assertEqual(value.shape, torch.Size([]))
                            self.assertEqual(value.device, env.rew_buf.device)
                            self.assertEqual(value.dtype, dtype)
                            self.assertFalse(value.requires_grad)
                            self.assertAlmostEqual(value.item(), expected[name])
                        torch.testing.assert_close(sum(components.values()), env.rew_buf.mean())
                        reward = [-3., 4. * step + 1.] if clipping else [-4. * step - 2., 4. * step + 1.]
                        torch.testing.assert_close(env.rew_buf, env.rew_buf.new_tensor(reward))
                        self.assertEqual(env.calls, Counter({name: step for name in
                                                           ("vector", "negative", "scalar", "termination")}))
                    # Existing episode accounting still accumulates weighted
                    # values, and its diagnostic metrics remain independent.
                    self.assertEqual(env.episode_sums["vector"].tolist(), [6., 18.])
                    self.assertEqual(env.episode_sums["scalar"].tolist(), [2., 2.])
                    self.assertEqual(env.episode_metric_sums["vector"].tolist(), [202., 202.])

    def test_disabled_termination_is_visible_as_zero_without_calling_it(self):
        env = RewardLoggingEnv(termination_scale=0.)
        with patch.object(env, "_reward_termination", side_effect=AssertionError("disabled termination")):
            env.step(torch.zeros(2, 1))
        components = env.extras["reward_components"]
        self.assertEqual(components["termination"].item(), 0.)
        self.assertEqual(components["disabled"].item(), 0.)
        torch.testing.assert_close(sum(components.values()), env.rew_buf.mean())

    def test_ten_iteration_interval_and_final_partial_keep_window_means(self):
        for clipping in (False, True):
            for timesteps in (240, 241):
                with self.subTest(clipping=clipping, timesteps=timesteps):
                    trainer, agent, env = make_reward_trainer(timesteps, clipping)
                    trainer.train()
                    windows = [(240, 120.5)]
                    if timesteps == 241:
                        windows.append((241, 241.))
                    expected_steps = [end for end, _ in windows]
                    for name in expected_components(1., clipping):
                        logged = agent.writer.values(component_tag(name))
                        self.assertEqual([step for _, step in logged], expected_steps)
                        for (value, end), (_, mean_step) in zip(logged, windows):
                            self.assertAlmostEqual(value, expected_components(mean_step, clipping)[name])
                    for end, mean_step in windows:
                        total = dict((step, value) for value, step in agent.writer.values(TOTAL))[end]
                        components = [dict((step, value) for value, step in agent.writer.values(component_tag(name)))[end]
                                      for name in expected_components(mean_step, clipping)]
                        self.assertAlmostEqual(sum(components), total)
                    self.assertEqual(agent.writer.values(RATE), [(end / 1000., end) for end in expected_steps])
                    self.assertEqual(agent.writer.values(COUNT), [(float(end), end) for end in expected_steps])
                    self.assertEqual(agent.writer.values(FAMILY_RATE), [(end / 2000., end) for end in expected_steps])
                    self.assertEqual(agent.write_interval, 240)
                    self.assertEqual(agent.writer.flush_count, 1)
                    self.assertFalse(agent.tracking_data)
                    self.assertEqual(env.calls["termination"], timesteps)

    def test_disabled_logging_does_not_write_component_records(self):
        trainer, agent, _ = make_reward_trainer(2, write_interval=0)
        trainer.train()
        self.assertEqual(agent.writer.scalars, [])
        self.assertEqual(agent.writer.flush_count, 0)

    def test_component_tags_and_accounting_survive_tensorboard_roundtrip(self):
        trainer, agent, _ = make_reward_trainer(2, clipping=True)
        with tempfile.TemporaryDirectory() as directory:
            agent.writer = SummaryWriter(log_dir=directory)
            try:
                trainer.train()
            finally:
                agent.writer.close()
            events = EventAccumulator(directory, size_guidance={"scalars": 0}).Reload()
            expected = expected_components(1.5, clipping=True)
            actual_tags = {tag for tag in events.Tags()["scalars"] if tag.startswith("Reward components / ")}
            self.assertEqual(actual_tags, {component_tag(name) for name in expected})
            values = []
            for name, value in expected.items():
                recorded = events.Scalars(component_tag(name))
                self.assertEqual(len(recorded), 1)
                self.assertEqual(recorded[0].step, 2)
                self.assertAlmostEqual(recorded[0].value, value)
                values.append(recorded[0].value)
            self.assertAlmostEqual(sum(values), events.Scalars(TOTAL)[0].value)


if __name__ == "__main__":
    unittest.main()
