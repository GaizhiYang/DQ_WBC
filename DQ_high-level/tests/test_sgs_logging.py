"""SGS recent-episode windows and the actual scalar writer, without a simulator."""

import ast
from collections import defaultdict, deque
from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import torch

from test_sgs_env import TestEnvironment, make_bank, make_active_near_goal_bank


ROOT = Path(__file__).resolve().parents[2]
AGENT_SOURCE = ROOT / "third_party/skrl/skrl/agents/torch/base.py"


def agent_method(name, **extra_namespace):
    tree = ast.parse(AGENT_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {"torch": torch, "np": np, **extra_namespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(AGENT_SOURCE), "exec"), namespace)
    return namespace[name]


def finalize_logging(trainer):
    path = ROOT / "DQ_high-level/train_multistate_DQ_teacher.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "finalize_teacher_logging")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), "exec"), namespace)
    namespace[node.name](trainer)


class MemoryWriter:
    def __init__(self):
        self.values = {}
        self.flush_count = 0

    def add_scalar(self, key, value, step):
        self.values[key] = (float(value), step)

    def flush(self):
        self.flush_count += 1


class TrackingAgent:
    record_transition = agent_method("record_transition")
    write_tracking_data = agent_method("write_tracking_data")
    track_data_latest = agent_method("track_data_latest")

    def __init__(self):
        self.write_interval = 24
        self._cumulative_rewards = None
        self._cumulative_timesteps = None
        self._track_rewards = deque()
        self._track_timesteps = deque()
        self.tracking_data = defaultdict(list)
        self.writer = MemoryWriter()

    def record(self, info, reward=1.):
        self.record_transition(states=torch.zeros(1, 1), actions=torch.zeros(1, 1),
                               rewards=torch.tensor([[reward]]), next_states=torch.zeros(1, 1),
                               terminated=torch.zeros(1, 1, dtype=torch.bool),
                               truncated=torch.zeros(1, 1, dtype=torch.bool),
                               infos=info, timestep=0, timesteps=120000)


def four_family_environment(num_envs=4):
    env = TestEnvironment(num_envs=num_envs, bank=make_bank(num_configs=8, num_families=4))
    env.eval = False
    with redirect_stdout(io.StringIO()):
        env.initialize(tuple(index % 8 for index in range(num_envs)))
    return env


class SGSFamilyLoggingTests(unittest.TestCase):
    def test_active_three_families_keep_recent_windows_in_terminal_and_writer(self):
        env = TestEnvironment(num_envs=303, bank=make_active_near_goal_bank())
        env.eval = False
        ids = list(range(303))
        config_ids = [2 * (index % 3) for index in ids]
        with redirect_stdout(io.StringIO()):
            env.initialize(tuple(config_ids))
            env.finish(ids, [False] * 303, config_ids)
            env.finish(ids, [index % 3 == 2 for index in ids], config_ids)
        self.assertEqual(env.recent_episode_stats.family_count.tolist(), [100] * 3)
        self.assertEqual(env.recent_episode_stats.family_successes.tolist(), [0, 0, 100])
        agent = TrackingAgent()
        agent.record(env.extras)
        agent.write_tracking_data(24, 120000)
        for name, rate in (("full_task", 0.), ("pregrasp", 0.), ("near_goal", 1.)):
            self.assertEqual(agent.writer.values["SuccessRate / " + name], (rate, 24))
            self.assertEqual(agent.writer.values["SGS / episodes_" + name], (100., 24))
        self.assertFalse(any("at_grasp" in key for key in agent.writer.values))
        stream = io.StringIO()
        with redirect_stdout(stream):
            env._print_sgs_success_rates(force=True)
        self.assertIn("near_goal=100.00% (100/100)", stream.getvalue())
        self.assertNotIn("at_grasp", stream.getvalue())
        self.assertNotIn("overall", stream.getvalue())

    def test_four_family_rates_use_completed_episodes_and_original_reset_family(self):
        env = four_family_environment()
        with redirect_stdout(io.StringIO()):
            env.finish([0, 1, 2, 3], [True, False, True, False], [0, 1, 2, 3])
            env.finish([1, 2, 3], [True, False, False], [1, 2, 3])
            env.finish([2], [False], [2])
        self.assertEqual(env.sgs_family_episodes.tolist(), [1, 2, 3, 2])
        self.assertEqual(env.sgs_family_successes.tolist(), [1, 1, 1, 0])
        recent = env.recent_episode_stats
        self.assertEqual(recent.family_count.tolist(), [1, 2, 3, 2])
        self.assertEqual(recent.family_successes.tolist(), [1, 1, 1, 0])
        for name, rate in zip(env.sgs_family_names, (1., .5, 1 / 3, 0.)):
            self.assertAlmostEqual(env.extras["success_rate"]["SGS / success_" + name], rate)
            self.assertAlmostEqual(env.extras["scalar_snapshots"]["SuccessRate / " + name], rate)
            self.assertEqual(env.extras["scalar_snapshots"]["SGS / successes_" + name],
                             env.sgs_family_successes[env.sgs_family_names.index(name)].item())

        for name in ("overall", "family_mean", "sugar_box"):
            self.assertNotIn("SuccessRate / " + name, env.extras["scalar_snapshots"])
            self.assertNotIn("SuccessRate / " + name, env.extras["success_rate"])

        # Neither a still-running success flag nor a cancelled reset is a result.
        env.episode_success.fill_(True)
        old_episodes = env.sgs_family_episodes.clone()
        old_successes = env.sgs_family_successes.clone()
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([7])):
            with redirect_stdout(io.StringIO()):
                env.reset_idx(torch.tensor([3]))
        torch.testing.assert_close(env.sgs_family_episodes, old_episodes)
        torch.testing.assert_close(env.sgs_family_successes, old_successes)
        torch.testing.assert_close(recent.family_count, old_episodes)
        torch.testing.assert_close(recent.family_successes, old_successes)
        self.assertEqual(env.sgs_aborted.item(), 1)

    def test_recent_windows_evict_old_episodes_across_natural_reset_batches(self):
        env = four_family_environment(num_envs=404)
        ids = list(range(env.num_envs))
        config_ids = [index % 8 for index in ids]
        with redirect_stdout(io.StringIO()):
            env.finish(ids, [True] * 404, config_ids)
            env.finish(ids, [False] * 404, config_ids)
            env.finish(ids, [index % 4 == 2 for index in ids], config_ids)
        recent = env.recent_episode_stats
        self.assertEqual(env.sgs_family_episodes.tolist(), [303] * 4)
        self.assertEqual(recent.family_count.tolist(), [100] * 4)
        self.assertEqual(recent.family_successes.tolist(), [0, 0, 100, 0])
        snapshots = env.extras["scalar_snapshots"]
        for name, rate in zip(env.sgs_family_names, (0., 0., 1., 0.)):
            self.assertEqual(snapshots["SuccessRate / " + name], rate)
            self.assertEqual(snapshots["SGS / episodes_" + name], 100.)
        stream = io.StringIO()
        with redirect_stdout(stream):
            env._print_sgs_success_rates(force=True)
        self.assertIn("SuccessRate (each family last 100):", stream.getvalue())
        self.assertIn("at_grasp=100.00% (100/100)", stream.getvalue())
        self.assertNotIn("overall", stream.getvalue())
        self.assertNotIn("family_mean", stream.getvalue())
        self.assertNotIn("/303)", stream.getvalue())

    def test_oversized_same_step_batch_preserves_supplied_completion_order(self):
        env = four_family_environment(num_envs=1105)
        # Events in one physics step have no finer timestamp. Use the supplied
        # order consistently, including a non-sorted reset-index batch.
        ids = torch.cat((torch.arange(500, 1105), torch.arange(500)))
        successes = (torch.arange(env.num_envs) >= 900) | (torch.arange(env.num_envs) < 100)
        families = env.sgs_bank.tensors["family_id"][env.active_config_id[ids]]
        outcomes = successes[ids]
        env.episode_success.copy_(successes)
        env._sgs_pending_done.fill_(True)
        env._sgs_natural_reset = True
        with patch.object(env.sgs_sampler, "sample", return_value=ids % 8):
            with redirect_stdout(io.StringIO()):
                env._reset_sgs_envs(ids)
        env._sgs_natural_reset = False
        recent = env.recent_episode_stats
        for family in range(4):
            expected = outcomes[families == family][-100:]
            self.assertEqual(recent.family_count[family].item(), len(expected))
            self.assertEqual(recent.family_successes[family].item(), expected.sum().item())
        self.assertEqual(env.sgs_family_episodes.sum().item(), 1105)

    def test_logging_does_not_change_sampler_outcomes_probabilities_or_rng(self):
        env = four_family_environment()
        with redirect_stdout(io.StringIO()):
            env.finish([0, 1, 2, 3], [True, False, True, False], [0, 1, 2, 3])
        before = {name: getattr(env.sgs_sampler, name).clone()
                  for name in env.sgs_sampler._TENSOR_STATE}
        probability = env.sgs_sampler.probabilities().clone()
        sampler_rng = env.sgs_sampler.generator.get_state().clone()
        global_rng = torch.random.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):
            # The display buffer itself is independent of sampler history.
            env.recent_episode_stats.record(torch.tensor([0, 1]), torch.tensor([False, True]))
            for _ in range(5):
                env._record_sgs_metrics()
                env._print_sgs_success_rates(force=True)
        for name, value in before.items():
            torch.testing.assert_close(getattr(env.sgs_sampler, name), value)
        torch.testing.assert_close(env.sgs_sampler.probabilities(), probability)
        torch.testing.assert_close(env.sgs_sampler.generator.get_state(), sampler_rng)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
        self.assertEqual(env.recent_episode_stats.family_count.tolist(), [2, 2, 1, 1])

    def test_unfinished_families_are_na_in_terminal_and_do_not_emit_a_zero_rate(self):
        env = four_family_environment()
        stream = io.StringIO()
        with redirect_stdout(stream):
            env._print_sgs_success_rates(force=True)
        message = stream.getvalue()
        self.assertEqual(message.count("N/A"), 4)
        self.assertEqual(message.count("(0/0)"), 4)
        self.assertIn("SuccessRate (each family last 100):", message)
        for name in ("overall", "family_mean", "sugar_box"):
            self.assertNotIn(name, message)
            self.assertNotIn("SuccessRate / " + name, env.extras["scalar_snapshots"])
        for name in env.sgs_family_names:
            self.assertIn(name, message)
            self.assertNotIn("SuccessRate / " + name, env.extras["scalar_snapshots"])
            self.assertEqual(env.extras["scalar_snapshots"]["SGS / episodes_" + name], 0)

    def test_missing_family_remains_unknown_without_suppressing_other_families(self):
        for missing_family in range(4):
            with self.subTest(missing_family=missing_family):
                env = four_family_environment()
                completed = [family for family in range(4) if family != missing_family]
                with redirect_stdout(io.StringIO()):
                    env.finish(completed, [True] * 3, completed)
                for family in completed:
                    name = env.sgs_family_names[family]
                    self.assertEqual(env.extras["scalar_snapshots"]["SuccessRate / " + name], 1.)
                missing_name = env.sgs_family_names[missing_family]
                self.assertNotIn("SuccessRate / " + missing_name, env.extras["scalar_snapshots"])
                stream = io.StringIO()
                with redirect_stdout(stream):
                    env._print_sgs_success_rates(force=True)
                self.assertIn(missing_name + "=N/A (0/0)", stream.getvalue())
                with redirect_stdout(io.StringIO()):
                    env.finish([missing_family], [False], [missing_family])
                self.assertEqual(env.extras["scalar_snapshots"]["SuccessRate / " + missing_name], 0.)

    def test_terminal_prints_four_latest_rates_once_per_interval_without_new_resets(self):
        env = four_family_environment()
        env.recent_episode_stats.record(
            torch.tensor([0, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
            torch.tensor([True, True, False, True, False, False, False, False, False, False]))
        env.global_step_counter = 0
        with redirect_stdout(io.StringIO()):
            env._record_sgs_metrics()
            env._print_sgs_success_rates(force=True)
        stream = io.StringIO()
        with redirect_stdout(stream):
            env.global_step_counter = 23
            env._print_sgs_success_rates()
        self.assertEqual(stream.getvalue(), "")

        # step() must also print when all episodes are still running.
        env.next_done.zero_()
        env.global_step_counter = 24
        with redirect_stdout(stream), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            env.step(torch.zeros(env.num_envs, 9))
            env._print_sgs_success_rates()
        lines = [line for line in stream.getvalue().splitlines() if "full_task" in line]
        self.assertEqual(len(lines), 1)
        for fragment in ("full_task", "pregrasp", "at_grasp", "near_goal",
                         "100.00%", "50.00%", "25.00%", "0.00%",
                         "(1/1)", "(1/2)", "(1/4)", "(0/3)"):
            self.assertIn(fragment, lines[0])
        with redirect_stdout(stream):
            env.global_step_counter = 48
            env._print_sgs_success_rates()
        self.assertEqual(len([line for line in stream.getvalue().splitlines()
                              if "full_task" in line]), 2)

    def test_scalar_snapshots_serialize_latest_window_rate_not_time_average(self):
        logger = TrackingAgent()
        for rate, count, reward in ((1., 1., 1.), (.25, 4., 3.)):
            logger.record({
                "success_rate": {"SGS / success_near_goal": rate},
                "scalar_snapshots": {"SuccessRate / near_goal": rate,
                                     "SGS / success_near_goal": rate,
                                     "SGS / episodes_near_goal": count}}, reward=reward)
        logger.write_tracking_data(timestep=24, timesteps=120000)
        values = logger.writer.values
        self.assertEqual(values["SuccessRate / near_goal"], (.25, 24))
        self.assertEqual(values["SGS / success_near_goal"], (.25, 24))
        self.assertEqual(values["SGS / episodes_near_goal"], (4., 24))
        self.assertEqual(values["Reward / Instantaneous reward (mean)"], (2., 24))
        self.assertFalse(logger.tracking_data)

    def test_wandb_uses_the_same_tensorboard_scalar_writer(self):
        writer = MemoryWriter()
        create_writer = Mock(return_value=writer)
        initialize = agent_method("init", copy=copy, os=os, SummaryWriter=create_writer)
        logger = SimpleNamespace(
            cfg={"experiment": {"wandb": True, "wandb_kwargs": {
                "project": "test-only", "tensorboard": False, "name": "sgs-families"}}},
            models={}, experiment_dir="/unused/sgs-families", write_interval=24,
            checkpoint_interval=0)
        fake_wandb = SimpleNamespace(init=Mock())
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            initialize(logger)
        self.assertTrue(fake_wandb.init.call_args.kwargs["sync_tensorboard"])
        create_writer.assert_called_once_with(log_dir=logger.experiment_dir)
        self.assertIs(logger.writer, writer)

    def test_final_flush_includes_outcomes_settled_after_the_last_agent_write(self):
        env = four_family_environment()
        logger = TrackingAgent()
        logger.record(env.extras)
        logger.write_tracking_data(timestep=96, timesteps=120000)
        self.assertNotIn("SuccessRate / near_goal", logger.writer.values)

        env.global_step_counter = 96
        with redirect_stdout(io.StringIO()):
            env.finish([0, 1, 2, 3], [True, False, True, False], [0, 1, 2, 3])
        stream = io.StringIO()
        with redirect_stdout(stream):
            finalize_logging(SimpleNamespace(env=env, agents=logger, timesteps=120000))
        self.assertIn("final", stream.getvalue())
        self.assertIn("near_goal=0.00% (0/1)", stream.getvalue())
        for name, rate in zip(env.sgs_family_names, (1., 0., 1., 0.)):
            self.assertEqual(logger.writer.values["SuccessRate / " + name], (rate, 96))
            self.assertEqual(logger.writer.values["SGS / episodes_" + name], (1., 96))
        for name in ("overall", "family_mean", "sugar_box"):
            self.assertNotIn("SuccessRate / " + name, logger.writer.values)
        self.assertNotIn("SGS / episodes_overall", logger.writer.values)
        self.assertEqual(logger.writer.flush_count, 1)


if __name__ == "__main__":
    unittest.main()
