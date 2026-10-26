"""SGS family accounting and the actual scalar writer, without a simulator."""

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

from test_sgs_env import TestEnvironment, make_bank


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


def four_family_environment():
    env = TestEnvironment(bank=make_bank(num_configs=8, num_families=4))
    env.eval = False
    with redirect_stdout(io.StringIO()):
        env.initialize((0, 1, 2, 3))
    return env


class SGSFamilyLoggingTests(unittest.TestCase):
    def test_four_family_rates_use_completed_episodes_and_original_reset_family(self):
        env = four_family_environment()
        with redirect_stdout(io.StringIO()):
            env.finish([0, 1, 2, 3], [True, False, True, False], [0, 1, 2, 3])
            env.finish([1, 2, 3], [True, False, False], [1, 2, 3])
            env.finish([2], [False], [2])
        self.assertEqual(env.sgs_family_episodes.tolist(), [1, 2, 3, 2])
        self.assertEqual(env.sgs_family_successes.tolist(), [1, 1, 1, 0])
        for name, rate in zip(env.sgs_family_names, (1., .5, 1 / 3, 0.)):
            self.assertAlmostEqual(env.extras["success_rate"]["SGS / success_" + name], rate)
            self.assertAlmostEqual(env.extras["scalar_snapshots"]["SuccessRate / " + name], rate)
            self.assertEqual(env.extras["scalar_snapshots"]["SGS / successes_" + name],
                             env.sgs_family_successes[env.sgs_family_names.index(name)].item())

        # Neither a still-running success flag nor a cancelled reset is a result.
        env.episode_success.fill_(True)
        old_episodes = env.sgs_family_episodes.clone()
        old_successes = env.sgs_family_successes.clone()
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([7])):
            with redirect_stdout(io.StringIO()):
                env.reset_idx(torch.tensor([3]))
        torch.testing.assert_close(env.sgs_family_episodes, old_episodes)
        torch.testing.assert_close(env.sgs_family_successes, old_successes)
        self.assertEqual(env.sgs_aborted.item(), 1)

    def test_unfinished_families_are_na_in_terminal_and_do_not_emit_a_zero_rate(self):
        env = four_family_environment()
        stream = io.StringIO()
        with redirect_stdout(stream):
            env._print_sgs_success_rates(force=True)
        message = stream.getvalue()
        self.assertEqual(message.count("N/A"), 4)
        self.assertEqual(message.count("(0/0)"), 4)
        for name in env.sgs_family_names:
            self.assertIn(name, message)
            self.assertNotIn("SuccessRate / " + name, env.extras["scalar_snapshots"])
            self.assertEqual(env.extras["scalar_snapshots"]["SGS / episodes_" + name], 0)

    def test_terminal_prints_four_latest_rates_once_per_interval_without_new_resets(self):
        env = four_family_environment()
        env.sgs_family_successes[:] = torch.tensor([1, 1, 1, 0])
        env.sgs_family_episodes[:] = torch.tensor([1, 2, 4, 3])
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

    def test_scalar_snapshots_serialize_latest_cumulative_rate_not_time_average(self):
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
        self.assertEqual(logger.writer.flush_count, 1)


if __name__ == "__main__":
    unittest.main()
