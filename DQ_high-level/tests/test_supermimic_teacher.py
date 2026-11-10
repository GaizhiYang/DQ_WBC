"""Execute the SuperMimic entrypoint without importing or starting Isaac Gym."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import (apply_pose_prediction_config, configure_teacher_experiment,
                          get_params, load_cfg, teacher_run_end)


def load_trainer(namespace):
    path = ROOT / "train_supermimic_teacher.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == "get_supermimic_trainer")
    namespace["__file__"] = str(path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


class FakeEnv:
    """Wrapper-shaped tensors and explicit, partial reset semantics."""

    def __init__(self, num_envs, events, terminal_steps, interrupt_at=None):
        self.num_envs = num_envs
        self.num_actions = 9
        self.events = events
        self.terminal_steps = terminal_steps
        self.interrupt_at = interrupt_at
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.completed_waves = 0
        self.reset_waves = 0
        self.pending = torch.ones(num_envs, dtype=torch.bool)
        self._env = SimpleNamespace(state=torch.zeros(num_envs, 51), extras={})

    def reset(self):
        self.resets += 1
        ids = self.pending.nonzero(as_tuple=False).flatten()
        values = ids.float() * 10 if self.steps == 0 else -100.0 * (self.steps + 1) - ids.float()
        self._env.state[ids] = values[:, None]
        if self.steps:
            self.reset_waves += 1
        snapshots = {"SGS / episodes_near_goal": self.reset_waves,
                     "SGS / successes_near_goal": int(self.reset_waves > 0)}
        if self.reset_waves:
            snapshots["SuccessRate / near_goal"] = 1.0 / self.reset_waves
        # SGS diagnostics settle during reset, after step() returned its infos.
        self._env.extras["scalar_snapshots"] = snapshots
        self.pending[:] = False
        self.events.append(("reset", self.steps))
        return torch.zeros(self.num_envs, 1), {}

    def step(self, action):
        assert action.shape == (self.num_envs, self.num_actions)
        if self.steps + 1 == self.interrupt_at:
            raise KeyboardInterrupt
        self.steps += 1
        self._env.state += 1
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.long)
        truncated = torch.zeros_like(terminated)
        if self.steps in self.terminal_steps:
            terminated[0] = 1
            truncated[1] = 1
            self.completed_waves += 1
            self._env.extras["success_rate"] = {
                "SuccessRate / sugar_box": 1.0 / (2 * self.completed_waves),
                "SuccessWindow / completed": 2 * self.completed_waves,
                "SuccessWindow / successes": 1,
            }
        self.pending = (terminated | truncated)[:, 0].bool()
        self.events.append(("step", self.steps))
        reward = torch.arange(self.steps, self.steps + self.num_envs, dtype=torch.long)[:, None]
        self._env.extras["reward_components"] = {
            "approach": reward.float().mean() + 1,
            "penalty": torch.tensor(-1.),
            "disabled": torch.tensor(0.),
        }
        # Independent dictionaries ensure the trainer must refresh snapshots
        # after reset rather than relying on a mutable alias of step infos.
        return torch.zeros(self.num_envs, 1), reward, terminated, truncated, deepcopy(self._env.extras)

    def close(self):
        self.closed = True


class TrainerHarness:
    def __init__(self, directory, *, timesteps=10, stop_after=None, num_envs=3,
                 terminal_steps=(1, 10), object_name="sugar_box", wandb=False,
                 interrupt_at=None):
        options = ["--task", "B1Z1PickMulti", "--rl_device", "cpu", "--headless",
                   "--timesteps", str(timesteps), "--num_envs", str(num_envs),
                   "--experiment_dir", directory, "--wandb_name", "entry-smoke"]
        if object_name is not None:
            options += ["--object_name", object_name]
        if stop_after is not None:
            options += ["--stop_after", str(stop_after)]
        if wandb:
            options += ["--wandb"]
        self.args = get_params(options)
        self.events = []
        self.env = FakeEnv(num_envs, self.events, terminal_steps, interrupt_at)
        self.prediction_calls = []
        self.create_calls = []
        self.logs = []
        self.init_calls = []
        self.sdk_calls = []
        self.metric_definitions = []
        self.printed = []
        self.wandb_finished = False
        harness = self

        class Adapter:
            def __init__(self, raw_env, device):
                assert raw_env is harness.env._env
                self.raw_env = raw_env

            def capture_state(self):
                harness.events.append(("capture", harness.env.steps))
                return self.raw_env.state.clone()

            def compute_geometric_proxy_reward(self, state, next_state):
                assert state.shape == next_state.shape == (num_envs, 51)
                return next_state[:, 0] - state[:, 0]

        class Agent:
            warmup_iterations = 1000

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.transitions = []
                self.actions_from = []
                self.updates = 0
                self.checkpoints = []
                self.replay_buffer = SimpleNamespace(capacity=kwargs["replay_capacity"])

            def attach_state_adapter(self, adapter):
                self.adapter = adapter

            def select_action(self, state, **kwargs):
                self.actions_from.append(state.clone())
                return torch.zeros(num_envs, self.kwargs["action_dim"])

            def observe_transition(self, **transition):
                harness.events.append(("observe", harness.env.steps))
                self.transitions.append({name: value.clone() for name, value in transition.items()})

            def update(self):
                self.updates += 1
                harness.events.append(("update", harness.env.steps))
                return {"phase": "warmup"}

            def get_stats(self):
                return {"buffer_stats": {"buffer_size": len(self.transitions) * num_envs,
                                         "buffer_utilization": 0.01}}

            def save(self, path):
                self.checkpoints.append(path)

        def get_prediction(**kwargs):
            self.prediction_calls.append(kwargs)
            return None, None, None

        def create_environment(**kwargs):
            self.create_calls.append(deepcopy(kwargs["cfg"]))
            return self.env

        def finish():
            self.sdk_calls.append("finish")
            self.wandb_finished = True

        def init(**kwargs):
            self.sdk_calls.append("init")
            self.init_calls.append(deepcopy(kwargs))
            return SimpleNamespace(url="https://wandb.ai/test/supermimic/runs/fake")

        def log(info, step):
            self.sdk_calls.append("log")
            self.logs.append((deepcopy(info), step))

        def define_metric(*args, **kwargs):
            self.sdk_calls.append("define_metric")
            self.metric_definitions.append((args, kwargs))

        self.run = load_trainer({
            "get_params": lambda: self.args,
            "set_seed": lambda seed: None,
            "teacher_config_path": lambda *args, **kwargs: str(ROOT / "data/cfg/DQ_teacher.yaml"),
            "load_cfg": load_cfg,
            "configure_teacher_experiment": configure_teacher_experiment,
            "apply_pose_prediction_config": apply_pose_prediction_config,
            "teacher_run_end": teacher_run_end,
            "get_predict_point": get_prediction,
            "create_env": create_environment,
            "DQStateAdapter": Adapter,
            "SuperMimicAgent": Agent,
            "Path": Path, "deepcopy": deepcopy, "os": os, "torch": torch, "np": np,
            "cprint": lambda message, *args, **kwargs: self.printed.append(message),
            "tqdm": lambda iterable, **kwargs: iterable,
            "wandb": SimpleNamespace(init=init, log=log, define_metric=define_metric,
                                      finish=finish),
        })


class SuperMimicTeacherTests(unittest.TestCase):
    def test_native_columns_and_truncations_have_one_scalar_per_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, wandb=True)
            agent = harness.run()
            first = agent.transitions[0]
            self.assertEqual(first["done"].dtype, torch.bool)
            self.assertEqual(first["done"].shape, (3,))
            self.assertEqual(first["done"].tolist(), [True, True, False])
            self.assertEqual(first["native_reward"].dtype, torch.float32)
            self.assertEqual(first["native_reward"].shape, (3,))
            self.assertEqual(first["native_reward"].tolist(), [1, 2, 3])
            self.assertEqual(first["reward"].shape, (3,))
            self.assertEqual(first["reward"].tolist(), [1, 1, 1])
            info, step = harness.logs[0]
            self.assertEqual(info["total_episodes"], 4)
            # Include episodes completed at step 1 as well as the log step 10.
            self.assertEqual(info["episode_reward_mean"], 30.)
            self.assertAlmostEqual(info["episode_reward_std"], float(np.std([1, 2, 54, 63])))
            self.assertEqual(info["episode_length_mean"], 5.)
            self.assertEqual(info["episode_count"], 4)
            self.assertEqual(info["global_step"], 30)
            self.assertEqual(step, 30)
            self.assertTrue(harness.wandb_finished)

    def test_reward_components_average_all_steps_and_final_partial_window(self):
        for timesteps, expected_iterations in ((10, [10]), (13, [10, 13])):
            with self.subTest(timesteps=timesteps), tempfile.TemporaryDirectory() as directory:
                harness = TrainerHarness(directory, timesteps=timesteps, wandb=True)
                harness.run()
                self.assertEqual([info["iteration"] for info, _ in harness.logs], expected_iterations)
                self.assertEqual([step for _, step in harness.logs], [step * 3 for step in expected_iterations])
                for (info, _), expected_reward in zip(harness.logs, [6.5, 13.]):
                    self.assertEqual(info["Reward / Instantaneous reward (mean)"], expected_reward)
                    self.assertEqual(info["Reward / Geometric proxy reward (mean)"], 1.)
                    self.assertEqual(info["Reward components / approach (mean)"], expected_reward + 1)
                    self.assertEqual(info["Reward components / penalty (mean)"], -1.)
                    self.assertEqual(info["Reward components / disabled (mean)"], 0.)
                    component_sum = sum(value for key, value in info.items()
                                        if key.startswith("Reward components / "))
                    self.assertEqual(component_sum, info["Reward / Instantaneous reward (mean)"])
                if timesteps == 13:
                    final = harness.logs[-1][0]
                    self.assertEqual(final["episode_count"], 0)
                    self.assertNotIn("episode_reward_mean", final)
                    self.assertNotIn("episode_reward_std", final)

    def test_success_rates_are_latest_snapshots_including_post_reset_sgs(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=13, wandb=True)
            harness.run()
            first, final = (info for info, _ in harness.logs)
            for info in (first, final):
                self.assertEqual(info["SuccessRate / sugar_box"], .25)
                self.assertEqual(info["SuccessWindow / completed"], 4)
                self.assertEqual(info["SuccessWindow / successes"], 1)
                # The second family snapshot appears only in the step-10 reset.
                self.assertEqual(info["SuccessRate / near_goal"], .5)
                self.assertEqual(info["SGS / episodes_near_goal"], 2)
                self.assertEqual(info["SGS / successes_near_goal"], 1)

    def test_no_completed_episodes_does_not_invent_success_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=13, terminal_steps=(), wandb=True)
            harness.run()
            self.assertEqual(len(harness.logs), 2)
            for info, _ in harness.logs:
                self.assertEqual(info["total_episodes"], 0)
                self.assertEqual(info["episode_count"], 0)
                self.assertFalse(any(key.startswith("SuccessRate / ") for key in info))
                self.assertNotIn("episode_reward_mean", info)

    def test_wandb_configures_iteration_axis_and_prints_the_run_link(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=1, terminal_steps=(), wandb=True)
            agent = harness.run()
            self.assertIn((("iteration",), {}), harness.metric_definitions)
            self.assertIn((("*",), {"step_metric": "iteration", "step_sync": True}),
                          harness.metric_definitions)
            self.assertTrue(any("https://wandb.ai/test/supermimic/runs/fake" in line
                                for line in harness.printed))
            self.assertEqual(harness.sdk_calls.count("init"), 1)
            self.assertEqual(harness.sdk_calls.count("finish"), 1)
            init = harness.init_calls[0]
            self.assertEqual(init["project"], harness.args.wandb_project)
            self.assertEqual(init["name"], harness.args.wandb_name)
            config = init["config"]
            self.assertEqual(config["algorithm"], "SuperMimic")
            self.assertEqual(config["supermimic"],
                             {**agent.kwargs, "warmup_iterations": agent.warmup_iterations})
            self.assertTrue({"state_dim", "action_dim", "world_model_lr", "policy_lr", "critic_lr",
                             "imagination_horizon", "num_ensemble", "world_model_hidden",
                             "policy_hidden", "world_model_batch_size", "imagination_batch_size",
                             "policy_update_steps", "replay_capacity", "min_buffer_size"}
                            .issubset(config["supermimic"]))
            self.assertEqual(config["state_dim"], agent.kwargs["state_dim"])
            self.assertEqual(config["action_dim"], harness.env.num_actions)
            self.assertEqual(config["num_envs"], harness.env.num_envs)
            self.assertEqual(config["env"]["numEnvs"], harness.env.num_envs)
            self.assertEqual(config["seed"], harness.args.seed)
            self.assertEqual(config["timesteps"], harness.args.timesteps)
            self.assertEqual(config["arguments"], vars(harness.args))
            self.assertEqual(config["logging"]["interval_vector_steps"], 10)
            self.assertEqual(config["logging"]["x_axis"], "iteration")
            self.assertTrue({"reward_components", "episode_rewards", "family_success_window"}
                            .issubset(config["logging"]))

    def test_disabled_wandb_does_not_call_the_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=13, wandb=False)
            harness.run()
            self.assertEqual(harness.sdk_calls, [])
            self.assertEqual(harness.logs, [])
            self.assertTrue(harness.env.closed)

    def test_wandb_finish_failure_still_closes_environment_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=1, terminal_steps=(), wandb=True)
            with patch.object(harness.run.__globals__["wandb"], "finish",
                              side_effect=RuntimeError("W&B finish failed")) as finish:
                with self.assertRaisesRegex(RuntimeError, "W&B finish failed"):
                    harness.run()
                finish.assert_called_once_with()
            self.assertTrue(harness.env.closed)

    def test_interrupt_flushes_completed_steps_and_closes_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=20, interrupt_at=14, wandb=True)
            with self.assertRaises(KeyboardInterrupt):
                harness.run()
            self.assertEqual(harness.env.steps, 13)
            self.assertEqual([info["iteration"] for info, _ in harness.logs], [10, 13])
            self.assertEqual([step for _, step in harness.logs], [30, 39])
            final = harness.logs[-1][0]
            self.assertEqual(final["global_step"], 39)
            self.assertEqual(final["Reward / Instantaneous reward (mean)"], 13.)
            self.assertEqual(harness.sdk_calls.count("finish"), 1)
            self.assertTrue(harness.env.closed)

    def test_terminal_transition_is_stored_before_partial_reset_and_recapture(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=2, terminal_steps=(1,))
            agent = harness.run()
            self.assertEqual(harness.events, [
                ("reset", 0), ("capture", 0),
                ("step", 1), ("capture", 1), ("observe", 1),
                ("reset", 1), ("capture", 1), ("update", 1),
                ("step", 2), ("capture", 2), ("observe", 2), ("update", 2),
            ])
            self.assertEqual(agent.transitions[0]["next_state"][:, 0].tolist(), [1, 11, 21])
            self.assertEqual(agent.transitions[1]["state"][:, 0].tolist(), [-200, -201, 21])
            self.assertTrue(torch.equal(agent.actions_from[1], agent.transitions[1]["state"]))
            self.assertEqual(harness.env.resets, 2)
            self.assertTrue(harness.env.closed)

    def test_timesteps_counts_vector_steps_even_when_below_environment_count(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=3, num_envs=8, terminal_steps=())
            agent = harness.run()
            self.assertEqual(harness.env.steps, 3)
            self.assertEqual(agent.updates, 3)
            self.assertEqual(len(agent.transitions), 3)
            self.assertEqual(Path(agent.checkpoints[-1]).name, "agent_24")

    def test_stop_after_caps_execution_without_changing_the_training_plan(self):
        for stop_after, expected in ((2, 2), (12, 5)):
            with self.subTest(stop_after=stop_after), tempfile.TemporaryDirectory() as directory:
                harness = TrainerHarness(directory, timesteps=5, stop_after=stop_after, terminal_steps=())
                agent = harness.run()
                self.assertEqual(harness.env.steps, expected)
                self.assertEqual(agent.updates, expected)
                self.assertEqual(harness.args.timesteps, 5)
                self.assertEqual(Path(agent.checkpoints[-1]).name, f"agent_{expected * 3}")

    def test_single_object_filter_preserves_prediction_dictionary_index(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = TrainerHarness(directory, timesteps=1, terminal_steps=())
            harness.run()
            self.assertEqual(list(harness.create_calls[0]["env"]["asset"]["asset_multi"]), ["sugar_box"])
            self.assertEqual(harness.prediction_calls[0]["object_indices"], [1])
            self.assertEqual(harness.prediction_calls[0]["num_env"], 3)
            saved = load_cfg(Path(directory) / "entry-smoke/DQ_teacher.yaml")
            self.assertEqual(saved["env"]["asset"]["asset_multi"]["sugar_box"]["dict_idx"], 1)


if __name__ == "__main__":
    unittest.main()
