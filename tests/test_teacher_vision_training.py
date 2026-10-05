"""Exercise actual PPO/checkpoint integration without importing Isaac Gym.

Run with: python -m unittest discover -s tests -p 'test_teacher_vision_training.py'
"""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from modules.teacher_vision import IMAGE_OBS_DIM, PRIVILEGED_OBS_DIM
from utils.teacher_vision_training import (
    TeacherVisionTrainingState, initialize_from_teacher, make_agent,
    restore_experiment,
)
from utils.teacher_vision_wrapper import TeacherVisionWrapper
from test_teacher_vision_models import original_teacher_classes


class ReusingCameraEnv:
    """Two environments with deliberately mutable observation storage."""

    def __init__(self, origin_step=120000):
        self.device = self.rl_device = "cpu"
        self.num_envs = 2
        self.num_agents = 1
        self.num_states = IMAGE_OBS_DIM + 61
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (PRIVILEGED_OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (9,), np.float32)
        self.global_step_counter = origin_step
        self.local_step = 0
        self.observations = {
            "obs": torch.zeros(self.num_envs, PRIVILEGED_OBS_DIM),
            "states": torch.zeros(self.num_envs, self.num_states),
        }
        self.snapshots = []
        self.executed_actions = []

    def _observe(self):
        privileged = self.observations["obs"]
        student = self.observations["states"]
        privileged[0].fill_(0.1 + self.local_step * 0.03)
        privileged[1].fill_(0.2 + self.local_step * 0.04)
        student[:, :IMAGE_OBS_DIM].fill_(0.25 + self.local_step * 0.01)
        # The wrapper must discard these duplicated state values.
        student[:, IMAGE_OBS_DIM:].fill_(999)
        self.snapshots.append(torch.cat([privileged, student[:, :IMAGE_OBS_DIM]], dim=-1))
        return self.observations

    def reset(self):
        return self._observe()

    def step(self, actions):
        self.executed_actions.append(actions.clone())
        self.local_step += 1
        self.global_step_counter += 1
        rewards = torch.tensor([0.03, 0.07]) + self.local_step * 0.01
        return self._observe(), rewards, torch.zeros(2, dtype=torch.bool), {}


class TeacherVisionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original_policy, cls.original_value = original_teacher_classes()

    def setUp(self):
        torch.manual_seed(31)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = {
            "rollouts": 2,
            "minibatch_size": 2,
            "learning_epochs": 1,
            "learning_rate": 1e-4,
            "checkpoint_interval": 0,
        }
        self.config = {"teacher_vision": dict(self.settings, vision_mode="images", seed=31)}
        self.options = {"roboinfo": True, "observe_gait_commands": True, "rand_control": False}

    def _legacy_checkpoint(self):
        actor = self.original_policy((PRIVILEGED_OBS_DIM,), (9,), "cpu", 1024, 128, None, None, None)
        critic = self.original_value((PRIVILEGED_OBS_DIM,), (9,), "cpu", 1024, 128, None, None, None)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.02)
        data = torch.randn(4, PRIVILEGED_OBS_DIM)
        loss = actor.compute({"states": data}, "policy")[0].square().mean()
        loss = loss + critic.compute({"states": data}, "value")[0].square().mean()
        loss.backward()
        optimizer.step()
        scheduler = KLAdaptiveLR(optimizer)
        scheduler.step(0.0)
        state_scaler = RunningStandardScaler(PRIVILEGED_OBS_DIM, device="cpu")
        value_scaler = RunningStandardScaler(1, device="cpu")
        state_scaler(torch.randn(8, PRIVILEGED_OBS_DIM) * 1.2 + 0.1, train=True)
        value_scaler(torch.linspace(-1, 1, 8).reshape(-1, 1), train=True)
        return {
            "policy": deepcopy(actor.state_dict()),
            "value": deepcopy(critic.state_dict()),
            "state_preprocessor": deepcopy(state_scaler.state_dict()),
            "value_preprocessor": deepcopy(value_scaler.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "scheduler": deepcopy(scheduler.state_dict()),
        }

    def _agent(self, name, config=None, options=None, origin_step=120000):
        env = TeacherVisionWrapper(ReusingCameraEnv(origin_step))
        agent = make_agent(env, self.settings, "images", self.directory.name, name)
        agent.checkpoint_modules["teacher_vision_state"] = TeacherVisionTrainingState(
            env, config if config is not None else self.config,
            options if options is not None else self.options, origin_step,
        )
        self.addCleanup(lambda: agent.writer.close() if hasattr(agent, "writer") else None)
        return env, agent

    def _train_one_rollout(self, env, agent):
        trainer = SequentialTrainer(env=env, agents=agent, cfg={
            "timesteps": 2, "headless": True, "disable_progressbar": True,
            "close_environment_at_exit": False,
        })
        trainer.train()

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_legacy_warm_start_loads_models_scalers_and_keeps_adam_fresh(self):
        _, agent = self._agent("warm_start")
        legacy = self._legacy_checkpoint()
        self.assertTrue(legacy["optimizer"]["state"])
        initial_scheduler = deepcopy(agent.scheduler.state_dict())
        initialize_from_teacher(agent, legacy)
        for name in ("policy", "value"):
            actual = agent.checkpoint_modules[name].state_dict()
            for key, tensor in legacy[name].items():
                torch.testing.assert_close(actual[key], tensor, atol=0, rtol=0)
        for name in ("state_preprocessor", "value_preprocessor"):
            self.assert_nested_equal(agent.checkpoint_modules[name].state_dict(), legacy[name])
        self.assertEqual(agent.optimizer.state_dict()["state"], {})
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], self.settings["learning_rate"])
        self.assert_nested_equal(agent.scheduler.state_dict(), initial_scheduler)
        self.assertEqual(torch.count_nonzero(agent.policy.visual_adapter.weight).item(), 0)

    def test_real_ppo_records_pre_step_observations_and_updates_parameters(self):
        env, agent = self._agent("ppo_update")
        initialize_from_teacher(agent, self._legacy_checkpoint())
        original_policy = agent.policy.net[0].weight.detach().clone()
        original_value = agent.value.net[-1].weight.detach().clone()
        original_scalers = {
            name: deepcopy(agent.checkpoint_modules[name].state_dict())
            for name in ("state_preprocessor", "value_preprocessor")
        }
        self._train_one_rollout(env, agent)
        self.assertTrue(agent.memory.filled)
        self.assertEqual(env._env.local_step, 2)
        observations = agent.memory.get_tensor_by_name("states")
        self.assertEqual(observations.shape, (2, 2, PRIVILEGED_OBS_DIM + IMAGE_OBS_DIM))
        for timestep in range(2):
            torch.testing.assert_close(observations[timestep], env._env.snapshots[timestep])
            self.assertFalse(torch.equal(observations[timestep], env._env.snapshots[timestep + 1]))
        torch.testing.assert_close(
            agent.memory.get_tensor_by_name("actions"), torch.stack(env._env.executed_actions)
        )
        for name in ("log_prob", "values", "returns", "advantages"):
            self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())
        self.assertTrue(agent.optimizer.state_dict()["state"])
        self.assertFalse(torch.equal(original_policy, agent.policy.net[0].weight))
        self.assertFalse(torch.equal(original_value, agent.value.net[-1].weight))
        self.assertGreater(torch.count_nonzero(agent.policy.visual_adapter.weight).item(), 0)
        for name, expected in original_scalers.items():
            self.assert_nested_equal(agent.checkpoint_modules[name].state_dict(), expected)

    def test_frozen_weights_give_unit_ppo_ratio_and_keep_pixels_and_statistics(self):
        env, agent = self._agent("ratio")
        initialize_from_teacher(agent, self._legacy_checkpoint())
        # Exercise a nonzero visual contribution, not just its zero-init bypass.
        with torch.no_grad():
            agent.policy.visual_adapter.weight.normal_(std=0.01)
        scaler = agent.checkpoint_modules["state_preprocessor"]
        initial_stats = deepcopy(scaler.state_dict())
        states, _ = env.reset()
        agent.set_mode("eval")
        with torch.no_grad():
            actions, old_log_prob, _ = agent.act(states, timestep=0, timesteps=2)
        agent.set_mode("train")
        normalized = scaler(states, train=True)
        _, new_log_prob, _ = agent.policy.act(
            {"states": normalized, "taken_actions": actions}, role="policy"
        )
        torch.testing.assert_close(torch.exp(new_log_prob - old_log_prob), torch.ones_like(old_log_prob),
                                   atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normalized[:, PRIVILEGED_OBS_DIM:], states[:, PRIVILEGED_OBS_DIM:],
                                   atol=0, rtol=0)
        self.assert_nested_equal(scaler.state_dict(), initial_stats)

    def test_augmented_checkpoint_restores_adam_scheduler_models_and_metadata(self):
        env, agent = self._agent("before_save")
        initialize_from_teacher(agent, self._legacy_checkpoint())
        self._train_one_rollout(env, agent)
        # Make scheduler state observably different from a fresh instance.
        agent.scheduler.step(0.0)
        checkpoint_path = Path(self.directory.name) / "agent_2.pt"
        agent.save(str(checkpoint_path))
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = saved["teacher_vision_state"]
        self.assertEqual(metadata["origin_step"], 120000)
        self.assertEqual(metadata["global_step"], 120002)
        self.assertEqual(metadata["completed_steps"], 2)
        self.assert_nested_equal(metadata["experiment_config"], self.config)
        self.assert_nested_equal(metadata["environment_options"], self.options)

        # Match the CLI contract: build using the checkpoint's saved config and
        # options, then restore module state and the environment curriculum step.
        restored_env, restored = self._agent(
            "restored", metadata["experiment_config"], metadata["environment_options"],
            origin_step=metadata["origin_step"],
        )
        restored_env._env.global_step_counter = 0
        self.assertEqual(restored.optimizer.state_dict()["state"], {})
        restore_experiment(restored, saved)
        for name in ("policy", "value", "optimizer", "scheduler", "state_preprocessor",
                     "value_preprocessor", "teacher_vision_state"):
            self.assert_nested_equal(restored.checkpoint_modules[name].state_dict(), saved[name])
        self.assertEqual(restored_env.global_step_counter, 120002)
        self.assertTrue(restored.optimizer.state_dict()["state"])
        self.assertEqual(restored.scheduler.get_last_lr(), agent.scheduler.get_last_lr())
        self.assertEqual(restored.optimizer.param_groups[0]["lr"], agent.optimizer.param_groups[0]["lr"])

        # A resumed optimizer must also successfully step with the restored
        # tensors; matching serialized values alone would miss incompatible Adam
        # moment shapes or parameter ordering.
        before = restored.policy.net[-1].weight.detach().clone()
        self._train_one_rollout(restored_env, restored)
        self.assertFalse(torch.equal(before, restored.policy.net[-1].weight))
        self.assertEqual(restored_env.global_step_counter, 120004)


if __name__ == "__main__":
    unittest.main()
