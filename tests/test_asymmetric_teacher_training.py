"""CPU integration of visual warm start, real PPO and version 2 state."""

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
from learning.asymmetric_teacher_trainer import AsymmetricTeacherTrainer
from modules.teacher_vision import TeacherVisionPolicy, TeacherVisionValue, PACKED_OBS_DIM
from modules.asymmetric_teacher import KEEP_FUSION_INDICES
from utils.asymmetric_teacher_training import (
    AsymmetricTeacherTrainingState, initialize_from_teacher, make_agent,
    restore_experiment, validate_settings, feature_flags,
)
from utils.config import load_cfg
from utils.teacher_vision_wrapper import TeacherVisionWrapper
from test_teacher_vision_training import ReusingCameraEnv


class SyntheticFeatureWrapper(TeacherVisionWrapper):
    """Supply nonzero task/belief inputs while exercising genuine PPO updates."""

    def __init__(self, env, m1=False, m2=False):
        super().__init__(env)
        self.m1, self.m2 = m1, m2
        self._teacher_vision_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (PACKED_OBS_DIM + 5 * m1 + 16 * m2,), np.float32)

    def _pack_observation(self, observations):
        packed = super()._pack_observation(observations)
        parts = [packed]
        if self.m1:
            parts.append(torch.tensor([[.8, .2, .1, .3, 1.], [.4, .1, .3, .1, 0.]]))
        if self.m2:
            parts.append(torch.linspace(-.2, .8, 32).reshape(2, 16))
        return torch.cat(parts, dim=-1)


class AsymmetricTeacherTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        cls.teacher = TeacherVisionPolicy((PACKED_OBS_DIM,), (9,), "cpu")
        value = TeacherVisionValue((PACKED_OBS_DIM,), (9,), "cpu")
        with torch.no_grad():
            cls.teacher.visual_adapter.weight.normal_(std=0.01)
        optimizer = torch.optim.Adam(
            list(cls.teacher.parameters()) + list(value.parameters()), lr=0.02)
        observations = torch.randn(2, PACKED_OBS_DIM)
        loss = cls.teacher.compute({"states": observations}, "policy")[0].square().mean()
        loss += value.compute({"states": observations}, "value")[0].square().mean()
        loss.backward()
        optimizer.step()
        scheduler = KLAdaptiveLR(optimizer)
        scheduler.step(0.0)
        state_scaler = RunningStandardScaler(1276, device="cpu")
        value_scaler = RunningStandardScaler(1, device="cpu")
        state_scaler(torch.randn(8, 1276) + 0.25, train=True)
        value_scaler(torch.linspace(-1, 1, 8)[:, None], train=True)
        cls.source = {
            "policy": deepcopy(cls.teacher.state_dict()),
            "value": deepcopy(value.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "scheduler": deepcopy(scheduler.state_dict()),
            "state_preprocessor": deepcopy(state_scaler.state_dict()),
            "value_preprocessor": deepcopy(value_scaler.state_dict()),
            "teacher_vision_state": {
                "experiment_config": {"teacher_vision": {"vision_mode": "images"}},
                "global_step": 120048,
            },
        }

    def setUp(self):
        torch.manual_seed(71)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = deepcopy(load_cfg(str(ROOT / "DQ_high-level/data/cfg/DQ_asymmetric_teacher.yaml"))["asymmetric_teacher"])
        self.settings.pop("schedule", None)
        self.settings.update({
            "rollouts": 2, "minibatch_size": 2, "learning_epochs": 1,
            "learning_rate": 1e-4, "checkpoint_interval": 0,
            "eval_interval": 0, "eval_steps": 2,
            "time_limit_semantics": "task_deadline", "seed": 71,
            "total_steps": 16, "mode": "m0",
        })
        self.config = {"asymmetric_teacher": self.settings}
        self.options = {"roboinfo": True, "observe_gait_commands": True}

    def _agent(self, name, *, deterministic=False, mode="m0"):
        settings = deepcopy(self.settings)
        settings["mode"] = mode
        env = SyntheticFeatureWrapper(ReusingCameraEnv(120048), **feature_flags(settings))
        agent = make_agent(env, settings, self.directory.name, name,
                           deterministic=deterministic)
        state = AsymmetricTeacherTrainingState(env, {"asymmetric_teacher": settings}, self.options, 120048)
        agent.checkpoint_modules["asymmetric_teacher_state"] = state
        self.addCleanup(lambda: agent.writer.close() if hasattr(agent, "writer") else None)
        return env, agent, state

    def _train_to(self, env, agent, steps):
        trainer = AsymmetricTeacherTrainer(env, agent, cfg={
            "timesteps": steps, "headless": True, "disable_progressbar": True,
            "close_environment_at_exit": False, "eval_interval": 0,
            "eval_steps": 2, "checkpoint_interval": 0,
        })
        trainer.train()
        return trainer

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

    def test_visual_warm_start_preserves_visual_weights_critic_scalers_with_fresh_adam(self):
        env, agent, state = self._agent("migration")
        self.assertTrue(self.source["optimizer"]["state"])
        scheduler_before = deepcopy(agent.scheduler.state_dict())
        initialize_from_teacher(agent, self.source)
        self.assertEqual(agent.optimizer.state_dict()["state"], {})
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], 1e-4)
        self.assert_nested_equal(agent.scheduler.state_dict(), scheduler_before)
        for name in ("value", "state_preprocessor", "value_preprocessor"):
            self.assert_nested_equal(agent.checkpoint_modules[name].state_dict(), self.source[name])
        torch.testing.assert_close(agent.policy.actor.visual_adapter.weight,
                                   self.source["policy"]["visual_adapter.weight"], atol=0, rtol=0)
        self.assertGreater(torch.count_nonzero(agent.policy.actor.visual_adapter.weight).item(), 0)
        torch.testing.assert_close(agent.policy.actor.proprio_adapter.weight,
                                   self.source["policy"]["net.0.weight"][:, KEEP_FUSION_INDICES], atol=0, rtol=0)
        torch.testing.assert_close(agent.policy.log_std_parameter,
                                   self.source["policy"]["log_std_parameter"], atol=0, rtol=0)
        self.assertFalse(any(key.startswith("transition.") for key in agent.policy.state_dict()))
        self.assertEqual(state.state_dict()["global_step"], 120048)

    def test_real_ppo_updates_deployable_actor_and_critic_but_freezes_scalers(self):
        env, agent, state = self._agent("ppo")
        initialize_from_teacher(agent, self.source)
        actor_before = agent.policy.actor.action_head[-1].weight.detach().clone()
        visual_before = agent.policy.actor.visual_adapter.weight.detach().clone()
        critic_before = agent.value.net[-1].weight.detach().clone()
        self._train_to(env, agent, 2)
        self.assertFalse(torch.equal(actor_before, agent.policy.actor.action_head[-1].weight))
        self.assertFalse(torch.equal(visual_before, agent.policy.actor.visual_adapter.weight))
        self.assertFalse(torch.equal(critic_before, agent.value.net[-1].weight))
        self.assertTrue(all(parameter.requires_grad for parameter in agent.policy.parameters()))
        for name in ("state_preprocessor", "value_preprocessor"):
            self.assert_nested_equal(agent.checkpoint_modules[name].state_dict(), self.source[name])
        recorded = agent.memory.get_tensor_by_name("states")
        for step in range(2):
            torch.testing.assert_close(recorded[step], env._env.snapshots[step], atol=0, rtol=0)
        for name in ("log_prob", "returns", "advantages"):
            self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())
        self.assertEqual(state.completed_steps, 2)
        self.assertEqual(env.global_step_counter, 120050)
        self.assertFalse(agent.cfg["time_limit_bootstrap"])

    def test_m1_and_m2_new_adapters_are_zero_initialized_and_learn_in_real_ppo(self):
        for mode in ("m1", "m2"):
            with self.subTest(mode=mode):
                env, agent, state = self._agent("ppo_" + mode, mode=mode)
                initialize_from_teacher(agent, self.source)
                adapters = [agent.value.task_adapter]
                if mode == "m2":
                    adapters.extend([agent.policy.actor.belief_adapter, agent.value.belief_adapter])
                for adapter in adapters:
                    self.assertIsNone(adapter.bias)
                    self.assertEqual(torch.count_nonzero(adapter.weight).item(), 0)
                for name, tensor in self.source["value"].items():
                    torch.testing.assert_close(agent.value.state_dict()[name], tensor, atol=0, rtol=0)
                observations, _ = env.reset()
                normalized = agent._state_preprocessor(observations, train=True)
                # Camera pixels, normalized task state and belief are already
                # in their intended units and bypass the frozen teacher scaler.
                torch.testing.assert_close(normalized[:, 1276:], observations[:, 1276:], atol=0, rtol=0)
                self._train_to(env, agent, 2)
                for adapter in adapters:
                    self.assertTrue(torch.isfinite(adapter.weight).all())
                    self.assertGreater(torch.count_nonzero(adapter.weight).item(), 0)
                self.assertEqual(state.completed_steps, 2)
                for name in ("state_preprocessor", "value_preprocessor"):
                    self.assert_nested_equal(agent.checkpoint_modules[name].state_dict(), self.source[name])

    def test_warm_start_preserves_trained_adapters_and_zeros_only_new_branches(self):
        env, m1, _ = self._agent("source_m1", mode="m1")
        initialize_from_teacher(m1, self.source)
        self._train_to(env, m1, 2)
        source = {key: deepcopy(module.state_dict()) for key, module in m1.checkpoint_modules.items()}
        m2_env, m2, _ = self._agent("upgrade_m2", mode="m2")
        initialize_from_teacher(m2, source)
        torch.testing.assert_close(m2.value.task_adapter.weight, m1.value.task_adapter.weight, atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(m2.policy.actor.belief_adapter.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(m2.value.belief_adapter.weight).item(), 0)
        self.assertEqual(m2.optimizer.state_dict()["state"], {})
        self._train_to(m2_env, m2, 2)
        source = {key: deepcopy(module.state_dict()) for key, module in m2.checkpoint_modules.items()}
        _, restarted, state = self._agent("new_m2", mode="m2")
        initialize_from_teacher(restarted, source)
        self.assert_nested_equal(restarted.policy.state_dict(), source["policy"])
        self.assert_nested_equal(restarted.value.state_dict(), source["value"])
        self.assertEqual(restarted.optimizer.state_dict()["state"], {})
        self.assertEqual(state.completed_steps, 0)
        _, m0, _ = self._agent("cannot_drop_belief")
        with self.assertRaises((ValueError, RuntimeError)):
            initialize_from_teacher(m0, source)

    def test_version_one_warm_start_discards_old_privileged_branch_and_optimizer(self):
        _, source_agent, _ = self._agent("v1_fixture")
        initialize_from_teacher(source_agent, self.source)
        source = {key: deepcopy(module.state_dict()) for key, module in source_agent.checkpoint_modules.items()}
        source["asymmetric_teacher_state"]["version"] = 1
        # Construct the old checkpoint layout from the original teacher, with
        # a live privilege coefficient to ensure migration cannot retain it.
        source["policy"]["_alpha"] = torch.tensor(0.75, dtype=torch.float64)
        removed_columns = tuple(range(6)) + tuple(range(58, 63)) + tuple(range(72, 206))
        source["policy"]["transition.projection.weight"] = self.source["policy"]["net.0.weight"][:, removed_columns].clone()
        for key, tensor in self.source["policy"].items():
            if key.startswith(("feature_encoder.", "key_proj.", "query_proj.", "value_proj.", "output_proj.")):
                source["policy"]["transition." + key] = tensor.clone()
        source["optimizer"] = self.source["optimizer"]
        _, migrated, state = self._agent("v1_to_m2", mode="m2")
        initialize_from_teacher(migrated, source)
        self.assertFalse(any(key.startswith("transition.") or key == "_alpha" for key in migrated.policy.state_dict()))
        torch.testing.assert_close(migrated.policy.actor.proprio_adapter.weight,
                                   source_agent.policy.actor.proprio_adapter.weight, atol=0, rtol=0)
        self.assertEqual(migrated.optimizer.state_dict()["state"], {})
        self.assertEqual(state.completed_steps, 0)

    def test_complete_checkpoint_restores_adam_scheduler_and_continues_ppo(self):
        env, agent, state = self._agent("before_save")
        initialize_from_teacher(agent, self.source)
        self._train_to(env, agent, 6)
        state.best_eval_score = 0.625
        state.last_eval_step = 4
        agent.scheduler.step(0.0)
        path = Path(self.directory.name) / "agent_6.pt"
        agent.save(str(path))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        metadata = saved["asymmetric_teacher_state"]
        self.assertEqual(metadata["version"], 2)
        self.assertEqual((metadata["origin_step"], metadata["completed_steps"], metadata["global_step"]),
                         (120048, 6, 120054))
        restored_env, restored, restored_state = self._agent("restored")
        restored_env._env.global_step_counter = 0
        restore_experiment(restored, saved)
        for name in ("policy", "value", "optimizer", "scheduler", "state_preprocessor",
                     "value_preprocessor", "asymmetric_teacher_state"):
            self.assert_nested_equal(restored.checkpoint_modules[name].state_dict(), saved[name])
        self.assertEqual(restored_env.global_step_counter, 120054)
        self.assertEqual(restored.scheduler.get_last_lr(), agent.scheduler.get_last_lr())
        before = restored.policy.actor.action_head[-1].weight.detach().clone()
        old_steps = [int(v["step"]) for v in restored.optimizer.state.values() if "step" in v]
        self._train_to(restored_env, restored, 8)
        new_steps = [int(v["step"]) for v in restored.optimizer.state.values() if "step" in v]
        self.assertFalse(torch.equal(before, restored.policy.actor.action_head[-1].weight))
        self.assertTrue(all(new > old for old, new in zip(old_steps, new_steps)))
        self.assertEqual(restored_env._env.local_step, 2)
        self.assertEqual(restored_state.completed_steps, 8)
        self.assertEqual(restored_env.global_step_counter, 120056)

        # A play-only load can omit optimizer state without changing the policy.
        _, evaluation, _ = self._agent("evaluation", deterministic=True)
        restore_experiment(evaluation, {k: v for k, v in saved.items()
                                       if k not in ("optimizer", "scheduler")}, evaluation=True)
        self.assert_nested_equal(evaluation.policy.state_dict(), saved["policy"])

        # A warm start intentionally retains weights but starts a new optimizer
        # and training budget, even when its source is another version 2 run.
        _, warm_started, warm_state = self._agent("new_experiment")
        initialize_from_teacher(warm_started, saved)
        self.assert_nested_equal(warm_started.policy.state_dict(), saved["policy"])
        self.assertEqual(warm_started.optimizer.state_dict()["state"], {})
        self.assertEqual(warm_state.completed_steps, 0)

    def test_resume_rejects_incomplete_or_inconsistent_checkpoint(self):
        _, agent, state = self._agent("bad_restore")
        initialize_from_teacher(agent, self.source)
        checkpoint = {key: deepcopy(module.state_dict())
                      for key, module in agent.checkpoint_modules.items()}
        for key in ("optimizer", "scheduler", "asymmetric_teacher_state"):
            invalid = dict(checkpoint)
            del invalid[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                restore_experiment(agent, invalid)
        invalid = deepcopy(checkpoint)
        invalid["asymmetric_teacher_state"]["version"] = 1
        with self.assertRaisesRegex(ValueError, "teacher_init_checkpoint"):
            restore_experiment(agent, invalid)
        with self.assertRaisesRegex(ValueError, "images visual teacher"):
            initialize_from_teacher(agent, dict(self.source, teacher_vision_state={
                "experiment_config": {"teacher_vision": {"vision_mode": "zero"}}}))

    def test_budget_boundaries_and_metadata_validation(self):
        env, _, state = self._agent("budget")
        state.mark_completed(6)
        self.assertEqual(env.global_step_counter, 120054)
        for step in (4, 7, 18):
            with self.subTest(step=step), self.assertRaises(ValueError):
                state.mark_completed(step)
        saved = state.state_dict()
        for key, value in (("global_step", 6), ("completed_steps", 7), ("version", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                state.load_state_dict(dict(saved, **{key: value}))
        changed = deepcopy(saved)
        changed["experiment_config"]["asymmetric_teacher"]["total_steps"] = 18
        with self.assertRaises(ValueError):
            state.load_state_dict(changed)

    def test_invalid_budgets_modes_and_unsupported_timeouts_are_rejected(self):
        changes = (
            ("rollouts", 0), ("minibatch_size", 0), ("learning_epochs", 0),
            ("eval_interval", 3), ("checkpoint_interval", 3),
            ("time_limit_semantics", "collection_truncation"),
            ("total_steps", 11), ("total_steps", 0), ("mode", "unknown"),
        )
        for key, value in changes:
            settings = deepcopy(self.settings)
            settings[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(settings)

    def test_perception_and_reward_settings_are_validated_before_training(self):
        for field, value in (("camera_delay_frames", [1]), ("frame_dropout_prob", 1.1),
                             ("depth_noise_std_m", -0.1), ("measurement_std_m", 0),
                             ("max_depth_m", 3.01)):
            settings = deepcopy(self.settings)
            settings["mode"] = "m2"
            settings.setdefault("perception", {})[field] = value
            with self.subTest(perception=field), self.assertRaises(ValueError):
                validate_settings(settings)
        for field, value in (("weight", -1), ("sigma_m", 0), ("sigma_m", float("nan")),
                             ("prediction_horizon_s", -1), ("prediction_horizon_s", float("inf"))):
            settings = deepcopy(self.settings)
            settings.setdefault("perception_reward", {})[field] = value
            with self.subTest(reward=field), self.assertRaises(ValueError):
                validate_settings(settings)
        self.assertEqual(feature_flags({"mode": "m0"}), {"m1": False, "m2": False})
        self.assertEqual(feature_flags({"mode": "m1"}), {"m1": True, "m2": False})
        self.assertEqual(feature_flags({"mode": "m2"}), {"m1": True, "m2": True})


if __name__ == "__main__":
    unittest.main()
