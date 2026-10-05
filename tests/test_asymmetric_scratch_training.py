"""Actual CPU PPO checks for scratch initialization and rollout normalization."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DQ_high-level"))
sys.path.insert(0, str(ROOT / "third_party/skrl"))

from learning.asymmetric_teacher_trainer import AsymmetricTeacherTrainer
from modules.asymmetric_teacher import PROPRIO_INDICES, packed_obs_dim
from modules.teacher_vision import PRIVILEGED_OBS_DIM, PACKED_OBS_DIM
from utils.asymmetric_scratch_preprocessor import (
    IdentityValuePreprocessor, RolloutPrefixRunningStandardScaler,
)
from utils.asymmetric_teacher_preprocessor import export_deployment, load_deployment
from utils.asymmetric_teacher_training import (
    AsymmetricTeacherTrainingState, feature_flags, make_agent,
    restore_experiment, validate_settings,
)
from test_asymmetric_teacher_training import SyntheticFeatureWrapper
from test_teacher_vision_training import ReusingCameraEnv


class LargeRewardEnv(ReusingCameraEnv):
    def step(self, actions):
        observations, rewards, done, info = super().step(actions)
        return observations, rewards + 100., done, info


class AsymmetricScratchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(271)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = {
            "mode": "m2", "rollouts": 2, "minibatch_size": 2,
            "learning_epochs": 2, "learning_rate": 1e-4, "total_steps": 8,
            "eval_interval": 0, "eval_steps": 2, "checkpoint_interval": 0,
            "seed": 271, "initialization": "scratch", "normalization": "rollout",
        }

    def agent(self, name="scratch", *, settings=None, reward_env=False):
        settings = deepcopy(self.settings if settings is None else settings)
        raw = (LargeRewardEnv if reward_env else ReusingCameraEnv)(0)
        env = SyntheticFeatureWrapper(raw, **feature_flags(settings))
        agent = make_agent(env, settings, self.directory.name, name)
        state = AsymmetricTeacherTrainingState(env, {"asymmetric_teacher": settings}, {}, 0)
        agent.checkpoint_modules["asymmetric_teacher_state"] = state
        self.addCleanup(lambda: agent.writer.close() if hasattr(agent, "writer") else None)
        return env, agent

    def trainer(self, env, agent, steps=2):
        return AsymmetricTeacherTrainer(env, agent, cfg={
            "timesteps": steps, "headless": True, "disable_progressbar": True,
            "close_environment_at_exit": False, "eval_interval": 0,
            "eval_steps": 2, "checkpoint_interval": 0,
        })

    def assert_statistics(self, scaler, expected):
        for key, value in expected.items():
            torch.testing.assert_close(scaler.state_dict()[key], value, atol=0, rtol=0)

    def test_setting_validation_and_migration_defaults(self):
        for key, value in (("normalization", "minibatch"), ("initialization", "invalid")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_settings(dict(self.settings, **{key: value}))
        settings = deepcopy(self.settings)
        settings.pop("normalization")
        settings.pop("initialization")
        _, agent = self.agent(settings=settings)
        self.assertNotIsInstance(agent._state_preprocessor, RolloutPrefixRunningStandardScaler)
        self.assertNotIsInstance(agent._value_preprocessor, IdentityValuePreprocessor)
        self.assertEqual(torch.count_nonzero(agent.policy.actor.visual_adapter.weight), 0)

    def test_first_real_ppo_backward_trains_cnn_and_all_adapters(self):
        env, agent = self.agent()
        parameters = {
            "cnn": agent.policy.actor.shared_cnn.encoder[0].weight,
            "visual": agent.policy.actor.visual_adapter.weight,
            "actor_belief": agent.policy.actor.belief_adapter.weight,
            "critic_task": agent.value.task_adapter.weight,
            "critic_belief": agent.value.belief_adapter.weight,
        }
        before = {name: tensor.detach().clone() for name, tensor in parameters.items()}
        self.assertTrue(all(torch.count_nonzero(tensor) for tensor in parameters.values()))
        gradients = []
        handle = parameters["cnn"].register_hook(lambda gradient: gradients.append(float(gradient.abs().sum())))
        self.addCleanup(handle.remove)
        scaler = agent._state_preprocessor
        initial = deepcopy(scaler.state_dict())
        original_update = agent._update
        boundaries = []

        def checked_update(*args, **kwargs):
            self.assert_statistics(scaler, initial)
            original_update(*args, **kwargs)
            self.assert_statistics(scaler, initial)
            boundaries.append(int(scaler.current_count))

        agent._update = checked_update
        self.trainer(env, agent).train()
        self.assertEqual(boundaries, [1])
        self.assertGreater(gradients[0], 0)
        for name, tensor in parameters.items():
            self.assertFalse(torch.equal(before[name], tensor), name)
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(int(scaler.current_count), 5)
        self.assertEqual(agent.checkpoint_modules["asymmetric_teacher_state"].completed_steps, 2)
        self.assertEqual(env.global_step_counter, 2)
        for name in ("returns", "advantages", "log_prob"):
            self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())

    def test_recomputed_likelihoods_match_with_train_true_scaler_calls(self):
        env, agent = self.agent()
        states, _ = env.reset()
        scaler = agent._state_preprocessor
        for _ in range(2):
            statistics = deepcopy(scaler.state_dict())
            agent.set_mode("eval")
            with torch.no_grad():
                actions, old_log_prob, _ = agent.act(states, timestep=0, timesteps=2)
            agent.set_mode("train")
            for _ in range(3):
                processed = scaler(states, train=True)
                _, log_prob, _ = agent.policy.act({"states": processed, "taken_actions": actions}, "policy")
                torch.testing.assert_close(log_prob, old_log_prob, atol=1e-6, rtol=0)
                torch.testing.assert_close(processed[:, PRIVILEGED_OBS_DIM:], states[:, PRIVILEGED_OBS_DIM:], atol=0, rtol=0)
                self.assert_statistics(scaler, statistics)
            scaler.update_rollout(states)

    def test_raw_returns_above_five_are_not_clipped(self):
        env, agent = self.agent(reward_env=True)
        self.trainer(env, agent).train()
        returns = agent.memory.get_tensor_by_name("returns")
        self.assertTrue(torch.isfinite(returns).all())
        self.assertGreater(float(returns.min()), 90.)
        value_scaler = agent._value_preprocessor
        original = deepcopy(value_scaler.state_dict())
        values = torch.tensor([[-1234.], [6789.]])
        for inverse in (True, False):
            torch.testing.assert_close(value_scaler(values, train=True, inverse=inverse), values, atol=0, rtol=0)
        self.assert_statistics(value_scaler, original)

    def test_resume_export_and_evaluation_retain_normalization_contract(self):
        env, agent = self.agent("before_save")
        trainer = self.trainer(env, agent)
        trainer.train()
        before_eval = deepcopy(agent._state_preprocessor.state_dict())
        trainer.evaluate_policy(3)
        self.assert_statistics(agent._state_preprocessor, before_eval)
        self.assertEqual(int(agent._state_preprocessor.current_count), 5)
        saved = {name: deepcopy(module.state_dict()) for name, module in agent.checkpoint_modules.items()}
        new_env, restored = self.agent("resume")
        restore_experiment(restored, saved)
        self.assert_statistics(restored._state_preprocessor, before_eval)
        self.assertIsInstance(restored._value_preprocessor, IdentityValuePreprocessor)
        self.trainer(new_env, restored, steps=4).train()
        self.assertEqual(int(restored._state_preprocessor.current_count), 9)
        states, _ = new_env.reset()
        deployment = load_deployment(export_deployment(restored.policy, restored._state_preprocessor))
        with torch.no_grad():
            expected = restored.policy.compute({"states": restored._state_preprocessor(states)}, "policy")[0]
            actual = deployment(states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM],
                                states[:, list(PROPRIO_INDICES)], states[:, -16:])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_single_sample_variance_and_prefix_only_population_statistics(self):
        size = packed_obs_dim(True, True)
        scaler = RolloutPrefixRunningStandardScaler(size=size, device="cpu")
        states = torch.zeros(1, size)
        states[:, :PRIVILEGED_OBS_DIM] = 3.
        states[:, PRIVILEGED_OBS_DIM:] = float("nan")
        scaler.update_rollout(states)
        self.assertEqual(int(scaler.current_count), 2)
        torch.testing.assert_close(scaler.running_mean, torch.full((PRIVILEGED_OBS_DIM,), 1.5, dtype=torch.float64))
        # Combine the initial mean0/variance1/count1 prior with one sample 3.
        torch.testing.assert_close(scaler.running_variance, torch.full((PRIVILEGED_OBS_DIM,), 2.75, dtype=torch.float64))
        self.assertTrue(torch.isfinite(scaler.running_variance).all())
        with self.assertRaisesRegex(ValueError, "empty"):
            scaler.update_rollout(states[:0])
        states[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            scaler.update_rollout(states)

    def test_identity_value_scaler_rejects_scaled_teacher_statistics(self):
        scaler = IdentityValuePreprocessor(size=1, device="cpu")
        state = deepcopy(scaler.state_dict())
        state["running_variance"].fill_(7.)
        with self.assertRaisesRegex(ValueError, "identity value statistics"):
            scaler.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
