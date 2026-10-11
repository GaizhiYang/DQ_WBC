"""CPU regression tests for removing five velocities from the GFM Actor only."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from utils.wrapper import IsaacGymPreview3Wrapper
from utils.teacher_grasp_training import (
    ENVIRONMENT_OPTIONS, TeacherGraspTrainingState, resolve_selection,
)
from test_teacher_vision_models import original_teacher_classes


def options(**overrides):
    values = {name: False for name in ENVIRONMENT_OPTIONS}
    values.update(task="B1Z1PickMulti", seed=43, roboinfo=True,
                  observe_gait_commands=True, grasp_selector="gfm",
                  karl_switch_margin_deg=None, karl_orientation_preference=None,
                  actor_drop_velocity_obs=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class RawTeacherEnv:
    num_envs, num_states, num_agents = 2, 0, 1
    device = rl_device = "cpu"
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1276,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (9,), np.float32)

    def __init__(self):
        self.obs = torch.randn(self.num_envs, 1276) * .1
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.global_step_counter = 0

    def reset(self):
        self.reset_buf.zero_()
        return {"obs": self.obs}

    def step(self, actions):
        self.global_step_counter += 1
        self.obs[:, 1024] += .03
        self.obs[:, 1082:1087] += .1
        self.obs[:, -9:] = actions
        return {"obs": self.obs}, actions[:, 0].clone(), self.reset_buf, {}


class TeacherVelocityAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy_class, cls.value_class = original_teacher_classes()

    def setUp(self):
        torch.manual_seed(43)
        self.states = torch.randn(3, 1276) * .1

    def policy(self, enabled=False):
        return self.policy_class((1276,), (9,), "cpu", 1024, 128, None, None, None,
                                 actor_drop_velocity_obs=enabled)

    def value(self):
        return self.value_class((1276,), (9,), "cpu", 1024, 128, None, None, None)

    @staticmethod
    def legacy_fused(model, states):
        """The original author's complete GFM input construction."""
        features = model.feature_encoder(states[:, :1024])
        poses = torch.cat([states[:, -189:-99].reshape(-1, 30, 3),
                           states[:, -99:-9].reshape(-1, 30, 3)], dim=-1)
        grasp = model.attention_forward(poses, features, states[:, 1024:1030])
        return torch.cat([states[:, 1024:1087], states[:, -9:], features, grasp], -1)

    def test_default_actor_exactly_matches_original_fused_input_and_checkpoint_shapes(self):
        default = self.policy_class((1276,), (9,), "cpu", 1024, 128, None, None, None)
        explicit = self.policy(False)
        explicit.load_state_dict(default.state_dict(), strict=True)
        expected = default.net(self.legacy_fused(default, self.states))
        actual, std, _ = default.compute({"states": self.states}, "policy")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        restored, restored_std, _ = explicit.compute({"states": self.states}, "policy")
        torch.testing.assert_close(restored, actual, rtol=0, atol=0)
        torch.testing.assert_close(restored_std, std, rtol=0, atol=0)
        self.assertEqual(default.num_observations, 1276)
        self.assertEqual(default.actor_num_observations, 1276)
        self.assertEqual(default.net[0].in_features, 206)
        reduced = self.policy(True)
        self.assertEqual(reduced.num_observations, 1276)
        self.assertEqual(reduced.actor_num_observations, 1271)
        self.assertEqual(reduced.net[0].in_features, 201)
        self.assertEqual(set(default.state_dict()), set(reduced.state_dict()))
        for name, tensor in default.state_dict().items():
            if name == "net.0.weight":
                self.assertEqual(reduced.state_dict()[name].shape, (512, 201))
            else:
                self.assertEqual(reduced.state_dict()[name].shape, tensor.shape)
        with self.assertRaises(RuntimeError):
            reduced.load_state_dict(default.state_dict(), strict=True)

    def test_reduced_actor_rejects_wrong_layout_or_pretruncated_ppo_input(self):
        for obs_dim, actions, features, encode in ((1271, 9, 1024, 128),
                                                   (1276, 10, 1024, 128),
                                                   (1276, 9, 0, 0)):
            with self.subTest(layout=(obs_dim, actions, features, encode)), self.assertRaises(ValueError):
                self.policy_class((obs_dim,), (actions,), "cpu", features, encode,
                                  None, None, None, actor_drop_velocity_obs=True)
        with self.assertRaises(ValueError):
            self.policy(True).compute({"states": self.states[:, :1271]}, "policy")

    def test_actor_is_invariant_to_both_removed_velocity_blocks_and_has_zero_gradients(self):
        policy = self.policy(True)
        initial, std, _ = policy.compute({"states": self.states}, "policy")
        for start, end in ((1082, 1085), (1085, 1087), (1082, 1087)):
            changed = self.states.clone()
            changed[:, start:end] += torch.randn(3, end - start) * 100
            mean, changed_std, _ = policy.compute({"states": changed}, "policy")
            torch.testing.assert_close(mean, initial, rtol=0, atol=0)
            torch.testing.assert_close(changed_std, std, rtol=0, atol=0)
        states = self.states.clone().requires_grad_()
        policy.compute({"states": states}, "policy")[0].square().sum().backward()
        self.assertEqual(torch.count_nonzero(states.grad[:, 1082:1087]).item(), 0)
        # These groups must still influence the Actor/GFM; the removal must
        # not accidentally shift the pose, candidate or previous-action slices.
        for start, end in ((0, 1024), (1024, 1030), (1030, 1082),
                           (1087, 1177), (1177, 1267), (1267, 1276)):
            with self.subTest(retained_group=(start, end)):
                self.assertGreater(states.grad[:, start:end].abs().sum().item(), 0)

    def test_gfm_receives_the_same_features_candidates_and_object_pose(self):
        policy = self.policy(True)
        captured = {}
        attention = policy.attention_forward

        def recording_attention(grasps, obj_feat, pose):
            captured.update(grasps=grasps, obj_feat=obj_feat, pose=pose)
            return attention(grasps, obj_feat, pose)

        policy.attention_forward = recording_attention
        policy.compute({"states": self.states}, "policy")
        torch.testing.assert_close(captured["pose"], self.states[:, 1024:1030])
        torch.testing.assert_close(captured["obj_feat"], policy.feature_encoder(self.states[:, :1024]))
        torch.testing.assert_close(captured["grasps"][..., :3], self.states[:, 1087:1177].reshape(3, 30, 3))
        torch.testing.assert_close(captured["grasps"][..., 3:], self.states[:, 1177:1267].reshape(3, 30, 3))

    def test_critic_keeps_complete_input_and_depends_on_all_five_velocities(self):
        value = self.value()
        self.assertEqual(value.num_observations, 1276)
        self.assertEqual(value.net[0].in_features, 206)
        actual = value.compute({"states": self.states}, "value")[0]
        expected = value.net(self.legacy_fused(value, self.states))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Positive weights make dependence on every direct input coordinate
        # deterministic, rather than relying on a chance random initialization.
        with torch.no_grad():
            for layer in value.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(.001)
                    layer.bias.fill_(.01)
        states = self.states.clone().requires_grad_()
        value.compute({"states": states}, "value")[0].sum().backward()
        self.assertTrue(torch.all(states.grad[:, 1082:1087] > 0))
        for start, end in ((1082, 1085), (1085, 1087)):
            changed = self.states.clone()
            changed[:, start:end] += 1
            self.assertTrue(torch.all(value.compute({"states": changed}, "value")[0] >
                                      value.compute({"states": self.states}, "value")[0]))

    def test_observation_normalization_preserves_actor_invariance(self):
        policy = self.policy(True)
        scaler = RunningStandardScaler(1276, device="cpu")
        scaler(torch.randn(10, 1276), train=True)
        perturbed = self.states.clone()
        perturbed[:, 1082:1087] += 100
        first = policy.compute({"states": scaler(self.states)}, "policy")[0]
        second = policy.compute({"states": scaler(perturbed)}, "policy")[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(scaler.state_dict()["running_mean"].shape, (1276,))

    def test_settings_keep_baseline_metadata_and_reject_incompatible_flags_or_checkpoints(self):
        _, baseline = resolve_selection(options(), {"env": {}})
        self.assertNotIn("actor_drop_velocity_obs", baseline)
        cfg, reduced = resolve_selection(options(actor_drop_velocity_obs=True), {"env": {}})
        self.assertTrue(reduced["actor_drop_velocity_obs"])
        self.assertEqual(cfg["grasp_selection"], reduced)
        for overrides in ({"grasp_selector": "karl"}, {"grasp_selector": "geometric"},
                          {"roboinfo": False}, {"no_feature": True},
                          {"last_commands": True}, {"pitch_control": True},
                          {"task": "B1Z1Pick"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_selection(options(actor_drop_velocity_obs=True, **overrides), {"env": {}})
        for invalid_cfg in ({"env": {"lastCommands": True}},
                            {"env": {}, "sensor": {"enableCamera": True}}):
            with self.subTest(cfg=invalid_cfg), self.assertRaises(ValueError):
                resolve_selection(options(actor_drop_velocity_obs=True), invalid_cfg)
        legacy = {"policy": {"query_proj.weight": torch.zeros(64, 134)}}
        _, settings = resolve_selection(options(), {"env": {}}, legacy)
        self.assertNotIn("actor_drop_velocity_obs", settings)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_selection(options(actor_drop_velocity_obs=True), {"env": {}}, legacy)

    def test_checkpoint_weight_shape_cannot_disagree_with_saved_actor_configuration(self):
        cfg, reduced = resolve_selection(options(actor_drop_velocity_obs=True), {"env": {}})
        env = IsaacGymPreview3Wrapper(RawTeacherEnv())
        metadata = TeacherGraspTrainingState(env, reduced, cfg, options()).state_dict()
        with self.assertRaisesRegex(ValueError, "shape conflicts"):
            resolve_selection(options(), {"env": {}},
                {"grasp_selection_state": metadata, "policy": {"net.0.weight": torch.zeros(512, 206)}})
        # Old GFM checkpoints have no ablation metadata; a 201-wide Actor
        # must be rejected instead of silently guessing its observation layout.
        with self.assertRaisesRegex(ValueError, "shape conflicts"):
            resolve_selection(options(), {"env": {}}, {"policy": {
                "query_proj.weight": torch.zeros(64, 134), "net.0.weight": torch.zeros(512, 201)}})

    def test_real_ppo_update_full_critic_preprocessing_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(actor_drop_velocity_obs=True)
            experiment_cfg, selection = resolve_selection(args, {"env": {}})

            def build():
                env = IsaacGymPreview3Wrapper(RawTeacherEnv())
                models = {"policy": self.policy(True), "value": self.value()}
                cfg = deepcopy(PPO_DEFAULT_CONFIG)
                cfg.update(rollouts=2, learning_epochs=1, mini_batches=2,
                           state_preprocessor=RunningStandardScaler,
                           state_preprocessor_kwargs={"size": env.observation_space, "device": "cpu"},
                           value_preprocessor=RunningStandardScaler,
                           value_preprocessor_kwargs={"size": 1, "device": "cpu"})
                cfg["experiment"].update(directory=directory, write_interval=0, checkpoint_interval=0)
                agent = PPO(models=models, memory=RandomMemory(memory_size=2, num_envs=2, device="cpu"),
                            observation_space=env.observation_space, action_space=env.action_space,
                            device="cpu", cfg=cfg)
                agent.checkpoint_modules["grasp_selection_state"] = TeacherGraspTrainingState(
                    env, selection, experiment_cfg, args)
                return env, agent

            env, agent = build()
            before_policy = agent.policy.net[0].weight.detach().clone()
            before_value = agent.value.net[0].weight.detach().clone()
            trainer = SequentialTrainer(env=env, agents=agent,
                cfg={"timesteps": 2, "headless": True, "disable_progressbar": True,
                     "close_environment_at_exit": False})
            trainer.train()
            self.assertFalse(torch.equal(before_policy, agent.policy.net[0].weight))
            self.assertFalse(torch.equal(before_value, agent.value.net[0].weight))
            self.assertEqual(agent.memory.get_tensor_by_name("states").shape[-1], 1276)
            for name in ("returns", "advantages", "log_prob"):
                self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())
            self.assertEqual(agent._state_preprocessor.state_dict()["running_mean"].shape, (1276,))
            path = str(Path(directory) / "agent_2.pt")
            agent.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["policy"]["net.0.weight"].shape, (512, 201))
            self.assertEqual(saved["value"]["net.0.weight"].shape, (512, 206))
            self.assertEqual(saved["state_preprocessor"]["running_mean"].shape, (1276,))
            _, restored_selection = resolve_selection(options(), {"env": {}}, saved)
            self.assertEqual(restored_selection, selection)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve_selection(options(actor_drop_velocity_obs=False), {"env": {}}, saved)
            baseline_metadata = TeacherGraspTrainingState(env,
                {key: val for key, val in selection.items() if key != "actor_drop_velocity_obs"},
                {"env": {}}, options()).state_dict()
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve_selection(options(actor_drop_velocity_obs=True), {"env": {}},
                                  {"grasp_selection_state": baseline_metadata})
            restored_env, restored = build()
            restored.load(path)
            self.assertEqual(restored_env.global_step_counter, 2)
            for name in ("policy", "value", "state_preprocessor", "value_preprocessor"):
                for key, tensor in saved[name].items():
                    torch.testing.assert_close(restored.checkpoint_modules[name].state_dict()[key], tensor)
            self.assertTrue(restored.optimizer.state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
