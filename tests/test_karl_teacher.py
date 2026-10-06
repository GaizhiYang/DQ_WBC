"""Geometry, partial resets, PPO replay and checkpoint regression tests (CPU)."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))
from modules.karl_grasp_selector import KarlGraspSelector, grasp_costs, rpy_to_quaternion, quaternion_to_rpy
from modules.karl_teacher import KarlTeacherPolicy, KarlTeacherValue
from utils.karl_teacher_wrapper import KarlTeacherWrapper
from utils.teacher_grasp_training import (
    ENVIRONMENT_OPTIONS, TeacherGraspTrainingState, resolve_selection, checkpoint_step,
)
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from test_teacher_vision_models import original_teacher_classes


def args(**overrides):
    options = {key: False for key in ENVIRONMENT_OPTIONS}
    options.update(task="B1Z1PickMulti", seed=43, grasp_selector=None,
                   karl_switch_margin_deg=None, karl_orientation_preference=None)
    options.update(overrides)
    return SimpleNamespace(**options)


class RawTeacherEnv:
    """Reuses buffers and exposes terminal obs before partial reset, like DQ."""
    num_envs, num_states, num_agents = 2, 0, 1
    device = rl_device = "cpu"
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1276,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (9,), np.float32)

    def __init__(self):
        self.obs = torch.zeros(2, 1276)
        self.obs[:, :1024] = torch.randn(2, 1024) * .1
        self.obs[:, -189:-99] = torch.arange(90).float() / 100
        self.reset_buf = torch.zeros(2, dtype=torch.long)
        self.global_step_counter = 0

    def reset(self):
        self.reset_buf.zero_()
        return {"obs": self.obs}

    def step(self, actions):
        self.global_step_counter += 1
        self.obs[:, 1024] += .03
        self.obs[:, -9:] = actions
        self.reset_buf[0] = int(self.global_step_counter % 2 == 0)
        return {"obs": self.obs}, actions[:, 0].clone(), self.reset_buf, {}


class KarlGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def select(self, yaw, margin=30, ee_yaw=0):
        selector = KarlGraspSelector(1, "cpu", margin)
        poses = torch.zeros(1, len(yaw), 6, dtype=torch.float64)
        poses[0, :, 5] = torch.deg2rad(torch.tensor(yaw, dtype=torch.float64))
        ee = torch.tensor([[0., 0., 0., 0., 0., math.radians(ee_yaw)]], dtype=torch.float64)
        return selector, poses, ee

    def test_hysteresis_keeps_zero_on_first_frame_and_switches_on_large_improvement(self):
        sel, poses, ee = self.select([20, 0])
        self.assertEqual(sel.select(poses, ee)[1]["karl_grasp_index"].item(), 0)
        poses[:, 0, 5] = math.radians(60)
        self.assertEqual(sel.select(poses, ee)[1]["karl_grasp_index"].item(), 1)
        poses[:, 0, 5], poses[:, 1, 5] = 0, math.radians(10)
        self.assertEqual(sel.select(poses, ee)[1]["karl_grasp_index"].item(), 1)

    def test_strict_threshold_equality_does_not_switch(self):
        sel, poses, ee = self.select([30, 0])
        costs = grasp_costs(poses, ee)
        sel.margin = float(costs[0, 0] - costs[0, 1])
        self.assertFalse(sel.select(poses, ee)[1]["karl_grasp_switched"].item())

    def test_wraparound_and_quaternion_sign_invariance(self):
        sel, poses, ee = self.select([-179, 140], margin=0, ee_yaw=179)
        costs = grasp_costs(poses, ee)
        self.assertAlmostEqual(costs[0, 0].item(), math.radians(2), places=9)
        poses[:, :, 5] += 2 * math.pi
        torch.testing.assert_close(grasp_costs(poses, ee), costs)
        self.assertEqual(sel.select(poses, ee)[1]["karl_grasp_index"].item(), 0)

    def test_rotation_formula_against_scipy_full_rotations(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.RandomState(43)
        rotations = rng.uniform(-2, 2, (12, 3))
        q = rpy_to_quaternion(torch.from_numpy(rotations))
        np.testing.assert_allclose(q.numpy(), Rotation.from_euler("xyz", rotations).as_quat(), atol=1e-12)
        poses = torch.zeros(1, 12, 6, dtype=torch.float64)
        poses[0, :, 3:] = torch.from_numpy(rotations)
        ee = poses[:, 0].clone()
        expected = (Rotation.from_euler("xyz", rotations[0]).inv() * Rotation.from_euler("xyz", rotations)).magnitude()
        np.testing.assert_allclose(grasp_costs(poses, ee)[0].numpy(), expected, atol=5e-8)

    def test_rpy_roundtrip_including_gimbal_lock_and_legacy_order(self):
        from scipy.spatial.transform import Rotation
        rpy = torch.tensor([[.7, math.pi/2, 1.2], [-1.1, -math.pi/2, -.9],
                            [.2, .3, -.8], [.2, math.pi/2-1e-5, -.8]], dtype=torch.float64)
        recovered = quaternion_to_rpy(rpy_to_quaternion(rpy))
        np.testing.assert_allclose(Rotation.from_euler("xyz", recovered.numpy()).as_matrix(),
                                   Rotation.from_euler("xyz", rpy.numpy()).as_matrix(), atol=1e-9)
        torch.testing.assert_close(recovered[2], rpy[2])

    def test_positions_do_not_affect_cost_and_selected_pose_is_real_candidate(self):
        sel, poses, ee = self.select([80, 0], margin=0)
        poses[0, :, :3] = torch.tensor([[0, 0, 0], [20, 30, 40]])
        selected, _ = sel.select(poses, ee)
        torch.testing.assert_close(selected, poses[:, 1])

    def test_original_orientation_penalty_and_total_cost_hysteresis(self):
        sel, poses, ee = self.select([80, 100], ee_yaw=80)
        costs = grasp_costs(poses, ee, "karl")
        torch.testing.assert_close(costs, torch.tensor([[1., math.radians(20)]], dtype=torch.float64))
        sel.orientation_preference = "karl"
        self.assertTrue(sel.select(poses, ee)[1]["karl_grasp_switched"].item())

    def test_invalid_current_switches_and_all_invalid_falls_back_to_ee(self):
        sel, poses, ee = self.select([0, 10])
        poses[:, 0, 0] = float("nan")
        selected, metrics = sel.select(poses, ee)
        self.assertEqual(metrics["karl_grasp_index"].item(), 1)
        self.assertTrue(torch.isfinite(selected).all())
        poses[:] = float("nan")
        selected, metrics = sel.select(poses, ee)
        torch.testing.assert_close(selected, ee)
        self.assertTrue(metrics["karl_no_valid_grasp"].item())

    def test_selection_does_not_build_autograd_graph(self):
        sel, poses, ee = self.select([90, 0])
        self.assertFalse(sel.select(poses.requires_grad_(), ee)[0].requires_grad)


class KarlTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_packed_layout_fresh_storage_and_partial_episode_reset(self):
        raw = RawTeacherEnv()
        env = KarlTeacherWrapper(raw)
        packed, _ = env.reset()
        self.assertEqual(packed.shape, (2, 1102))
        torch.testing.assert_close(packed[:, :1087], raw.obs[:, :1087])
        torch.testing.assert_close(packed[:, -15:-12], raw.obs[:, -189:-186])
        saved = packed.clone()
        env.selector.indices[:] = torch.tensor([4, 7])
        raw.reset_buf[:] = torch.tensor([1, 0])
        env.reset()
        self.assertEqual(env.selector.indices.tolist(), [0, 7])
        raw.obs.zero_()
        torch.testing.assert_close(packed, saved)

    def test_termination_does_not_reset_before_terminal_observation(self):
        raw = RawTeacherEnv()
        env = KarlTeacherWrapper(raw)
        env.reset()
        env.selector.indices[:] = 8
        raw.global_step_counter = 1
        packed, _, done, _, _ = env.step(torch.zeros(2, 9))
        self.assertTrue(done[0])
        self.assertEqual(env.selector.indices.tolist(), [8, 8])
        torch.testing.assert_close(packed[:, -15:-12], raw.obs[:, -165:-162])
        env.reset()
        self.assertEqual(env.selector.indices.tolist(), [0, 8])

    def test_backbone_matches_baseline_and_removes_only_attention_parameters(self):
        old_policy, old_value = original_teacher_classes()
        for old_cls, new_cls in ((old_policy, KarlTeacherPolicy), (old_value, KarlTeacherValue)):
            old = old_cls((1276,), (9,), "cpu", 1024, 128, None, None, None)
            new = new_cls((1102,), (9,), "cpu")
            removed = set(old.state_dict()) - set(new.state_dict())
            self.assertEqual(removed, {name + suffix for name in ("key_proj", "value_proj", "query_proj", "output_proj")
                                      for suffix in (".weight", ".bias")})
            self.assertEqual(sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in new.parameters()), 9926)
            for key, tensor in new.state_dict().items():
                self.assertEqual(tensor.shape, old.state_dict()[key].shape)
            self.assertEqual(new.net[0].in_features, 206)

    def test_replay_likelihood_independent_of_new_selector_state_and_shuffle(self):
        env = KarlTeacherWrapper(RawTeacherEnv())
        packed, _ = env.reset()
        policy = KarlTeacherPolicy(env.observation_space, env.action_space)
        scaler = RunningStandardScaler(1102, device="cpu")
        scaler(torch.randn(10, 1102), train=True)
        normalized = scaler(packed)
        action, before, _ = policy.act({"states": normalized}, "policy")
        env.selector.indices[:] = 12
        env.step(torch.ones(2, 9))
        order = torch.tensor([1, 0])
        _, after, _ = policy.act({"states": scaler(packed[order]), "taken_actions": action[order]}, "policy")
        torch.testing.assert_close((after - before[order]).exp(), torch.ones_like(after), atol=1e-6, rtol=1e-6)

    def test_default_baseline_and_legacy_checkpoint_cannot_load_as_karl(self):
        cfg = {"env": {}}
        self.assertEqual(resolve_selection(args(), cfg)[1]["mode"], "gfm")
        legacy = {"policy": {"query_proj.weight": torch.zeros(64, 134)}}
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_selection(args(grasp_selector="karl"), cfg, legacy)
        for flag in ("no_feature", "last_commands", "pitch_control"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                resolve_selection(args(grasp_selector="karl", roboinfo=True, **{flag: True}), cfg)

    def test_real_ppo_update_and_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args(grasp_selector="karl", roboinfo=True)
            cfg, selection = resolve_selection(options, {"env": {}})

            def build():
                env = KarlTeacherWrapper(RawTeacherEnv())
                models = {"policy": KarlTeacherPolicy(env.observation_space, env.action_space),
                          "value": KarlTeacherValue(env.observation_space, env.action_space)}
                settings = deepcopy(PPO_DEFAULT_CONFIG)
                settings.update(rollouts=2, learning_epochs=1, mini_batches=2,
                                state_preprocessor=RunningStandardScaler,
                                state_preprocessor_kwargs={"size": 1102, "device": "cpu"},
                                value_preprocessor=RunningStandardScaler,
                                value_preprocessor_kwargs={"size": 1, "device": "cpu"},
                                learning_rate_scheduler=KLAdaptiveLR,
                                learning_rate_scheduler_kwargs={"kl_threshold": .008})
                settings["experiment"].update(directory=directory, write_interval=0, checkpoint_interval=0)
                agent = PPO(models=models, memory=RandomMemory(memory_size=2, num_envs=2, device="cpu"),
                            observation_space=env.observation_space, action_space=env.action_space,
                            device="cpu", cfg=settings)
                agent.checkpoint_modules["grasp_selection_state"] = TeacherGraspTrainingState(env, selection, cfg, options)
                agent.checkpoint_modules["scheduler"] = agent.scheduler
                return env, agent

            env, agent = build()
            before_policy = agent.policy.net[0].weight.detach().clone()
            before_value = agent.value.net[0].weight.detach().clone()
            first_obs = env.reset()[0].clone()
            trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 2, "headless": True,
                                        "disable_progressbar": True, "close_environment_at_exit": False})
            trainer.train()
            self.assertFalse(torch.equal(before_policy, agent.policy.net[0].weight))
            self.assertFalse(torch.equal(before_value, agent.value.net[0].weight))
            torch.testing.assert_close(agent.memory.get_tensor_by_name("states")[0], first_obs)
            for name in ("returns", "advantages", "log_prob"):
                self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())
            path = str(Path(directory) / "best_agent.pt")
            agent.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            resolved_cfg, restored_settings = resolve_selection(args(), {"env": {}}, saved)
            self.assertEqual(restored_settings, selection)
            self.assertEqual(checkpoint_step(saved, path), 2)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve_selection(args(karl_switch_margin_deg=0), {"env": {}}, saved)
            restored_env, restored = build()
            restored.load(path)
            self.assertEqual(restored_env.global_step_counter, 2)
            self.assertEqual(restored_env.selector.indices.tolist(), [0, 0])
            for name in ("policy", "value", "state_preprocessor", "value_preprocessor"):
                for key, tensor in saved[name].items():
                    torch.testing.assert_close(restored.checkpoint_modules[name].state_dict()[key], tensor)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            self.assertEqual(restored.scheduler.state_dict(), saved["scheduler"])

            # Exercise the actual SKRL evaluation path: this repository used
            # to ignore the requested duration and run 50000 steps instead.
            eval_env, eval_agent = build()
            evaluator = SequentialTrainer(env=eval_env, agents=eval_agent,
                cfg={"timesteps": 2, "evaluation_steps": 2, "headless": True,
                     "disable_progressbar": True, "close_environment_at_exit": False})
            evaluator.eval()
            self.assertEqual(eval_env.global_step_counter, 2)
            self.assertFalse(eval_agent.optimizer.state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
