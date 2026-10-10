"""Minimal teacher configuration, actual PPO update and checkpoint regression."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.karl_teacher import KarlTeacherValue
from modules.minimal_asymmetric_teacher import MinimalAsymmetricTeacherPolicy
from utils.config import get_params
from utils.geometric_teacher_wrapper import GeometricTeacherWrapper
from utils.teacher_grasp_training import TeacherGraspTrainingState, checkpoint_step, resolve_selection
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from test_geometric_teacher import RawGeometricTeacherEnv


def options(*flags):
    with patch.object(sys, "argv", ["train_multistate_DQ_teacher.py", *flags]):
        args = get_params()
    args.task = args.task or "B1Z1PickMulti"
    return args


def minimal_options(*flags):
    return options("--teacher_actor", "minimal", "--grasp_selector", "geometric",
                   "--roboinfo", "--observe_gait_commands", *flags)


class MinimalConfigurationTests(unittest.TestCase):
    def test_default_stays_privileged_and_minimal_changes_only_selection_config(self):
        self.assertIsNone(options().teacher_actor)
        _, default = resolve_selection(options(), {"env": {}})
        self.assertEqual((default["mode"], default["teacher_actor"]), ("gfm", "privileged"))
        original = {"env": {"numEnvs": 128, "holdSteps": 25}, "reward": {"lifting": .8}}
        cfg, selected = resolve_selection(minimal_options(), deepcopy(original))
        self.assertEqual(selected["teacher_actor"], "minimal")
        self.assertEqual(selected["mode"], "geometric")
        self.assertEqual({k: v for k, v in cfg.items() if k != "grasp_selection"}, original)
        self.assertEqual(cfg["grasp_selection"], selected)

    def test_unsupported_modes_rejected_before_environment_creation(self):
        for mode in ("gfm", "karl"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires --grasp_selector geometric"):
                resolve_selection(options("--teacher_actor", "minimal", "--grasp_selector", mode),
                                  {"env": {}})
        for flags, error in (
            (("--use_tanh",), "omit --use_tanh"),
            (("--teacher_init_checkpoint", "teacher.pt"), "starts from scratch"),
            (("--no_feature",), "omit --no_feature"),
            (("--last_commands",), "omit --no_feature"),
            (("--pitch_control",), "omit --no_feature"),
        ):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, error):
                resolve_selection(minimal_options(*flags), {"env": {}})
        with self.assertRaisesRegex(ValueError, "sensor.enableCamera"):
            resolve_selection(minimal_options(), {"env": {}, "sensor": {"enableCamera": True}})
        with self.assertRaisesRegex(ValueError, "requires --roboinfo"):
            resolve_selection(options("--teacher_actor", "minimal", "--grasp_selector", "geometric"),
                              {"env": {}})

    def test_checkpoint_auto_restore_and_explicit_actor_conflicts(self):
        args = minimal_options("--seed", "44")
        cfg, selected = resolve_selection(args, {"env": {"numEnvs": 2}})
        state = TeacherGraspTrainingState(SimpleNamespace(global_step_counter=123), selected, cfg, args)
        saved = {"grasp_selection_state": state.state_dict()}
        restored_args = options()
        restored_cfg, restored = resolve_selection(restored_args, {"env": {}}, saved)
        self.assertEqual(restored, selected)
        self.assertEqual(restored_cfg, cfg)
        self.assertEqual(restored_args.seed, 44)
        self.assertTrue(restored_args.roboinfo and restored_args.observe_gait_commands)
        self.assertEqual(checkpoint_step(saved, "best_agent.pt"), 123)
        with self.assertRaisesRegex(ValueError, "teacher_actor conflicts"):
            resolve_selection(options("--teacher_actor", "privileged"), {"env": {}}, saved)
        with self.assertRaisesRegex(ValueError, "grasp_selector conflicts"):
            resolve_selection(options("--grasp_selector", "karl"), {"env": {}}, saved)

    def test_legacy_version1_settings_without_actor_load_as_privileged(self):
        args = options("--grasp_selector", "geometric", "--roboinfo")
        cfg, selected = resolve_selection(args, {"env": {}})
        state = TeacherGraspTrainingState(SimpleNamespace(global_step_counter=19), selected, cfg, args).state_dict()
        del state["settings"]["teacher_actor"]
        del state["experiment_config"]["grasp_selection"]["teacher_actor"]
        saved = {"grasp_selection_state": state}
        restored_args = options()
        restored_cfg, restored = resolve_selection(restored_args, {"env": {}}, saved)
        self.assertEqual(restored["teacher_actor"], "privileged")
        raw = SimpleNamespace(global_step_counter=0)
        wrapper = SimpleNamespace(_env=raw)
        loader = TeacherGraspTrainingState(wrapper, restored, restored_cfg, restored_args)
        loader.load_state_dict(state)
        self.assertEqual(raw.global_step_counter, 19)
        self.assertNotIn("teacher_actor", state["settings"])  # No mutation of old checkpoint.
        with self.assertRaisesRegex(ValueError, "teacher_actor conflicts"):
            resolve_selection(options("--teacher_actor", "minimal"), {"env": {}}, saved)
        legacy_gfm = {"policy": {"query_proj.weight": torch.zeros(64, 134)}}
        self.assertEqual(resolve_selection(options(), {"env": {}}, legacy_gfm)[1]["teacher_actor"],
                         "privileged")
        with self.assertRaisesRegex(ValueError, "teacher_actor conflicts"):
            resolve_selection(options("--teacher_actor", "minimal"), {"env": {}}, legacy_gfm)


class MinimalPPOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def build(self, directory, deterministic=False):
        args = minimal_options()
        cfg, selected = resolve_selection(args, {"env": {}})
        raw = RawGeometricTeacherEnv()
        provider = SimpleNamespace(context=lambda: raw.context)
        with patch("utils.geometric_teacher_wrapper.TeacherGraspGeometry", return_value=provider):
            env = GeometricTeacherWrapper(raw, selected["geometric"])
        models = {
            "policy": MinimalAsymmetricTeacherPolicy(env.observation_space, env.action_space,
                                                     "cpu", deterministic=deterministic),
            "value": KarlTeacherValue(env.observation_space, env.action_space, "cpu"),
        }
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
        agent.checkpoint_modules["grasp_selection_state"] = TeacherGraspTrainingState(env, selected, cfg, args)
        agent.checkpoint_modules["scheduler"] = agent.scheduler
        return env, agent

    def test_real_ppo_update_save_resume_and_deterministic_play(self):
        with tempfile.TemporaryDirectory() as directory:
            env, agent = self.build(directory)
            initial_policy = {k: v.clone() for k, v in agent.policy.state_dict().items()}
            initial_value = agent.value.net[0].weight.detach().clone()
            first_obs = env.reset()[0].clone()
            trainer = SequentialTrainer(env=env, agents=agent, cfg={
                "timesteps": 2, "headless": True, "disable_progressbar": True,
                "close_environment_at_exit": False,
            })
            trainer.train()
            self.assertTrue(any(not torch.equal(initial_policy[k], v)
                                for k, v in agent.policy.state_dict().items()
                                if k.endswith("weight")))
            self.assertFalse(torch.equal(initial_value, agent.value.net[0].weight))
            self.assertTrue({id(p) for p in agent.policy.parameters()}.isdisjoint(
                {id(p) for p in agent.value.parameters()}))
            torch.testing.assert_close(agent.memory.get_tensor_by_name("states")[0], first_obs)
            for name in ("returns", "advantages", "log_prob"):
                self.assertTrue(torch.isfinite(agent.memory.get_tensor_by_name(name)).all())
            self.assertEqual(tuple(agent.memory.get_tensor_by_name("log_prob").shape), (2, 2, 1))
            path = str(Path(directory) / "best_agent.pt")
            agent.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            _, restored_selection = resolve_selection(options(), {"env": {}}, checkpoint)
            self.assertEqual(restored_selection["teacher_actor"], "minimal")
            self.assertEqual(checkpoint_step(checkpoint, path), 2)
            restored_env, restored = self.build(directory)
            restored.load(path)
            self.assertEqual(restored_env.global_step_counter, 2)
            for name in ("policy", "value", "state_preprocessor", "value_preprocessor"):
                for key, value in checkpoint[name].items():
                    torch.testing.assert_close(restored.checkpoint_modules[name].state_dict()[key], value)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            self.assertEqual(restored.scheduler.state_dict(), checkpoint["scheduler"])
            resumed = SequentialTrainer(env=restored_env, agents=restored, cfg={
                "timesteps": 4, "initial_timestep": 2, "headless": True,
                "disable_progressbar": True, "close_environment_at_exit": False,
            })
            resumed.train()
            self.assertEqual(restored_env.global_step_counter, 4)

            eval_env, eval_agent = self.build(directory, deterministic=True)
            eval_agent.load(path)
            frozen_stats = deepcopy(eval_agent._state_preprocessor.state_dict())
            policy_before = deepcopy(eval_agent.policy.state_dict())
            observation = eval_env.reset()[0]
            with torch.no_grad():
                action1, logp, _ = eval_agent.act(observation, 0, 2)
                action2, _, _ = eval_agent.act(observation, 0, 2)
            torch.testing.assert_close(action1, action2, rtol=0, atol=0)
            self.assertEqual(tuple(logp.shape), (2, 1))
            evaluator = SequentialTrainer(env=eval_env, agents=eval_agent, cfg={
                "timesteps": 2, "evaluation_steps": 2, "headless": True,
                "disable_progressbar": True, "close_environment_at_exit": False,
            })
            evaluator.eval()
            self.assertEqual(eval_env.global_step_counter, 4)
            for key, value in frozen_stats.items():
                torch.testing.assert_close(eval_agent._state_preprocessor.state_dict()[key], value)
            for key, value in policy_before.items():
                torch.testing.assert_close(eval_agent.policy.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
