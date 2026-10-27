"""CPU checks for teacher configuration and strict evaluation (no Isaac Gym)."""

import ast
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module("teacher_config_test", ROOT / "utils/config.py")
evaluator = load_module("teacher_evaluator_test", ROOT / "eval_sgs_teacher.py")


def entry_function(name):
    path = ROOT / "train_multistate_DQ_teacher.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"Path": Path, "json": json, "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TeacherConfigurationTests(unittest.TestCase):
    def test_sgs_and_pose_prediction_settings_coexist_and_survive_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.touch()
            args = config.get_params([
                "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                "--sgs_mode", "sgs", "--sgs_bank", str(bank),
                "--timesteps", "120000", "--stop_after", "24",
                "--future_pose_obs", "--future_pose_horizons", "0.2", "0.5",
                "--ekf_history", "7", "--ekf_q_scale", "2", "--ekf_r_scale", "0.02",
            ])
            source = ROOT / "data/cfg/DQ_teacher.yaml"
            cfg = config.configure_teacher_experiment(config.load_cfg(source), args)
            settings = config.apply_pose_prediction_config(cfg, args)
            self.assertEqual(cfg["env"]["sgs"]["mode"], "sgs")
            self.assertEqual(cfg["env"]["successMode"], "hold")
            self.assertEqual(config.teacher_run_end(args.timesteps, stop_after=args.stop_after), 24)
            self.assertEqual(settings, dict(enabled=True, horizons=[.2, .5],
                                            history=7, qScale=2., rScale=.02))
            saved = Path(directory) / "run"
            config.copy_cfg(source, saved, cfg=cfg)
            restored = config.load_cfg(saved / source.name)
            self.assertEqual(config.apply_pose_prediction_config(restored, config.get_params([])), settings)
            config.apply_pose_prediction_config(restored, config.get_params(["--no_future_pose_obs"]))
            self.assertFalse(restored["env"]["posePrediction"]["enabled"])
            self.assertEqual(restored["env"]["sgs"], cfg["env"]["sgs"])

    def test_defaults_preserve_disabled_sampler_and_cuda(self):
        args = config.get_params([])
        cfg = config.configure_teacher_experiment({"env": {}}, args)
        self.assertEqual(cfg["env"]["sgs"]["mode"], "disabled")
        self.assertEqual(cfg["env"]["successMode"], "legacy")
        self.assertIsNone(cfg["env"]["sgs"]["bank_size"])
        self.assertEqual(cfg["env"]["sgs"]["p_init"], 0.0)
        self.assertEqual(args.rl_device, "cuda:0")
        self.assertEqual(args.sim_device, "cuda:0")

    def test_bank_overrides_and_seed_are_saved_without_shortening_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.touch()  # Its contents are checked by the environment's bank loader.
            args = config.get_params(["--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                                      "--sgs_mode", "sgs", "--sgs_bank", str(bank), "--seed", "45",
                                      "--timesteps", "120000", "--stop_after", "24"])
            cfg = config.configure_teacher_experiment({"env": {}}, args)
            self.assertEqual(cfg["env"]["sgs"]["bank_path"], str(bank))
            self.assertEqual(cfg["env"]["sgs"]["mode"], "sgs")
            self.assertEqual(cfg["env"]["successMode"], "hold")
            self.assertEqual(cfg["seed"], 45)
            self.assertEqual(args.timesteps, 120000)
            self.assertEqual(config.teacher_run_end(args.timesteps, stop_after=args.stop_after), 24)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "starts from scratch"):
                config.configure_teacher_experiment({"env": {}}, args)

    def test_enabled_sampler_requires_existing_bank(self):
        args = config.get_params(["--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                                  "--sgs_mode", "uniform"])
        with self.assertRaisesRegex(ValueError, "validated bank"):
            config.configure_teacher_experiment({"env": {}}, args)

    def test_explicit_bank_size_remains_an_optional_positive_integer_assertion(self):
        args = config.get_params([])
        cfg = config.configure_teacher_experiment({"env": {"sgs": {"bank_size": 2048}}}, args)
        self.assertEqual(cfg["env"]["sgs"]["bank_size"], 2048)
        for invalid in (0, -1, True, 2.5, "2048"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "bank_size"):
                config.configure_teacher_experiment({"env": {"sgs": {"bank_size": invalid}}}, args)

    def test_strict_eval_overrides_saved_training_sampler(self):
        args = config.get_params(["--checkpoint", "agent_24.pt", "--eval_episodes", "3"])
        cfg = {"env": {"sgs": {"mode": "sgs", "bank_path": "not-needed.pt"},
                       "robotStartDistanceRange": [0.85, 1.1], "holdSteps": 1}}
        config.configure_teacher_experiment(cfg, args, is_eval=True, strict_eval=True)
        self.assertEqual(cfg["env"]["sgs"]["mode"], "disabled")
        self.assertEqual(cfg["env"]["successMode"], "hold")
        self.assertEqual(cfg["env"]["holdSteps"], 15)
        self.assertEqual(cfg["env"]["robotStartDistanceRange"], [2.0, 2.0])
        self.assertEqual(cfg["env"]["D1_bench_task_level"], "Level02")
        self.assertEqual(args.task, "B1Z1PickMulti")
        self.assertEqual(args.object_name, "sugar_box")

    def test_stop_after_is_relative_to_invocation_not_a_new_schedule(self):
        self.assertEqual(config.teacher_run_end(120000, 5000, 24), 5024)
        self.assertEqual(config.teacher_run_end(120000, 119990, 24), 120000)
        with self.assertRaises(ValueError):
            config.teacher_run_end(120000, stop_after=0)
        with self.assertRaises(ValueError):
            config.teacher_run_end(5000, 5000)

    def test_eval_loads_snapshot_and_model_flags_without_overriding_eval_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoints").mkdir()
            checkpoint = run / "checkpoints/best_agent.pt"
            checkpoint.touch()
            saved_cfg = run / "custom_teacher.yaml"
            saved_cfg.write_text("env: {}\n")
            (run / "teacher_run.json").write_text(json.dumps({
                "config_file": saved_cfg.name,
                "arguments": {"roboinfo": True, "no_velocity_obs": True, "rand_control": True, "seed": 43}}))
            args = config.get_params(["--checkpoint", str(checkpoint), "--seed", "99"])
            entry_function("restore_eval_arguments")(args)
            selected = entry_function("teacher_config_path")(args, is_eval=True, strict_eval=True)
            self.assertEqual(selected, str(saved_cfg))
            self.assertTrue(args.roboinfo)
            self.assertTrue(args.no_velocity_obs)
            self.assertTrue(args.rand_control)
            self.assertEqual(args.seed, 99)
            self.assertEqual(entry_function("checkpoint_step")(checkpoint), 0)
            self.assertEqual(entry_function("checkpoint_step")("agent_24.pt"), 24)


class FrozenScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.tensor(7))

    def forward(self, states, train=False):
        if train:
            self.count += 1
        return states


class EvaluationAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = SimpleNamespace(deterministic=False)
        self._state_preprocessor = FrozenScaler()
        self._value_preprocessor = FrozenScaler()

    def set_mode(self, mode):
        self.mode = mode

    def act(self, states, **kwargs):
        assert not torch.is_grad_enabled()
        assert self.policy.deterministic
        if getattr(self, "consume_rng", False):
            torch.rand(37)
        return torch.zeros_like(self._state_preprocessor(states)), None, {}

    def record_transition(self, **kwargs):
        raise AssertionError("Evaluation must not record training data")

    def post_interaction(self, **kwargs):
        raise AssertionError("Evaluation must not update PPO")


class EvaluationEnv:
    def __init__(self):
        self.num_envs = 2
        self.max_episode_length = 3
        self.hold_steps = 15
        self.cfg = {"env": {"successMode": "hold", "sgs": {"mode": "disabled"}}}
        self.age = torch.zeros(2, dtype=torch.long)
        self.reset_buf = torch.ones(2, dtype=torch.bool)
        self.episode_success = torch.zeros(2, dtype=torch.bool)
        self.lifted_object = torch.zeros(2, dtype=torch.bool)
        self.reset_cases = []
        self.episode_lengths = torch.tensor([1, 3])

    def reset(self):
        self.reset_cases.append(torch.rand(2))
        self.age[self.reset_buf] = 0
        self.episode_success[self.reset_buf] = False
        self.lifted_object[self.reset_buf] = False
        self.reset_buf[:] = False
        return torch.zeros(2, 1), {}

    def step(self, actions):
        self.age += 1
        # Env 0 repeatedly fails in one step; env 1 succeeds after three steps.
        self.reset_buf = self.age >= self.episode_lengths
        self.lifted_object[1] = self.age[1] >= 2
        self.episode_success[1] = self.reset_buf[1]
        return torch.zeros(2, 1), torch.zeros(2, 1), self.reset_buf[:, None], torch.zeros(2, 1, dtype=torch.bool), {}


class StrictEvaluationTests(unittest.TestCase):
    def test_slow_success_cases_are_not_displaced_by_fast_failures(self):
        agent = EvaluationAgent()
        trainer = SimpleNamespace(env=EvaluationEnv(), agents=agent, headless=True, timesteps=120000,
                                  teacher_args=SimpleNamespace(seed=99))
        result = evaluator.evaluate_teacher(trainer, episodes=3)
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["hold_successes"], 1)
        self.assertEqual(result["hold_success_rate"], 1 / 3)
        self.assertEqual(result["control_steps"], 4)
        self.assertEqual(result["waves"], 2)
        self.assertEqual(len(trainer.env.reset_cases), 2)
        self.assertTrue(result["scalers_unchanged"])
        self.assertTrue(result["deterministic_policy"])
        self.assertEqual(agent._state_preprocessor.count.item(), 7)
        self.assertEqual(agent._value_preprocessor.count.item(), 7)
        self.assertFalse(agent._state_preprocessor.training)
        # RNG consumed by the previous policy must not change the reset cases.
        torch.rand(37)
        other = SimpleNamespace(env=EvaluationEnv(), agents=EvaluationAgent(), headless=True,
                                timesteps=120000, teacher_args=SimpleNamespace(seed=99))
        other.env.episode_lengths[1] = 2
        other.agents.consume_rng = True
        evaluator.evaluate_teacher(other, episodes=3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(trainer.env.reset_cases, other.env.reset_cases)))

    def test_evaluation_detects_scaler_updates(self):
        class UpdatingAgent(EvaluationAgent):
            def act(self, states, **kwargs):
                self._state_preprocessor(states, train=True)
                return super().act(states, **kwargs)

        trainer = SimpleNamespace(env=EvaluationEnv(), agents=UpdatingAgent(), headless=True,
                                  timesteps=120000, teacher_args=SimpleNamespace(seed=99))
        with self.assertRaisesRegex(RuntimeError, "modified the frozen scaler"):
            evaluator.evaluate_teacher(trainer, episodes=2)

    def test_evaluation_rejects_training_sampler(self):
        env = EvaluationEnv()
        env.cfg["env"]["sgs"]["mode"] = "sgs"
        trainer = SimpleNamespace(env=env, agents=EvaluationAgent(), headless=True, timesteps=120000)
        with self.assertRaisesRegex(ValueError, "training bank"):
            evaluator.evaluate_teacher(trainer, 3)


if __name__ == "__main__":
    unittest.main()
