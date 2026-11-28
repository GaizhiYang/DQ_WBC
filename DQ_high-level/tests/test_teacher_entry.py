"""CPU checks for teacher configuration and strict evaluation (no Isaac Gym)."""

import ast
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
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
    namespace = {"Path": Path, "json": json, "re": re,
                 "saved_history_frames": config.saved_history_frames}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TeacherConfigurationTests(unittest.TestCase):
    def test_ekf_launcher_keeps_level03_and_accepts_level02_override(self):
        source = ROOT / "data/cfg/DQ_teacher.yaml"
        for extra, expected in (([], "Level03"), (["--motion_level", "Level02"], "Level02")):
            with self.subTest(level=expected):
                output = subprocess.check_output(
                    ["bash", str(ROOT / "train_teacher_ekf.sh"), "--dry-run", "--no-wandb", *extra],
                    text=True)
                command = next(line[len("Command: "):] for line in output.splitlines()
                               if line.startswith("Command: "))
                args = config.get_params(shlex.split(command)[2:])
                cfg = config.configure_teacher_experiment(config.load_cfg(source), args)
                self.assertEqual(cfg["env"]["D1_bench_task_level"], expected)
                self.assertEqual(cfg["env"]["sgs"]["mode"], "disabled")
                self.assertFalse(cfg["env"]["tsil"]["enabled"])
                self.assertEqual(args.ekf_q_scale, 10.)
                self.assertEqual(args.ekf_r_scale, .0001)
                if expected == "Level03":
                    self.assertEqual(cfg["env"]["level03Motion"]["translationAxes"], "xy")
                    self.assertEqual(cfg["env"]["level03Motion"]["minSpeed"], .25)
                    self.assertEqual(cfg["env"]["level03Motion"]["maxSpeed"], .45)
                    self.assertEqual(cfg["env"]["level03Motion"]["maxYawRate"], .4)
                    self.assertEqual(cfg["env"]["level03Motion"]["stopProbability"], 0.)
                    self.assertEqual(cfg["env"]["level03Motion"]["reverseProbability"], 0.)
        # Other launchers sharing this YAML retain their original default.
        self.assertEqual(config.load_cfg(source)["env"]["D1_bench_task_level"], "Level02")

    def test_motion_level_override_is_saved_and_absent_flag_keeps_yaml(self):
        source = ROOT / "data/cfg/DQ_teacher.yaml"
        cfg = config.configure_teacher_experiment(
            config.load_cfg(source), config.get_params(["--motion_level", "Level03"]))
        with tempfile.TemporaryDirectory() as directory:
            config.copy_cfg(source, directory, cfg=cfg)
            restored = config.configure_teacher_experiment(
                config.load_cfg(Path(directory) / source.name), config.get_params([]))
        self.assertEqual(restored["env"]["D1_bench_task_level"], "Level03")

    def test_level03_acceleration_limits_must_be_finite_and_positive(self):
        self.assertEqual(config.level03_motion_limits({}), (1., 1.8))
        self.assertEqual(config.level03_motion_limits({"level03Motion": {
            "linearAcceleration": .2, "yawAcceleration": .4}}), (.2, .4))
        for key in ("linearAcceleration", "yawAcceleration"):
            for value in (0, -1, True, float("nan"), float("inf"), "0.3"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    config.level03_motion_limits({"level03Motion": {key: value}})

    def test_level03_minimum_speed_is_validated_and_old_configs_keep_legacy_floor(self):
        for value in (-.1, True, float("nan"), float("inf"), ".25", .45, .46):
            with self.subTest(minimum=value), self.assertRaisesRegex(ValueError, "minSpeed"):
                config.level03_motion_config({"level03Motion": {"minSpeed": value}})
        self.assertEqual(config.level03_motion_config({"level03Motion": {"minSpeed": 0.}})["minSpeed"], 0.)
        self.assertEqual(config.level03_motion_config({})["minSpeed"], .25)
        # Saved runs before the speed-band setting explicitly store maxSpeed.
        legacy = config.level03_motion_config({"level03Motion": {"maxSpeed": .4, "maxYawRate": .3}})
        self.assertEqual(legacy["minSpeed"], 0.)
        self.assertEqual(legacy["maxSpeed"], .4)
        current = config.level03_motion_config({"level03Motion": {
            "minSpeed": .25, "maxSpeed": .45, "maxYawRate": .4}})
        self.assertEqual(current["minSpeed"], .25)

    def test_level03_axes_default_to_xy_and_old_explicit_speed_configs_keep_y(self):
        self.assertEqual(config.level03_motion_config({})["translationAxes"], "xy")
        legacy = config.level03_motion_config({"level03Motion": {"maxSpeed": .4}})
        self.assertEqual(legacy["translationAxes"], "y")
        # Even the preceding no-stop speed-band configuration still drove Y only.
        prior_band = config.level03_motion_config({"level03Motion": {"minSpeed": .25, "maxSpeed": .45}})
        self.assertEqual(prior_band["translationAxes"], "y")
        for axes in ("y", "xy"):
            settings = config.level03_motion_config({"level03Motion": {"translationAxes": axes}})
            self.assertEqual(settings["translationAxes"], axes)
        for axes in ("x", "YX", "", None, True, 2, ["x", "y"]):
            with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, "translationAxes"):
                config.level03_motion_config({"level03Motion": {"translationAxes": axes}})

    def test_level03_car_limits_and_sampling_config_are_validated_and_saved(self):
        args = config.get_params(["--motion_level", "Level03"])
        cfg = config.configure_teacher_experiment({"env": {}}, args)
        settings = cfg["env"]["level03Motion"]
        self.assertEqual(settings["translationAxes"], "xy")
        self.assertEqual(settings["minSpeed"], .25)
        self.assertEqual(settings["maxSpeed"], .45)
        self.assertEqual(settings["maxYawRate"], .4)
        self.assertEqual(settings["driveDuration"], [6., 10.])
        self.assertEqual(settings["throttleUpdateTime"], [.4, .8])
        self.assertEqual(settings["steeringUpdateTime"], [.3, .65])
        self.assertEqual(settings["turnHoldTime"], [2., 4.])
        self.assertEqual(settings["stopHoldTime"], [.4, .9])
        self.assertEqual(settings["stopProbability"], 0.)
        self.assertEqual(settings["reverseProbability"], 0.)
        self.assertEqual(settings["straightProbability"], .35)
        self.assertEqual(settings["inputResponseTime"], .12)
        self.assertEqual(settings["throttleTrackingGain"], .4)
        self.assertEqual(settings["throttleVariation"], .18)
        self.assertEqual(settings["cruiseThrottleRange"], [.56, .95])
        self.assertEqual(settings["steeringTrackingGain"], .6)
        self.assertEqual(settings["steeringVariation"], .08)
        self.assertEqual(settings["turnMagnitudeRange"], [.45, .95])
        for key, invalid in (("maxSpeed", 0), ("maxYawRate", float("inf")),
                             ("stopProbability", 1.01), ("straightProbability", -.1),
                             ("driveDuration", [10., 6.]), ("stopHoldTime", [0., .9]),
                             ("throttleUpdateTime", [.6]), ("stopHoldTime", [True, .9])):
            with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(ValueError, key):
                config.level03_motion_config({"level03Motion": {key: invalid}})

    def test_level03_driver_config_validates_and_survives_saved_config(self):
        cases = [("reverseProbability", (-.1, 1.01, True, float("nan"), float("inf"), "0.4")),
                 ("inputResponseTime", (0., -.1, True, float("nan"), float("inf"), "0.18"))]
        for key in ("driveDuration", "throttleUpdateTime", "steeringUpdateTime", "turnHoldTime", "stopHoldTime"):
            cases.append((key, ([6., 4.], [0., 6.], [True, 6.], [4.],
                                [4., float("nan")], [4., float("inf")], "4,6")))
        for key in ("throttleTrackingGain", "steeringTrackingGain"):
            cases.append((key, (0., -.1, 1.01, True, float("nan"), float("inf"), "0.4")))
        for key in ("throttleVariation", "steeringVariation"):
            cases.append((key, (-.1, 1.01, True, float("nan"), float("inf"), "0.1")))
        for key in ("cruiseThrottleRange", "turnMagnitudeRange"):
            cases.append((key, ([-.1, .9], [.1, 1.1], [.8, .2], [True, .9], [.2],
                                [.2, float("nan")], [.2, float("inf")], ".2,.8")))
        for key, invalid_values in cases:
            for invalid in invalid_values:
                with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(ValueError, key):
                    config.level03_motion_config({"level03Motion": {key: invalid}})
        with self.assertRaises(ValueError):
            config.level03_motion_config({"level03Motion": {"stopProbability": .7, "reverseProbability": .5}})
        for key in ("throttleVariation", "steeringVariation"):
            for value in (0., 1.):
                with self.subTest(key=key, boundary=value):
                    self.assertEqual(config.level03_motion_config({"level03Motion": {key: value}})[key], value)
        for key in ("throttleTrackingGain", "steeringTrackingGain"):
            self.assertEqual(config.level03_motion_config({"level03Motion": {key: 1.}})[key], 1.)
        for key in ("cruiseThrottleRange", "turnMagnitudeRange"):
            for bounds in ([0., 0.], [0., 1.], [1., 1.]):
                with self.subTest(key=key, boundary=bounds):
                    self.assertEqual(config.level03_motion_config({"level03Motion": {key: bounds}})[key], bounds)

        source = ROOT / "data/cfg/DQ_teacher.yaml"
        for probability in (0., 1.):
            cfg = config.load_cfg(source)
            cfg["env"]["level03Motion"].update(
                stopProbability=0., reverseProbability=probability, driveDuration=[7., 9.],
                throttleUpdateTime=[.7, 1.1], steeringUpdateTime=[.5, .8],
                turnHoldTime=[3., 4.], stopHoldTime=[.5, .8], inputResponseTime=.2,
                throttleTrackingGain=.5, throttleVariation=0., cruiseThrottleRange=[.2, .8],
                steeringTrackingGain=1., steeringVariation=.2, turnMagnitudeRange=[.3, .9])
            cfg = config.configure_teacher_experiment(cfg, config.get_params(["--motion_level", "Level03"]))
            with tempfile.TemporaryDirectory() as directory:
                config.copy_cfg(source, directory, cfg=cfg)
                restored = config.configure_teacher_experiment(
                    config.load_cfg(Path(directory) / source.name), config.get_params([]))
            settings = restored["env"]["level03Motion"]
            self.assertEqual(settings["reverseProbability"], probability)
            self.assertEqual(settings["driveDuration"], [7., 9.])
            self.assertEqual(settings["throttleUpdateTime"], [.7, 1.1])
            self.assertEqual(settings["steeringUpdateTime"], [.5, .8])
            self.assertEqual(settings["turnHoldTime"], [3., 4.])
            self.assertEqual(settings["stopHoldTime"], [.5, .8])
            self.assertEqual(settings["inputResponseTime"], .2)
            self.assertEqual(settings["throttleTrackingGain"], .5)
            self.assertEqual(settings["throttleVariation"], 0.)
            self.assertEqual(settings["cruiseThrottleRange"], [.2, .8])
            self.assertEqual(settings["steeringTrackingGain"], 1.)
            self.assertEqual(settings["steeringVariation"], .2)
            self.assertEqual(settings["turnMagnitudeRange"], [.3, .9])

    def test_level03_legacy_sampling_keys_warn_and_are_removed_from_saved_config(self):
        legacy = {"maxSpeed": .22, "commandHoldTime": [100., 100.],
                  "fullTurnProbability": 1., "fullTurnHoldTime": [100., 100.]}
        with self.assertWarns(Warning):
            cfg = config.configure_teacher_experiment(
                {"env": {"level03Motion": legacy}}, config.get_params(["--motion_level", "Level03"]))
        settings = cfg["env"]["level03Motion"]
        self.assertEqual(settings["maxSpeed"], .22)
        self.assertEqual(settings["minSpeed"], 0.)
        self.assertEqual(settings["driveDuration"], [6., 10.])
        for key in ("commandHoldTime", "fullTurnProbability", "fullTurnHoldTime"):
            self.assertNotIn(key, settings)
        source = ROOT / "data/cfg/DQ_teacher.yaml"
        with tempfile.TemporaryDirectory() as directory:
            config.copy_cfg(source, directory, cfg=cfg)
            restored = config.load_cfg(Path(directory) / source.name)
        self.assertEqual(restored["env"]["level03Motion"], settings)

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
                                            history=7, qScale=2., rScale=.02, gfmFuturePoses=True))
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
        self.assertEqual(cfg["env"]["successMode"], "hold")
        self.assertEqual(cfg["env"]["holdSteps"], 10)
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

    def test_sampler_and_success_aliases_preserve_the_same_task(self):
        source = ROOT / "data/cfg/DQ_teacher.yaml"
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.touch()
            for mode in ("disabled", "uniform", "sgs"):
                for alias in ("legacy", "hold"):
                    with self.subTest(mode=mode, success_mode=alias):
                        args = config.get_params([
                            "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                            "--sgs_mode", mode, "--sgs_bank", str(bank),
                            "--success_mode", alias])
                        cfg = config.configure_teacher_experiment(config.load_cfg(source), args)
                        self.assertEqual(cfg["env"]["successMode"], "hold")
                        self.assertEqual(cfg["env"]["holdSteps"], 10)
                        self.assertEqual(cfg["env"]["numEnvs"], 4096)
                        self.assertEqual(cfg["reward"]["scales"]["pick_up"], 50.0)

    def test_custom_hold_length_is_preserved_and_invalid_lengths_are_rejected(self):
        args = config.get_params([])
        cfg = config.configure_teacher_experiment({"env": {"holdSteps": 15}}, args)
        self.assertEqual(cfg["env"]["holdSteps"], 15)
        for invalid in (0, -1, True, 2.5, "10"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "holdSteps"):
                config.configure_teacher_experiment({"env": {"holdSteps": invalid}}, args)

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
        self.assertEqual(cfg["env"]["holdSteps"], 1)
        self.assertEqual(cfg["env"]["robotStartDistanceRange"], [2.0, 2.0])
        self.assertEqual(cfg["env"]["D1_bench_task_level"], "Level02")
        self.assertEqual(args.task, "B1Z1PickMulti")
        self.assertEqual(args.object_name, "sugar_box")

    def test_strict_eval_uses_default_ten_steps_when_no_training_value_is_saved(self):
        args = config.get_params(["--checkpoint", "agent_24.pt"])
        cfg = config.configure_teacher_experiment({"env": {}}, args,
                                                  is_eval=True, strict_eval=True)
        self.assertEqual(cfg["env"]["holdSteps"], 10)

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
    def test_temporal_lookup_counts_follow_counted_episodes_including_partial_wave(self):
        from utils.tsil import (LOOKUP_NEAREST, LOOKUP_NO_SUCCESS, LOOKUP_OUT_OF_SUPPORT,
                                LOOKUP_INVALID)

        class TemporalEvaluationEnv(EvaluationEnv):
            tsil_enabled = True
            tsil_time_reference = .3
            dt, control_freq_low = .1, 1

            def __init__(self):
                super().__init__()
                self.tsil_state = SimpleNamespace(
                    state_dict=lambda: {"targets": torch.tensor([.3])}, bank_fingerprint="test")
                self.wave = 0

            def reset(self):
                states = super().reset()
                # Last slot in the final wave is inactive and must not count.
                statuses = [(LOOKUP_NEAREST, LOOKUP_NO_SUCCESS),
                            (LOOKUP_OUT_OF_SUPPORT, LOOKUP_INVALID),
                            (LOOKUP_NEAREST, LOOKUP_INVALID)]
                self.tsil_lookup_status = torch.tensor(statuses[self.wave])
                self.wave += 1
                return states

        trainer = SimpleNamespace(env=TemporalEvaluationEnv(), agents=EvaluationAgent(),
                                  headless=True, timesteps=120000, teacher_args=SimpleNamespace(seed=99))
        result = evaluator.evaluate_teacher(trainer, episodes=5)
        self.assertEqual(result["target_lookup_counts"],
                         dict(nearest=2, no_success=1, out_of_support=1, invalid=1))
        self.assertEqual(result["target_lookup_nearest_fraction"], 2 / 5)
        self.assertTrue(result["tsil_targets_unchanged"])

    def test_success_alias_does_not_change_evaluation_or_saved_hold_length(self):
        for alias in ("legacy", "hold"):
            with self.subTest(success_mode=alias):
                env = EvaluationEnv()
                env.cfg["env"]["successMode"] = alias
                trainer = SimpleNamespace(env=env, agents=EvaluationAgent(), headless=True,
                                          timesteps=120000, teacher_args=SimpleNamespace(seed=99))
                result = evaluator.evaluate_teacher(trainer, episodes=3)
                self.assertEqual(result["hold_steps"], env.hold_steps)
                self.assertEqual(result["hold_success_rate"], 1 / 3)

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
