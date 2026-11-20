"""T1 action parameterization: legacy clipping versus normalized unit actions.

legacy:     raw outputs are clipped at the physical limits (unchanged behavior).
normalized: raw outputs are clipped to [-1, 1], then scaled to the same limits.
Runs on CPU without Isaac Gym, through the production methods.
"""

import ast
import math
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

# Module imports keep the gripper tests from being collected a second time here.
import test_gripper_control as gripper
from test_teacher_gfm import Policy
from utils import config as config_utils


ROOT = Path(__file__).resolve().parents[1]
ACTIONS, GRASPS = 9, 180
LIMITS = [0.02] * 3 + [0.06] * 3 + [1.0, 0.45, 0.6]


def module_constant(name):
    """A literal module-level constant of b1z1_base.py, read without importing it."""
    tree = ast.parse((ROOT / "envs" / "b1z1_base.py").read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == name for target in node.targets))


# The dead-zone branch of pre_physics_step reads these module constants.
for _name in ("LIN_VEL_X_CLIP", "ANG_VEL_YAW_CLIP"):
    gripper.ControlEnvironment.pre_physics_step.__globals__.setdefault(_name, module_constant(_name))


def saturation_method():
    """_record_action_saturation with the module-level dimension names it reads."""
    source = ROOT / "envs" / "b1z1_base.py"
    tree = ast.parse(source.read_text())
    names = next(node for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "ACTION_DIMENSION_NAMES"
                         for target in node.targets))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "B1Z1Base")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_record_action_saturation")
    namespace = {"torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[names, method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_record_action_saturation"], namespace["ACTION_DIMENSION_NAMES"]


RECORD_SATURATION, DIMENSION_NAMES = saturation_method()


class ConfigurationTests(unittest.TestCase):
    def test_missing_key_means_legacy_and_repository_yaml_keeps_legacy(self):
        self.assertEqual(config_utils.validate_action_scaling({}), "legacy")
        cfg = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertEqual(cfg["env"]["actionScaling"], "legacy")

    def test_invalid_mode_and_tanh_combination_are_rejected(self):
        with self.assertRaises(ValueError):
            config_utils.validate_action_scaling({"actionScaling": "tanh"})
        with self.assertRaises(ValueError):
            config_utils.validate_action_scaling({"actionScaling": "normalized", "useTanh": True})
        self.assertEqual(config_utils.validate_action_scaling(
            {"actionScaling": "legacy", "useTanh": True}), "legacy")

    def test_cli_overrides_yaml_and_absent_cli_keeps_yaml(self):
        parse = config_utils.get_params
        self.assertIsNone(parse([]).action_scaling)
        cfg = {"env": {"actionScaling": "normalized"}}
        self.assertEqual(config_utils.apply_action_scaling_config(cfg, parse([])), "normalized")
        self.assertEqual(config_utils.apply_action_scaling_config(
            cfg, parse(["--action_scaling", "legacy"])), "legacy")
        self.assertEqual(cfg["env"]["actionScaling"], "legacy")
        cfg = {"env": {}}
        self.assertEqual(config_utils.apply_action_scaling_config(cfg, parse([])), "legacy")
        self.assertEqual(cfg["env"]["actionScaling"], "legacy")
        with self.assertRaises(SystemExit):
            parse(["--action_scaling", "tanh"])

    def test_student_restores_teacher_mode_only_when_saved(self):
        for saved, expected in (({"actionScaling": "normalized"}, "normalized"), ({}, None)):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                (run / "checkpoints").mkdir()
                checkpoint = run / "checkpoints" / "agent_1.pt"
                checkpoint.touch()
                (run / "DQ_teacher.yaml").write_text(yaml.safe_dump({"env": saved}))
                cfg = {"env": {}}
                config_utils.restore_student_teacher_observation_config(cfg, str(checkpoint))
                self.assertEqual(cfg["env"].get("actionScaling"), expected)


class StepMappingTests(unittest.TestCase):
    """pre_physics_step and clip_commands, with the gripper test's stand-in state."""

    helpers = gripper.GripperControlTests()

    def step(self, raw, mode=None, curriculum=True):
        raw = torch.tensor(raw, dtype=torch.float32)
        env = self.helpers.environment([0.0] * raw.shape[0])
        self.helpers.prepare_step(env)
        if mode is not None:
            env.cfg["env"]["actionScaling"] = mode
        env.commands_curriculum = curriculum
        before = raw.clone()
        env.pre_physics_step(raw)
        torch.testing.assert_close(raw, before)  # PPO's stored sample is not mutated
        return env

    def test_legacy_mode_keeps_clipping_at_physical_limits(self):
        raw = [[0.5, -0.5, 0.01, 0.1, -0.1, 0.03, 2.0, 1.0, -1.0]]
        for mode in (None, "legacy"):
            with self.subTest(mode=mode):
                env = self.step(raw, mode)
                torch.testing.assert_close(env.delta_goal_cart, torch.tensor([[0.02, -0.02, 0.01]]))
                torch.testing.assert_close(env.delta_goal_orn, torch.tensor([[0.06, -0.06, 0.03]]))
                torch.testing.assert_close(env.commands[:, [0, 2]], torch.tensor([[0.45, -0.6]]))
                # Observations and rate penalties still see the raw output.
                torch.testing.assert_close(env.action_history_buf[:, -1], torch.tensor(raw))
                torch.testing.assert_close(env.actions, torch.tensor(raw))

    def test_normalized_mode_scales_unit_actions_and_saturates_beyond_one(self):
        raw = [[0.5, -0.5, 2.0, 0.5, -3.0, 0.25, -2.0, 0.5, -0.5]]
        clipped = torch.tensor(raw).clamp(-1, 1)
        env = self.step(raw, "normalized")
        torch.testing.assert_close(env.delta_goal_cart, torch.tensor([[0.01, -0.01, 0.02]]))
        torch.testing.assert_close(env.delta_goal_orn, torch.tensor([[0.03, -0.06, 0.015]]))
        torch.testing.assert_close(env.commands[:, [0, 2]], torch.tensor([[0.225, -0.3]]))
        torch.testing.assert_close(env.gripper_dof_pos, torch.zeros(1, 1))  # fully closed
        torch.testing.assert_close(env.actions, clipped)
        torch.testing.assert_close(env.action_history_buf[:, -1], clipped)
        torch.testing.assert_close(env.clipped_actions[:, :6],
                                   torch.cat((env.delta_goal_cart, env.delta_goal_orn), dim=-1))

    def test_unit_bounds_reach_the_same_physical_limits_as_legacy(self):
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                legacy = self.step([[sign * 100.0] * ACTIONS], "legacy")
                normalized = self.step([[sign] * ACTIONS], "normalized")
                for name in ("delta_goal_cart", "delta_goal_orn", "commands", "gripper_dof_pos"):
                    torch.testing.assert_close(getattr(normalized, name), getattr(legacy, name))

    def test_normalized_mode_without_command_curriculum_uses_point_four(self):
        env = self.step([[0.0] * 7 + [0.5, 0.0]], "normalized", curriculum=False)
        torch.testing.assert_close(env.commands[:, 0], torch.tensor([0.2]))

    def test_normalized_mode_keeps_the_small_command_dead_zone(self):
        raw = torch.tensor([[0.0] * 7 + [0.3, 0.5], [0.0] * 7 + [0.5, 0.0]])
        env = self.helpers.environment([0.0, 0.0])
        self.helpers.prepare_step(env)
        env.cfg["env"].update(actionScaling="normalized", smallValueSetZero=True)
        env.commands_curriculum = True
        env.pre_physics_step(raw)
        # 0.3 * 0.45 = 0.135 m/s and 0.5 * 0.6 = 0.3 rad/s are both inside the
        # dead zone (0.15 m/s, 0.35 rad/s), so the whole command is zeroed.
        self.assertEqual((module_constant("LIN_VEL_X_CLIP"), module_constant("ANG_VEL_YAW_CLIP")),
                         (0.15, 0.35))
        torch.testing.assert_close(env.commands[0], torch.zeros(3))
        torch.testing.assert_close(env.commands[1, 0], torch.tensor(0.225))


class SaturationMetricTests(unittest.TestCase):
    def environment(self, mode, use_tanh=False):
        env = type("Env", (), {"_record_action_saturation": RECORD_SATURATION})()
        env.cfg = {"env": {"useTanh": use_tanh}}
        env.action_scaling = mode
        env.commands_curriculum = True
        env.extras = {}
        return env

    def test_fraction_outside_the_executed_range_per_dimension(self):
        legacy = torch.tensor([LIMITS, [2 * value for value in LIMITS],
                               [0.0] * ACTIONS, [-3 * value for value in LIMITS]])
        env = self.environment("legacy")
        env._record_action_saturation(legacy)
        self.assertEqual(tuple(env.extras["action_saturation"]), DIMENSION_NAMES)
        for value in env.extras["action_saturation"].values():
            self.assertAlmostEqual(float(value), 0.5)  # equal to the limit is not saturated
        env = self.environment("normalized")
        env._record_action_saturation(legacy)  # |0.02| < 1: nothing saturates in unit space
        self.assertEqual(float(env.extras["action_saturation"]["ee_dx"]), 0.0)
        self.assertEqual(float(env.extras["action_saturation"]["gripper"]), 0.5)

    def test_tanh_and_other_action_widths_are_not_reported(self):
        env = self.environment("legacy", use_tanh=True)
        env._record_action_saturation(torch.ones(2, ACTIONS))
        self.assertNotIn("action_saturation", env.extras)
        env = self.environment("legacy")
        env._record_action_saturation(torch.ones(2, ACTIONS + 1))
        self.assertNotIn("action_saturation", env.extras)


class PolicyInitializationTests(unittest.TestCase):
    features, body, future = 16, 24, 13

    def model(self, mode=None):
        torch.manual_seed(5)
        kwargs = {"num_future_pose_obs": self.future}
        if mode is not None:
            kwargs["action_scaling"] = mode
        width = self.features + self.body + self.future + GRASPS + ACTIONS
        return Policy(width, ACTIONS, "cpu", self.features, 128, None, None, None, **kwargs)

    def states(self, batch=4):
        width = self.features + self.body + self.future + GRASPS + ACTIONS
        return torch.randn(batch, width, generator=torch.Generator().manual_seed(3))

    def test_legacy_initialization_is_unchanged(self):
        reference, legacy = self.model(), self.model("legacy")
        torch.testing.assert_close(legacy.log_std_parameter, torch.zeros(ACTIONS))
        for (name, expected), (_, value) in zip(reference.state_dict().items(),
                                                legacy.state_dict().items()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(value, expected)

    def test_normalized_initialization_starts_near_zero_with_half_range_std(self):
        policy = self.model("normalized")
        torch.testing.assert_close(policy.log_std_parameter.exp(), torch.full((ACTIONS,), 0.5))
        torch.testing.assert_close(policy.net[-1].bias, torch.zeros(ACTIONS))
        legacy = self.model("legacy")
        torch.testing.assert_close(policy.net[-1].weight, legacy.net[-1].weight * 0.01)
        mean = policy.compute({"states": self.states()}, role="policy")[0]
        self.assertLess(float(mean.abs().max()), 0.1)

    def test_normalized_std_is_capped_at_the_full_range(self):
        policy = self.model("normalized")
        with torch.no_grad():
            policy.log_std_parameter.fill_(1.0)
        policy.act({"states": self.states()}, role="policy")
        self.assertTrue(math.isclose(float(policy.get_log_std(role="policy").max()), 0.0,
                                     abs_tol=1e-6))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model("tanh")


class EvaluationMetadataTests(unittest.TestCase):
    def test_eval_and_temporal_checkpoints_restore_the_mode(self):
        source = (ROOT / "train_multistate_DQ_teacher.py").read_text()
        restore = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                       and node.name == "restore_eval_arguments")
        names = {node.value for node in ast.walk(restore) if isinstance(node, ast.Constant)}
        self.assertIn("action_scaling", names)
        tsil = (ROOT / "utils" / "tsil_checkpoint.py").read_text()
        observation_arguments = next(
            node.value for node in ast.parse(tsil).body if isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "OBSERVATION_ARGUMENTS" for target in node.targets))
        self.assertIn("action_scaling", ast.literal_eval(observation_arguments))


if __name__ == "__main__":
    unittest.main()
