"""Exercise continuous gripper commands without importing Isaac Gym."""

import ast
import math
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "envs" / "b1z1_base.py"


def quat_apply(quaternion, vector):
    cross = torch.cross(quaternion[..., :3], vector, dim=-1)
    return vector + 2 * (quaternion[..., 3:] * cross
                         + torch.cross(quaternion[..., :3], cross, dim=-1))


def quat_mul(left, right):
    vector = (left[..., 3:] * right[..., :3] + right[..., 3:] * left[..., :3]
              + torch.cross(left[..., :3], right[..., :3], dim=-1))
    scalar = left[..., 3:] * right[..., 3:] - (left[..., :3] * right[..., :3]).sum(-1, keepdim=True)
    return torch.cat((vector, scalar), dim=-1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    return torch.stack((sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy), dim=-1)


def source_method(name):
    """Compile the production method, keeping simulator imports out of tests."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "B1Z1Base")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "quat_apply": quat_apply,
                 "quat_mul": quat_mul, "quat_from_euler_xyz": quat_from_euler_xyz}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


class ControlEnvironment:
    set_gripper = source_method("set_gripper")
    pre_physics_step = source_method("pre_physics_step")
    clip_commands = source_method("clip_commands")
    _setup_obs_and_action_info = source_method("_setup_obs_and_action_info")


class GripperControlTests(unittest.TestCase):
    device = "cpu"

    def tensor(self, values):
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def environment(self, commands, lower=None, upper=None):
        env = ControlEnvironment()
        env.device, env.num_envs = self.device, len(commands)
        lower = [-math.pi / 2] if lower is None else lower
        upper = [0.0] if upper is None else upper
        env.num_gripper_dof = len(lower)
        env.actions = torch.zeros(env.num_envs, 9, device=self.device)
        env.actions[:, 6] = self.tensor(commands)
        # Prefix joints ensure that only the final gripper limits are used.
        env.dof_limits_lower = self.tensor([-2.0] * 6 + lower)
        env.dof_limits_upper = self.tensor([2.0] * 6 + upper)
        env.gripper_dof_pos = torch.full((env.num_envs, env.num_gripper_dof),
                                         123.0, device=self.device)
        env._dof_pos = torch.full((env.num_envs, 6 + env.num_gripper_dof),
                                  0.125, device=self.device)
        env._dof_vel = torch.full_like(env._dof_pos, 0.25)
        return env

    def prepare_step(self, env, use_tanh=False):
        count = env.num_envs
        env.cfg = {"env": {"useTanh": use_tanh, "smallValueSetZero": False}}
        env.stu_distill = False
        env.global_step_counter, env.total_timesteps = 0, 100
        env.arm_delay = False
        env.action_history_buf = torch.zeros(count, 2, 9, device=self.device)
        env.command_history_buf = torch.zeros_like(env.action_history_buf)
        env.curr_ee_goal_cart = torch.zeros(count, 3, device=self.device)
        env.curr_ee_goal_orn_rpy = torch.zeros_like(env.curr_ee_goal_cart)
        env.arm_base = torch.zeros_like(env.curr_ee_goal_cart)
        env._robot_root_states = torch.zeros(count, 13, device=self.device)
        env._robot_root_states[:, 6] = 1.0
        env.pitch_control, env.floating_base = False, False
        env.commands_curriculum = False
        env.commands = torch.zeros(count, 3, device=self.device)
        env.clipped_actions = torch.zeros_like(env.actions)
        env.stop_pick, env.rand_depth_clip = False, False

    def test_b1_closed_partial_and_open_targets(self):
        env = self.environment([-1.0, -0.5, 0.0, 0.5, 1.0])
        env.set_gripper()
        expected = self.tensor([[0.0], [-math.pi / 8], [-math.pi / 4],
                                [-3 * math.pi / 8], [-math.pi / 2]])
        torch.testing.assert_close(env.gripper_dof_pos, expected)

    def test_commands_outside_the_range_saturate_at_joint_limits(self):
        env = self.environment([-100.0, -1.0, 1.0, 100.0])
        env.set_gripper()
        torch.testing.assert_close(env.gripper_dof_pos,
                                   self.tensor([[0.0], [0.0], [-math.pi / 2], [-math.pi / 2]]))

    def test_increasing_command_increases_opening_monotonically(self):
        env = self.environment(torch.linspace(-1, 1, 101).tolist())
        env.set_gripper()
        self.assertTrue(bool((torch.diff(env.gripper_dof_pos[:, 0]) < 0).all()))
        torch.testing.assert_close(env.gripper_dof_pos[50], self.tensor([-math.pi / 4]))

    def test_two_dof_gripper_uses_each_joint_own_limits(self):
        env = self.environment([-1.0, 0.0, 1.0], lower=[-1.5, 0.1], upper=[0.1, 0.9])
        env.set_gripper()
        torch.testing.assert_close(env.gripper_dof_pos,
                                   self.tensor([[0.1, 0.9], [-0.7, 0.5], [-1.5, 0.1]]))

    def test_single_and_two_dof_targets_preserve_batch_shape_and_device(self):
        for count in (1, 4):
            for dofs in (1, 2):
                with self.subTest(count=count, dofs=dofs):
                    env = self.environment([0.0] * count,
                                           lower=[-1.0] * dofs, upper=[0.0] * dofs)
                    env.set_gripper()
                    self.assertEqual(env.gripper_dof_pos.shape, (count, dofs))
                    self.assertEqual(env.gripper_dof_pos.device, env.actions.device)
                    torch.testing.assert_close(env.gripper_dof_pos,
                                               torch.full_like(env.gripper_dof_pos, -0.5))

    def test_changing_one_environment_command_does_not_change_other_targets(self):
        env = self.environment([-0.8, 0.0, 0.6], lower=[-1.0, -2.0], upper=[0.0, 0.0])
        env.set_gripper()
        initial = env.gripper_dof_pos.clone()
        env.actions[1, 6] = 0.4
        env.set_gripper()
        torch.testing.assert_close(env.gripper_dof_pos[[0, 2]], initial[[0, 2]])
        torch.testing.assert_close(env.gripper_dof_pos[1], self.tensor([-0.7, -1.4]))

    def test_setting_targets_does_not_teleport_measured_joints_or_mutate_actions(self):
        env = self.environment([-2.0, 0.25, 2.0])
        before = {name: getattr(env, name).clone()
                  for name in ("actions", "_dof_pos", "_dof_vel",
                               "dof_limits_lower", "dof_limits_upper")}
        env.set_gripper()
        for name, expected in before.items():
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(env, name), expected)
        self.assertFalse(torch.equal(env.gripper_dof_pos, env._dof_pos[:, -1:]))

    def test_unsupported_gripper_dof_count_is_rejected(self):
        for dofs in (0, 3):
            with self.subTest(dofs=dofs):
                env = self.environment([0.0, 0.5], lower=[-1.0] * dofs, upper=[0.0] * dofs)
                with self.assertRaises(NotImplementedError):
                    env.set_gripper()

    def test_step_records_clipped_continuous_gripper_commands_in_nine_actions(self):
        commands = [-2.0, -0.5, 0.0, 0.5, 2.0]
        expected_commands = self.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        for dofs in (1, 2):
            for use_tanh in (False, True):
                with self.subTest(dofs=dofs, use_tanh=use_tanh):
                    env = self.environment(commands, lower=[-1.0] * dofs, upper=[0.0] * dofs)
                    self.prepare_step(env, use_tanh=use_tanh)
                    raw = env.actions.clone()
                    env.pre_physics_step(raw)
                    self.assertEqual(env.actions.shape, (5, 9))
                    self.assertEqual(env.clipped_actions.shape, (5, 9))
                    self.assertEqual(env.command_history_buf.shape, (5, 2, 9))
                    torch.testing.assert_close(env.clipped_actions[:, 6], expected_commands)
                    torch.testing.assert_close(env.command_history_buf[:, -1, 6], expected_commands)
                    torch.testing.assert_close(env.gripper_dof_pos[:, 0],
                                               self.tensor([0.0, -0.25, -0.5, -0.75, -1.0]))
                    torch.testing.assert_close(raw[:, 6], self.tensor(commands))

    def test_default_teacher_action_space_remains_nine_dimensional(self):
        env = ControlEnvironment()
        env.cfg = {"env": {}}
        env.floating_base, env.pitch_control = False, False
        env.use_roboinfo, env.enable_camera = True, False
        env._setup_obs_and_action_info(removed_dim=0)
        self.assertEqual(env.cfg["env"]["numActions"], 9)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class GripperControlCudaTests(GripperControlTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
