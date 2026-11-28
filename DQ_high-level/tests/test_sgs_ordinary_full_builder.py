"""Ordinary SGS starts retain reset state; curriculum preparation is isolated."""

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_sgs_env import NAMESPACE, quat_from_euler_xyz
from utils.config import level03_motion_config
from utils.sgs_bank_builder import PhysicsBankBuilder, snapshot
from utils.sgs_build_config import load_build_config
from utils.sgs_grounding import FOOT_NAMES
from utils.table_motion import RemoteCarMotion


@contextmanager
def quaternion_imports():
    """Replace Isaac Gym's import boundary with equivalent CPU quaternion math."""
    package, math_module = ModuleType("isaacgym"), ModuleType("isaacgym.torch_utils")
    for name, value in NAMESPACE.items():
        if name.startswith("quat_") or name == "euler_from_quat":
            setattr(math_module, name, value)
    package.torch_utils = math_module
    with patch.dict(sys.modules, {"isaacgym": package, "isaacgym.torch_utils": math_module}):
        yield


class OrdinaryFullBuilderTests(unittest.TestCase):
    def builder(self, level="Level03"):
        count = 4
        env = SimpleNamespace(num_envs=count, device="cpu", table_level=level,
                              sim_params=SimpleNamespace(dt=.005))
        root = torch.zeros(count, 13)
        root[:, 6] = 1.
        env._robot_root_states = root.clone()
        env._robot_root_states[:, :3] = torch.tensor([-2., 2., .5998])
        yaw = torch.deg2rad(torch.tensor([-14., -3., 8., 14.]))
        env._robot_root_states[:, 3:7] = quat_from_euler_xyz(torch.zeros(count), torch.zeros(count), yaw)
        env._robot_root_states[:, 7:13] = torch.arange(6) * .004
        env._cube_root_states, env._table_root_states = root.clone(), root.clone()
        env._cube_root_states[:, :3] = torch.tensor([0., 2., .585])
        env._table_root_states[:, :3] = torch.tensor([0., 2., .50])
        env._cube_root_states[:, 3:7] = env._robot_root_states[:, 3:7]
        # Preserve the contact-induced velocity produced by the reset's own
        # physics tick; it is not a commanded nonzero Level03 start.
        env._table_root_states[:, 7] = .0003
        env.table_heights = torch.full((count,), .55)
        env.init_height = torch.full((count,), .035)
        env.init_quat = root[:, 3:7].clone()
        env.cube_start_pose = SimpleNamespace(p=SimpleNamespace(x=0., y=2.))
        env._dof_pos = torch.zeros(count, 19)
        env._dof_pos[:, 12:18] = torch.arange(6) * .1
        env._dof_pos[:, -1] = torch.tensor([-.2, -.5, -.8, -1.1])
        env._dof_vel = torch.full_like(env._dof_pos, .03)
        env.curr_ee_goal_cart = torch.tensor([[.46, .0, .66]]).repeat(count, 1)
        env.curr_ee_goal_orn_rpy = torch.tensor([[1.57, .0, .0]]).repeat(count, 1)
        env.gripper_dof_pos = torch.tensor([[-.25], [-.55], [-.85], [-1.15]])
        env.dof_limits_lower = torch.full((19,), -1.57)
        env.dof_limits_upper = torch.full((19,), 1.57)
        env.dof_limits_upper[-1] = 0.
        env.vel_line_forward_y_level02 = torch.tensor([.2, -.2, .25, -.25])
        env.lin_circle_select = torch.tensor([False, True, True, False])
        env.vel_angular_yaw = torch.tensor([.1, -.1, .2, -.2])
        env.rand_change_step = torch.tensor([40, 45, 50, 55])
        env._rigid_body_state = torch.zeros(count, 6, 13)
        env._rigid_body_state[:, :, 6] = 1.
        env._rigid_body_state[:, :, 2] = .12  # Ordinary startup has airborne feet.
        env._rigid_body_state[:, :, 7] = .023
        env._rigid_body_pos = env._rigid_body_state[:, :, :3]
        env._contact_forces = torch.zeros(count, 6, 3)
        env.body_names_to_idx = dict(zip((*FOOT_NAMES, "trunk", "gripperMover"), range(6)))
        env.ee_j_eef = torch.ones(count, 6, 6)
        env.table_chassis_collision = torch.zeros(count, dtype=torch.bool)
        env.level03_motion = RemoteCarMotion(count, "cpu", level03_motion_config({}))
        env.reset_idx = Mock(side_effect=env.level03_motion.reset)
        env.gym = SimpleNamespace(simulate=Mock())

        builder = object.__new__(PhysicsBankBuilder)
        builder.env = env
        builder.reset_config = load_build_config()
        builder._place_robot, builder._ground_robot = Mock(), Mock()
        builder._physics_window = Mock()
        builder.sampling_audit = Mock()
        builder.per_family = count
        builder.family_names = ("full_task", "pregrasp", "at_grasp", "near_goal")
        builder.parts, builder.counts = [[] for _ in range(4)], [0] * 4
        return builder

    def test_full_snapshot_preserves_ordinary_roots_dofs_targets_and_transients(self):
        for level in ("Level02", "Level03"):
            with self.subTest(level=level), quaternion_imports():
                builder = self.builder(level)
                expected = snapshot(builder.env)
                state, checks = builder._full()
                builder.env.reset_idx.assert_called_once_with()
                builder.env.gym.simulate.assert_not_called()
                builder._place_robot.assert_not_called()
                builder._ground_robot.assert_not_called()
                builder._physics_window.assert_not_called()
                for name, value in expected.items():
                    torch.testing.assert_close(state[name], value, atol=0., rtol=0., msg=name)
                self.assertTrue(all(bool(value.all()) for value in checks.values()))
                self.assertNotIn("base_motion_alignment", checks)
                self.assertNotIn("feet_near_ground", checks)
                torch.testing.assert_close(state["sampled_object_yaw_offset_deg"],
                                           state["object_yaw_offset_deg"], atol=0., rtol=0.)
                builder.env._dof_pos.add_(1.)
                torch.testing.assert_close(state["dof_pos"], expected["dof_pos"], atol=0., rtol=0.)

    def test_course_initialization_samples_motion_and_object_yaw_without_mutating_full(self):
        with quaternion_imports():
            builder = self.builder()
            initial, _ = builder._full()
            before = {name: value.clone() for name, value in initial.items()}
            builder.reset_config["object"]["yaw_offset_range_deg"] = [25., 25.]
            course = builder._curriculum_initial(initial)
            for name, value in before.items():
                torch.testing.assert_close(initial[name], value, atol=0., rtol=0., msg=name)
            for prefix in ("level03_", "level03_x_"):
                self.assertTrue(course[prefix + "driver_initialized"].all())
                self.assertTrue((course[prefix + "driver_direction"].abs() == 1).all())
                self.assertTrue((course[prefix + "motion_speed"] == 0).all())
                self.assertFalse(initial[prefix + "driver_initialized"].any())
            expected_q = quat_from_euler_xyz(torch.zeros(4), torch.zeros(4),
                                             torch.full((4,), torch.deg2rad(torch.tensor(25.)).item()))
            torch.testing.assert_close(course["object_root"][:, 3:7], expected_q)
            self.assertNotEqual(course["object_root"].data_ptr(), initial["object_root"].data_ptr())
            for name in ("robot_root", "table_root", "dof_pos", "dof_vel", "gripper_target"):
                torch.testing.assert_close(course[name], initial[name], atol=0., rtol=0.)

    def test_rest_drift_exemption_applies_only_to_ordinary_full_rows(self):
        with quaternion_imports():
            builder = self.builder()
            state, checks = builder._full()
            self.assertGreater(state["table_root"][:, 7].abs().min(), 1e-5)
            self.assertEqual(builder._collect(0, state, checks), 4)
            course = builder._curriculum_initial(state)
            self.assertEqual(builder._collect(1, course, checks), 0)
            self.assertEqual(builder.counts, [4, 0, 0, 0])
            torch.testing.assert_close(builder.parts[0][0]["table_root"], state["table_root"])

    def test_ordinary_snapshot_checks_contact_and_finiteness_without_pose_resampling(self):
        with quaternion_imports():
            builder = self.builder()
            builder.env.table_chassis_collision[1] = True
            builder.env._dof_vel[2, 0] = float("nan")
            state, checks = builder._full()
            self.assertEqual(checks["table_chassis_clear"].tolist(), [True, False, True, True])
            self.assertEqual(checks["finite_state"].tolist(), [True, True, False, True])
            self.assertEqual(builder._collect(0, state, checks), 2)
            builder.env.reset_idx.assert_called_once_with()

    def test_at_grasp_targets_candidate_exactly_with_nonzero_opening(self):
        with quaternion_imports():
            builder = self.builder()
            env = builder.env
            initial, _ = builder._full()
            env._robot_root_states[:, 3:7] = torch.tensor([0., 0., 0., 1.])
            env.arm_base = torch.zeros(4, 3)
            point = torch.tensor([[.45, .0, .60]]).repeat(4, 1)
            env.get_grasp_world = Mock(return_value=(point[:, None, :], torch.zeros(4, 1, 3)))
            env.ee_pos = point.clone()
            env.ee_orn = torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1)
            builder.kinematics = SimpleNamespace(solve=Mock(return_value=(
                torch.zeros(4, 6), torch.zeros(4), torch.zeros(4))))
            builder._base_in_range = Mock(return_value=torch.ones(4, dtype=torch.bool))
            builder._base_motion_aligned = Mock(return_value=torch.ones(4, dtype=torch.bool))

            def checked_window(gripper):
                env.gripper_dof_pos.copy_(gripper)
                return {"physical_window": torch.ones(4, dtype=torch.bool)}, {
                    "validation_peak_contact_force_n": torch.zeros(4),
                    "validation_max_object_displacement_m": torch.zeros(4)}

            builder._physics_window = Mock(side_effect=checked_window)
            with patch("utils.sgs_bank_builder.restore_snapshot"):
                state, checks = builder._grasp_pose(initial, "at_grasp")
            requested_position, requested_rotation = builder.kinematics.solve.call_args.args[:2]
            torch.testing.assert_close(requested_position, point, atol=0., rtol=0.)
            torch.testing.assert_close(requested_rotation, torch.eye(3).expand(4, 3, 3), atol=0., rtol=0.)
            torch.testing.assert_close(state["sampled_candidate_local_offset"], torch.zeros(4, 3))
            torch.testing.assert_close(state["sampled_ee_rotation_offset_deg"], torch.zeros(4, 3))
            self.assertTrue(((state["gripper_opening"] >= .2)
                             & (state["gripper_opening"] <= 1.)).all())
            self.assertTrue(checks["ee_distance"].all())
            self.assertTrue(checks["ee_orientation"].all())
            self.assertTrue(checks["gripper_opening"].all())


if __name__ == "__main__":
    unittest.main()
