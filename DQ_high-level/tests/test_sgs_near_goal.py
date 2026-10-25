"""Check near-goal acceptance with independent scripted physical observations.

No Isaac Gym module is imported. These tests provide histories with known
outcomes, then exercise the real hold acceptance and diagnostic accumulation.
"""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from utils.sgs_bank_builder import PhysicsBankBuilder
from utils.sgs_grounding import (
    FOOT_NAMES, FOOT_RADIUS, grasp_contact_body_indices,
)


class NearGoalHoldTests(unittest.TestCase):
    def make_builder(self, frames, count=3):
        env = SimpleNamespace(
            num_envs=count, device="cpu", sim_params=SimpleNamespace(dt=.04),
            table_heights=torch.full((count,), .5),
            init_height=torch.full((count,), .1), lifted_success_threshold=.2,
        )
        env._cube_root_states = torch.zeros(count, 13)
        env._cube_root_states[:, 2] = .675
        env._cube_root_states[:, 6] = 1
        env._robot_root_states = torch.zeros(count, 13)
        env._robot_root_states[:, 6] = 1
        names = (*FOOT_NAMES, "link06", "gripperMover")
        env.body_names_to_idx = dict(zip(names, range(len(names))))
        env._rigid_body_pos = torch.zeros(count, len(names), 3)
        env._rigid_body_pos[:, :4, 2] = FOOT_RADIUS + .001
        env._contact_forces = torch.zeros(count, len(names), 3)
        env._contact_forces[:, 4:, 0] = 5
        env.relative_position = torch.zeros(count, 3)
        env.relative_quaternion = torch.zeros(count, 4)
        env.relative_quaternion[:, 3] = 1
        env.contacts = torch.ones(count, dtype=torch.bool)
        env.between = torch.ones(count, dtype=torch.bool)
        env.clear_of_table = torch.ones(count, dtype=torch.bool)

        builder = object.__new__(PhysicsBankBuilder)
        builder.env = env
        builder.reset_config = {
            "near_goal": {
                "hold_time_s": len(frames) * env.sim_params.dt,
                "object_lift_height_range_m": [.05, .10],
                "max_relative_slip_m": .01,
                "max_relative_rotation_deg": 10.,
                "min_contact_fraction": .8,
                "max_object_speed_m_s": .1,
            },
            "validation": {"max_contact_force_n": 5000.},
        }

        def advance(_position, _quaternion, _closed, step, _table):
            frames[step](env)

        builder._near_goal_step = Mock(side_effect=advance)
        builder._frame_checks = Mock(side_effect=lambda: {
            "frame_finite": torch.ones(count, dtype=torch.bool),
        })
        builder._grasp_geometry = Mock(side_effect=lambda: (
            env.contacts.clone(), env.between.clone(), env.clear_of_table.clone()))
        builder._grasp_relative_pose = Mock(side_effect=lambda: (
            env.relative_position.clone(), env.relative_quaternion.clone()))
        return builder

    def hold(self, builder):
        env = builder.env
        position = torch.zeros(env.num_envs, 3)
        quaternion = torch.zeros(env.num_envs, 4)
        quaternion[:, 3] = 1
        closed = torch.zeros(env.num_envs, 1)
        return builder._near_goal_hold(position, quaternion, closed)

    @staticmethod
    def unchanged(_env):
        pass

    def test_airborne_motionless_object_without_finger_contact_is_not_a_grasp(self):
        def observed_frame(env):
            # Same valid height and zero slip for both states. A temporarily
            # suspended object must not pass just because it has not fallen yet.
            env.contacts[0] = False
            env._contact_forces[0, 4:] = 0

        builder = self.make_builder([observed_frame] * 5, count=2)
        checks, evidence = self.hold(builder)
        self.assertEqual(checks["finger_contact"].tolist(), [False, True])
        self.assertEqual(checks["final_finger_contact"].tolist(), [False, True])
        torch.testing.assert_close(evidence["near_goal_contact_fraction"], torch.tensor([0., 1.]))
        self.assertTrue(bool(checks["lift_height"].all()))
        self.assertTrue(bool(checks["relative_slip"].all()))
        self.assertTrue(all(bool(values[1]) for values in checks.values()))

    def test_transient_slip_and_rotation_fail_even_after_pose_returns(self):
        def transient(env):
            env.relative_position[0, 0] = .025
            angle = math.radians(20.) / 2
            env.relative_quaternion[1] = torch.tensor([0., 0., math.sin(angle), math.cos(angle)])

        def recovered(env):
            env.relative_position.zero_()
            env.relative_quaternion.zero_()
            env.relative_quaternion[:, 3] = 1

        builder = self.make_builder([self.unchanged, transient, recovered,
                                     self.unchanged, self.unchanged])
        checks, evidence = self.hold(builder)
        self.assertEqual(checks["relative_slip"].tolist(), [False, True, True])
        self.assertEqual(checks["relative_rotation"].tolist(), [True, False, True])
        torch.testing.assert_close(evidence["near_goal_relative_slip_m"], torch.tensor([.025, 0., 0.]))
        torch.testing.assert_close(evidence["near_goal_relative_rotation_deg"],
                                   torch.tensor([0., 20., 0.]), atol=1e-4, rtol=1e-4)
        self.assertEqual(int(torch.count_nonzero(builder.env.relative_position)), 0)
        self.assertTrue(all(bool(values[2]) for values in checks.values()))

    def test_missing_final_contact_fails_despite_sufficient_contact_fraction(self):
        def release(env):
            env.contacts[0] = False

        builder = self.make_builder([self.unchanged] * 4 + [release], count=2)
        checks, evidence = self.hold(builder)
        torch.testing.assert_close(evidence["near_goal_contact_fraction"], torch.tensor([.8, 1.]))
        self.assertEqual(checks["finger_contact"].tolist(), [True, True])
        self.assertEqual(checks["final_finger_contact"].tolist(), [False, True])

    def test_already_successful_height_is_rejected_even_with_grip_and_loose_range(self):
        def too_high(env):
            env._cube_root_states[0, 2] = .81  # 21 cm above original support.

        builder = self.make_builder([too_high] * 5, count=2)
        # Isolate the defensive success check from the normal stricter YAML
        # validation, which would already reject this configuration range.
        builder.reset_config["near_goal"]["object_lift_height_range_m"] = [.05, .25]
        checks, _ = self.hold(builder)
        self.assertTrue(bool(checks["lift_height"].all()))
        self.assertEqual(checks["not_task_success"].tolist(), [False, True])

    def test_feet_need_final_ground_proximity_without_standing_or_load_requirement(self):
        def moving_robot(env):
            env._robot_root_states[:, 7:] = 5.
            # Feet carry no load. One state has a foot outside the geometry
            # limit; the other is moving but all foot bottoms remain nearby.
            env._rigid_body_pos[0, 0, 2] = FOOT_RADIUS + .03

        builder = self.make_builder([moving_robot] * 5, count=2)
        checks, _ = self.hold(builder)
        self.assertEqual(checks["feet_near_ground"].tolist(), [False, True])
        self.assertEqual(int(torch.count_nonzero(builder.env._contact_forces[:, :4])), 0)
        self.assertTrue(all(bool(values[1]) for values in checks.values()))

    def test_table_support_or_leaving_grip_region_cannot_be_hidden_by_final_recovery(self):
        def invalid(env):
            env.clear_of_table[0] = False
            env.between[1] = False

        def recovered(env):
            env.clear_of_table.fill_(True)
            env.between.fill_(True)

        builder = self.make_builder([invalid, recovered] + [self.unchanged] * 3)
        checks, _ = self.hold(builder)
        self.assertEqual(checks["object_clear_of_table"].tolist(), [False, True, True])
        self.assertEqual(checks["held_region"].tolist(), [True, False, True])
        self.assertTrue(all(bool(values[2]) for values in checks.values()))


class GraspContactBodyIndicesTests(unittest.TestCase):
    def test_collapsed_fixed_finger_uses_link06(self):
        env = SimpleNamespace(body_names_to_idx={"link06": 17, "gripperMover": 18,
                                                 "ee_gripper_link": 19})
        self.assertEqual(grasp_contact_body_indices(env), [17, 18])

    def test_uncollapsed_fixed_finger_uses_stator_instead_of_parent_wrist(self):
        env = SimpleNamespace(body_names_to_idx={"link06": 17, "gripperStator": 18,
                                                 "gripperMover": 20, "ee_gripper_link": 19})
        self.assertEqual(grasp_contact_body_indices(env), [18, 20])


if __name__ == "__main__":
    unittest.main()
