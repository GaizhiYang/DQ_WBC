"""Check SGS heading acceptance against commanded Level02 object motion."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from utils.sgs_bank_builder import PhysicsBankBuilder, level02_object_motion_xy


class MotionHeadingTests(unittest.TestCase):
    device = "cpu"

    def state(self, robot_yaws, table_yaws=None, table_vy=None,
              rotation_enabled=None, table_yaw_rate=None):
        count = len(robot_yaws)

        def values(items, default, dtype=torch.float32):
            return torch.tensor(items if items is not None else [default] * count,
                                dtype=dtype, device=self.device)

        def roots(yaws):
            result = torch.zeros(count, 13, device=self.device)
            half_angles = torch.deg2rad(values(yaws, 0.0)) / 2
            result[:, 5], result[:, 6] = half_angles.sin(), half_angles.cos()
            return result

        return {
            "robot_root": roots(robot_yaws),
            "table_root": roots(table_yaws),
            "object_root": roots(None),
            "table_vy": values(table_vy, 1.0),
            "rotation_enabled": values(rotation_enabled, False, torch.bool),
            "table_yaw_rate": values(table_yaw_rate, 0.0),
        }

    def aligned(self, state):
        builder = object.__new__(PhysicsBankBuilder)
        builder.env = SimpleNamespace(device=self.device,
                                      num_envs=len(state["robot_root"]))
        return builder._base_motion_aligned(state)

    def assert_motion(self, state, expected):
        actual = level02_object_motion_xy(state)
        expected = torch.tensor(expected, device=self.device, dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.device, state["table_root"].device)
        self.assertEqual(actual.dtype, state["table_root"].dtype)

    def test_translation_preserves_forward_and_reverse_motion(self):
        state = self.state([0.0, 0.0], table_vy=[0.2, -0.3])
        self.assert_motion(state, [[0.0, 0.2], [0.0, -0.3]])

    def test_only_acute_headings_are_accepted(self):
        state = self.state([90.0, 45.0, 135.0, -90.0, 0.0, 180.0])
        self.assertEqual(self.aligned(state).tolist(),
                         [True, True, True, False, False, False])

    def test_reversing_motion_reverses_heading_acceptance(self):
        state = self.state([90.0, -90.0, 90.0, -90.0],
                           table_vy=[1.0, 1.0, -1.0, -1.0])
        self.assertEqual(self.aligned(state).tolist(), [True, False, False, True])

    def test_signed_ninety_and_yaw_branch_cut_remain_geometric(self):
        # A table yaw of +90 degrees makes its local +y motion point world -x.
        state = self.state([179.0, -179.0, 180.0, -180.0, 90.0, -90.0, 0.0],
                           table_yaws=[90.0] * 7)
        self.assertEqual(self.aligned(state).tolist(),
                         [True, True, True, True, False, False, False])

    def test_headings_just_inside_and_outside_the_right_angle(self):
        state = self.state([0.01, -0.01, 179.99, 180.01])
        self.assertEqual(self.aligned(state).tolist(), [True, False, True, False])

    def test_table_local_translation_is_rotated_into_world(self):
        state = self.state([90.0, 180.0, -90.0, 0.0],
                           table_yaws=[0.0, 90.0, 180.0, -90.0],
                           table_vy=[2.0] * 4)
        self.assert_motion(state, [[0.0, 2.0], [-2.0, 0.0],
                                   [0.0, -2.0], [2.0, 0.0]])
        self.assertEqual(self.aligned(state).tolist(), [True] * 4)

    def test_rotation_uses_object_offset_from_the_table_center(self):
        state = self.state([math.degrees(math.atan2(4.0, -6.0))],
                           table_vy=[0.0], rotation_enabled=[True],
                           table_yaw_rate=[2.0])
        state["table_root"][0, :2] = torch.tensor([10.0, -7.0], device=self.device)
        state["object_root"][0, :2] = torch.tensor([12.0, -4.0], device=self.device)
        self.assert_motion(state, [[-6.0, 4.0]])
        self.assertEqual(self.aligned(state).tolist(), [True])

    def test_rotation_and_translation_are_combined_before_checking_heading(self):
        state = self.state([0.0, 180.0], table_yaws=[90.0, 90.0],
                           rotation_enabled=[True, True], table_yaw_rate=[0.5, 0.5])
        state["object_root"][:, :2] = torch.tensor([1.0, 2.0], device=self.device)
        self.assert_motion(state, [[-2.0, 0.5], [-2.0, 0.5]])
        self.assertEqual(self.aligned(state).tolist(), [False, True])

    def test_disabled_rotation_does_not_change_reference_motion(self):
        state = self.state([90.0, 90.0], rotation_enabled=[False, True],
                           table_yaw_rate=[-2.0, -2.0])
        state["object_root"][:, 0] = 1.0
        self.assert_motion(state, [[0.0, 1.0], [0.0, -1.0]])
        self.assertEqual(self.aligned(state).tolist(), [True, False])

    def test_measured_object_velocity_does_not_override_commanded_motion(self):
        state = self.state([90.0, 90.0, 90.0])
        state["object_root"][:, 7:10] = torch.tensor(
            [[0.0, 0.0, 0.0], [1e-10, -1e-10, 0.0], [0.0, -20.0, 0.0]],
            device=self.device)
        self.assert_motion(state, [[0.0, 1.0]] * 3)
        self.assertEqual(self.aligned(state).tolist(), [True] * 3)

    def test_stationary_and_exactly_cancelled_motion_are_rejected(self):
        state = self.state([0.0, 90.0, 180.0], table_vy=[0.0, 0.0, 1.0],
                           rotation_enabled=[False, True, True],
                           table_yaw_rate=[0.0, 1.0, -1.0])
        # Rotation at the center gives no motion; the last row cancels local +y.
        state["object_root"][2, 0] = 1.0
        self.assert_motion(state, [[0.0, 0.0]] * 3)
        self.assertEqual(self.aligned(state).tolist(), [False] * 3)

    def test_motion_and_heading_checks_preserve_the_saved_state(self):
        state = self.state([45.0, -135.0], table_yaws=[30.0, -70.0],
                           table_vy=[0.2, -0.4], rotation_enabled=[True, False],
                           table_yaw_rate=[0.5, -0.3])
        state["table_root"][:, :2] = torch.tensor([2.0, -3.0], device=self.device)
        state["object_root"][:, :2] = torch.tensor([2.5, -2.8], device=self.device)
        before = {name: value.clone() for name, value in state.items()}
        level02_object_motion_xy(state)
        accepted = self.aligned(state)
        self.assertEqual(accepted.dtype, torch.bool)
        self.assertEqual(accepted.shape, (2,))
        self.assertEqual(accepted.device, state["robot_root"].device)
        for name, value in state.items():
            with self.subTest(field=name):
                self.assertTrue(torch.equal(value, before[name]))

    def test_collection_stores_only_aligned_rows_for_each_family(self):
        state = self.state([90.0, 0.0, -90.0, 45.0])
        state["grasp_candidate_index"] = torch.arange(4, device=self.device)
        for family, check_name in enumerate(("base_motion_alignment",
                                            "base_motion_alignment",
                                            "final_base_motion_alignment")):
            with self.subTest(family=family):
                builder = object.__new__(PhysicsBankBuilder)
                builder.per_family = 4
                builder.counts = [0, 0, 0]
                builder.parts = [[], [], []]
                builder.sampling_audit = Mock()
                checks = {"geometry": torch.ones(4, dtype=torch.bool, device=self.device),
                          check_name: self.aligned(state)}
                self.assertEqual(builder._collect(family, state, checks), 2)
                self.assertEqual(builder.counts[family], 2)
                saved = builder.parts[family][0]
                self.assertEqual(saved["grasp_candidate_index"].tolist(), [0, 3])
                self.assertEqual(saved["family_id"].tolist(), [family, family])
                self.assertEqual(builder.sampling_audit.record.call_args.args[3].tolist(),
                                 [0, 3])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class MotionHeadingCudaTests(MotionHeadingTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
