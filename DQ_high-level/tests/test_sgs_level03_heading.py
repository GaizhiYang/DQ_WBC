"""Level03 reset headings follow the saved departure intention, from rest."""

import unittest
from unittest.mock import Mock

import torch

from utils.sgs_build_audit import BuildAudit
from utils.sgs_motion import (
    base_motion_angle_deg, level02_object_motion_xy, object_motion_xy,
)


def roots(yaws):
    angles = torch.deg2rad(torch.tensor(yaws, dtype=torch.float64)) / 2
    result = torch.zeros(len(yaws), 13, dtype=torch.float64)
    result[:, 5], result[:, 6] = angles.sin(), angles.cos()
    return result


class Level03HeadingTests(unittest.TestCase):
    def state(self, table_yaws, directions, robot_yaws=None):
        count = len(table_yaws)
        return {
            "table_root": roots(table_yaws),
            "object_root": roots([0.] * count),
            "robot_root": roots(robot_yaws or [0.] * count),
            "level03_driver_direction": torch.tensor(directions, dtype=torch.float64),
        }

    def test_signed_local_y_intention_rotates_to_world(self):
        state = self.state([0., 0., 90., 90., -90., 180.], [1., -1., 1., -1., 1., 1.])
        expected = torch.tensor([[0., 1.], [0., -1.], [-1., 0.], [1., 0.],
                                 [1., 0.], [0., -1.]], dtype=torch.float64)
        torch.testing.assert_close(object_motion_xy(state), expected, atol=1e-12, rtol=0.)
        self.assertEqual(object_motion_xy(state).dtype, state["table_root"].dtype)

    def test_reference_ignores_stationary_roots_and_legacy_motion_parameters(self):
        state = self.state([0., 0.], [1., -1.], [90., -90.])
        state.update(table_vy=torch.tensor([-9., 9.]),
                     rotation_enabled=torch.tensor([True, True]),
                     table_yaw_rate=torch.tensor([20., -20.]))
        state["object_root"][:, 0] = 2.
        before = {key: value.clone() for key, value in state.items()}
        torch.testing.assert_close(base_motion_angle_deg(state), torch.zeros(2, dtype=torch.float64),
                                   atol=1e-12, rtol=0.)
        for key, value in state.items():
            torch.testing.assert_close(value, before[key], atol=0., rtol=0.)

    def test_xy_departure_uses_both_saved_directions(self):
        state = self.state([0., 0., 0., 90.], [1., 1., -1., 1.],
                           [45., 135., -45., 135.])
        state["level03_x_driver_direction"] = torch.tensor([1., -1., 1., 1.])
        expected = torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., 1.]],
                                dtype=torch.float64) / 2 ** .5
        torch.testing.assert_close(object_motion_xy(state), expected)
        torch.testing.assert_close(base_motion_angle_deg(state),
                                   torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0.)

    def test_xy_departure_accounts_for_initial_pedal_ratio(self):
        state = self.state([0.], [1.])
        state["level03_x_driver_direction"] = torch.tensor([-1.])
        state["level03_driver_throttle_goal"] = torch.tensor([.6], dtype=torch.float64)
        state["level03_x_driver_throttle_goal"] = torch.tensor([.8], dtype=torch.float64)
        torch.testing.assert_close(object_motion_xy(state),
                                   torch.tensor([[-.8, .6]], dtype=torch.float64))

    def test_reversing_saved_intention_changes_heading_without_velocity_jump(self):
        state = self.state([0., 0., 90., 90.], [1., -1., 1., -1.], [60., 60., 150., 150.])
        torch.testing.assert_close(base_motion_angle_deg(state),
                                   torch.tensor([30., 150., 30., 150.], dtype=torch.float64))
        self.assertTrue((state["table_root"][:, 7:] == 0).all())

    def test_absent_direction_is_invalid_for_heading(self):
        state = self.state([0.], [0.], [90.])
        self.assertTrue(torch.isnan(base_motion_angle_deg(state)).all())

    def test_level02_keeps_commanded_translation_and_yaw_tangent(self):
        state = self.state([90., 0.], [1., 1.])
        del state["level03_driver_direction"]
        state.update(table_vy=torch.tensor([.2, -.3]),
                     rotation_enabled=torch.tensor([True, False]),
                     table_yaw_rate=torch.tensor([.5, 9.]))
        state["object_root"][:, :2] = torch.tensor([1., 2.])
        torch.testing.assert_close(object_motion_xy(state), level02_object_motion_xy(state))
        torch.testing.assert_close(object_motion_xy(state),
                                   torch.tensor([[-1.2, .5], [0., -.3]], dtype=torch.float64))

    def test_audit_reports_departure_heading_without_legacy_fields(self):
        state = self.state([0., 90.], [1., -1.], [60., -30.])
        count = 2
        state.update(
            family_id=torch.zeros(count, dtype=torch.long),
            table_height=torch.full((count,), .55),
            gripper_opening=torch.ones(count), object_yaw_offset_deg=torch.zeros(count),
            validation_peak_contact_force_n=torch.zeros(count),
            validation_max_object_displacement_m=torch.zeros(count),
            candidate_local_offset=torch.zeros(count, 3), ee_rotation_offset_deg=torch.zeros(count, 3),
        )
        coverage = BuildAudit(("full_task",)).summary(state)["families"]["full_task"]["coverage"]
        angles = coverage["base_motion_angle_deg"]
        self.assertAlmostEqual(angles["min"], 30.)
        self.assertAlmostEqual(angles["max"], 30.)
        self.assertEqual(angles["nonfinite_count"], 0)

        state["level03_x_driver_direction"] = torch.tensor([1., -1.])
        state["robot_root"] = roots([45., -45.])
        coverage = BuildAudit(("full_task",)).summary(state)["families"]["full_task"]["coverage"]
        self.assertAlmostEqual(coverage["base_motion_angle_deg"]["max"], 0.)

    def test_collection_rejects_table_contact_drift_without_editing_snapshot(self):
        from utils.sgs_bank_builder import PhysicsBankBuilder
        state = self.state([0.] * 4, [1.] * 4)
        state["grasp_candidate_index"] = torch.arange(4)
        state["table_root"][1, 7] = 2e-5
        state["table_root"][2, 12] = 2e-5
        state["table_root"][3, 9] = -.05  # Vertical support drift is separate.
        before = state["table_root"].clone()
        builder = object.__new__(PhysicsBankBuilder)
        builder.per_family, builder.counts, builder.parts = 4, [0], [[]]
        builder.sampling_audit = Mock()
        self.assertEqual(builder._collect(0, state, {"geometry": torch.ones(4, dtype=torch.bool)}), 2)
        self.assertEqual(builder.parts[0][0]["grasp_candidate_index"].tolist(), [0, 3])
        torch.testing.assert_close(state["table_root"], before, atol=0., rtol=0.)
        checks = builder.sampling_audit.record.call_args.args[2]
        self.assertEqual(checks["level03_stationary_support"].tolist(), [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
