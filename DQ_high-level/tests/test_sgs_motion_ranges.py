"""Verify configurable unsigned angles to Level02 motion on CPU and CUDA."""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.sgs_bank_builder import PhysicsBankBuilder
from utils.sgs_build_config import load_build_config, validate_build_config
from utils.sgs_motion import base_motion_angle_deg, level02_object_motion_xy


class MotionRangeConfigTests(unittest.TestCase):
    def config(self):
        config = load_build_config()
        config["heading_reference"] = "object_motion"
        for family in ("full_task", "pregrasp", "near_goal"):
            config[family]["base"]["heading_offset_range_deg"] = [45.0, 75.0]
        return config

    def test_motion_normalization_is_idempotent_and_does_not_mutate_input(self):
        config = self.config()
        before = deepcopy(config)
        normalized = validate_build_config(config)
        self.assertEqual(validate_build_config(normalized), normalized)
        self.assertEqual(normalized["heading_reference"], "object_motion")
        normalized["pregrasp"]["base"]["heading_offset_range_deg"][0] = 10.0
        self.assertEqual(config, before)

    def test_motion_accepts_fixed_angles_and_ranges_ending_at_ninety(self):
        for family in ("full_task", "pregrasp", "near_goal"):
            for bounds in ([0.0, 0.0], [45.0, 45.0], [45.0, 75.0],
                           [75.0, 90.0], [0.0, 90.0], [89.99, 90.0]):
                with self.subTest(family=family, bounds=bounds):
                    config = self.config()
                    config[family]["base"]["heading_offset_range_deg"] = bounds
                    normalized = validate_build_config(config)
                    self.assertEqual(normalized[family]["base"]["heading_offset_range_deg"], bounds)

    def test_motion_rejects_signed_obtuse_and_only_perpendicular_ranges(self):
        for family in ("full_task", "pregrasp", "near_goal"):
            for bounds in ([-1.0, 45.0], [45.0, 90.01], [90.0, 90.0],
                           [91.0, 91.0], [75.0, 45.0]):
                with self.subTest(family=family, bounds=bounds):
                    config = self.config()
                    config[family]["base"]["heading_offset_range_deg"] = bounds
                    with self.assertRaisesRegex(ValueError, "heading"):
                        validate_build_config(config)

    def test_unknown_heading_reference_is_rejected(self):
        for reference in ("world_x", "object_velocity", "", None, True, 1):
            with self.subTest(reference=reference):
                config = self.config()
                config["heading_reference"] = reference
                with self.assertRaisesRegex(ValueError, "heading_reference"):
                    validate_build_config(config)

    def test_omitted_or_explicit_legacy_reference_retains_signed_domain(self):
        config = self.config()
        config.pop("heading_reference")
        config["full_task"]["base"]["heading_offset_range_deg"] = [-180.0, 180.0]
        self.assertEqual(validate_build_config(config), config)
        config["heading_reference"] = "object_bearing"
        self.assertEqual(validate_build_config(config), config)


class MotionRangeTests(unittest.TestCase):
    device = "cpu"

    def tensor(self, values, dtype=torch.float32):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def state(self, robot_yaws, table_yaws=None, table_vy=None,
              rotation_enabled=None, table_yaw_rate=None):
        count = len(robot_yaws)

        def roots(yaws):
            root = torch.zeros(count, 13, device=self.device)
            angles = self.tensor([0.0] * count if yaws is None else yaws) * math.pi / 360
            root[:, 5], root[:, 6] = angles.sin(), angles.cos()
            return root

        state = {"robot_root": roots(robot_yaws), "table_root": roots(table_yaws),
                 "object_root": roots(None),
                 "table_vy": self.tensor([1.0] * count if table_vy is None else table_vy),
                 "rotation_enabled": self.tensor(
                     [False] * count if rotation_enabled is None else rotation_enabled, torch.bool),
                 "table_yaw_rate": self.tensor(
                     [0.0] * count if table_yaw_rate is None else table_yaw_rate)}
        state["robot_root"][:, 0] = -1.0
        return state

    def builder(self, state, reference="object_motion"):
        builder = object.__new__(PhysicsBankBuilder)
        builder.reset_config = {} if reference is None else {"heading_reference": reference}
        builder.env = SimpleNamespace(
            device=self.device, num_envs=len(state["robot_root"]),
            _cube_root_states=state["object_root"], _table_root_states=state["table_root"],
            vel_line_forward_y_level02=state["table_vy"],
            lin_circle_select=state["rotation_enabled"], vel_angular_yaw=state["table_yaw_rate"])
        return builder

    @staticmethod
    def base(bounds=(45.0, 75.0)):
        return {"x_range_m": [0.99, 1.01], "y_range_m": [-0.01, 0.01],
                "heading_offset_range_deg": list(bounds)}

    @staticmethod
    def write_yaw(state, yaw):
        state["robot_root"][:, 3:7] = 0.0
        state["robot_root"][:, 5] = (yaw / 2).sin()
        state["robot_root"][:, 6] = (yaw / 2).cos()

    def sample_fixed(self, state, angle, dx=None, dy=None):
        builder = self.builder(state)
        count = len(state["robot_root"])
        dx = self.tensor([1.0] * count if dx is None else dx)
        dy = self.tensor([0.0] * count if dy is None else dy)
        with patch.object(builder, "_sample_range", return_value=self.tensor([angle] * count)):
            return builder._sample_base_yaw(self.base((angle, angle)), dx, dy)

    def test_direct_sampling_stays_in_requested_motion_angle_and_passes_acceptance(self):
        state = self.state([0.0] * 1024)
        builder = self.builder(state)
        yaw = builder._sample_base_yaw(self.base(), self.tensor([1.0] * 1024),
                                       self.tensor([0.0] * 1024))
        self.write_yaw(state, yaw)
        angle = base_motion_angle_deg(state)
        self.assertTrue(bool(((angle >= 45.0 - 1e-4) & (angle <= 75.0 + 1e-4)).all()))
        self.assertGreater(angle.unique().numel(), 900)
        self.assertTrue(bool(builder._base_in_range(state, self.base()).all()))
        self.assertTrue(bool(builder._base_motion_aligned(state).all()))
        self.assertEqual(yaw.shape, (1024,))
        self.assertEqual(yaw.device, state["robot_root"].device)

    def test_sampling_uses_motion_reference_and_reverses_with_signed_velocity(self):
        state = self.state([0.0, 0.0], table_vy=[1.0, -1.0])
        yaw = self.sample_fixed(state, 60.0)
        torch.testing.assert_close(torch.rad2deg(yaw), self.tensor([30.0, -30.0]))
        self.write_yaw(state, yaw)
        torch.testing.assert_close(base_motion_angle_deg(state), self.tensor([60.0, 60.0]))

    def test_sampling_selects_the_side_closer_to_the_object_bearing(self):
        state = self.state([0.0, 0.0])
        yaw = self.sample_fixed(state, 60.0, dx=[1.0, -1.0])
        torch.testing.assert_close(torch.rad2deg(yaw), self.tensor([30.0, 150.0]))

    def test_sampling_rotates_table_local_motion_into_world(self):
        state = self.state([0.0], table_yaws=[30.0])
        yaw = self.sample_fixed(state, 60.0)
        torch.testing.assert_close(torch.rad2deg(yaw), self.tensor([60.0]))
        self.write_yaw(state, yaw)
        torch.testing.assert_close(base_motion_angle_deg(state), self.tensor([60.0]))

    def test_sampling_includes_rotation_at_the_object_offset(self):
        state = self.state([0.0], rotation_enabled=[True], table_yaw_rate=[0.5])
        state["object_root"][0, :2] = self.tensor([1.0, 2.0])
        torch.testing.assert_close(level02_object_motion_xy(state), self.tensor([[-1.0, 1.5]]))
        yaw = self.sample_fixed(state, 60.0)
        expected = math.degrees(math.atan2(1.5, -1.0)) - 60.0
        torch.testing.assert_close(torch.rad2deg(yaw), self.tensor([expected]))
        self.write_yaw(state, yaw)
        torch.testing.assert_close(base_motion_angle_deg(state), self.tensor([60.0]))

    def test_acceptance_measures_motion_angle_instead_of_bearing_offset(self):
        state = self.state([30.0, -30.0, 60.0])
        motion = self.builder(state)._base_in_range(state, self.base())
        legacy = self.builder(state, reference=None)._base_in_range(state, self.base())
        self.assertEqual(motion.tolist(), [True, False, False])
        self.assertEqual(legacy.tolist(), [False, False, True])

    def test_legacy_sampler_keeps_signed_bearing_offsets(self):
        state = self.state([0.0, 0.0])
        for reference in (None, "object_bearing"):
            with self.subTest(reference=reference):
                builder = self.builder(state, reference=reference)
                with patch.object(builder, "_sample_range", return_value=self.tensor([-10.0, 15.0])):
                    yaw = builder._sample_base_yaw(self.base((-10.0, 15.0)),
                                                   self.tensor([1.0, 0.0]), self.tensor([0.0, 1.0]))
                torch.testing.assert_close(torch.rad2deg(yaw), self.tensor([-10.0, 105.0]))

    def test_unsigned_angle_and_side_choice_wrap_across_plus_minus_180(self):
        state = self.state([135.0, -135.0], table_yaws=[90.0, 90.0])
        torch.testing.assert_close(base_motion_angle_deg(state), self.tensor([45.0, 45.0]))
        self.assertEqual(self.builder(state)._base_in_range(state, self.base((45.0, 45.0))).tolist(),
                         [True, True])
        bearing = self.tensor([-179.0, 179.0]) * math.pi / 180
        yaw = self.sample_fixed(state, 45.0, dx=bearing.cos().tolist(), dy=bearing.sin().tolist())
        torch.testing.assert_close(yaw.cos(), self.tensor([-math.sqrt(0.5)] * 2))
        torch.testing.assert_close(yaw.sin(), self.tensor([-math.sqrt(0.5), math.sqrt(0.5)]))

    def test_fixed_motion_angle_keeps_one_degree_physical_tolerance(self):
        state = self.state([44.5, 45.5, 43.9, 46.1])
        self.assertEqual(self.builder(state)._base_in_range(state, self.base((45.0, 45.0))).tolist(),
                         [True, True, False, False])

    def test_finite_range_has_no_extra_physical_padding(self):
        state = self.state([45.0, 15.0, 45.01, 14.99])
        self.assertEqual(self.builder(state)._base_in_range(state, self.base()).tolist(),
                         [True, True, False, False])

    def test_range_ending_at_ninety_still_rejects_perpendicular_heading(self):
        state = self.state([15.0, 0.0, 0.01, -0.01, 15.01])
        builder = self.builder(state)
        accepted = builder._base_in_range(state, self.base((75.0, 90.0)))
        self.assertEqual(accepted.tolist(), [True, True, True, False, False])
        self.assertEqual((accepted & builder._base_motion_aligned(state)).tolist(),
                         [True, False, True, False, False])

    def test_zero_or_cancelled_motion_has_no_valid_angle(self):
        state = self.state([30.0, 30.0], table_vy=[0.0, 1.0],
                           rotation_enabled=[False, True], table_yaw_rate=[0.0, -1.0])
        state["object_root"][1, 0] = 1.0
        # Keep both robots at object-minus-base x=1, so only direction can reject.
        state["robot_root"][:, 0] = state["object_root"][:, 0] - 1.0
        builder = self.builder(state)
        self.assertEqual(builder._base_in_range(state, self.base()).tolist(), [False, False])
        self.assertEqual(builder._base_motion_aligned(state).tolist(), [False, False])

    def test_sampling_and_acceptance_do_not_mutate_saved_motion(self):
        state = self.state([30.0, 30.0], table_yaws=[0.0, 30.0], table_vy=[0.2, -0.3])
        before = {name: value.clone() for name, value in state.items()}
        builder = self.builder(state)
        self.sample_fixed(state, 60.0)
        builder._base_in_range(state, self.base())
        for name, expected in before.items():
            with self.subTest(field=name):
                torch.testing.assert_close(state[name], expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class MotionRangeCudaTests(MotionRangeTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
