"""Check configurable reset geometry on CPU/CUDA without starting Isaac Gym."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import torch

from utils.sgs_bank_builder import PhysicsBankBuilder, pregrasp_local_offset
from utils.sgs_build_config import load_build_config


class BuilderRangeTests(unittest.TestCase):
    device = "cpu"

    def builder(self, count=1):
        base = {
            "x_range_m": [0.68, 0.90],
            "y_range_m": [-0.015, 0.015],
            "heading_offset_range_deg": [-2.0, 2.0],
        }
        config = {
            "schema_version": 3,
            "object": {"yaw_offset_range_deg": [-15, 15]},
            "validation": {"physics_steps": 5, "max_contact_force_n": 5000.,
                           "max_object_displacement_m": .02},
            "full_task": {"base": deepcopy(base), "arm_pose": "default"},
            "pregrasp": {"base": deepcopy(base),
                         "ee_approach_distance_range_m": [0.05, 0.10],
                         "ee_lateral_offset_range_m": [-.01, .01],
                         "ee_vertical_offset_range_m": [-.01, .01],
                         "ee_rotation_offset_range_deg": {
                             "roll": [-5, 5], "pitch": [-5, 5], "yaw": [-5, 5]}},
            "at_grasp": {"base": deepcopy(base),
                         "gripper_opening_range": [0.0, 1.0]},
        }
        urdf = Path(__file__).resolve().parents[1] / "data/asset/b1z1-col/urdf/b1z1.urdf"
        limits = ET.parse(urdf).getroot().find("joint[@name='z1_jointGripper']/limit")
        env = SimpleNamespace(num_envs=count, device=self.device,
                              lifted_success_threshold=0.2,
                              dof_limits_lower=torch.tensor([float(limits.attrib["lower"])],
                                                            device=self.device),
                              dof_limits_upper=torch.tensor([float(limits.attrib["upper"])],
                                                            device=self.device))
        return PhysicsBankBuilder(env, per_family=1, reset_config=config)

    def state(self, offsets, yaws):
        """World object-minus-base XY and absolute base yaw, in degrees."""
        count = len(offsets)
        obj = torch.zeros(count, 13, device=self.device)
        obj[:, :3] = torch.tensor([5.0, -3.0, 0.6], device=self.device)
        obj[:, 6] = 1
        root = obj.clone()
        root[:, :2] -= torch.tensor(offsets, device=self.device)
        angles = torch.deg2rad(torch.tensor(yaws, device=self.device)) / 2
        root[:, 5] = torch.sin(angles)
        root[:, 6] = torch.cos(angles)
        return {"robot_root": root, "object_root": obj}

    def test_sample_range_uses_requested_device_shape_and_signed_bounds(self):
        builder = self.builder(257)
        values = builder._sample_range([-0.4, -0.1])
        self.assertEqual(values.shape, (257,))
        self.assertEqual(values.device.type, torch.device(self.device).type)
        self.assertTrue(bool(((values >= -0.4) & (values <= -0.1)).all()))
        self.assertGreater(values.unique().numel(), 1)

    def test_fixed_sampling_keeps_the_fixed_target(self):
        values = self.builder(19)._sample_range([0.73, 0.73])
        self.assertTrue(torch.equal(values, torch.full_like(values, 0.73)))

    def test_pregrasp_disturbances_preserve_euclidean_distance_and_approach_side(self):
        distance = torch.tensor([.05, .075, .10], device=self.device)
        lateral = torch.tensor([.01, -.01, 0.], device=self.device)
        vertical = torch.tensor([-.01, 0., .01], device=self.device)
        actual = pregrasp_local_offset(distance, lateral, vertical)
        torch.testing.assert_close(actual.norm(dim=1), distance)
        torch.testing.assert_close(actual[:, 1], lateral)
        torch.testing.assert_close(actual[:, 2], vertical)
        self.assertTrue(bool((actual[:, 0] < 0).all()))
        self.assertEqual(actual.device, distance.device)

    def test_pregrasp_impossible_transverse_offset_is_rejected(self):
        distance = torch.tensor([.05], device=self.device)
        for lateral in (.05, .06):
            with self.assertRaisesRegex(ValueError, "must fit"):
                pregrasp_local_offset(distance, torch.full_like(distance, lateral),
                                      torch.zeros_like(distance))

    def test_opening_sampling_accepts_closed_open_and_partial_ranges(self):
        builder = self.builder(257)
        for bounds in ([0., 0.], [1., 1.], [.2, .8]):
            with self.subTest(bounds=bounds):
                values = builder._sample_range(bounds)
                self.assertTrue(bool(((values >= bounds[0]) & (values <= bounds[1])).all()))
                if bounds[0] == bounds[1]:
                    self.assertTrue(torch.equal(values, torch.full_like(values, bounds[0])))
                else:
                    self.assertGreater(values.unique().numel(), 1)

    def test_opening_to_joint_uses_b1_closed_and_open_urdf_limits(self):
        opening = torch.tensor([0., .25, .5, .75, 1.], device=self.device)
        actual = self.builder(5)._opening_to_joint(opening)
        expected = torch.tensor([0., -math.pi / 8, -math.pi / 4,
                                 -3 * math.pi / 8, -math.pi / 2], device=self.device)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.device, opening.device)

    def test_joint_to_opening_recovers_b1_closed_partial_and_open_values(self):
        joints = torch.tensor([0., -math.pi / 8, -math.pi / 4,
                               -3 * math.pi / 8, -math.pi / 2], device=self.device)
        actual = self.builder(5)._joint_to_opening(joints)
        expected = torch.tensor([0., .25, .5, .75, 1.], device=self.device)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.device, joints.device)

    def test_finite_width_intervals_have_no_five_millimeter_padding(self):
        values = torch.tensor([0.049, 0.05, 0.075, 0.10, 0.101], device=self.device)
        accepted = self.builder()._in_range(values, [0.05, 0.10])
        self.assertEqual(accepted.tolist(), [False, True, True, True, False])

    def test_fixed_distance_allows_only_documented_physical_tolerance(self):
        values = torch.tensor([1.994, 1.996, 2.0, 2.004, 2.006], device=self.device)
        self.assertEqual(self.builder()._in_range(values, [2.0, 2.0]).tolist(),
                         [False, True, True, True, False])

    def test_signed_world_xy_is_not_radial_distance_or_robot_frame(self):
        state = self.state([(0.8, 0.3), (0.8, -0.3), (-0.8, 0.3), (0.7, 0.3)],
                           [45.0] * 4)
        base = {"x_range_m": [0.79, 0.81], "y_range_m": [0.29, 0.31],
                "heading_offset_range_deg": [-180.0, 180.0]}
        self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                         [True, False, False, False])

    def test_heading_is_relative_to_object_bearing_not_world_x(self):
        state = self.state([(1.0, 1.0)] * 3, [45.0, 0.0, 55.0])
        base = {"x_range_m": [1.0, 1.0], "y_range_m": [1.0, 1.0],
                "heading_offset_range_deg": [-0.5, 0.5]}
        self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                         [True, False, False])

    def test_positive_heading_offset_means_counterclockwise(self):
        state = self.state([(1.0, -1.0)] * 3, [-35.0, -55.0, -45.0])
        base = {"x_range_m": [1.0, 1.0], "y_range_m": [-1.0, -1.0],
                "heading_offset_range_deg": [9.0, 11.0]}
        self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                         [True, False, False])

    def test_heading_wraps_across_absolute_yaw_branch_cut(self):
        angle = math.radians(179.0)
        offset = (math.cos(angle), math.sin(angle))
        state = self.state([offset, offset], [-179.0, 177.0])
        base = {"x_range_m": [-1.01, -0.99], "y_range_m": [0.01, 0.03],
                "heading_offset_range_deg": [1.5, 2.5]}
        self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                         [True, False])

    def test_fixed_opposite_heading_has_circular_one_degree_tolerance(self):
        state = self.state([(1.0, 0.0)] * 4, [180.0, -179.5, 179.5, 178.5])
        for target in (-180.0, 180.0):
            with self.subTest(target=target):
                base = {"x_range_m": [1.0, 1.0], "y_range_m": [0.0, 0.0],
                        "heading_offset_range_deg": [target, target]}
                self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                                 [True, True, True, False])

    def test_upper_180_boundary_accepts_equivalent_minus_180(self):
        state = self.state([(1.0, 0.0)] * 3, [-180.0, 179.5, -179.5])
        base = {"x_range_m": [1.0, 1.0], "y_range_m": [0.0, 0.0],
                "heading_offset_range_deg": [170.0, 180.0]}
        self.assertEqual(self.builder()._base_in_range(state, base).tolist(),
                         [True, True, False])

    def test_geometry_checks_do_not_mutate_saved_state_or_config(self):
        state = self.state([(1.0, 1.0), (1.0, -1.0)], [45.0, -45.0])
        before = {key: value.clone() for key, value in state.items()}
        base = {"x_range_m": [0.9, 1.1], "y_range_m": [-1.1, 1.1],
                "heading_offset_range_deg": [-2.0, 2.0]}
        base_before = deepcopy(base)
        result = self.builder()._base_in_range(state, base)
        self.assertEqual(result.tolist(), [True, True])
        self.assertEqual(result.dtype, torch.bool)
        for key in before:
            self.assertTrue(torch.equal(before[key], state[key]))
        self.assertEqual(base, base_before)

    def test_schema_five_constructs_only_three_families_and_keeps_grasp_mesh(self):
        builder = self.builder()
        env = builder.env
        asset_root = Path(__file__).resolve().parents[1] / "data/asset"
        env.cfg = {"env": {"asset": {"assetRoot": str(asset_root),
                                    "assetFileObj": "obj_set",
                                    "asset_multi": {"sugar_box": {}}}}}
        config = load_build_config()
        rebuilt = PhysicsBankBuilder(env, per_family=2, reset_config=config)
        self.assertEqual(rebuilt.family_names, ("full_task", "pregrasp", "near_goal"))
        self.assertEqual(rebuilt.counts, [0, 0, 0])
        self.assertEqual(len(rebuilt.parts), 3)
        self.assertNotIn("at_grasp", rebuilt.reset_config)
        self.assertGreater(rebuilt.object_vertices.shape[0], 0)
        self.assertEqual(rebuilt.object_vertices.device.type, torch.device(self.device).type)

    def test_schema_five_build_routes_near_goal_and_assigns_contiguous_family_ids(self):
        builder = self.builder(2)
        env = builder.env
        config = load_build_config()
        builder.reset_config = config
        builder.family_names = ("full_task", "pregrasp", "near_goal")
        builder.per_family = 2
        env.obj_list = ["sugar_box"]
        env.init_height = torch.tensor([.045], device=self.device)
        env.num_dofs = 19
        env._rigid_body_pos = torch.zeros(2, 27, 3, device=self.device)
        env.hold_steps = 15
        env.cfg = {"sim": {"dt": .005}, "env": {"asset": {"assetFileRobot": "b1z1.urdf"}}}
        state = {"grasp_candidate_index": torch.tensor([-1, -1], device=self.device)}
        checks = {"valid": torch.ones(2, dtype=torch.bool, device=self.device)}
        builder._full = Mock(return_value=(state, checks))
        builder._grasp_pose = Mock(return_value=(state, checks))
        builder._near_goal = Mock(return_value=(state, checks))
        builder.sampling_audit = Mock()
        builder.sampling_audit.summary.return_value = {}
        with patch("utils.sgs_bank_builder.foot_indices", return_value=[3, 7, 11, 15]), \
                patch("utils.sgs.ConfigurationBank", side_effect=lambda tensors, metadata: (tensors, metadata)):
            tensors, metadata = builder.build(max_batches=1)
        builder._grasp_pose.assert_called_once_with(state, "pregrasp")
        builder._near_goal.assert_called_once_with(state)
        self.assertEqual(tensors["family_id"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(metadata["family_names"], ["full_task", "pregrasp", "near_goal"])
        self.assertEqual(metadata["version"], "b1z1-sugar-sgs-v10")
        validation = metadata["physical_validation"]
        self.assertEqual(validation["validation_profile"], "geometry_grasp_v3")
        self.assertEqual(validation["accepted_per_family"], [2, 2, 2])
        self.assertTrue(validation["near_goal_grasp_verified"])
        self.assertTrue(validation["scripted_grasp_required"])
        self.assertIn("cold-reset grip verification", validation["validation_by_family"]["near_goal"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class BuilderRangeCudaTests(BuilderRangeTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
