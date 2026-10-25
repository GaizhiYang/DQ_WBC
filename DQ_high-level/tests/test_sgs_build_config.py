"""Validate user-editable bank settings without importing Isaac Gym or torch."""

import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import yaml


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "sgs_build_config.py"
SPEC = importlib.util.spec_from_file_location("dq_sgs_build_config", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
load_build_config = MODULE.load_build_config
validate_build_config = MODULE.validate_build_config


class BuildConfigTests(unittest.TestCase):
    def setUp(self):
        # Keep validation cases independent of the user's editable YAML values.
        self.config = {
            "schema_version": 3,
            "object": {"yaw_offset_range_deg": [-15., 15.]},
            "validation": {"physics_steps": 5, "max_object_displacement_m": .02,
                           "max_contact_force_n": 5000.},
            "full_task": {"base": {"x_range_m": [2., 2.], "y_range_m": [0., 0.],
                                   "heading_offset_range_deg": [-15., 15.]},
                          "arm_pose": "default"},
            "pregrasp": {"base": {"x_range_m": [.68, .95], "y_range_m": [-.2, .2],
                                  "heading_offset_range_deg": [-15., 15.]},
                         "ee_approach_distance_range_m": [.05, .15],
                         "ee_lateral_offset_range_m": [-.01, .01],
                         "ee_vertical_offset_range_m": [-.01, .01],
                         "ee_rotation_offset_range_deg": {
                             "roll": [-5., 5.], "pitch": [-5., 5.], "yaw": [-5., 5.]}},
            "at_grasp": {"base": {"x_range_m": [.68, .85], "y_range_m": [-.1, .1],
                                  "heading_offset_range_deg": [-10., 10.]},
                         "gripper_opening_range": [.5, 1.]},
        }

    def test_repository_default_config_is_valid_for_teacher(self):
        loaded = load_build_config()
        normalized = validate_build_config(loaded, lifted_threshold=0.2)
        self.assertTrue(MODULE.DEFAULT_BUILD_CONFIG_PATH.is_absolute())
        self.assertEqual(normalized["full_task"]["arm_pose"], "default")
        self.assertEqual(normalized["schema_version"], 4)
        self.assertEqual(normalized, loaded)
        self.assertEqual(set(normalized), {
            "schema_version", "object", "validation", "full_task", "pregrasp", "at_grasp", "near_goal"})
        self.assertEqual(set(normalized["pregrasp"]["ee_rotation_offset_range_deg"]),
                         {"roll", "pitch", "yaw"})

    def test_default_loading_does_not_depend_on_working_directory(self):
        original = Path.cwd()
        expected = load_build_config()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(load_build_config(), expected)
        finally:
            os.chdir(original)

    def test_custom_ranges_signed_offsets_and_full_heading_boundaries(self):
        self.config["full_task"]["base"] = {
            "x_range_m": [-3, -1], "y_range_m": [-2, 2],
            "heading_offset_range_deg": [-180, 180],
        }
        self.config["pregrasp"]["base"] = {
            "x_range_m": [0, 0], "y_range_m": [-.8, -.7],
            "heading_offset_range_deg": [90, 90],
        }
        self.config["pregrasp"]["ee_approach_distance_range_m"] = (.01, .04)
        self.config["pregrasp"]["ee_lateral_offset_range_m"] = [-.002, .004]
        self.config["pregrasp"]["ee_vertical_offset_range_m"] = [-.003, .001]
        self.config["pregrasp"]["ee_rotation_offset_range_deg"] = {
            "roll": [0, 0], "pitch": [-2, 3], "yaw": [7, 8]}
        self.config["object"]["yaw_offset_range_deg"] = [-90, -30]
        self.config["validation"] = {
            "physics_steps": 7, "max_object_displacement_m": .03, "max_contact_force_n": 4000}
        self.config["at_grasp"]["gripper_opening_range"] = [.2, .8]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "自定义配置.yaml"
            path.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
            result = validate_build_config(load_build_config(path), lifted_threshold=.2)
        self.assertEqual(result["full_task"]["base"]["x_range_m"], [-3., -1.])
        self.assertEqual(result["pregrasp"]["ee_approach_distance_range_m"], [.01, .04])
        self.assertEqual(result["at_grasp"]["gripper_opening_range"], [.2, .8])
        self.assertEqual(result["pregrasp"]["ee_lateral_offset_range_m"], [-.002, .004])
        self.assertEqual(result["pregrasp"]["ee_vertical_offset_range_m"], [-.003, .001])
        self.assertEqual(result["pregrasp"]["ee_rotation_offset_range_deg"]["yaw"], [7., 8.])
        self.assertEqual(result["object"]["yaw_offset_range_deg"], [-90., -30.])
        self.assertEqual(result["validation"], self.config["validation"])

    def test_normalization_returns_detached_mutable_lists(self):
        original = copy.deepcopy(self.config)
        result = validate_build_config(self.config)
        result["pregrasp"]["base"]["x_range_m"][0] = 100.
        result["at_grasp"]["gripper_opening_range"].append(100.)
        result["object"]["yaw_offset_range_deg"][0] = -180.
        result["pregrasp"]["ee_rotation_offset_range_deg"]["roll"][0] = -45.
        result["pregrasp"]["ee_lateral_offset_range_m"][0] = -.1
        result["validation"]["physics_steps"] = 100
        self.assertEqual(self.config, original)

    def test_missing_and_unknown_keys_are_rejected_at_each_level(self):
        locations = [(), ("pregrasp",), ("at_grasp", "base"), ("object",),
                     ("validation",), ("pregrasp", "ee_rotation_offset_range_deg")]
        for location in locations:
            for operation in ("missing", "unknown"):
                config = copy.deepcopy(self.config)
                target = config
                for key in location:
                    target = target[key]
                if operation == "missing":
                    target.pop(next(iter(target)))
                else:
                    target["typo"] = 1
                with self.subTest(location=location, operation=operation):
                    with self.assertRaisesRegex(ValueError, operation):
                        validate_build_config(config)

    def test_schema_version_requires_supported_integer(self):
        for version in (True, False, 3., 4., "3", "4", 0, 1, 2, 5, None):
            with self.subTest(version=version):
                self.config["schema_version"] = version
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    validate_build_config(self.config)

    def test_ranges_reject_nonfinite_boolean_string_and_reversed_values(self):
        locations = [
            ("full_task", "base", "x_range_m"),
            ("pregrasp", "base", "y_range_m"),
            ("at_grasp", "base", "heading_offset_range_deg"),
            ("pregrasp", "ee_approach_distance_range_m"),
            ("at_grasp", "gripper_opening_range"),
            ("object", "yaw_offset_range_deg"),
            ("pregrasp", "ee_lateral_offset_range_m"),
            ("pregrasp", "ee_vertical_offset_range_m"),
            ("pregrasp", "ee_rotation_offset_range_deg", "roll"),
        ]
        invalid = ([True, 1], [0, False], [float("nan"), 1], [0, float("inf")],
                   [float("-inf"), 0], ["0.05", .1], [2, 1], [.1], [.1, .2, .3],
                   "0.05,0.10", None)
        for location in locations:
            for value in invalid:
                config = copy.deepcopy(self.config)
                target = config
                for key in location[:-1]:
                    target = target[key]
                target[location[-1]] = value
                with self.subTest(location=location, value=value):
                    with self.assertRaises(ValueError):
                        validate_build_config(config)

    def test_angles_must_fit_signed_degree_domain(self):
        for location in (("full_task", "base", "heading_offset_range_deg"),
                         ("object", "yaw_offset_range_deg")):
            for value in ([-181, 0], [0, 180.1]):
                config = copy.deepcopy(self.config)
                target = config
                for key in location[:-1]:
                    target = target[key]
                target[location[-1]] = value
                with self.subTest(location=location, value=value):
                    with self.assertRaisesRegex(ValueError, "degrees"):
                        validate_build_config(config)

    def test_object_yaw_can_be_fixed_or_cover_signed_boundaries(self):
        for value in ([0, 0], [-180, 180], [15, 15]):
            self.config["object"]["yaw_offset_range_deg"] = value
            with self.subTest(value=value):
                self.assertEqual(validate_build_config(self.config)["object"]["yaw_offset_range_deg"],
                                 value)

    def test_pregrasp_rotation_stays_in_local_small_angle_domain(self):
        for axis in ("roll", "pitch", "yaw"):
            for value in ([-45., 45.], [0., 0.], [-3., -3.]):
                config = copy.deepcopy(self.config)
                config["pregrasp"]["ee_rotation_offset_range_deg"][axis] = value
                with self.subTest(axis=axis, value=value):
                    self.assertEqual(validate_build_config(config)["pregrasp"][
                        "ee_rotation_offset_range_deg"][axis], value)
            for value in ([-45.001, 0.], [0., 45.001]):
                config = copy.deepcopy(self.config)
                config["pregrasp"]["ee_rotation_offset_range_deg"][axis] = value
                with self.subTest(axis=axis, value=value):
                    with self.assertRaisesRegex(ValueError, r"\[-45, 45\]"):
                        validate_build_config(config)

    def test_completely_infeasible_lateral_vertical_ranges_are_rejected(self):
        for lateral, vertical in (([.15, .16], [0, 0]),
                                  ([.09, .10], [.12, .13]),
                                  ([-.20, -.16], [-.01, .01])):
            config = copy.deepcopy(self.config)
            config["pregrasp"]["ee_lateral_offset_range_m"] = lateral
            config["pregrasp"]["ee_vertical_offset_range_m"] = vertical
            with self.subTest(lateral=lateral, vertical=vertical):
                with self.assertRaisesRegex(ValueError, "lateral/vertical"):
                    validate_build_config(config)
    def test_partly_feasible_ranges_are_preserved_for_proposal_filtering(self):
        self.config["pregrasp"]["ee_lateral_offset_range_m"] = [-.05, .05]
        self.config["pregrasp"]["ee_vertical_offset_range_m"] = [-.05, .05]
        self.assertEqual(validate_build_config(self.config), self.config)
        # Even an offset beyond d_min can be feasible for a larger sampled d.
        self.config["pregrasp"]["ee_lateral_offset_range_m"] = [.10, .14]
        self.config["pregrasp"]["ee_vertical_offset_range_m"] = [0, 0]
        self.assertEqual(validate_build_config(self.config), self.config)

    def near_goal_config(self):
        config = copy.deepcopy(self.config)
        config["schema_version"] = 4
        config["near_goal"] = {
            "base": copy.deepcopy(config["at_grasp"]["base"]),
            "object_lift_height_range_m": [.05, .10],
            "close_time_s": .20, "lift_time_s": .40, "hold_time_s": .20,
            "max_relative_slip_m": .01, "max_relative_rotation_deg": 10.,
            "min_finger_contact_force_n": .2, "min_contact_fraction": .8,
            "max_object_speed_m_s": .1,
        }
        return config

    def test_three_and_four_family_schemas_have_explicit_contracts(self):
        three = validate_build_config(self.config)
        self.assertEqual(three, self.config)
        self.assertNotIn("near_goal", three)
        four = self.near_goal_config()
        self.assertEqual(validate_build_config(four), four)
        missing = copy.deepcopy(four)
        missing.pop("near_goal")
        with self.assertRaisesRegex(ValueError, "missing keys near_goal"):
            validate_build_config(missing)
        four["schema_version"] = 3
        with self.assertRaisesRegex(ValueError, "unknown keys.*near_goal"):
            validate_build_config(four)

    def test_near_goal_missing_unknown_and_duplicate_fields_are_rejected(self):
        for key in self.near_goal_config()["near_goal"]:
            config = self.near_goal_config()
            config["near_goal"].pop(key)
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing keys"):
                validate_build_config(config)
        config = self.near_goal_config()
        config["near_goal"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_build_config(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("near_goal:\n  hold_time_s: .2\n  hold_time_s: .3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate SGS YAML key"):
                load_build_config(path)

    def test_near_goal_has_independent_base_and_detached_parameters(self):
        config = self.near_goal_config()
        config["near_goal"]["base"]["x_range_m"] = [.72, .81]
        config["near_goal"]["base"]["heading_offset_range_deg"] = [-5, 7]
        before = copy.deepcopy(config)
        result = validate_build_config(config)
        result["near_goal"]["base"]["x_range_m"][0] = 2.
        result["near_goal"]["object_lift_height_range_m"][0] = .3
        result["near_goal"]["hold_time_s"] = 1.
        self.assertEqual(config, before)
        self.assertEqual(config["at_grasp"]["base"]["x_range_m"], [.68, .85])
        for key, value in (("x_range_m", [.9, .8]),
                           ("heading_offset_range_deg", [-181., 0.]),
                           ("y_range_m", [False, 1.])):
            config = self.near_goal_config()
            config["near_goal"]["base"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_build_config(config)
        config = self.near_goal_config()
        config["near_goal"]["base"].update(x_range_m=[0, 0], y_range_m=[0, 0])
        with self.assertRaisesRegex(ValueError, "both be fixed at zero"):
            validate_build_config(config)

    def test_near_goal_height_is_positive_and_strictly_below_success(self):
        for value in ([0., .1], [-.1, .1], [.1, .05], [.05, .2], [.2, .2],
                      [.05, float("inf")], [.05, True]):
            config = self.near_goal_config()
            config["near_goal"]["object_lift_height_range_m"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_build_config(config)
        config = self.near_goal_config()
        config["near_goal"]["object_lift_height_range_m"] = [.05, .199]
        self.assertEqual(validate_build_config(config)["near_goal"]["object_lift_height_range_m"],
                         [.05, .199])
        config["near_goal"]["object_lift_height_range_m"] = [.075, .075]
        self.assertEqual(validate_build_config(config)["near_goal"]["object_lift_height_range_m"],
                         [.075, .075])

    def test_near_goal_uses_actual_teacher_threshold_in_validation_and_loader(self):
        config = self.near_goal_config()
        with self.assertRaisesRegex(ValueError, "strictly below"):
            validate_build_config(config, lifted_threshold=.10)
        config["near_goal"]["object_lift_height_range_m"] = [.15, .25]
        with self.assertRaisesRegex(ValueError, "strictly below"):
            validate_build_config(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "near_goal.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(load_build_config(path, lifted_threshold=.3), config)
        for threshold in (0., -.1, True, "0.2", float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "lifted_threshold"):
                validate_build_config(self.near_goal_config(), lifted_threshold=threshold)

    def test_near_goal_positive_fields_are_finite_positive_numbers(self):
        fields = ("close_time_s", "lift_time_s", "hold_time_s", "max_relative_slip_m",
                  "max_relative_rotation_deg", "min_finger_contact_force_n",
                  "min_contact_fraction", "max_object_speed_m_s")
        for key in fields:
            for value in (0., -1., True, "1", None, float("nan"), float("inf")):
                config = self.near_goal_config()
                config["near_goal"][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_build_config(config)

    def test_near_goal_hold_window_and_contact_fraction_have_physical_bounds(self):
        for key, value in (("hold_time_s", .099), ("min_contact_fraction", 1.001),
                           ("max_relative_rotation_deg", 180.001)):
            config = self.near_goal_config()
            config["near_goal"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                validate_build_config(config)
        config = self.near_goal_config()
        config["near_goal"].update(hold_time_s=.1, min_contact_fraction=1.,
                                   max_relative_rotation_deg=180.)
        self.assertEqual(validate_build_config(config), config)

    def test_zero_offsets_preserve_axial_pregrasp_targets(self):
        pregrasp = self.config["pregrasp"]
        pregrasp["ee_lateral_offset_range_m"] = [0., 0.]
        pregrasp["ee_vertical_offset_range_m"] = [0., 0.]
        pregrasp["ee_rotation_offset_range_deg"] = {
            axis: [0., 0.] for axis in ("roll", "pitch", "yaw")}
        pregrasp["ee_approach_distance_range_m"] = [.001, .001]
        self.assertEqual(validate_build_config(self.config)["pregrasp"], pregrasp)

    def test_physics_window_requires_positive_integer(self):
        for value in (True, False, 0, -1, 1., 2.5, "5", None):
            self.config["validation"]["physics_steps"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_build_config(self.config)
        for value in (1, 5, 25):
            self.config["validation"]["physics_steps"] = value
            self.assertEqual(validate_build_config(self.config)["validation"]["physics_steps"], value)

    def test_physics_limits_must_be_finite_positive_numbers(self):
        for key in ("max_object_displacement_m", "max_contact_force_n"):
            for value in (0., -1., True, "1", None, float("nan"), float("inf")):
                config = copy.deepcopy(self.config)
                config["validation"][key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        validate_build_config(config)

    def test_coincident_fixed_base_and_object_are_rejected(self):
        for family in ("full_task", "pregrasp", "at_grasp"):
            config = copy.deepcopy(self.config)
            config[family]["base"]["x_range_m"] = [0, 0]
            config[family]["base"]["y_range_m"] = [0, 0]
            with self.subTest(family=family):
                with self.assertRaisesRegex(ValueError, "both be fixed at zero"):
                    validate_build_config(config)

    def test_pregrasp_distance_is_positive(self):
        for value in ([0, .1], [-.1, .1]):
            self.config["pregrasp"]["ee_approach_distance_range_m"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive"):
                    validate_build_config(self.config)

    def test_only_default_full_task_arm_pose_is_supported(self):
        for value in ("custom", [0] * 6, None, True):
            self.config["full_task"]["arm_pose"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "arm_pose"):
                    validate_build_config(self.config)

    def test_gripper_opening_accepts_endpoints_and_fixed_values(self):
        for value in ([0., 1.], [0., 0.], [1., 1.], [.2, .8], [.5, .5]):
            self.config["at_grasp"]["gripper_opening_range"] = value
            with self.subTest(value=value):
                result = validate_build_config(self.config)
                self.assertEqual(result["at_grasp"]["gripper_opening_range"], value)

    def test_gripper_opening_rejects_values_outside_normalized_limits(self):
        for value in ([-.001, 1.], [0., 1.001], [-1., -1.], [2., 2.]):
            self.config["at_grasp"]["gripper_opening_range"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
                    validate_build_config(self.config)

    def test_lift_success_threshold_no_longer_restricts_at_grasp(self):
        for threshold in (None, .2, .01, 0.):
            with self.subTest(threshold=threshold):
                self.assertEqual(validate_build_config(self.config, lifted_threshold=threshold),
                                 validate_build_config(self.config))

    def test_legacy_held_configuration_requires_explicit_migration(self):
        legacy = copy.deepcopy(self.config)
        legacy["schema_version"] = 1
        legacy["grasped_before_lift"] = legacy.pop("at_grasp")
        legacy["grasped_before_lift"].pop("gripper_opening_range")
        legacy["grasped_before_lift"]["object_lift_height_range_m"] = [.03, .08]
        for version in (1, 2):
            legacy["schema_version"] = version
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "migrate to schema_version: 3"):
                    validate_build_config(legacy)

    def test_schema_two_requires_explicit_new_fields(self):
        legacy = copy.deepcopy(self.config)
        legacy["schema_version"] = 2
        legacy.pop("object")
        legacy.pop("validation")
        for key in ("ee_lateral_offset_range_m", "ee_vertical_offset_range_m",
                    "ee_rotation_offset_range_deg"):
            legacy["pregrasp"].pop(key)
        with self.assertRaisesRegex(ValueError, "migrate to schema_version: 3; add object, validation"):
            validate_build_config(legacy)

    def test_lift_setting_is_rejected_in_the_new_family(self):
        self.config["at_grasp"]["object_lift_height_range_m"] = [.03, .08]
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_build_config(self.config)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in (
                "schema_version: 3\nschema_version: 3\n",
                "pregrasp:\n  base:\n    x_range_m: [.68, .9]\n    x_range_m: [.7, .8]\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content):
                    with self.assertRaisesRegex(ValueError, "Duplicate SGS YAML key"):
                        load_build_config(path)

    def test_empty_nonmapping_and_invalid_yaml_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in ("", "[]", "true", "pregrasp: [broken"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content):
                    with self.assertRaises(ValueError):
                        load_build_config(path)


if __name__ == "__main__":
    unittest.main()
