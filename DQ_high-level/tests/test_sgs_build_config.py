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
        self.config = load_build_config()

    def test_repository_default_config_is_valid_for_teacher(self):
        normalized = validate_build_config(self.config, lifted_threshold=0.2)
        self.assertTrue(MODULE.DEFAULT_BUILD_CONFIG_PATH.is_absolute())
        self.assertEqual(normalized["full_task"]["base"]["x_range_m"], [2., 2.])
        self.assertEqual(normalized["full_task"]["arm_pose"], "default")
        self.assertEqual(normalized["schema_version"], 2)
        self.assertEqual(normalized["pregrasp"]["ee_approach_distance_range_m"], [.05, .15])
        self.assertEqual(normalized["at_grasp"]["gripper_opening_range"], [0., 1.])
        self.assertEqual(normalized["full_task"]["base"]["heading_offset_range_deg"],
                         [-9.1673, 9.1673])
        self.assertEqual(normalized["pregrasp"]["base"]["x_range_m"], [.68, .95])
        self.assertEqual(normalized["pregrasp"]["base"]["y_range_m"], [-.2, .2])
        self.assertEqual(normalized["at_grasp"]["base"]["x_range_m"], [.68, .85])
        self.assertEqual(normalized["at_grasp"]["base"]["y_range_m"], [-.1, .1])

    def test_default_loading_does_not_depend_on_working_directory(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(load_build_config(), self.config)
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
        self.config["at_grasp"]["gripper_opening_range"] = [.2, .8]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "自定义配置.yaml"
            path.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
            result = validate_build_config(load_build_config(path), lifted_threshold=.2)
        self.assertEqual(result["full_task"]["base"]["x_range_m"], [-3., -1.])
        self.assertEqual(result["pregrasp"]["ee_approach_distance_range_m"], [.01, .04])
        self.assertEqual(result["at_grasp"]["gripper_opening_range"], [.2, .8])

    def test_normalization_returns_detached_mutable_lists(self):
        original = copy.deepcopy(self.config)
        result = validate_build_config(self.config)
        result["pregrasp"]["base"]["x_range_m"][0] = 100.
        result["at_grasp"]["gripper_opening_range"].append(100.)
        self.assertEqual(self.config, original)

    def test_missing_and_unknown_keys_are_rejected_at_each_level(self):
        locations = [(), ("pregrasp",), ("at_grasp", "base")]
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
        for version in (True, False, 2., "2", 0, 1, 3, None):
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
        for value in ([-181, 0], [0, 180.1]):
            self.config["full_task"]["base"]["heading_offset_range_deg"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "degrees"):
                    validate_build_config(self.config)

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
                with self.assertRaisesRegex(ValueError, "migrate to schema_version: 2"):
                    validate_build_config(legacy)

    def test_lift_setting_is_rejected_in_the_new_family(self):
        self.config["at_grasp"]["object_lift_height_range_m"] = [.03, .08]
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_build_config(self.config)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in (
                "schema_version: 2\nschema_version: 2\n",
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
