"""Ordinary full-task snapshots coexist with strict curriculum-bank validation."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
from pathlib import Path
import tempfile
import unittest

import torch

import test_sgs_env as env_test
from utils.config import level03_motion_config
from utils.sgs import (ConfigurationBank, ORDINARY_FULL_TASK_PROFILE,
                       ORDINARY_FULL_TASK_PROFILE_V2, ORDINARY_FULL_LEVEL03_RESET_MODE)
from utils.sgs_level03 import prepare_rest_motion, restore_motion, snapshot_motion
from utils.table_motion import RemoteCarMotion


def ordinary_bank(level="Level03", with_at_grasp=False):
    original = (env_test.make_near_goal_bank() if with_at_grasp
                else env_test.make_active_near_goal_bank())
    tensors = {key: value.clone() for key, value in original.tensors.items()}
    metadata = deepcopy(original.metadata)
    metadata["table_level"] = level
    metadata["physical_validation"].update(
        validation_profile=(ORDINARY_FULL_TASK_PROFILE_V2 if with_at_grasp
                            else ORDINARY_FULL_TASK_PROFILE),
        feet_near_ground=False,
        feet_near_ground_families=list(original.family_names[1:]),
        full_task_reset_source="ordinary_reset")
    metadata["construction"]["reset_config"]["full_task"] = {"source": "ordinary_reset"}
    full = tensors["family_id"] == 0
    feet = metadata["physical_validation"]["grounding"]["foot_body_indices"]
    # Ordinary reset at base z=.6 does not promise planted feet, nor exactly
    # zero body/contact velocity after its one physical simulation tick.
    for index in feet:
        tensors["rigid_body_state"][full, index, 2] = .12
    tensors["table_root"][full, 7] = .0002
    tensors["robot_root"][full, 9] = -.049
    tensors["dof_pos"][full, -1] = -.31
    if with_at_grasp:
        metadata["construction"]["reset_config"].update(
            schema_version=6, at_grasp={"gripper_opening_range": [.2, 1.]})
        metadata["construction"].update(at_grasp_position_tolerance_m=.005,
                                        at_grasp_angle_tolerance_rad=.10,
                                        gripper_joint_limits_rad=[-math.pi / 2, 0.])
        n = original.num_configs
        tensors["gripper_opening"] = torch.zeros(n)
        tensors["candidate_local_offset"] = torch.zeros(n, 3)
        tensors["ee_rotation_offset_deg"] = torch.zeros(n, 3)
        tensors["grasp_candidate_index"] = torch.full((n,), -1.)
        at = tensors["family_id"] == original.family_names.index("at_grasp")
        tensors["gripper_opening"][at] = .8 / (math.pi / 2)
        tensors["grasp_candidate_index"][at] = 12.
        tensors["dof_pos"][at, -1] = -.8
        tensors["gripper_target"][at, 0] = -1.
    if level == "Level03":
        settings = level03_motion_config({})
        motion = RemoteCarMotion(original.num_configs, "cpu", settings)
        prepare_rest_motion(motion, .005)
        motion.reset(full.nonzero().flatten())
        tensors.update(snapshot_motion(motion))
        for name in ("table_vy", "table_yaw_rate", "rotation_enabled"):
            tensors[name].zero_()
        metadata.update(level03_motion_config=settings,
                        level03_reset_mode=ORDINARY_FULL_LEVEL03_RESET_MODE)
    return ConfigurationBank(tensors, metadata)


class OrdinaryFullBankTests(unittest.TestCase):
    def test_roundtrip_preserves_actual_ordinary_root_and_joint_states(self):
        for level in ("Level02", "Level03"):
            bank = ordinary_bank(level)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ordinary.pt"
                bank.save(path)
                restored = ConfigurationBank.load(path)
            self.assertEqual(bank.fingerprint, restored.fingerprint)
            self.assertEqual(bank.metadata, restored.metadata)
            for name, value in bank.tensors.items():
                torch.testing.assert_close(restored.tensors[name], value, atol=0., rtol=0.)

    def test_only_full_task_is_exempt_from_the_curriculum_foot_height_rule(self):
        bank = ordinary_bank()
        feet = bank.metadata["physical_validation"]["grounding"]["foot_body_indices"]
        for family in (1, 2):
            state = {name: value.clone() for name, value in bank.tensors.items()}
            row = (state["family_id"] == family).nonzero()[0, 0]
            state["rigid_body_state"][row, feet[0], 2] = .12
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "feet outside"):
                ConfigurationBank(state, bank.metadata)

    def test_metadata_must_explicitly_and_consistently_scope_the_new_contract(self):
        bank = ordinary_bank("Level02")
        mutations = [
            lambda m: m["physical_validation"].update(feet_near_ground=True),
            lambda m: m["physical_validation"].update(feet_near_ground_families=["near_goal"]),
            lambda m: m["physical_validation"].update(full_task_reset_source="geometry"),
            lambda m: m["physical_validation"].update(validation_profile="geometry_grasp_v3"),
            lambda m: m["construction"]["reset_config"].update(full_task={"source": "geometry"}),
            lambda m: m.update(family_names=["full_task", "near_goal", "pregrasp"]),
        ]
        for index, mutate in enumerate(mutations):
            metadata = deepcopy(bank.metadata)
            mutate(metadata)
            with self.subTest(case=index), self.assertRaises(ValueError):
                ConfigurationBank(bank.tensors, metadata)
        state = {name: value.clone() for name, value in bank.tensors.items()}
        state["family_id"][state["family_id"] == 0] = 1
        with self.assertRaisesRegex(ValueError, "family IDs"):
            ConfigurationBank(state, bank.metadata)

    def test_legacy_profile_still_rejects_airborne_feet(self):
        legacy = env_test.make_active_near_goal_bank()
        state = {name: value.clone() for name, value in legacy.tensors.items()}
        state["rigid_body_state"][0, 0, 2] = .12
        with self.assertRaisesRegex(ValueError, "feet outside"):
            ConfigurationBank(state, legacy.metadata)

    def test_driver_initialization_is_family_specific_for_both_axes(self):
        bank = ordinary_bank()
        full = bank.tensors["family_id"] == 0
        for prefix in ("level03_", "level03_x_"):
            self.assertFalse(bool(bank.tensors[prefix + "driver_initialized"][full].any()))
            self.assertTrue(bool(bank.tensors[prefix + "driver_initialized"][~full].all()))
            for field, row, value in (("driver_phase", 0, 1),
                                      ("motion_speed", 0, .01),
                                      ("motion_command_changed", 0, True),
                                      ("driver_phase", 2, 1)):
                state = {name: item.clone() for name, item in bank.tensors.items()}
                state[prefix + field][row] = value
                with self.subTest(prefix=prefix, field=field, row=row), self.assertRaises(ValueError):
                    ConfigurationBank(state, bank.metadata)

    def test_new_profile_cannot_use_legacy_all_initialized_reset_mode(self):
        bank = ordinary_bank()
        metadata = deepcopy(bank.metadata)
        metadata["level03_reset_mode"] = "rest_with_sampled_driver_intention"
        with self.assertRaisesRegex(ValueError, ORDINARY_FULL_LEVEL03_RESET_MODE):
            ConfigurationBank(bank.tensors, metadata)

    def test_saved_intention_cannot_be_smuggled_into_an_ordinary_full_row(self):
        bank = ordinary_bank()
        state = {name: value.clone() for name, value in bank.tensors.items()}
        for key in state:
            if key.startswith(("level03_",)):
                state[key][0] = state[key][2]
        with self.assertRaisesRegex(ValueError, "remain uninitialized"):
            ConfigurationBank(state, bank.metadata)

    def test_planar_contact_drift_is_preserved_only_in_ordinary_rows(self):
        bank = ordinary_bank()
        for row in (2, 4):
            state = {name: value.clone() for name, value in bank.tensors.items()}
            state["table_root"][row, 7] = .0002
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "planar velocity"):
                ConfigurationBank(state, bank.metadata)

    def test_restored_full_driver_starts_the_same_way_as_an_ordinary_reset(self):
        bank = ordinary_bank()
        settings = level03_motion_config({})
        restored = RemoteCarMotion(2, "cpu", settings)
        fresh = RemoteCarMotion(2, "cpu", settings)
        restored.step(.1)
        restore_motion(restored, bank.batch(torch.tensor([0, 1])))
        rng = torch.get_rng_state()
        expected = torch.stack([fresh.step(.005) for _ in range(20)])
        torch.set_rng_state(rng)
        actual = torch.stack([restored.step(.005) for _ in range(20)])
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)

    def test_production_loader_accepts_new_profile_for_both_motion_levels(self):
        initialize = env_test.source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_init_sgs")
        for level in ("Level02", "Level03"):
            bank = ordinary_bank(level)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ordinary.pt"
                bank.save(path)
                env = env_test.SGSInitializationTests().environment(path)
                env.table_level = level
                env.level03_motion_cfg = level03_motion_config({})
                env.cfg["env"]["sgs"]["bank_size"] = None
                with redirect_stdout(io.StringIO()):
                    initialize(env)
            self.assertEqual(env.sgs_bank.fingerprint, bank.fingerprint)


class OrdinaryFullFourFamilyBankTests(unittest.TestCase):
    def test_four_family_roundtrip_keeps_at_grasp_distinct_from_near_goal(self):
        for level in ("Level02", "Level03"):
            bank = ordinary_bank(level, with_at_grasp=True)
            self.assertEqual(bank.family_names, ("full_task", "pregrasp", "at_grasp", "near_goal"))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "four.pt"
                bank.save(path)
                saved = ConfigurationBank.load(path)
            self.assertEqual(saved.fingerprint, bank.fingerprint)
            self.assertEqual(saved.tensors["family_id"].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
            self.assertTrue(bool((saved.tensors["gripper_target"][4:6] < 0).all()))
            self.assertTrue(bool((saved.tensors["gripper_target"][6:] == 0).all()))

    def test_four_family_profile_cannot_be_relabelled_as_three_family(self):
        bank = ordinary_bank(with_at_grasp=True)
        for mutate in (
            lambda m: m["physical_validation"].update(validation_profile=ORDINARY_FULL_TASK_PROFILE),
            lambda m: m["physical_validation"].update(feet_near_ground_families=["pregrasp", "near_goal"]),
            lambda m: m["construction"]["reset_config"].update(schema_version=5),
            lambda m: m.update(family_names=["full_task", "pregrasp", "near_goal"]),
        ):
            metadata = deepcopy(bank.metadata)
            mutate(metadata)
            with self.assertRaises(ValueError):
                ConfigurationBank(bank.tensors, metadata)

    def test_arrival_and_open_gripper_evidence_are_required(self):
        bank = ordinary_bank(with_at_grasp=True)
        alterations = [
            ("candidate_local_offset", [.006, 0., 0.]),
            ("ee_rotation_offset_deg", [0., 0., 6.]),
            ("gripper_opening", 0.),
            ("gripper_opening", .8),  # disagrees with the actual joint angle
            ("grasp_candidate_index", -1.),
            ("grasp_candidate_index", 1.5),
            ("gripper_target", [0.]),
        ]
        for name, value in alterations:
            state = {key: item.clone() for key, item in bank.tensors.items()}
            state[name][4] = torch.as_tensor(value)
            with self.subTest(field=name, value=value), self.assertRaisesRegex(ValueError, "at_grasp"):
                ConfigurationBank(state, bank.metadata)
        for field in ("candidate_local_offset", "gripper_opening", "ee_rotation_offset_deg"):
            state = dict(bank.tensors)
            del state[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "at_grasp"):
                ConfigurationBank(state, bank.metadata)

    def test_at_grasp_opening_bounds_cannot_include_closed(self):
        bank = ordinary_bank(with_at_grasp=True)
        for bounds in ([0., 1.], [-.1, .8], [.2, 1.1], [.8, .2]):
            metadata = deepcopy(bank.metadata)
            metadata["construction"]["reset_config"]["at_grasp"]["gripper_opening_range"] = bounds
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "opening_range"):
                ConfigurationBank(bank.tensors, metadata)

    def test_arrival_orientation_uses_quaternion_angle_instead_of_euler_norm(self):
        bank = ordinary_bank(with_at_grasp=True)
        state = {key: item.clone() for key, item in bank.tensors.items()}
        # The Euler norm exceeds .1 rad, but the represented rotation is .09997.
        state["ee_rotation_offset_deg"][4] = 3.34
        ConfigurationBank(state, bank.metadata)
        state["ee_rotation_offset_deg"][4] = 3.35
        with self.assertRaisesRegex(ValueError, "not aligned"):
            ConfigurationBank(state, bank.metadata)

    def test_opening_requires_consistent_joint_limits_and_construction_metadata(self):
        bank = ordinary_bank(with_at_grasp=True)
        for limits in (None, [-1., 0.], [-math.pi / 2, .1], [0., 0.]):
            metadata = deepcopy(bank.metadata)
            metadata["construction"]["gripper_joint_limits_rad"] = limits
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "at_grasp"):
                ConfigurationBank(bank.tensors, metadata)
        for mutate in (lambda m: m.update(construction=None),
                       lambda m: m["construction"].update(reset_config=None)):
            metadata = deepcopy(bank.metadata)
            mutate(metadata)
            with self.assertRaisesRegex(ValueError, "construction config"):
                ConfigurationBank(bank.tensors, metadata)

    def test_at_grasp_must_be_grounded_and_unlifted(self):
        bank = ordinary_bank(with_at_grasp=True)
        state = {key: item.clone() for key, item in bank.tensors.items()}
        state["rigid_body_state"][4, 0, 2] = .12
        with self.assertRaisesRegex(ValueError, "feet outside"):
            ConfigurationBank(state, bank.metadata)
        state = {key: item.clone() for key, item in bank.tensors.items()}
        state["object_root"][4, 2] += .05
        with self.assertRaisesRegex(ValueError, "before lifting"):
            ConfigurationBank(state, bank.metadata)

    def test_new_four_family_bank_loads_through_the_production_initializer(self):
        initialize = env_test.source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_init_sgs")
        for level in ("Level02", "Level03"):
            bank = ordinary_bank(level, with_at_grasp=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "four.pt"
                bank.save(path)
                env = env_test.SGSInitializationTests().environment(path)
                env.table_level = level
                env.level03_motion_cfg = level03_motion_config({})
                env.cfg["env"]["sgs"]["bank_size"] = None
                with redirect_stdout(io.StringIO()):
                    initialize(env)
                self.assertEqual(env.sgs_family_names, bank.family_names)

    def test_schema_four_bank_stays_readable_without_the_new_at_grasp_evidence(self):
        legacy = env_test.make_near_goal_bank()
        self.assertEqual(legacy.family_names, ("full_task", "pregrasp", "at_grasp", "near_goal"))
        self.assertNotIn("candidate_local_offset", legacy.tensors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            legacy.save(path)
            saved = ConfigurationBank.load(path)
        self.assertEqual(legacy.fingerprint, saved.fingerprint)


if __name__ == "__main__":
    unittest.main()
