"""Ground-support regressions without importing Isaac Gym.

Run with ``python -m unittest discover -s tests -p test_sgs_grounding.py``.
The same physical cases run on CUDA when a GPU is available.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from test_sgs import ConfigurationBank, bank_fixture


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "sgs_grounding.py"
SPEC = importlib.util.spec_from_file_location("dq_sgs_grounding", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
grounded_mask = MODULE.grounded_mask
reset_supported_mask = MODULE.reset_supported_mask
approx_grounded_mask = MODULE.approx_grounded_mask


class ApproximateGroundingTests(unittest.TestCase):
    device = "cpu"

    def test_nearby_collision_bottoms_are_accepted_without_force_or_velocity(self):
        bottom = torch.tensor([[0., -.004, .019, .005]], device=self.device)
        result = approx_grounded_mask(bottom)
        self.assertEqual(result.tolist(), [True])
        self.assertEqual(result.device, bottom.device)
        self.assertEqual(result.dtype, torch.bool)

    def test_airborne_penetrating_and_nonfinite_feet_are_rejected_independently(self):
        bottom = torch.zeros(6, 4, device=self.device)
        bottom[0] = .08
        bottom[1, 0] = -.006
        bottom[2, 1] = .021
        bottom[3, 2] = float("nan")
        bottom[4, 3] = float("inf")
        self.assertEqual(approx_grounded_mask(bottom).tolist(), [False] * 5 + [True])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaApproximateGroundingTests(ApproximateGroundingTests):
    device = "cuda:0"


class GroundingTests(unittest.TestCase):
    device = "cpu"

    def standing(self, count=1):
        root = torch.zeros(count, 13, device=self.device)
        root[:, :3] = torch.tensor([-2., 2., .52], device=self.device)
        root[:, 6] = 1.  # Unit quaternion, xyzw.
        root[:, 7:10] = torch.tensor([.03, -.02, .01], device=self.device)
        root[:, 10:13] = torch.tensor([.1, -.1, .05], device=self.device)
        # One foot is close to the ground but unloaded; three carry the robot.
        bottom = torch.tensor([[-.001, .002, 0., .006]], device=self.device).repeat(count, 1)
        force = torch.zeros(count, 4, 3, device=self.device)
        force[:, :, 2] = torch.tensor([120., 150., 90., 0.], device=self.device)
        return root, bottom, force

    def test_three_load_bearing_feet_and_low_body_motion_are_accepted(self):
        root, bottom, force = self.standing(2)
        force[1, 3, 2] = 80.  # Four-foot support also qualifies.
        result = grounded_mask(root, bottom, force)
        self.assertEqual(result.tolist(), [True, True])
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.device, root.device)

    def test_original_eight_centimeter_airborne_reset_is_rejected(self):
        root, bottom, force = self.standing(2)
        root[:, 2] = .60
        bottom[:] = .08
        force[0] = 0.  # The original airborne snapshot has no contacts.
        force[1, :, 2] = 100.  # Stale forces cannot make an airborne state valid.
        self.assertEqual(grounded_mask(root, bottom, force).tolist(), [False, False])

    def test_one_raised_foot_is_rejected_even_with_three_support_contacts(self):
        root, bottom, force = self.standing()
        bottom[0, 3] = .05
        self.assertFalse(grounded_mask(root, bottom, force).item())

    def test_collision_penetration_is_rejected_despite_large_contact_force(self):
        root, bottom, force = self.standing()
        bottom[0, 0] = -.015
        force[0, 0, 2] = 1000.
        self.assertFalse(grounded_mask(root, bottom, force).item())

    def test_two_support_feet_are_insufficient_even_if_total_load_is_large(self):
        root, bottom, force = self.standing()
        force[:] = 0.
        force[0, :2, 2] = 300.
        self.assertFalse(grounded_mask(root, bottom, force).item())

    def test_horizontal_and_downward_forces_do_not_count_as_ground_support(self):
        root, bottom, force = self.standing(2)
        force[:] = 0.
        force[0, :, :2] = 200.  # Side contacts, not upward support.
        force[1, :, 2] = -100.
        self.assertEqual(grounded_mask(root, bottom, force).tolist(), [False, False])

    def test_touchdown_and_fast_body_rotation_are_not_settled_stances(self):
        root, bottom, force = self.standing(4)
        root[0, 7:10] = torch.tensor([0., 0., -.4], device=self.device)
        root[1, 7:10] = torch.tensor([.12, .12, 0.], device=self.device)
        root[2, 10:13] = torch.tensor([0., .8, 0.], device=self.device)
        root[3, 10:13] = torch.tensor([.35, .35, .35], device=self.device)
        self.assertEqual(grounded_mask(root, bottom, force).tolist(), [False] * 4)

    def test_nonfinite_pose_clearance_or_contact_force_is_never_accepted(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                root, bottom, force = self.standing(3)
                root[0, 0] = invalid
                bottom[1, 3] = invalid
                force[2, 3, 0] = invalid  # Even an unloaded foot must be finite.
                self.assertEqual(grounded_mask(root, bottom, force).tolist(), [False] * 3)

    def test_batch_keeps_valid_neighbors_and_does_not_modify_snapshot(self):
        root, bottom, force = self.standing(4)
        bottom[1] = .08
        force[2] = 0.
        before = tuple(value.clone() for value in (root, bottom, force))
        self.assertEqual(grounded_mask(root, bottom, force).tolist(), [True, False, False, True])
        for actual, expected in zip((root, bottom, force), before):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaGroundingTests(GroundingTests):
    device = "cuda:0"


class ResetSupportTests(unittest.TestCase):
    device = "cpu"

    def redistributed_load(self, count=1):
        bottom = torch.tensor([[0., -.001, .002, .003]], device=self.device).repeat(count, 1)
        forces = torch.zeros(count, 4, 3, device=self.device)
        forces[:, :, 2] = torch.tensor([240., 200., 0., 0.], device=self.device)
        return bottom, forces

    def test_first_tick_allows_two_loaded_feet_with_other_feet_near_ground(self):
        bottom, forces = self.redistributed_load()
        result = reset_supported_mask(bottom, forces)
        self.assertEqual(result.tolist(), [True])
        self.assertEqual(result.device, bottom.device)
        self.assertEqual(result.dtype, torch.bool)
        # This first-tick load redistribution cannot be accepted as a settled
        # configuration-bank snapshot: that still requires three support feet.
        root = torch.zeros(1, 13, device=self.device)
        root[:, 2] = .52
        root[:, 6] = 1.
        self.assertFalse(grounded_mask(root, bottom, forces).item())

    def test_airborne_reset_still_fails_with_stale_four_foot_forces(self):
        bottom, forces = self.redistributed_load()
        bottom[:] = .08
        forces[:, :, 2] = 100.
        self.assertFalse(reset_supported_mask(bottom, forces).item())

    def test_single_support_foot_cannot_pass_by_carrying_more_load(self):
        bottom, forces = self.redistributed_load()
        forces[:] = 0.
        forces[:, 0, 2] = 600.
        self.assertFalse(reset_supported_mask(bottom, forces).item())

    def test_raised_or_penetrating_unloaded_foot_is_not_ignored(self):
        bottom, forces = self.redistributed_load(2)
        bottom[0, 3] = .025
        bottom[1, 3] = -.015
        self.assertEqual(reset_supported_mask(bottom, forces).tolist(), [False, False])

    def test_nonfinite_foot_data_fails_without_affecting_valid_neighbor(self):
        bottom, forces = self.redistributed_load(3)
        bottom[0, 3] = float("nan")
        forces[1, 3, 0] = float("nan")
        self.assertEqual(reset_supported_mask(bottom, forces).tolist(), [False, False, True])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaResetSupportTests(ResetSupportTests):
    device = "cuda:0"


class GroundedBankTests(unittest.TestCase):
    def supported_bank(self):
        tensors, metadata = bank_fixture()
        feet = [2, 6, 12, 18]
        tensors["robot_root"][:, 2] = .52
        tensors["rigid_body_state"][:, :, 2] = .5
        tensors["rigid_body_state"][:, feet, 2] = torch.tensor([.039, .042, .040, .046])
        forces = torch.zeros(6, 4, 3)
        forces[:, :, 2] = torch.tensor([120., 150., 90., 0.])
        tensors["initial_foot_contact_force"] = forces
        metadata["physical_validation"].update({
            "grounded_initial_states": True,
            "grounding": {"foot_body_indices": feet, "foot_radius_m": .04},
        })
        return tensors, metadata

    def test_supported_bank_round_trip_retains_grounding_evidence(self):
        tensors, metadata = self.supported_bank()
        bank = ConfigurationBank(tensors, metadata)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "grounded.pt"
            bank.save(path)
            for device in devices:
                with self.subTest(device=device):
                    loaded = ConfigurationBank.load(path, device=device, expected_metadata={
                        "physical_validation": {"grounded_initial_states": True},
                    })
                    self.assertEqual(loaded.fingerprint, bank.fingerprint)
                    self.assertEqual(loaded.metadata["physical_validation"],
                                     metadata["physical_validation"])
                    batch = loaded.batch(torch.tensor([4, 0, 4], device=device))
                    torch.testing.assert_close(batch["initial_foot_contact_force"].cpu(),
                                               tensors["initial_foot_contact_force"][[4, 0, 4]])

    def test_grounded_metadata_does_not_override_airborne_snapshot(self):
        tensors, metadata = self.supported_bank()
        feet = metadata["physical_validation"]["grounding"]["foot_body_indices"]
        tensors["robot_root"][1, 2] = .60
        tensors["rigid_body_state"][1, feet, 2] += .08
        # The other five entries and all contact-force evidence remain valid.
        with self.assertRaisesRegex(ValueError, "ungrounded or unsettled"):
            ConfigurationBank(tensors, metadata)

    def test_missing_or_incomplete_contact_evidence_is_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                tensors, metadata = self.supported_bank()
                if missing:
                    tensors.pop("initial_foot_contact_force")
                else:
                    tensors["initial_foot_contact_force"] = tensors[
                        "initial_foot_contact_force"][:, :3]
                with self.assertRaisesRegex(ValueError, "initial foot contact forces"):
                    ConfigurationBank(tensors, metadata)

    def test_invalid_foot_indices_cannot_substitute_other_bodies(self):
        for indices in ([2, 2, 12, 18], [-1, 6, 12, 18], [2, 6, 12, 23],
                        [2., 6, 12, 18], [2, 6, 12]):
            with self.subTest(indices=indices):
                tensors, metadata = self.supported_bank()
                metadata["physical_validation"]["grounding"]["foot_body_indices"] = indices
                with self.assertRaisesRegex(ValueError, "grounding geometry"):
                    ConfigurationBank(tensors, metadata)
        tensors, metadata = self.supported_bank()
        # In-range indices pointing to elevated body links also fail the check.
        metadata["physical_validation"]["grounding"]["foot_body_indices"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "ungrounded or unsettled"):
            ConfigurationBank(tensors, metadata)

    def test_legacy_bank_can_be_inspected_but_cannot_claim_grounded_training(self):
        tensors, metadata = bank_fixture()
        bank = ConfigurationBank(tensors, metadata)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            bank.save(path)
            self.assertEqual(ConfigurationBank.load(path).fingerprint, bank.fingerprint)
            with self.assertRaisesRegex(ValueError, "grounded_initial_states"):
                ConfigurationBank.load(path, expected_metadata={
                    "physical_validation": {"grounded_initial_states": True},
                })


class GeometryBankTests(unittest.TestCase):
    def geometry_bank(self):
        tensors, metadata = bank_fixture()
        feet = [2, 6, 12, 18]
        tensors["robot_root"][:, 2] = .52
        tensors["robot_root"][:, 7:10] = .7
        tensors["robot_root"][:, 10:13] = 1.
        tensors["rigid_body_state"][:, :, 2] = .5
        tensors["rigid_body_state"][:, feet, 2] = torch.tensor([.036, .040, .045, .059])
        metadata["family_names"] = ["full_task", "pregrasp", "at_grasp"]
        metadata["physical_validation"] = {
            "passed": True, "validation_profile": "geometry_v1",
            "feet_near_ground": True, "collision_checked": True,
            "object_attachment": False,
            "grounding": {
                "foot_body_indices": feet, "foot_radius_m": MODULE.FOOT_RADIUS,
                "limits": dict(MODULE.APPROX_GROUND_LIMITS),
            },
        }
        return tensors, metadata

    def test_round_trip_allows_moving_base_without_contact_evidence(self):
        tensors, metadata = self.geometry_bank()
        bank = ConfigurationBank(tensors, metadata)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "geometry.pt"
            bank.save(path)
            for device in devices:
                with self.subTest(device=device):
                    loaded = ConfigurationBank.load(path, device=device, expected_metadata={
                        "family_names": ["full_task", "pregrasp", "at_grasp"],
                        "physical_validation": {"validation_profile": "geometry_v1"},
                    })
                    self.assertEqual(loaded.fingerprint, bank.fingerprint)
                    batch = loaded.batch(torch.tensor([5, 1, 0], device=device))
                    torch.testing.assert_close(batch["robot_root"].cpu(),
                                               tensors["robot_root"][[5, 1, 0]])
                    self.assertNotIn("initial_foot_contact_force", batch)

    def test_zero_contact_force_does_not_disqualify_geometry_bank(self):
        tensors, metadata = self.geometry_bank()
        tensors["initial_foot_contact_force"] = torch.zeros(6, 4, 3)
        ConfigurationBank(tensors, metadata)

    def test_geometry_flag_cannot_accept_original_eight_centimeter_gap(self):
        tensors, metadata = self.geometry_bank()
        feet = metadata["physical_validation"]["grounding"]["foot_body_indices"]
        tensors["rigid_body_state"][2, feet, 2] = .12
        with self.assertRaisesRegex(ValueError, "approximate ground limits"):
            ConfigurationBank(tensors, metadata)

    def test_geometry_flags_must_explicitly_certify_requested_checks(self):
        for name, values in {
                "passed": [False, 1, "true", None],
                "feet_near_ground": [False, 1, "true", None],
                "collision_checked": [False, 1, "true", None],
                "object_attachment": [True, 0, "false", None]}.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    tensors, metadata = self.geometry_bank()
                    metadata["physical_validation"][name] = value
                    with self.assertRaisesRegex(ValueError, "geometry_v1 validation flag"):
                        ConfigurationBank(tensors, metadata)

    def test_old_grasped_family_cannot_be_relabelled_as_geometry_profile(self):
        tensors, metadata = self.geometry_bank()
        metadata["family_names"][2] = "grasped"
        with self.assertRaisesRegex(ValueError, "family_names"):
            ConfigurationBank(tensors, metadata)

    def test_geometry_profile_cannot_claim_stable_standing(self):
        tensors, metadata = self.geometry_bank()
        metadata["physical_validation"]["grounded_initial_states"] = True
        with self.assertRaisesRegex(ValueError, "does not certify stable"):
            ConfigurationBank(tensors, metadata)

    def test_geometry_limits_and_foot_indices_cannot_be_bypassed_in_metadata(self):
        for field, invalid in (("foot_body_indices", [2, 2, 12, 18]),
                               ("foot_body_indices", [2, 6, 12, 23]),
                               ("foot_radius_m", .03),
                               ("limits", {"min_foot_bottom_m": -.1,
                                           "max_foot_bottom_m": .1})):
            with self.subTest(field=field):
                tensors, metadata = self.geometry_bank()
                metadata["physical_validation"]["grounding"][field] = invalid
                with self.assertRaises(ValueError):
                    ConfigurationBank(tensors, metadata)

    def test_historical_bank_does_not_satisfy_new_training_profile(self):
        tensors, metadata = GroundedBankTests().supported_bank()
        bank = ConfigurationBank(tensors, metadata)
        with self.assertRaisesRegex(ValueError, "validation_profile"):
            bank.validate({"physical_validation": {"validation_profile": "geometry_v1"}})


if __name__ == "__main__":
    unittest.main()
