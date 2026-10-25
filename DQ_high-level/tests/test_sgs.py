"""Sampler and reset-bank checks without importing or initializing Isaac Gym.

Run with the dqwbc Python, e.g. ``python -m unittest discover -s tests -p test_sgs.py``.
CUDA checks are enabled when a CUDA device is available.
"""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "sgs.py"
SPEC = importlib.util.spec_from_file_location("dq_sgs", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SGSSampler = MODULE.SGSSampler
ConfigurationBank = MODULE.ConfigurationBank


def bank_fixture(num_configs=6, num_dofs=19, num_bodies=23):
    roots = torch.zeros(num_configs, 13)
    roots[:, 6] = 1
    tensors = {
        "family_id": torch.arange(num_configs) % 3,
        "robot_root": roots.clone(),
        "object_root": roots.clone(),
        "table_root": roots.clone(),
        "rigid_body_state": roots[:, None, :].repeat(1, num_bodies, 1),
        "ee_jacobian": torch.eye(6)[None].repeat(num_configs, 1, 1),
        "dof_pos": torch.zeros(num_configs, num_dofs),
        "dof_vel": torch.zeros(num_configs, num_dofs),
        "ee_goal_cart": torch.zeros(num_configs, 3),
        "ee_goal_orn_rpy": torch.zeros(num_configs, 3),
        "gripper_target": torch.zeros(num_configs, 1),
        "table_vy": torch.linspace(-0.3, 0.3, num_configs),
        "rotation_enabled": torch.arange(num_configs) % 2 == 0,
        "table_yaw_rate": torch.zeros(num_configs),
        "table_height": torch.full((num_configs,), 0.55),
    }
    metadata = {
        "object_name": "sugar_box", "robot_asset": "b1z1/robot.urdf",
        "num_dofs": num_dofs, "num_bodies": num_bodies,
        "hold_steps": 15, "lifted_threshold": 0.2,
        "version": "test-v1", "optional": {"seed": 43},
        "physical_validation": {"passed": True, "test_steps": 3},
    }
    return tensors, metadata


def near_goal_bank_fixture():
    """Four-class physical contract, with the final two rows genuinely elevated."""
    from utils.sgs_grounding import APPROX_GROUND_LIMITS, FOOT_RADIUS
    tensors, metadata = bank_fixture(num_configs=8, num_bodies=5)
    tensors["family_id"] = torch.arange(8) // 2
    feet = [0, 2, 3, 4]
    tensors["rigid_body_state"][:, feet, 2] = FOOT_RADIUS
    tensors["object_root"][:, 2] = tensors["table_height"] + .045
    tensors["object_root"][6:, 2] += torch.tensor([.06, .09])
    tensors["lift_height"] = tensors["object_root"][:, 2] - tensors["table_height"] - .045
    for name, value in (("near_goal_relative_slip_m", .003),
                        ("near_goal_relative_rotation_deg", 2.),
                        ("near_goal_contact_fraction", .95),
                        ("near_goal_object_speed_m_s", .02),
                        ("near_goal_restore_verified", 1.)):
        tensors[name] = torch.zeros(8)
        tensors[name][6:] = value
    metadata.update({
        "version": "four-family-test", "object_init_height_m": .045,
        "family_names": ["full_task", "pregrasp", "at_grasp", "near_goal"],
        "physical_validation": {
            "passed": True, "validation_profile": "geometry_grasp_v2",
            "feet_near_ground": True, "collision_checked": True, "object_attachment": False,
            "stable_standing_required": False, "near_goal_grasp_verified": True,
            "grounding": {"foot_body_indices": feet, "foot_radius_m": FOOT_RADIUS,
                          "limits": dict(APPROX_GROUND_LIMITS)},
        },
        "construction": {"reset_config": {"near_goal": {
            "object_lift_height_range_m": [.05, .10],
            "max_relative_slip_m": .01, "max_relative_rotation_deg": 10.,
            "min_contact_fraction": .8, "max_object_speed_m_s": .1,
        }}},
    })
    return tensors, metadata


class RingReference:
    """Deliberately sequential oracle, not a copy of the vectorized algorithm."""

    def __init__(self, num_configs, history_size):
        self.h = history_size
        self.history = [[False] * history_size for _ in range(num_configs)]
        self.events = [[] for _ in range(num_configs)]
        self.head = [0] * num_configs
        self.last_seen = [-1] * num_configs

    def observe(self, ids, values, step):
        for config, value in zip(ids, values):
            self.history[config][self.head[config]] = bool(value)
            self.head[config] = (self.head[config] + 1) % self.h
            self.events[config].append(bool(value))
            self.last_seen[config] = step

    def check(self, test, sampler):
        test.assertEqual(sampler.history.cpu().tolist(), self.history)
        test.assertEqual(sampler.head.cpu().tolist(), self.head)
        test.assertEqual(sampler.count.cpu().tolist(),
                         [min(self.h, len(events)) for events in self.events])
        test.assertEqual(sampler.success_sum.cpu().tolist(),
                         [sum(events[-self.h:]) for events in self.events])
        test.assertEqual(sampler.visits_total.cpu().tolist(),
                         [len(events) for events in self.events])
        test.assertEqual(sampler.last_seen.cpu().tolist(), self.last_seen)


class SamplerTests(unittest.TestCase):
    device = "cpu"

    def make(self, *args, **kwargs):
        return SGSSampler(*args, device=self.device, **kwargs)

    def test_empty_history_is_uniform_but_not_observed_failure(self):
        sampler = self.make(5)
        self.assertEqual(sampler.p_init, 0.0)
        torch.testing.assert_close(sampler.probabilities().cpu(), torch.full((5,), 0.2))
        self.assertEqual(sampler.success_rates().cpu().tolist(), [0.0] * 5)
        self.assertEqual(sampler.count.sum().item(), 0)
        self.assertEqual(sampler.metrics()["unknown_count"].item(), 5)
        sampler.observe([], [], step=3)
        self.assertEqual(sampler.sample(0).numel(), 0)

    def test_default_kernel_matches_paper_numeric_ratios(self):
        sampler = self.make(9)
        counts = [0, 1, 10, 25, 50, 75, 90, 99, 100]
        for config, successes in enumerate(counts):
            sampler.observe([config] * 100,
                            [True] * successes + [False] * (100 - successes))
        probabilities = sampler.probabilities()
        ratios = (probabilities / probabilities[4]).cpu()
        expected = torch.tensor([
            0.142101804, 0.447347798, 0.774785878, 0.930650252,
            1.0, 0.930650252, 0.774785878, 0.447347798, 0.142101804,
        ])
        torch.testing.assert_close(ratios, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(bool((probabilities > 0).all()))
        torch.testing.assert_close(sampler.probabilities("uniform").cpu(),
                                   torch.full((9,), 1.0 / 9))

    def test_floor_temperature_and_asymmetric_target(self):
        for target, concentration, temperature, epsilon in (
                (0.66, 5.0, 2.0, 1e-8), (0.5, 10.0, 0.2, 1e-4),
                (0.0, 1.0, 2.0, 1e-4), (1.0, 1.0, 2.0, 1e-4),
                (0.5, 0.0, 2.0, 1e-4)):
            sampler = self.make(
                3, target_success_rate=target, concentration=concentration,
                temperature=temperature, epsilon=epsilon)
            sampler.observe([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1])
            unnormalized = []
            for rate in (0.0, 0.5, 1.0):
                weight = (rate + epsilon) ** (concentration * target)
                weight *= (1.0 - rate + epsilon) ** (concentration * (1.0 - target))
                unnormalized.append((max(weight, epsilon) + epsilon) ** (1 / max(temperature, 1)))
            expected = torch.tensor(unnormalized)
            expected /= expected.sum()
            torch.testing.assert_close(sampler.probabilities().cpu(), expected)

    def test_repeated_ids_and_oversized_groups_match_sequential_ring(self):
        generator = torch.Generator().manual_seed(913)
        sampler = self.make(7, history_size=5)
        reference = RingReference(7, 5)
        batches = [
            ([2, 1, 2, 2, 1], [1, 0, 0, 1, 1]),
            ([2] * 23 + [0, 1, 0, 6], [index % 3 == 0 for index in range(23)] + [1, 0, 0, 1]),
        ]
        for length in (1, 100, 37, 99, 250):
            batches.append((torch.randint(0, 7, (length,), generator=generator).tolist(),
                            torch.randint(0, 2, (length,), generator=generator).tolist()))
        for step, (ids, outcomes) in enumerate(batches, 1):
            sampler.observe(torch.tensor(ids, device=self.device),
                            torch.tensor(outcomes, device=self.device), step=step)
            reference.observe(ids, outcomes, step)
            reference.check(self, sampler)
        self.assertEqual(sampler.history.device.type, torch.device(self.device).type)

    def test_short_history_and_first_success_do_not_fill_unused_slots(self):
        sampler = self.make(3)
        sampler.observe([0, 1], [1, 0], step=4)
        self.assertEqual(sampler.count.cpu().tolist(), [1, 1, 0])
        self.assertEqual(sampler.success_rates().cpu().tolist(), [1.0, 0.0, 0.0])
        torch.testing.assert_close(sampler.probabilities().cpu(), torch.full((3,), 1 / 3))
        sampler.observe([0], [0], step=7)
        self.assertAlmostEqual(sampler.success_rates()[0].item(), 0.5)
        self.assertGreater(sampler.probabilities()[0].item(), sampler.probabilities()[2].item())

    def test_replacement_sampling_and_rng_snapshot(self):
        sampler = self.make(3, seed=987)
        sampler.observe([0, 0, 1, 2], [0, 1, 0, 1], step=20)
        sampler.sample(23)
        state = sampler.state_dict()
        expected = sampler.sample(500)
        restored = self.make(3, seed=1)
        restored.load_state_dict(state)
        actual = restored.sample(500)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.numel(), 500)
        self.assertLessEqual(actual.unique().numel(), 3)
        restored.observe([2, 2], [1, 0], step=21)
        self.assertEqual(state["tensors"]["visits_total"].tolist(), [2, 1, 1])

    def test_statistics_remain_device_tensors_and_preserve_family_mass(self):
        sampler = self.make(6)
        sampler.observe([0, 0, 2, 3], [0, 1, 1, 0], step=10)
        metrics = sampler.metrics(step=15, family_id=[0, 0, 1, 1, 2, 2])
        self.assertTrue(all(isinstance(value, torch.Tensor) and value.ndim == 0
                            for value in metrics.values()))
        self.assertEqual(metrics["unknown_count"].item(), 3)
        self.assertEqual(metrics["visits_total"].item(), 4)
        self.assertEqual(metrics["last_visit_age_mean"].item(), 5)
        self.assertAlmostEqual(sum(metrics["family_{}_probability".format(i)].item()
                                   for i in range(3)), 1.0, places=6)

    def test_four_family_metrics_preserve_probability_and_outcome_attribution(self):
        sampler = self.make(8)
        ids = torch.arange(8, device=self.device) // 2
        initial = sampler.metrics(family_id=ids, num_families=4)
        for family in range(4):
            self.assertAlmostEqual(initial[f"family_{family}_probability"].item(), .25)
        sampler.observe([6, 7, 7], [1, 0, 1])
        metrics = sampler.metrics(family_id=ids, num_families=4)
        self.assertEqual(metrics["family_3_visits"].item(), 3)
        self.assertEqual(metrics["family_2_visits"].item(), 0)
        self.assertAlmostEqual(sum(metrics[f"family_{i}_probability"].item() for i in range(4)), 1., places=6)

    def test_checkpoint_rejects_incompatible_parameters_and_corruption(self):
        sampler = self.make(3, history_size=5)
        sampler.observe([0, 1], [1, 0], step=2)
        state = sampler.state_dict()
        incompatible = self.make(3, history_size=6)
        with self.assertRaisesRegex(ValueError, "parameters"):
            incompatible.load_state_dict(state)
        corrupted = copy.deepcopy(state)
        corrupted["tensors"]["success_sum"][0] = 0
        with self.assertRaisesRegex(ValueError, "success sums"):
            sampler.load_state_dict(corrupted)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaSamplerTests(SamplerTests):
    device = "cuda:0"


class ConfigurationBankTests(unittest.TestCase):
    def test_four_family_roundtrip_preserves_grasp_evidence_and_closed_target(self):
        tensors, metadata = near_goal_bank_fixture()
        bank = ConfigurationBank(tensors, metadata)
        self.assertEqual(bank.family_names, ("full_task", "pregrasp", "at_grasp", "near_goal"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "near.pt"
            bank.save(path)
            loaded = ConfigurationBank.load(path)
            self.assertEqual(bank.fingerprint, loaded.fingerprint)
            self.assertEqual(loaded.batch([6, 7])["near_goal_restore_verified"].tolist(), [1., 1.])
            torch.testing.assert_close(loaded.batch([6, 7])["gripper_target"], torch.zeros(2, 1))

    def test_four_family_rejects_forged_height_contact_hold_and_restore_evidence(self):
        modifications = (
            ("lift_height", .07), ("near_goal_relative_slip_m", .02),
            ("near_goal_relative_rotation_deg", 12.), ("near_goal_contact_fraction", .5),
            ("near_goal_object_speed_m_s", .2), ("near_goal_restore_verified", 0.),
            ("gripper_target", -1.),
        )
        for name, value in modifications:
            with self.subTest(field=name):
                tensors, metadata = near_goal_bank_fixture()
                tensors[name][6] = value
                with self.assertRaisesRegex(ValueError, "near_goal"):
                    ConfigurationBank(tensors, metadata)
        for actual_lift in (.01, .11, .21):
            tensors, metadata = near_goal_bank_fixture()
            tensors["object_root"][6, 2] = tensors["table_height"][6] + .045 + actual_lift
            tensors["lift_height"][6] = actual_lift
            with self.subTest(actual_lift=actual_lift), self.assertRaisesRegex(ValueError, "near_goal"):
                ConfigurationBank(tensors, metadata)

    def test_four_family_requires_new_profile_and_declared_verification(self):
        for key, value in (("validation_profile", "geometry_v1"),
                           ("near_goal_grasp_verified", False), ("stable_standing_required", True)):
            tensors, metadata = near_goal_bank_fixture()
            metadata["physical_validation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ConfigurationBank(tensors, metadata)
        tensors, metadata = near_goal_bank_fixture()
        tensors.pop("near_goal_restore_verified")
        with self.assertRaisesRegex(ValueError, "evidence"):
            ConfigurationBank(tensors, metadata)

    def test_fixed_near_goal_height_tolerance_never_allows_completed_task(self):
        tensors, metadata = near_goal_bank_fixture()
        metadata["construction"]["reset_config"]["near_goal"]["object_lift_height_range_m"] = [.07, .07]
        tensors["lift_height"][6:] = torch.tensor([.066, .074])
        tensors["object_root"][6:, 2] = tensors["table_height"][6:] + .045 + tensors["lift_height"][6:]
        ConfigurationBank(tensors, metadata)
        metadata["construction"]["reset_config"]["near_goal"]["object_lift_height_range_m"] = [.198, .198]
        tensors["lift_height"][6:] = .201
        tensors["object_root"][6:, 2] = tensors["table_height"][6:] + .045 + .201
        with self.assertRaisesRegex(ValueError, "already successful"):
            ConfigurationBank(tensors, metadata)

    def test_four_family_requires_nonempty_near_goal_and_per_row_evidence_shape(self):
        tensors, metadata = near_goal_bank_fixture()
        tensors["family_id"][6:] = 2
        with self.assertRaisesRegex(ValueError, "nonempty near_goal"):
            ConfigurationBank(tensors, metadata)
        tensors, metadata = near_goal_bank_fixture()
        tensors["near_goal_contact_fraction"] = tensors["near_goal_contact_fraction"][:, None]
        with self.assertRaisesRegex(ValueError, "per-configuration"):
            ConfigurationBank(tensors, metadata)

    def test_near_goal_speed_evidence_must_cover_actual_saved_velocity_norm(self):
        tensors, metadata = near_goal_bank_fixture()
        # Each component is below the recorded .02 m/s, but the vector norm is
        # .025 m/s. Verify the physical root, not only the reported scalar.
        tensors["object_root"][6, 7:10] = torch.tensor([.015, .020, 0.])
        with self.assertRaisesRegex(ValueError, "understates saved object velocity"):
            ConfigurationBank(tensors, metadata)
        # The record is the maximum of saved-state and cold-replay speeds, so a
        # conservative value above the saved root speed is valid.
        tensors["near_goal_object_speed_m_s"][6] = .03
        ConfigurationBank(tensors, metadata)

    def test_round_trip_metadata_hash_and_batch(self):
        tensors, metadata = bank_fixture()
        bank = ConfigurationBank(tensors, metadata)
        expected = bank.batch(torch.tensor([4, 1, 4]))
        self.assertEqual(expected["family_id"].tolist(), [1, 1, 1])
        tensors["table_vy"].zero_()
        self.assertNotEqual(bank.tensors["table_vy"].abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.pt"
            bank.save(path)
            loaded = ConfigurationBank.load(
                path, expected_metadata={"object_name": "sugar_box", "hold_steps": 15,
                                         "physical_validation": {"passed": True}})
            self.assertEqual(bank.fingerprint, loaded.fingerprint)
            for name in bank.tensors:
                torch.testing.assert_close(bank.tensors[name], loaded.tensors[name])
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                ConfigurationBank.load(path, expected_metadata={"hold_steps": 20})
            payload = torch.load(path, weights_only=True)
            payload["tensors"]["table_vy"][0] += 0.01
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ConfigurationBank.load(path)

    def test_invalid_schema_states_and_post_construction_mutation(self):
        tensors, metadata = bank_fixture()
        invalid = tensors.copy()
        invalid["dof_vel"] = torch.zeros(6, 18)
        with self.assertRaisesRegex(ValueError, "shape"):
            ConfigurationBank(invalid, metadata)
        invalid = tensors.copy()
        invalid["table_height"] = torch.full((6,), float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            ConfigurationBank(invalid, metadata)
        invalid = tensors.copy()
        invalid["object_root"] = torch.zeros(6, 13)
        with self.assertRaisesRegex(ValueError, "quaternion"):
            ConfigurationBank(invalid, metadata)
        invalid = tensors.copy()
        invalid["rigid_body_state"] = tensors["rigid_body_state"].clone()
        invalid["rigid_body_state"][0, 1, 3:7] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            ConfigurationBank(invalid, metadata)
        invalid = tensors.copy()
        invalid["ee_jacobian"] = torch.zeros(6, 6, 7)
        with self.assertRaisesRegex(ValueError, "shape"):
            ConfigurationBank(invalid, metadata)
        invalid = metadata.copy()
        invalid.pop("version")
        with self.assertRaisesRegex(ValueError, "missing bank metadata"):
            ConfigurationBank(tensors, invalid)
        bank = ConfigurationBank(tensors, metadata)
        bank.tensors["table_vy"][0] = 0.123
        with self.assertRaisesRegex(ValueError, "mutated"):
            bank.state_dict()

    def test_optional_table_interval_validates_integer_values_and_range(self):
        tensors, metadata = bank_fixture()
        for dtype in (torch.long, torch.float32):
            tensors["table_change_interval"] = torch.full((6,), 50, dtype=dtype)
            ConfigurationBank(tensors, metadata)
        for invalid in (39.0, 60.0, 50.5, float("nan")):
            tensors["table_change_interval"] = torch.full((6,), invalid)
            with self.assertRaisesRegex(ValueError, "table_change_interval"):
                ConfigurationBank(tensors, metadata)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_load_on_cuda_and_index_on_device(self):
        tensors, metadata = bank_fixture()
        bank = ConfigurationBank(tensors, metadata)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.pt"
            bank.save(path)
            loaded = ConfigurationBank.load(path, device="cuda:0")
            self.assertEqual(loaded.fingerprint, bank.fingerprint)
            batch = loaded.batch(torch.tensor([5, 0], device="cuda:0"))
            self.assertTrue(all(value.is_cuda for value in batch.values()))


if __name__ == "__main__":
    unittest.main()
