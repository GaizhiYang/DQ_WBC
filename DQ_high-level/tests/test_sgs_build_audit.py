"""Offline construction accounting and coverage checks on CPU/CUDA."""

from collections import OrderedDict
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "sgs_build_audit.py"
SPEC = importlib.util.spec_from_file_location("dq_sgs_build_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BuildAudit = MODULE.BuildAudit
FAMILIES = ("full_task", "pregrasp", "at_grasp")


class BuildAuditTests(unittest.TestCase):
    device = "cpu"

    def vector(self, values, dtype=torch.long):
        return torch.tensor(values, device=self.device, dtype=dtype)

    def proposals(self):
        ids = self.vector([12, 12, 23, 99, 23, 12])
        checks = OrderedDict([
            ("ik", self.vector([1, 0, 1, 0, 1, 1], torch.bool)),
            ("collision", self.vector([1, 0, 0, 1, 1, 1], torch.bool)),
            ("distance", self.vector([1, 1, 0, 0, 1, 1], torch.bool)),
        ])
        return ids, checks, self.vector([0, 4])

    def bank(self, include_full=True):
        family = [0, 1, 1, 2] if include_full else [1, 1, 2]
        count = len(family)
        robot = torch.zeros(count, 13, device=self.device)
        obj = robot.clone()
        robot[:, 6] = obj[:, 6] = 1
        offsets = [[2., 0.], [.8, .1], [.9, -.1], [.75, 0.]]
        headings = [0., 10., -10., 5.]
        candidate = [-1, 12, 12, 23]
        local = [[0., 0., 0.], [-.08, .006, .008], [-.10, -.01, 0.], [0., .002, 0.]]
        if not include_full:
            offsets, headings, candidate, local = offsets[1:], headings[1:], candidate[1:], local[1:]
        delta = self.vector(offsets, torch.float32)
        robot[:, :2] = self.vector([5., -3.], torch.float32)
        obj[:, :2] = robot[:, :2] + delta
        yaw = delta[:, 1].atan2(delta[:, 0]) + torch.deg2rad(self.vector(headings, torch.float32))
        robot[:, 5], robot[:, 6] = (yaw / 2).sin(), (yaw / 2).cos()
        return {
            "family_id": self.vector(family), "robot_root": robot, "object_root": obj,
            "table_height": torch.full((count,), .55, device=self.device),
            "gripper_opening": torch.full((count,), .75, device=self.device),
            "object_yaw_offset_deg": torch.zeros(count, device=self.device),
            "candidate_local_offset": self.vector(local, torch.float32),
            "ee_rotation_offset_deg": torch.zeros(count, 3, device=self.device),
            "validation_peak_contact_force_n": torch.full((count,), 42., device=self.device),
            "validation_max_object_displacement_m": torch.zeros(count, device=self.device),
            "grasp_candidate_index": self.vector(candidate, torch.float32),
        }

    def test_overlapping_rejections_are_separate_from_first_failure_and_quota(self):
        audit = BuildAudit(FAMILIES)
        audit.record(1, *self.proposals())
        report = audit.summary()
        self.assertEqual(report["totals"], {
            "proposed": 6, "valid": 3, "stored": 2, "invalid": 3, "valid_not_stored": 1})
        family = report["families"]["pregrasp"]
        self.assertEqual(family["rejection_counts"], {"ik": 2, "collision": 2, "distance": 2})
        self.assertEqual(family["first_rejection_counts"], {"ik": 2, "collision": 1, "distance": 0})
        self.assertEqual(sum(family["first_rejection_counts"].values()), family["invalid"])
        candidates = {row["candidate_id"]: row for row in family["candidates"]}
        self.assertEqual(set(candidates), {12, 23, 99})
        self.assertEqual([candidates[12][name] for name in ("proposed", "valid", "stored")], [3, 2, 1])
        self.assertEqual(candidates[12]["valid_not_stored"], 1)
        self.assertEqual(candidates[99]["stored"], 0)
        self.assertEqual(candidates[99]["first_rejection_counts"]["ik"], 1)

    def test_repeated_batches_accumulate_without_losing_rejected_candidates(self):
        audit = BuildAudit(FAMILIES)
        audit.record(1, *self.proposals())
        checks = {name: self.vector([1], torch.bool) for name in ("ik", "collision", "distance")}
        audit.record(1, self.vector([12]), checks, self.vector([]))
        family = audit.summary()["families"]["pregrasp"]
        self.assertEqual([family[name] for name in ("proposed", "valid", "stored", "valid_not_stored")],
                         [7, 4, 2, 2])
        self.assertEqual(len(family["candidates"]), 3)
        self.assertAlmostEqual(family["acceptance_rate"], 4 / 7)

    def test_full_task_and_empty_proposals_do_not_invent_candidate_ids(self):
        audit = BuildAudit(FAMILIES)
        audit.record(0, self.vector([-1, -1], torch.float32),
                     {"geometry": self.vector([1, 0], torch.bool)}, self.vector([0]))
        audit.record(2, self.vector([]), {"geometry": self.vector([], torch.bool)}, self.vector([]))
        report = audit.summary()
        self.assertEqual(report["families"]["full_task"]["candidates"], [])
        self.assertEqual(report["families"]["full_task"]["stored"], 1)
        self.assertIsNone(report["families"]["at_grasp"]["acceptance_rate"])
        self.assertEqual(report["families"]["at_grasp"]["proposed"], 0)

    def test_record_and_summary_leave_inputs_and_internal_counters_unmodified(self):
        audit = BuildAudit(FAMILIES)
        ids, checks, chosen = self.proposals()
        before = [ids.clone(), chosen.clone()] + [value.clone() for value in checks.values()]
        audit.record(1, ids, checks, chosen)
        for actual, expected in zip([ids, chosen] + list(checks.values()), before):
            torch.testing.assert_close(actual, expected)
        first = audit.summary()
        first["families"]["pregrasp"]["rejection_counts"]["ik"] = -100
        first["families"]["pregrasp"]["candidates"][0]["proposed"] = -100
        self.assertEqual(audit.summary()["families"]["pregrasp"]["rejection_counts"]["ik"], 2)
        self.assertEqual(audit.summary()["families"]["pregrasp"]["candidates"][0]["proposed"], 3)
        bank = self.bank()
        snapshot = {key: value.clone() for key, value in bank.items()}
        report = audit.summary(bank)
        json.dumps(report, allow_nan=False)
        for name, value in bank.items():
            torch.testing.assert_close(value, snapshot[name])

    def test_invalid_record_is_atomic(self):
        audit = BuildAudit(FAMILIES)
        audit.record(1, *self.proposals())
        before = deepcopy(audit.summary())
        ids, checks, _ = self.proposals()
        for indices in ([1], [0, 0], [6], [-1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                audit.record(1, ids, checks, self.vector(indices))
            self.assertEqual(audit.summary(), before)
        with self.assertRaises(ValueError):
            audit.record(1, ids, dict(reversed(list(checks.items()))), self.vector([0]))
        self.assertEqual(audit.summary(), before)

    def test_candidate_ids_and_checks_reject_ambiguous_values(self):
        for candidate in ([1.5], [float("nan")], [-2.]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                BuildAudit(FAMILIES).record(0, self.vector(candidate, torch.float32),
                                           {"ok": self.vector([1], torch.bool)}, self.vector([]))
        with self.assertRaises(ValueError):
            BuildAudit(FAMILIES).record(1, self.vector([1]),
                                       {"ok": self.vector([1.], torch.float32)}, self.vector([]))
        # Snapshot candidate indices are floats, but still exact integers.
        audit = BuildAudit(FAMILIES)
        audit.record(1, self.vector([12.], torch.float32),
                     {"ok": self.vector([1], torch.bool)}, self.vector([0]))
        self.assertEqual(audit.summary()["families"]["pregrasp"]["stored"], 1)

    def test_coverage_uses_actual_rows_and_signed_relative_geometry(self):
        report = BuildAudit(FAMILIES).summary(self.bank())
        near = report["families"]["pregrasp"]
        coverage = near["coverage"]
        self.assertEqual(near["bank_count"], 2)
        self.assertEqual(near["stored_candidate_counts"], [{"candidate_id": 12, "count": 2}])
        self.assertEqual(near["stored_candidate_count"], 1)
        self.assertAlmostEqual(coverage["base_x_m"]["min"], .8, places=5)
        self.assertAlmostEqual(coverage["base_y_m"]["min"], -.1, places=5)
        self.assertAlmostEqual(coverage["base_heading_offset_deg"]["min"], -10., places=4)
        self.assertAlmostEqual(coverage["base_heading_offset_deg"]["max"], 10., places=4)
        self.assertAlmostEqual(coverage["ee_candidate_distance_m"]["min"], math.sqrt(.08**2 + .01**2), places=6)
        self.assertAlmostEqual(coverage["candidate_local_y_m"]["min"], -.01, places=6)
        self.assertNotIn("ee_candidate_distance_m", report["families"]["full_task"]["coverage"])
        self.assertEqual(report["families"]["full_task"]["stored_candidate_counts"], [])

    def test_constant_empty_and_nonfinite_histograms_are_json_safe(self):
        report = BuildAudit(FAMILIES).summary(self.bank(include_full=False))
        json.dumps(report, allow_nan=False)
        empty = report["families"]["full_task"]["coverage"]["table_height_m"]
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["min"])
        self.assertEqual(empty["histogram"]["counts"], [])
        fixed = report["families"]["pregrasp"]["coverage"]["gripper_opening"]
        self.assertEqual(fixed["min"], .75)
        self.assertEqual(fixed["max"], .75)
        self.assertEqual(len(fixed["histogram"]["edges"]), 11)
        self.assertEqual(len(fixed["histogram"]["counts"]), 10)
        self.assertEqual(sum(fixed["histogram"]["counts"]), 2)
        self.assertEqual(fixed["histogram"]["range"], "constant_value_display_interval")
        values = self.vector([0., 1., float("nan"), float("inf")], torch.float32)
        stats = MODULE._distribution(values)
        self.assertEqual(stats["nonfinite_count"], 2)
        self.assertEqual(sum(stats["histogram"]["counts"]), 2)
        json.dumps(stats, allow_nan=False)

    def test_four_family_accounting_and_near_goal_evidence_coverage(self):
        bank = {name: torch.cat((value, value[-1:].clone()))
                for name, value in self.bank().items()}
        bank["family_id"][-1] = 3
        bank["grasp_candidate_index"][-1] = 7
        expected = {
            "lift_height": .075,
            "near_goal_relative_slip_m": .003,
            "near_goal_relative_rotation_deg": 2.,
            "near_goal_contact_fraction": .95,
            "near_goal_object_speed_m_s": .02,
            "near_goal_restore_verified": 1.,
        }
        for name, value in expected.items():
            bank[name] = self.vector([0., 0., 0., 0., value], torch.float32)
        audit = BuildAudit(FAMILIES + ("near_goal",))
        audit.record(3, self.vector([7, 8]),
                     {"held": self.vector([True, False], torch.bool)}, self.vector([0]))
        report = audit.summary(bank)
        near = report["families"]["near_goal"]
        self.assertEqual((near["proposed"], near["valid"], near["stored"]), (2, 1, 1))
        self.assertEqual(near["stored_candidate_counts"], [{"candidate_id": 7, "count": 1}])
        for name, value in expected.items():
            metric = near["coverage"][name]
            self.assertAlmostEqual(metric["min"], value, places=6)
            self.assertAlmostEqual(metric["max"], value, places=6)
            self.assertEqual(sum(metric["histogram"]["counts"]), 1)
        self.assertIn("lift_height", report["families"]["at_grasp"]["coverage"])
        self.assertNotIn("near_goal_restore_verified", report["families"]["at_grasp"]["coverage"])
        json.dumps(report, allow_nan=False)

    def test_legacy_bank_omits_unavailable_grasp_evidence_instead_of_inventing_it(self):
        report = BuildAudit(FAMILIES).summary(self.bank())
        for name in FAMILIES:
            self.assertNotIn("lift_height", report["families"][name]["coverage"])
            self.assertNotIn("near_goal_contact_fraction", report["families"][name]["coverage"])

    def test_optional_grasp_evidence_requires_one_value_per_configuration(self):
        bank = self.bank()
        bank["near_goal_relative_slip_m"] = torch.zeros(4, 1, device=self.device)
        with self.assertRaisesRegex(ValueError, "invalid coverage tensor shape"):
            BuildAudit(FAMILIES).summary(bank)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class BuildAuditCudaTests(BuildAuditTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
