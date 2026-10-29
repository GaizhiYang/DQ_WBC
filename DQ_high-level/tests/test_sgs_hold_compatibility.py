"""Compatibility of immutable geometry banks with the aligned teacher task."""

from copy import deepcopy
import unittest

from utils.sgs import validate_task_hold_steps


class HoldCompatibilityTests(unittest.TestCase):
    def metadata(self, steps=15, profile="geometry_grasp_v3", passed=True):
        return {"hold_steps": steps, "physical_validation": {
            "passed": passed, "validation_profile": profile}}

    def test_matching_task_requires_no_migration(self):
        self.assertEqual(validate_task_hold_steps(self.metadata(10), 10), "exact")

    def test_known_alignment_preserves_original_evidence(self):
        metadata = self.metadata()
        original = deepcopy(metadata)
        self.assertEqual(validate_task_hold_steps(metadata, 10), "geometry_bank_15_to_10")
        self.assertEqual(metadata, original)

    def test_unvalidated_or_unsupported_bank_cannot_change_task(self):
        for metadata in (self.metadata(passed=False), self.metadata(profile="unknown"),
                         {"hold_steps": 15}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "hold_steps"):
                validate_task_hold_steps(metadata, 10)

    def test_other_hold_changes_require_explicit_revalidation(self):
        for source, target in ((10, 15), (20, 10), (15, 9), (15, 20)):
            with self.subTest(source=source, target=target), self.assertRaisesRegex(ValueError, "hold_steps"):
                validate_task_hold_steps(self.metadata(source), target)

    def test_boolean_zero_and_noninteger_lengths_are_rejected(self):
        for value in (False, 0, -1, 10.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    validate_task_hold_steps(self.metadata(value), 10)
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    validate_task_hold_steps(self.metadata(), value)


if __name__ == "__main__":
    unittest.main()
