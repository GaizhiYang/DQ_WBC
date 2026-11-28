"""Regression coverage for episode-level uncertainty and pooled capture metadata."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_online_ekf_check import _complete_capture
from summarize_online_ekf import paired_episode_bootstrap, summarize


class SummaryTests(unittest.TestCase):
    def test_bootstrap_pairs_methods_and_weights_whole_episodes(self):
        # Repeating overlapping origins in episode 0 must not give it 100x
        # the weight of episode 1 in the episode-level effect or interval.
        position = np.zeros((101, 4, 1))
        position[:100, 0] = 1
        position[:100, 1:] = 2
        position[100, 0] = 5
        position[100, 1:] = 2
        keys = np.zeros((101, 3), dtype=np.int64)
        keys[-1, -1] = 1
        rows = paired_episode_bootstrap(position, position * 2, keys,
            ["online_ekf", "hold", "cv_last2", "cv_window5"], [.6], samples=200)
        self.assertEqual(len(rows), 6)
        for row in rows:
            factor = 1 if row["metric"] == "position_cm" else 2
            self.assertEqual(row["episodes"], 2)
            self.assertAlmostEqual(row["mean_difference"], factor)
            self.assertAlmostEqual(row["ci95_lower"], -factor)
            self.assertAlmostEqual(row["ci95_upper"], 3 * factor)
        reduced = paired_episode_bootstrap(position[[0, -1]], position[[0, -1]] * 2,
            keys[[0, -1]], ["online_ekf", "hold", "cv_last2", "cv_window5"], [.6], samples=200)
        self.assertEqual(rows, reduced)

    def test_capture_identity_keeps_episode_zero_distinct(self):
        errors = np.zeros((2, 4, 1))
        errors[:, 1:] = 1
        keys = np.array([[0, 0, 0], [1, 0, 0]])
        rows = paired_episode_bootstrap(errors, errors, keys,
            ["online_ekf", "hold", "cv_last2", "cv_window5"], [.8], samples=10)
        self.assertTrue(all(row["episodes"] == 2 for row in rows))
        self.assertTrue(all(row["ci95_upper"] == -1 for row in rows))

    @staticmethod
    def make_run(root, level, seed):
        directory = root / (level.lower() + "_seed" + str(seed))
        directory.mkdir()
        metadata = dict(error=None, requested_steps=20, completed_steps=20,
                        motion_level=level, seed=seed, num_envs=2)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        np.savez(directory / "capture.npz", dense_truth_times=np.array([0., .025, .05]),
                 dense_truth_episode_ids=np.zeros((3, 2), dtype=np.int64))
        errors = np.ones((2, 4, 2)) * seed / 100.
        for frame in ("relative_evaluation", "world_diagnostic/evaluation"):
            output = directory / frame
            output.mkdir(parents=True)
            methods = ["online_ekf", "hold", "cv_last2", "cv_window5"]
            if frame.startswith("world"):
                methods[0] = "world_ekf"
            np.savez(output / "raw_errors.npz", methods=np.asarray(methods), horizons=[.2, .8],
                position_error_m=errors, rotation_error_deg=errors * 100,
                origin_speed_change_m_s2=np.array([.1, -.1]),
                current_phase=np.array([0, 1]), future_phase_changed=np.array([[False, True], [False, False]]),
                env_ids=np.array([0, 1]), episode_ids=np.array([0, 0]), origin_times=np.array([.025, .05]))
        (directory / "world_diagnostic/online_replay.json").write_text(
            json.dumps(dict(matches_online_within_tolerance=True)))
        return directory

    def test_summary_uses_actual_metadata_and_reports_per_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for level in ("Level02", "Level03"):
                for seed in (11, 19):
                    self.make_run(root, level, seed)
            result = summarize(root, bootstrap_samples=20, plot=False)
            self.assertEqual(result["counts"]["level03"]["seeds"], [11, 19])
            self.assertEqual(result["counts"]["level03"]["environments_per_capture"], [2, 2])
            self.assertEqual(result["counts"]["level03"]["evaluated_episodes"], 4)
            self.assertEqual(result["truth_sample_interval_s"]["median"], .025)
            self.assertTrue(any(row["scope"] == "phase_changes_within_0.8s" for row in result["metrics"]))
            self.assertFalse(any("0.9" in row["scope"] for row in result["metrics"]))
            transition_rows = [row for row in result["metrics"]
                               if row["scope"] == "phase_change_within_horizon"]
            self.assertTrue(transition_rows)
            self.assertTrue(all(row["horizon_s"] == .8 for row in transition_rows))
            self.assertTrue(all(row["samples"] == 2 for row in transition_rows))
            self.assertTrue(any(row["scope"] == "speeding_up" for row in result["metrics"]))
            self.assertTrue(any(row["scope"] == "slowing_down" for row in result["metrics"]))
            self.assertEqual(len(result["seed_metrics"]), 2 * 2 * 2 * 4 * 2)
            self.assertEqual({row["seed"] for row in result["seed_metrics"]}, {11, 19})
            self.assertTrue(result["all_replays_match"])
            for name in ("metrics.csv", "seed_metrics.csv", "paired_episode_comparisons.csv", "summary.json"):
                self.assertTrue((root / name).is_file())

    def test_reuse_requires_complete_matching_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_run(Path(temporary), "Level03", 11)
            _complete_capture(directory, "Level03", 11, 2, 20)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                _complete_capture(directory, "Level03", 11, 2, 30)
            metadata = json.loads((directory / "metadata.json").read_text())
            metadata["error"] = "interrupted"
            (directory / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _complete_capture(directory, "Level03", 11, 2, 20)

    def test_selected_seeds_ignore_other_complete_and_incomplete_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for level in ("Level02", "Level03"):
                for seed in (11, 19):
                    self.make_run(root, level, seed)
                # A failed/unrelated run has no metadata and must not enter a
                # summary explicitly restricted to seed 19.
                (root / (level.lower() + "_seed47")).mkdir()
            result = summarize(root, bootstrap_samples=20, plot=False, seeds=[19])
            self.assertEqual(len(result["captures"]), 2)
            self.assertEqual({capture["seed"] for capture in result["captures"]}, {19})
            self.assertEqual({row["seed"] for row in result["seed_metrics"]}, {19})
            for counts in result["counts"].values():
                self.assertEqual(counts["seeds"], [19])
                self.assertEqual(counts["evaluated_origins"], 2)
            with self.assertRaisesRegex(ValueError, "Missing selected capture"):
                summarize(root, bootstrap_samples=20, plot=False, seeds=[23])


if __name__ == "__main__":
    unittest.main()
