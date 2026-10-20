"""Regression checks for reset boundaries, timestamps, and pooled metrics."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pose_trajectory import PoseTrajectoryRecorder
from evaluate_object_pose_prediction import evaluate_recordings, metric_rows


class TrajectoryTests(unittest.TestCase):
    def test_asynchronous_resets_preserve_terminal_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "trajectories"
            recorder = PoseTrajectoryRecorder(output, 2)
            recorder.append(.14, .14, [[1,0,0,0,0,0,2], [10,0,0,0,0,0,1]], [False,False], [False,False], 0)
            recorder.append(.30, .16, [[2,0,0,0,0,0,1], [11,0,0,0,0,0,1]], [True,False], [True,False], 1)
            recorder.append(.44, .14, [[100,0,0,0,0,0,1], [12,0,0,0,0,0,1]], [False,False], [False,False], 2)
            recorder.close()
            files = sorted(output.glob("*.csv"))
            self.assertEqual(len(files), 3)
            with files[0].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["x"]) for row in rows], [1,2])
            self.assertEqual(rows[-1]["terminated"], "1")
            self.assertAlmostEqual(float(rows[-1]["timestamp"]), .16)
            self.assertEqual(float(rows[0]["qw"]), 1.)
            with files[1].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["timestamp"]), 0.)
            self.assertEqual(float(rows[0]["x"]), 100.)
            with files[2].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(float(rows[-1]["timestamp"]), .30)
            self.assertEqual(len(json.loads((output / "episodes.json").read_text())), 3)

    def test_reject_duplicate_simulation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = PoseTrajectoryRecorder(Path(tmp) / "trajectories", 1)
            try:
                recorder.append(.14, .14, [[0,0,0,0,0,0,1]], [False], [False], 0)
                with self.assertRaisesRegex(ValueError, "strictly increase"):
                    recorder.append(.14, .14, [[0,0,0,0,0,0,1]], [False], [False], 1)
            finally:
                recorder.close()

    def test_metrics_use_forecast_samples_not_mean_of_episode_means(self):
        # Two episodes with 1 and 2 origins: pooled mean = (1+3+3)/3 cm.
        errors = np.array([.01, .03, .03]).reshape(3,1,1)
        rows = metric_rows(errors, errors*100, SimpleNamespace(steps=[1], prediction_dt=.15), ["ekf"])
        self.assertAlmostEqual(rows[0]["position_mean_cm"], 7/3)
        self.assertAlmostEqual(rows[0]["position_rmse_cm"], np.sqrt(19/3))
        self.assertEqual(rows[0]["samples"], 3)

    def test_offline_replay_keeps_large_reset_jump_out_of_forecasts(self):
        paparazzo = Path.home() / "Paparazzo"
        if not (paparazzo / "scripts/evaluate_pose_prediction.py").exists():
            self.skipTest("Local Paparazzo benchmark not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = PoseTrajectoryRecorder(root / "trajectories", 1)
            clock = 0.
            for episode in range(2):
                for step in range(30):
                    dt = .14 if step % 2 else .16
                    clock += dt
                    pose = [100*episode + .2*clock, 0, .7, 0, 0, 0, 1]
                    recorder.append(clock, dt, [pose], [step == 29], [False], episode*30+step)
            recorder.close()
            args = SimpleNamespace(paparazzo_dir=paparazzo, history=8, prediction_dt=.15,
                steps=[1,2], stride=1, velocity_window=5, pos_noise=0., rot_noise_deg=0.,
                q_scale=1., r_scale=1., max_plots=0, pre_lift_only=False, seed=43)
            rows = evaluate_recordings(root, root / "evaluation", args)
            cv = [row for row in rows if row["method"] == "constant_velocity"]
            self.assertTrue(all(row["position_mean_cm"] < 1e-8 for row in cv))
            summary = json.loads((root / "evaluation/summary.json").read_text())
            self.assertEqual(summary["evaluated_episodes"], 2)
            self.assertFalse(summary["skipped"])
            for path in (root / "evaluation/predictions").glob("*.npz"):
                with np.load(path) as data:
                    self.assertEqual(data["predictions"].shape[1:], (3,2,4,4))
                    self.assertEqual(data["predicted_poses"].shape[1:], (3,2,7))
                    np.testing.assert_allclose(data["predicted_poses"][..., :3], data["predictions"][..., :3, 3])
                    np.testing.assert_allclose(np.linalg.norm(data["predicted_poses"][..., 3:], axis=-1), 1.)


if __name__ == "__main__":
    unittest.main()
