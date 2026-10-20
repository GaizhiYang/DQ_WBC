"""Check episode separation, shared origins, and validation-independent selection."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tune_object_pose_prediction import split_episodes, short_horizon_score, tune
from utils.pose_trajectory import PoseTrajectoryRecorder


class TuningTests(unittest.TestCase):
    def test_split_keeps_simultaneous_envs_on_same_side_of_reset(self):
        paths = [Path("env_{:04d}_episode_{:06d}.csv".format(env, episode))
                 for env in range(3) for episode in range(2)]
        calibration, validation = split_episodes(paths)
        self.assertEqual(len(calibration), 3)
        self.assertEqual(len(validation), 3)
        self.assertTrue(all(path.stem.endswith("000000") for path in calibration))
        self.assertTrue(all(path.stem.endswith("000001") for path in validation))
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_episodes(calibration)

    def test_short_term_objective_ignores_long_term_errors(self):
        score = short_horizon_score(np.array([[.01, .02, 1.]]), np.array([[1., 2., 100.]]),
                                    [.15, .3, .9], .3, 1., 1.)
        self.assertAlmostEqual(score, 3.)

    def test_validation_changes_cannot_select_different_parameters(self):
        paparazzo = Path.home() / "Paparazzo"
        if not (paparazzo / "scripts/evaluate_pose_prediction.py").exists():
            self.skipTest("Local Paparazzo benchmark not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = PoseTrajectoryRecorder(root / "trajectories", 1)
            for episode in range(2):
                for step in range(25):
                    t = .15 * step
                    pose = [.2 * t + .05 * t*t, 0, .6, 0, 0, 0, 1]
                    recorder.append(.15 * (episode*25+step+1), .15, [pose], [step==24], [False], step)
            recorder.close()
            args = SimpleNamespace(input=root, output=root / "first", paparazzo_dir=paparazzo,
                history=5, prediction_dt=.15, steps=[1,2,4], stride=2, velocity_window=5,
                pos_noise=0., rot_noise_deg=0., q_scale=1., r_scale=1., max_plots=0,
                pre_lift_only=False, seed=43, history_grid=[3,8], q_scale_grid=[1.],
                r_scale_grid=[.01], velocity_window_grid=[2,3], optimize_horizon=.3,
                position_scale_cm=1., rotation_scale_deg=1.)
            with contextlib.redirect_stdout(io.StringIO()):
                first = tune(args)
            for path in (args.output / "validation_predictions").glob("*.npz"):
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["origin_indices"], np.arange(7,21,2))
                    self.assertEqual(data["predicted_poses"].shape, (7,5,3,7))
                    np.testing.assert_allclose(np.linalg.norm(data["predicted_poses"][...,3:],axis=-1),1.)
            original_scores = (args.output / "validation_scores.csv").read_text()
            path = root / "trajectories/env_0000_episode_000001.csv"
            with path.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                row["x"] = str(float(row["x"]) * 5)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            args.output = root / "second"
            with contextlib.redirect_stdout(io.StringIO()):
                second = tune(args)
            self.assertEqual(first["selected_parameters"], second["selected_parameters"])
            self.assertNotEqual(original_scores, (args.output / "validation_scores.csv").read_text())


if __name__ == "__main__":
    unittest.main()
