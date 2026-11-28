"""Production replay and reset-warm-up coverage for world diagnostics."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from modules.pose_prediction import RollingSE3Predictor
from utils.ekf_world_diagnostic import evaluate_world_capture


def poses(times, world=False):
    times = np.asarray(times)
    result = np.zeros(times.shape + (7,), dtype=np.float32)
    result[..., 0] = ((2. + .2 * times) if world else (.4 - .1 * times)) + .04 * times**2
    result[..., 1] = .2
    result[..., 5] = np.sin(.1 * times)
    result[..., 6] = np.cos(.1 * times)
    return result


class WorldDiagnosticTests(unittest.TestCase):
    @staticmethod
    def capture(q_scale=1., r_scale=.01):
        times = np.arange(11) * .2
        horizons = np.array([.3, .6])
        predictor = RollingSE3Predictor(1, "cpu", history=5, horizons=tuple(horizons), q_scale=q_scale, r_scale=r_scale)
        predictions, valid, histories, history_times = [], [], [], []
        for time, pose in zip(times, poses(times)):
            predictor.push(torch.from_numpy(pose[None]), float(time))
            prediction, is_valid = predictor.predict()
            predictions.append(prediction.numpy().copy())
            valid.append(is_valid.numpy().copy())
            histories.append(predictor.poses.numpy().copy())
            history_times.append(predictor.times.numpy().copy())
        # Dense truth starts after the initial/reset sample. The first valid
        # online fit therefore has no complete world history and must be skipped.
        dense_times = np.arange(1, 21) * .1
        return dict(times=times, horizons=horizons,
            episode_ids=np.zeros((len(times), 1), dtype=np.int64),
            relative_poses=poses(times)[:, None], world_poses=poses(times, True)[:, None],
            predictions=np.asarray(predictions), valid=np.asarray(valid),
            history_poses=np.asarray(histories), history_times=np.asarray(history_times),
            lifted=np.zeros((len(times), 1), dtype=bool),
            driver_phase=np.zeros((len(times), 1), dtype=np.int64),
            dense_truth_times=dense_times,
            dense_truth_episode_ids=np.zeros((len(dense_times), 1), dtype=np.int64),
            dense_truth_relative_poses=poses(dense_times)[:, None],
            dense_truth_world_poses=poses(dense_times, True)[:, None])

    def test_replay_matches_production_and_world_excludes_uncovered_history(self):
        capture = self.capture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            result = evaluate_world_capture(directory, directory / "world", device="cpu")
            self.assertTrue(result["replay_summary"]["matches_online_within_tolerance"])
            self.assertEqual(result["replay_summary"]["origins"], 7)
            evaluation = result["evaluation"]
            self.assertEqual(evaluation["summary"]["world_history_uncovered_origins"], 1)
            self.assertEqual(evaluation["summary"]["pose_frame"], "world coordinates")
            self.assertEqual({row["method"] for row in evaluation["rows"]},
                             {"world_ekf", "hold", "cv_last2", "cv_window5"})
            with np.load(evaluation["raw_errors_path"], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["origin_indices"], [5, 6, 7])
                self.assertEqual(raw["methods"][0], "world_ekf")
                self.assertGreater(raw["target_poses"][..., 0].min(), 2.)
            self.assertTrue(Path(result["replay_errors_path"]).is_file())
            self.assertEqual({row["method"] for row in evaluation["event_rows"]},
                             {"world_ekf", "hold", "cv_last2", "cv_window5"})

    def test_replay_uses_captured_noise_settings(self):
        settings = dict(history=5, q_scale=3., r_scale=.2)
        capture = self.capture(settings["q_scale"], settings["r_scale"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            wrong_settings = evaluate_world_capture(directory, directory / "wrong", device="cpu")
            self.assertFalse(wrong_settings["replay_summary"]["matches_online_within_tolerance"])
            (directory / "metadata.json").write_text(json.dumps(dict(ekf=settings)))
            result = evaluate_world_capture(directory, directory / "world", device="cpu")
            self.assertTrue(result["replay_summary"]["matches_online_within_tolerance"])
            self.assertEqual(result["replay_summary"]["settings"], settings)


if __name__ == "__main__":
    unittest.main()
