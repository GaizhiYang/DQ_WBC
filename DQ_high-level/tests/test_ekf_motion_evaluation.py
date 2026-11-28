"""Small CPU regressions for causal, reset-safe EKF capture scoring."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.ekf_motion_evaluation import causal_baselines, evaluate_capture, interpolate_poses


def trajectory(times, offset=0.):
    times = np.asarray(times, dtype=np.float64)
    position = np.array([offset, 0.1, -0.2]) + times[..., None] * np.array([0.2, 0.1, -0.1])
    rotation = (Rotation.from_euler("xyz", [0.3, -0.2, 0.4])
                * Rotation.from_rotvec(times.reshape(-1, 1) * np.array([0.1, -0.2, 0.3])))
    return np.concatenate((position, rotation.as_quat().reshape(times.shape + (4,))), axis=-1)


class CaptureEvaluationTests(unittest.TestCase):
    def assert_pose_equal(self, actual, expected):
        np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=2e-12)
        angles = (Rotation.from_quat(actual[..., 3:].reshape(-1, 4)).inv()
                  * Rotation.from_quat(expected[..., 3:].reshape(-1, 4))).magnitude()
        np.testing.assert_allclose(angles, 0., atol=2e-12)

    def test_causal_baselines_fit_nonuniform_translation_and_body_rotation(self):
        times = np.array([0., 0.12, 0.29, 0.44, 0.61])
        poses = trajectory(times)
        poses[1::2, 3:] *= -1.  # Quaternion sign must not change the fit.
        horizons = np.array([0.3, 0.6, 0.9])
        predictions = causal_baselines(poses, times, horizons)
        self.assert_pose_equal(predictions[0], np.repeat(poses[-1:], len(horizons), axis=0))
        expected = trajectory(times[-1] + horizons)
        self.assert_pose_equal(predictions[1], expected)
        self.assert_pose_equal(predictions[2], expected)

    def test_acceleration_exposes_window_lag_without_future_information(self):
        times = np.arange(5) * .1
        poses = np.zeros((5, 7))
        poses[:, 0] = .5 * times**2
        poses[:, 6] = 1.
        predictions = causal_baselines(poses, times, [.3])
        # x(t)=0.5*t^2 gives analytic future x(0.7)=0.245. A five-sample
        # slope averages older, slower velocities and lags more than last-two.
        errors = .245 - predictions[:, 0, 0]
        np.testing.assert_allclose(errors, [.165, .06, .105], atol=1e-12)

    def test_ground_truth_slerp_interpolates_and_refuses_extrapolation(self):
        times = np.array([0., 0.23, 0.61, 1.1])
        targets = np.array([[0.1, 0.8], [0.61, 0.]])
        self.assert_pose_equal(interpolate_poses(times, trajectory(times), targets), trajectory(targets))
        for invalid in (-0.001, 1.101):
            with self.assertRaisesRegex(ValueError, "extrapolation"):
                interpolate_poses(times, trajectory(times), [invalid])

    def test_ground_truth_accepts_roundoff_at_episode_end(self):
        times = np.array([0., .1, .3])
        self.assert_pose_equal(interpolate_poses(times, trajectory(times), [.1 + .2]), trajectory([.3]))

    @staticmethod
    def capture():
        length, history = 18, 5
        times = np.arange(length) * 0.25
        episodes = (np.arange(length) // 9)[:, None]
        poses = np.empty((length, 1, 7))
        predictions = np.empty((length, 1, 2, 7))
        history_poses = np.empty((length, 1, history, 7))
        history_times = np.empty((length, 1, history))
        valid = np.zeros((length, 1), dtype=bool)
        horizons = np.array([0.3, 0.6])
        for index in range(length):
            episode_start = 9 * (index // 9)
            local_time = times[index] - times[episode_start]
            offset = 100. * (index // 9)  # Deliberate reset discontinuity.
            poses[index, 0] = trajectory(local_time, offset)
            predictions[index, 0] = trajectory(local_time + horizons, offset)
            indices = np.maximum(np.arange(index - history + 1, index + 1), episode_start)
            history_times[index, 0] = times[indices]
            history_poses[index, 0] = trajectory(times[indices] - times[episode_start], offset)
            valid[index, 0] = index - episode_start >= history - 1
        lifted = np.zeros((length, 1), dtype=bool)
        lifted[[4, 7], 0] = True
        phase = np.zeros((length, 1), dtype=np.int64)
        phase[5:7, 0] = 1
        phase[7:9, 0] = 2
        phase[14:, 0] = 2
        return dict(times=times, episode_ids=episodes, relative_poses=poses,
                    world_poses=trajectory(times, -500.)[:, None], predictions=predictions,
                    valid=valid, history_poses=history_poses, history_times=history_times,
                    driver_phase=phase, lifted=lifted, horizons=horizons)

    def test_shared_origins_are_relative_causal_and_never_cross_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            capture = self.capture()
            np.savez(directory / "capture.npz", **capture)
            (directory / "metadata.json").write_text(json.dumps({"level": "Level03", "seed": 1}))
            result = evaluate_capture(directory, directory / "evaluation")
            with np.load(result["raw_errors_path"], allow_pickle=False) as data:
                raw = {key: data[key] for key in data.files}
            # Every method/horizon uses these exact origins. In particular,
            # origins near the 100 m reset jump cannot borrow the next episode.
            np.testing.assert_array_equal(raw["origin_indices"], [5, 13, 14])
            np.testing.assert_array_equal(raw["episode_ids"], [0, 1, 1])
            self.assertEqual(raw["position_error_m"].shape, (3, 4, 2))
            np.testing.assert_allclose(raw["position_error_m"][:, [0, 2, 3]], 0., atol=2e-12)
            np.testing.assert_allclose(raw["rotation_error_deg"][:, [0, 2, 3]], 0., atol=2e-12)
            hold_error = np.linalg.norm([0.2, 0.1, -0.1]) * capture["horizons"]
            np.testing.assert_allclose(raw["position_error_m"][:, 1], np.tile(hold_error, (3, 1)))
            np.testing.assert_array_equal(raw["future_phase_changed"][0], [False, True])
            np.testing.assert_array_equal(raw["future_lifted"][0], [False, True])
            self.assertEqual(result["summary"]["evaluated_origins"], 3)
            self.assertEqual(result["summary"]["evaluated_episodes"], 2)
            self.assertEqual(result["summary"]["metadata"]["level"], "Level03")
            self.assertEqual({row["scope"] for row in result["rows"]}, {"all", "drive", "brake", "pause"})
            self.assertEqual(len(result["episode_rows"]), 2 * 4 * 2)
            for name in ("metrics.csv", "episode_metrics.csv", "event_metrics.csv", "summary.json", "raw_errors.npz"):
                self.assertTrue((directory / "evaluation" / name).is_file())
            changed_rows = [row for row in result["event_rows"] if row["scope"] == "phase_changed"]
            self.assertTrue(changed_rows)
            for row in changed_rows:
                self.assertEqual(row["samples"], {.3: 1, .6: 2}[row["horizon_s"]])

            # Changing only future truth changes errors, never the forecasts:
            # all baselines must consume the captured predictor history.
            capture["relative_poses"][6, 0, 0] += 1.
            np.savez(directory / "capture.npz", **capture)
            changed = evaluate_capture(directory, directory / "changed_truth")
            with np.load(changed["raw_errors_path"], allow_pickle=False) as altered:
                np.testing.assert_array_equal(altered["predictions"], raw["predictions"])
                self.assertGreater(altered["position_error_m"][0, 0, 0], 0.7)

    def test_reset_contaminated_history_is_excluded_even_when_marked_valid(self):
        capture = self.capture()
        # A plausible increasing five-sample history that leaks past the
        # observed terminal time of the previous episode must be rejected.
        capture["history_times"][13, 0] = [1.75, 2.25, 2.5, 3., 3.25]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            result = evaluate_capture(directory, directory / "evaluation")
            self.assertEqual(result["summary"]["exclusions"]["invalid_history"], 1)
            with np.load(result["raw_errors_path"], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["origin_indices"], [5, 14])

    def test_dense_events_detect_transition_between_high_level_samples(self):
        capture = self.capture()
        dense_times = np.arange(0., 4.251, .125)
        dense_episodes = (dense_times >= 2.25).astype(np.int64)[:, None]
        dense_poses = trajectory(dense_times)
        second = dense_times >= 2.25
        dense_poses[second] = trajectory(dense_times[second] - 2.25, 100.)
        # At 1.375 s the driver briefly changes and comes back before the
        # next high-level sample at 1.5 s. Only dense metadata sees the event.
        dense_phase = capture["driver_phase"][np.minimum((dense_times / .25).astype(int), 17)].copy()
        dense_phase[np.isclose(dense_times, 1.375), 0] = 2
        dense_lifted = np.zeros_like(dense_episodes, dtype=bool)
        dense_lifted[np.isclose(dense_times, 1.375), 0] = True
        capture.update(dense_truth_times=dense_times, dense_truth_episode_ids=dense_episodes,
                       dense_truth_relative_poses=dense_poses[:, None],
                       dense_truth_driver_phase=dense_phase, dense_truth_lifted=dense_lifted)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            result = evaluate_capture(directory, directory / "evaluation")
            with np.load(result["raw_errors_path"], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["future_phase_changed"][0], [True, True])
                np.testing.assert_array_equal(raw["future_lifted"][0], [True, True])
            self.assertEqual(result["summary"]["event_sampling"]["driver_phase"],
                             "dense truth plus high-level samples")

    def test_speed_change_labels_use_past_same_episode_commands_only(self):
        capture = self.capture()
        times = capture["times"]
        commands = np.zeros((len(times), 1, 6))
        # The first episode accelerates in reverse. The second slows while
        # moving forwards, then accelerates. Magnitude, not signed speed,
        # determines the labels in both directions.
        commands[:9, 0, 0] = -.1 * times[:9]
        commands[9:, 0, 0] = .4 - .1 * (times[9:] - times[9])
        commands[14:, 0, 0] = commands[13, 0, 0] + .1 * (times[14:] - times[13])
        capture.update(dense_truth_times=times, dense_truth_episode_ids=capture["episode_ids"],
                       dense_truth_relative_poses=capture["relative_poses"],
                       dense_truth_table_commands=commands)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            result = evaluate_capture(directory, directory / "evaluation")
            with np.load(result["raw_errors_path"], allow_pickle=False) as raw:
                np.testing.assert_allclose(raw["origin_speed_change_m_s2"], [.1, -.1, .1], atol=1e-12)
            rows = [row for row in result["event_rows"] if row["scope"] in ("speeding_up", "slowing_down")]
            self.assertEqual(len(rows), 2 * 4 * 2)
            for row in rows:
                self.assertEqual(row["samples"], 2 if row["scope"] == "speeding_up" else 1)

    def test_dense_truth_supplies_targets_and_episode_coverage(self):
        capture = self.capture()
        # The second dense episode deliberately ends before the high-level
        # capture, so its available truth must govern forecast eligibility.
        first_times = np.arange(0., 2.001, 0.125)
        second_times = np.arange(2.25, 3.876, 0.125)
        dense_times = np.r_[first_times, second_times]
        dense_poses = np.concatenate((trajectory(first_times), trajectory(second_times - 2.25, 100.)))[:, None]
        dense_poses[np.flatnonzero(dense_times == 1.625), 0, 0] += 0.5
        capture.update(dense_truth_times=dense_times,
                       dense_truth_episode_ids=np.r_[np.zeros(len(first_times)), np.ones(len(second_times))].astype(np.int64)[:, None],
                       dense_truth_relative_poses=dense_poses)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "capture.npz", **capture)
            result = evaluate_capture(directory, directory / "evaluation")
            self.assertEqual(result["summary"]["truth_source"], "dense_truth_relative_poses")
            with np.load(result["raw_errors_path"], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["origin_indices"], [5, 13])
                # t = 1.55 interpolates the dense 1.625 s perturbation, which
                # is absent from the high-level samples and all histories.
                self.assertAlmostEqual(raw["position_error_m"][0, 0, 0], 0.2)
                np.testing.assert_allclose(raw["position_error_m"][1, [0, 2, 3]], 0., atol=2e-12)


if __name__ == "__main__":
    unittest.main()
