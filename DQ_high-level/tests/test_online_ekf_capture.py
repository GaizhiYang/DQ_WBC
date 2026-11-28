"""Reset-boundary capture checks without loading Isaac Gym."""

import unittest

import numpy as np

from collect_online_ekf import append_truth


class DenseTruthCaptureTests(unittest.TestCase):
    def test_reset_replaces_terminal_sample_without_duplicate_time(self):
        dense = {"times": [], "episode_ids": [], "poses": []}
        append_truth(dense, dict(times=.98, episode_ids=np.array([0, 0]), poses=np.array([1., 2.])))
        append_truth(dense, dict(times=1., episode_ids=np.array([0, 0]), poses=np.array([1.1, 2.1])))
        # Only environment zero resets; environment one keeps its exact truth.
        append_truth(dense, dict(times=1., episode_ids=np.array([1, 0]), poses=np.array([100., 2.1])))
        append_truth(dense, dict(times=1.02, episode_ids=np.array([1, 0]), poses=np.array([100.1, 2.2])))
        np.testing.assert_allclose(dense["times"], [.98, 1., 1.02])
        np.testing.assert_array_equal(np.asarray(dense["episode_ids"])[:, 0], [0, 1, 1])
        np.testing.assert_allclose(np.asarray(dense["poses"])[:, 1], [2., 2.1, 2.2])
        self.assertEqual(dense["poses"][1][0], 100.)

    def test_backward_time_is_rejected(self):
        dense = {"times": []}
        append_truth(dense, dict(times=1.))
        with self.assertRaisesRegex(ValueError, "backwards"):
            append_truth(dense, dict(times=.9))


if __name__ == "__main__":
    unittest.main()
