"""Causality, production equivalence and explicit tuning guardrails."""

import unittest

import numpy as np
import torch

from modules.pose_prediction import RollingSE3Predictor
from tune_online_ekf import BASELINE, config_id, predict_case, rank_configs


class OnlineEKFTuningTests(unittest.TestCase):
    def test_short_history_uses_only_causal_suffix_and_matches_production(self):
        times = np.array([[.02, .18, .32, .48, .625]], dtype=np.float64)
        poses = np.zeros((1, 5, 7), dtype=np.float32)
        poses[..., 0] = .3*times + .04*times**2
        poses[..., 1] = .1*times
        poses[..., 5] = np.sin(.1*times)
        poses[..., 6] = np.cos(.1*times)
        case = dict(histories=poses, history_times=times, horizons=np.array([.3, .6, .9]))
        config = dict(history=3, q_scale=1., r_scale=.1)
        actual = predict_case(case, config, "cpu")
        production = RollingSE3Predictor(1, "cpu", horizons=(.3, .6, .9), **config)
        for i in range(2, 5):
            production.push(torch.from_numpy(poses[:, i]), float(times[0, i]))
            production.predict()
        np.testing.assert_allclose(actual, production.future_poses.numpy(), atol=2e-6)
        altered = dict(case, histories=poses.copy(), targets=np.ones((1, 3, 7)))
        altered["histories"][:, :2, :3] += 100.
        # Discarded history and future truth must not enter the fit.
        np.testing.assert_array_equal(predict_case(altered, config, "cpu"), actual)

    def test_rotation_tail_guardrail_is_explicit_and_changes_ranking(self):
        rows = []
        for config, mean, rotation_p95 in ((BASELINE, 10., 5.),
                (dict(history=3, q_scale=1., r_scale=.1), 8., 6.),
                (dict(history=4, q_scale=10., r_scale=.0001), 9., 5.1)):
            for horizon in (.6, .9):
                for scope in ("all", "brake"):
                    rows.append(dict(config_id=config_id(config), **config, scope=scope,
                        horizon_s=horizon, position_mean_cm=mean, position_p95_cm=20.,
                        rotation_mean_deg=1., rotation_p95_deg=rotation_p95))
        self.assertEqual(rank_configs(rows)[0]["configuration"]["history"], 3)
        stricter = rank_configs(rows, rotation_p95_guardrail=True)
        self.assertEqual(stricter[0]["configuration"]["history"], 4)
        aggressive = next(row for row in stricter if row["configuration"]["history"] == 3)
        self.assertFalse(aggressive["guardrails_pass"])
        self.assertIn("rotation_p95_deg@0.6", aggressive["guardrail_failures"])


if __name__ == "__main__":
    unittest.main()
