"""Numerical/reference, coordinate-frame and episode-isolation regression tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    # Isaac Gym requires this import before torch, even in a CPU-only test.
    from isaacgym import gymapi
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Run these tests in the dqwbc environment: {}".format(exc))

from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, quat_conjugate, quat_mul, euler_from_quat
from modules.kalman_filter import SE3KF, ekf_predict_horizon
from modules.pose_prediction import RollingSE3Predictor, poses_to_transforms, se3_exp_batch, se3_log_batch
from utils.object_pose import object_pose_in_teacher_frame
from utils.config import get_params, apply_pose_prediction_config


def reference(poses, times):
    transforms = np.tile(np.eye(4), (len(poses), 1, 1))
    transforms[:, :3, 3] = poses[:, :3]
    transforms[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    kf = SE3KF(T0=transforms[0])
    kf.R = kf.R * .01
    for i in range(1, len(poses)):
        kf.predict(times[i] - times[i-1])
        kf.update_with_absolute_pose(transforms[i])
    return np.stack(ekf_predict_horizon(kf, N=4, dt=.15))[[1, 3]]


class FuturePoseTests(unittest.TestCase):
    def check_reference(self, device, dtype, tolerance):
        times = np.array([10000., 10000.14, 10000.30, 10000.445, 10000.605])
        t = times - times[0]
        histories = []
        for env in range(3):
            pos = np.column_stack((env*100 + .2*t + .05*t*t, -.1*t, .6 + .002*np.sin(t)))
            q = Rotation.from_euler("xyz", np.column_stack((.3+.1*t, -.2+.04*t*t, env + .4*t))).as_quat()
            q[1::2] *= -1  # q and -q represent the same orientation.
            histories.append(np.column_stack((pos, q)))
        predictor = RollingSE3Predictor(3, device, dtype=dtype)
        for i, timestamp in enumerate(times):
            predictor.push(torch.tensor(np.stack(histories)[:, i], device=device, dtype=dtype), timestamp)
        predicted, valid = predictor.predict()
        self.assertTrue(valid.all())
        actual = poses_to_transforms(predicted.reshape(-1, 7)).reshape(3, 2, 4, 4).cpu().numpy()
        expected = np.stack([reference(poses, times) for poses in histories])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
        # Forecasting must not advance or otherwise mutate the measured history.
        before = predictor.poses.clone()
        predictor.predict()
        torch.testing.assert_close(predictor.poses, before)

    def test_matches_numpy_reference_variable_dt_float64(self):
        self.check_reference("cpu", torch.float64, 1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_matches_numpy_reference_on_gpu_float32(self):
        self.check_reference("cuda:0", torch.float32, 1e-4)

    def test_reset_warmup_duplicate_sample_and_independent_environments(self):
        predictor = RollingSE3Predictor(2, "cpu", dtype=torch.float64)
        for i in range(5):
            pose = torch.tensor([[.03*i, 0., .6, 0., 0., 0., 1.], [10+.03*i, 0., .7, 0., 0., 0., 1.]], dtype=torch.float64)
            predictor.push(pose, .15*i)
            future, valid = predictor.predict()
            if i < 4:
                self.assertFalse(valid.any())
                torch.testing.assert_close(future, pose[:, None, :].expand(-1, 2, -1))
        count = predictor.count.clone()
        predictor.push(pose + 100, .6)  # Repeated timestamp must not become a new sample.
        torch.testing.assert_close(predictor.count, count)
        unaffected = predictor.poses[1].clone()
        predictor.reset(torch.tensor([0]))
        reset_pose = torch.tensor([[1000., 0., .65, 0., 0., 0., 1.]], dtype=torch.float64)
        predictor.push(reset_pose, .605, torch.tensor([0]))
        future, valid = predictor.predict()
        self.assertEqual(predictor.count.tolist(), [1, 5])
        self.assertEqual(valid.tolist(), [False, True])
        torch.testing.assert_close(future[0], reset_pose.expand(2, -1))
        torch.testing.assert_close(predictor.poses[1], unaffected)
        # Refill the reset episode with a stationary object: no old velocity.
        for i in range(1, 5):
            predictor.push(reset_pose, .605 + .16*i, torch.tensor([0]))
        future, valid = predictor.predict()
        self.assertTrue(valid.all())
        torch.testing.assert_close(future[0], reset_pose.expand(2, -1))

    def test_invalid_measurement_is_isolated(self):
        predictor = RollingSE3Predictor(2, "cpu", history=2)
        pose = torch.tensor([[0., 0., .6, 0., 0., 0., 1.]]).repeat(2, 1)
        predictor.push(pose, 0.)
        predictor.push(pose, .15)
        predictor.predict()
        pose[0, 0] = float("nan")
        predictor.push(pose, .30)
        future, valid = predictor.predict()
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(torch.isfinite(future).all())

    def test_zero_and_near_pi_rotation_are_finite(self):
        xi = torch.tensor([[.1, -.2, .05, 0., 0., 0.], [.1, .2, .3, np.pi-1e-7, 0., 0.]], dtype=torch.float64)
        recovered = se3_log_batch(se3_exp_batch(xi))
        torch.testing.assert_close(recovered, xi, rtol=1e-7, atol=1e-8)

    def test_shared_frame_matches_original_and_preserves_world_height(self):
        dtype = torch.float64
        robot_position = torch.tensor([[2., -1., .5]], dtype=dtype)
        robot_q = torch.tensor(Rotation.from_euler("xyz", [[.2, -.3, 1.2]]).as_quat(), dtype=dtype)
        yaw_q = torch.tensor(Rotation.from_euler("z", [1.2]).as_quat(), dtype=dtype)
        arm = robot_position + quat_apply(robot_q, torch.tensor([[.3, 0., .09]], dtype=dtype))
        pos = torch.tensor([[2.6, -.5, .65]], dtype=dtype)
        q = torch.tensor(Rotation.from_euler("xyz", [[.5, .1, 1.7]]).as_quat(), dtype=dtype)
        actual = object_pose_in_teacher_frame(pos, q, arm, yaw_q)
        expected_pos = quat_rotate_inverse(yaw_q, pos - arm)
        expected_pos[:, 2] = pos[:, 2]
        expected_rpy = torch.stack(euler_from_quat(quat_mul(quat_conjugate(yaw_q), q)), dim=-1)
        torch.testing.assert_close(actual, torch.cat((expected_pos, expected_rpy), dim=-1), rtol=0, atol=0)
        self.assertEqual(actual[0, 2].item(), .65)
        # Same current reference at both future horizons, even with base roll/pitch.
        future_pos = pos.repeat(2, 1) + torch.tensor([[.06, 0., 0.], [.12, 0., .01]], dtype=dtype)
        future = object_pose_in_teacher_frame(future_pos, q.repeat(2, 1), arm.repeat(2, 1), yaw_q.repeat(2, 1))
        expected_xy = Rotation.from_euler("z", -1.2).apply((future_pos-pos).numpy())[:, :2]
        np.testing.assert_allclose((future[:, :2]-actual[:, :2]).numpy(), expected_xy, atol=1e-12)
        torch.testing.assert_close(future[:, 2], future_pos[:, 2])

    def test_saved_config_survives_resume_and_explicit_disable(self):
        cfg = {"env": {}}
        settings = apply_pose_prediction_config(cfg, get_params(["--future_pose_obs"]))
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["horizons"], [.3, .6])
        apply_pose_prediction_config(cfg, get_params([]))
        self.assertTrue(settings["enabled"])
        apply_pose_prediction_config(cfg, get_params(["--no_future_pose_obs"]))
        self.assertFalse(settings["enabled"])


if __name__ == "__main__":
    unittest.main()
