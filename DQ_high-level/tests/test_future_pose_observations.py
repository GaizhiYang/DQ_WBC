"""Numerical/reference, coordinate-frame and episode-isolation regression tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    # Isaac Gym requires this import before torch, even in a CPU-only test.
    from isaacgym import gymapi
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Run these tests in the dqwbc environment: {}".format(exc))

from isaacgym.torch_utils import quat_apply
from modules.kalman_filter import SE3KF, ekf_predict_horizon
from modules.pose_prediction import RollingSE3Predictor, poses_to_transforms, se3_exp_batch, se3_log_batch
from utils.object_pose import (
    object_pose_in_arm_frame,
    object_pose_in_teacher_frame,
    pose_to_rpy_observation,
)
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

    def test_arm_frame_uses_full_robot_orientation_and_relative_height(self):
        dtype = torch.float64
        robot_position = torch.tensor([[2., -1., .5], [-1., 3., .8]], dtype=dtype)
        robot_rotation = Rotation.from_euler("xyz", [[.2, -.3, 1.2], [-.4, .25, -.7]])
        robot_q = torch.tensor(robot_rotation.as_quat(), dtype=dtype)
        arm = robot_position + quat_apply(robot_q, torch.tensor([[.3, 0., .09]], dtype=dtype).repeat(2, 1))
        expected_position = np.array([[.4, -.1, .25], [-.2, .3, -.12]])
        expected_rpy = np.array([[.3, -.2, .6], [-.4, .25, -.5]])
        relative_rotation = Rotation.from_euler("xyz", expected_rpy)
        pos = arm + torch.tensor(robot_rotation.apply(expected_position), dtype=dtype)
        q = torch.tensor((robot_rotation * relative_rotation).as_quat(), dtype=dtype)

        local_pose = object_pose_in_arm_frame(pos, q, arm, robot_q)
        actual = object_pose_in_teacher_frame(pos, q, arm, robot_q)
        expected = torch.tensor(np.column_stack((expected_position, expected_rpy)), dtype=dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(pose_to_rpy_observation(local_pose), expected, rtol=0, atol=1e-12)
        self.assertTrue(torch.all(torch.abs(actual[:, 2] - pos[:, 2]) > .1))

        # Reconstruct the original rigid transform, including all three axes.
        arm_world = poses_to_transforms(torch.cat((arm, robot_q), dim=-1))
        object_world = poses_to_transforms(torch.cat((pos, q), dim=-1))
        torch.testing.assert_close(arm_world @ poses_to_transforms(local_pose), object_world, rtol=0, atol=1e-12)

    def test_arm_frame_is_invariant_to_common_world_rigid_transform(self):
        dtype = torch.float64
        robot_rotation = Rotation.from_euler("xyz", [[.3, -.4, .8], [-.2, .1, -.5]])
        object_rotation = Rotation.from_euler("xyz", [[-.1, .5, -.6], [.6, -.3, .2]])
        arm = np.array([[2., -1., .7], [-3., 2., .9]])
        position = np.array([[2.5, -.8, 1.1], [-2.4, 1.9, .5]])
        world_rotation = Rotation.from_euler("xyz", [.5, -.2, 1.1])
        translation = np.array([10., -20., 4.])

        def convert(p, q, a, robot_q):
            return object_pose_in_arm_frame(*[
                torch.tensor(value, dtype=dtype) for value in (p, q, a, robot_q)
            ])

        original = convert(position, object_rotation.as_quat(), arm, robot_rotation.as_quat())
        transformed = convert(
            world_rotation.apply(position) + translation,
            (world_rotation * object_rotation).as_quat(),
            world_rotation.apply(arm) + translation,
            (world_rotation * robot_rotation).as_quat(),
        )
        torch.testing.assert_close(poses_to_transforms(original), poses_to_transforms(transformed), rtol=0, atol=1e-12)
        torch.testing.assert_close(pose_to_rpy_observation(original), pose_to_rpy_observation(transformed), rtol=0, atol=1e-12)

    def check_forecasts_relative_motion_of_stationary_world_object(self, device, dtype):
        times = np.array([0., .14, .30, .445, .605])
        horizons = (.6, .9)
        initial_arm = np.eye(4)
        initial_arm[:3, :3] = Rotation.from_euler("xyz", [.2, -.15, .4]).as_matrix()
        initial_arm[:3, 3] = [.8, -.3, .7]
        # A translating and rotating robot, with constant arm-frame body twist.
        # Use independent SciPy matrix exponentials for the ground truth.
        vx, vy, vz, wx, wy, wz = [.15, -.08, .03, .10, -.12, .20]
        generator = np.array([
            [0., -wz, wy, vx], [wz, 0., -wx, vy],
            [-wy, wx, 0., vz], [0., 0., 0., 0.],
        ])
        object_world = np.eye(4)
        object_world[:3, :3] = Rotation.from_euler("xyz", [.2, .4, -.3]).as_matrix()
        object_world[:3, 3] = [1.7, .1, 1.]
        object_position = torch.tensor(object_world[None, :3, 3], device=device, dtype=dtype)
        object_quaternion = torch.tensor(Rotation.from_matrix(object_world[:3, :3]).as_quat()[None], device=device, dtype=dtype)
        predictor = RollingSE3Predictor(1, device, dtype=dtype, horizons=horizons)
        relative_history = []
        for timestamp in times:
            arm_world = initial_arm @ expm(generator * timestamp)
            robot_q = torch.tensor(Rotation.from_matrix(arm_world[:3, :3]).as_quat()[None], device=device, dtype=dtype)
            # Derive the arm origin from the robot root and mounting offset.
            offset = quat_apply(robot_q, torch.tensor([[.3, 0., .09]], device=device, dtype=dtype))
            root_world = torch.tensor(arm_world[None, :3, 3], device=device, dtype=dtype) - offset
            relative_pose = object_pose_in_arm_frame(
                object_position, object_quaternion, root_world + offset, robot_q
            )
            relative_history.append(relative_pose)
            predictor.push(relative_pose, timestamp)

        # History must contain T_A(t)O(t), not world measurements or all samples
        # re-expressed in just the latest robot frame.
        torch.testing.assert_close(predictor.poses[0], torch.cat(relative_history), rtol=0, atol=1e-6)
        future, valid = predictor.predict()
        self.assertTrue(valid.all())
        actual = poses_to_transforms(future[0]).cpu().numpy()
        expected = np.stack([
            np.linalg.inv(initial_arm @ expm(generator * (times[-1] + horizon))) @ object_world
            for horizon in horizons
        ])
        # The finite history starts with zero velocity; fitting constant relative
        # motion leaves a small, bounded EKF estimation error.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-4)
        encoded = pose_to_rpy_observation(future[0]).cpu().numpy()
        np.testing.assert_allclose(encoded[:, :3], expected[:, :3, 3], rtol=0, atol=4e-4)
        np.testing.assert_allclose(Rotation.from_euler("xyz", encoded[:, 3:]).as_matrix(), expected[:, :3, :3], rtol=0, atol=4e-4)

        # A world-space predictor for this stationary object, followed only by
        # conversion into the current A frame, would predict no relative motion.
        current_relative = poses_to_transforms(relative_history[-1])[0].cpu().numpy()
        self.assertGreater(np.linalg.norm(actual[0, :3, 3] - current_relative[:3, 3]), .1)
        self.assertGreater(np.linalg.norm(actual[1, :3, 3] - current_relative[:3, 3]), .18)
        self.assertGreater(np.linalg.norm(actual[0, :3, :3] - current_relative[:3, :3]), .1)

    def test_relative_arm_frame_forecast_includes_robot_motion_cpu(self):
        self.check_forecasts_relative_motion_of_stationary_world_object("cpu", torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_relative_arm_frame_forecast_includes_robot_motion_gpu(self):
        self.check_forecasts_relative_motion_of_stationary_world_object("cuda:0", torch.float32)

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
