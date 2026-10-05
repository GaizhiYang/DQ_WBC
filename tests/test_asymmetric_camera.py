"""Camera-coordinate and capture-time contracts without an IsaacGym runtime."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DQ_high-level"))
from utils.asymmetric_camera import record_camera_geometry, sensor_packet, visible_surface_points
from utils.asymmetric_perception import AsymmetricPerception, PerceptionConfig


class FakeGym:
    """Row-vector OpenGL view matrices with simulation-global translations."""

    def __init__(self):
        self.time = 1_800_000_000.0625
        self.origins = np.array([[0., 0., 0.], [6., -3., 1.5]])
        self.local_positions = np.array([[.3, .2, .8], [.6, -.1, 1.1]])
        self.rotations = np.array([np.eye(3), [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])

    def get_camera_view_matrix(self, sim, environment, camera):
        rotation = self.rotations[camera]
        position = self.origins[environment] + self.local_positions[camera]
        # Derive world->OpenGL camera explicitly. Its row-vector translation
        # is on the final row, not on the final column of the returned matrix.
        view = np.eye(4)
        view[:3, :3] = rotation
        view[3, :3] = -position @ rotation
        return view

    def get_camera_proj_matrix(self, sim, environment, camera):
        projection = np.eye(4)
        projection[0, 0], projection[1, 1] = ((1.5, 2.) if camera == 0 else (2.25, 3.))
        return projection

    def get_env_origin(self, environment):
        return SimpleNamespace(**dict(zip(("x", "y", "z"), self.origins[environment])))

    def get_sim_time(self, sim):
        return self.time


def make_env():
    env = SimpleNamespace(gym=FakeGym(), sim=object(), envs=[0, 1],
                          camera_handles=[[0, 1], [0, 1]], num_envs=2,
                          device="cpu", cfg={"sensor": {"resized_resolution": [96, 54]}})
    env._cube_root_states = torch.arange(26, dtype=torch.float32).reshape(2, 13).requires_grad_()
    env._robot_root_states = torch.zeros(2, 13)
    env._robot_root_states[:, :3] = torch.tensor([[1., 2., .6], [1., 2., .6]])
    env._robot_root_states[:, 6] = 1
    frames = torch.zeros(2, 4, 54, 96)
    frames[:, 0] = 1
    frames[:, 2], frames[:, 3] = .25, .5
    env._camera_frame_observation = lambda: frames.flatten(1)
    return env


class AsymmetricCameraTests(unittest.TestCase):
    def test_inverse_transpose_optical_axes_and_environment_origin_alignment(self):
        env = make_env()
        record_camera_geometry(env)
        transforms = env._m2_camera_transforms
        expected_rotation = torch.tensor([[[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                                          [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]])
        expected_position = torch.tensor([[.3, .2, .8], [.6, -.1, 1.1]])
        for environment in (0, 1):
            torch.testing.assert_close(transforms[environment, :, :3, :3], expected_rotation)
            torch.testing.assert_close(transforms[environment, :, :3, 3], expected_position)
        # OpenCV +x is right, +y down, +z forwards. Exercise all three axis
        # signs using a concrete ray, independent of the inverse implementation.
        ray = torch.tensor([.5, 1., 2., 1.])
        expected_ray = torch.tensor([[.8, -.8, -1.2, 1.], [1.6, .4, -.9, 1.]])
        torch.testing.assert_close(transforms[0] @ ray, expected_ray)
        torch.testing.assert_close(transforms[1] @ ray, expected_ray)

    def test_intrinsics_use_resized_pixel_dimensions_per_camera(self):
        env = make_env()
        record_camera_geometry(env)
        expected = torch.tensor([[[72., 0., 48.], [0., 54., 27.], [0., 0., 1.]],
                                 [[108., 0., 48.], [0., 81., 27.], [0., 0., 1.]]])
        torch.testing.assert_close(env._m2_camera_intrinsics, expected[None].expand(2, -1, -1, -1))

    def test_capture_timestamp_and_reward_truth_are_independent_snapshots(self):
        env = make_env()
        record_camera_geometry(env)
        captured_time = env._m2_camera_timestamps.clone()
        captured_gt = env._m2_reward_capture_object_states
        expected_gt = env._cube_root_states.detach().clone()
        self.assertEqual(captured_time.dtype, torch.float64)
        self.assertEqual(captured_time[0, 0].item(), env.gym.time)
        self.assertFalse(captured_gt.requires_grad)
        self.assertNotEqual(captured_gt.data_ptr(), env._cube_root_states.data_ptr())
        with torch.no_grad():
            env._cube_root_states.add_(10)
        env.gym.time += .125
        torch.testing.assert_close(captured_gt, expected_gt)
        torch.testing.assert_close(env._m2_camera_timestamps, captured_time)
        record_camera_geometry(env)
        torch.testing.assert_close(env._m2_reward_capture_object_states, expected_gt + 10)
        torch.testing.assert_close(captured_gt, expected_gt)
        torch.testing.assert_close(env._m2_camera_timestamps, captured_time + .125)

    def test_sensor_packet_preserves_capture_latency_units_and_sensor_only_keys(self):
        env = make_env()
        record_camera_geometry(env)
        env.gym.time += .125
        packet = sensor_packet(env)
        self.assertEqual(set(packet), {"images", "T_world_camera", "intrinsics", "timestamps", "now", "T_world_base"})
        self.assertEqual(packet["images"].shape, (2, 2, 2, 54, 96))
        torch.testing.assert_close(packet["now"][:, None] - packet["timestamps"], torch.full((2, 2), .125, dtype=torch.float64))
        torch.testing.assert_close(packet["images"][:, 0, 0], torch.ones(2, 54, 96))
        torch.testing.assert_close(packet["images"][:, 1, 0], torch.zeros(2, 54, 96))
        torch.testing.assert_close(packet["images"][:, 0, 1], torch.full((2, 54, 96), .75))
        torch.testing.assert_close(packet["images"][:, 1, 1], torch.full((2, 54, 96), 1.5))
        torch.testing.assert_close(packet["T_world_base"][:, :3, 3], env._robot_root_states[:, :3])

    def test_reward_surface_depth_gates_match_the_deployable_filter(self):
        env = make_env()
        record_camera_geometry(env)
        packet = sensor_packet(env)
        images = torch.zeros_like(packet["images"])
        images[:, :, 0, 27, 48:53] = 1
        images[:, :, 1, 27, 48:53] = torch.tensor([.2, .7, 1.5, 2., float("nan")])
        config = PerceptionConfig(min_depth_m=.5, max_depth_m=2., min_valid_pixels=2)
        world, valid, count = visible_surface_points(images, packet["T_world_camera"], packet["intrinsics"], config)
        self.assertTrue(valid.all())
        torch.testing.assert_close(count, torch.full((2, 2), 2))
        estimator = AsymmetricPerception(2, 54, 96, config=config)
        out = estimator.step(images, packet["T_world_camera"], packet["intrinsics"], packet["timestamps"],
                             packet["T_world_base"], packet["now"], corrupt=False)
        torch.testing.assert_close(world, out["measurement_world"])
        torch.testing.assert_close(valid, out["valid"])


if __name__ == "__main__":
    unittest.main()
