"""Candidate rotations must preserve geometry and expose roll/pitch/yaw.

SciPy supplies independent matrix references, including gimbal lock. Load
the production quaternion module directly so these tests need no simulator.
"""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from PredictPiontTransform.predictpoint_mul import PredictPoint
from test_sgs_env import method_node


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grasp_rotation_test_torch_utils",
    ROOT.parent / "third_party/isaacgym/python/isaacgym/torch_utils.py")
QUATERNIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUATERNIONS)


def rotation_cases():
    # 0.005 rad is inside the old abs(sin(pitch)) > 0.9999 special branch;
    # 0.02 is outside. Check both sides of both singular pitch angles.
    pitches = [-0.4, 0.7] + [
        sign * np.pi / 2 + offset
        for sign in (-1, 1)
        for offset in (-0.02, -0.005, -1e-7, -1e-12, 0.,
                       1e-12, 1e-7, 0.005, 0.02)]
    return np.array([[roll, pitch, yaw]
                     for roll in (0.31, -1.2, 2.6)
                     for yaw in (-0.72, 0.8, -2.4)
                     for pitch in pitches])


def tensor_settings():
    settings = [("cpu", torch.float64, 2e-13),
                ("cpu", torch.float32, 2e-6)]
    if torch.cuda.is_available():
        settings.append(("cuda", torch.float32, 2e-6))
    return settings


def matrices_from_rpy(rpy, order="xyz"):
    angles = rpy.detach().cpu().double().numpy().reshape(-1, 3)
    if order == "zyx":
        angles = angles[:, ::-1]
    return Rotation.from_euler(order, angles).as_matrix()


def make_predictor(device):
    # Exercise the real constructor, including its fixed TCP matrices.
    with redirect_stdout(io.StringIO()):
        return PredictPoint(
            torch.zeros(1, 1, 6), torch.eye(4)[None, None],
            torch.zeros(1, 1, 6), device=device)


class QuaternionRotationTests(unittest.TestCase):
    def test_regular_angles_are_rpy_and_legacy_api_stays_ypr(self):
        angles = np.array([[0.31, 0.7, -0.72], [-1.2, -0.4, 2.6]])
        for device, dtype, tolerance in tensor_settings():
            with self.subTest(device=device, dtype=dtype):
                quats = torch.tensor(Rotation.from_euler("xyz", angles).as_quat(),
                                     device=device, dtype=dtype)
                rpy = QUATERNIONS.quat_to_rpy(quats)
                np.testing.assert_allclose(rpy.cpu().numpy(), angles,
                                           atol=tolerance, rtol=0)
                torch.testing.assert_close(
                    QUATERNIONS.quat_to_euler_zyx(quats), rpy.flip([-1]),
                    atol=0., rtol=0.)
                # A single quaternion must also preserve its unbatched shape.
                self.assertEqual(QUATERNIONS.quat_to_rpy(quats[0]).shape, (3,))

    def test_singular_and_near_singular_quaternions_reconstruct_rotation(self):
        rotations = Rotation.from_euler("xyz", rotation_cases())
        expected = rotations.as_matrix()
        for device, dtype, tolerance in tensor_settings():
            with self.subTest(device=device, dtype=dtype):
                # A high-dimensional batch with sign and non-unit scale
                # variations also checks normalization and q == -q.
                quats = torch.tensor(rotations.as_quat(), device=device,
                                     dtype=dtype).reshape(3, 3, 20, 4)
                factors = torch.tensor([1., -1., 0.3, -12.5],
                                       device=device, dtype=dtype)
                quats = factors[:, None, None, None, None] * quats[None]
                before = quats.clone()
                rpy = QUATERNIONS.quat_to_rpy(quats)
                self.assertEqual(rpy.shape, (4, 3, 3, 20, 3))
                self.assertEqual(rpy.device, quats.device)
                self.assertEqual(rpy.dtype, dtype)
                self.assertTrue(torch.isfinite(rpy).all())
                self.assertTrue((rpy[..., 1].abs() <= np.pi / 2 + tolerance).all())
                np.testing.assert_allclose(
                    matrices_from_rpy(rpy), np.tile(expected, (4, 1, 1)),
                    atol=tolerance, rtol=0)
                legacy = QUATERNIONS.quat_to_euler_zyx(quats)
                np.testing.assert_allclose(
                    matrices_from_rpy(legacy.flip([-1])),
                    np.tile(expected, (4, 1, 1)), atol=tolerance, rtol=0)
                torch.testing.assert_close(quats, before, atol=0., rtol=0.)


class PredictPointRotationTests(unittest.TestCase):
    def test_decomposition_preserves_calibrated_rotation_and_translation(self):
        angles = rotation_cases()
        # The TCP correction is fixed in production; build its independent
        # reference from rotations, not from the attributes being tested.
        rx = Rotation.from_euler("x", np.pi / 2).as_matrix()
        rz = Rotation.from_euler("z", np.pi / 2).as_matrix()
        calibration = rx @ rz @ rx
        positions = np.linspace(-0.4, 0.8, len(angles) * 3).reshape(-1, 3)
        for device, dtype, tolerance in tensor_settings():
            predictor = make_predictor(device)
            np.testing.assert_allclose(predictor.Rx_rot.cpu().numpy(), rx, atol=1e-15)
            np.testing.assert_allclose(predictor.Rz_rot.cpu().numpy(), rz, atol=1e-15)
            predictor.Rx_rot = predictor.Rx_rot.to(dtype=dtype)
            predictor.Rz_rot = predictor.Rz_rot.to(dtype=dtype)
            for order in ("xyz", "zyx"):
                with self.subTest(device=device, dtype=dtype, order=order):
                    scipy_angles = angles if order == "xyz" else angles[:, ::-1]
                    expected = Rotation.from_euler(order, scipy_angles).as_matrix()
                    transforms = np.broadcast_to(np.eye(4), (len(angles), 4, 4)).copy()
                    # Ensure the *calibrated* rotation crosses gimbal lock.
                    transforms[:, :3, :3] = expected @ calibration.T
                    transforms[:, :3, 3] = positions
                    transforms = torch.tensor(transforms, device=device,
                                              dtype=dtype).reshape(2, 3, 30, 4, 4)
                    before = transforms.clone()
                    xyz, rpy = predictor.decompose_pose_to_xyz_rpy(transforms, order)
                    self.assertEqual(rpy.shape, (2, 3, 30, 3))
                    self.assertEqual(xyz.shape, rpy.shape)
                    self.assertEqual(rpy.dtype, dtype)
                    self.assertEqual(rpy.device, transforms.device)
                    np.testing.assert_allclose(matrices_from_rpy(rpy, order), expected,
                                               atol=tolerance, rtol=0)
                    torch.testing.assert_close(xyz, transforms[..., :3, 3],
                                               atol=0., rtol=0.)
                    torch.testing.assert_close(transforms, before, atol=0., rtol=0.)

    def test_get_grasp_world_keeps_object_transform_and_tcp_chain(self):
        predictor = make_predictor("cpu")
        object_pose = np.array([1.2, -0.6, 0.8, 0.21, -0.34, 0.57])
        object_transform = np.eye(4)
        object_transform[:3, :3] = Rotation.from_euler("xyz", object_pose[3:]).as_matrix()
        object_transform[:3, 3] = object_pose[:3]
        local_grasps = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        local_grasps[:, :3, :3] = Rotation.from_euler(
            "xyz", [[0.4, 0.2, -0.7], [-1.1, 0.8, 0.3], [0.2, -0.6, 1.4]]).as_matrix()
        local_grasps[:, :3, 3] = [[0.1, 0.2, 0.3], [-0.1, 0.4, 0.2], [0.3, 0.1, -0.2]]
        predictor.T_grasp_in_obj = torch.tensor(local_grasps[None], dtype=torch.float32)
        xyz, rpy = predictor.get_grasp_world(
            torch.tensor(object_pose[None], dtype=torch.float32), torch.tensor([0]))
        expected = object_transform[None] @ local_grasps
        rx = Rotation.from_euler("x", np.pi / 2).as_matrix()
        rz = Rotation.from_euler("z", np.pi / 2).as_matrix()
        np.testing.assert_allclose(xyz[0].numpy(), expected[:, :3, 3], atol=3e-7, rtol=0)
        np.testing.assert_allclose(matrices_from_rpy(rpy),
                                   expected[:, :3, :3] @ rx @ rz @ rx,
                                   atol=5e-7, rtol=0)


class CandidateObservationTests(unittest.TestCase):
    def test_actual_observation_tail_contains_local_rpy_instead_of_ypr(self):
        # Use production _compute_observations and production quaternion
        # functions; stub only the parent observation and simulator state.
        class Base:
            def _compute_observations(self, ids):
                return torch.arange(64, dtype=torch.float).expand(len(ids), -1)

        node = ast.ClassDef(name="Task", bases=[ast.Name(id="Base", ctx=ast.Load())],
                            keywords=[], decorator_list=[], body=[method_node(
                                "b1z1_pickmulti.py", "B1Z1PickMulti", "_compute_observations")])
        namespace = dict(vars(QUATERNIONS), Base=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(ROOT / "envs/b1z1_pickmulti.py"), "exec"), namespace)
        task = namespace["Task"]()
        task.no_feature = False
        task.cfg = {"env": {}}
        task.feature_obs = torch.arange(1024, dtype=torch.float)[None]
        task.action_history_buf = torch.arange(9, dtype=torch.float)[None, None]
        task.obs_buf = torch.zeros(1, 1284)
        task._future_object_observations = lambda ids, pose: torch.arange(13.)[None]
        task._tsil_observations = lambda ids: torch.zeros(len(ids), 0)
        base = Rotation.from_euler("xyz", [0.41, -0.28, 0.83])
        task._robot_root_states = torch.zeros(1, 13)
        task._robot_root_states[0, 3:7] = torch.tensor(base.as_quat(), dtype=torch.float32)
        arm_base = np.array([1.4, -0.5, 0.7])
        task.arm_base = torch.tensor(arm_base[None], dtype=torch.float32)
        # Different nonzero roll/yaw values make a YPR swap observable.
        angles = np.array([[0.31, 0.45, -0.72], [-1.2, -0.4, 0.8],
                           [0.4, np.pi / 2 - 0.005, -1.1]])
        angles = np.tile(angles, (10, 1))
        local_rotations = Rotation.from_euler("xyz", angles)
        world_angles = (base * local_rotations).as_euler("xyz")
        local_positions = np.tile([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4],
                                   [0.4, -0.3, 0.2]], (10, 1))
        world_positions = base.apply(local_positions) + arm_base
        task.get_grasp_world = lambda pose, ids: (
            torch.tensor(world_positions[None], dtype=torch.float32),
            torch.tensor(world_angles[None], dtype=torch.float32))
        task._compute_observations(torch.tensor([0]))
        rpy = task.obs_buf[:, -99:-9].reshape(30, 3)
        xyz = task.obs_buf[:, -189:-99].reshape(30, 3)
        # Compare regular angles explicitly, plus every reconstructed matrix.
        np.testing.assert_allclose(rpy.numpy().reshape(10, 3, 3)[:, :2],
                                   angles.reshape(10, 3, 3)[:, :2], atol=5e-7, rtol=0)
        np.testing.assert_allclose(matrices_from_rpy(rpy), local_rotations.as_matrix(),
                                   atol=8e-7, rtol=0)
        np.testing.assert_allclose(xyz.numpy(), local_positions, atol=3e-7, rtol=0)
        torch.testing.assert_close(task.obs_buf[:, -9:], task.action_history_buf[:, -1])


if __name__ == "__main__":
    unittest.main()
