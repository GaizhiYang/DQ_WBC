"""Independent geometry/velocity/gradient checks for imagined B1Z1 kinematics.

Run with ``python -m unittest discover -s DQ_high-level/tests -p
test_world_model_kinematics.py``. CUDA tests run on actual hardware when present;
these checks do not import Isaac Gym or substitute for live-simulator parity.
"""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.kinematics import B1Z1Kinematics, DEFAULT_DOF_NAMES, DEFAULT_URDF


def matrix_oracle(q):
    """Separate homogeneous-matrix URDF implementation, base at identity."""
    root = ET.parse(DEFAULT_URDF).getroot()
    pending = list(root.findall("joint"))
    transforms = {"base": np.eye(4)}
    index = {name: i for i, name in enumerate(DEFAULT_DOF_NAMES)}

    def rotation(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross

    while pending:
        ready = [joint for joint in pending if joint.find("parent").get("link") in transforms]
        assert ready, "Disconnected oracle tree"
        for joint in ready:
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").replace("0.09>>", "0.09").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (rotation([0, 0, 1], rpy[2])
                                  @ rotation([0, 1, 0], rpy[1])
                                  @ rotation([1, 0, 0], rpy[0]))
            if joint.get("type") != "fixed":
                axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                transform[:3, :3] = transform[:3, :3] @ rotation(axis, q[index[joint.get("name")]])
            transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ transform
            pending.remove(joint)
    return transforms


def quat_matrix(q):
    """Independent test-only xyzw quaternion to matrix."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


class KinematicsChecks:
    device = "cpu"

    def inputs(self, batch=(3,)):
        torch.manual_seed(813)
        base = torch.randn(*batch, 13, dtype=torch.float64, device=self.device)
        base[..., 3:7] /= torch.linalg.vector_norm(base[..., 3:7], dim=-1, keepdim=True)
        q = torch.randn(*batch, 19, dtype=torch.float64, device=self.device) * 0.6
        dq = torch.randn_like(q)
        return base, q, dq

    def fk(self, **kwargs):
        return B1Z1Kinematics(device=self.device, dtype=torch.float64, **kwargs)

    def test_zero_pose_matches_urdf_dimensions_and_fixed_gripper_stator(self):
        fk = self.fk()
        base = torch.zeros(2, 13, device=self.device, dtype=torch.float64)
        base[:, 6] = 1
        q = torch.zeros(2, 19, device=self.device, dtype=torch.float64)
        q[1, -1] = 0.8
        result = fk(base, q, torch.zeros_like(q))
        expected_ee = torch.tensor([0.4732, 0, 0.2505], dtype=q.dtype, device=self.device)
        torch.testing.assert_close(result["ee_pos"], expected_ee.expand(2, 3))
        feet = torch.tensor([[0.3455, -0.19875, -0.7], [0.3455, 0.19875, -0.7],
                             [-0.3455, -0.19875, -0.7], [-0.3455, 0.19875, -0.7]],
                            dtype=q.dtype, device=self.device)
        torch.testing.assert_close(result["feet_pos"], feet.expand(2, 4, 3))
        self.assertTrue(all(value.device == base.device for value in result.values()))

    def test_random_pose_matches_independent_matrix_oracle(self):
        base, q, dq = self.inputs()
        result = self.fk()(base, q, dq)
        for i in range(base.shape[0]):
            base_cpu, q_cpu = base[i].cpu().numpy(), q[i].cpu().numpy()
            transform = np.eye(4)
            transform[:3, :3] = quat_matrix(base_cpu[3:7])
            transform[:3, 3] = base_cpu[:3]
            expected = transform @ matrix_oracle(q_cpu)["ee_gripper_link"]
            np.testing.assert_allclose(result["ee_pos"][i].cpu().numpy(), expected[:3, 3], atol=1e-12)
            np.testing.assert_allclose(quat_matrix(result["ee_quat"][i].cpu().numpy()),
                                       expected[:3, :3], atol=1e-12)

    def test_world_velocity_matches_pose_finite_difference(self):
        base, q, dq = self.inputs(batch=(2, 3))
        fk, epsilon = self.fk(), 1e-6
        result = fk(base, q, dq)

        def advance(sign):
            shifted = base.clone()
            dt = sign * epsilon
            # Isaac stores root COM velocity, while root position is the link
            # origin. Independently rotate the known merged-mount COM offset.
            vector = fk.base_com.expand_as(base[..., :3])
            quaternion = base[..., 3:7]
            cross = 2 * torch.cross(quaternion[..., :3], vector, dim=-1)
            world_com = vector + quaternion[..., 3:] * cross + torch.cross(quaternion[..., :3], cross, dim=-1)
            origin_velocity = base[..., 7:10] - torch.cross(base[..., 10:13], world_com, dim=-1)
            shifted[..., :3] += dt * origin_velocity
            omega = base[..., 10:13]
            norm = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
            delta = torch.cat((omega / norm * (norm * dt / 2).sin(),
                               (norm * dt / 2).cos()), -1)
            shifted[..., 3:7] = multiply(delta, base[..., 3:7])
            return fk(shifted, q + dt * dq, dq)

        plus, minus = advance(1), advance(-1)
        for position, velocity in (("ee_pos", "ee_vel"), ("feet_pos", "feet_vel")):
            derivative = (plus[position] - minus[position]) / (2 * epsilon)
            torch.testing.assert_close(result[velocity], derivative, atol=2e-9, rtol=2e-8)
        conjugate = torch.cat((-minus["ee_quat"][..., :3], minus["ee_quat"][..., 3:]), -1)
        delta = multiply(plus["ee_quat"], conjugate)
        torch.testing.assert_close(result["ee_omega"], delta[..., :3] / epsilon,
                                   atol=2e-9, rtol=2e-8)

    def test_simulator_dof_order_can_be_explicitly_remapped(self):
        base, q, dq = self.inputs()
        expected = self.fk()(base, q, dq)
        permutation = tuple(reversed(range(19)))
        remapped = self.fk(dof_names=[DEFAULT_DOF_NAMES[i] for i in permutation])
        actual = remapped(base, q[..., list(permutation)], dq[..., list(permutation)])
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])

    def test_root_com_is_merged_mount_and_can_be_overridden(self):
        fk = self.fk()
        expected = torch.tensor([.29665016, -.00013615, .11495843], dtype=torch.float64, device=self.device)
        torch.testing.assert_close(fk.base_com, expected)
        base, q, dq = self.inputs()
        corrected = base.clone()
        corrected[..., 7:10] = fk.root_origin_velocity(base)
        zero_com = torch.zeros(3, dtype=base.dtype, device=self.device)
        default, explicit = fk(base, q, dq), fk(corrected, q, dq, base_com=zero_com)
        for key in default:
            torch.testing.assert_close(default[key], explicit[key])
        # The live PhysX merged rigid body can recompute COM from collision
        # geometry, so callers must be able to override its inertial URDF value.
        actual_com = torch.tensor([.3, 0., .1155], dtype=torch.float64, device=self.device)
        compiled = self.fk(base_com=actual_com)
        with_override = fk(base, q, dq, base_com=actual_com)
        for key, value in compiled(base, q, dq).items():
            torch.testing.assert_close(value, with_override[key])

    def test_fk_remains_differentiable_through_pose_and_velocity(self):
        base, q, dq = (tensor.requires_grad_() for tensor in self.inputs(batch=(1,)))
        fk = self.fk()

        def output(b, p, v):
            value = fk(b, p, v)
            return torch.cat((value["ee_pos"], value["ee_quat"], value["ee_vel"],
                              value["ee_omega"], value["feet_pos"].flatten(-2),
                              value["feet_vel"].flatten(-2)), -1)

        self.assertTrue(torch.autograd.gradcheck(output, (base, q, dq), fast_mode=True))
        loss = output(base, q, dq).square().sum()
        loss.backward()
        for tensor in (base, q, dq):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0)
        # EE is defined on the stator; no false opening->EE kinematics gradient.
        self.assertEqual(float(q.grad[..., -1].abs().sum()), 0)


class TestKinematicsCPU(KinematicsChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
class TestKinematicsCUDA(KinematicsChecks, unittest.TestCase):
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


if __name__ == "__main__":
    unittest.main()
