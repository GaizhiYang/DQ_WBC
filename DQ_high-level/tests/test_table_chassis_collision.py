"""CPU checks of actual collider geometry and production failure bookkeeping."""

import math
from pathlib import Path
from types import MethodType
import unittest

import numpy as np
import torch

from utils.table_collision import (box_separation, quaternion_box_axes,
                                   read_chassis_box, table_chassis_contact)
from test_sgs_env import (TestEnvironment, source_method, quat_apply, quat_mul,
                          quat_from_euler_xyz)


ROOT = Path(__file__).resolve().parents[1]


class BoxCollisionTests(unittest.TestCase):
    def setUp(self):
        half, offset, rotation = read_chassis_box(
            ROOT / "data/asset/b1z1-col/urdf/b1z1.urdf")
        self.assertEqual(half, [.3235, .15, .075])
        self.assertEqual(offset, [0., 0., 0.])
        self.assertEqual(rotation, [0., 0., 0., 1.])
        self.chassis = torch.tensor(half)
        self.table = torch.tensor([.1, .1, .05])

    def contact(self, positions, forces=None, table_quat=None, rest_offset=0.):
        positions = torch.as_tensor(positions, dtype=torch.float32)
        n = positions.shape[0]
        identity = torch.tensor([[0., 0., 0., 1.]]).expand(n, -1)
        return table_chassis_contact(
            torch.zeros(n, 3), identity, self.chassis, positions,
            identity if table_quat is None else table_quat, self.table,
            torch.zeros(n, 3) if forces is None else forces, rest_offset)

    def test_touching_and_overlap_need_no_force_but_clearance_does(self):
        result = self.contact([[.4235, 0., 0.], [.42, 0., 0.], [.425, 0., 0.],
                               [0., .25, 0.], [0., .252, 0.], [0., 0., .126]])
        self.assertEqual(result.tolist(), [True, True, False, True, False, False])

    def test_solver_gap_needs_chassis_contact_and_is_not_contact_offset(self):
        positions = [[.425, 0., 0.], [.425, 0., 0.], [.43, 0., 0.], [.45, 0., 0.]]
        forces = torch.tensor([[0., 0., 0.], [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]])
        self.assertEqual(self.contact(positions, forces).tolist(), [False, True, False, False])
        self.assertEqual(self.contact(positions, forces, rest_offset=.01).tolist(),
                         [False, True, True, False])

    def test_rotated_table_and_vertical_separation(self):
        # A 45-degree square reaches farther along x than its unrotated box.
        angle = torch.full((3,), math.pi / 4)
        q = quat_from_euler_xyz(torch.zeros(3), torch.zeros(3), angle)
        self.assertEqual(self.contact([[.46, 0., 0.], [.47, 0., 0.], [.46, 0., .13]],
                                     table_quat=q).tolist(), [True, False, False])

    def test_random_orientations_match_independent_vertex_sat(self):
        generator = torch.Generator().manual_seed(31)
        n = 160
        qa = torch.randn(n, 4, generator=generator, dtype=torch.float64)
        qb = torch.randn(n, 4, generator=generator, dtype=torch.float64)
        qa /= qa.norm(dim=-1, keepdim=True)
        qb /= qb.norm(dim=-1, keepdim=True)
        centers = (torch.rand(n, 3, generator=generator, dtype=torch.float64) - .5) * .8
        half_a, half_b = self.chassis.double(), self.table.double()
        got = box_separation(torch.zeros_like(centers), qa, half_a, centers, qb, half_b)
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        # Construct matrices independently through rotating the basis vectors.
        eye = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        axes_a = quat_apply(qa[:, None, :].expand(-1, 3, -1), eye).numpy()
        axes_b = quat_apply(qb[:, None, :].expand(-1, 3, -1), eye).numpy()
        expected = []
        for i in range(n):
            va = (corners * half_a.numpy()) @ axes_a[i]
            vb = (corners * half_b.numpy()) @ axes_b[i] + centers[i].numpy()
            axes = list(axes_a[i]) + list(axes_b[i])
            axes += [np.cross(a, b) for a in axes_a[i] for b in axes_b[i]]
            separated = False
            for axis in axes:
                if np.linalg.norm(axis) < 1.e-6:
                    continue
                pa, pb = va @ axis, vb @ axis
                separated |= pa.max() < pb.min() or pb.max() < pa.min()
            expected.append(not separated)
        self.assertEqual((got <= 0).tolist(), expected)
        # The result must not depend on environment origins or robot orientation.
        common = torch.tensor([[.2, -.3, .1, .9]], dtype=torch.float64)
        common = (common / common.norm()).expand(n, -1)
        shift = torch.tensor([10., -4., 2.], dtype=torch.float64)
        moved = box_separation(
            shift.expand(n, -1), quat_mul(common, qa), half_a,
            quat_apply(common, centers) + shift, quat_mul(common, qb), half_b)
        torch.testing.assert_close(got, moved)


class CollisionTerminationTests(unittest.TestCase):
    def environment(self):
        env = TestEnvironment()
        env.initialize()
        env.table_chassis_collision = torch.zeros(env.num_envs, dtype=torch.bool)
        env.chassis_idx = 0
        env._rigid_body_pos = env._rigid_body_state[..., :3]
        env._rigid_body_rot = env._rigid_body_state[..., 3:7]
        env._chassis_half_extents = torch.tensor([.3235, .15, .075])
        env._table_half_extents = torch.tensor([.1, .1, .05])
        env._chassis_box_offset = torch.zeros(4, 3)
        env._chassis_box_rotation = torch.tensor([[0., 0., 0., 1.]]).expand(4, -1)
        env._table_chassis_rest_offset = 0.
        update = source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_update_table_chassis_collision")
        update.__globals__["table_chassis_contact"] = table_chassis_contact
        env._update_table_chassis_collision = MethodType(update, env)
        return env

    def test_transient_hit_is_latched_and_other_body_contacts_do_not_trigger(self):
        env = self.environment()
        env._rigid_body_pos[:, 0] = torch.tensor([0., 0., .6])
        env._table_root_states[:, :3] = torch.tensor([.42, 0., .6])
        env._table_root_states[1:, 0] = .425  # close but no chassis force
        env._contact_forces[:] = 1000.  # feet/arm/gripper/object can all be in contact
        env._contact_forces[:, 0] = 0.
        env._update_table_chassis_collision()
        self.assertEqual(env.table_chassis_collision.tolist(), [True, False, False, False])
        env._table_root_states[:, 0] = 1.
        env._update_table_chassis_collision()
        self.assertTrue(env.table_chassis_collision[0])
        env.table_chassis_collision[2] = True
        env._reset_pick_state(torch.tensor([0]))
        self.assertEqual(env.table_chassis_collision.tolist(), [False, False, True, False])

    def test_hit_wins_over_same_step_success_and_timeout(self):
        env = self.environment()
        env.table_chassis_collision[0] = True
        env.pick_counter[:] = env.hold_steps - 1
        env.best_pick_counter[:] = env.hold_steps - 1
        env.progress_buf[:] = env.max_episode_length
        env.grasp_step()
        self.assertEqual(env._terminate_buf.tolist(), [1, 0, 0, 0])
        self.assertEqual(env.reset_buf.tolist(), [1, 1, 1, 1])
        self.assertEqual(env.pick_success.tolist(), [False, True, True, True])
        self.assertEqual(env.episode_success.tolist(), [False, True, True, True])
        self.assertEqual(env.success_counter.tolist(), [0, 1, 1, 1])
        self.assertFalse(bool(env.timeout_buf.any()))

    def test_sgs_reset_clears_latch_only_for_reset_slots(self):
        env = self.environment()
        env.table_chassis_collision[:] = True
        env._reset_sgs_envs(torch.tensor([0, 2]))
        self.assertEqual(env.table_chassis_collision.tolist(), [False, True, False, True])

    def test_collision_clears_previously_latched_success(self):
        env = self.environment()
        env.pick_success_recorded[0] = True
        env.table_chassis_collision[0] = True
        env.grasp_step()
        self.assertFalse(env.episode_success[0])
        self.assertFalse(env.pick_success[0])
        self.assertTrue(env.reset_buf[0])


if __name__ == "__main__":
    unittest.main()
