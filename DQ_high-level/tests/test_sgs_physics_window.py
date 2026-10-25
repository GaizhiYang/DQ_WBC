"""Exercise the continuous validation window with scripted CPU physics frames."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import utils.sgs_bank_builder as bank_builder
from utils.sgs_grounding import FOOT_NAMES, FOOT_RADIUS


class PhysicsWindowTests(unittest.TestCase):
    def make_builder(self, frames, count=3):
        """Keep real tensor checks; replace only the simulator and target sync."""
        env = SimpleNamespace(num_envs=count, device="cpu", sim=object())
        env._root_states = torch.zeros(count * 3, 13)
        roots = env._root_states.view(count, 3, 13)
        roots[:, :, 6] = 1
        roots[:, :, 7:] = 7
        env._robot_root_states = roots[:, 0]
        env._cube_root_states = roots[:, 1]
        env._table_root_states = roots[:, 2]
        env._cube_root_states[:, 2] = .6
        env._robot_actor_ids = torch.arange(count) * 3
        env._cube_actor_ids = env._robot_actor_ids + 1
        env._table_actor_ids = env._robot_actor_ids + 2
        env.table_heights = torch.full((count,), .5)
        env.table_dimz = .2
        env.init_height = .1

        env._dof_state = torch.zeros(count * 3, 2)
        dofs = env._dof_state.view(count, 3, 2)
        env._dof_pos, env._dof_vel = dofs[:, :, 0], dofs[:, :, 1]
        env._dof_pos[:] = torch.tensor([.2, -.3, -.4])
        env._dof_vel[:] = 9
        env.dof_limits_lower = torch.full((3,), -2.)
        env.dof_limits_upper = torch.full((3,), 2.)
        env.gripper_dof_pos = torch.zeros(count, 1)
        env.last_low_actions = torch.ones(count, 12)
        env._sgs_pending_reset = torch.ones(count, dtype=torch.bool)
        env._sgs_has_pending_reset = True

        names = (*FOOT_NAMES, "base", "gripperMover")
        env.body_names_to_idx = dict(zip(names, range(len(names))))
        env._rigid_body_pos = torch.zeros(count, len(names), 3)
        env._rigid_body_pos[:, :4, 2] = FOOT_RADIUS + .001
        env._contact_forces = torch.zeros(count, len(names), 3)
        env.ee_pos = torch.ones(count, 3)
        env.ee_orn = torch.zeros(count, 4)
        env.ee_orn[:, 3] = 1
        env.get_torques = Mock(return_value=torch.zeros(count, 3))
        env._refresh_sim_tensors = Mock()
        env.update_roboinfo = Mock()

        before_frames = []
        env.before_root_frames = []
        env.root_uploads = []

        def upload_roots(_sim, state, actor_ids, actor_count):
            self.assertEqual(actor_ids.numel(), actor_count)
            env.root_uploads.append((actor_ids.clone(), state[actor_ids.long()].clone()))

        def simulate(_sim):
            before_frames.append((env._dof_vel.clone(), roots[:, :, 7:].clone()))
            env.before_root_frames.append(roots.clone())
            frames[len(before_frames) - 1](env)

        env.gym = SimpleNamespace(
            set_dof_state_tensor_indexed=Mock(),
            set_actor_root_state_tensor_indexed=Mock(side_effect=upload_roots),
            set_dof_position_target_tensor=Mock(),
            set_dof_actuation_force_tensor=Mock(),
            simulate=Mock(side_effect=simulate),
            fetch_results=Mock(),
        )
        builder = object.__new__(bank_builder.PhysicsBankBuilder)
        builder.env = env
        builder.reset_config = {"validation": {
            "physics_steps": len(frames),
            "max_contact_force_n": 5000.,
            "max_object_displacement_m": .02,
        }}
        return builder, before_frames

    def run_window(self, builder):
        isaacgym = ModuleType("isaacgym")
        isaacgym.gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor)
        gripper = torch.full((builder.env.num_envs, 1), -.4)
        with patch.dict(sys.modules, {"isaacgym": isaacgym}), patch.object(
                bank_builder, "synchronize_targets") as sync:
            checks, evidence = builder._physics_window(gripper)
        sync.assert_called_once()
        args = sync.call_args.args
        self.assertIs(args[0], builder.env)
        torch.testing.assert_close(args[1], builder.env.ee_pos)
        torch.testing.assert_close(args[2], builder.env.ee_orn)
        self.assertNotEqual(args[1].data_ptr(), builder.env.ee_pos.data_ptr())
        self.assertNotEqual(args[2].data_ptr(), builder.env.ee_orn.data_ptr())
        torch.testing.assert_close(args[3], gripper)
        return checks, evidence

    def test_upload_once_preserves_velocities_and_sampled_joint_targets(self):
        def advance(env):
            env._dof_pos.add_(.1)
            env._dof_vel.add_(1)
            env._root_states[:, 7:].add_(1)
            env.ee_pos.add_(.1)

        builder, before_frames = self.make_builder([advance] * 3)
        env = builder.env
        initial_positions = env._dof_pos.clone()
        checks, _ = self.run_window(builder)

        env.gym.set_dof_state_tensor_indexed.assert_called_once()
        self.assertEqual(len(env.root_uploads), 3)
        torch.testing.assert_close(env.root_uploads[0][0], torch.cat((
            env._robot_actor_ids, env._cube_actor_ids, env._table_actor_ids)).to(torch.int32))
        for actor_ids, _ in env.root_uploads[1:]:
            torch.testing.assert_close(actor_ids, env._table_actor_ids.to(torch.int32))
        env.gym.set_dof_position_target_tensor.assert_called_once()
        for callback in (env.gym.simulate, env.gym.fetch_results,
                         env.gym.set_dof_actuation_force_tensor,
                         env.get_torques, env._refresh_sim_tensors):
            self.assertEqual(callback.call_count, 3)
        env.update_roboinfo.assert_called_once()
        for step, (joint_velocity, root_velocity) in enumerate(before_frames):
            torch.testing.assert_close(joint_velocity, torch.full_like(joint_velocity, step))
            torch.testing.assert_close(root_velocity[:, :2],
                                       torch.full_like(root_velocity[:, :2], step))
            torch.testing.assert_close(root_velocity[:, 2], torch.zeros_like(root_velocity[:, 2]))
        torch.testing.assert_close(env._dof_vel, torch.full_like(env._dof_vel, 3))
        targets = env.gym.set_dof_position_target_tensor.call_args.args[1]
        torch.testing.assert_close(targets, initial_positions)
        torch.testing.assert_close(env._dof_pos, initial_positions + .3)
        self.assertNotEqual(targets.data_ptr(), env._dof_pos.data_ptr())
        self.assertTrue(targets.is_contiguous())
        self.assertFalse(env._sgs_has_pending_reset)
        self.assertFalse(bool(env._sgs_pending_reset.any()))
        self.assertEqual(int(torch.count_nonzero(env.last_low_actions)), 0)
        self.assertTrue(all(bool(value.all()) for value in checks.values()))

    def test_falling_table_is_supported_each_tick_without_resetting_robot_or_object(self):
        def advance(env):
            env._table_root_states[:, 2] -= .05
            env._table_root_states[:, 9] = -1
            env._robot_root_states[:, 0] += .01
            env._robot_root_states[:, 7:] += 1
            env._cube_root_states[:, 0] += .005
            env._cube_root_states[:, 7:] += 1

        builder, _ = self.make_builder([advance] * 3)
        env = builder.env
        # Even the first tick must place the tabletop at its configured height.
        env._table_root_states[:, 2] = -.2
        checks, _ = self.run_window(builder)
        expected_height = env.table_heights - env.table_dimz / 2
        for step, roots in enumerate(env.before_root_frames):
            torch.testing.assert_close(roots[:, 2, 2], expected_height)
            torch.testing.assert_close(roots[:, 2, 7:], torch.zeros_like(roots[:, 2, 7:]))
            torch.testing.assert_close(roots[:, 0, 0], torch.full_like(expected_height, step * .01))
            torch.testing.assert_close(roots[:, 1, 0], torch.full_like(expected_height, step * .005))
            torch.testing.assert_close(roots[:, :2, 7:], torch.full_like(roots[:, :2, 7:], step))
        self.assertEqual(len(env.root_uploads), 3)
        for actor_ids, states in env.root_uploads[1:]:
            torch.testing.assert_close(actor_ids, env._table_actor_ids.to(torch.int32))
            torch.testing.assert_close(states[:, 2], expected_height)
            torch.testing.assert_close(states[:, 7:], torch.zeros_like(states[:, 7:]))
        env.gym.set_dof_state_tensor_indexed.assert_called_once()
        torch.testing.assert_close(env._robot_root_states[:, 0], torch.full_like(expected_height, .03))
        torch.testing.assert_close(env._cube_root_states[:, 0], torch.full_like(expected_height, .015))
        self.assertTrue(all(bool(value.all()) for value in checks.values()))

    def test_early_body_and_gripper_collisions_survive_a_clean_last_frame(self):
        def collision(env):
            env._contact_forces[0, env.body_names_to_idx["base"], 0] = 10
            env._contact_forces[1, env.body_names_to_idx["gripperMover"], 0] = 300

        def recover(env):
            env._contact_forces.zero_()

        builder, _ = self.make_builder([collision, recover])
        checks, _ = self.run_window(builder)
        self.assertEqual(checks["non_grasp_collision"].tolist(), [False, True, True])
        self.assertEqual(checks["grasp_collision"].tolist(), [True, False, True])
        self.assertTrue(bool(checks["peak_contact_force"].all()))
        self.assertEqual(int(torch.count_nonzero(builder.env._contact_forces)), 0)

    def test_force_and_displacement_peaks_survive_final_recovery(self):
        def disturbance(env):
            env._contact_forces[0, 0, 2] = 6000
            env._cube_root_states[1, 0] = .03

        def recover(env):
            env._contact_forces.zero_()
            env._cube_root_states[:, 0] = 0

        builder, _ = self.make_builder([disturbance, recover])
        checks, evidence = self.run_window(builder)
        self.assertEqual(checks["peak_contact_force"].tolist(), [False, True, True])
        self.assertEqual(checks["object_displacement"].tolist(), [True, False, True])
        torch.testing.assert_close(evidence["validation_peak_contact_force_n"],
                                   torch.tensor([6000., 0., 0.]))
        torch.testing.assert_close(evidence["validation_max_object_displacement_m"],
                                   torch.tensor([0., .03, 0.]))
        self.assertTrue(bool(checks["object_on_table"].all()))
        self.assertTrue(bool(checks["non_grasp_collision"].all()))

    def test_only_final_ground_proximity_is_required_without_standing_stability(self):
        def early_airborne(env):
            env._rigid_body_pos[:, :4, 2] = FOOT_RADIUS + .1

        def final_frame(env):
            env._rigid_body_pos[:, :4, 2] = FOOT_RADIUS + .001
            env._rigid_body_pos[1, 0, 2] = FOOT_RADIUS + .03
            # No support forces and high finite velocity intentionally fail a
            # stable-standing criterion; only final foot geometry is required.
            env._robot_root_states[:, 7:] = 5
            env._dof_vel[:] = 5

        builder, _ = self.make_builder([early_airborne, final_frame])
        checks, _ = self.run_window(builder)
        self.assertEqual(checks["feet_near_ground"].tolist(), [True, False, True])
        for name, values in checks.items():
            if name != "feet_near_ground":
                self.assertTrue(bool(values.all()), name)
        self.assertEqual(int(torch.count_nonzero(builder.env._contact_forces)), 0)


if __name__ == "__main__":
    unittest.main()
