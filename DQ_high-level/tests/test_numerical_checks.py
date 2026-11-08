"""CPU checks of the actual simulator-entry methods without importing Isaac Gym."""

import ast
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.numerical_checks import require_finite_tensors


def quat_apply(quat, vector):
    uv = torch.cross(quat[..., :3], vector, dim=-1)
    return vector + 2 * (quat[..., 3:] * uv + torch.cross(quat[..., :3], uv, dim=-1))


def production_method(name):
    path = ROOT / "envs" / "b1z1_base.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "B1Z1Base")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = dict(torch=torch, random=random, time=time, quat_apply=quat_apply,
                     quat_rotate=quat_apply, reindex_all=lambda tensor: tensor,
                     gymtorch=SimpleNamespace(unwrap_tensor=lambda tensor: tensor))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace[name]


class CPUEnvironment:
    step = production_method("step")
    floating_table_control = production_method("floating_table_control")
    floating_base_control = production_method("floating_base_control")

    def __init__(self, floating=False):
        n = self.num_envs = 3
        self.global_step_counter = 144
        self.device = self.rl_device = "cpu"
        self.sim = object()
        self.events = []
        self.gym = SimpleNamespace(**{
            name: Mock(side_effect=lambda *args, _name=name: self.events.append(_name))
            for name in ("set_dof_position_target_tensor", "set_dof_actuation_force_tensor",
                         "set_actor_root_state_tensor_indexed", "set_dof_state_tensor_indexed",
                         "simulate", "fetch_results")})
        self._root_states = torch.zeros(n * 3, 13)
        self._root_states[:, 6] = 1
        self._robot_actor_ids = torch.arange(n) * 3
        self._table_actor_ids = self._robot_actor_ids + 1
        self._cube_actor_ids = self._robot_actor_ids + 2
        self._robot_root_states = self._root_states.view(n, 3, 13)[:, 0]
        self._dof_state = torch.zeros(n * 19, 2)
        self._sgs_has_pending_reset = False
        self._sgs_pending_reset = torch.zeros(n, dtype=torch.bool)
        self.table_level = "Level00"
        self.compute_table_command_level_00 = Mock()
        self.table_commands = torch.zeros(n, 6)
        self.table_dimz = .05
        self.table_z_lower, self.table_z_upper = .5, .6
        self.commands = torch.zeros(n, 3)
        self.targets, self.torques = torch.zeros(n, 19), torch.zeros(n, 19)
        self.get_all_pos_targets = Mock(side_effect=lambda *args: self.targets)
        self.get_torques = Mock(side_effect=lambda: self.torques)
        self.pred_success = self.rand_control = False
        self.dr_randomizations = {}
        self.clip_actions = 1.
        self.pre_physics_step = Mock(return_value=(torch.zeros(n, 3), torch.ones(n, 4)))
        self.img_delay_frame = 0
        self.control_freq_low, self.control_freq_inv = 2, 2
        self.floating_base = floating
        self._compute_low_level_observations = Mock()
        self.low_obs_buf = torch.zeros(n, 71)
        self.low_level_policy = Mock(return_value=torch.zeros(n, 12))
        self.last_low_actions = torch.zeros(n, 12)
        self._refresh_sim_tensors = Mock()
        self.headless, self.enable_camera, self.camera_test = True, False, False
        self.one_epoch_step = 0
        self.post_physics_step = Mock()
        self.extras, self.obs_dict = {}, {}
        self.obs_buf = torch.zeros(n, 10)
        self.clip_obs = 100.
        self.num_states = 0
        self.rew_buf = torch.zeros(n)
        self.reset_buf = self.timeout_buf = torch.zeros(n, dtype=torch.bool)


class FiniteTensorTests(unittest.TestCase):
    def test_finite_check_preserves_values_gradients_and_rng(self):
        values = torch.randn(4, 6, requires_grad=True)
        before, rng = values.detach().clone(), torch.get_rng_state().clone()
        require_finite_tensors({"values": values, "empty": torch.empty(0, 3)},
                               phase="test", step=1)
        torch.testing.assert_close(values, before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        values.square().sum().backward()
        torch.testing.assert_close(values.grad, 2 * before)

    def test_failure_lists_all_bad_tensors_and_mapped_environment_ids(self):
        one = torch.tensor([[float("nan"), 0], [0, 1.]])
        two = torch.tensor([[0, float("inf")], [float("-inf"), 0.]])
        with self.assertRaises(FloatingPointError) as caught:
            require_finite_tensors({"targets": one, "roots": two},
                phase="physx.upload", step="high=144, low=2, physics=3",
                env_ids={"roots": torch.tensor([17, 29])})
        message = str(caught.exception)
        for text in ("phase=physx.upload", "step=high=144, low=2, physics=3",
                     "targets: env_ids=[0]", "roots: env_ids=[17, 29]"):
            self.assertIn(text, message)

    def test_scalar_vector_empty_and_invalid_row_mapping(self):
        require_finite_tensors({}, phase="empty")
        require_finite_tensors({"scalar": torch.tensor(1.), "vector": torch.ones(3)}, phase="ok")
        for value in (torch.tensor(float("nan")), torch.tensor([0., float("inf")])):
            with self.assertRaises(FloatingPointError):
                require_finite_tensors({"value": value}, phase="bad")
        with self.assertRaisesRegex(ValueError, "map every row"):
            require_finite_tensors({"x": torch.zeros(2, 3)}, phase="shape",
                                   env_ids={"x": torch.tensor([5])})


class SimulatorBoundaryTests(unittest.TestCase):
    def test_nan_and_both_infinities_rejected_before_clamp_or_pre_physics(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                env = CPUEnvironment()
                actions = torch.zeros(3, 9)
                actions[2, 0] = value
                with self.assertRaisesRegex(FloatingPointError,
                        r"phase=env.step.actions; step=high=144; actions: env_ids=\[2\]"):
                    env.step(actions)
                env.pre_physics_step.assert_not_called()
                self.assertEqual(env.events, [])

    def test_randomization_cannot_introduce_nonfinite_actions(self):
        env = CPUEnvironment()
        env.dr_randomizations = {"actions": {"noise_lambda": lambda value: value * float("inf")}}
        with self.assertRaisesRegex(FloatingPointError, "phase=env.step.randomized_actions"):
            env.step(torch.zeros(3, 9))
        env.pre_physics_step.assert_not_called()
        self.assertEqual(env.events, [])

    def test_targets_and_torques_checked_before_any_upload_or_simulate(self):
        for name, value in (("targets", float("nan")), ("torques", float("inf"))):
            with self.subTest(name=name):
                env = CPUEnvironment()
                getattr(env, name)[1, 5] = value
                with self.assertRaisesRegex(FloatingPointError,
                        r"phase=physx.upload; step=high=144, low=0, physics=0; dof_.*env_ids=\[1\]"):
                    env.step(torch.zeros(3, 9))
                self.assertEqual(env.events, [])

    def test_finite_substeps_aggregate_checks_and_preserve_submission_count(self):
        env = CPUEnvironment()
        with patch("utils.numerical_checks.require_finite_tensors",
                   wraps=require_finite_tensors) as check:
            env.step(torch.full((3, 9), 2.))
        self.assertEqual(check.call_count, 5)  # One action check plus four physical substeps.
        for call in check.call_args_list[1:]:
            self.assertEqual(set(call.args[0]), {"dof_targets", "dof_torques", "actor_root_states"})
        torch.testing.assert_close(env.pre_physics_step.call_args.args[0], torch.ones(3, 9))
        for name in ("set_dof_position_target_tensor", "set_dof_actuation_force_tensor",
                     "set_actor_root_state_tensor_indexed", "simulate"):
            self.assertEqual(getattr(env.gym, name).call_count, 4)
        self.assertEqual(env.events[:4], ["set_actor_root_state_tensor_indexed",
            "set_dof_position_target_tensor", "set_dof_actuation_force_tensor", "simulate"])

    def test_nonfinite_uploaded_roots_block_all_setters(self):
        env = CPUEnvironment()
        env._root_states[env._table_actor_ids[2], 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, r"actor_root_states: env_ids=\[2\]"):
            env.step(torch.zeros(3, 9))
        self.assertEqual(env.events, [])

    def test_reset_dof_failure_preserves_pending_reset_and_blocks_all_setters(self):
        env = CPUEnvironment()
        env._sgs_has_pending_reset = True
        env._sgs_pending_reset[2] = True
        env._dof_state.view(3, 19, 2)[2, 3, 1] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, r"reset_dof_state: env_ids=\[2\]"):
            env.floating_table_control()
        self.assertEqual(env.events, [])
        self.assertTrue(env._sgs_has_pending_reset)
        self.assertTrue(env._sgs_pending_reset[2])

    def test_only_uploaded_roots_checked_then_reset_adds_robot_and_object_rows(self):
        env = CPUEnvironment()
        env._root_states[env._cube_actor_ids[1], 0] = float("inf")
        env.floating_table_control()  # Object row is not part of this indexed upload.
        env.events.clear()
        env._sgs_has_pending_reset = True
        env._sgs_pending_reset[1] = True
        with self.assertRaisesRegex(FloatingPointError, r"actor_root_states: env_ids=\[1\]"):
            env.floating_table_control()
        self.assertEqual(env.events, [])

    def test_finite_reset_preserves_indexed_payloads_and_clears_pending_reset(self):
        env = CPUEnvironment()
        env._sgs_has_pending_reset = True
        env._sgs_pending_reset[[0, 2]] = True
        env.floating_table_control()
        self.assertFalse(env._sgs_has_pending_reset)
        self.assertFalse(bool(env._sgs_pending_reset.any()))
        torch.testing.assert_close(env.gym.set_dof_state_tensor_indexed.call_args.args[2],
                                   torch.tensor([0, 6], dtype=torch.int32))
        torch.testing.assert_close(env.gym.set_actor_root_state_tensor_indexed.call_args.args[2],
                                   torch.tensor([1, 4, 7, 0, 6, 2, 8], dtype=torch.int32))
        self.assertEqual(env.events, ["set_dof_state_tensor_indexed",
                                     "set_actor_root_state_tensor_indexed"])

    def test_floating_base_targets_and_roots_checked_before_setters(self):
        for name in ("targets", "roots"):
            env = CPUEnvironment(floating=True)
            if name == "targets":
                env.targets[1, 0] = float("nan")
            else:
                env._root_states[env._robot_actor_ids[1], 0] = float("nan")
            with self.assertRaisesRegex(FloatingPointError, "physx.floating_base_upload"):
                env.step(torch.zeros(3, 9))
            self.assertEqual(env.events, [])

    def test_floating_base_held_targets_checked_before_each_simulation(self):
        env = CPUEnvironment(floating=True)
        env._refresh_sim_tensors.side_effect = lambda: env.targets.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError,
                "phase=physx.simulate; step=high=144, low=0, physics=1"):
            env.step(torch.zeros(3, 9))
        self.assertEqual(env.gym.simulate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
