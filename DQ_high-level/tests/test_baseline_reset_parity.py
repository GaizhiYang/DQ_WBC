"""Exercise ordinary 633 reset ordering without loading Isaac Gym.

Both production reset layers execute; only simulator submission, actor sampling,
and observations are replaced. This locks down the behaviors that previously
changed implicitly when --success_mode hold was supplied.
"""

import ast
from types import SimpleNamespace
import unittest

import torch

from test_sgs_env import NAMESPACE, TestEnvironment, method_node


def reset_environment_class():
    base = ast.ClassDef(
        name="B1Z1Base", bases=[ast.Name(id="TestEnvironment", ctx=ast.Load())],
        keywords=[], decorator_list=[], body=[
            method_node("b1z1_base.py", "B1Z1Base", name)
            for name in ("_reset_envs", "_reset_env_tensors")])
    task = ast.ClassDef(
        name="B1Z1PickMulti", bases=[ast.Name(id="B1Z1Base", ctx=ast.Load())],
        keywords=[], decorator_list=[], body=[
            method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name)
            for name in ("_reset_envs", "_reset_env_tensors")])
    namespace = dict(NAMESPACE, TestEnvironment=TestEnvironment)
    module = ast.Module(body=[base, task], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), __file__, "exec"), namespace)
    return namespace["B1Z1PickMulti"]


ResetEnvironment = reset_environment_class()


class BaselineResetParityTests(unittest.TestCase):
    def environment(self, mode):
        env = ResetEnvironment(num_envs=2)
        env.sgs_enabled = False
        env.success_mode = mode
        env.local_step_counter = 1
        env.randomize = False
        env.episode_counter[:] = torch.tensor([2, 3])
        env.success_counter[:] = torch.tensor([5, 7])
        env.success_onetime_counter[:] = torch.tensor([4, 6])
        env.curr_ee_goal_cart[:] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        env.curr_ee_goal_orn_rpy.fill_(2.)
        env.init_ee_goal_cart = torch.tensor([[.46, 0., .66]]).repeat(2, 1)
        env.rand_change_step[:] = torch.tensor([41, 57])
        env.gait_indices = torch.tensor([.25, .75])
        env.clock_inputs = torch.ones(2, 4)
        env.gait_wait_timer = torch.tensor([1, 2])
        env.is_walking = torch.ones(2, dtype=torch.bool)
        env.low_obs_history_buf.fill_(8.)
        env.actions.fill_(9.)
        env.gripper_dof_pos.fill_(-.3)
        env.commands.fill_(.4)
        env.progress_buf.fill_(12)
        env.pick_counter.fill_(7)
        env.best_pick_counter.fill_(9)
        for name in ("lifted_object", "lifted_now", "pick_success",
                     "pick_success_recorded", "pick_hold_progress", "episode_outcome_recorded"):
            getattr(env, name).fill_(True)
        env.events = []
        env._reset_actors = lambda ids: env.events.append("actors")
        env._refresh_sim_tensors = lambda: env.events.append("refresh")
        env.render = lambda: env.events.append("render")

        def observations(ids):
            env.events.append("observe")
            env.observed_goals = env.curr_ee_goal_cart[ids].clone()
            env.observed_orn = env.curr_ee_goal_orn_rpy[ids].clone()

        env._compute_observations = observations
        # Deliberately expose only the setters used by the original reset.
        # Extra target/force setters introduced by hold mode fail this test.
        env.gym = SimpleNamespace(
            set_dof_state_tensor_indexed=lambda *args: env.events.append("dof_state"),
            set_actor_root_state_tensor_indexed=lambda *args: env.events.append("actor_state"),
            simulate=lambda *args: env.events.append("simulate"),
        )
        return env

    def test_hold_and_legacy_preserve_original_simulation_and_observation_order(self):
        for mode in ("hold", "legacy"):
            with self.subTest(mode=mode):
                env = self.environment(mode)
                cached_names = ("rand_change_step", "gait_indices", "clock_inputs",
                                "gait_wait_timer", "is_walking", "low_obs_history_buf",
                                "actions", "gripper_dof_pos")
                before = {name: getattr(env, name).clone() for name in cached_names}
                env._reset_envs(torch.tensor([0]))
                self.assertEqual(env.events, ["actors", "dof_state", "actor_state", "simulate",
                                              "refresh", "render", "refresh", "observe"])
                # 633 computes the initial observation before restoring the
                # command goal; alignment must not silently change this order.
                torch.testing.assert_close(env.observed_goals, torch.tensor([[1., 2., 3.]]))
                torch.testing.assert_close(env.observed_orn, torch.full((1, 3), 2.))
                torch.testing.assert_close(env.curr_ee_goal_cart[0], env.init_ee_goal_cart[0])
                torch.testing.assert_close(env.curr_ee_goal_cart[1], torch.tensor([4., 5., 6.]))
                for name, value in before.items():
                    torch.testing.assert_close(getattr(env, name), value, msg=name)

    def test_reset_clears_selected_pick_history_without_counting_success_again(self):
        env = self.environment("hold")
        env._reset_envs(torch.tensor([0]))
        self.assertEqual(env.episode_counter.tolist(), [3, 3])
        self.assertEqual(env.success_counter.tolist(), [5, 7])
        self.assertEqual(env.success_onetime_counter.tolist(), [4, 6])
        self.assertEqual(env.progress_buf.tolist(), [0, 12])
        self.assertEqual(env.pick_counter.tolist(), [0, 7])
        self.assertEqual(env.best_pick_counter.tolist(), [0, 9])
        for name in ("lifted_object", "lifted_now", "pick_success", "pick_success_recorded",
                     "pick_hold_progress", "episode_outcome_recorded"):
            self.assertEqual(getattr(env, name).tolist(), [False, True], name)
        self.assertEqual(env.episode_success.data_ptr(), env.pick_success_recorded.data_ptr())
        self.assertEqual(env.episode_success.tolist(), [False, True])
        torch.testing.assert_close(env.commands[0], torch.zeros(3))
        torch.testing.assert_close(env.commands[1], torch.full((3,), .4))


if __name__ == "__main__":
    unittest.main()
