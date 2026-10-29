"""Exercise actual task termination/rewards on CPU without constructing a simulator.

Run this module in the Isaac Gym environment as its own process: Isaac Gym must
be imported before torch. No task constructor, model, or physics scene is used.
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from isaacgym import gymapi  # Must precede importing torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.b1z1_pickmulti import B1Z1PickMulti
from utils.pick_success import RollingEpisodeSuccess


def make_task(eval_mode=False, reward_scales=None, num_envs=1, wandb_enabled=False):
    """Supply only state used by the real termination and reward methods."""
    task = object.__new__(B1Z1PickMulti)
    task.device = "cpu"
    task.num_environments = num_envs
    task.eval = eval_mode
    task.pred_success = False
    task.enable_camera = False
    task.cfg = {"env": {"wandb": wandb_enabled}, "reward": {
        "only_positive_rewards": False, "base_height_target": .55,
    }}
    task.hold_steps = 10
    task.max_episode_length = 100
    task.lifted_success_threshold = .20
    task.table_heights = torch.full((num_envs,), .5)
    task.init_height = torch.full((num_envs,), .045)
    task._robot_root_states = torch.zeros((num_envs, 13))
    task._robot_root_states[:, 2] = .55
    task._robot_root_states[:, 6] = 1.
    task.initial_yaw_z = torch.zeros(num_envs)
    task._cube_root_states = torch.zeros((num_envs, 13))
    task._cube_root_states[:, 2] = .755  # Sugar box lifted by 21 cm.
    task._cube_root_states[:, 6] = 1.
    task.arm_base = task._robot_root_states[:, :3].clone()
    task.ee_pos = task._cube_root_states[:, :3].clone()
    task.ee_pos[:, 0] += .05
    task.curr_ee_goal_cart = task.ee_pos - task.arm_base
    task.actions = torch.zeros((num_envs, 9))
    task.actions[:, 6] = -1.
    for name in ("progress_buf", "reset_buf", "_terminate_buf", "pick_counter",
                 "best_pick_counter", "success_counter", "success_onetime_counter"):
        setattr(task, name, torch.zeros(num_envs, dtype=torch.long))
    for name in ("timeout_buf", "lifted_now", "lifted_object", "pick_success",
                 "pick_success_recorded", "pick_hold_progress", "episode_outcome_recorded"):
        setattr(task, name, torch.zeros(num_envs, dtype=torch.bool))
    task.obj_list = ["sugar_box"]
    task.extras = {}
    task.success_window = RollingEpisodeSuccess(1000)
    task.surpass_dist_flag = torch.ones(num_envs, dtype=torch.bool)
    task.rew_buf = torch.zeros(num_envs)
    task.reward_scales = dict({"pick_up": 3.5} if reward_scales is None else reward_scales)
    task._prepare_reward_function()
    return task


def step_task(task):
    """Use the same check-before-reward order as post_physics_step."""
    task.progress_buf += 1
    task.check_termination()
    task.compute_reward()
    return task.rew_buf.clone()


class PickTaskTests(unittest.TestCase):
    def test_train_and_eval_require_ten_steps_and_award_success_only_once(self):
        traces = []
        for eval_mode in (False, True):
            task = make_task(eval_mode=eval_mode)
            trace = []
            for step in range(1, 11):
                reward = step_task(task).item()
                trace.append((reward, task.reset_buf.item(), task.success_counter.item()))
                self.assertEqual(task.pick_counter.item(), step)
                self.assertEqual(reward, 3.5 if step == 10 else 0.)
                self.assertEqual(bool(task.reset_buf.item()), step == 10)
                self.assertEqual(task.success_counter.item(), int(step == 10))
                self.assertFalse(task.timeout_buf.item())
            self.assertEqual(task.success_onetime_counter.item(), 1)
            self.assertEqual(step_task(task).item(), 0.)
            self.assertTrue(task.reset_buf.item())
            self.assertFalse(task.pick_success.item())
            self.assertFalse(task.timeout_buf.item())
            self.assertEqual(task.success_counter.item(), 1)
            self.assertEqual(task.episode_sums["pick_up"].item(), 3.5)
            self.assertEqual(task.extras["success_rate"]["SuccessWindow / completed"], 1)
            self.assertEqual(task.extras["success_rate"]["SuccessWindow / successes"], 1)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_yaw_failure_prevents_success_on_the_tenth_hold_step(self):
        task = make_task()
        for _ in range(9):
            step_task(task)
        half_yaw = math.radians(71.) / 2
        task._robot_root_states[0, 3:7] = torch.tensor(
            [0., 0., math.sin(half_yaw), math.cos(half_yaw)])
        self.assertEqual(step_task(task).item(), 0.)
        self.assertTrue(task._terminate_buf.item())
        self.assertTrue(task.reset_buf.item())
        self.assertFalse(task.pick_success.item())
        self.assertFalse(task.timeout_buf.item())
        self.assertEqual(task.success_counter.item(), 0)
        self.assertEqual(task.pick_counter.item(), 0)

    def test_success_on_steps_99_and_100_is_not_a_timeout(self):
        for completion_step in (99, 100):
            with self.subTest(completion_step=completion_step):
                task = make_task()
                task.progress_buf[:] = completion_step - 10
                for _ in range(9):
                    self.assertEqual(step_task(task).item(), 0.)
                    self.assertFalse(task.reset_buf.item())
                self.assertEqual(step_task(task).item(), 3.5)
                self.assertEqual(task.progress_buf.item(), completion_step)
                self.assertTrue(task.pick_success.item())
                self.assertTrue(task.reset_buf.item())
                self.assertFalse(task.timeout_buf.item())

    def test_incomplete_hold_at_step_100_is_a_timeout(self):
        task = make_task()
        task.progress_buf[:] = 91
        for _ in range(9):
            self.assertEqual(step_task(task).item(), 0.)
        self.assertEqual(task.progress_buf.item(), 100)
        self.assertEqual(task.pick_counter.item(), 9)
        self.assertTrue(task.timeout_buf.item())
        self.assertTrue(task.reset_buf.item())
        self.assertFalse(task.pick_success.item())
        self.assertEqual(task.success_counter.item(), 0)

    def test_zero_success_reward_still_completes_and_counts(self):
        task = make_task(reward_scales={"pick_up": 0.})
        self.assertNotIn("pick_up", task.reward_names)
        for _ in range(10):
            self.assertEqual(step_task(task).item(), 0.)
        self.assertTrue(task.reset_buf.item())
        self.assertTrue(task.pick_success.item())
        self.assertEqual(task.success_counter.item(), 1)
        self.assertFalse(task.timeout_buf.item())
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / completed"], 1)
        self.assertEqual(task.extras["success_rate"]["SuccessRate / sugar_box"], 1.)

    def test_old_subthreshold_window_has_no_holding_reward(self):
        task = make_task(reward_scales={"pick_up": 3.5, "grasp_base_height": .5})
        for center_height in (.715, .725, .744):
            task._cube_root_states[:, 2] = center_height
            for _ in range(12):
                self.assertEqual(step_task(task).item(), 0.)
                self.assertFalse(task.lifted_now.item())
                self.assertFalse(task.lifted_object.item())
                self.assertEqual(task.pick_counter.item(), 0)
        self.assertEqual(task.episode_sums["grasp_base_height"].item(), 0.)
        self.assertEqual(task.success_counter.item(), 0)

    def test_interrupted_nine_step_hold_cannot_collect_same_holding_rewards_again(self):
        task = make_task(reward_scales={"pick_up": 3.5, "grasp_base_height": .5})
        for _ in range(9):
            self.assertEqual(step_task(task).item(), .5)
        task._cube_root_states[:, 2] = .725
        self.assertEqual(step_task(task).item(), 0.)
        self.assertEqual(task.pick_counter.item(), 0)
        self.assertFalse(task.reset_buf.item())
        task._cube_root_states[:, 2] = .755
        for _ in range(9):
            self.assertEqual(step_task(task).item(), 0.)
            self.assertFalse(task.reset_buf.item())
        self.assertEqual(step_task(task).item(), 4.)
        self.assertEqual(task.episode_sums["grasp_base_height"].item(), 5.)
        self.assertEqual(task.episode_sums["pick_up"].item(), 3.5)
        self.assertEqual(task.success_counter.item(), 1)

    def test_actual_tensor_reset_clears_only_the_selected_episode(self):
        task = make_task(num_envs=2)
        for _ in range(10):
            step_task(task)
        # The tensor-reset method forwards tensors to Gym; replace only those
        # external side effects, leaving the production reset logic untouched.
        task.gym = SimpleNamespace(
            set_dof_state_tensor_indexed=lambda *args: None,
            set_actor_root_state_tensor_indexed=lambda *args: None,
            simulate=lambda *args: None,
        )
        task.sim = None
        task._refresh_sim_tensors = lambda: None
        task.render = lambda: None
        task._robot_actor_ids = torch.tensor([0, 3])
        task._table_actor_ids = torch.tensor([1, 4])
        task._cube_actor_ids = torch.tensor([2, 5])
        task._dof_state = torch.zeros((2, 20, 2))
        task._root_states = torch.zeros((6, 13))
        task.rand_cmd_scale = False
        task.object_pose_predictor = None
        for name in ("num_steps_to_change_mode", "increase_to_change_step",
                     "curr_dist", "closest_dist", "reach_counter",
                     "curr_height", "highest_object"):
            setattr(task, name, torch.ones(2))
        for name in ("last_actions", "last_low_actions", "clipped_actions", "commands"):
            setattr(task, name, torch.ones((2, 9)))
        task.action_history_buf = torch.ones((2, 3, 9))
        task.command_history_buf = torch.ones((2, 3, 3))
        task._reset_env_tensors(torch.tensor([0]))
        for name in ("lifted_object", "lifted_now", "pick_success",
                     "pick_success_recorded", "pick_hold_progress", "episode_outcome_recorded"):
            self.assertEqual(getattr(task, name).tolist(), [False, True], name)
        self.assertEqual(task.pick_counter.tolist(), [0, 10])
        self.assertEqual(task.best_pick_counter.tolist(), [0, 10])
        self.assertEqual(task.progress_buf.tolist(), [0, 10])
        self.assertEqual(task.success_counter.tolist(), [1, 1])
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / completed"], 2)
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / successes"], 2)
        # Initialization/reset at progress zero never creates another outcome.
        task.reset_buf[0] = 1
        task._record_episode_outcomes()
        self.assertEqual(task.episode_outcome_recorded.tolist(), [False, True])
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / completed"], 2)
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / successes"], 2)
        task.progress_buf[0] = 1
        task._record_episode_outcomes()
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / completed"], 3)
        self.assertEqual(task.extras["success_rate"]["SuccessWindow / successes"], 2)
        self.assertAlmostEqual(task.extras["success_rate"]["SuccessRate / sugar_box"], 2 / 3)

    def test_all_terminal_outcomes_reach_infos_on_same_step_in_every_mode(self):
        snapshots = []
        for eval_mode in (False, True):
            for wandb_enabled in (False, True):
                with self.subTest(eval_mode=eval_mode, wandb_enabled=wandb_enabled):
                    task = make_task(eval_mode=eval_mode, num_envs=3,
                                     wandb_enabled=wandb_enabled)
                    for _ in range(9):
                        step_task(task)
                    # Env 0 completes; env 1 times out one hold step short;
                    # env 2 violates the yaw limit on its would-be success step.
                    task.progress_buf[1] = 99
                    task.pick_counter[1] = 8
                    half_yaw = math.radians(71.) / 2
                    task._robot_root_states[2, 3:7] = torch.tensor(
                        [0., 0., math.sin(half_yaw), math.cos(half_yaw)])
                    task.progress_buf += 1
                    task.check_termination()
                    task.compute_reward()
                    rates = dict(task.extras["success_rate"])
                    self.assertEqual(rates["SuccessWindow / completed"], 3)
                    self.assertEqual(rates["SuccessWindow / successes"], 1)
                    self.assertAlmostEqual(rates["SuccessRate / sugar_box"], 1 / 3)
                    self.assertFalse(any(key.startswith("Onetime_") for key in rates))
                    self.assertEqual(task.episode_outcome_recorded.tolist(), [True] * 3)
                    self.assertEqual(task.timeout_buf.tolist(), [False, True, False])
                    self.assertEqual(task.pick_success.tolist(), [True, False, False])
                    self.assertEqual(task.rew_buf.tolist(), [3.5, 0., 0.])
                    # Recording outcomes again in the same episode must not
                    # append the same success, timeout, or hard failure twice.
                    task._record_episode_outcomes()
                    self.assertEqual(task.extras["success_rate"], rates)
                    snapshots.append(rates)
        self.assertTrue(all(snapshot == snapshots[0] for snapshot in snapshots))

    def test_unfinished_and_initial_resets_do_not_enter_the_window(self):
        task = make_task(num_envs=3)
        task.reset_buf[:] = 1  # Initialization reset, no step has occurred.
        task._record_episode_outcomes()
        self.assertFalse(any(key.startswith("SuccessRate /")
                             for key in task.extras.get("success_rate", {})))
        self.assertFalse(task.episode_outcome_recorded.any().item())
        task.progress_buf[:] = 1
        task.reset_buf[:] = torch.tensor([1, 0, 0])
        task.pick_success_recorded[:] = True
        task._record_episode_outcomes()
        rates = task.extras["success_rate"]
        self.assertEqual(rates["SuccessWindow / completed"], 1)
        self.assertEqual(rates["SuccessWindow / successes"], 1)
        self.assertEqual(rates["SuccessRate / sugar_box"], 1.)
        self.assertEqual(task.episode_outcome_recorded.tolist(), [True, False, False])

    def test_window_uses_available_completed_episodes_as_denominator(self):
        task = make_task(num_envs=3)
        task.progress_buf[:] = 1
        task.reset_buf[:] = 1
        task.pick_success_recorded[:] = torch.tensor([True, False, True])
        task._record_episode_outcomes()
        rates = task.extras["success_rate"]
        self.assertEqual(rates["SuccessWindow / completed"], 3)
        self.assertEqual(rates["SuccessWindow / successes"], 2)
        self.assertAlmostEqual(rates["SuccessRate / sugar_box"], 2 / 3)

    def test_reward_triggered_termination_is_recorded_after_rewards(self):
        task = make_task(reward_scales={"reach": 1., "pick_up": 3.5})
        task.progress_buf += 1
        task.check_termination()
        self.assertFalse(task.reset_buf.item())
        # Legacy optional reach shaping also sets done. It ends the episode,
        # but cannot count as completing the ten-step grasp-and-lift target.
        task.compute_reward()
        self.assertTrue(task.reset_buf.item())
        self.assertFalse(task.pick_success_recorded.item())
        rates = task.extras["success_rate"]
        self.assertEqual(rates["SuccessWindow / completed"], 1)
        self.assertEqual(rates["SuccessWindow / successes"], 0)
        self.assertEqual(rates["SuccessRate / sugar_box"], 0.)

    def test_global_window_keeps_last_thousand_and_groups_objects_within_it(self):
        task = make_task(num_envs=1002)
        task.obj_list = ["sugar_box", "mustard_bottle"]
        task.progress_buf[:] = 1
        task.reset_buf[:] = 1
        # Simultaneous completions append in env-id order: the two earliest
        # successes are evicted, leaving only env 1001 successful in the window.
        task.pick_success_recorded[[0, 1, 1001]] = True
        task._record_episode_outcomes()
        rates = task.extras["success_rate"]
        self.assertEqual(rates["SuccessWindow / completed"], 1000)
        self.assertEqual(rates["SuccessWindow / successes"], 1)
        self.assertEqual(rates["SuccessRate / sugar_box"], 0.)
        self.assertAlmostEqual(rates["SuccessRate / mustard_bottle"], 1 / 500)
        self.assertAlmostEqual(rates["SuccessRate / total"], 1 / 1000)


if __name__ == "__main__":
    unittest.main()
