"""633 terminal-time success accounting stays independent of reset sampling.

Run actual terminal/reward/window/reset methods with CPU state; no simulator,
policy, CUDA context, or external W&B run is created.
"""

from contextlib import redirect_stdout
import io
import unittest

import torch

from test_sgs_env import BaseStandIn, TestEnvironment, source_method


class AccountingEnv(TestEnvironment):
    _reset_envs = source_method("b1z1_base.py", "B1Z1Base", "_reset_envs")

    def __init__(self, num_envs=2, evaluate=False, hold_steps=10):
        super().__init__(num_envs=num_envs)
        self.eval = evaluate
        self.hold_steps = hold_steps
        self.local_step_counter = 1
        self.randomize = False
        self.init_ee_goal_cart = torch.tensor([[.46, 0., .66]]).repeat(num_envs, 1)
        with redirect_stdout(io.StringIO()):
            self.initialize(ids=tuple(range(num_envs)))
        self.sgs_enabled = False

    def _reset_env_tensors(self, ids):
        BaseStandIn._reset_env_tensors(self, ids)
        self._reset_pick_state(ids)

    def _reset_actors(self, ids):
        self.surpass_dist_flag[ids] = True

    def _refresh_sim_tensors(self):
        self._update_curr_dist()

    def reset_ids(self, *ids):
        self._reset_envs(torch.tensor(ids, dtype=torch.long))


class SuccessRateTests(unittest.TestCase):
    def test_initial_reset_does_not_fabricate_outcomes(self):
        env = AccountingEnv()
        env.episode_counter.fill_(-1)
        env.pick_success_recorded.fill_(True)
        env.reset_ids(0, 1)
        self.assertEqual(env.episode_counter.tolist(), [0, 0])
        self.assertEqual(env.success_counter.tolist(), [0, 0])
        self.assertFalse(env.pick_success_recorded.any().item())
        self.assertEqual(env.success_window.metrics(env.obj_list)["SuccessWindow / completed"], 0)

    def test_success_is_counted_and_logged_on_terminal_step_before_reset(self):
        env = AccountingEnv()
        for _ in range(9):
            self.assertEqual(env.grasp_step([True, False]).tolist(), [0., 0.])
            self.assertEqual(env.success_counter.tolist(), [0, 0])
        self.assertEqual(env.grasp_step([True, False]).tolist(), [1., 0.])
        self.assertEqual(env.success_counter.tolist(), [1, 0])
        self.assertEqual(env.episode_counter.tolist(), [0, 0])
        self.assertEqual(env.extras["success_rate"]["SuccessRate / sugar_box"], 1.)
        self.assertEqual(env.extras["success_rate"]["SuccessWindow / completed"], 1)
        env.reset_ids(0)
        self.assertEqual(env.success_counter.tolist(), [1, 0])
        self.assertEqual(env.episode_counter.tolist(), [1, 0])
        self.assertFalse(env.pick_success_recorded.any().item())
        # The remaining environment times out without completing its hold.
        env.progress_buf[1] = env.max_episode_length
        env.check_termination()
        env._record_episode_outcomes()
        self.assertEqual(env.extras["success_rate"]["SuccessWindow / completed"], 2)
        self.assertEqual(env.extras["success_rate"]["SuccessRate / sugar_box"], .5)
        env.reset_ids(1)
        self.assertEqual(env.success_counter.tolist(), [1, 0])

    def test_reset_selected_environment_clears_hold_state_without_recounting(self):
        env = AccountingEnv()
        for _ in range(10):
            env.grasp_step()
        env.reset_ids(0)
        for name in ("pick_success", "pick_success_recorded", "pick_hold_progress",
                     "episode_outcome_recorded"):
            self.assertEqual(getattr(env, name).tolist(), [False, True], name)
        self.assertEqual(env.pick_counter.tolist(), [0, 10])
        self.assertEqual(env.best_pick_counter.tolist(), [0, 10])
        self.assertEqual(env.success_counter.tolist(), [1, 1])
        env.reset_ids(1)
        self.assertEqual(env.success_counter.tolist(), [1, 1])
        self.assertEqual(env.extras["success_rate"]["SuccessWindow / completed"], 2)

    def test_train_eval_and_configured_hold_use_identical_reward_threshold(self):
        for evaluate in (False, True):
            for hold_steps in (3, 10, 20):
                with self.subTest(evaluate=evaluate, hold_steps=hold_steps):
                    env = AccountingEnv(evaluate=evaluate, hold_steps=hold_steps)
                    for step in range(1, hold_steps + 1):
                        reward = env.grasp_step()
                        completed = step == hold_steps
                        self.assertEqual(reward.tolist(), [float(completed)] * 2)
                        self.assertEqual(env.reset_buf.bool().tolist(), [completed] * 2)
                        self.assertEqual(env.success_counter.tolist(), [int(completed)] * 2)
                    self.assertEqual(env.grasp_step().tolist(), [0., 0.])
                    self.assertEqual(env.success_counter.tolist(), [1, 1])

    def test_interrupted_holds_do_not_count_and_milestones_are_not_repaid(self):
        env = AccountingEnv()
        for lifted in [True] * 9 + [False] + [True] * 9:
            self.assertEqual(env.grasp_step(lifted).tolist(), [0., 0.])
        self.assertEqual(env.pick_counter.tolist(), [9, 9])
        self.assertFalse(env.pick_hold_progress.any().item())
        self.assertEqual(env.success_counter.tolist(), [0, 0])
        self.assertEqual(env.grasp_step().tolist(), [1., 1.])


if __name__ == "__main__":
    unittest.main()
