"""CPU regression tests for episode success accounting.

Extract the actual methods with AST so importing this test does not import Isaac
Gym or create a simulator. Physics, observation refresh, and actor resets are
stubbed; reward conditions, counter settlement, tensor clearing, and the logged
single-object success rate execute the production methods unchanged.
"""

import ast
import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ENV_DIR = Path(__file__).resolve().parents[1] / "envs"


def source_method(filename, class_name, method_name):
    path = ENV_DIR / filename
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {"torch": torch, "np": SimpleNamespace(pi=math.pi)}
    extracted = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace[method_name]


class AccountingEnv:
    _reward_pick_up = source_method("reward_vec_task.py", "RewardVecTask", "_reward_pick_up")
    _reset_envs = source_method("b1z1_base.py", "B1Z1Base", "_reset_envs")
    _reset_env_tensors = source_method("b1z1_base.py", "B1Z1Base", "_reset_env_tensors")
    _record_single_object_success = source_method(
        "b1z1_pickmulti.py", "B1Z1PickMulti", "_record_single_object_success")

    def __init__(self, num_envs=1, evaluate=False, hold_steps=15):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.eval = evaluate
        self.hold_steps = hold_steps
        self.global_step_counter = 1
        self.local_step_counter = 1
        self.randomize = False
        self.rand_cmd_scale = False
        self.pred_success = False
        self.obj_list = ["sugar_box"]
        self.cfg = {"env": {"wandb": True}}
        self.extras = {}

        for name in ("episode_counter", "success_counter", "success_onetime_counter",
                     "pick_counter", "reach_counter", "progress_buf", "reset_buf",
                     "_terminate_buf", "num_steps_to_change_mode", "increase_to_change_step"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long))
        self.episode_counter.fill_(-1)
        self.reset_buf.fill_(1)
        for name in ("episode_success", "episode_onetime_success", "lifted_object"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool))
        self.surpass_dist_flag = torch.ones(num_envs, dtype=torch.bool)
        for name in ("curr_dist", "closest_dist"):
            setattr(self, name, torch.zeros(num_envs))
        for name in ("last_actions", "last_low_actions", "clipped_actions", "commands",
                     "curr_ee_goal_cart", "init_ee_goal_cart", "curr_ee_goal_orn_rpy"):
            setattr(self, name, torch.zeros(num_envs, 3))
        self.action_history_buf = torch.zeros(num_envs, 3, 3)
        self.command_history_buf = torch.zeros(num_envs, 3, 3)

    def _reset_actors(self, env_ids):
        # The real actor reset changes this flag before resetting other tensors.
        self.surpass_dist_flag[env_ids] = True

    def _refresh_sim_tensors(self):
        pass

    def _compute_observations(self, env_ids):
        pass

    def reset_ids(self, *ids):
        self._reset_envs(torch.tensor(ids, dtype=torch.long))

    def step_lift(self, *lifted):
        self.global_step_counter += 1
        self.progress_buf += 1
        self.lifted_object[:] = torch.tensor(lifted, dtype=torch.bool)
        reward, _ = self._reward_pick_up()
        return reward

    def logged_rate(self, one_time=False):
        with contextlib.redirect_stdout(io.StringIO()):
            self._record_single_object_success(one_time=one_time)
        prefix = "Onetime_SuccessRate" if one_time else "SuccessRate"
        return self.extras["success_rate"][f"{prefix} / sugar_box"]


class SuccessRateTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = AccountingEnv(**kwargs)
        env.reset_ids(*range(env.num_envs))
        return env

    def test_initial_reset_does_not_count_a_completed_episode(self):
        env = AccountingEnv(num_envs=2)
        # Even stale flags must not turn initialization into a successful episode.
        env.episode_success[:] = True
        env.episode_onetime_success[:] = True
        env.reset_ids(0, 1)
        self.assertEqual(env.episode_counter.tolist(), [0, 0])
        self.assertEqual(env.success_counter.tolist(), [0, 0])
        self.assertEqual(env.success_onetime_counter.tolist(), [0, 0])
        self.assertFalse(env.episode_success.any().item())
        self.assertFalse(env.episode_onetime_success.any().item())
        self.assertEqual(env.logged_rate(), 0.0)

    def test_fifteen_step_hold_and_completed_episode_denominator(self):
        env = self.make_env(num_envs=2)
        for _ in range(14):
            self.assertEqual(env.step_lift(True, False).tolist(), [0.0, 0.0])
        self.assertEqual(env.pick_counter.tolist(), [14, 0])
        self.assertEqual(env.reset_buf.tolist(), [0, 0])
        self.assertFalse(env.episode_success.any().item())
        self.assertEqual(env.success_counter.tolist(), [0, 0])

        self.assertEqual(env.step_lift(True, False).tolist(), [1.0, 0.0])
        self.assertEqual(env.reset_buf.tolist(), [1, 0])
        self.assertEqual(env.episode_success.tolist(), [True, False])
        self.assertEqual(env.success_counter.tolist(), [0, 0])
        env.reset_ids(0)
        self.assertEqual(env.success_counter.tolist(), [1, 0])
        self.assertEqual(env.episode_counter.tolist(), [1, 0])
        self.assertEqual(env.pick_counter.tolist(), [0, 0])
        self.assertFalse(env.episode_success.any().item())
        self.assertEqual(env.logged_rate(), 1.0)  # Env 1 is still in its first episode.

        env.reset_buf[1] = 1  # Timeout/failure in the other environment.
        env.reset_ids(1)
        self.assertEqual(env.episode_counter.tolist(), [1, 1])
        self.assertEqual(env.logged_rate(), 0.5)
        self.assertEqual(env.success_onetime_counter.tolist(), [0, 0])

    def test_repeated_short_lifts_do_not_count_as_success(self):
        env = self.make_env()
        # Leaving the lift condition can occur without a terminal drop.
        for lifted in [True] * 7 + [False] + [True] * 9 + [False] + [True] * 14:
            self.assertEqual(env.step_lift(lifted).item(), 0.0)
            self.assertEqual(env.reset_buf.item(), 0)
            self.assertFalse(env.episode_success.item())
            self.assertEqual(env.success_counter.item(), 0)
        env.reset_buf[0] = 1
        env.reset_ids(0)
        self.assertEqual(env.episode_counter.item(), 1)
        self.assertEqual(env.success_counter.item(), 0)
        self.assertEqual(env.logged_rate(), 0.0)

    def test_eval_success_and_onetime_settle_only_the_reset_environment(self):
        env = self.make_env(num_envs=2, evaluate=True)
        env.surpass_dist_flag[:] = torch.tensor([True, False])
        self.assertEqual(env.step_lift(True, True).tolist(), [1.0, 1.0])
        self.assertEqual(env.reset_buf.tolist(), [1, 1])
        self.assertEqual(env.episode_success.tolist(), [True, True])
        self.assertEqual(env.episode_onetime_success.tolist(), [True, False])
        self.assertEqual(env.success_counter.tolist(), [0, 0])

        env.reset_ids(0)
        self.assertEqual(env.success_counter.tolist(), [1, 0])
        self.assertEqual(env.success_onetime_counter.tolist(), [1, 0])
        self.assertEqual(env.episode_counter.tolist(), [1, 0])
        self.assertEqual(env.episode_success.tolist(), [False, True])
        env.reset_ids(1)
        self.assertEqual(env.success_counter.tolist(), [1, 1])
        self.assertEqual(env.success_onetime_counter.tolist(), [1, 0])
        self.assertFalse(env.episode_success.any().item())
        self.assertFalse(env.episode_onetime_success.any().item())
        self.assertEqual(env.logged_rate(), 1.0)
        self.assertEqual(env.logged_rate(one_time=True), 0.5)

        env.step_lift(False, False)
        env.reset_buf[0] = 1
        env.reset_ids(0)
        self.assertEqual(env.episode_counter.tolist(), [2, 1])
        self.assertEqual(env.success_counter.tolist(), [1, 1])
        self.assertEqual(env.success_onetime_counter.tolist(), [1, 0])
        self.assertAlmostEqual(env.logged_rate(), 2 / 3)

    def test_configured_hold_controls_logging_without_changing_reward_threshold(self):
        for hold_steps in (3, 20):
            with self.subTest(hold_steps=hold_steps):
                env = self.make_env(hold_steps=hold_steps)
                for step in range(1, hold_steps + 1):
                    reward = env.step_lift(True)
                    # Preserve the existing hardcoded reward threshold of 15.
                    self.assertEqual(reward.item(), float(step >= 15))
                    self.assertEqual(env.reset_buf.item(), int(step >= hold_steps))
                    self.assertEqual(env.episode_success.item(), step >= hold_steps)
                    self.assertEqual(env.success_counter.item(), 0)
                env.reset_ids(0)
                self.assertEqual(env.success_counter.item(), 1)
                self.assertEqual(env.logged_rate(), 1.0)


if __name__ == "__main__":
    unittest.main()
