"""CPU tests of completed-episode statistics without importing Isaac Gym.

Extract the production methods unchanged so their tensor behavior, reset
ordering and returned logging data are exercised with real CPU torch tensors.
Only simulator operations and the parent physics step are stubbed.
"""

import ast
import copy
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Run these tests in the dqwbc environment: {}".format(exc))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.episode_success import EpisodeSuccessWindow


def extract_class(filename, class_name, methods, bases, namespace):
    tree = ast.parse((ROOT / "envs" / filename).read_text())
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = copy.deepcopy(source)
    selected.bases = [ast.Name(id=name, ctx=ast.Load()) for name in bases]
    selected.keywords = []
    selected.decorator_list = []
    selected.body = [node for node in selected.body
                     if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected.body} == set(methods)
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(ROOT / "envs" / filename), "exec"), namespace)
    return namespace[class_name]


def make_environment(object_names=("apple",), num_envs=1, evaluation=False):
    printed = []
    namespace = {
        "torch": torch, "Dict": Dict, "Tuple": Tuple, "Any": Any,
        "tqdm": SimpleNamespace(write=printed.append),
        "gymtorch": SimpleNamespace(unwrap_tensor=lambda value: value),
        "torch_rand_float": lambda low, high, shape, device: torch.full(shape, .5),
    }
    base = extract_class("b1z1_base.py", "B1Z1Base",
                         {"_reset_env_tensors"}, ["object"], namespace)
    base.step = lambda self, actions: self._physics_step(actions)
    extract_class("reward_vec_task.py", "RewardVecTask",
                  {"_reward_pick_up"}, ["object"], namespace)
    task = extract_class("b1z1_pickmulti.py", "B1Z1PickMulti",
                         {"_record_completed_episode_success", "_reset_env_tensors", "step"},
                         ["B1Z1Base", "RewardVecTask"], namespace)
    env = task()
    env.num_envs = num_envs
    env.device = "cpu"
    env.obj_list = list(object_names)
    env.success_window = EpisodeSuccessWindow(object_names)
    env.extras = {}
    env.eval = evaluation
    env.global_step_counter = 1
    env.hold_steps = 3
    for name in ("episode_success", "success_recorded", "lifted_object", "lifted_now"):
        setattr(env, name, torch.zeros(num_envs, dtype=torch.bool))
    env.surpass_dist_flag = torch.ones(num_envs, dtype=torch.bool)
    for name in ("reset_buf", "progress_buf", "pick_counter", "reach_counter",
                 "_terminate_buf", "num_steps_to_change_mode", "increase_to_change_step"):
        setattr(env, name, torch.zeros(num_envs, dtype=torch.long))
    for name in ("curr_dist", "closest_dist", "curr_height", "highest_object"):
        setattr(env, name, torch.zeros(num_envs))
    for name in ("last_actions", "last_low_actions", "clipped_actions", "commands"):
        setattr(env, name, torch.zeros(num_envs, 2))
    env.action_history_buf = torch.zeros(num_envs, 2, 2)
    env.command_history_buf = torch.zeros(num_envs, 2, 2)
    env.rand_cmd_scale = False
    for offset, name in enumerate(("_robot_actor_ids", "_table_actor_ids", "_cube_actor_ids")):
        setattr(env, name, torch.arange(num_envs) * 3 + offset)
    env._dof_state = torch.zeros(num_envs, 2)
    env._root_states = torch.zeros(num_envs * 3, 13)
    env.sim = None
    env.gym = SimpleNamespace(
        set_dof_state_tensor_indexed=lambda *args: None,
        set_actor_root_state_tensor_indexed=lambda *args: None,
        simulate=lambda *args: None,
    )
    env._refresh_sim_tensors = lambda: None
    env.render = lambda: None
    env._reset_objs = lambda env_ids: None
    env.object_pose_predictor = None
    env.pred_success = False
    env.near_goal_stop = False
    env.obj_move_prob = 0
    return env, printed


class EpisodeSuccessWindowTests(unittest.TestCase):
    def test_empty_and_warmup_use_actual_completed_episode_count(self):
        window = EpisodeSuccessWindow(["apple", "pear"])
        metrics = window.metrics()
        self.assertEqual(metrics["SuccessRate / Total"], 0)
        self.assertEqual(metrics["SuccessWindow / Episodes"], 0)
        self.assertEqual(metrics["SuccessRate / pear"], 0)
        for name, success in [("apple", True), ("pear", False), ("apple", False)]:
            window.append(name, success)
        metrics = window.metrics()
        self.assertEqual(metrics["SuccessCount / Total"], 1)
        self.assertEqual(metrics["SuccessWindow / Episodes"], 3)
        self.assertAlmostEqual(metrics["SuccessRate / Total"], 1 / 3)
        self.assertEqual(metrics["SuccessRate / apple"], .5)
        self.assertEqual(metrics["SuccessRate / pear"], 0)

    def test_window_boundary_evicts_old_successes(self):
        window = EpisodeSuccessWindow(["apple"])
        for _ in range(1000):
            window.append("apple", True)
        self.assertEqual(window.metrics()["SuccessRate / Total"], 1)
        self.assertEqual(window.metrics()["SuccessWindow / Episodes"], 1000)
        window.append("apple", False)
        self.assertEqual(window.metrics()["SuccessCount / Total"], 999)
        self.assertEqual(window.metrics()["SuccessRate / Total"], .999)
        for _ in range(999):
            window.append("apple", False)
        self.assertEqual(window.metrics()["SuccessCount / Total"], 0)
        self.assertEqual(window.metrics()["SuccessRate / Total"], 0)
        self.assertEqual(window.metrics()["SuccessWindow / Episodes"], 1000)

    def test_object_denominators_share_the_global_window(self):
        window = EpisodeSuccessWindow(["apple", "pear", "unused"])
        for _ in range(800):
            window.append("apple", True, True)
        for _ in range(400):
            window.append("pear", False, False)
        metrics = window.metrics(include_one_time=True)
        self.assertEqual(metrics["SuccessWindow / apple"], 600)
        self.assertEqual(metrics["SuccessWindow / pear"], 400)
        self.assertEqual(metrics["SuccessWindow / unused"], 0)
        self.assertEqual(metrics["SuccessRate / Total"], .6)
        self.assertEqual(metrics["SuccessRate / apple"], 1)
        self.assertEqual(metrics["SuccessRate / pear"], 0)
        self.assertEqual(metrics["Onetime_SuccessRate / Total"], .6)
        self.assertEqual(metrics["Onetime_SuccessRate / apple"], 1)
        self.assertNotIn("Onetime_SuccessRate / Total", window.metrics())


class CompletedEpisodeIntegrationTests(unittest.TestCase):
    def test_only_finished_noninitial_episodes_are_recorded_once(self):
        env, printed = make_environment(num_envs=4)
        env.reset_buf[:] = torch.tensor([1, 0, 1, 1])
        env.progress_buf[:] = torch.tensor([0, 5, 5, 5])
        env.episode_success[:] = torch.tensor([True, True, True, False])
        env._record_completed_episode_success()
        metrics = env.extras["success_rate"]
        self.assertEqual(metrics["SuccessWindow / Episodes"], 2)
        self.assertEqual(metrics["SuccessRate / Total"], .5)
        self.assertEqual(env.success_recorded.tolist(), [False, False, True, True])
        env._record_completed_episode_success()
        self.assertEqual(len(printed), 1)
        self.assertIs(env.extras["success_rate"], metrics)
        env.reset_buf[1] = 1
        env._record_completed_episode_success()
        self.assertEqual(env.extras["success_rate"]["SuccessWindow / Episodes"], 3)
        self.assertAlmostEqual(env.extras["success_rate"]["SuccessRate / Total"], 2 / 3)

    def test_reset_clears_episode_flags_and_preserves_window_for_next_episode(self):
        env, _ = make_environment(num_envs=2)
        env.progress_buf[:] = 10
        env.reset_buf[:] = 1
        env.episode_success[:] = True
        env._record_completed_episode_success()
        env._reset_env_tensors(torch.tensor([0]))
        self.assertEqual(env.episode_success.tolist(), [False, True])
        self.assertEqual(env.success_recorded.tolist(), [False, True])
        self.assertEqual(env.progress_buf.tolist(), [0, 10])
        self.assertEqual(env.reset_buf.tolist(), [0, 1])
        self.assertEqual(len(env.success_window.episodes), 2)
        env._record_completed_episode_success()
        self.assertEqual(len(env.success_window.episodes), 2)
        env.progress_buf[0] = 7
        env.reset_buf[0] = 1
        env._record_completed_episode_success()
        metrics = env.extras["success_rate"]
        self.assertEqual(metrics["SuccessWindow / Episodes"], 3)
        self.assertEqual(metrics["SuccessCount / Total"], 2)
        self.assertAlmostEqual(metrics["SuccessRate / Total"], 2 / 3)

    def test_single_object_and_reordered_uneven_object_assignments(self):
        for names in [("only_object",), ("pear", "apple", "orange")]:
            with self.subTest(objects=names):
                env, _ = make_environment(names, num_envs=7)
                env.progress_buf[:] = 1
                env.reset_buf[:] = 1
                env.episode_success[:] = torch.tensor([True, False, False, True, True, False, False])
                env._record_completed_episode_success()
                metrics = env.extras["success_rate"]
                self.assertEqual(metrics["SuccessCount / Total"], 3)
                self.assertEqual(metrics["SuccessWindow / Episodes"], 7)
                for index, name in enumerate(names):
                    expected = env.episode_success[index::len(names)].tolist()
                    self.assertEqual(metrics[f"SuccessWindow / {name}"], len(expected))
                    self.assertEqual(metrics[f"SuccessRate / {name}"], sum(expected) / len(expected))

    def test_batch_larger_than_window_keeps_last_1000_in_environment_order(self):
        env, _ = make_environment(("pear", "apple", "orange"), num_envs=1205)
        env.progress_buf[:] = 1
        env.reset_buf[:] = 1
        env.episode_success[:205] = True
        env._record_completed_episode_success()
        metrics = env.extras["success_rate"]
        self.assertEqual(metrics["SuccessWindow / Episodes"], 1000)
        self.assertEqual(metrics["SuccessCount / Total"], 0)
        self.assertEqual(metrics["SuccessRate / Total"], 0)
        self.assertEqual(list(env.success_window.episodes), [
            (env.obj_list[index % 3], False, False) for index in range(205, 1205)
        ])

    def test_console_and_step_infos_publish_the_same_current_window(self):
        env, printed = make_environment(num_envs=2)
        def physics_step(actions):
            env.progress_buf[:] = 1
            env.reset_buf[:] = 1
            env.episode_success[:] = torch.tensor([True, False])
            return {}, torch.zeros(2), env.reset_buf, env.extras
        env._physics_step = physics_step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Existing torch.range call.
            _, _, _, infos = env.step(torch.zeros(2, 9))
        self.assertIs(infos, env.extras)
        self.assertEqual(infos["success_rate"]["SuccessRate / Total"], .5)
        self.assertEqual(infos["success_rate"]["SuccessWindow / Episodes"], 2)
        self.assertEqual(len(printed), 1)
        self.assertIn("last 1000 completed episodes", printed[0])
        console_metrics = ast.literal_eval(printed[0].split(": ", 1)[1])
        self.assertEqual(console_metrics, infos["success_rate"])

    def test_evaluation_one_time_success_is_an_episode_outcome(self):
        env, _ = make_environment(num_envs=3, evaluation=True)
        env.progress_buf[:] = 1
        env.reset_buf[:] = 1
        env.episode_success[:] = torch.tensor([True, True, False])
        env.surpass_dist_flag[:] = torch.tensor([True, False, True])
        env._record_completed_episode_success()
        metrics = env.extras["success_rate"]
        self.assertAlmostEqual(metrics["SuccessRate / Total"], 2 / 3)
        self.assertAlmostEqual(metrics["Onetime_SuccessRate / Total"], 1 / 3)


class PickupSuccessTests(unittest.TestCase):
    def test_lift_dropped_before_hold_threshold_is_a_failed_episode(self):
        env, _ = make_environment()
        env.lifted_object[:] = True
        for _ in range(env.hold_steps - 1):
            env._reward_pick_up()
        self.assertFalse(env.episode_success.item())
        self.assertEqual(env.reset_buf.item(), 0)
        env.lifted_object[:] = False
        env.reset_buf[:] = 1  # A drop terminates the episode.
        env.progress_buf[:] = 3
        env._reward_pick_up()
        env._record_completed_episode_success()
        self.assertEqual(env.pick_counter.item(), 0)
        self.assertEqual(env.extras["success_rate"]["SuccessCount / Total"], 0)

    def test_reaching_hold_threshold_records_one_success(self):
        env, _ = make_environment()
        env.lifted_object[:] = True
        for step in range(env.hold_steps):
            env.progress_buf[:] = step + 1
            env._reward_pick_up()
            env._record_completed_episode_success()
            self.assertEqual(len(env.success_window.episodes), int(step + 1 == env.hold_steps))
        self.assertTrue(env.episode_success.item())
        self.assertEqual(env.reset_buf.item(), 1)
        self.assertEqual(env.extras["success_rate"]["SuccessRate / Total"], 1)
        env._record_completed_episode_success()
        self.assertEqual(len(env.success_window.episodes), 1)

    def test_evaluation_preserves_immediate_lift_success(self):
        env, _ = make_environment(evaluation=True)
        env.lifted_object[:] = True
        env.progress_buf[:] = 1
        reward, _ = env._reward_pick_up()
        env._record_completed_episode_success()
        self.assertEqual(reward.item(), 1)
        self.assertTrue(env.episode_success.item())
        self.assertEqual(env.pick_counter.item(), 0)
        self.assertEqual(env.extras["success_rate"]["SuccessRate / Total"], 1)


if __name__ == "__main__":
    unittest.main()
