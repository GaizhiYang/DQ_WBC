"""Exercise the real environment methods without importing Isaac Gym.

The extracted methods run against CPU tensors and a small camera API double.
This checks reset/observation contracts; it does not validate GPU rendering.
"""

import ast
from pathlib import Path
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / "DQ_high-level"


def load_methods(filename, class_name, names, base=object, extra_globals=None):
    path = ROOT / "envs" / filename
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in original.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(methods) != len(names):
        raise AssertionError("A requested environment method was not found")
    extracted = ast.ClassDef(name=class_name, bases=[ast.Name(id="Base", ctx=ast.Load())],
                             keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    namespace = {"Base": base, "torch": torch, "np": np}
    namespace.update(extra_globals or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


BufferReturnBase = load_methods("vec_task.py", "VecTask", {"reset", "get_state"})
EnvironmentMethods = load_methods(
    "b1z1_base.py", "B1Z1Base",
    {"reset", "_reset_envs", "_reset_env_tensors", "_compute_states_buf",
     "obtain_imgs", "_camera_frame_observation",
     "_refresh_camera_observations_after_reset", "make_img_obs"},
    base=BufferReturnBase)


class CameraDouble:
    def __init__(self):
        self.events = []
        self.accessing = False

    def fetch_results(self, sim, wait):
        self.events.append("fetch")

    def step_graphics(self, sim):
        self.events.append("graphics")

    def render_all_camera_sensors(self, sim):
        if self.accessing:
            raise AssertionError("Rendering during image tensor access")
        self.events.append("render")

    def start_access_image_tensors(self, sim):
        self.accessing = True
        self.events.append("access")

    def end_access_image_tensors(self, sim):
        self.accessing = False
        self.events.append("release")


def make_environment(refresh=True):
    env = EnvironmentMethods()
    env.device = env.rl_device = "cpu"
    env.num_envs, env.num_features, env.num_actions = 3, 1024, 9
    env.num_states = 3 * 4 * 2 + 61
    env.clip_obs = float("inf")
    env.camera_mode, env.camera_history_len = "full", 3
    env.enable_camera, env.camera_test = True, False
    env.refresh_camera_on_reset = refresh
    env.randomize = env.rand_cmd_scale = False
    env.local_step_counter = 10
    env.sim = object()
    env.gym = CameraDouble()
    env.events = []
    env.episode_counter = torch.zeros(3, dtype=torch.long)
    env.progress_buf = torch.tensor([8, 8, 8])
    env.reset_buf = torch.tensor([0, 1, 0])
    env._terminate_buf = torch.ones(3, dtype=torch.long)
    env.num_steps_to_change_mode = torch.ones(3, dtype=torch.long)
    env.increase_to_change_step = torch.ones(3, dtype=torch.long)
    env.last_actions = torch.ones(3, 9)
    env.last_low_actions = torch.ones(3, 18)
    env.clipped_actions = torch.ones(3, 9)
    env.commands = torch.ones(3, 3)
    env.action_history_buf = torch.ones(3, 3, 9)
    env.command_history_buf = torch.ones(3, 3, 9)
    env.curr_dist = torch.ones(3)
    env.closest_dist = torch.ones(3)
    env.reach_counter = torch.ones(3, dtype=torch.long)
    env.pick_counter = torch.ones(3, dtype=torch.long)
    env.curr_ee_goal_cart = torch.full((3, 3), 9.)
    env.curr_ee_goal_orn_rpy = torch.full((3, 3), 8.)
    env.init_ee_goal_cart = torch.tensor([[.46, 0., .66]]).repeat(3, 1)
    env.obs_buf = torch.arange(3 * 1276, dtype=torch.float).reshape(3, 1276)
    env.states_buf = torch.full((3, env.num_states), -5.)
    env.camera_history_buf = torch.arange(3 * 3 * 8, dtype=torch.float).reshape(3, 3, 8)
    env._camera_history_initialized = torch.ones(3, dtype=torch.bool)
    env.camera_sensor_dict = {name: [object()] * 3 for name in (
        "forward_depth", "forward_seg", "forward_color",
        "wrist_depth", "wrist_seg", "wirst_color")}
    env.obs_dict = {"obs": torch.full_like(env.obs_buf, -99.),
                    "states": torch.full_like(env.states_buf, -99.)}
    env.frame_number = 1
    env.cfg = {"sensor": {"recordCameraGeometry": False}}

    def reset_actors(self, env_ids):
        self.events.append("reset_actors")

    def refresh_tensors(self):
        self.events.append("refresh_robot")

    def update_roboinfo(self):
        self.events.append("update_roboinfo")

    def compute_observations(self, env_ids):
        self.events.append("teacher_obs")
        self.observed_goal = self.curr_ee_goal_cart[env_ids].clone()
        self.obs_buf[env_ids, 1076:1079] = self.curr_ee_goal_cart[env_ids]
        self.obs_buf[env_ids, 1079:1082] = self.curr_ee_goal_orn_rpy[env_ids]
        self.obs_buf[env_ids, -9:] = self.action_history_buf[env_ids, -1]

    def get_camera_obs(self):
        if not self.gym.accessing:
            raise AssertionError("Reading camera tensors without access")
        self.events.append("camera_read")
        return tuple(torch.full((3, 2), self.frame_number * 10. + channel)
                     for channel in range(6))

    for name, function in {
        "reset_idx": lambda self, ids: self._reset_envs(ids),
        "_reset_actors": reset_actors,
        "_refresh_sim_tensors": refresh_tensors,
        "update_roboinfo": update_roboinfo,
        "_compute_observations": compute_observations,
        "_get_camera_obs": get_camera_obs,
    }.items():
        setattr(env, name, types.MethodType(function, env))
    return env


class TeacherVisionResetTests(unittest.TestCase):
    def test_partial_reset_returns_fresh_goals_images_and_states(self):
        env = make_environment()
        old_history, old_states, old_obs = (
            env.camera_history_buf.clone(), env.states_buf.clone(), env.obs_buf.clone())
        previous_return = env.obs_dict["states"]
        result = env.reset()
        self.assertTrue(torch.equal(env.observed_goal, env.init_ee_goal_cart[1:2]))
        self.assertLess(env.events.index("update_roboinfo"), env.events.index("teacher_obs"))
        self.assertLess(env.events.index("teacher_obs"), env.events.index("camera_read"))
        self.assertEqual(env.gym.events.count("render"), 1)
        self.assertFalse(env.gym.accessing)
        frame = env._camera_frame_observation()[1]
        self.assertTrue(torch.equal(env.camera_history_buf[1], frame.repeat(3, 1)))
        self.assertTrue(torch.equal(result["states"][1, :24], frame.repeat(3)))
        self.assertTrue(torch.equal(result["states"][1, 24:76], env.obs_buf[1, 1030:1082]))
        self.assertTrue(torch.equal(result["states"][1, -9:], torch.zeros(9)))
        self.assertIsNot(result["states"], previous_return)
        for index in (0, 2):
            self.assertTrue(torch.equal(env.camera_history_buf[index], old_history[index]))
            self.assertTrue(torch.equal(env.states_buf[index], old_states[index]))
            self.assertTrue(torch.equal(env.obs_buf[index], old_obs[index]))

    def test_first_step_keeps_the_reset_frame_in_history(self):
        env = make_environment()
        env.reset()
        reset_frame = env.camera_history_buf[1, 0].clone()
        env.progress_buf += 1
        env.frame_number = 2
        env.obtain_imgs()
        env.make_img_obs()
        self.assertTrue(torch.equal(env.camera_history_buf[1, :2], reset_frame.repeat(2, 1)))
        self.assertTrue(torch.equal(env.camera_history_buf[1, 2], env._camera_frame_observation()[1]))

    def test_initial_reset_initializes_goals_and_all_camera_histories(self):
        env = make_environment()
        env.local_step_counter = 0
        env.reset_buf[:] = 1
        env._camera_history_initialized[:] = False
        del env.init_ee_goal_cart
        result = env.reset()
        self.assertTrue(env._camera_history_initialized.all())
        self.assertTrue(torch.equal(env.observed_goal, env.init_ee_goal_cart))
        self.assertTrue(torch.equal(result["states"][:, :24], env.camera_history_buf.flatten(1)))
        self.assertEqual(env.gym.events.count("render"), 1)

    def test_legacy_reset_does_not_refresh_or_advance_camera_history(self):
        env = make_environment(refresh=False)
        before = env.camera_history_buf.clone()
        env.reset()
        self.assertTrue(torch.equal(env.observed_goal, torch.full((1, 3), 9.)))
        self.assertEqual(env.gym.events.count("render"), 0)
        self.assertTrue(torch.equal(env.camera_history_buf, before))

    def test_no_done_reset_has_no_camera_or_history_effect(self):
        env = make_environment()
        env.reset_buf[:] = 0
        before = env.camera_history_buf.clone()
        env.reset()
        self.assertEqual(env.gym.events, [])
        self.assertTrue(torch.equal(env.camera_history_buf, before))

    def test_unready_camera_buffers_are_not_accessed(self):
        env = make_environment()
        del env.camera_history_buf
        env._refresh_camera_observations_after_reset(torch.tensor([1]))
        self.assertEqual(env.gym.events, [])
        env = make_environment()
        env.camera_sensor_dict["wrist_depth"] = []
        env._refresh_camera_observations_after_reset(torch.tensor([1]))
        self.assertEqual(env.gym.events, [])

    def test_camera_access_is_released_when_image_processing_raises(self):
        env = make_environment()

        def failing_read(self):
            raise RuntimeError("mock image processing failure")

        env._get_camera_obs = types.MethodType(failing_read, env)
        with self.assertRaisesRegex(RuntimeError, "mock image processing failure"):
            env.obtain_imgs()
        self.assertFalse(env.gym.accessing)
        self.assertEqual(env.gym.events[-1], "release")


class TerminationBase:
    def check_termination(self):
        self.reset_buf[:] = 0


TerminationMethods = load_methods(
    "b1z1_pickmulti.py", "B1Z1PickMulti", {"check_termination"},
    base=TerminationBase, extra_globals={"quat_apply": lambda quaternion, vector: vector})


class CameraConstraintTests(unittest.TestCase):
    def test_camera_constraints_can_be_disabled_without_disabling_camera(self):
        env = TerminationMethods()
        env.num_envs, env.device = 3, "cpu"
        env.enable_camera = True
        env.reset_buf = torch.zeros(3, dtype=torch.long)
        env._robot_root_states = torch.zeros(3, 13)
        env._robot_root_states[:, 0] = torch.tensor([-1., -1., 1.])
        env._cube_root_states = torch.zeros(3, 13)
        env._cube_root_states[:, 0] = torch.tensor([0., -2., 2.])
        env._cube_root_states[:, 2] = .015
        env.ee_pos = env._cube_root_states[:, :3].clone()
        env.base_yaw_quat = torch.zeros(3, 4)
        env.table_heights = env.init_height = torch.zeros(3)
        env.lifted_object = torch.zeros(3, dtype=torch.bool)
        env.lifted_success_threshold = .35
        env.cfg = {"sensor": {"onboard_camera": {"position": [0., 0., 0.]}}}
        env.terminate_on_camera_constraint = True
        env.check_termination()
        self.assertTrue(torch.equal(env.reset_buf, torch.tensor([0, 1, 1])))
        env.terminate_on_camera_constraint = False
        env.check_termination()
        self.assertTrue(torch.equal(env.reset_buf, torch.zeros(3, dtype=torch.long)))

    def test_default_camera_flags_preserve_legacy_configuration(self):
        tree = ast.parse((ROOT / "envs" / "b1z1_base.py").read_text())
        original = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "B1Z1Base")
        init = next(node for node in original.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        settings = [node for node in init.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr in {
                        "terminate_on_camera_constraint", "refresh_camera_on_reset"}
                            for target in node.targets)]
        code = compile(ast.fix_missing_locations(ast.Module(body=settings, type_ignores=[])),
                       "camera_settings", "exec")
        for enabled in (False, True):
            env = types.SimpleNamespace(cfg={"env": {}}, enable_camera=enabled)
            exec(code, {"self": env})
            self.assertEqual(env.terminate_on_camera_constraint, enabled)
            self.assertFalse(env.refresh_camera_on_reset)
        env = types.SimpleNamespace(cfg={"env": {
            "terminateOnCameraConstraint": False, "refreshCameraOnReset": True}},
            enable_camera=True)
        exec(code, {"self": env})
        self.assertFalse(env.terminate_on_camera_constraint)
        self.assertTrue(env.refresh_camera_on_reset)


if __name__ == "__main__":
    unittest.main()
