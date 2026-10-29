"""Execute the real SGS environment methods with CPU simulator stand-ins.

Only physics submission is stubbed. Reset bookkeeping, cache restoration,
sampler updates, terminal predicates and hold rewards execute repository code.
AST extraction avoids importing Isaac Gym or creating a CUDA context.
"""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

from utils.episode_statistics import RecentEpisodeStatistics
from utils.pick_success import RollingEpisodeSuccess, advance_pick_hold, lifted_grasp_mask


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sgs_env_test_sampler", ROOT / "utils" / "sgs.py")
SGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SGS)


def quat_apply(q, v):
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(q[..., :3], uv, dim=-1))


def quat_rotate_inverse(q, v):
    inverse = q.clone()
    inverse[..., :3].neg_()
    return quat_apply(inverse, v)


def quat_mul(a, b):
    vector = (a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
              + torch.cross(a[..., :3], b[..., :3], dim=-1))
    scalar = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((vector, scalar), -1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy), -1)


def euler_from_quat(q):
    x, y, z, w = q.unbind(-1)
    return (torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
            torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


NAMESPACE = {
    "torch": torch, "F": torch.nn.functional,
    "tqdm": SimpleNamespace(write=print),
    "quat_apply": quat_apply, "quat_rotate_inverse": quat_rotate_inverse,
    "quat_rotate": quat_apply,
    "quat_mul": quat_mul, "quat_from_euler_xyz": quat_from_euler_xyz,
    "euler_from_quat": euler_from_quat,
    "gymtorch": SimpleNamespace(unwrap_tensor=lambda value: value),
    "torch_rand_float": lambda low, high, shape, device: torch.full(
        shape, (low + high) / 2, device=device),
    "np": SimpleNamespace(pi=torch.pi, deg2rad=lambda value: value * torch.pi / 180.),
    "advance_pick_hold": advance_pick_hold, "lifted_grasp_mask": lifted_grasp_mask,
}


def method_node(filename, class_name, method_name):
    module = ast.parse((ROOT / "envs" / filename).read_text())
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    return method


def source_method(filename, class_name, method_name):
    namespace = NAMESPACE.copy()
    module = ast.Module(body=[method_node(filename, class_name, method_name)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "envs" / filename), "exec"), namespace)
    return namespace[method_name]


class SimSubmission:
    def __init__(self):
        self.robot_ids = []
        self.actor_ids = []
        self.root_payloads = []

    def set_dof_state_tensor_indexed(self, sim, state, ids, count):
        assert count == len(ids)
        self.robot_ids.append(ids.clone())

    def set_actor_root_state_tensor_indexed(self, sim, state, ids, count):
        assert count == len(ids)
        self.actor_ids.append(ids.clone())
        self.root_payloads.append(state.clone())

    def simulate(self, sim):
        raise AssertionError("an asynchronous reset must not advance any environment")


class BaseStandIn:
    _reset_env_tensors = source_method("b1z1_base.py", "B1Z1Base", "_reset_env_tensors")
    check_termination = source_method("b1z1_base.py", "B1Z1Base", "check_termination")
    update_roboinfo = source_method("b1z1_base.py", "B1Z1Base", "update_roboinfo")
    floating_table_control = source_method("b1z1_base.py", "B1Z1Base", "floating_table_control")
    compute_table_command_level_02 = source_method(
        "b1z1_base.py", "B1Z1Base", "compute_table_command_level_02")

    def reset(self):
        ids = self.reset_buf.nonzero().flatten()
        self.reset_idx(ids)
        return self.obs.clone()

    def reset_idx(self, ids):
        self._reset_sgs_envs(ids)

    def step(self, actions):
        self.floating_table_control()
        self.reset_buf.copy_(self.next_done.long())
        self.one_epoch_step += 1
        return self.obs, torch.zeros(self.num_envs), self.reset_buf, self.extras


def production_environment_class():
    methods = ("reset", "_reset_sgs_envs", "_record_sgs_metrics", "_print_sgs_success_rates",
               "update_roboinfo", "step", "check_termination", "_yaw_deviation",
               "_reset_pick_state", "_record_episode_outcomes", "_update_curr_dist")
    cls = ast.ClassDef(
        name="B1Z1PickMulti", bases=[ast.Name(id="B1Z1Base", ctx=ast.Load())],
        keywords=[], body=[method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name)
                           for name in methods], decorator_list=[])
    namespace = dict(NAMESPACE, B1Z1Base=BaseStandIn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(ROOT / "envs" / "b1z1_pickmulti.py"), "exec"), namespace)
    return namespace["B1Z1PickMulti"]


ProductionEnvironment = production_environment_class()


def make_bank(num_configs=6, num_families=3):
    n, d, b = num_configs, 19, 5
    roots = torch.zeros(n, 13)
    roots[:, 6] = 1
    robot, obj, table = roots.clone(), roots.clone(), roots.clone()
    robot[:, 0] = torch.arange(n) * 0.1 - 1.0
    robot[:, 2] = 0.6
    obj[:, 0] = robot[:, 0] + 0.7
    obj[:, 2] = 0.695  # elevated but below success threshold
    table[:, 0] = obj[:, 0]
    table[:, 2] = 0.525
    body = robot[:, None].repeat(1, b, 1)
    body[:, 1, :3] = obj[:, :3] + torch.tensor([0.03, 0.0, 0.04])
    arm_base = robot[:, :3] + torch.tensor([0.3, 0.0, 0.09])
    tensors = {
        "family_id": torch.arange(n) % num_families,
        "robot_root": robot, "object_root": obj, "table_root": table,
        "rigid_body_state": body,
        "ee_jacobian": torch.eye(6)[None].repeat(n, 1, 1),
        "dof_pos": torch.arange(n, dtype=torch.float)[:, None].repeat(1, d) * 0.01,
        "dof_vel": torch.zeros(n, d), "ee_goal_cart": body[:, 1, :3] - arm_base,
        "ee_goal_orn_rpy": torch.zeros(n, 3), "gripper_target": torch.zeros(n, 1),
        "table_vy": torch.linspace(-0.3, 0.3, n),
        "rotation_enabled": torch.arange(n) % 2 == 0,
        "table_yaw_rate": torch.linspace(-0.2, 0.2, n),
        "table_height": torch.full((n,), 0.55),
    }
    metadata = {
        "object_name": "sugar_box", "robot_asset": "robot.urdf", "num_dofs": d,
        "num_bodies": b, "hold_steps": 15, "lifted_threshold": 0.2,
        "version": "test", "physical_validation": {"passed": True},
    }
    if num_families == 4:
        metadata["family_names"] = ["full_task", "pregrasp", "at_grasp", "near_goal"]
    return SGS.ConfigurationBank(tensors, metadata)


def make_near_goal_bank():
    from test_sgs import near_goal_bank_fixture
    fixture, metadata = near_goal_bank_fixture()
    bank = make_bank(num_configs=8, num_families=4)
    tensors = bank.tensors
    tensors["family_id"] = fixture["family_id"]
    tensors["object_root"][:, 2] = fixture["object_root"][:, 2]
    tensors["rigid_body_state"][:, [0, 2, 3, 4], 2] = .04
    tensors["rigid_body_state"][:, 1, :3] = tensors["object_root"][:, :3] + torch.tensor([.03, 0., .04])
    tensors["ee_goal_cart"] = (tensors["rigid_body_state"][:, 1, :3]
                               - tensors["robot_root"][:, :3] - torch.tensor([.3, 0., .09]))
    tensors["dof_pos"][6:, -1] = -.7  # contact blocks physical closure to zero
    for name in fixture:
        if name.startswith("near_goal_") or name == "lift_height":
            tensors[name] = fixture[name]
    metadata.update(robot_asset="robot.urdf", table_level="Level02")
    return SGS.ConfigurationBank(tensors, metadata)


def make_active_near_goal_bank():
    bank = make_near_goal_bank()
    retained = torch.tensor([0, 1, 2, 3, 6, 7])
    tensors = {name: value[retained].clone() for name, value in bank.tensors.items()}
    tensors["family_id"][tensors["family_id"] == 3] = 2
    metadata = bank.metadata
    metadata["family_names"] = ["full_task", "pregrasp", "near_goal"]
    metadata["physical_validation"]["validation_profile"] = "geometry_grasp_v3"
    return SGS.ConfigurationBank(tensors, metadata)


class SGSInitializationTests(unittest.TestCase):
    initialize_sgs = staticmethod(source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_init_sgs"))

    def geometry_bank(self):
        from utils.sgs_grounding import APPROX_GROUND_LIMITS, FOOT_RADIUS
        bank = make_bank()
        metadata, tensors = bank.metadata, bank.tensors
        metadata["table_level"] = "Level02"
        metadata["family_names"] = ["full_task", "pregrasp", "at_grasp"]
        feet = [0, 2, 3, 4]
        tensors["rigid_body_state"][:, feet, 2] = FOOT_RADIUS
        metadata["physical_validation"] = {
            "passed": True, "validation_profile": "geometry_v1",
            "feet_near_ground": True, "collision_checked": True,
            "object_attachment": False,
            "grounding": {"foot_body_indices": feet, "foot_radius_m": FOOT_RADIUS,
                          "limits": dict(APPROX_GROUND_LIMITS)},
        }
        return SGS.ConfigurationBank(tensors, metadata)

    def environment(self, path, mode="sgs"):
        return SimpleNamespace(
            cfg={"seed": 43, "env": {"sgs": {
                "mode": mode, "bank_path": str(path), "bank_size": 6},
                "asset": {"assetFileRobot": "robot.urdf"}}},
            obj_list=["sugar_box"], table_level="Level02", enable_camera=False,
            floating_base=False, obj_move_prob=0., success_mode="hold", device="cpu",
            num_dofs=19, num_envs=4, _rigid_body_pos=torch.zeros(4, 5, 3),
            hold_steps=15, lifted_success_threshold=.2, init_height=torch.full((4,), .045))

    def test_supported_profiles_initialize_both_modes_and_preserve_old_bank_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "physical.pt"
            for bank in (self.geometry_bank(), make_near_goal_bank(), make_active_near_goal_bank()):
                bank.save(path)
                for mode in ("sgs", "uniform"):
                    with self.subTest(families=bank.family_names, mode=mode):
                        env = self.environment(path, mode=mode)
                        env.cfg["env"]["sgs"]["bank_size"] = None
                        with redirect_stdout(io.StringIO()):
                            self.initialize_sgs(env)
                        self.assertEqual(env.sgs_family_names, bank.family_names)
                        self.assertEqual(len(env.sgs_family_samples), len(bank.family_names))
                        self.assertEqual(env.recent_episode_stats.family_count.tolist(),
                                         [0] * len(bank.family_names))
                        self.assertEqual(env.sgs_sampler.num_configs, bank.num_configs)
                        self.assertEqual(env.sgs_bank.fingerprint, bank.fingerprint)

    def test_near_goal_metadata_must_match_task_object_support_height(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "near.pt"
            for bank in (make_near_goal_bank(), make_active_near_goal_bank()):
                bank.save(path)
                env = self.environment(path)
                env.cfg["env"]["sgs"]["bank_size"] = None
                env.init_height.fill_(.08)
                with self.subTest(families=bank.family_names), self.assertRaisesRegex(
                        ValueError, "object_init_height_m"):
                    self.initialize_sgs(env)

    def test_geometry_bank_initializes_gpu_compatible_sampler_in_both_training_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "geometry.pt"
            bank = self.geometry_bank()
            bank.save(path)
            for mode in ("sgs", "uniform"):
                with self.subTest(mode=mode):
                    env = self.environment(path, mode=mode)
                    with redirect_stdout(io.StringIO()):
                        self.initialize_sgs(env)
                    self.assertTrue(env.sgs_enabled)
                    self.assertEqual(env.sgs_bank.fingerprint, bank.fingerprint)
                    self.assertEqual(env.active_config_id.tolist(), [-1] * 4)
                    ids = env.sgs_sampler.sample(300)
                    self.assertEqual(ids.unique().tolist(), list(range(6)))
                    self.assertEqual(env.sgs_bank.batch(ids)["family_id"].unique().tolist(),
                                     [0, 1, 2])

    def test_historical_grasped_bank_is_rejected_by_new_training_loader(self):
        bank = make_bank()
        metadata = bank.metadata
        metadata["table_level"] = "Level02"
        metadata["family_names"] = ["full_task", "pregrasp", "grasped"]
        legacy = SGS.ConfigurationBank(bank.tensors, metadata)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            legacy.save(path)
            with self.assertRaisesRegex(ValueError, "family_names"):
                self.initialize_sgs(self.environment(path))

    def test_bank_size_and_nonempty_family_requirements_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "geometry.pt"
            bank = self.geometry_bank()
            bank.save(path)
            env = self.environment(path)
            env.cfg["env"]["sgs"]["bank_size"] = 12
            with self.assertRaisesRegex(ValueError, "bank_size"):
                self.initialize_sgs(env)
            tensors = {key: value.clone() for key, value in bank.tensors.items()}
            tensors["family_id"].zero_()
            SGS.ConfigurationBank(tensors, bank.metadata).save(path)
            with self.assertRaisesRegex(ValueError, "equal nonempty"):
                self.initialize_sgs(self.environment(path))


class TestEnvironment(ProductionEnvironment):
    _reward_pick_up = source_method("reward_vec_task.py", "RewardVecTask", "_reward_pick_up")
    _reward_approaching = source_method("reward_vec_task.py", "RewardVecTask", "_reward_approaching")
    _reward_lifting = source_method("reward_vec_task.py", "RewardVecTask", "_reward_lifting")

    def __init__(self, num_envs=4, bank=None):
        self.num_envs = num_envs
        self.device = "cpu"
        self.sgs_enabled = True
        self.object_pose_predictor = None
        self._sgs_natural_reset = False
        self._sgs_has_pending_reset = False
        self.sgs_bank = make_bank() if bank is None else bank
        self.sgs_family_names = self.sgs_bank.family_names
        self.sgs_sampler = SGS.SGSSampler(self.sgs_bank.num_configs)
        self.active_config_id = torch.full((num_envs,), -1, dtype=torch.long)
        self.global_step_counter = 20
        self.sgs_family_episodes = torch.zeros(len(self.sgs_family_names), dtype=torch.long)
        self.sgs_family_successes = torch.zeros(len(self.sgs_family_names), dtype=torch.long)
        self.sgs_family_samples = torch.zeros(len(self.sgs_family_names), dtype=torch.long)
        self.recent_episode_stats = RecentEpisodeStatistics(
            len(self.sgs_family_names), device=self.device, family_window=100)
        self.obj_list = ["sugar_box"]
        self.sgs_aborted = torch.zeros((), dtype=torch.long)
        self.sgs_log_interval = 24
        self._sgs_last_print_step = None
        self.extras = {}
        self.sim = object()
        self.gym = SimSubmission()
        self.table_dimz = 0.05
        self.table_level = "Level02"
        self.table_z_lower, self.table_z_upper = 0.5, 0.6
        self.rand_cmd_scale = False
        self.success_mode = "hold"
        self.eval = True
        self.hold_steps = 10
        self.max_episode_length = 100
        self.lifted_success_threshold = 0.2
        self.reward_scales = {"limit_yaw_rotation_penalty": 0.4}
        self.initial_yaw_z = torch.tensor(0.)
        self.enable_camera = False
        self.pred_success = False
        self.near_goal_stop = False
        self.obj_move_prob = 0.0
        self.cfg = {"env": {"cameraVisibilityReset": False}}
        self._root_states = torch.zeros(num_envs * 3, 13)
        roots = self._root_states.view(num_envs, 3, 13)
        roots[..., 6] = 1
        self._robot_root_states, self._table_root_states, self._cube_root_states = roots.unbind(1)
        self._dof_state = torch.zeros(num_envs, 19, 2)
        self._dof_pos, self._dof_vel = self._dof_state.unbind(-1)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._rigid_body_state = torch.zeros(num_envs, 5, 13)
        self._rigid_body_state[..., 6] = 1
        self.ee_j_eef = torch.zeros(num_envs, 6, 6)
        self._robot_actor_ids = torch.arange(num_envs) * 3
        self._table_actor_ids = self._robot_actor_ids + 1
        self._cube_actor_ids = self._robot_actor_ids + 2
        self.init_height = torch.full((num_envs,), 0.045)
        for name in (
                "episode_counter", "success_counter", "success_onetime_counter",
                "pick_counter", "best_pick_counter", "reach_counter", "progress_buf", "reset_buf",
                "_terminate_buf", "num_steps_to_change_mode", "increase_to_change_step",
                "rand_change_step"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long))
        self.episode_counter.fill_(-1)
        self.reset_buf.fill_(1)
        for name in (
                "_sgs_pending_done", "_sgs_pending_reset", "episode_success", "episode_onetime_success",
                "surpass_dist_flag", "lifted_object", "lifted_now", "timeout_buf",
                "lin_circle_select", "next_done", "pick_success", "pick_success_recorded",
                "pick_hold_progress", "episode_outcome_recorded"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool))
        self.episode_success = self.pick_success_recorded
        self.success_window = RollingEpisodeSuccess(window_size=1000)
        for name in (
                "episode_step_sum", "one_epoch_step", "curr_dist", "closest_dist",
                "curr_height", "highest_object", "table_heights",
                "vel_line_forward_y_level02", "vel_angular_yaw"):
            setattr(self, name, torch.zeros(num_envs))
        for name in (
                "last_actions", "last_low_actions", "clipped_actions", "commands",
                "curr_ee_goal_cart", "curr_ee_goal_orn_rpy", "last_ee_pos",
                "ee_goal_global_cart", "arm_base"):
            setattr(self, name, torch.zeros(num_envs, 3))
        self.ee_goal_global_orn_quat = torch.zeros(num_envs, 4)
        self.gripper_dof_pos = torch.zeros(num_envs, 1)
        self.gripper_dof_vel = torch.zeros(num_envs, 1)
        self.table_commands = torch.zeros(num_envs, 6)
        self.action_history_buf = torch.zeros(num_envs, 3, 9)
        self.command_history_buf = torch.zeros(num_envs, 3, 9)
        self.actions = torch.zeros(num_envs, 9)
        self.low_obs_history_buf = torch.zeros(num_envs, 10, 4)
        self.foot_contacts_from_sensor = torch.ones(num_envs, 4, dtype=torch.bool)
        self.force_sensor_tensor = torch.ones(num_envs, 4, 6)
        self._contact_forces = torch.ones(num_envs, 5, 3)
        self.obs = torch.zeros(num_envs, 13)
        self._update_base_yaw_quat()

    @property
    def ee_pos(self):
        return self._rigid_body_state[:, 1, :3]

    def _update_base_yaw_quat(self):
        self.base_yaw_quat = self._robot_root_states[:, 3:7].clone()

    def _reset_objs(self, ids):
        if len(ids):
            raise AssertionError("SGS must not randomly replace the active object")

    def _compute_observations(self, ids):
        self.obs[ids] = torch.cat(
            (self._robot_root_states[ids, :7], self.ee_pos[ids],
             self.ee_goal_global_cart[ids]), dim=1)

    def initialize(self, ids=(0, 1, 2, 3)):
        with patch.object(self.sgs_sampler, "sample", return_value=torch.tensor(ids)):
            self.reset()

    def grasp_step(self, lifted=True):
        """Run real terminal predicates on a closed grasp with tracked IK."""
        lifted = torch.as_tensor(lifted, dtype=torch.bool).expand(self.num_envs)
        self.actions[:, 6] = -1
        self._cube_root_states[:, 2] = (
            self.table_heights + self.init_height + torch.where(lifted, .21, .05))
        self._rigid_body_state[:, 1, :3] = self._cube_root_states[:, :3]
        self.update_roboinfo()
        self.curr_ee_goal_cart[:] = quat_rotate_inverse(
            self._robot_root_states[:, 3:7], self.ee_pos - self.arm_base)
        self.progress_buf += 1
        self.check_termination()
        reward, _ = self._reward_pick_up()
        self._record_episode_outcomes()
        return reward

    def finish(self, env_ids, successes, next_ids):
        self.next_done.zero_()
        self.next_done[env_ids] = True
        self.episode_success[env_ids] = torch.tensor(successes)
        self.episode_onetime_success[env_ids] = torch.tensor(successes)
        # Terminal predicates own lifetime counters; SGS reset must only update
        # configuration statistics, without counting these outcomes again.
        self.success_counter[env_ids] += torch.tensor(successes, dtype=torch.long)
        self.success_onetime_counter[env_ids] += torch.tensor(successes, dtype=torch.long)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.step(torch.zeros(self.num_envs, 9))
        with patch.object(self.sgs_sampler, "sample", return_value=torch.tensor(next_ids)):
            self.reset()


class SGSEnvironmentTests(unittest.TestCase):
    def test_sgs_reset_clears_pose_history_before_observations_only_for_reset_envs(self):
        spec = importlib.util.spec_from_file_location(
            "sgs_pose_prediction_test", ROOT / "modules" / "pose_prediction.py")
        prediction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prediction)
        env = TestEnvironment()
        predictor = prediction.RollingSE3Predictor(env.num_envs, "cpu")
        env.object_pose_predictor = predictor
        env.gym.get_sim_time = lambda sim: 1.0
        env.initialize()
        self.assertEqual(predictor.count.tolist(), [1] * 4)
        for step in range(1, 5):
            poses = env._cube_root_states[:, :7].clone()
            poses[:, 0] += 100.0 + step * .03
            predictor.push(poses, 1.0 + .15 * step)
        predictor.predict()
        self.assertTrue(predictor.valid.all().item())
        untouched = {name: getattr(predictor, name)[[1, 3]].clone()
                     for name in ("poses", "times", "count", "future_poses", "valid")}
        env.gym.get_sim_time = lambda sim: 1.75
        compute_observations = env._compute_observations

        def check_first_observations(ids):
            self.assertEqual(ids.tolist(), [0, 2])
            self.assertEqual(predictor.count[ids].tolist(), [1, 1])
            self.assertFalse(predictor.valid[ids].any().item())
            torch.testing.assert_close(predictor.poses[ids, -1], env._cube_root_states[ids, :7])
            torch.testing.assert_close(
                predictor.future_poses[ids], env._cube_root_states[ids, None, :7].expand(-1, 2, -1))
            return compute_observations(ids)

        with patch.object(env, "_compute_observations", side_effect=check_first_observations) as observe:
            env.finish([0, 2], [True, False], [4, 5])
            observe.assert_called_once()
        for name, expected in untouched.items():
            torch.testing.assert_close(getattr(predictor, name)[[1, 3]], expected)
        self.assertEqual(env.sgs_sampler.visits_total.tolist(), [1, 0, 1, 0, 0, 0])
        # A stationary new episode must not inherit the old episode's motion.
        ids = torch.tensor([0, 2])
        poses = env._cube_root_states[ids, :7].clone()
        for step in range(1, 5):
            predictor.push(poses, 1.75 + .15 * step, ids)
        predictor.predict()
        self.assertTrue(predictor.valid[ids].all().item())
        torch.testing.assert_close(predictor.future_poses[ids], poses[:, None].expand(-1, 2, -1))

    def test_near_goal_reset_restores_actual_aperture_closed_target_and_clean_success(self):
        bank = make_near_goal_bank()
        env = TestEnvironment(bank=bank)
        env.pick_counter.fill_(14)
        env.episode_success.fill_(True)
        env.initialize(ids=(6, 7, 6, 0))
        torch.testing.assert_close(env.curr_height[:3], torch.tensor([.06, .09, .06]))
        torch.testing.assert_close(env.highest_object, env.curr_height)
        torch.testing.assert_close(env._dof_pos[:3, -1], torch.full((3,), -.7))
        torch.testing.assert_close(env.gripper_dof_pos[:3], torch.zeros(3, 1))
        self.assertFalse(bool(env.episode_success.any()))
        self.assertEqual(env.pick_counter.tolist(), [0] * 4)
        self.assertEqual(env.sgs_family_samples.tolist(), [1, 0, 0, 3])
        self.assertEqual(env.sgs_sampler.visits_total.sum().item(), 0)
        self.assertIn("success_near_goal", env.extras["SGS"])
        self.assertAlmostEqual(env.extras["SGS"]["sample_fraction_near_goal"].item(), .75)
        env.finish([0], [True], [1])
        self.assertEqual(env.sgs_family_episodes.tolist(), [0, 0, 0, 1])
        self.assertEqual(env.sgs_family_successes.tolist(), [0, 0, 0, 1])
        self.assertEqual(env.sgs_sampler.visits_total[6].item(), 1)

    def test_active_near_goal_id_two_restores_and_completes_without_at_grasp(self):
        bank = make_active_near_goal_bank()
        env = TestEnvironment(bank=bank)
        with redirect_stdout(io.StringIO()):
            env.initialize(ids=(4, 5, 4, 0))
            torch.testing.assert_close(env.curr_height[:3], torch.tensor([.06, .09, .06]))
            torch.testing.assert_close(env._dof_pos[:3, -1], torch.full((3,), -.7))
            torch.testing.assert_close(env.gripper_dof_pos[:3], torch.zeros(3, 1))
            self.assertEqual(env.sgs_family_samples.tolist(), [1, 0, 3])
            self.assertEqual(env.pick_counter.tolist(), [0] * 4)
            self.assertFalse(bool(env.episode_success.any()))
            env.finish([0, 1], [True, False], [0, 2])
        self.assertEqual(env.sgs_family_episodes.tolist(), [0, 0, 2])
        self.assertEqual(env.sgs_family_successes.tolist(), [0, 0, 1])
        self.assertEqual(env.recent_episode_stats.family_count.tolist(), [0, 0, 2])
        self.assertEqual(env.sgs_sampler.visits_total.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(env.extras["scalar_snapshots"]["SuccessRate / near_goal"], .5)
        self.assertFalse(any("at_grasp" in key for key in env.extras["scalar_snapshots"]))

    def test_initial_reset_never_records_stale_success_or_failure(self):
        env = TestEnvironment()
        env.episode_success.fill_(True)
        env.initialize()
        self.assertEqual(env.sgs_sampler.visits_total.sum().item(), 0)
        self.assertEqual(env.episode_counter.tolist(), [0] * 4)
        self.assertEqual(env.success_counter.tolist(), [0] * 4)
        self.assertFalse(env.episode_success.any().item())
        self.assertEqual(env.sgs_aborted.item(), 0)
        self.assertFalse(env._sgs_natural_reset)
        self.assertTrue(env._sgs_has_pending_reset)
        self.assertTrue(env._sgs_pending_reset.all().item())
        self.assertEqual(env.gym.actor_ids, [])
        self.assertEqual(env.gym.robot_ids, [])

    def test_async_completion_attributes_old_config_and_preserves_other_envs(self):
        env = TestEnvironment()
        env.initialize()
        untouched = {
            "root": env._root_states.view(4, 3, 13)[[1, 3]].clone(),
            "body": env._rigid_body_state[[1, 3]].clone(),
            "obs": env.obs[[1, 3]].clone(),
        }
        env.actions[[1, 3]] = 9
        env.low_obs_history_buf[[1, 3]] = 8
        env.low_obs_history_buf[[0, 2]] = 7
        env.foot_contacts_from_sensor[[1, 3]] = True
        env.finish([0, 2], [True, False], [4, 5])
        self.assertEqual(env.active_config_id.tolist(), [4, 1, 5, 3])
        self.assertEqual(env.sgs_sampler.visits_total.tolist(), [1, 0, 1, 0, 0, 0])
        self.assertEqual(env.sgs_sampler.success_sum.tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(env.episode_counter.tolist(), [1, 0, 1, 0])
        self.assertEqual(env.success_counter.tolist(), [1, 0, 0, 0])
        self.assertEqual(env.sgs_family_episodes.tolist(), [1, 0, 1])
        self.assertIn("success_at_grasp", env.extras["SGS"])
        self.assertIn("success_full_task", env.extras["SGS"])
        self.assertNotIn("success_grasped", env.extras["SGS"])
        torch.testing.assert_close(env._root_states.view(4, 3, 13)[[1, 3]], untouched["root"])
        torch.testing.assert_close(env._rigid_body_state[[1, 3]], untouched["body"])
        torch.testing.assert_close(env.obs[[1, 3]], untouched["obs"])
        self.assertTrue(bool((env.actions[[1, 3]] == 9).all()))
        self.assertTrue(bool((env.low_obs_history_buf[[1, 3]] == 8).all()))
        # Bank restoration isolates controller caches for selected trials.
        self.assertTrue(bool((env.low_obs_history_buf[[0, 2]] == 0).all()))
        self.assertTrue(env.foot_contacts_from_sensor[[1, 3]].all().item())
        self.assertFalse(env.foot_contacts_from_sensor[[0, 2]].any().item())
        self.assertEqual(env._sgs_pending_reset.tolist(), [True, False, True, False])
        env.floating_table_control()
        self.assertEqual(env.gym.robot_ids[-1].tolist(), [0, 6])
        self.assertEqual(env.gym.actor_ids[-1].tolist(), [1, 4, 7, 10, 0, 6, 2, 8])

    def test_pending_resets_accumulate_and_commit_once_with_table_upload(self):
        env = TestEnvironment()
        env.initialize()
        env.floating_table_control()
        self.assertFalse(env._sgs_has_pending_reset)
        self.assertFalse(env._sgs_pending_reset.any().item())
        self.assertEqual(len(env.gym.actor_ids), 1)
        self.assertEqual(len(env.gym.robot_ids), 1)
        for env_id, config_id in ((0, 4), (2, 5)):
            with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([config_id])):
                env.reset_idx(torch.tensor([env_id]))
        self.assertEqual(env._sgs_pending_reset.tolist(), [True, False, True, False])
        self.assertEqual(len(env.gym.actor_ids), 1)
        self.assertEqual(len(env.gym.robot_ids), 1)
        env.floating_table_control()
        self.assertEqual(env.gym.robot_ids[-1].tolist(), [0, 6])
        self.assertEqual(env.gym.actor_ids[-1].tolist(), [1, 4, 7, 10, 0, 6, 2, 8])
        uploaded = env.gym.root_payloads[-1]
        for env_id, config_id in ((0, 4), (2, 5)):
            torch.testing.assert_close(
                uploaded[env._robot_actor_ids[env_id]], env.sgs_bank.tensors["robot_root"][config_id])
            torch.testing.assert_close(
                uploaded[env._cube_actor_ids[env_id]], env.sgs_bank.tensors["object_root"][config_id])
        self.assertFalse(env._sgs_has_pending_reset)
        self.assertFalse(env._sgs_pending_reset.any().item())
        env.floating_table_control()
        self.assertEqual(len(env.gym.robot_ids), 2)
        self.assertEqual(len(env.gym.actor_ids), 3)
        self.assertEqual(env.gym.actor_ids[-1].tolist(), [1, 4, 7, 10])

    def test_multiple_completed_episodes_of_same_config_are_all_recorded(self):
        env = TestEnvironment()
        env.initialize(ids=(0, 0, 0, 0))
        env.finish([0, 1, 2], [True, False, True], [1, 2, 3])
        self.assertEqual(env.sgs_sampler.count[0].item(), 3)
        self.assertEqual(env.sgs_sampler.history[0, :3].tolist(), [True, False, True])
        self.assertAlmostEqual(env.sgs_sampler.success_rates()[0].item(), 2 / 3, places=6)

    def test_real_terminal_outcomes_reach_sampler_once_before_resampling(self):
        env = TestEnvironment()
        env.initialize()
        for _ in range(9):
            env.grasp_step()
        env.progress_buf[[1, 3]] = env.max_episode_length - 1
        self.assertEqual(env.grasp_step([True, False, True, False]).tolist(), [1., 0., 1., 0.])
        self.assertTrue(env.reset_buf.bool().all().item())
        self.assertEqual(env.success_counter.tolist(), [1, 0, 1, 0])
        self.assertEqual(env.success_window.metrics(env.obj_list)["SuccessWindow / completed"], 4)
        # The real task step copies terminal flags before trainer.reset().
        env._sgs_pending_done.copy_(env.reset_buf.bool())
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([4, 5, 4, 5])):
            env.reset()
        self.assertEqual(env.active_config_id.tolist(), [4, 5, 4, 5])
        self.assertEqual(env.sgs_sampler.visits_total.tolist(), [1, 1, 1, 1, 0, 0])
        self.assertEqual(env.sgs_sampler.success_sum.tolist(), [1, 0, 1, 0, 0, 0])
        self.assertEqual(env.success_counter.tolist(), [1, 0, 1, 0])
        self.assertEqual(env.pick_counter.tolist(), [0] * 4)
        self.assertEqual(env.best_pick_counter.tolist(), [0] * 4)
        self.assertFalse(env.pick_success_recorded.any().item())
        self.assertFalse(env.episode_outcome_recorded.any().item())
        self.assertEqual(env.episode_success.data_ptr(), env.pick_success_recorded.data_ptr())
        self.assertEqual(env.success_window.metrics(env.obj_list)["SuccessWindow / completed"], 4)
        self.assertTrue(env._sgs_pending_reset.all().item())
        self.assertEqual(env.gym.robot_ids, [])  # Deferred upload; no reset-time physics.
        env.reset()
        self.assertEqual(env.sgs_sampler.visits_total.sum().item(), 4)
        self.assertEqual(env.success_counter.tolist(), [1, 0, 1, 0])

    def test_cancelled_or_repeated_reset_is_not_a_failed_episode(self):
        env = TestEnvironment()
        env.initialize()
        env.episode_success[0] = True
        env._sgs_pending_done[0] = True
        # An explicit reset_idx cancels even when a stale done flag is present.
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([5])):
            env.reset_idx(torch.tensor([0]))
        self.assertEqual(env.sgs_aborted.item(), 1)
        self.assertEqual(env.sgs_sampler.visits_total.sum().item(), 0)
        self.assertEqual(env.episode_counter.sum().item(), 0)
        self.assertFalse(env._sgs_pending_done[0])
        env.reset()  # No finished IDs: no new event and no additional sampling.
        self.assertEqual(env.sgs_aborted.item(), 1)
        self.assertEqual(env.sgs_family_samples.sum().item(), 5)
        env.finish([1], [False], [4])
        before = env.sgs_sampler.state_dict()
        env.reset()
        for name, values in before["tensors"].items():
            torch.testing.assert_close(getattr(env.sgs_sampler, name), values)

    def test_reset_caches_do_not_create_fake_progress_reward(self):
        env = TestEnvironment()
        env.initialize()
        approaching, _ = env._reward_approaching()
        lifting, _ = env._reward_lifting()
        torch.testing.assert_close(approaching, torch.zeros(4))
        torch.testing.assert_close(lifting, torch.zeros(4))
        torch.testing.assert_close(env.ee_goal_global_cart, env.ee_pos)
        torch.testing.assert_close(env.ee_j_eef, torch.eye(6)[None].repeat(4, 1, 1))
        env.check_termination()
        self.assertFalse(env.reset_buf.any().item())
        self.assertFalse(env._terminate_buf.any().item())
        self.assertFalse(env.lifted_object.any().item())
        self.assertFalse(env.foot_contacts_from_sensor.any().item())
        self.assertEqual(env.force_sensor_tensor.abs().sum().item(), 0)
        self.assertEqual(env._contact_forces.abs().sum().item(), 0)
        self.assertTrue(all(isinstance(value, float)
                            for value in env.extras["success_rate"].values()))

    def test_hold_eval_and_initial_training_step_require_full_hold(self):
        for evaluate, global_step in ((True, 20), (False, 0)):
            env = TestEnvironment()
            env.initialize()
            env.eval = evaluate
            env.global_step_counter = global_step
            for _ in range(9):
                reward = env.grasp_step()
                self.assertFalse(env.reset_buf.any().item())
                self.assertFalse(env.episode_success.any().item())
                self.assertEqual(reward.sum().item(), 0)
            reward = env.grasp_step()
            self.assertTrue(env.reset_buf.bool().all().item())
            self.assertTrue(env.episode_success.all().item())
            self.assertEqual(env.success_onetime_counter.tolist(), [1] * 4)
            self.assertEqual(env.success_counter.tolist(), [1] * 4)
            self.assertEqual(reward.tolist(), [1.] * 4)
            self.assertEqual(env.grasp_step().tolist(), [0.] * 4)
            self.assertEqual(env.success_counter.tolist(), [1] * 4)

    def test_interrupted_hold_restarts_counter(self):
        env = TestEnvironment()
        env.initialize()
        for lifted in [True] * 9 + [False] + [True] * 9:
            env.grasp_step(lifted)
        self.assertEqual(env.pick_counter.tolist(), [9] * 4)
        self.assertFalse(env.pick_hold_progress.any().item())
        self.assertFalse(env.episode_success.any().item())
        env.grasp_step()
        self.assertTrue(env.episode_success.all().item())

    def test_timeout_does_not_hide_real_failure_on_last_hold_step(self):
        env = TestEnvironment()
        env.initialize()
        # Both objects are high enough and close to the gripper. Env 0 has an
        # IK failure coincident with timeout; Env 1 has only a timeout.
        env._cube_root_states[:, 2] = 0.9
        env.actions[:, 6] = -1
        env._rigid_body_state[:, 1, :3] = env._cube_root_states[:, :3]
        env.update_roboinfo()
        local = quat_rotate_inverse(
            env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
        env.curr_ee_goal_cart[:] = local
        env.curr_ee_goal_cart[0, 2] += 0.4
        env.progress_buf.fill_(env.max_episode_length)
        env.pick_counter.fill_(9)
        env.best_pick_counter.fill_(9)
        env.check_termination()
        self.assertFalse(env.timeout_buf.any().item())
        self.assertEqual(env._terminate_buf.tolist(), [1, 0, 0, 0])
        env._reward_pick_up()
        self.assertEqual(env.episode_success.tolist(), [False, True, True, True])


if __name__ == "__main__":
    unittest.main()
