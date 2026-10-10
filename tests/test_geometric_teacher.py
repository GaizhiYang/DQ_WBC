"""CPU regression tests for physical grasp scoring and experiment isolation."""
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.geometric_grasp_selector import (
    GEOMETRIC_DEFAULTS, GeometricGraspSelector, geometric_grasp_costs,
    validate_geometric_settings,
)
from utils.teacher_grasp_training import (
    ENVIRONMENT_OPTIONS, TeacherGraspTrainingState, resolve_selection,
)
from modules.karl_teacher import KarlTeacherPolicy
from utils.geometric_teacher_wrapper import GeometricTeacherWrapper
from utils.grasp_geometry import (
    TeacherGraspGeometry, load_gripper_envelope_corners, load_object_collision_bounds,
)
from skrl.resources.preprocessors.torch import RunningStandardScaler


def options(**overrides):
    values = {key: False for key in ENVIRONMENT_OPTIONS}
    values.update(task="B1Z1PickMulti", seed=43, grasp_selector=None,
                  karl_switch_margin_deg=None, karl_orientation_preference=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def scene(num_envs=1, num_candidates=2):
    """A small box above a distant table; +X of every grasp points down."""
    def batch(values):
        return torch.tensor(values, dtype=torch.float64).repeat(num_envs, 1)

    context = {
        "base_quaternion": batch([0., 0., 0., 1.]),
        "arm_base": batch([0., 0., 0.]),
        "object_pose": batch([0., 0., 1., 0., 0., 0., 1.]),
        "object_center_local": batch([0., 0., 0.]),
        "object_half_extents": batch([.1, .06, .05]),
        "table_pose": batch([0., 0., .45, 0., 0., 0., 1.]),
        "table_half_extents": batch([1., 1., .05]),
        "gripper_corners": torch.tensor([[-.005, -.005, -.005],
                                          [.005, .005, .005]], dtype=torch.float64),
        "closing": torch.zeros(num_envs, dtype=torch.bool),
    }
    poses = torch.zeros(num_envs, num_candidates, 6, dtype=torch.float64)
    poses[:, :, 2] = 1.04
    poses[:, :, 4] = math.pi / 2
    ee = batch([-.25, 0., 1.1, 0., math.pi / 2, 0.])
    return poses, ee, context


def weights(**overrides):
    values = dict(center_weight=0., topdown_weight=0., height_weight=0.,
                  distance_weight=0., rotation_weight=0.)
    values.update(overrides)
    return values


class GeometricSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_selection_prefers_center_and_keeps_exact_candidate(self):
        poses, ee, context = scene()
        poses[:, 0, 0] = .08
        selector = GeometricGraspSelector(1, "cpu")
        selected, metrics = selector.select(poses, ee, context)
        self.assertEqual(metrics["geometric_grasp_index"].item(), 1)
        torch.testing.assert_close(selected, poses[:, 1])
        self.assertTrue(selector.initialized.item())

    def test_topdown_uses_approach_x_axis(self):
        poses, ee, context = scene()
        poses[:, 0, 4] = 0.  # +X points sideways; candidate 1 points down.
        costs, _ = geometric_grasp_costs(poses, ee, context, weights(topdown_weight=1.))
        self.assertTrue(torch.isfinite(costs).all())
        self.assertLess(costs[0, 1].item(), costs[0, 0].item())

    def test_costs_invariant_to_rolled_robot_base_coordinates(self):
        poses, ee, context = scene()
        poses[0, 0, :3] += torch.tensor([.06, .01, -.01])
        poses[0, 0, 3:] = torch.tensor([.2, 1.1, -.5])
        before, _ = geometric_grasp_costs(poses, ee, context)

        rotation = Rotation.from_euler("xyz", [.65, -.45, .85])
        origin = np.array([.3, -.4, .2])

        def to_base(world_poses):
            local = world_poses.clone()
            source = world_poses.reshape(-1, 6).numpy()
            result = local.reshape(-1, 6)
            result[:, :3] = torch.from_numpy(rotation.inv().apply(source[:, :3] - origin))
            result[:, 3:] = torch.from_numpy(
                (rotation.inv() * Rotation.from_euler("xyz", source[:, 3:])).as_euler("xyz"))
            return local

        changed = deepcopy(context)
        changed["base_quaternion"][0] = torch.from_numpy(rotation.as_quat())
        changed["arm_base"][0] = torch.from_numpy(origin)
        after, _ = geometric_grasp_costs(to_base(poses), to_base(ee), changed)
        torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-7)

    def test_upper_region_preference_does_not_move_target(self):
        poses, ee, context = scene()
        poses[:, 0, 2], poses[:, 1, 2] = .98, 1.04
        selector = GeometricGraspSelector(1, "cpu", weights(height_weight=1.))
        selected, metrics = selector.select(poses, ee, context)
        self.assertEqual(metrics["geometric_grasp_index"].item(), 1)
        torch.testing.assert_close(selected, poses[:, 1])

    def test_table_collision_checks_gripper_extent_not_only_tcp(self):
        poses, ee, context = scene(num_candidates=1)
        poses[:, :, 2] = .58
        context["object_pose"][:, 2] = .6
        context["object_half_extents"][:, 2] = .1
        context["gripper_corners"] = torch.zeros(1, 3, dtype=torch.float64)
        clear, _ = geometric_grasp_costs(poses, ee, context)
        self.assertTrue(torch.isfinite(clear).all())
        # Downward +X makes the finger tip reach z=.49, below table top .50.
        context["gripper_corners"] = torch.tensor([[.09, 0., 0.], [-.01, 0., 0.]],
                                                    dtype=torch.float64)
        colliding, _ = geometric_grasp_costs(poses, ee, context)
        self.assertTrue(torch.isinf(colliding).all())

    def test_object_validity_uses_transformed_mesh_center_and_oriented_bounds(self):
        poses, ee, context = scene()
        rotation = Rotation.from_euler("z", math.pi / 2)
        context["object_pose"][0, :3] = torch.tensor([1., 2., 1.])
        context["object_pose"][0, 3:] = torch.from_numpy(rotation.as_quat())
        context["object_center_local"][0] = torch.tensor([.3, -.2, 0.])
        context["object_half_extents"][0] = torch.tensor([.1, .01, .05])
        local_points = np.array([[.3, -.12, .04], [.38, -.2, .04]])
        poses[0, :, :3] = torch.from_numpy(rotation.apply(local_points) + [1., 2., 1.])
        costs, _ = geometric_grasp_costs(poses, ee, context)
        self.assertTrue(torch.isinf(costs[0, 0]))
        self.assertTrue(torch.isfinite(costs[0, 1]))

    def test_small_score_improvement_does_not_switch_but_large_improvement_does(self):
        poses, ee, context = scene()
        poses[:, 1, 0] = .06
        selector = GeometricGraspSelector(1, "cpu", weights(center_weight=3., switch_margin=.1))
        self.assertEqual(selector.select(poses, ee, context)[1]["geometric_grasp_index"].item(), 0)
        poses[:, 0, 0], poses[:, 1, 0] = .001, 0.
        held = selector.select(poses, ee, context)[1]
        self.assertEqual(held["geometric_grasp_index"].item(), 0)
        self.assertFalse(held["geometric_grasp_switched"].item())
        poses[:, 0, 0] = .08
        changed = selector.select(poses, ee, context)[1]
        self.assertEqual(changed["geometric_grasp_index"].item(), 1)
        self.assertTrue(changed["geometric_grasp_switched"].item())

    def test_near_closure_latches_target_until_gripper_reopens(self):
        poses, ee, context = scene()
        poses[:, 1, 0] = .07
        selector = GeometricGraspSelector(1, "cpu", weights(center_weight=3.))
        selector.select(poses, ee, context)
        context["closing"][:] = True
        ee = poses[:, 0].clone()
        ee[:, 2] += .02
        self.assertTrue(selector.select(poses, ee, context)[1]["geometric_locked"].item())
        poses[:, 0, 0], poses[:, 1, 0] = .08, 0.
        ee[:, 0] = -.3  # Once latched, EE distance alone must not release it.
        held = selector.select(poses, ee, context)[1]
        self.assertEqual(held["geometric_grasp_index"].item(), 0)
        self.assertTrue(held["geometric_locked"].item())
        context["closing"][:] = False
        released = selector.select(poses, ee, context)[1]
        self.assertFalse(released["geometric_locked"].item())
        self.assertEqual(released["geometric_grasp_index"].item(), 1)

    def test_invalid_locked_target_forces_reselection(self):
        poses, ee, context = scene()
        poses[:, 1, 0] = .06
        selector = GeometricGraspSelector(1, "cpu")
        selector.select(poses, ee, context)
        context["closing"][:] = True
        ee = poses[:, 0].clone()
        self.assertTrue(selector.select(poses, ee, context)[1]["geometric_locked"].item())
        poses[:, 0, :] = float("nan")
        selected, metrics = selector.select(poses, ee, context)
        self.assertEqual(metrics["geometric_grasp_index"].item(), 1)
        self.assertFalse(metrics["geometric_no_valid_grasp"].item())
        torch.testing.assert_close(selected, poses[:, 1])

    def test_all_invalid_has_explicit_fallback_and_recovers_to_valid_candidate(self):
        poses, ee, context = scene()
        selector = GeometricGraspSelector(1, "cpu")
        poses[:] = float("nan")
        selected, metrics = selector.select(poses, ee, context)
        self.assertEqual(metrics["geometric_grasp_index"].item(), -1)
        self.assertTrue(metrics["geometric_no_valid_grasp"].item())
        self.assertEqual(metrics["geometric_valid_count"].item(), 0)
        self.assertTrue(torch.isfinite(selected).all())
        torch.testing.assert_close(selected, ee)
        poses[:, 1] = torch.tensor([0., 0., 1.04, 0., math.pi / 2, 0.])
        selected, metrics = selector.select(poses, ee, context)
        self.assertEqual(metrics["geometric_grasp_index"].item(), 1)
        self.assertFalse(metrics["geometric_no_valid_grasp"].item())
        torch.testing.assert_close(selected, poses[:, 1])

    def test_partial_reset_clears_only_requested_episode_state(self):
        poses, ee, context = scene(num_envs=2)
        poses[:, 0, 0] = .07
        selector = GeometricGraspSelector(2, "cpu")
        selector.select(poses, ee, context)
        context["closing"][:] = True
        ee = poses[:, 1].clone()
        selector.select(poses, ee, context)
        self.assertTrue(selector.locked.all())
        selector.reset(torch.tensor([0]))
        self.assertEqual(selector.initialized.tolist(), [False, True])
        self.assertEqual(selector.locked.tolist(), [False, True])
        self.assertEqual(selector.indices[1].item(), 1)
        selector.reset()
        self.assertFalse(selector.initialized.any())
        self.assertFalse(selector.locked.any())

    def test_selection_does_not_retain_autograd_graph(self):
        poses, ee, context = scene()
        poses.requires_grad_()
        ee.requires_grad_()
        selected, metrics = GeometricGraspSelector(1, "cpu").select(poses, ee, context)
        self.assertFalse(selected.requires_grad)
        for tensor in metrics.values():
            if isinstance(tensor, torch.Tensor):
                self.assertFalse(tensor.requires_grad)

    def test_invalid_world_context_cannot_make_a_valid_grasp(self):
        for invalid in ("base_quaternion", "object_quaternion", "table_pose"):
            poses, ee, context = scene()
            if invalid == "base_quaternion":
                context["base_quaternion"].zero_()
            elif invalid == "object_quaternion":
                context["object_pose"][:, 3:7].zero_()
            else:
                context["table_pose"][:, 2] = float("nan")
            with self.subTest(invalid=invalid):
                selected, metrics = GeometricGraspSelector(1, "cpu").select(poses, ee, context)
                self.assertTrue(metrics["geometric_no_valid_grasp"].all())
                self.assertEqual(metrics["geometric_valid_count"].item(), 0)
                self.assertEqual(metrics["geometric_grasp_index"].item(), -1)
                self.assertTrue(torch.isfinite(selected).all())
                torch.testing.assert_close(selected, ee)


class GeometricConfigurationTests(unittest.TestCase):
    def test_partial_geometric_yaml_merges_defaults_and_preserves_overrides(self):
        cfg = {"env": {}, "grasp_selection": {
            "mode": "geometric", "geometric": {"height_fraction": .6, "center_weight": 4.}}}
        resolved_cfg, settings = resolve_selection(options(roboinfo=True), cfg)
        expected = dict(GEOMETRIC_DEFAULTS, height_fraction=.6, center_weight=4.)
        self.assertEqual(settings["geometric"], expected)
        self.assertEqual(settings["mode"], "geometric")
        self.assertEqual(resolved_cfg["grasp_selection"], settings)
        for mode in ("gfm", "karl"):
            original = {"mode": mode, "switch_margin_deg": 30., "orientation_preference": "none",
                        "teacher_actor": "privileged"}
            _, baseline = resolve_selection(options(grasp_selector=mode, roboinfo=True), {"env": {}})
            self.assertEqual(baseline, original)

    def test_defaults_are_complete_and_old_modes_keep_privileged_actor(self):
        baseline = {"mode": "gfm", "switch_margin_deg": 30., "orientation_preference": "none",
                    "teacher_actor": "privileged"}
        self.assertEqual(resolve_selection(options(), {"env": {}})[1], baseline)
        karl = dict(baseline, mode="karl")
        self.assertEqual(resolve_selection(options(grasp_selector="karl", roboinfo=True),
                                           {"env": {}})[1], karl)
        _, geometric = resolve_selection(options(grasp_selector="geometric", roboinfo=True), {"env": {}})
        self.assertEqual(geometric["geometric"], GEOMETRIC_DEFAULTS)
        self.assertNotIn("geometric", baseline)

    def test_invalid_values_and_cross_mode_flags_are_rejected(self):
        for invalid in ({"center_weight": -1.}, {"switch_margin": float("nan")},
                        {"table_clearance": -.001}, {"lock_distance": float("inf")}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_geometric_settings(invalid)
        cases = [options(grasp_selector="geometric", roboinfo=True, karl_switch_margin_deg=30.),
                 options(grasp_selector="geometric", roboinfo=True, karl_orientation_preference="none"),
                 options(grasp_selector="karl", roboinfo=True, geometric_switch_margin=.1),
                 options(geometric_center_weight=3.)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_selection(args, {"env": {}})

    def test_geometric_checkpoint_restores_all_settings_and_environment_convention(self):
        args = options(grasp_selector="geometric", roboinfo=True, observe_gait_commands=True,
                       geometric_center_weight=4.5, geometric_switch_margin=.2)
        cfg, settings = resolve_selection(args, {"env": {"numEnvs": 7}})
        env = SimpleNamespace(global_step_counter=321)
        state = TeacherGraspTrainingState(env, settings, cfg, args).state_dict()
        saved = {"grasp_selection_state": state}
        resumed_args = options()
        resumed_cfg, resumed = resolve_selection(resumed_args, {"env": {"numEnvs": 1}}, saved)
        self.assertEqual(resumed, settings)
        self.assertEqual(resumed_cfg, cfg)
        self.assertTrue(resumed_args.roboinfo)
        self.assertTrue(resumed_args.observe_gait_commands)
        resolve_selection(options(geometric_center_weight=4.5), {"env": {}}, saved)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_selection(options(geometric_center_weight=3.), {"env": {}}, saved)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_selection(options(grasp_selector="karl"), {"env": {}}, saved)
        corrupt = deepcopy(saved)
        del corrupt["grasp_selection_state"]["settings"]["geometric"]["center_weight"]
        with self.assertRaisesRegex(ValueError, "checkpoint settings"):
            resolve_selection(options(), {"env": {}}, corrupt)

    def test_geometric_mode_requires_original_teacher_observation_and_viewer_for_markers(self):
        for override in ({"roboinfo": False}, {"no_feature": True},
                         {"last_commands": True}, {"pitch_control": True},
                         {"vis_selected_grasp": True, "headless": True}):
            values = {"grasp_selector": "geometric", "roboinfo": True}
            values.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                resolve_selection(options(**values), {"env": {}})
        _, settings = resolve_selection(options(grasp_selector="geometric", roboinfo=True,
                                                vis_selected_grasp=True, grasp_vis_envs=5), {"env": {}})
        self.assertEqual(settings["mode"], "geometric")


class RawGeometricTeacherEnv:
    """A reused-buffer environment exposing terminal observations before reset."""
    num_envs, num_states, num_agents = 2, 0, 1
    device = rl_device = "cpu"
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1276,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (9,), np.float32)

    def __init__(self):
        poses, ee, context = scene(2, 30)
        self.poses = poses.float()
        self.poses[:, :, 0] = .08
        self.poses[:, 7, 0] = 0.
        self.context = {key: value.float() if value.dtype == torch.float64 else value
                        for key, value in context.items()}
        self.obs = torch.zeros(2, 1276)
        self.obs[:, :1024] = torch.linspace(-.1, .1, 1024)
        self.obs[:, 1030:1036] = ee.float()
        self.reset_buf = torch.zeros(2, dtype=torch.long)
        self.global_step_counter = 0
        self.write_poses()

    def write_poses(self):
        self.obs[:, -189:-99] = self.poses[:, :, :3].reshape(2, 90)
        self.obs[:, -99:-9] = self.poses[:, :, 3:].reshape(2, 90)

    def reset(self):
        self.reset_buf.zero_()
        return {"obs": self.obs}

    def step(self, actions):
        self.global_step_counter += 1
        self.obs[:, -9:] = actions
        self.context["closing"] = actions[:, 6] < 0
        self.reset_buf[0] = 1
        self.write_poses()
        return {"obs": self.obs}, actions[:, 0].clone(), self.reset_buf, {}


class GeometricWrapperTests(unittest.TestCase):
    def build(self):
        raw = RawGeometricTeacherEnv()
        provider = SimpleNamespace(context=lambda: raw.context)
        with patch("utils.geometric_teacher_wrapper.TeacherGraspGeometry", return_value=provider):
            wrapper = GeometricTeacherWrapper(raw, weights(center_weight=3.))
        return raw, wrapper

    def test_packed_snapshot_and_visualizer_use_the_same_selected_pose(self):
        raw, env = self.build()

        class VisualizerSpy:
            def update(self, selected, metrics):
                self.selected = selected.clone()
                self.indices = metrics["geometric_grasp_index"].clone()

            def reset(self, ids=None):
                pass

        spy = VisualizerSpy()
        env._grasp_visualizer = spy
        packed, info = env.reset()
        self.assertEqual(packed.shape, (2, 1102))
        self.assertEqual(env.observation_space.shape, (1102,))
        torch.testing.assert_close(packed[:, :1087], raw.obs[:, :1087])
        torch.testing.assert_close(packed[:, -15:-9], raw.poses[:, 7])
        torch.testing.assert_close(packed[:, -15:-9], spy.selected)
        torch.testing.assert_close(info["geometric_grasp_index"], spy.indices)
        torch.testing.assert_close(packed[:, -9:], raw.obs[:, -9:])
        saved = packed.clone()
        raw.obs.add_(.25)
        torch.testing.assert_close(packed, saved)

    def test_terminal_observation_keeps_state_until_actual_partial_reset(self):
        raw, env = self.build()
        self.assertEqual(env.reset()[1]["geometric_grasp_index"].tolist(), [7, 7])
        # Candidate 3 improves only slightly, so an existing episode holds 7.
        # A fresh episode must choose 3 directly without hysteresis.
        raw.poses[:, 7, 0] = .001
        raw.poses[:, 3, 0] = 0.
        terminal, _, terminated, _, info = env.step(torch.zeros(2, 9))
        self.assertEqual(terminated.flatten().tolist(), [1, 0])
        self.assertEqual(info["geometric_grasp_index"].tolist(), [7, 7])
        torch.testing.assert_close(terminal[:, -15:-9], raw.poses[:, 7])
        reset, info = env.reset()
        self.assertEqual(info["geometric_grasp_index"].tolist(), [3, 7])
        torch.testing.assert_close(reset[0, -15:-9], raw.poses[0, 3])
        torch.testing.assert_close(reset[1, -15:-9], raw.poses[1, 7])

    def test_ppo_replay_likelihood_does_not_depend_on_later_selection_state(self):
        raw, env = self.build()
        packed, _ = env.reset()
        policy = KarlTeacherPolicy(env.observation_space, env.action_space)
        scaler = RunningStandardScaler(1102, device="cpu")
        scaler(torch.randn(12, 1102), train=True)
        action, before, _ = policy.act({"states": scaler(packed)}, "policy")
        raw.poses[:, 7, 0] = .08
        raw.poses[:, 3, 0] = 0.
        _, _, _, _, info = env.step(torch.zeros(2, 9))
        self.assertEqual(info["geometric_grasp_index"].tolist(), [3, 3])
        order = torch.tensor([1, 0])
        _, after, _ = policy.act({"states": scaler(packed[order]),
                                  "taken_actions": action[order]}, "policy")
        torch.testing.assert_close((after - before[order]).exp(), torch.ones_like(after),
                                   atol=1e-6, rtol=1e-6)

    def test_diagnostics_average_all_environments_and_steps_once_per_rollout(self):
        raw, env = self.build()
        received = []
        env.diagnostics_callback = lambda key, value: received.append((key, value))
        env.reset()
        for step in range(24):
            # First 12 steps: one of two environments has no valid candidates.
            # Next 12 steps: both are valid. Rollout no-valid mean must be .25.
            raw.poses[0, :, 0] = 10. if step < 12 else .08
            if step >= 12:
                raw.poses[0, 7, 0] = 0.
            env.step(torch.zeros(2, 9))
            if step < 23:
                self.assertEqual(received, [])
        self.assertEqual(len(received), len(env.DIAGNOSTICS))
        measured = dict(received)
        self.assertAlmostEqual(measured["Grasp geometric / no_valid_grasp"], .25)
        self.assertAlmostEqual(measured["Grasp geometric / valid_count"], 22.5)
        self.assertEqual(env._diagnostic_steps, 0)
        self.assertIsNone(env._diagnostic_sum)
        env.step(torch.zeros(2, 9))
        self.assertEqual(len(received), len(env.DIAGNOSTICS))


class GeometricAssetTests(unittest.TestCase):
    ROBOT_URDF = ROOT / "DQ_high-level/data/asset/b1z1-col/urdf/b1z1.urdf"

    def test_real_z1_envelope_uses_tcp_frame_and_contains_moving_finger(self):
        corners = load_gripper_envelope_corners(self.ROBOT_URDF)
        self.assertEqual(corners.shape, (8, 3))
        self.assertTrue(np.isfinite(corners).all())
        lower, upper = corners.min(0), corners.max(0)
        # The URDF TCP lies 135 mm along stator +X. The fixed fingertips
        # extend just 15 mm beyond it; forgetting this offset is a large error.
        self.assertTrue(-.16 < lower[0] < -.12)
        self.assertTrue(.015 < upper[0] < .025)
        # Independently sample the URDF's distal moving box through angles
        # between the loader's samples, including both joint endpoints.
        box = np.asarray(list(product((-1., 1.), repeat=3))) * [.013, .03, .009]
        box += [.085, 0., 0.]
        angles = np.linspace(-math.pi / 2, 0., 169)
        points = np.concatenate([
            Rotation.from_euler("y", angle).apply(box) + [.049 - .135, 0., 0.]
            for angle in angles])
        self.assertTrue((points >= lower - 1e-10).all())
        self.assertTrue((points <= upper + 1e-10).all())

    def test_object_loader_honors_mesh_scale_origin_and_rejects_unknown_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_dir = root / "obj_set" / "test_box"
            object_dir.mkdir(parents=True)
            mesh = object_dir / "collision.obj"
            mesh.write_text("".join("v %s %s %s\n" % point
                                    for point in product((0., 1.), (0., 2.), (0., 3.))))
            path = object_dir / "model.urdf"
            valid = ('<robot name="fixture"><link name="body"><collision>'
                     '<origin xyz="1 2 3" rpy="0 0 %.17g"/>'
                     '<geometry><mesh filename="collision.obj" scale="2 3 4"/></geometry>'
                     '</collision></link></robot>') % (math.pi / 2)
            path.write_text(valid)
            lower, upper = load_object_collision_bounds(path)
            np.testing.assert_allclose(lower, [-5., 2., 3.], atol=1e-12)
            np.testing.assert_allclose(upper, [1., 4., 15.], atol=1e-12)

            # Simulation does not apply asset_multi.scale. Cached bounds must
            # therefore apply only the actual URDF mesh scale tested above.
            env = SimpleNamespace(num_envs=2, rl_device="cpu", obj_list=["test_box"],
                                  table_dims=SimpleNamespace(x=2., y=2., z=.1),
                                  cfg={"env": {"asset": {
                                      "assetRoot": str(root), "assetFileObj": "obj_set",
                                      "assetFileRobot": str(self.ROBOT_URDF),
                                      "asset_multi": {"test_box": {"scale": 7.}}}}})
            geometry = TeacherGraspGeometry(env)
            np.testing.assert_allclose(geometry.object_center_local.numpy(), [[-2., 3., 9.]] * 2)
            np.testing.assert_allclose(geometry.object_half_extents.numpy(), [[3., 1., 6.]] * 2)

            path.write_text('<robot name="fixture"><link name="body"><collision>'
                            '<geometry><cylinder radius=".1" length=".2"/></geometry>'
                            '</collision></link></robot>')
            with self.assertRaisesRegex(ValueError, "Unsupported collision geometry"):
                load_object_collision_bounds(path)
            path.write_text(valid.replace("collision.obj", "missing.obj"))
            with self.assertRaises(FileNotFoundError):
                load_object_collision_bounds(path)


if __name__ == "__main__":
    unittest.main()
