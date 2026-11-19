"""Direction 8 grasp tokens: geometry, layout, scaler and models, on CPU without Isaac Gym.

The h = 0 tokens must describe exactly the candidates of the old 180-dim block
(production `_grasp_candidate_observations`), and each h > 0 token must be the
same transform applied to the EKF forecast for that horizon.
"""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import math
import unittest

import gymnasium
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from modules.grasp_tokens import (GraspTokenLayout, build_grasp_tokens, layout_from_config,
                                  rotation_features, split_grasp_tokens)
from PredictPiontTransform.predictpoint_mul import PredictPoint
from skrl.resources.preprocessors.torch import RunningStandardScaler
from test_critic_time_obs import SetupBase, task_with
from test_sgs_env import NAMESPACE, method_node
from test_teacher_gfm import ROOT, Policy, Value
from utils import config as config_utils
from utils.state_scaler import PassthroughStandardScaler, TokenStandardScaler, teacher_state_preprocessor


SPEC = importlib.util.spec_from_file_location(
    "grasp_token_test_torch_utils",
    ROOT.parent / "third_party/isaacgym/python/isaacgym/torch_utils.py")
QUATERNIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUATERNIONS)

ENVS, CANDIDATES, ACTIONS = 5, 30, 9
ARM_OFFSET = torch.tensor([0.3, 0.0, 0.09])


def random_quaternions(count, seed):
    return torch.tensor(Rotation.random(count, random_state=seed).as_quat(), dtype=torch.float32)


def pick_task():
    """Production candidate/token methods on a real PredictPoint (fixed gripper matrices)."""
    methods = ("_grasp_candidate_observations", "_grasp_token_observations",
               "_current_object_pose_in_arm_frame")
    node = ast.ClassDef(name="Task", bases=[ast.Name(id="PredictPoint", ctx=ast.Load())],
                        keywords=[], decorator_list=[],
                        body=[method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name) for name in methods])
    namespace = dict(vars(QUATERNIONS), PredictPoint=PredictPoint,
                     build_grasp_tokens=build_grasp_tokens,
                     object_pose_in_arm_frame=NAMESPACE["object_pose_in_arm_frame"])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(ROOT / "envs/b1z1_pickmulti.py"), "exec"), namespace)
    with redirect_stdout(io.StringIO()):
        predictor = PredictPoint(torch.zeros(1, 1, 6), torch.eye(4)[None, None],
                                 torch.zeros(1, 1, 6), device="cpu")
    task = namespace["Task"].__new__(namespace["Task"])
    task.__dict__.update(predictor.__dict__)

    generator = torch.Generator().manual_seed(8)
    grasps = torch.eye(4).repeat(ENVS, CANDIDATES, 1, 1)
    grasps[..., :3, :3] = torch.tensor(Rotation.random(ENVS * CANDIDATES, random_state=3).as_matrix(),
                                       dtype=torch.float32).view(ENVS, CANDIDATES, 3, 3)
    grasps[..., :3, 3] = 0.08 * torch.randn(ENVS, CANDIDATES, 3, generator=generator)
    task.T_grasp_in_obj = grasps
    robot = torch.zeros(ENVS, 13)
    robot[:, :3] = torch.randn(ENVS, 3, generator=generator)
    robot[:, 3:7] = random_quaternions(ENVS, 4)
    cube = torch.zeros(ENVS, 13)
    cube[:, :3] = robot[:, :3] + 0.4 * torch.randn(ENVS, 3, generator=generator)
    cube[:, 3:7] = random_quaternions(ENVS, 5)
    task._robot_root_states, task._cube_root_states = robot, cube
    task.arm_base = QUATERNIONS.quat_apply(robot[:, 3:7], ARM_OFFSET.expand(ENVS, 3)) + robot[:, :3]
    task.device = "cpu"
    task.lifted_object = torch.zeros(ENVS, dtype=torch.bool)
    # Tail of the robot observation: object world pose [xyz, roll/pitch/yaw].
    rpy = torch.stack(QUATERNIONS.euler_from_quat(cube[:, 3:7]), dim=-1)
    task.robot_obs = torch.cat((torch.zeros(ENVS, 4), cube[:, :3], rpy), dim=-1)
    return task


def ypr_matrices(ypr):
    angles = ypr.detach().double().numpy().reshape(-1, 3)
    return Rotation.from_euler("ZYX", angles).as_matrix().reshape(*ypr.shape[:-1], 3, 3)


class TokenGeometryTests(unittest.TestCase):
    def setUp(self):
        self.task = pick_task()
        self.ids = torch.arange(ENVS)
        old = self.task._grasp_candidate_observations(self.task.robot_obs, self.ids)
        self.old_positions = old[:, :3 * CANDIDATES].view(ENVS, CANDIDATES, 3)
        self.old_ypr = old[:, 3 * CANDIDATES:].view(ENVS, CANDIDATES, 3)

    def tokens(self, rotation="rot6d", horizons=(0.0,)):
        self.task.grasp_token_layout = layout = GraspTokenLayout(horizons=horizons, rotation=rotation)
        block = self.task._grasp_token_observations(self.ids)
        self.assertEqual(block.shape, (ENVS, layout.width))
        return split_grasp_tokens(block, layout)

    def test_current_tokens_match_the_old_candidate_block(self):
        tokens, active = self.tokens("rot6d")
        torch.testing.assert_close(tokens[..., :3], self.old_positions, atol=2e-5, rtol=0)
        expected = torch.tensor(ypr_matrices(self.old_ypr), dtype=torch.float32)
        torch.testing.assert_close(tokens[..., 3:6], expected[..., :, 0], atol=2e-5, rtol=0)
        torch.testing.assert_close(tokens[..., 6:9], expected[..., :, 1], atol=2e-5, rtol=0)
        # Reach features follow from the position.
        torch.testing.assert_close(tokens[..., 9], torch.linalg.norm(self.old_positions[..., :2], dim=-1),
                                   atol=2e-5, rtol=0)
        torch.testing.assert_close(tokens[..., 10], torch.atan2(self.old_positions[..., 1],
                                                                self.old_positions[..., 0]), atol=2e-4, rtol=0)
        self.assertFalse(active.bool().any())

    def test_euler_tokens_repeat_the_old_yaw_pitch_roll(self):
        tokens, _ = self.tokens("euler")
        difference = torch.remainder(tokens[..., 3:6] - self.old_ypr + math.pi, 2 * math.pi) - math.pi
        torch.testing.assert_close(difference, torch.zeros_like(difference), atol=5e-5, rtol=0)

    def test_future_tokens_apply_the_same_transform_to_each_forecast(self):
        forecasts = torch.zeros(ENVS, 2, 7)
        forecasts[..., :3] = torch.randn(ENVS, 2, 3, generator=torch.Generator().manual_seed(2))
        forecasts[..., 3:] = random_quaternions(2 * ENVS, 6).view(ENVS, 2, 4)
        valid = torch.tensor([True, True, False, True, False])
        self.task.lifted_object = torch.tensor([False, True, False, False, False])
        self.task.object_pose_predictor = type("EKF", (), {"future_poses": forecasts, "valid": valid})()
        tokens, active = self.tokens("rot6d", horizons=(0.0, 0.6, 0.9))
        current, _ = self.tokens("rot6d")
        torch.testing.assert_close(tokens[:, :CANDIDATES], current)
        single = GraspTokenLayout(horizons=(0.0,))
        correction = self.task.Rx_rot @ self.task.Rz_rot @ self.task.Rx_rot
        for k in range(2):
            expected = build_grasp_tokens(forecasts[:, k:k + 1, :3], forecasts[:, k:k + 1, 3:],
                                          self.task.T_grasp_in_obj, correction,
                                          torch.zeros(ENVS, dtype=torch.bool), single)
            expected, _ = split_grasp_tokens(expected, single)
            torch.testing.assert_close(tokens[:, (k + 1) * CANDIDATES:(k + 2) * CANDIDATES], expected)
        # Valid forecast and not lifted.
        self.assertEqual(active.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0])

    def test_rot6d_is_continuous_across_the_yaw_wrap_where_euler_jumps(self):
        angles = torch.tensor([math.pi - 1e-3, -math.pi + 1e-3])
        matrices = torch.tensor(Rotation.from_euler("z", angles.numpy()).as_matrix(), dtype=torch.float32)
        rot6d = rotation_features(matrices, "rot6d")
        euler = rotation_features(matrices, "euler")
        self.assertLess((rot6d[0] - rot6d[1]).abs().max().item(), 3e-3)
        self.assertGreater((euler[0, 0] - euler[1, 0]).abs().item(), 6.0)


class TokenLayoutTests(unittest.TestCase):
    def test_widths(self):
        self.assertEqual(GraspTokenLayout().width, 30 * 11 + 1)
        self.assertEqual(GraspTokenLayout(rotation="euler").width, 30 * 8 + 1)
        self.assertEqual(GraspTokenLayout(horizons=(0.0, 0.6, 0.9)).width, 90 * 11 + 1)

    def test_config(self):
        prediction = {"enabled": True, "horizons": [0.6, 0.9]}
        self.assertIsNone(layout_from_config({"env": {}}))
        self.assertIsNone(layout_from_config({"env": {"graspTokens": {"enabled": False}}}))
        layout = layout_from_config({"env": {"graspTokens": {"enabled": True, "future": True},
                                             "posePrediction": prediction}})
        self.assertEqual(layout.horizons, (0.0, 0.6, 0.9))
        # Saved configs from before the normalization option keep the raw tokens.
        self.assertEqual(layout.normalization, "raw")
        shared = layout_from_config({"env": {"graspTokens": {"enabled": True, "normalization": "shared"}}})
        self.assertEqual(shared.normalization, "shared")
        self.assertEqual(shared.width, GraspTokenLayout().width)
        for tokens, prediction_cfg in (({"enabled": True, "future": True}, {"enabled": False}),
                                       ({"enabled": False, "future": True}, prediction),
                                       ({"enabled": "true"}, prediction),
                                       ({"enabled": True, "rotation": "quat"}, prediction),
                                       ({"enabled": True, "normalization": "layer"}, prediction)):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                layout_from_config({"env": {"graspTokens": tokens, "posePrediction": prediction_cfg}})

    def test_cli_overrides_only_explicit_flags(self):
        cfg = {"env": {"graspTokens": {"enabled": True, "future": True}}}
        args = type("Args", (), {"grasp_tokens": None, "grasp_token_rotation": "euler",
                                 "grasp_future_tokens": False, "grasp_token_norm": None})()
        self.assertEqual(config_utils.apply_grasp_token_config(cfg, args),
                         {"enabled": True, "future": False, "rotation": "euler", "normalization": "raw"})
        args.grasp_token_norm = "shared"
        self.assertEqual(config_utils.apply_grasp_token_config(cfg, args)["normalization"], "shared")

    def test_repository_config_uses_shared_normalization(self):
        cfg = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertEqual(cfg["env"]["graspTokens"]["normalization"], "shared")
        self.assertIs(cfg["env"]["criticGfm"], True)

    def test_new_cli_flags(self):
        parse = lambda *argv: config_utils.get_params(list(argv))
        args = parse()
        self.assertIsNone(args.grasp_token_norm)
        self.assertIsNone(args.critic_gfm)
        self.assertFalse(args.value_norm_returns_only)
        args = parse("--grasp_token_norm", "raw", "--no_critic_gfm", "--value_norm_returns_only")
        self.assertEqual((args.grasp_token_norm, args.critic_gfm, args.value_norm_returns_only),
                         ("raw", False, True))
        self.assertTrue(parse("--no_critic_gfm", "--critic_gfm").critic_gfm)

    def setup_task(self, tokens=None, tsil=True, critic_time=True):
        task = task_with("_setup_obs_and_action_info", base=SetupBase)
        task.no_feature, task.num_features = False, 1024
        env = {"noVelocityObs": True, "lastCommands": False, "tsil": {"enabled": tsil},
               "criticTimeObs": critic_time,
               "posePrediction": {"enabled": True, "horizons": [.6, .9]}}
        if tokens is not None:
            env["graspTokens"] = tokens
        task.cfg = {"env": env}
        task._setup_obs_and_action_info()
        return task

    def test_token_block_replaces_the_candidate_block_and_shifts_end_indices(self):
        old = self.setup_task()
        self.assertEqual(old.num_grasp_obs, 180)
        self.assertEqual(old.tsil_time_indices, (-191, -190))
        for future in (False, True):
            with self.subTest(future=future):
                task = self.setup_task({"enabled": True, "future": future})
                shift = task.grasp_token_layout.width - 180
                self.assertEqual(task.dimensions["num_obs"], old.dimensions["num_obs"] + shift)
                self.assertEqual(task.dimensions["removed_dim"], old.dimensions["removed_dim"] + shift)
                self.assertEqual(task.critic_time_obs_index, old.critic_time_obs_index - shift)
                self.assertEqual(task.tsil_time_indices,
                                 tuple(index - shift for index in old.tsil_time_indices))


class TokenModelTests(unittest.TestCase):
    feature_width, body_width = 16, 24
    layout = GraspTokenLayout(horizons=(0.0, 0.6, 0.9))

    def state(self, active):
        generator = torch.Generator().manual_seed(11)
        random = lambda width: 0.3 * torch.randn(2, width, generator=generator)
        future = random(13)
        future[:, -1] = 1.0
        block = random(self.layout.width)
        block[:, -1] = active
        return torch.cat((random(self.feature_width), random(self.body_width), future, block,
                          random(ACTIONS)), dim=-1)

    def model(self, cls, layout=None, **kwargs):
        torch.manual_seed(5)
        width = self.feature_width + self.body_width + 13 + self.layout.width + ACTIONS
        return cls(width, ACTIONS, "cpu", self.feature_width, 128, None, None, None,
                   num_future_pose_obs=13, grasp_token_layout=layout or self.layout, **kwargs)

    def test_outputs_mlp_width_and_future_mask(self):
        future = torch.arange(self.layout.num_tokens) >= CANDIDATES
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                model = self.model(cls)
                self.assertEqual(model.selected_dim, 64)
                self.assertFalse(hasattr(model, "output_proj"))
                self.assertEqual(model.net[0].in_features,
                                 self.body_width + 13 + ACTIONS + 128 + 64)
                for active in (0.0, 1.0):
                    output = model.compute({"states": self.state(active)}, role="policy")[0]
                    self.assertEqual(output.shape, (2, ACTIONS if cls is Policy else 1))
                    attention = model.last_attention
                    torch.testing.assert_close(attention.sum(-1), torch.ones(2))
                    if active:
                        self.assertTrue((attention[:, future] > 0).all())
                    else:
                        self.assertEqual(attention[:, future].abs().max().item(), 0.0)

    def test_tokens_receive_gradients_and_checkpoints_round_trip(self):
        model = self.model(Policy)
        state = self.state(1.0).requires_grad_(True)
        model.compute({"states": state}, role="policy")[0].sum().backward()
        start = self.feature_width + self.body_width + 13
        self.assertGreater(state.grad[:, start:start + self.layout.width - 1].abs().sum().item(), 0)
        copy = self.model(Policy)
        copy.load_state_dict(model.state_dict(), strict=True)
        self.assertNotIn("token_h", model.state_dict())

    def test_rollout_attention_diagnostics(self):
        model, calls = self.model(Policy), []
        model.attention_tracker = lambda tag, value: calls.append(tag)
        model.compute({"states": self.state(1.0)}, role="policy")       # update pass: not logged
        self.assertEqual(calls, [])
        with torch.no_grad():
            for _ in range(2):
                model.compute({"states": self.state(1.0)}, role="policy")
        self.assertEqual(calls.count("GFM / attention entropy"), 2)
        self.assertEqual(calls.count("GFM / future token mass"), 2)
        self.assertEqual(calls.count("GFM / argmax switch rate"), 1)

    def test_original_gfm_logs_policy_and_critic_attention(self):
        width = self.feature_width + self.body_width + 13 + 180 + ACTIONS
        state = 0.3 * torch.randn(2, width, generator=torch.Generator().manual_seed(12))
        for cls, tag in ((Policy, "GFM"), (Value, "GFM critic")):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(5)
                model, calls = cls(width, ACTIONS, "cpu", self.feature_width, 128, None, None, None,
                                   num_future_pose_obs=13), []
                model.attention_tracker = lambda name, value: calls.append(name)
                with torch.no_grad():
                    for _ in range(2):
                        model.compute({"states": state}, role="policy")
                self.assertEqual(model.last_attention.shape, (2, CANDIDATES))
                self.assertEqual(calls.count(f"{tag} / attention entropy"), 2)
                self.assertEqual(calls.count(f"{tag} / argmax switch rate"), 1)
                self.assertFalse(any("future token mass" in name for name in calls))

    def test_shared_normalization_layer_norms_the_embedding(self):
        layout = GraspTokenLayout(horizons=(0.0, 0.6, 0.9), normalization="shared")
        self.assertFalse(hasattr(self.model(Policy), "output_norm"))
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                model, captured = self.model(cls, layout), []
                self.assertIsInstance(model.output_norm, torch.nn.LayerNorm)
                self.assertIn("output_norm.weight", model.state_dict())
                model.net[0].register_forward_hook(lambda module, inputs, output: captured.append(inputs[0]))
                model.compute({"states": self.state(1.0)}, role="policy")
                embedding = captured[0][:, -64:]
                torch.testing.assert_close(embedding.mean(-1), torch.zeros(2), atol=1e-5, rtol=0)
                torch.testing.assert_close(embedding.var(-1, unbiased=False), torch.ones(2), atol=2e-2, rtol=0)
                copy = self.model(cls, layout)
                copy.load_state_dict(model.state_dict(), strict=True)

    def test_critic_without_gfm_ignores_the_grasp_block(self):
        value, calls = self.model(Value, critic_gfm=False), []
        value.attention_tracker = lambda name, number: calls.append(name)
        self.assertEqual(value.selected_dim, 0)
        self.assertFalse([key for key in value.state_dict()
                          if key.startswith(("key_proj", "value_proj", "query_proj", "output_"))])
        self.assertEqual(value.net[0].in_features, self.body_width + 13 + ACTIONS + 128)
        state = self.state(1.0)
        changed = state.clone()
        start = self.feature_width + self.body_width + 13
        changed[:, start:start + self.layout.width] = 5.0
        with torch.no_grad():
            first = value.compute({"states": state}, role="value")[0]
            second = value.compute({"states": changed}, role="value")[0]
        self.assertEqual(first.shape, (2, 1))
        torch.testing.assert_close(first, second)
        self.assertIsNone(value.last_attention)
        self.assertEqual(calls, [])
        copy = self.model(Value, critic_gfm=False)
        copy.load_state_dict(value.state_dict(), strict=True)
        # The actor keeps its GFM.
        self.assertEqual(self.model(Policy).selected_dim, 64)


class PassthroughScalerTests(unittest.TestCase):
    def test_token_slice_stays_raw_and_other_dimensions_are_scaled(self):
        observation = gymnasium.spaces.Box(-np.inf, np.inf, shape=(20,))
        action = gymnasium.spaces.Box(-1, 1, shape=(3,))
        layout = GraspTokenLayout(num_candidates=1, rotation="euler")    # width 9
        scaler_class, kwargs = teacher_state_preprocessor(observation, action, "cpu", layout)
        self.assertIs(scaler_class, PassthroughStandardScaler)
        self.assertEqual(kwargs["passthrough"], (8, 17))
        self.assertIs(teacher_state_preprocessor(observation, action, "cpu")[0], RunningStandardScaler)
        scaler = scaler_class(**kwargs)
        data = 3.0 + 2.0 * torch.randn(4000, 20, generator=torch.Generator().manual_seed(1))
        scaled = scaler(data, train=True)
        torch.testing.assert_close(scaled[:, 8:17], data[:, 8:17])
        outside = torch.cat((scaled[:, :8], scaled[:, 17:]), dim=-1)
        self.assertLess(outside.mean(0).abs().max().item(), 0.05)
        self.assertLess((outside.std(0) - 1).abs().max().item(), 0.05)
        torch.testing.assert_close(scaler(data[:3], inverse=True)[:, 8:17], data[:3, 8:17])
        reference = RunningStandardScaler(observation, device="cpu")
        reference.load_state_dict(scaler.state_dict(), strict=True)


class TokenScalerTests(unittest.TestCase):
    observation = gymnasium.spaces.Box(-np.inf, np.inf, shape=(40,))
    action = gymnasium.spaces.Box(-1, 1, shape=(3,))
    # 3 tokens x 8 features + flag = 25 dims, ending before the 3 action dims: (12, 37).
    layout = GraspTokenLayout(num_candidates=3, rotation="euler", normalization="shared")

    def scaler(self):
        scaler_class, kwargs = teacher_state_preprocessor(self.observation, self.action, "cpu", self.layout)
        self.assertIs(scaler_class, TokenStandardScaler)
        self.assertEqual((kwargs["token_block"], kwargs["feature_dim"]), ((12, 37), 8))
        return scaler_class(**kwargs), kwargs

    @staticmethod
    def data():
        generator = torch.Generator().manual_seed(4)
        data = 3.0 + 2.0 * torch.randn(6000, 40, generator=generator)
        # Token k is offset by 4k, so the tokens' means differ.
        data[:, 12:36] += 4.0 * torch.arange(3.0).repeat_interleave(8)
        data[:, 36] = (torch.rand(6000, generator=generator) > 0.5).float()
        return data

    def test_tokens_share_statistics_and_the_flag_stays_raw(self):
        scaler, _ = self.scaler()
        data = self.data()
        scaled = scaler(data, train=True)
        tokens = scaled[:, 12:36].reshape(-1, 3, 8)
        pooled = tokens.reshape(-1, 8)
        self.assertLess(pooled.mean(0).abs().max().item(), 0.05)
        self.assertLess((pooled.std(0) - 1).abs().max().item(), 0.05)
        # Shared statistics keep the tokens' relative offsets (per-token scaling would erase them).
        per_token = tokens.mean(0).mean(-1)
        self.assertLess(per_token[0].item(), -0.9)
        self.assertGreater(per_token[2].item(), 0.9)
        same = data[:2].clone()
        same[:, 20:28] = same[:, 12:20]
        out = scaler(same)
        torch.testing.assert_close(out[:, 20:28], out[:, 12:20])
        torch.testing.assert_close(scaled[:, 36], data[:, 36])
        outside = torch.cat((scaled[:, :12], scaled[:, 37:]), dim=-1)
        self.assertLess(outside.mean(0).abs().max().item(), 0.05)
        self.assertLess((outside.std(0) - 1).abs().max().item(), 0.05)
        torch.testing.assert_close(scaler(scaled[:3], inverse=True)[:, 12:37], data[:3, 12:37],
                                   atol=1e-4, rtol=0)

    def test_rollout_batches_checkpoints_and_validation(self):
        scaler, kwargs = self.scaler()
        data = self.data()
        self.assertEqual(scaler(data.view(100, 60, 40), train=True).shape, (100, 60, 40))
        self.assertEqual(scaler.token_count.item(), 1 + 6000 * 3)
        copy = TokenStandardScaler(**kwargs)
        copy.load_state_dict(scaler.state_dict(), strict=True)
        torch.testing.assert_close(copy(data[:5]), scaler(data[:5]))
        self.assertIn("token_variance", scaler.state_dict())
        with self.assertRaises(ValueError):
            TokenStandardScaler(40, token_block=(12, 36), feature_dim=8, device="cpu")


if __name__ == "__main__":
    unittest.main()
