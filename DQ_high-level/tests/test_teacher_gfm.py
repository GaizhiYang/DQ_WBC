"""Run the teacher's real EKF-to-attention path without importing Isaac Gym."""

import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import yaml
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from modules.predictattention import PredictAttentionSelector, teacher_gfm_kwargs
from utils import config as config_utils


ROOT = Path(__file__).resolve().parents[1]


def teacher_models():
    source = ROOT / "train_multistate_DQ_teacher.py"
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in ("Policy", "Value")]
    namespace = dict(torch=torch, nn=nn, Model=Model, GaussianMixin=GaussianMixin,
                     DeterministicMixin=DeterministicMixin,
                     PredictAttentionSelector=PredictAttentionSelector,
                     cprint=lambda *args, **kwargs: None)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Policy"], namespace["Value"]


Policy, Value = teacher_models()


class TeacherGFMConfigurationTests(unittest.TestCase):
    def test_fresh_configuration_enables_future_query_and_keeps_tsil_width(self):
        cfg = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertIs(cfg["env"]["posePrediction"]["gfmFuturePoses"], True)
        env = SimpleNamespace(cfg=cfg, num_future_pose_obs=13, num_tsil_obs=2)
        self.assertEqual(teacher_gfm_kwargs(env),
                         {"num_future_pose_obs": 13, "num_tsil_obs": 2})

    def test_missing_or_disabled_saved_flag_keeps_legacy_query(self):
        for settings in ({}, {"enabled": True}, {"gfmFuturePoses": False}):
            with self.subTest(settings=settings):
                env = SimpleNamespace(cfg={"env": {"posePrediction": settings}},
                                      num_future_pose_obs=13, num_tsil_obs=2)
                self.assertEqual(teacher_gfm_kwargs(env),
                                 {"num_future_pose_obs": 0, "num_tsil_obs": 2})
        self.assertEqual(teacher_gfm_kwargs(SimpleNamespace(cfg={"env": {}})),
                         {"num_future_pose_obs": 0, "num_tsil_obs": 0})

    def test_prediction_disabled_uses_no_future_channels_even_with_new_flag(self):
        env = SimpleNamespace(cfg={"env": {"posePrediction": {
            "enabled": False, "gfmFuturePoses": True}}}, num_future_pose_obs=0)
        self.assertEqual(teacher_gfm_kwargs(env),
                         {"num_future_pose_obs": 0, "num_tsil_obs": 0})

    def test_saved_flag_requires_a_boolean(self):
        for flag in ("false", 0, 1, None, []):
            with self.subTest(flag=flag):
                env = SimpleNamespace(cfg={"env": {"posePrediction": {"gfmFuturePoses": flag}}})
                with self.assertRaisesRegex(ValueError, "gfmFuturePoses"):
                    teacher_gfm_kwargs(env)


class StudentTeacherGFMConfigurationTests(unittest.TestCase):
    @staticmethod
    def student_config():
        return {"env": {
            "posePrediction": {"enabled": False, "horizons": [0.3, 0.6],
                               "history": 5, "qScale": 1.0, "rScale": 0.01},
            "noVelocityObs": False, "numEnvs": 3, "cameraMode": "full",
            "asset": {"asset_multi": {"sugar_box": {"dict_idx": 1}}}},
            "reward": {"pick_up": 50.0}, "seed": 12}

    @staticmethod
    def saved_run(directory, settings, filename="DQ_teacher.yaml", metadata=False):
        run = Path(directory)
        checkpoint = run / "checkpoints" / "agent_100.pt"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        path = run / filename
        path.write_text(yaml.safe_dump(settings))
        if metadata:
            # Run metadata may record the config's old absolute source path.
            (run / "teacher_run.json").write_text(json.dumps({
                "config_file": "/old/machine/data/cfg/" + filename}))
        return checkpoint, path

    @staticmethod
    def selector_width(cfg):
        settings = cfg["env"].get("posePrediction", {})
        future_width = (6 * len(settings.get("horizons", [0.3, 0.6])) + 1
                        if settings.get("enabled", False) else 0)
        env = SimpleNamespace(cfg=cfg, num_future_pose_obs=future_width, num_tsil_obs=0)
        return PredictAttentionSelector(
            future_pose_dim=teacher_gfm_kwargs(env)["num_future_pose_obs"]).query_proj.in_features

    def test_custom_saved_config_restores_full_prediction_settings_and_new_query(self):
        prediction = {"enabled": True, "horizons": [0.2, 0.7], "history": 9,
                      "qScale": 2.5, "rScale": 0.04, "gfmFuturePoses": True}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, selected = self.saved_run(directory, {"env": {
                "posePrediction": prediction, "noVelocityObs": True,
                "numEnvs": 999, "cameraMode": "teacher"}}, "custom_teacher.yaml", metadata=True)
            (Path(directory) / "DQ_teacher.yaml").write_text("env: {}\n")
            cfg = self.student_config()
            expected = deepcopy(cfg)
            expected["env"]["posePrediction"] = prediction
            expected["env"]["noVelocityObs"] = True
            self.assertEqual(config_utils.restore_student_teacher_observation_config(cfg, checkpoint),
                             selected.resolve())
            self.assertEqual(cfg, expected)
            self.assertEqual(self.selector_width(cfg), 147)

    def test_default_saved_yaml_keeps_old_future_query_when_flag_is_missing(self):
        prediction = {"enabled": True, "horizons": [0.4, 0.9], "history": 7,
                      "qScale": 3.0, "rScale": 0.02}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, selected = self.saved_run(directory, {"env": {
                "posePrediction": prediction, "noVelocityObs": True}})
            cfg = self.student_config()
            self.assertEqual(config_utils.restore_student_teacher_observation_config(cfg, checkpoint),
                             selected.resolve())
            self.assertEqual(cfg["env"]["posePrediction"], prediction)
            self.assertTrue(cfg["env"]["noVelocityObs"])
            self.assertEqual(self.selector_width(cfg), 134)

    def test_legacy_saved_yaml_without_observation_settings_does_not_enable_new_query(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.saved_run(directory, {"env": {"numEnvs": 999}})
            cfg = self.student_config()
            cfg["env"]["posePrediction"]["gfmFuturePoses"] = True
            cfg["env"]["posePrediction"]["enabled"] = True
            cfg["env"]["noVelocityObs"] = True
            config_utils.restore_student_teacher_observation_config(cfg, checkpoint)
            self.assertEqual(cfg["env"]["posePrediction"], {})
            self.assertFalse(cfg["env"]["noVelocityObs"])
            self.assertEqual(cfg["env"]["numEnvs"], 3)
            self.assertEqual(self.selector_width(cfg), 134)

    def test_missing_adjacent_yaml_keeps_student_config_without_new_default_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoints" / "agent_100.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            cfg = self.student_config()
            before = deepcopy(cfg)
            self.assertIsNone(config_utils.restore_student_teacher_observation_config(cfg, checkpoint))
            self.assertEqual(cfg, before)
            self.assertEqual(self.selector_width(cfg), 134)

    def test_explicit_cli_prediction_overrides_are_applied_after_saved_settings(self):
        prediction = {"enabled": True, "horizons": [0.2, 0.7], "history": 9,
                      "qScale": 2.5, "rScale": 0.04, "gfmFuturePoses": True}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.saved_run(directory, {"env": {"posePrediction": prediction}})
            cfg = self.student_config()
            config_utils.restore_student_teacher_observation_config(cfg, checkpoint)
            args = config_utils.get_params([
                "--future_pose_horizons", "0.1", "0.4", "0.8", "--ekf_history", "3",
                "--ekf_q_scale", "1.5", "--ekf_r_scale", "0.005"])
            settings = config_utils.apply_pose_prediction_config(cfg, args)
            self.assertEqual(settings, {"enabled": True, "horizons": [0.1, 0.4, 0.8],
                                        "history": 3, "qScale": 1.5, "rScale": 0.005,
                                        "gfmFuturePoses": True})
            self.assertEqual(self.selector_width(cfg), 153)

    def test_student_entry_restores_only_during_training_and_before_cli_overrides(self):
        source = ROOT / "train_multi_bc_deter_DQ_stu.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "get_trainer")

        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets)

        start = next(i for i, node in enumerate(function.body) if assigns(node, "cfg"))
        stop = next(i for i, node in enumerate(function.body) if assigns(node, "multi_obj"))
        # Execute the real configuration section; simulator imports and setup stay outside it.
        code = compile(ast.Module(body=function.body[start:stop], type_ignores=[]), str(source), "exec")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.saved_run(directory, {"env": {"posePrediction": {
                "enabled": True, "horizons": [0.2, 0.7], "gfmFuturePoses": True}}})
            for is_eval in (False, True):
                with self.subTest(is_eval=is_eval):
                    args = config_utils.get_params([
                        "--teacher_ckpt_path", str(checkpoint), "--future_pose_horizons", "0.1", "0.5"])
                    args.eval = is_eval
                    calls = []

                    def restore(cfg, path):
                        calls.append(path)
                        return config_utils.restore_student_teacher_observation_config(cfg, path)

                    namespace = dict(args=args, file_path="student.yaml", os=__import__("os"),
                                     load_cfg=lambda path: self.student_config(),
                                     restore_student_teacher_observation_config=restore,
                                     apply_pose_prediction_config=config_utils.apply_pose_prediction_config,
                                     cprint=lambda *args, **kwargs: None)
                    exec(code, namespace)
                    cfg = namespace["cfg"]
                    self.assertEqual(len(calls), 0 if is_eval else 1)
                    self.assertEqual(cfg["env"]["posePrediction"]["horizons"], [0.1, 0.5])
                    self.assertEqual(self.selector_width(cfg), 134 if is_eval else 147)


class TeacherGFMTests(unittest.TestCase):
    device = "cpu"
    feature_width = 16
    body_width = 24

    def fixture(self, future_width=13, temporal_width=0, action_width=9):
        generator = torch.Generator().manual_seed(423)

        def random(width):
            return torch.randn(2, width, generator=generator).to(self.device) * 0.3

        feature, body = random(self.feature_width), random(self.body_width)
        future = random(future_width)
        if future_width:
            future[:, -1] = torch.tensor([1.0, 0.0], device=self.device)
        temporal, positions, rotations, actions = (random(temporal_width), random(90),
                                                   random(90), random(action_width))
        state = torch.cat((feature, body, future, temporal, positions, rotations, actions), dim=-1)
        return SimpleNamespace(
            state=state, future=future, temporal=temporal, actions=actions,
            pose=body[:, :6], feature=feature,
            candidates=torch.cat((positions.view(2, 30, 3), rotations.view(2, 30, 3)), dim=-1),
            future_slice=slice(self.feature_width + self.body_width,
                               self.feature_width + self.body_width + future_width),
            temporal_width=temporal_width, action_width=action_width)

    def model(self, cls, fixture, query_future_width=None, legacy_defaults=False):
        torch.manual_seed(17)
        if query_future_width is None:
            query_future_width = fixture.future.shape[-1]
        kwargs = {} if legacy_defaults else {
            "num_future_pose_obs": query_future_width, "num_tsil_obs": fixture.temporal_width}
        return cls(fixture.state.shape[-1], fixture.action_width, self.device,
                   self.feature_width, 128, None, None, None, **kwargs).to(self.device)

    @staticmethod
    def capture(model, state):
        captured = {}
        handles = [
            model.query_proj.register_forward_pre_hook(
                lambda module, args: captured.update(query=args[0])),
            model.key_proj.register_forward_pre_hook(
                lambda module, args: captured.update(candidates=args[0])),
            model.feature_encoder.register_forward_hook(
                lambda module, args, output: captured.update(features=output)),
            model.output_proj.register_forward_hook(
                lambda module, args, output: captured.update(selected=output)),
            model.net[0].register_forward_pre_hook(
                lambda module, args: captured.update(mlp=args[0])),
        ]
        try:
            output = model.compute({"states": state}, role="policy")[0]
        finally:
            for handle in handles:
                handle.remove()
        return output, captured

    def test_policy_and_value_query_receive_exact_current_future_and_validity_channels(self):
        for cls in (Policy, Value):
            for temporal_width, action_width in ((0, 9), (2, 9), (2, 10)):
                with self.subTest(model=cls.__name__, tsil=temporal_width, actions=action_width):
                    fixture = self.fixture(temporal_width=temporal_width, action_width=action_width)
                    model = self.model(cls, fixture)
                    output, captured = self.capture(model, fixture.state)
                    expected = torch.cat((captured["features"], fixture.pose, fixture.future), dim=-1)
                    torch.testing.assert_close(captured["query"], expected)
                    torch.testing.assert_close(captured["candidates"], fixture.candidates)
                    self.assertEqual(model.query_proj.in_features, 147)
                    self.assertEqual(output.shape, (2, action_width if cls is Policy else 1))
                    # The existing MLP still receives state + actions + feature + selected grasp.
                    expected_mlp_width = fixture.state.shape[-1] - self.feature_width + 128 - 180 + 6
                    self.assertEqual(captured["mlp"].shape, (2, expected_mlp_width))
                    torch.testing.assert_close(captured["mlp"][:, -6:], captured["selected"])

    def test_both_predicted_poses_and_validity_have_gradients_through_grasp_selection(self):
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                fixture = self.fixture(temporal_width=2)
                state = fixture.state.detach().requires_grad_(True)
                model = self.model(cls, fixture)
                _, captured = self.capture(model, state)
                # Differentiate GFM itself, excluding the direct EKF-to-MLP path.
                captured["selected"].square().sum().backward()
                gradients = state.grad[:, fixture.future_slice]
                for name, span in (("first", slice(0, 6)), ("second", slice(6, 12)),
                                   ("validity", slice(12, 13))):
                    with self.subTest(channel=name):
                        self.assertGreater(gradients[:, span].abs().sum().item(), 1e-9)

    def test_changing_either_future_pose_changes_selected_grasp(self):
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                fixture = self.fixture(temporal_width=2)
                model = self.model(cls, fixture)
                _, original = self.capture(model, fixture.state)
                for horizon in (0, 1):
                    state = fixture.state.clone()
                    start = fixture.future_slice.start + 6 * horizon
                    state[:, start:start + 6] += 1.0
                    _, changed = self.capture(model, state)
                    self.assertGreater((changed["selected"] - original["selected"]).abs().max().item(), 1e-7)

    def test_tsil_channels_do_not_enter_query_or_replace_future_pose(self):
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                fixture = self.fixture(temporal_width=2)
                model = self.model(cls, fixture)
                _, original = self.capture(model, fixture.state)
                changed = fixture.state.clone()
                changed[:, fixture.future_slice.stop:fixture.future_slice.stop + 2] += 100.0
                _, result = self.capture(model, changed)
                torch.testing.assert_close(result["query"], original["query"])
                torch.testing.assert_close(result["selected"], original["selected"])

    def test_horizon_count_controls_query_width_without_changing_candidate_shape(self):
        for cls in (Policy, Value):
            for horizons in (1, 3):
                with self.subTest(model=cls.__name__, horizons=horizons):
                    fixture = self.fixture(future_width=6 * horizons + 1, temporal_width=2)
                    model = self.model(cls, fixture)
                    _, captured = self.capture(model, fixture.state)
                    self.assertEqual(captured["query"].shape, (2, 134 + 6 * horizons + 1))
                    torch.testing.assert_close(captured["query"][:, 134:], fixture.future)
                    self.assertEqual(captured["candidates"].shape, (2, 30, 6))

    def test_gfm_future_disabled_preserves_legacy_query_but_keeps_mlp_future_input(self):
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                fixture = self.fixture(temporal_width=2)
                model = self.model(cls, fixture, query_future_width=0)
                _, original = self.capture(model, fixture.state)
                changed = fixture.state.clone()
                changed[:, fixture.future_slice] += 1.0
                _, result = self.capture(model, changed)
                self.assertEqual(model.query_proj.in_features, 134)
                torch.testing.assert_close(result["query"], original["query"])
                torch.testing.assert_close(result["selected"], original["selected"])
                self.assertGreater((result["mlp"] - original["mlp"]).abs().max().item(), 0.5)

    def test_prediction_disabled_and_old_constructor_defaults_keep_134_query(self):
        fixture = self.fixture(future_width=0)
        for cls in (Policy, Value):
            with self.subTest(model=cls.__name__):
                model = self.model(cls, fixture, legacy_defaults=True)
                output, captured = self.capture(model, fixture.state)
                self.assertEqual(captured["query"].shape, (2, 134))
                torch.testing.assert_close(captured["query"][:, 128:], fixture.pose)
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_legacy_and_future_model_checkpoints_round_trip_with_strict_loading(self):
        for future_width in (0, 13):
            for cls in (Policy, Value):
                with self.subTest(future_width=future_width, model=cls.__name__):
                    fixture = self.fixture(future_width=future_width)
                    source = self.model(cls, fixture, legacy_defaults=not future_width)
                    expected, _ = self.capture(source, fixture.state)
                    stream = io.BytesIO()
                    torch.save(source.state_dict(), stream)
                    stream.seek(0)
                    restored = self.model(cls, fixture, legacy_defaults=not future_width)
                    restored.load_state_dict(torch.load(stream, map_location=self.device, weights_only=True),
                                             strict=True)
                    actual, _ = self.capture(restored, fixture.state)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(restored.query_proj.weight.shape, (64, 134 + future_width))

    def test_future_selector_requires_configured_pose_and_validity_width(self):
        selector = PredictAttentionSelector(future_pose_dim=13).to(self.device)
        fixture = self.fixture()
        feature = torch.zeros(2, 128, device=self.device)
        for future in (None, fixture.future[:, :12], fixture.future[:, :7]):
            with self.subTest(width=None if future is None else future.shape[-1]):
                with self.assertRaisesRegex(ValueError, "future|validity"):
                    selector.attention_forward(fixture.candidates, feature, fixture.pose, future)

    def test_legacy_selector_matches_original_attention_equations(self):
        selector = PredictAttentionSelector().to(self.device)
        fixture = self.fixture(future_width=0)
        features = torch.zeros(2, 128, device=self.device)
        query = selector.query_proj(torch.cat((features, fixture.pose), dim=-1)).unsqueeze(1)
        keys = selector.key_proj(fixture.candidates)
        weights = torch.softmax((query @ keys.transpose(-2, -1)) / 8.0, dim=-1)
        expected = selector.output_proj((weights @ selector.value_proj(fixture.candidates)).squeeze(1))
        actual = selector.attention_forward(fixture.candidates, features, fixture.pose)
        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TeacherGFMCudaTests(TeacherGFMTests):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
