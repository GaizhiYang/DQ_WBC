"""Minimal actor information boundary and standalone deployment equivalence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.minimal_asymmetric_teacher import (
    ACTOR_INDICES, DeploymentActor, MinimalAsymmetricTeacherPolicy, select_actor_observation,
)
from skrl.resources.preprocessors.torch import RunningStandardScaler
from utils.minimal_asymmetric_deployment import build_deployment, export_deployment


class MinimalActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(51)
        self.policy = MinimalAsymmetricTeacherPolicy((1102,), (9,), "cpu")
        self.raw = torch.randn(8, 1102)

    def checkpoint(self):
        scaler = RunningStandardScaler(1102, device="cpu")
        scaler.running_mean.copy_(torch.randn(1102, dtype=torch.float64))
        scaler.running_variance.copy_(torch.rand(1102, dtype=torch.float64) * 4)
        # Distinguishes sqrt(var)+epsilon from sqrt(var+epsilon).
        scaler.running_variance[1030] = 0
        scaler.running_mean[1030] = 0
        scaler.current_count.fill_(12801)
        checkpoint = {
            "policy": deepcopy(self.policy.state_dict()),
            "state_preprocessor": deepcopy(scaler.state_dict()),
            "grasp_selection_state": {
                "version": 1,
                "settings": {"teacher_actor": "minimal", "mode": "geometric"},
                "environment_options": {"use_tanh": False, "stop_pick": False, "control_freq": None},
            },
        }
        return checkpoint, scaler

    def test_input_contract_and_architecture(self):
        packed = torch.arange(1102).reshape(1, 1102)
        selected = select_actor_observation(packed)
        self.assertEqual(selected.shape, (1, 67))
        self.assertEqual(selected[0].tolist(), list(ACTOR_INDICES))
        layers = [module for module in self.policy.actor.modules() if isinstance(module, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(67, 512), (512, 256), (256, 128), (128, 9)])
        self.assertFalse(any("feature_encoder" in key or "query_proj" in key for key in self.policy.state_dict()))

    def test_ignored_privileges_including_nan_cannot_change_actor_or_its_gradients(self):
        keep = set(ACTOR_INDICES)
        removed = [index for index in range(1102) if index not in keep]
        expected = self.policy.compute({"states": self.raw}, "policy")[0]
        changed = self.raw.clone()
        changed[:, removed] = float("nan")
        actual = self.policy.compute({"states": changed}, "policy")[0]
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        differentiable = self.raw.clone().requires_grad_()
        self.policy.compute({"states": differentiable}, "policy")[0].sum().backward()
        self.assertEqual(torch.count_nonzero(differentiable.grad[:, removed]).item(), 0)
        self.assertTrue((differentiable.grad[:, list(ACTOR_INDICES)].abs().sum(0) > 0).all())

    def test_stochastic_and_deterministic_distribution_interface(self):
        means = self.policy.compute({"states": self.raw}, "policy")[0]
        actions, log_prob, outputs = self.policy.act({"states": self.raw}, "policy")
        self.assertEqual(actions.shape, (8, 9))
        self.assertEqual(log_prob.shape, (8, 1))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertFalse(torch.equal(actions, means))
        torch.testing.assert_close(outputs["mean_actions"], means)
        self.policy.deterministic = True
        torch.testing.assert_close(self.policy.act({"states": self.raw}, "policy")[0], means, atol=0, rtol=0)

    def test_wrong_protocol_and_tanh_are_rejected(self):
        for kwargs in ({"use_tanh": True}, {"clip_actions": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "unsquashed Gaussian"):
                MinimalAsymmetricTeacherPolicy((1102,), (9,), "cpu", **kwargs)
        with self.assertRaises(ValueError):
            MinimalAsymmetricTeacherPolicy((1276,), (9,), "cpu")
        with self.assertRaises(ValueError):
            MinimalAsymmetricTeacherPolicy((1102,), (10,), "cpu")
        with self.assertRaises(ValueError):
            select_actor_observation(torch.zeros(1, 1276))
        with self.assertRaises(ValueError):
            DeploymentActor()(torch.zeros(1, 1102))

    def test_export_matches_full_training_scaler_and_mean_exactly(self):
        checkpoint, scaler = self.checkpoint()
        raw = self.raw.clone()
        raw[:, 1030] = torch.linspace(-2e-8, 2e-8, len(raw))
        raw[0, 1040], raw[1, 1090] = 1e20, -1e20
        expected = self.policy.compute({"states": scaler(raw)}, "policy")[0]
        deployment, metadata = build_deployment(checkpoint)
        actual = deployment(select_actor_observation(raw))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(deployment.running_mean.shape, (67,))
        self.assertEqual(deployment.current_count.item(), 12801)
        self.assertEqual(metadata["packed_training_indices"], list(ACTOR_INDICES))
        self.assertEqual(metadata["environment_options"]["stop_pick"], False)
        # Normalization buffers are frozen; repeated inference does not update count.
        deployment(select_actor_observation(raw * 3))
        self.assertEqual(deployment.current_count.item(), 12801)

    def test_script_is_standalone_and_embeds_only_actor_statistics(self):
        checkpoint, scaler = self.checkpoint()
        expected = self.policy.compute({"states": scaler(self.raw)}, "policy")[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actor.ts.pt"
            export_deployment(checkpoint, path)
            extra = {"metadata.json": ""}
            loaded = torch.jit.load(str(path), _extra_files=extra)
            torch.testing.assert_close(loaded(select_actor_observation(self.raw)), expected, atol=0, rtol=0)
            metadata = json.loads(extra["metadata.json"])
            self.assertEqual(metadata["input_shape"], ["batch", 67])
            self.assertFalse(metadata["training_use_tanh"])
            self.assertFalse(any("feature_encoder" in key for key in loaded.state_dict()))
            self.assertEqual(loaded.running_mean.numel(), 67)
            with self.assertRaises((RuntimeError, torch.jit.Error)):
                loaded(torch.zeros(1, 1102))
            # Isolated Python cannot import this repo or SKRL. torch.jit.load must still work.
            program = "import torch,sys; m=torch.jit.load(sys.argv[1]); assert m(torch.zeros(2,67)).shape==(2,9); assert 'isaacgym' not in sys.modules; assert 'skrl' not in sys.modules"
            result = subprocess.run([sys.executable, "-I", "-c", program, str(path)],
                                    cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_rejects_unidentified_wrong_or_corrupt_checkpoints(self):
        checkpoint, _ = self.checkpoint()
        cases = []
        wrong = deepcopy(checkpoint)
        del wrong["grasp_selection_state"]
        cases.append(wrong)
        for path, value in (("teacher_actor", "privileged"), ("mode", "karl")):
            wrong = deepcopy(checkpoint)
            wrong["grasp_selection_state"]["settings"][path] = value
            cases.append(wrong)
        wrong = deepcopy(checkpoint)
        wrong["grasp_selection_state"]["environment_options"]["use_tanh"] = True
        cases.append(wrong)
        wrong = deepcopy(checkpoint)
        wrong["state_preprocessor"]["running_variance"][1030] = -1
        cases.append(wrong)
        wrong = deepcopy(checkpoint)
        wrong["state_preprocessor"]["current_count"] = torch.tensor(0.)
        cases.append(wrong)
        wrong = deepcopy(checkpoint)
        wrong["state_preprocessor"]["running_mean"] = torch.zeros(1276)
        cases.append(wrong)
        wrong = deepcopy(checkpoint)
        wrong["policy"]["actor.net.0.weight"] = torch.zeros(512, 206)
        cases.append(wrong)
        for index, wrong in enumerate(cases):
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                build_deployment(wrong)

    def test_cli_exports_without_loading_isaacgym(self):
        checkpoint, _ = self.checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "checkpoint.pt", Path(temporary) / "actor.ts.pt"
            torch.save(checkpoint, source)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "third_party" / "skrl"))
            result = subprocess.run([
                sys.executable, str(ROOT / "DQ_high-level" / "export_minimal_asymmetric_teacher.py"),
                "--checkpoint", str(source), "--output", str(output),
            ], env=env, cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("raw [batch,67]", result.stdout)


if __name__ == "__main__":
    unittest.main()
