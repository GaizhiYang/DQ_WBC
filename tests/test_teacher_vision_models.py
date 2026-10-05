"""CPU tests of the camera teacher against the original, unmodified classes.

Run with: python -m unittest discover -s tests -p 'test_teacher_vision_models.py'
"""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from modules.predictattention import PredictAttentionSelector
from modules.teacher_vision import (
    IMAGE_OBS_DIM, PACKED_OBS_DIM, PRIVILEGED_OBS_DIM,
    TeacherVisionPolicy, TeacherVisionValue, load_legacy_teacher_weights,
)


def original_teacher_classes():
    """Execute actual model classes while excluding Isaac Gym/top-level setup."""
    source = ROOT / "DQ_high-level" / "train_multistate_DQ_teacher.py"
    module = ast.parse(source.read_text())
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name in ("Policy", "Value")]
    namespace = dict(
        torch=torch, nn=nn, np=np, Model=Model, GaussianMixin=GaussianMixin,
        DeterministicMixin=DeterministicMixin,
        PredictAttentionSelector=PredictAttentionSelector,
        cprint=lambda *args, **kwargs: None,
    )
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Policy"], namespace["Value"]


class TeacherVisionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original_policy, cls.original_value = original_teacher_classes()

    def setUp(self):
        torch.manual_seed(7)
        self.privileged = torch.randn(2, PRIVILEGED_OBS_DIM)
        self.images = torch.rand(2, IMAGE_OBS_DIM)
        self.packed = torch.cat([self.privileged, self.images], dim=-1)

    def original(self, cls):
        return cls((PRIVILEGED_OBS_DIM,), (9,), "cpu", 1024, 128, None, None, None)

    def policy(self, vision_mode="images"):
        return TeacherVisionPolicy((PACKED_OBS_DIM,), (9,), vision_mode=vision_mode)

    def test_original_weights_and_zero_adapter_reproduce_teacher_for_any_images(self):
        original = self.original(self.original_policy)
        expected, expected_std, _ = original.compute({"states": self.privileged}, "policy")
        for mode in ("images", "zero", "none"):
            with self.subTest(mode=mode):
                policy = self.policy(mode)
                result = load_legacy_teacher_weights(policy, original.state_dict())
                self.assertEqual(result.unexpected_keys, [])
                for key, old in original.state_dict().items():
                    self.assertEqual(policy.state_dict()[key].shape, old.shape)
                    torch.testing.assert_close(policy.state_dict()[key], old)
                for images in (self.images, 1 - self.images):
                    packed = torch.cat([self.privileged, images], dim=-1)
                    actual, actual_std, _ = policy.compute({"states": packed}, "policy")
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_std, expected_std)
                if mode == "none":
                    self.assertEqual(set(policy.state_dict()), set(original.state_dict()))
                    self.assertFalse(hasattr(policy, "shared_cnn"))
                else:
                    self.assertTrue(torch.count_nonzero(policy.visual_adapter.weight) == 0)

    def test_critic_exact_legacy_compatibility_and_image_independence(self):
        original = self.original(self.original_value)
        value = TeacherVisionValue((PACKED_OBS_DIM,), (9,))
        loaded = load_legacy_teacher_weights(value, original.state_dict())
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(set(value.state_dict()), set(original.state_dict()))
        expected = original.compute({"states": self.privileged}, "value")[0]
        packed = self.packed.clone().requires_grad_()
        actual = value.compute({"states": packed}, "value")[0]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(torch.count_nonzero(packed.grad[:, PRIVILEGED_OBS_DIM:]).item(), 0)
        self.assertGreater(torch.count_nonzero(packed.grad[:, :PRIVILEGED_OBS_DIM]).item(), 0)

    def test_zero_mode_removes_images_but_retains_guidance(self):
        policy = self.policy("zero")
        captured = []
        hook = policy.transformer_arm.register_forward_pre_hook(
            lambda module, args: captured.append(args[1].detach().clone())
        )
        first = policy._visual_features(self.privileged, self.images)
        second = policy._visual_features(self.privileged, torch.randn_like(self.images))
        hook.remove()
        torch.testing.assert_close(first, second)
        guidance = torch.cat([self.privileged[:, 1030:1082], self.privileged[:, -9:]], dim=-1)
        torch.testing.assert_close(captured[0], guidance)
        self.assertGreater(torch.count_nonzero(guidance).item(), 0)

    def test_gradients_reach_adapter_then_visual_backbone(self):
        policy = self.policy()
        policy.compute({"states": self.packed}, "policy")[0].square().sum().backward()
        self.assertGreater(policy.visual_adapter.weight.grad.abs().sum().item(), 0)
        # At the exact warm start, a zero adapter intentionally blocks upstream
        # visual gradients until its first optimizer update.
        self.assertEqual(policy.shared_cnn.encoder[0].weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()
        policy.compute({"states": self.packed}, "policy")[0].square().sum().backward()
        self.assertGreater(policy.shared_cnn.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(policy.transformer_arm.state_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(policy.transformer_base.state_proj.weight.grad.abs().sum().item(), 0)

    def test_gaussian_log_probability_matches_across_train_eval(self):
        policy = self.policy()
        with torch.no_grad():
            policy.visual_adapter.weight.normal_(std=0.01)
        for module in policy.modules():
            if isinstance(module, nn.Dropout):
                self.assertEqual(module.p, 0.0)
            if isinstance(module, nn.MultiheadAttention):
                self.assertEqual(module.dropout, 0.0)
        policy.eval()
        with torch.no_grad():
            actions, rollout_log_prob, _ = policy.act({"states": self.packed}, "policy")
        policy.train()
        _, update_log_prob, _ = policy.act(
            {"states": self.packed, "taken_actions": actions}, "policy"
        )
        torch.testing.assert_close(update_log_prob, rollout_log_prob, atol=1e-5, rtol=1e-5)
        mean, log_std, _ = policy.compute({"states": self.packed}, "policy")
        expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1, keepdim=True)
        torch.testing.assert_close(update_log_prob, expected)
        policy.deterministic = True
        torch.testing.assert_close(policy.act({"states": self.packed}, "policy")[0], mean)

    def test_legacy_loader_rejects_missing_original_keys_and_wrong_shapes(self):
        original = self.original(self.original_policy)
        policy = self.policy()
        missing = dict(original.state_dict())
        del missing["net.0.weight"]
        with self.assertRaisesRegex(ValueError, "net.0.weight"):
            load_legacy_teacher_weights(policy, missing)
        wrong = dict(original.state_dict())
        wrong["net.0.weight"] = torch.zeros(512, 334)
        with self.assertRaisesRegex(ValueError, "shape_mismatch"):
            load_legacy_teacher_weights(policy, wrong)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_legacy_teacher_weights(policy, policy.state_dict())

    def test_augmented_checkpoint_strict_round_trip(self):
        first = self.policy()
        with torch.no_grad():
            first.visual_adapter.weight.normal_(std=0.01)
        second = self.policy()
        second.load_state_dict(first.state_dict(), strict=True)
        torch.testing.assert_close(
            first.compute({"states": self.packed}, "policy")[0],
            second.compute({"states": self.packed}, "policy")[0],
        )

    def test_configuration_and_packed_shape_are_explicit(self):
        for kwargs in ({"no_feature": True}, {"pitch_control": True},
                       {"floating_base": True}, {"use_tanh": True},
                       {"clip_actions": True}, {"vision_mode": "bad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TeacherVisionPolicy((PACKED_OBS_DIM,), (9,), **kwargs)
        with self.assertRaises(ValueError):
            TeacherVisionPolicy((PRIVILEGED_OBS_DIM,), (9,))
        with self.assertRaises(ValueError):
            TeacherVisionPolicy((PACKED_OBS_DIM,), (10,))
        with self.assertRaises(ValueError):
            self.policy("none").compute({"states": self.privileged}, "policy")


if __name__ == "__main__":
    unittest.main()
