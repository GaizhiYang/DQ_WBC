"""CPU numerical and information-boundary tests for asymmetric M0/M1/M2."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.teacher_vision import TeacherVisionPolicy, TeacherVisionValue
from modules.asymmetric_teacher import (
    AsymmetricTeacherPolicy, AsymmetricTeacherValue, DeploymentActor,
    IMAGE_OBS_DIM, PACKED_OBS_DIM, PRIVILEGED_OBS_DIM, PROPRIO_INDICES,
    KEEP_FUSION_INDICES, initialize_policy_from_teacher,
    initialize_policy_from_asymmetric, initialize_policy_from_v1,
    initialize_value_from_teacher, packed_obs_dim,
)


FEATURES = ((False, False), (True, False), (False, True), (True, True))


class AsymmetricTeacherModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(71)
        self.teacher = TeacherVisionPolicy((PACKED_OBS_DIM,), (9,), device="cpu")
        with torch.no_grad():
            self.teacher.visual_adapter.weight.normal_(std=0.01)
            self.teacher.log_std_parameter.copy_(torch.linspace(-3, -1, 9))
        self.policy = self.make_policy()
        self.states = torch.cat([torch.randn(2, PRIVILEGED_OBS_DIM), torch.rand(2, IMAGE_OBS_DIM)], dim=-1)

    def make_policy(self, m1=False, m2=False):
        policy = AsymmetricTeacherPolicy((packed_obs_dim(m1, m2),), (9,), m1=m1, m2=m2)
        initialize_policy_from_teacher(policy, self.teacher.state_dict())
        return policy

    def packed(self, m1, m2):
        suffix = ([torch.randn(2, 5)] if m1 else []) + ([torch.randn(2, 16)] if m2 else [])
        return torch.cat([self.states] + suffix, dim=-1)

    def projected_teacher_mean(self, teacher):
        privileged = self.states[:, :PRIVILEGED_OBS_DIM]
        p = privileged[:, list(PROPRIO_INDICES)]
        hidden = nn.functional.linear(p, teacher.net[0].weight[:, KEEP_FUSION_INDICES], teacher.net[0].bias)
        if teacher.vision_mode != "none":
            hidden = hidden + teacher.visual_adapter(teacher._visual_features(privileged, self.states[:, PRIVILEGED_OBS_DIM:]))
        for layer in list(teacher.net.children())[1:]:
            hidden = layer(hidden)
        return hidden

    def test_visual_teacher_migration_preserves_all_deployable_weights(self):
        expected = self.projected_teacher_mean(self.teacher)
        actual, std, _ = self.policy.compute({"states": self.states}, "policy")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(std, self.teacher.log_std_parameter, atol=0, rtol=0)
        torch.testing.assert_close(self.policy.actor.proprio_adapter.weight,
                                   self.teacher.net[0].weight[:, KEEP_FUSION_INDICES])
        torch.testing.assert_close(self.policy.actor.visual_adapter.weight, self.teacher.visual_adapter.weight)
        self.assertFalse(hasattr(self.policy, "alpha"))
        self.assertFalse(hasattr(self.policy, "set_alpha"))
        self.assertFalse(hasattr(self.policy, "transition"))

    def test_every_actor_ignores_gt_and_task_memory_even_nan_and_has_zero_gt_gradient(self):
        keep = set(PROPRIO_INDICES)
        removed = [i for i in range(PRIVILEGED_OBS_DIM) if i not in keep]
        for m1, m2 in FEATURES:
            with self.subTest(m1=m1, m2=m2):
                policy, states = self.make_policy(m1, m2), self.packed(m1, m2)
                if m2:
                    nn.init.normal_(policy.actor.belief_adapter.weight, std=0.01)
                expected = policy.compute({"states": states}, "policy")[0]
                changed = states.clone()
                changed[:, removed] = float("nan")
                if m1:
                    changed[:, PACKED_OBS_DIM:PACKED_OBS_DIM + 5] = float("nan")
                actual = policy.compute({"states": changed}, "policy")[0]
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                differentiable = states.clone().requires_grad_()
                policy.compute({"states": differentiable}, "policy")[0].sum().backward()
                self.assertEqual(torch.count_nonzero(differentiable.grad[:, removed]).item(), 0)
                if m1:
                    self.assertEqual(torch.count_nonzero(differentiable.grad[:, PACKED_OBS_DIM:PACKED_OBS_DIM + 5]).item(), 0)
                self.assertGreater(differentiable.grad[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM].abs().sum().item(), 0)
                if m2:
                    self.assertGreater(differentiable.grad[:, -16:].abs().sum().item(), 0)

    def test_visual_backbone_and_guidance_receive_gradients(self):
        self.policy.train()
        self.policy.compute({"states": self.states}, "policy")[0].square().sum().backward()
        self.assertGreater(self.policy.actor.shared_cnn.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.policy.actor.transformer_arm.state_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.policy.actor.transformer_base.state_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.requires_grad for parameter in self.policy.parameters()))

    def test_pure_actor_matches_policy_and_has_no_gt_parameters(self):
        actor = self.policy.to_deploy_actor()
        self.assertIsInstance(actor, DeploymentActor)
        self.assertFalse(actor.training)
        self.assertEqual(sum(p.numel() for p in actor.parameters()) + 9, 5562530)
        forbidden = ("transition", "_alpha", "feature_encoder", "query_proj", "key_proj", "value_proj", "output_proj")
        self.assertFalse(any(part in key for key in self.policy.state_dict() for part in forbidden))
        proprio = self.states[:, list(PROPRIO_INDICES)]
        images = self.states[:, PRIVILEGED_OBS_DIM:]
        expected = self.policy.compute({"states": self.states}, "policy")[0]
        torch.testing.assert_close(actor(images, proprio), expected)
        torch.testing.assert_close(actor(images.reshape(2, 12, 54, 96), proprio), expected)
        self.assertNotEqual(actor.proprio_adapter.weight.data_ptr(), self.policy.actor.proprio_adapter.weight.data_ptr())

    def test_same_parameters_give_unit_ppo_ratio_across_train_eval_modes(self):
        policy, states = self.make_policy(True, True), self.packed(True, True)
        for module in policy.modules():
            if isinstance(module, nn.Dropout):
                self.assertEqual(module.p, 0.0)
            if isinstance(module, nn.MultiheadAttention):
                self.assertEqual(module.dropout, 0.0)
        policy.eval()
        with torch.no_grad():
            actions, old_log_prob, _ = policy.act({"states": states}, "policy")
        policy.train()
        _, new_log_prob, _ = policy.act({"states": states, "taken_actions": actions}, "policy")
        torch.testing.assert_close((new_log_prob - old_log_prob).exp(), torch.ones_like(old_log_prob), atol=1e-5, rtol=1e-5)
        mean, log_std, _ = policy.compute({"states": states}, "policy")
        exact = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1, keepdim=True)
        torch.testing.assert_close(new_log_prob, exact)

    def test_critic_zero_adapters_preserve_teacher_and_ignore_images(self):
        source = TeacherVisionValue((PACKED_OBS_DIM,), (9,))
        expected = source.compute({"states": self.states}, "value")[0]
        for m1, m2 in FEATURES:
            with self.subTest(m1=m1, m2=m2):
                value = AsymmetricTeacherValue((packed_obs_dim(m1, m2),), (9,), m1=m1, m2=m2)
                initialize_value_from_teacher(value, source.state_dict())
                states = self.packed(m1, m2)
                states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM] = float("nan")
                actual = value.compute({"states": states}, "value")[0]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                actual.sum().backward()
                if m1:
                    self.assertIsNone(value.task_adapter.bias)
                    self.assertEqual(torch.count_nonzero(value.task_adapter.weight).item(), 0)
                    self.assertGreater(value.task_adapter.weight.grad.abs().sum().item(), 0)
                if m2:
                    self.assertIsNone(value.belief_adapter.bias)
                    self.assertEqual(torch.count_nonzero(value.belief_adapter.weight).item(), 0)
                    self.assertGreater(value.belief_adapter.weight.grad.abs().sum().item(), 0)

    def test_critic_learns_to_distinguish_task_memory_and_belief(self):
        value = AsymmetricTeacherValue((packed_obs_dim(True, True),), (9,), m1=True, m2=True)
        nn.init.normal_(value.task_adapter.weight, std=0.05)
        nn.init.normal_(value.belief_adapter.weight, std=0.05)
        states = self.packed(True, True).requires_grad_()
        value.compute({"states": states}, "value")[0].sum().backward()
        self.assertGreater(states.grad[:, PACKED_OBS_DIM:PACKED_OBS_DIM + 5].abs().sum().item(), 0)
        self.assertGreater(states.grad[:, -16:].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(states.grad[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM]).item(), 0)

    def test_m2_requires_belief_and_zero_adapter_preserves_m0_initial_mean(self):
        for m1 in (False, True):
            policy = self.make_policy(m1, True)
            states = self.packed(m1, True)
            torch.testing.assert_close(policy.compute({"states": states}, "policy")[0],
                                       self.policy.compute({"states": self.states}, "policy")[0], atol=0, rtol=0)
            images, p = self.states[:, PRIVILEGED_OBS_DIM:], self.states[:, list(PROPRIO_INDICES)]
            with self.assertRaisesRegex(ValueError, "requires belief"):
                policy.actor(images, p)
            with self.assertRaisesRegex(ValueError, "requires belief"):
                policy.actor(images, p, torch.zeros(2, 15))
            with self.assertRaisesRegex(ValueError, "only accepted"):
                self.policy.actor(images, p, states[:, -16:])
            policy.compute({"states": states}, "policy")[0].sum().backward()
            self.assertGreater(policy.actor.belief_adapter.weight.grad.abs().sum().item(), 0)

    def test_legacy_teacher_requires_explicit_opt_in(self):
        legacy = TeacherVisionPolicy((PACKED_OBS_DIM,), (9,), vision_mode="none")
        with self.assertRaisesRegex(ValueError, "missing"):
            initialize_policy_from_teacher(self.policy, legacy.state_dict())
        initialize_policy_from_teacher(self.policy, legacy.state_dict(), allow_legacy=True)
        self.assertEqual(torch.count_nonzero(self.policy.actor.visual_adapter.weight).item(), 0)
        torch.testing.assert_close(self.policy.compute({"states": self.states}, "policy")[0],
                                   self.projected_teacher_mean(legacy), atol=0, rtol=0)

    def test_historical_v1_is_model_only_warm_start_without_retaining_auxiliary(self):
        state = dict(self.policy.state_dict())
        prefixes = ("feature_encoder.", "key_proj.", "value_proj.", "query_proj.", "output_proj.")
        state.update({"transition." + key: value for key, value in self.teacher.state_dict().items() if key.startswith(prefixes)})
        state["transition.projection.weight"] = torch.full((512, 145), float("nan"))
        state["_alpha"] = torch.tensor(0.375, dtype=torch.float64)
        target = self.make_policy(True, True)
        initialize_policy_from_v1(target, state)
        self.assertFalse(any(key.startswith("transition.") or key == "_alpha" for key in target.state_dict()))
        torch.testing.assert_close(target.compute({"states": self.packed(True, True)}, "policy")[0],
                                   self.policy.compute({"states": self.states}, "policy")[0], atol=0, rtol=0)
        del state["transition.projection.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            initialize_policy_from_v1(target, state)

    def test_v2_upgrade_preserves_existing_adapters_and_refuses_downgrade(self):
        upgraded = self.make_policy(True, True)
        nn.init.ones_(upgraded.actor.belief_adapter.weight)
        initialize_policy_from_asymmetric(upgraded, self.policy.state_dict())
        self.assertEqual(torch.count_nonzero(upgraded.actor.belief_adapter.weight).item(), 0)
        nn.init.normal_(upgraded.actor.belief_adapter.weight)
        copied = self.make_policy(True, True)
        initialize_policy_from_asymmetric(copied, upgraded.state_dict())
        torch.testing.assert_close(copied.actor.belief_adapter.weight, upgraded.actor.belief_adapter.weight)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            initialize_policy_from_asymmetric(self.policy, upgraded.state_dict())
        source = AsymmetricTeacherValue((packed_obs_dim(True, False),), (9,), m1=True)
        nn.init.normal_(source.task_adapter.weight)
        target = AsymmetricTeacherValue((packed_obs_dim(True, True),), (9,), m1=True, m2=True)
        initialize_value_from_teacher(target, source.state_dict())
        torch.testing.assert_close(target.task_adapter.weight, source.task_adapter.weight)
        self.assertEqual(torch.count_nonzero(target.belief_adapter.weight).item(), 0)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            initialize_value_from_teacher(source, target.state_dict())

    def test_loader_validates_all_keys_before_mutation_and_flags_validate_shape(self):
        before = self.policy.actor.proprio_adapter.weight.detach().clone()
        invalid = dict(self.teacher.state_dict())
        invalid["feature_encoder.0.weight"] = torch.randn(512, 1025)
        with self.assertRaisesRegex(ValueError, "shape_mismatch"):
            initialize_policy_from_teacher(self.policy, invalid)
        torch.testing.assert_close(before, self.policy.actor.proprio_adapter.weight, atol=0, rtol=0)
        missing_visual = dict(self.teacher.state_dict())
        del missing_visual["visual_adapter.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            initialize_policy_from_teacher(self.policy, missing_visual, allow_legacy=True)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            initialize_policy_from_teacher(self.policy, self.policy.state_dict())
        with self.assertRaisesRegex(ValueError, "boolean"):
            packed_obs_dim("false", False)
        with self.assertRaisesRegex(ValueError, "63489"):
            AsymmetricTeacherPolicy((PACKED_OBS_DIM,), (9,), m1=True)
        with self.assertRaisesRegex(ValueError, "packed states"):
            self.policy.compute({"states": self.packed(True, False)}, "policy")


if __name__ == "__main__":
    unittest.main()
