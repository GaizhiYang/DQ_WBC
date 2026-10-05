"""Version 2 deployment exports preserve training input and belief contracts."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.asymmetric_teacher import (
    AsymmetricTeacherPolicy, IMAGE_OBS_DIM, PACKED_OBS_DIM, PROPRIO_INDICES, packed_obs_dim,
)
from utils.teacher_vision_preprocessor import PrefixRunningStandardScaler
from utils.asymmetric_teacher_preprocessor import (
    DeployProprioNormalizer, DeploymentPolicy, export_deployment, load_deployment,
)


class AsymmetricTeacherIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(73)
        self.scaler = PrefixRunningStandardScaler(device="cpu", epsilon=2e-4, clip_threshold=3.2)
        with torch.no_grad():
            self.scaler.running_mean.copy_(torch.randn(1276, dtype=torch.float64))
            self.scaler.running_variance.copy_(torch.rand(1276, dtype=torch.float64) + 0.2)
        self.raw = torch.cat([torch.randn(2, 1276) * 5, torch.rand(2, IMAGE_OBS_DIM)], dim=-1)
        self.policy = AsymmetricTeacherPolicy((PACKED_OBS_DIM,), (9,))
        with torch.no_grad():
            self.policy.actor.visual_adapter.weight.normal_(std=0.01)

    def test_selected_statistics_match_training_normalization_exactly(self):
        normalizer = DeployProprioNormalizer.from_teacher_scaler(self.scaler)
        normalized = self.scaler(self.raw)
        actual = normalizer(self.raw[:, list(PROPRIO_INDICES)])
        torch.testing.assert_close(actual, normalized[:, list(PROPRIO_INDICES)], atol=0, rtol=0)
        self.assertEqual(normalizer.running_mean.numel(), 61)
        self.assertEqual(normalizer.running_variance.numel(), 61)
        with torch.no_grad():
            self.scaler.running_variance[1030] = 0
        normalizer = DeployProprioNormalizer.from_teacher_scaler(self.scaler)
        self.raw[:, 1030] = self.scaler.running_mean[1030].float() + 1e-4
        torch.testing.assert_close(normalizer(self.raw[:, list(PROPRIO_INDICES)]),
                                   self.scaler(self.raw)[:, list(PROPRIO_INDICES)], atol=0, rtol=0)

    def test_export_all_features_round_trip_raw_61_belief_and_no_gt_parameters(self):
        from utils.asymmetric_perception import BELIEF_CONTRACT
        for m1, m2 in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(m1=m1, m2=m2):
                dim = packed_obs_dim(m1, m2)
                policy = AsymmetricTeacherPolicy((dim,), (9,), m1=m1, m2=m2)
                nn_weights = [policy.actor.visual_adapter.weight]
                if m2:
                    nn_weights.append(policy.actor.belief_adapter.weight)
                with torch.no_grad():
                    for weight in nn_weights:
                        weight.normal_(std=0.01)
                scaler = PrefixRunningStandardScaler(size=dim, device="cpu", epsilon=2e-4, clip_threshold=3.2)
                scaler.load_state_dict(self.scaler.state_dict())
                suffix = ([torch.randn(2, 5)] if m1 else []) + ([torch.randn(2, 16)] if m2 else [])
                raw = torch.cat([self.raw] + suffix, dim=-1)
                normalized = scaler(raw)
                torch.testing.assert_close(normalized[:, 1276:], raw[:, 1276:], atol=0, rtol=0)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "nested" / "actor.pt"
                    payload = export_deployment(policy, scaler, path)
                    deployed = load_deployment(path)
                self.assertIsInstance(deployed, DeploymentPolicy)
                self.assertFalse(deployed.training)
                self.assertEqual(payload["format_version"], 2)
                self.assertEqual(payload["feature_flags"], {"m1": m1, "m2": m2})
                self.assertEqual(payload["metadata"]["belief_contract"], BELIEF_CONTRACT if m2 else None)
                self.assertEqual(payload["proprio_normalizer"]["mean"].shape, (61,))
                self.assertEqual(payload["proprio_normalizer"]["variance"].shape, (61,))
                forbidden = ("transition", "_alpha", "feature_encoder", "query_proj", "key_proj", "value_proj", "output_proj", "task_adapter")
                self.assertFalse(any(part in key for key in deployed.state_dict() for part in forbidden))
                images, proprio = raw[:, 1276:PACKED_OBS_DIM], raw[:, list(PROPRIO_INDICES)]
                belief = raw[:, -16:] if m2 else None
                expected = policy.compute({"states": normalized}, "policy")[0]
                torch.testing.assert_close(deployed(images, proprio, belief), expected, atol=0, rtol=0)
                flat = torch.cat([images, proprio] + ([belief] if m2 else []), dim=-1)
                torch.testing.assert_close(deployed(flat), expected, atol=0, rtol=0)
                torch.testing.assert_close(deployed.log_std, policy.log_std_parameter)
                with torch.no_grad():
                    policy.actor.proprio_adapter.weight.add_(1)
                torch.testing.assert_close(deployed(images, proprio, belief), expected, atol=0, rtol=0)

    def test_export_loader_rejects_old_versions_training_payload_and_gt_keys(self):
        with self.assertRaisesRegex(ValueError, "deployment-only"):
            load_deployment({"policy": self.policy.state_dict()})
        payload = export_deployment(self.policy, self.scaler)
        old = copy.deepcopy(payload)
        old["format_version"] = 1
        with self.assertRaisesRegex(ValueError, "version"):
            load_deployment(old)
        payload["actor_state_dict"]["transition.projection.weight"] = torch.zeros(512, 145)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            load_deployment(payload)

    def test_m2_belief_is_required_and_export_contract_is_validated(self):
        policy = AsymmetricTeacherPolicy((packed_obs_dim(True, True),), (9,), m1=True, m2=True)
        payload = export_deployment(policy, self.scaler)
        deployed = load_deployment(payload)
        images, proprio = self.raw[:, 1276:], self.raw[:, list(PROPRIO_INDICES)]
        with self.assertRaisesRegex(ValueError, "requires belief"):
            deployed(images, proprio)
        with self.assertRaisesRegex(ValueError, "62285"):
            deployed(torch.cat([images, proprio], dim=-1))
        with self.assertRaisesRegex(ValueError, "already contains"):
            deployed(torch.cat([images, proprio, torch.zeros(2, 16)], dim=-1), belief=torch.zeros(2, 16))
        invalid = copy.deepcopy(payload)
        invalid["metadata"]["belief_contract"]["dim"] = 15
        with self.assertRaisesRegex(ValueError, "belief contract"):
            load_deployment(invalid)
        invalid = copy.deepcopy(payload)
        invalid["feature_flags"]["m2"] = 1
        with self.assertRaisesRegex(ValueError, "feature flags"):
            load_deployment(invalid)
        invalid = copy.deepcopy(payload)
        invalid["feature_flags"]["m2"] = False
        invalid["metadata"]["belief_contract"] = None
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            load_deployment(invalid)

    def test_normalizer_and_input_validation(self):
        for mean, variance in ((torch.zeros(1276), torch.ones(1276)),
                               (torch.zeros(61), -torch.ones(61)),
                               (torch.full((61,), float("nan")), torch.ones(61))):
            with self.subTest(shape=mean.shape), self.assertRaises(ValueError):
                DeployProprioNormalizer(mean, variance)
        for kwargs in ({"epsilon": 0}, {"clip_threshold": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DeployProprioNormalizer(torch.zeros(61), torch.ones(61), **kwargs)
        deployed = load_deployment(export_deployment(self.policy, self.scaler))
        with self.assertRaisesRegex(ValueError, "62269"):
            deployed(self.raw)
        with self.assertRaisesRegex(ValueError, "61"):
            deployed(self.raw[:, 1276:], torch.zeros(2, 62))
        with self.assertRaisesRegex(ValueError, "only accepted"):
            deployed(self.raw[:, 1276:], self.raw[:, list(PROPRIO_INDICES)], torch.zeros(2, 16))


if __name__ == "__main__":
    unittest.main()
