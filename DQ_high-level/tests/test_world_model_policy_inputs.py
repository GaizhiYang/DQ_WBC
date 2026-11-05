"""Fixed normalization and measured-channel noise on CPU and CUDA."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.world_model.policy_inputs import (
    FixedScale, HighLevelObservationNoise, teacher_observation_vectors,
)


class PolicyInputChecks:
    device = "cpu"

    def test_default_layout_matches_teacher_and_keeps_derived_channels_clean(self):
        scale, noise = teacher_observation_vectors(device=self.device)
        self.assertEqual(scale.shape, (1284,))
        self.assertEqual(noise.shape, (1284,))
        torch.testing.assert_close(scale[:1024], torch.ones_like(scale[:1024]))
        for sl in (slice(1027, 1030), slice(1033, 1036), slice(1036, 1055),
                   slice(1079, 1082), slice(1085, 1088), slice(1091, 1094),
                   slice(1185, 1275)):
            torch.testing.assert_close(scale[sl], torch.full_like(scale[sl], math.pi))
        torch.testing.assert_close(noise[1024:1027], torch.full_like(noise[1024:1027], .005))
        torch.testing.assert_close(noise[1027:1030], torch.full_like(noise[1027:1030], .01))
        torch.testing.assert_close(noise[1055:1073], torch.full_like(noise[1055:1073], .005))
        self.assertEqual(float(noise[:1024].abs().sum()), 0.)
        self.assertEqual(float(noise[1073:].abs().sum()), 0.)
        # qd has already been scaled in the observation; don't rescale twice.
        torch.testing.assert_close(scale[1055:1073], torch.ones_like(scale[1055:1073]))

    def test_layout_sizes_follow_optional_velocity_horizons_and_assets(self):
        scale, noise = teacher_observation_vectors(velocity_obs=True, device=self.device)
        self.assertEqual(scale.numel(), 1289)
        torch.testing.assert_close(noise[1082:1087], torch.full_like(noise[1082:1087], .01))
        self.assertEqual(float(noise[1087:].abs().sum()), 0.)
        # 8 feature + 12 object/EE + 7 q/qd + 9 control/goal + 12 candidates
        # + 9 previous-action; absent forecasts have no validity placeholder.
        small, _ = teacher_observation_vectors(feature_dim=8, ndof=4, horizon_count=0,
                                                candidates=2, device=self.device)
        self.assertEqual(small.numel(), 57)
        two, _ = teacher_observation_vectors(feature_dim=8, ndof=4, horizon_count=2,
                                              candidates=2, device=self.device)
        self.assertEqual(two.numel(), 70)

    def test_fixed_scale_inverse_and_input_gradients_without_running_statistics(self):
        factors = torch.tensor([1., math.pi, .25], device=self.device)
        normalizer = FixedScale(size=SimpleNamespace(shape=(3,)), device=self.device, scale=factors)
        original = torch.tensor([[[2., 2 * math.pi, 1.], [100., 100 * math.pi, 25.]]], device=self.device)
        state_before = {name: value.clone() for name, value in normalizer.state_dict().items()}
        result = normalizer(original, train=True)
        torch.testing.assert_close(result, original.new_tensor([[[2., 2., 4.], [100., 100., 100.]]]))
        torch.testing.assert_close(normalizer(result, inverse=True), original)
        for name, value in normalizer.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
        source = original.detach().requires_grad_(True)
        self.assertFalse(normalizer(source).requires_grad)
        output = normalizer(source, train=True, no_grad=False)
        gradient, = torch.autograd.grad(output.sum(), source)
        torch.testing.assert_close(gradient, factors.reciprocal().expand_as(source))
        # Conversion to float64 should not silently reduce input precision.
        self.assertEqual(normalizer(source.double()).dtype, torch.float64)

    def test_noise_bounds_zero_channels_and_labels_unchanged(self):
        scale, half = teacher_observation_vectors(device=self.device)
        noise = HighLevelObservationNoise(len(half), self.device, half, noise_scale=1.5)
        observations = torch.zeros(128, len(half), device=self.device)
        # A view into a shared storage simulates independent policy observations
        # and physical labels still referencing an unmodified captured buffer.
        labels = observations[:, 1024:1073]
        original = observations.clone()
        result = noise(observations)
        torch.testing.assert_close(observations, original, rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.zeros_like(labels), rtol=0, atol=0)
        self.assertTrue(bool((result.abs() <= 1.5 * half).all()))
        torch.testing.assert_close(result[:, half == 0], torch.zeros_like(result[:, half == 0]), rtol=0, atol=0)
        self.assertGreater(float(result[:, half > 0].abs().sum()), 0.)
        standardized = FixedScale(len(scale), self.device, scale)(result)
        self.assertTrue(bool((standardized.abs() <= 1.5 * half / scale + 1e-8).all()))

    def test_noise_disable_and_input_jacobian(self):
        noise = HighLevelObservationNoise(3, self.device, [.005, .01, 0.])
        source = torch.randn(4, 3, device=self.device, requires_grad=True)
        noisy = noise(source)
        gradient, = torch.autograd.grad(noisy.sum(), source)
        torch.testing.assert_close(gradient, torch.ones_like(source), rtol=0, atol=0)
        noise.eval()
        torch.testing.assert_close(noise(source), source, rtol=0, atol=0)
        noise.train()
        disabled = HighLevelObservationNoise(3, self.device, [.005, .01, 0.], noise_scale=0.)
        torch.testing.assert_close(disabled(source), source, rtol=0, atol=0)

    def test_checkpoint_markers_reject_implicit_legacy_conversion(self):
        modules = (FixedScale(3, self.device, [1., 2., 3.]),
                   HighLevelObservationNoise(3, self.device, [.1, .2, 0.]))
        legacy = {"running_mean": torch.zeros(3), "running_variance": torch.ones(3),
                  "current_count": torch.ones(())}
        for module in modules:
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                module.load_state_dict(legacy, strict=False)
            good = {name: value.clone() for name, value in module.state_dict().items()}
            module.load_state_dict(good)
            marker = next(name for name in good if name.endswith("_version"))
            wrong = dict(good)
            wrong[marker] = torch.tensor(999, device=self.device, dtype=torch.int64)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                module.load_state_dict(wrong, strict=False)
            # Failed loads may have copied buffers before PyTorch raises; use a
            # known compatible marker to isolate the feature-width rejection.
            key = "scale" if isinstance(module, FixedScale) else "noise_half_width"
            wrong = dict(good)
            wrong[key] = torch.ones(2, device=self.device)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                module.load_state_dict(wrong)

    def test_invalid_shapes_and_scales_fail_early(self):
        for scale in ([1., 0., 1.], [1., float("nan"), 1.], [1., 2.]):
            with self.assertRaises(ValueError):
                FixedScale(3, self.device, scale)
        with self.assertRaises(ValueError):
            HighLevelObservationNoise(3, self.device, [.1, -.1, 0.])
        with self.assertRaises(ValueError):
            HighLevelObservationNoise(3, self.device, noise_scale=float("inf"))
        with self.assertRaises(ValueError):
            teacher_observation_vectors(ndof=0, device=self.device)
        for module in (FixedScale(3, self.device), HighLevelObservationNoise(3, self.device)):
            with self.assertRaisesRegex(ValueError, "dimension"):
                module(torch.zeros(2, 4, device=self.device))

    def test_real_wrapper_noises_collection_but_preserves_physical_transition_labels(self):
        from test_world_model_entry import FixtureEnvironment
        from utils.wrapper import IsaacGymPreview3Wrapper

        raw = FixtureEnvironment(self.device, enabled=True)
        raw.world_model_observation_noise = HighLevelObservationNoise(1, self.device, [.05])
        calls = []
        raw.world_model_observation_noise.register_forward_hook(lambda *_: calls.append(1))
        wrapper = IsaacGymPreview3Wrapper(raw)
        reset_observation, _ = wrapper.reset()
        self.assertGreater(float(reset_observation.abs().sum()), 0.)
        torch.testing.assert_close(raw.state, torch.zeros_like(raw.state), rtol=0, atol=0)
        collected, _, _, _, info = wrapper.step(torch.zeros(3, 9, device=self.device))
        clean = raw.state.clone()
        torch.testing.assert_close(clean, torch.ones_like(clean), rtol=0, atol=0)
        self.assertGreater(float((collected - clean).abs().sum()), 0.)
        self.assertLessEqual(float((collected - clean).abs().max()), .050001)
        transition = info["world_model"]
        torch.testing.assert_close(transition["state"], torch.zeros(2, 1, device=self.device), rtol=0, atol=0)
        torch.testing.assert_close(transition["next_state"], torch.ones(2, 1, device=self.device), rtol=0, atol=0)
        torch.testing.assert_close(wrapper._obs_dict["obs"], clean, rtol=0, atol=0)
        # PPO stores this collected realization. Repeated epoch preprocessing
        # is deterministic and must never call the observation-noise module.
        saved = collected.detach().clone()
        preprocessor = FixedScale(1, self.device, 2.)
        for epoch in range(5):
            torch.testing.assert_close(preprocessor(saved, train=epoch == 0), saved / 2., rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        wrapper.reset()
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(transition["next_state"], torch.ones(2, 1, device=self.device), rtol=0, atol=0)
        torch.testing.assert_close(collected, saved, rtol=0, atol=0)

    def test_adapter_policy_noise_matches_real_module_and_leaves_clean_observation_unchanged(self):
        from test_world_model_fixed8_ppo import make_fixed8_agent

        spec = make_fixed8_agent(self.device).wm_adapter
        state, context = spec.capture()
        clean = spec.observation(state, context)
        scale, widths = teacher_observation_vectors(
            feature_dim=spec.feature_table.shape[-1], ndof=spec.ndof,
            horizon_count=len(spec.horizons), candidates=spec.grasp_table.shape[1],
            velocity_obs=spec.velocity_obs, device=self.device)
        module = HighLevelObservationNoise(len(widths), self.device, widths)
        spec.env.world_model_observation_noise = module
        state_before, context_before = state.clone(), context.clone()
        torch.manual_seed(192)
        imagined = spec.policy_observation(state, context)
        torch.manual_seed(192)
        expected = module(clean)
        torch.testing.assert_close(imagined, expected, rtol=0, atol=0)
        self.assertGreater(float((imagined - clean).abs().sum()), 0.)
        torch.testing.assert_close(spec.observation(state, context), clean, rtol=0, atol=0)
        torch.testing.assert_close(state, state_before, rtol=0, atol=0)
        torch.testing.assert_close(context, context_before, rtol=0, atol=0)
        differentiable = state.detach().requires_grad_(True)
        prepared = FixedScale(len(scale), self.device, scale)(
            spec.policy_observation(differentiable, context), no_grad=False)
        gradient, = torch.autograd.grad(prepared[:, 1024:1027].sum(), differentiable)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient[:, spec.slices["object"]].abs().sum()), 0.)
        module.eval()
        torch.testing.assert_close(spec.policy_observation(state, context), clean, rtol=0, atol=0)


class CPUPolicyInputTests(PolicyInputChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAPolicyInputTests(PolicyInputChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
