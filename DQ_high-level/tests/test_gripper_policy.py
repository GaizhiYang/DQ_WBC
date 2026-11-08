"""Executed opening probabilities, gradient ownership and checkpoint semantics."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skrl.models.torch import Model
from utils.gripper_policy import ContinuousGripperGaussianMixin


class TinyOpeningPolicy(ContinuousGripperGaussianMixin, Model):
    def __init__(self, device="cpu", **options):
        Model.__init__(self, observation_space=2, action_space=9, device=device)
        ContinuousGripperGaussianMixin.__init__(self, **options)
        self.mean = torch.nn.Parameter(torch.zeros(9, device=device))
        self.log_std = torch.nn.Parameter(torch.full((9,), -.7, device=device))

    def compute(self, inputs, role=""):
        return self.mean.expand(len(inputs["states"]), -1), self.log_std, {}


class OpeningPolicyChecks:
    device = "cpu"

    def make(self, batch=128, **options):
        return TinyOpeningPolicy(self.device, **options), torch.zeros(batch, 2, device=self.device)

    def test_sampled_actions_are_bounded_stochastic_and_replay_ratio_is_exactly_one(self):
        actor, states = self.make()
        with torch.no_grad():
            first, old_logp, outputs = actor.act({"states": states}, role="policy")
            second, _, _ = actor.act({"states": states}, role="policy")
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(bool(((first[:, 6] > 0) & (first[:, 6] < 1)).all()))
        self.assertTrue(bool(((outputs["mean_actions"][:, 6] == .5)).all()))
        self.assertEqual(first.device.type, torch.device(self.device).type)
        _, recomputed, _ = actor.act({"states": states, "taken_actions": first}, role="policy")
        torch.testing.assert_close(recomputed, old_logp, rtol=0, atol=0)
        torch.testing.assert_close((recomputed - old_logp).exp(), torch.ones_like(old_logp), rtol=0, atol=0)

    def test_log_probability_includes_only_gripper_inverse_and_jacobian(self):
        actor, states = self.make(batch=4, reduction="none")
        actor = actor.double()
        states = states.double()
        latent = torch.linspace(-2., 2., 36, device=self.device, dtype=torch.float64).reshape(4, 9)
        taken = actor.mean_to_action(latent)
        _, logp, _ = actor.act({"states": states, "taken_actions": taken}, role="policy")
        reference = Normal(actor.mean, actor.log_std.exp()).log_prob(latent)
        reference[:, 6] -= torch.log(taken[:, 6]) + torch.log1p(-taken[:, 6])
        torch.testing.assert_close(logp, reference, rtol=1e-12, atol=1e-12)
        summed, _ = self.make(batch=4)
        summed.load_state_dict(actor.state_dict())
        _, reduced, _ = summed.act({"states": states.float(), "taken_actions": taken.float()}, role="policy")
        torch.testing.assert_close(reduced, reference.float().sum(-1, keepdim=True))

    def test_deterministic_mean_action_is_differentiable_and_independent_of_std(self):
        actor, states = self.make(batch=2, deterministic=True)
        with torch.no_grad():
            actor.mean.copy_(torch.linspace(-1., 1., 9, device=self.device))
        rng_before = torch.get_rng_state().clone()
        cuda_before = (torch.cuda.get_rng_state(self.device).clone()
                       if torch.device(self.device).type == "cuda" else None)
        first, _, outputs = actor.act({"states": states})
        with torch.no_grad():
            actor.log_std.fill_(2.)
        second, _, _ = actor.act({"states": states})
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, outputs["mean_actions"], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        if cuda_before is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(self.device), cuda_before, rtol=0, atol=0)
        actor.mean_to_action(actor.mean).sum().backward()
        expected = torch.ones_like(actor.mean)
        opening = actor.mean[6].sigmoid().detach()
        expected[6] = opening * (1 - opening)
        torch.testing.assert_close(actor.mean.grad, expected)
        self.assertIsNone(actor.log_std.grad)

    def test_ppo_likelihood_gradient_can_update_gripper_std(self):
        actor, states = self.make(batch=256)
        with torch.no_grad():
            taken, old_logp, _ = actor.act({"states": states}, role="policy")
        # A centered, nonconstant advantage driven by opening excursions makes
        # the gripper's variance useful; this is the actual PPO likelihood path.
        advantage = (taken[:, 6:7] - .5).square()
        advantage -= advantage.mean()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        before = actor.log_std.detach().clone()
        _, new_logp, _ = actor.act({"states": states, "taken_actions": taken}, role="policy")
        ratio = (new_logp - old_logp).exp()
        objective = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        self.assertGreater(float(actor.log_std.grad[6].abs()), 0.)
        self.assertTrue(bool(torch.isfinite(actor.log_std.grad).all()))
        optimizer.step()
        self.assertNotEqual(float(before[6]), float(actor.log_std[6]))

    def test_opt_in_keeps_sampling_rng_and_unsaturated_density_compatible(self):
        actor, states = self.make(batch=32)
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(self.device)
                    if torch.device(self.device).type == "cuda" else None)
        with torch.no_grad():
            actions, logp, default_outputs = actor.act({"states": states})
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, self.device)
        with torch.no_grad():
            opted_actions, opted_logp, outputs = actor.act({"states": states, "latent_ppo": True})
        self.assertNotIn("latent_actions", default_outputs)
        torch.testing.assert_close(opted_actions, actions, rtol=0, atol=0)
        torch.testing.assert_close(opted_logp, logp, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(actor.mean_to_action(outputs["latent_actions"]), actions, rtol=0, atol=0)
        jacobian = outputs["latent_log_abs_det_jacobian"]
        torch.testing.assert_close(outputs["latent_log_prob"] - jacobian, logp, rtol=2e-6, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "requires taken_latent_actions"):
            actor.act({"states": states, "taken_actions": actions, "latent_ppo": True})

    def test_opt_in_public_density_stays_finite_for_extreme_sampled_latents(self):
        actor, states = self.make(batch=8)
        for mean in (-1e20, -1000., 1000., 1e20):
            with self.subTest(mean=mean):
                with torch.no_grad():
                    actor.mean[6] = mean
                    actions, logp, outputs = actor.act({"states": states, "latent_ppo": True})
                self.assertTrue(bool(torch.isfinite(actions).all()))
                self.assertTrue(bool(torch.isfinite(logp).all()))
                expected = outputs["latent_log_prob"] - outputs["latent_log_abs_det_jacobian"]
                torch.testing.assert_close(logp, expected)
                _, replay_logp, replay = actor.act({"states": states, "taken_actions": actions,
                    "taken_latent_actions": outputs["latent_actions"], "latent_ppo": True})
                torch.testing.assert_close(replay_logp, logp, rtol=0, atol=0)
                torch.testing.assert_close(replay["latent_log_prob"], outputs["latent_log_prob"], rtol=0, atol=0)
                mean_score, std_score = torch.autograd.grad(replay_logp.sum(), (actor.mean, actor.log_std))
                self.assertTrue(bool(torch.isfinite(mean_score).all()))
                self.assertTrue(bool(torch.isfinite(std_score).all()))
                # The default baseline keeps its old inverse-opening formula;
                # the opt-in path must not evaluate or repair that overflow.
                _, default_logp, default = actor.act({"states": states, "taken_actions": actions})
                self.assertNotIn("latent_log_prob", default)
                if abs(mean) == 1e20:
                    self.assertTrue(bool(torch.isneginf(default_logp).all()))

    def test_saturated_openings_retain_distinct_latents_exact_ratio_and_correct_score(self):
        actor, states = self.make(batch=32)
        for mean in (-1000., -100., 100., 1000.):
            with self.subTest(mean=mean):
                with torch.no_grad():
                    actor.mean[6] = mean
                    taken, _, old = actor.act({"states": states, "latent_ppo": True})
                latent = old["latent_actions"].detach()
                self.assertEqual(len(taken[:, 6].unique()), 1)
                self.assertGreater(len(latent[:, 6].unique()), 1)
                _, _, replay = actor.act({"states": states, "taken_actions": taken,
                    "taken_latent_actions": latent, "latent_ppo": True})
                torch.testing.assert_close(replay["latent_log_prob"], old["latent_log_prob"], rtol=0, atol=0)
                torch.testing.assert_close((replay["latent_log_prob"] - old["latent_log_prob"]).exp(),
                    torch.ones_like(old["latent_log_prob"]), rtol=0, atol=0)
                mean_score, std_score = torch.autograd.grad(replay["latent_log_prob"].sum(),
                                                            (actor.mean, actor.log_std))
                delta = latent - actor.mean.detach()
                variance = actor.log_std.detach().exp().square()
                torch.testing.assert_close(mean_score, (delta / variance).sum(0))
                torch.testing.assert_close(std_score, (delta.square() / variance - 1).sum(0))
                self.assertTrue(bool(torch.isfinite(replay["latent_log_abs_det_jacobian"]).all()))
                # Perturb the distribution while retaining the SAME sampled z:
                # its old/new sigmoid Jacobian cancels before float arithmetic.
                with torch.no_grad():
                    actor.mean[6] += .125
                    actor.log_std[6] += .1
                _, _, updated = actor.act({"states": states, "taken_actions": taken,
                    "taken_latent_actions": latent, "latent_ppo": True})
                expected = Normal(actor.mean, actor.log_std.exp()).log_prob(latent).sum(-1, keepdim=True)
                torch.testing.assert_close(updated["latent_log_prob"], expected, rtol=0, atol=0)

    def test_extreme_means_keep_finite_density_and_exact_replayed_likelihood(self):
        actor, states = self.make(batch=8, deterministic=True)
        for mean in (-100., -20., 20., 100.):
            with self.subTest(latent_gripper_mean=mean):
                with torch.no_grad():
                    actor.mean[6] = mean
                    actor.log_std[6] = -20.
                    actions, old_logp, _ = actor.act({"states": states}, role="policy")
                _, logp, _ = actor.act({"states": states, "taken_actions": actions}, role="policy")
                self.assertTrue(bool(torch.isfinite(actions).all() & torch.isfinite(logp).all()))
                torch.testing.assert_close(logp, old_logp, rtol=0, atol=0)
                actor.zero_grad(set_to_none=True)
                logp.mean().backward()
                self.assertTrue(bool(torch.isfinite(actor.mean.grad).all()))
                self.assertTrue(bool(torch.isfinite(actor.log_std.grad).all()))

    def test_entropy_includes_sigmoid_jacobian_with_finite_parameter_gradients(self):
        actor, states = self.make(batch=3)
        actor.act({"states": states}, role="policy")
        entropy = actor.get_entropy(role="policy")
        latent = actor.distribution(role="policy").entropy()
        self.assertEqual(entropy.shape, (3, 9))
        torch.testing.assert_close(entropy[:, :6], latent[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(entropy[:, 7:], latent[:, 7:], rtol=0, atol=0)
        self.assertTrue(bool((entropy[:, 6] <= latent[:, 6] - 1.3862943).all()))
        # Independent midpoint integration in Normal quantile space validates
        # the quadrature against the continuous transformed entropy identity.
        probability = (torch.arange(100000, device=self.device, dtype=torch.float64) + .5) / 100000
        normal = Normal(torch.zeros_like(probability), torch.ones_like(probability))
        z = normal.icdf(probability) * actor.log_std[6].double().exp()
        jacobian = -torch.nn.functional.softplus(z) - torch.nn.functional.softplus(-z)
        expected = latent[0, 6].double() + jacobian.mean()
        torch.testing.assert_close(entropy[0, 6].double(), expected, rtol=1e-5, atol=1e-5)
        actor.zero_grad(set_to_none=True)
        entropy.sum().backward()
        self.assertTrue(bool(torch.isfinite(actor.mean.grad).all()))
        self.assertTrue(bool(torch.isfinite(actor.log_std.grad).all()))

    def test_checkpoint_marker_rejects_legacy_and_wrong_semantics_even_non_strict(self):
        actor, _ = self.make()
        copy, _ = self.make()
        current = deepcopy(actor.state_dict())
        copy.load_state_dict(current)
        self.assertIn("_gripper_action_semantics_version", current)
        self.assertNotIn("_gripper_entropy_nodes", current)
        for wrong in (None, torch.tensor(2)):
            damaged = deepcopy(current)
            if wrong is None:
                del damaged["_gripper_action_semantics_version"]
            else:
                damaged["_gripper_action_semantics_version"] = wrong
            for strict in (True, False):
                with self.assertRaisesRegex(RuntimeError, "Continuous gripper action semantics"):
                    copy.load_state_dict(damaged, strict=strict)

    def test_unsupported_extra_transforms_or_clipping_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "tanh"):
            self.make(transform_func=torch.distributions.transforms.TanhTransform())
        with self.assertRaisesRegex(ValueError, "clip_actions=False"):
            self.make(clip_actions=True)


class CPUOpeningPolicyTests(OpeningPolicyChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAOpeningPolicyTests(OpeningPolicyChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
