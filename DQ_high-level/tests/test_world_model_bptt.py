"""BPTT behavioral invariants on CPU and, when present, a real CUDA GPU.

These tests check future-state gradients, bootstrap semantics, policy drift
rollback and optimizer ownership without depending on Isaac Gym assets.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.gradients import freeze_parameters
from utils.world_model.imagination import (ImaginationConfig, ImaginationTrainer,
                                          diagonal_gaussian_kl, imagination_objective)


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, 0., 0., 0., 0., 0., 0.7, 0., 0.]))
        self.bias = nn.Parameter(torch.zeros(9))
        self.log_std = nn.Parameter(torch.full((9,), -2.0))

    def compute(self, inputs, role):
        return inputs["states"] * self.weight + self.bias, self.log_std, {}


class OpeningActor(ToyActor):
    """Match the teacher's latent Gaussian to normalized-opening mapping."""

    def mean_to_action(self, mean):
        return torch.cat((mean[:, :6], mean[:, 6:7].sigmoid(), mean[:, 7:]), -1)


class ToyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def compute(self, inputs, role):
        return self.weight * inputs["states"], {}


class ToyModel(nn.Module):
    def __init__(self, done=0.):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))
        self.done = done

    def step(self, state, action, context, dt):
        return torch.cat((state[:, :1] + self.gain * action[:, :1], state[:, 1:] + 1), -1)

    def termination_probability(self, next_state, context):
        return torch.full_like(next_state[:, 0], self.done)


class ToyAdapter:
    def __init__(self, reward_constant=0., reward_action=0.):
        self.reward_constant = reward_constant
        self.reward_action = reward_action

    def observation(self, state, context):
        return state[:, :1]

    def reward(self, previous, next_state, action, context):
        return self.reward_constant + self.reward_action * action[:, 0]


class TimeoutAdapter(ToyAdapter):
    def timeouts(self, state, context):
        return state[:, 1] >= 1


class BPTTChecks:
    device = "cpu"

    def make(self, adapter=None, model=None, actor=None, **options):
        actor, critic = (actor or ToyActor()).to(self.device), ToyCritic().to(self.device)
        model = (model or ToyModel()).to(self.device)
        defaults = dict(horizon=3, gamma=1.0, lr=1e-3, strength=1., max_kl=1.)
        defaults.update(options)
        trainer = ImaginationTrainer(model, adapter or ToyAdapter(), actor, critic,
                                     ImaginationConfig(**defaults))
        state = torch.tensor([[1., 0.], [1., 0.]], device=self.device)
        context = torch.zeros(2, 1, device=self.device)
        return trainer, state, context

    def assertTreeEqual(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assertTreeEqual(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assertTreeEqual(left, right)
        else:
            self.assertEqual(first, second)

    def rng_state(self):
        # CPU and CUDA streams must both remain untouched by mean-only actions.
        states = {"cpu": torch.get_rng_state().clone()}
        if torch.device(self.device).type == "cuda":
            states["cuda"] = torch.cuda.get_rng_state(self.device).clone()
        return states

    def test_mean_rollout_is_deterministic_without_action_rng_or_std_dependence(self):
        trainer, state, context = self.make()
        duration = torch.tensor([.14, .16, .14], device=self.device)[:, None, None].expand(3, 2, 1)
        rng_before = self.rng_state()
        with freeze_parameters(trainer.model, trainer.critic):
            first = trainer.imagine(state, context, dt=duration)
            second = trainer.imagine(state, context, dt=duration)
            # At the same mean, both tiny and huge Gaussian standard deviations
            # must produce exactly the same model actions and consequences.
            with torch.no_grad():
                trainer.actor.log_std.copy_(torch.linspace(-20., 2., 9, device=self.device))
            changed_std = trainer.imagine(state, context, dt=duration)
        for name, values in vars(first).items():
            torch.testing.assert_close(values, getattr(second, name), rtol=0, atol=0)
            torch.testing.assert_close(values, getattr(changed_std, name), rtol=0, atol=0)
        self.assertTreeEqual(rng_before, self.rng_state())
        self.assertTrue(trainer.actor.log_std.requires_grad)
        self.assertIsNone(trainer.actor.log_std.grad)

    def test_loaded_legacy_std_adam_moments_never_advance_on_accepted_or_rejected_bptt(self):
        for reject in (False, True):
            with self.subTest(rejected=reject):
                trainer, state, context = self.make()
                # Simulate an old stochastic-BPTT optimizer checkpoint that has
                # already accumulated a nonzero momentum for every std entry.
                legacy = torch.optim.Adam(trainer.actor.parameters(), lr=trainer.cfg.lr)
                legacy.zero_grad(set_to_none=True)
                trainer.actor.log_std.sum().backward()
                legacy.step()
                checkpoint = {"optimizer": deepcopy(legacy.state_dict()),
                              "accepted_updates": 4, "rejected_updates": 2}
                replacement, _, _ = self.make()
                replacement.actor.load_state_dict(trainer.actor.state_dict())
                replacement.load_state_dict(checkpoint)
                trainer = replacement
                self.assertTrue(any(parameter is trainer.actor.log_std
                                    for group in trainer.optimizer.param_groups
                                    for parameter in group["params"]))
                if reject:
                    trainer.optimizer.param_groups[0]["lr"] = 2.
                    trainer.cfg.max_kl = 1e-10
                before_std = trainer.actor.log_std.detach().clone()
                before_state = deepcopy(trainer.optimizer.state[trainer.actor.log_std])
                before_mean = trainer.actor.weight.detach().clone()
                self.assertGreater(float(before_state["exp_avg"].abs().sum()), 0.)
                self.assertGreater(float(before_state["exp_avg_sq"].sum()), 0.)
                # A stale gradient from PPO must also be cleared to None. A zero
                # tensor would let Adam move std using its old momentum.
                trainer.actor.log_std.grad = torch.full_like(trainer.actor.log_std, 19.)
                metrics = trainer.update(state, context, dt=.14)
                self.assertEqual(metrics["accepted"], float(not reject))
                self.assertEqual(metrics["rollback"], float(reject))
                self.assertGreater(metrics["actor_grad_norm"], 0.)
                torch.testing.assert_close(before_std, trainer.actor.log_std, rtol=0, atol=0)
                self.assertTreeEqual(before_state, trainer.optimizer.state[trainer.actor.log_std])
                self.assertIsNone(trainer.actor.log_std.grad)
                self.assertTrue(trainer.actor.log_std.requires_grad)
                if not reject:
                    self.assertFalse(torch.equal(before_mean, trainer.actor.weight))

    def test_tail_value_retains_full_closed_loop_gradient_and_freezes_evaluators(self):
        trainer, state, context = self.make()
        state.requires_grad_(True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            self.assertEqual(rollout.states.device.type, torch.device(self.device).type)
            objective = imagination_objective(rollout, 1.)
            objective.backward()
        # x_H = (1 + theta)^H: this detects detach at a later step and fixed
        # future actions; both errors remove part of the exact derivative.
        self.assertAlmostEqual(float(trainer.actor.weight.grad[0]), 3 * 1.2 ** 2, places=5)
        self.assertAlmostEqual(float(rollout.tail_values.mean()), 1.2 ** 3, places=5)
        self.assertIsNone(state.grad)
        self.assertIsNone(trainer.model.gain.grad)
        self.assertIsNone(trainer.critic.weight.grad)
        self.assertTrue(trainer.model.gain.requires_grad)
        self.assertTrue(trainer.critic.weight.requires_grad)
        # This particular toy model ignores eight coordinates, including the
        # gripper. The independent all-action dynamics test below checks that
        # the rollout keeps the gripper gradient when dynamics depend on it.
        self.assertEqual(float(trainer.actor.weight.grad[6]), 0.)
        self.assertEqual(float(trainer.actor.bias.grad[6]), 0.)
        self.assertIsNone(trainer.actor.log_std.grad)
        expected_gripper = .7 * (1.2 ** torch.arange(3, device=self.device))
        torch.testing.assert_close(rollout.actions[:, :, 6], expected_gripper[:, None].expand(-1, 2))

    def test_all_nine_mean_actions_receive_tail_gradient_and_gripper_matches_finite_difference(self):
        class AllActionModel(ToyModel):
            def step(self, state, action, context, dt):
                return torch.cat((state[:, :1] + self.gain * action.sum(-1, keepdim=True),
                                  state[:, 1:] + 1), -1)

        trainer, state, context = self.make(model=AllActionModel(), actor=OpeningActor())
        # Double precision makes the finite-difference comparison independent
        # of float32 subtraction error on both CPU and CUDA.
        trainer.actor.double()
        trainer.model.double()
        trainer.critic.double()
        state, context = state.double(), context.double()
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            objective = imagination_objective(rollout, 1.)
            objective.backward()
            gradient = trainer.actor.bias.grad.detach().clone()
            self.assertTrue(bool((gradient.abs() > 1e-5).all()))
            self.assertIsNone(trainer.actor.log_std.grad)
            self.assertIsNone(trainer.model.gain.grad)
            self.assertIsNone(trainer.critic.weight.grad)
            # Zero step rewards mean these gradients can only come from the
            # effect of actions on future state and its terminal value.
            self.assertTrue(bool((rollout.rewards == 0).all()))
            self.assertGreater(float(rollout.actions[-1, 0, 6]), float(rollout.actions[0, 0, 6]))
            self.assertTrue(bool(((rollout.actions[:, :, 6] > 0)
                                  & (rollout.actions[:, :, 6] < 1)).all()))
            epsilon = 1e-5
            with torch.no_grad():
                trainer.actor.bias[6] += epsilon
                plus = imagination_objective(trainer.imagine(state, context), 1.)
                trainer.actor.bias[6] -= 2 * epsilon
                minus = imagination_objective(trainer.imagine(state, context), 1.)
                trainer.actor.bias[6] += epsilon
            numerical_gradient = (plus - minus) / (2 * epsilon)
            torch.testing.assert_close(gradient[6], numerical_gradient, rtol=1e-7, atol=1e-8)
        before_mean = trainer.actor.bias.detach().clone()
        before_std = trainer.actor.log_std.detach().clone()
        metrics = trainer.update(state, context)
        self.assertEqual(metrics["accepted"], 1.)
        self.assertTrue(bool((trainer.actor.bias.detach() != before_mean).all()))
        torch.testing.assert_close(before_std, trainer.actor.log_std, rtol=0, atol=0)
        self.assertIsNone(trainer.actor.log_std.grad)

    def test_direct_action_reward_gradient_and_fixed_trust_weight(self):
        trainer, state, context = self.make(adapter=ToyAdapter(reward_action=2.),
                                                 horizon=1, use_tail_value=False)
        weight = torch.tensor([0.25, 0.75], device=self.device, requires_grad=True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            objective = imagination_objective(rollout, 1., weight)
            objective.backward()
        self.assertAlmostEqual(float(trainer.actor.weight.grad[0]), 1., places=6)
        self.assertIsNone(weight.grad)

    def test_terminal_reward_kept_but_natural_termination_drops_tail(self):
        trainer, state, context = self.make(adapter=ToyAdapter(reward_constant=2.),
                                                 model=ToyModel(done=1.), gamma=0.9)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
        self.assertAlmostEqual(float(objective), 2., places=6)
        self.assertTrue(bool((rollout.survival[1:] == 0).all()))

    def test_timeout_bootstraps_at_its_endpoint_and_cuts_future_rewards(self):
        trainer, state, context = self.make(adapter=TimeoutAdapter(reward_constant=2.), gamma=0.9,
                                                 bootstrap_timeouts=True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
            objective.backward()
        self.assertAlmostEqual(float(objective), 2. + 0.9 * 1.2, places=5)
        self.assertAlmostEqual(float(trainer.actor.weight.grad[0]), 0.9, places=5)

    def test_finite_task_timeout_keeps_its_reward_and_has_zero_bootstrap_by_default(self):
        trainer, state, context = self.make(adapter=TimeoutAdapter(reward_constant=2.), gamma=.9)
        self.assertFalse(trainer.cfg.bootstrap_timeouts)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
            objective.backward()
        self.assertAlmostEqual(float(objective), 2., places=6)
        self.assertEqual(float(trainer.actor.weight.grad[0]), 0.)
        self.assertTrue(bool((rollout.survival[1:] == 0).all()))
        self.assertTrue(bool((rollout.timeout_values == 0).all()))

    def test_continuous_gripper_retains_executed_opening_and_gradient_in_action_memory(self):
        class MemoryModel(ToyModel):
            def step(self, state, action, context, dt):
                return torch.cat((state[:, :1]+self.gain*action[:, :1], state[:, 1:2]+1, action), -1)

        adapter = ToyAdapter()
        adapter.clip_actions = .5
        trainer, state, context = self.make(adapter=adapter, model=MemoryModel(),
                                           actor=OpeningActor(), horizon=1)
        state[0, 0] = -2.
        state = torch.cat((state, torch.zeros(2, 9, device=self.device)), -1)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context)
            rollout.states[1, :, 2+6].sum().backward()
        opening = torch.sigmoid(torch.tensor(-1.4, device=self.device))
        expected = torch.stack((opening, opening.new_tensor(.5)))
        torch.testing.assert_close(rollout.actions[0, :, 6], expected)
        torch.testing.assert_close(rollout.states[1, :, 2+6], expected)
        # The unsaturated opening has its sigmoid derivative; only the sample
        # clipped by the actual execution bound loses its local gradient.
        torch.testing.assert_close(trainer.actor.bias.grad[6], opening * (1 - opening))
        torch.testing.assert_close(trainer.actor.weight.grad[6], -2 * opening * (1 - opening))
        self.assertIsNone(trainer.actor.log_std.grad)

    def test_accepted_update_only_changes_actor_and_preserves_scaler_statistics(self):
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        trainer, state, context = self.make()
        state_scaler = RunningStandardScaler(size=1, device=self.device)
        value_scaler = RunningStandardScaler(size=1, device=self.device)
        with torch.no_grad():
            state_scaler(torch.tensor([[-1.], [1.]], device=self.device), train=True)
            value_scaler(torch.tensor([[-2.], [2.]], device=self.device), train=True)
        trainer.state_preprocessor, trainer.value_preprocessor = state_scaler, value_scaler
        before_scalers = [deepcopy(s.state_dict()) for s in (state_scaler, value_scaler)]
        before_actor = deepcopy(trainer.actor.state_dict())
        before_model = deepcopy(trainer.model.state_dict())
        before_critic = deepcopy(trainer.critic.state_dict())
        # Existing gradients from another training phase must not accumulate.
        trainer.model.gain.grad = torch.full_like(trainer.model.gain, 17.)
        trainer.critic.weight.grad = torch.full_like(trainer.critic.weight, 23.)
        metrics = trainer.update(state, context,
                                 dt=torch.tensor([[0.14], [0.16]], device=self.device))
        self.assertEqual(metrics["accepted"], 1.)
        self.assertGreater(metrics["actor_grad_norm"], 0.)
        self.assertAlmostEqual(metrics["horizon_seconds_mean"], 0.45, places=6)
        self.assertFalse(torch.equal(before_actor["weight"], trainer.actor.weight))
        torch.testing.assert_close(before_actor["log_std"], trainer.actor.log_std, rtol=0, atol=0)
        self.assertIsNone(trainer.actor.log_std.grad)
        self.assertTrue(trainer.actor.log_std.requires_grad)
        self.assertNotIn(trainer.actor.log_std, trainer.optimizer.state)
        self.assertTreeEqual(before_model, trainer.model.state_dict())
        self.assertTreeEqual(before_critic, trainer.critic.state_dict())
        torch.testing.assert_close(trainer.model.gain.grad, torch.full_like(trainer.model.gain, 17.))
        torch.testing.assert_close(trainer.critic.weight.grad, torch.full_like(trainer.critic.weight, 23.))
        for before, scaler in zip(before_scalers, (state_scaler, value_scaler)):
            self.assertTreeEqual(before, scaler.state_dict())
        for state_item in trainer.optimizer.state.values():
            # PyTorch Adam's scalar step may intentionally be on CPU unless
            # capturable=True; the moment tensors and all model math are CUDA.
            self.assertEqual(state_item["exp_avg"].device.type, torch.device(self.device).type)

    def test_kl_rejection_restores_actor_and_existing_adam_moments(self):
        trainer, state, context = self.make()
        first = trainer.update(state, context)
        self.assertEqual(first["accepted"], 1.)
        for group in trainer.optimizer.param_groups:
            group["lr"] = 2.
        trainer.cfg.max_kl = 1e-10
        before_actor = deepcopy(trainer.actor.state_dict())
        before_optimizer = deepcopy(trainer.optimizer.state_dict())
        metrics = trainer.update(state, context)
        self.assertEqual(metrics["accepted"], 0.)
        self.assertEqual(metrics["rollback"], 1.)
        self.assertGreater(metrics["kl"], trainer.cfg.max_kl)
        self.assertTreeEqual(before_actor, trainer.actor.state_dict())
        self.assertTreeEqual(before_optimizer, trainer.optimizer.state_dict())
        self.assertEqual(trainer.accepted_updates, 1)
        self.assertEqual(trainer.rejected_updates, 1)
        restored = deepcopy(trainer.state_dict())
        trainer.load_state_dict(restored)
        self.assertTreeEqual(restored, trainer.state_dict())

    def test_full_gaussian_kl_includes_gripper_and_zero_weight_skips_update(self):
        zero = torch.zeros(2, 9, device=self.device)
        changed = zero.clone()
        changed[:, 6] = 0.2
        kl = diagonal_gaussian_kl(zero, zero, changed, zero)
        torch.testing.assert_close(kl, torch.full((2,), 0.02, device=self.device))
        trainer, state, context = self.make()
        before = deepcopy(trainer.actor.state_dict())
        result = trainer.update(state, context, weights=torch.zeros(2, device=self.device))
        self.assertEqual(result["skipped"], 1.)
        self.assertTreeEqual(before, trainer.actor.state_dict())
        self.assertEqual(len(trainer.optimizer.state), 0)

    def test_strength_scales_actual_adam_step_instead_of_cancelling_in_normalization(self):
        full, state, context = self.make(strength=1.)
        weak, _, _ = self.make(strength=.2)
        before = full.actor.weight.detach().clone()
        first = full.update(state, context)
        second = weak.update(state, context)
        self.assertEqual(first["accepted"], 1.)
        self.assertEqual(second["accepted"], 1.)
        full_delta = full.actor.weight.detach()-before
        weak_delta = weak.actor.weight.detach()-before
        torch.testing.assert_close(weak_delta, full_delta*.2, atol=2e-8, rtol=1e-4)


class CPUChecks(BPTTChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAChecks(BPTTChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
