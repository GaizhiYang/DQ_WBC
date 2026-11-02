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

    def make(self, adapter=None, model=None, **options):
        actor, critic = ToyActor().to(self.device), ToyCritic().to(self.device)
        model = (model or ToyModel()).to(self.device)
        defaults = dict(horizon=3, gamma=1.0, lr=1e-3, strength=1., max_kl=1.)
        defaults.update(options)
        trainer = ImaginationTrainer(model, adapter or ToyAdapter(), actor, critic,
                                     ImaginationConfig(**defaults))
        state = torch.tensor([[1., 0.], [1., 0.]], device=self.device)
        context = torch.zeros(2, 1, device=self.device)
        noise = torch.zeros(trainer.cfg.horizon, 2, 9, device=self.device)
        return trainer, state, context, noise

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

    def test_tail_value_retains_full_closed_loop_gradient_and_freezes_evaluators(self):
        trainer, state, context, noise = self.make()
        state.requires_grad_(True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, noise=noise)
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
        # Only the direct gripper head is asserted here. Real shared encoders
        # can still change its output indirectly, hence the all-action KL.
        self.assertEqual(float(trainer.actor.weight.grad[6]), 0.)
        self.assertEqual(float(trainer.actor.bias.grad[6]), 0.)
        self.assertEqual(float(trainer.actor.log_std.grad[6]), 0.)
        expected_gripper = .7 * (1.2 ** torch.arange(3, device=self.device))
        torch.testing.assert_close(rollout.actions[:, :, 6], expected_gripper[:, None].expand(-1, 2))

    def test_direct_action_reward_gradient_and_fixed_trust_weight(self):
        trainer, state, context, noise = self.make(adapter=ToyAdapter(reward_action=2.),
                                                 horizon=1, use_tail_value=False)
        weight = torch.tensor([0.25, 0.75], device=self.device, requires_grad=True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, noise=noise)
            objective = imagination_objective(rollout, 1., weight)
            objective.backward()
        self.assertAlmostEqual(float(trainer.actor.weight.grad[0]), 1., places=6)
        self.assertIsNone(weight.grad)

    def test_terminal_reward_kept_but_natural_termination_drops_tail(self):
        trainer, state, context, noise = self.make(adapter=ToyAdapter(reward_constant=2.),
                                                 model=ToyModel(done=1.), gamma=0.9)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, noise=noise)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
        self.assertAlmostEqual(float(objective), 2., places=6)
        self.assertTrue(bool((rollout.survival[1:] == 0).all()))

    def test_timeout_bootstraps_at_its_endpoint_and_cuts_future_rewards(self):
        trainer, state, context, noise = self.make(adapter=TimeoutAdapter(reward_constant=2.), gamma=0.9,
                                                 bootstrap_timeouts=True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, noise=noise)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
            objective.backward()
        self.assertAlmostEqual(float(objective), 2. + 0.9 * 1.2, places=5)
        self.assertAlmostEqual(float(trainer.actor.weight.grad[0]), 0.9, places=5)

    def test_finite_task_timeout_keeps_its_reward_and_has_zero_bootstrap_by_default(self):
        trainer, state, context, noise = self.make(adapter=TimeoutAdapter(reward_constant=2.), gamma=.9)
        self.assertFalse(trainer.cfg.bootstrap_timeouts)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, noise=noise)
            objective = imagination_objective(rollout, trainer.cfg.gamma)
            objective.backward()
        self.assertAlmostEqual(float(objective), 2., places=6)
        self.assertEqual(float(trainer.actor.weight.grad[0]), 0.)
        self.assertTrue(bool((rollout.survival[1:] == 0).all()))
        self.assertTrue(bool((rollout.timeout_values == 0).all()))

    def test_fixed_gripper_keeps_raw_clipped_values_in_predicted_action_memory(self):
        class MemoryModel(ToyModel):
            def step(self, state, action, context, dt):
                return torch.cat((state[:, :1]+self.gain*action[:, :1], state[:, 1:2]+1, action), -1)

        adapter = ToyAdapter()
        adapter.clip_actions = .5
        trainer, state, context, noise = self.make(adapter=adapter, model=MemoryModel(), horizon=1)
        state = torch.cat((state, torch.zeros(2, 9, device=self.device)), -1)
        commands = torch.tensor([-.23, .81], device=self.device, requires_grad=True)
        with freeze_parameters(trainer.model, trainer.critic):
            rollout = trainer.imagine(state, context, gripper_modes=commands, noise=noise)
            imagination_objective(rollout, 1.).backward()
        expected = torch.tensor([-.23, .5], device=self.device)
        torch.testing.assert_close(rollout.actions[0, :, 6], expected)
        torch.testing.assert_close(rollout.states[1, :, 2+6], expected)
        self.assertIsNone(commands.grad)
        self.assertEqual(float(trainer.actor.weight.grad[6]), 0.)
        # Reference commands are also sampled and outer-clipped, without an
        # artificial sign conversion before storing the previous action.
        reference = trainer.imagine(state, context, noise=noise)
        torch.testing.assert_close(reference.actions[0, :, 6], torch.full((2,), .5, device=self.device))

    def test_accepted_update_only_changes_actor_and_preserves_scaler_statistics(self):
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        trainer, state, context, noise = self.make()
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
        metrics = trainer.update(state, context, noise=noise,
                                 dt=torch.tensor([[0.14], [0.16]], device=self.device))
        self.assertEqual(metrics["accepted"], 1.)
        self.assertGreater(metrics["actor_grad_norm"], 0.)
        self.assertAlmostEqual(metrics["horizon_seconds_mean"], 0.45, places=6)
        self.assertFalse(torch.equal(before_actor["weight"], trainer.actor.weight))
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
        trainer, state, context, noise = self.make()
        first = trainer.update(state, context, noise=noise)
        self.assertEqual(first["accepted"], 1.)
        for group in trainer.optimizer.param_groups:
            group["lr"] = 2.
        trainer.cfg.max_kl = 1e-10
        before_actor = deepcopy(trainer.actor.state_dict())
        before_optimizer = deepcopy(trainer.optimizer.state_dict())
        metrics = trainer.update(state, context, noise=noise)
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
        trainer, state, context, noise = self.make()
        before = deepcopy(trainer.actor.state_dict())
        result = trainer.update(state, context, noise=noise, weights=torch.zeros(2, device=self.device))
        self.assertEqual(result["skipped"], 1.)
        self.assertTreeEqual(before, trainer.actor.state_dict())
        self.assertEqual(len(trainer.optimizer.state), 0)

    def test_strength_scales_actual_adam_step_instead_of_cancelling_in_normalization(self):
        full, state, context, noise = self.make(strength=1.)
        weak, _, _, _ = self.make(strength=.2)
        before = full.actor.weight.detach().clone()
        first = full.update(state, context, noise=noise)
        second = weak.update(state, context, noise=noise)
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
