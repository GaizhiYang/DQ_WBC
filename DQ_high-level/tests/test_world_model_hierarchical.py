"""Closed-loop timing, gradient and microstep-data invariants on CPU/CUDA."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model import HierarchicalDynamics, WMTrainerConfig, WorldModelTrainer
from utils.world_model.gradients import freeze_parameters


class LoopSpec:
    """Tiny physical plant with controller memory and an exogenous support."""

    state_dim = 11
    low_feature_dim = 4
    low_action_dim = 20
    context_dim = context_feature_dim = 1
    delta_dim = 2
    low_dt = 0.02

    def __init__(self, device):
        self.low_policy = torch.nn.Linear(4, 12, bias=False).to(device)
        with torch.no_grad():
            self.low_policy.weight.fill_(0.15)
        self.low_policy.requires_grad_(False)
        self.low_policy.eval()
        self.controls, self.integrated = [], []
        self.prepares = self.finishes = 0

    def get(self, state, name):
        assert name == "foot_contacts"
        return state[..., 5:9]

    def low_features(self, state):
        return torch.cat((state[..., :2], state[..., 2:3], state[..., 9:10]), -1)

    def low_action_features(self, state, action):
        return action

    def normalize_context(self, context):
        return context

    def prepare_high(self, state, action):
        self.prepares += 1
        result = state.clone()
        result[..., 2] = state[..., 2] + action[..., 0]
        return result

    def low_control(self, state, context):
        legs = self.low_policy(state[..., :4])
        self.controls.append(legs)
        updated = state.clone()
        updated[..., 3] = state[..., 0]
        updated[..., 4] = (state[..., 4] + 0.04) % 1
        action = torch.cat((legs, state[..., 2:3].expand(*state.shape[:-1], 8)), -1)
        return updated, action

    def integrate_low(self, state, delta, action, context, dt):
        velocity = state[..., 1:2] + dt * action[..., :1] + delta[..., :1]
        position = state[..., :1] + dt * velocity + delta[..., 1:2]
        result = torch.cat((position, velocity, state[..., 2:9],
                            state[..., 9:10] + dt * context, state[..., 10:]), -1)
        self.integrated.append(result)
        return result

    def with_contacts(self, state, contacts):
        return torch.cat((state[..., :5], contacts, state[..., 9:]), -1)

    def finish_high(self, previous, state, action, context):
        self.finishes += 1
        return torch.cat((state[..., :10], previous[..., 10:] + 1), -1)

    def error_vector(self, predicted, target):
        return predicted[..., :2] - target[..., :2]

    def auxiliary_loss(self, predicted, target):
        return (predicted[..., 0] - target[..., 0]).square()


class HierarchicalChecks:
    device = "cpu"

    def make(self):
        torch.manual_seed(17)
        spec = LoopSpec(self.device)
        model = HierarchicalDynamics(spec, hidden=(16, 16), contact_hidden=(8,), device=self.device)
        state = torch.zeros(4, spec.state_dim, device=self.device)
        state[:, 0] = 0.1
        context = torch.full((4, 1), 0.3, device=self.device)
        return spec, model, state, context

    def batch(self, steps=3):
        spec, model, state, context = self.make()
        actions = torch.ones(steps, 4, 20, device=self.device) * 0.2
        targets = [state]
        for action in actions:
            targets.append(spec.integrate_low(targets[-1], torch.zeros(4, 2, device=self.device),
                                              action, context, torch.tensor(.02, device=self.device)))
        states = torch.stack(targets).detach()
        states[1:, :, 5:9] = 1
        batch = SimpleNamespace(states=states, actions=actions,
                                context=context[None].expand(steps, -1, -1).clone(),
                                dt=torch.full((steps, 4, 1), .02, device=self.device),
                                terminated=torch.zeros(steps, 4, dtype=torch.bool, device=self.device))
        spec.integrated.clear()
        return spec, model, batch

    def test_high_action_has_eight_feedback_responses_and_one_goal_update(self):
        spec, model, state, context = self.make()
        action = torch.ones(4, 9, device=self.device, requires_grad=True)
        with freeze_parameters(model):
            result = model(state, action, context)
            result[:, 0].sum().backward()
        self.assertEqual((spec.prepares, spec.finishes, len(spec.controls)), (1, 1, 8))
        torch.testing.assert_close(result[:, 2], torch.ones(4, device=self.device))
        torch.testing.assert_close(result[:, 4], torch.full((4,), .32, device=self.device))
        torch.testing.assert_close(result[:, 9], torch.full((4,), .048, device=self.device))
        torch.testing.assert_close(result[:, 10], torch.ones(4, device=self.device))
        self.assertFalse(torch.equal(spec.controls[0], spec.controls[-1]))
        self.assertGreater(float(action.grad[:, 0].abs().sum()), 1e-4)
        self.assertTrue(all(parameter.grad is None for parameter in spec.low_policy.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(result.device.type, torch.device(self.device).type)

    def test_four_high_decisions_backpropagate_through_all_32_low_responses(self):
        spec, model, initial, context = self.make()
        model.double()
        spec.low_policy.double()
        initial, context = initial.double(), context.double()
        # Exercise a nonzero learned residual as well as the plant's explicit
        # integration; a zero output head cannot test its input Jacobian.
        with torch.no_grad():
            model.dynamics.net[-1].weight.fill_(.005)
        theta = torch.nn.Parameter(initial.new_tensor(.4))

        def rollout(parameter, first_action_offset=None):
            spec.controls.clear()
            spec.integrated.clear()
            spec.prepares = spec.finishes = 0
            state = initial
            actions, observations = [], []
            for high_step in range(4):
                # The same policy parameter sees newly predicted physical
                # observations at every high-level decision.
                observation = state[:, :2]
                observations.append(observation)
                command = parameter * (.5 + observation[:, 0] + .2 * observation[:, 1])
                action = torch.cat((command[:, None], state.new_zeros(len(state), 8)), -1)
                if high_step == 0 and first_action_offset is not None:
                    action = action + first_action_offset
                if action.requires_grad:
                    action.retain_grad()
                actions.append(action)
                state = model(state, action, context, .16)
            return state[:, 0].square().mean(), state, actions, observations

        with freeze_parameters(model, spec.low_policy):
            loss, endpoint, actions, observations = rollout(theta)
            loss.backward()
            self.assertEqual((spec.prepares, spec.finishes, len(spec.controls), len(spec.integrated)),
                             (4, 4, 32, 32))
            torch.testing.assert_close(endpoint[:, 10], endpoint.new_full((len(endpoint),), 4.))
            self.assertTrue(all(not torch.equal(observations[0], value) for value in observations[1:]))
            self.assertTrue(all(action.grad is not None and bool(torch.isfinite(action.grad).all())
                                for action in actions))
            self.assertGreater(float(theta.grad.abs()), 1e-8)
            self.assertGreater(float(actions[0].grad[:, 0].abs().sum()), 1e-8)
            theta_gradient = theta.grad.detach().clone()
            first_gradient = actions[0].grad.detach().clone()

            # Central differences rerun the real hierarchical operator. They
            # do not reproduce its eight-step recurrence in the test, and
            # detect detached intermediate observations/history even when the
            # forward values remain correct.
            epsilon = 1e-5
            direction = torch.zeros_like(actions[0])
            direction[:, 0] = direction.new_tensor([.7, -.2, .4, 1.1])
            with torch.no_grad():
                plus, *_ = rollout(theta.detach() + epsilon)
                minus, *_ = rollout(theta.detach() - epsilon)
                finite_theta = (plus - minus) / (2 * epsilon)
                plus, *_ = rollout(theta.detach(), epsilon * direction)
                minus, *_ = rollout(theta.detach(), -epsilon * direction)
                finite_first = (plus - minus) / (2 * epsilon)
            torch.testing.assert_close(theta_gradient, finite_theta, rtol=3e-5, atol=2e-8)
            torch.testing.assert_close((first_gradient * direction).sum(), finite_first,
                                       rtol=3e-5, atol=2e-8)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in spec.low_policy.parameters()))

    def test_frozen_controller_is_not_in_dynamics_optimizer_or_checkpoint(self):
        spec, model, _, _ = self.make()
        trainer = WorldModelTrainer(model)
        owned = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
        self.assertFalse(owned & {id(parameter) for parameter in spec.low_policy.parameters()})
        self.assertFalse(any("low_policy" in name for name in model.state_dict()))
        model.train()
        self.assertFalse(spec.low_policy.training)

    def test_microstep_fit_never_reads_future_states_or_future_table_context(self):
        spec, model, batch = self.batch()
        trainer = WorldModelTrainer(model)
        loss, metrics = trainer.loss_on(batch)
        baseline = [prediction.detach().clone() for prediction in spec.integrated]
        self.assertEqual(len(baseline), 3)
        self.assertEqual(len(spec.controls), 0)  # fit the recorded actuator inputs
        self.assertIn("contact_bce", metrics)
        self.assertTrue(torch.isfinite(loss))
        spec.integrated.clear()
        batch.states[1:, :, :2] += 12
        batch.context[1:] += 99
        changed_loss, _ = trainer.loss_on(batch)
        for first, second in zip(baseline, spec.integrated):
            torch.testing.assert_close(first, second)
        self.assertGreater(float(changed_loss), float(loss))

    def test_contact_predictions_are_binary_but_logits_receive_supervision(self):
        _, model, batch = self.batch()
        trainer = WorldModelTrainer(model)
        prediction, logits = model.low_step(batch.states[0], batch.actions[0], batch.context[0],
                                             return_logits=True)
        self.assertTrue(bool(((prediction[:, 5:9] == 0) | (prediction[:, 5:9] == 1)).all()))
        loss, _ = trainer.loss_on(batch)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.contact_head.parameters()), 0)
        self.assertEqual(logits.device.type, torch.device(self.device).type)
        self.assertIsNone(model.term_head)

    def test_joint_physical_and_contact_fit_reduces_loss(self):
        _, model, batch = self.batch()
        batch.states[1:, :, 0] += 0.1
        trainer = WorldModelTrainer(model, WMTrainerConfig(learning_rate=0.01))
        initial, _ = trainer.loss_on(batch)
        for _ in range(35):
            trainer.optimizer.zero_grad(set_to_none=True)
            loss, _ = trainer.loss_on(batch)
            loss.backward()
            trainer.optimizer.step()
        final, _ = trainer.loss_on(batch)
        self.assertLess(float(final), float(initial) * .5)

    def test_wrong_clock_or_action_interface_fails(self):
        _, model, state, context = self.make()
        with self.assertRaisesRegex(ValueError, "0.16"):
            model(state, torch.zeros(4, 9, device=self.device), context, .14)
        with self.assertRaisesRegex(ValueError, "actuator"):
            model.low_step(state, torch.zeros(4, 9, device=self.device), context)
        with self.assertRaisesRegex(ValueError, "0.02"):
            model.low_step(state, torch.zeros(4, 20, device=self.device), context, .16)
        with self.assertRaisesRegex(ValueError, "0.16"):
            model(state, torch.zeros(4, 9, device=self.device), context,
                  torch.full((4, 1), .14, device=self.device))
        with self.assertRaisesRegex(ValueError, "0.02"):
            model.low_step(state, torch.zeros(4, 20, device=self.device), context,
                           torch.full((4, 1), float("nan"), device=self.device))


class CPUHierarchicalTests(HierarchicalChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAHierarchicalTests(HierarchicalChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
