"""SuperMimic state-loss reduction and autoregressive gradient contracts (CPU)."""

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.world_model.trainer import WMTrainerConfig, WorldModelTrainer

SOURCE = Path("/home/hehui/SuperMimic-Reproduced/src/supermimic/world_model/world_model.py")


class PhysicalSpec:
    def physical_error_vector(self, predicted, target):
        # Fixed physical units, deliberately distinct from group weighting.
        return (predicted - target) / predicted.new_tensor([.05, .1, .5])

    def error_vector(self, predicted, target):
        return self.physical_error_vector(predicted, target) * predicted.new_tensor([.5, 2., 3.])

    def auxiliary_loss(self, predicted, target):
        return (predicted[..., 0] - target[..., 0]).square()


class Plant(torch.nn.Module):
    term_head = None

    def __init__(self, spec=None):
        super().__init__()
        self.spec = spec or PhysicalSpec()
        self.gain = torch.nn.Parameter(torch.tensor([.8, 1.1, .9], dtype=torch.float64))
        self.inputs = []

    def forward(self, state, action, context, dt):
        self.inputs.append(state.detach().clone())
        return state * self.gain + action

    def normalized_sq_error(self, predicted, target):
        return self.spec.error_vector(predicted, target).square().mean(-1)


def batch():
    return SimpleNamespace(
        states=torch.tensor([[[.1, .2, .3], [-.1, .3, -.2]],
                             [[.5, .1, -.1], [.2, .2, -.1]],
                             [[.2, .4, .1], [0., -.1, .2]]], dtype=torch.float64),
        actions=torch.tensor([[[.02, -.04, .1], [.1, .02, -.02]],
                              [[.03, .1, -.03], [-.02, -.1, .1]]], dtype=torch.float64),
        context=torch.zeros(2, 2, 1, dtype=torch.float64),
        dt=torch.full((2, 2, 1), .02, dtype=torch.float64),
        terminated=torch.zeros(2, 2, dtype=torch.bool))


class FittingLossTests(unittest.TestCase):
    def trainer(self, model, **kwargs):
        options = dict(state_loss_reduction="sum", auxiliary_weight=0.)
        options.update(kwargs)
        return WorldModelTrainer(model, WMTrainerConfig(**options))

    def test_sum_uses_raw_normalized_errors_and_keeps_nse_as_dimension_mean(self):
        model, data = Plant(), batch()
        model.spec.auxiliary_loss = Mock(side_effect=AssertionError("disabled FK auxiliary must not run"))
        first = data.states[0] * model.gain + data.actions[0]
        second = first * model.gain + data.actions[1]
        errors = torch.stack([model.spec.physical_error_vector(first, data.states[1]),
                              model.spec.physical_error_vector(second, data.states[2])])
        expected = errors.square().sum(-1).mean()
        actual, metrics = self.trainer(model).loss_on(data)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics["state_loss"], expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics["nse_1step"], errors[0].square().mean())
        torch.testing.assert_close(metrics["nse_kstep"], errors[1].square().mean())
        self.assertEqual(metrics["auxiliary_loss"].item(), 0.)
        model.spec.auxiliary_loss.assert_not_called()
        # Multiplying legacy weighted MSE by the dimension is NOT this loss.
        legacy = torch.stack([model.normalized_sq_error(first, data.states[1]),
                              model.normalized_sq_error(second, data.states[2])]).mean() * 3
        self.assertFalse(torch.isclose(actual, legacy))

    def test_two_step_gradient_includes_first_prediction_without_teacher_forcing(self):
        model, data = Plant(), batch()
        # Independently differentiate the explicit two-step recurrence.
        gain = model.gain.detach().clone().requires_grad_()
        first = data.states[0] * gain + data.actions[0]
        second = first * gain + data.actions[1]
        scale = gain.new_tensor([.05, .1, .5])
        expected = (((first-data.states[1])/scale).square().sum(-1).mean()
                    + ((second-data.states[2])/scale).square().sum(-1).mean()) / 2
        expected_gradient, = torch.autograd.grad(expected, gain)
        actual, _ = self.trainer(model).loss_on(data)
        actual_gradient, = torch.autograd.grad(actual, model.gain)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-13, atol=1e-13)
        original_inputs = model.inputs[:]
        model.inputs.clear()
        data.states[1].add_(17.)  # A supervised target must never become a rollout input.
        self.trainer(model).loss_on(data)
        for before, after in zip(original_inputs, model.inputs):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    @unittest.skipUnless(SOURCE.is_file(), "local SuperMimic reference checkout unavailable")
    def test_loss_and_gradient_match_unmodified_supermimic_window_loss(self):
        source = ast.parse(SOURCE.read_text())
        cls = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == "WorldModel")
        method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "window_loss")
        method.returns = None
        for argument in method.args.args:
            argument.annotation = None
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(SOURCE), "exec"), namespace)
        model, data = Plant(), batch()

        def source_step(state, action, context, target, done, *_):
            predicted = model(state, action, context, .02)
            squared = model.spec.physical_error_vector(predicted, target).square()
            zeros = torch.zeros_like(done, dtype=predicted.dtype)
            return predicted, squared, zeros, zeros

        reference = SimpleNamespace(cfg=SimpleNamespace(term_loss_weight=0.), _loss_step=source_step)
        refs = [SimpleNamespace(root_pos_w=None, root_quat_w=None)] * 2
        expected, expected_metrics = namespace["window_loss"](
            reference, data.states, data.actions, data.context[0], data.terminated, refs)
        actual, metrics = self.trainer(model).loss_on(data)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        left, = torch.autograd.grad(actual, model.gain)
        right, = torch.autograd.grad(expected, model.gain)
        torch.testing.assert_close(left, right, rtol=1e-13, atol=1e-13)
        for name in ("state_loss", "nse_1step", "nse_kstep"):
            torch.testing.assert_close(metrics[name], expected_metrics[name], rtol=1e-13, atol=1e-13)

    def test_mean_preserves_legacy_error_weights_and_explicit_auxiliary(self):
        model, data = Plant(), batch()
        first = model(data.states[0], data.actions[0], None, None)
        second = model(first, data.actions[1], None, None)
        state_error = torch.stack([model.normalized_sq_error(first, data.states[1]),
                                   model.normalized_sq_error(second, data.states[2])]).mean()
        auxiliary = torch.stack([model.spec.auxiliary_loss(first, data.states[1]),
                                 model.spec.auxiliary_loss(second, data.states[2])]).mean()
        actual, metrics = self.trainer(model, state_loss_reduction="mean", auxiliary_weight=.7).loss_on(data)
        torch.testing.assert_close(actual, state_error + .7 * auxiliary)
        torch.testing.assert_close(metrics["state_loss"], state_error)
        torch.testing.assert_close(metrics["auxiliary_loss"], auxiliary)

    def test_sum_falls_back_to_spec_error_vector_without_new_physical_api(self):
        spec = SimpleNamespace(error_vector=lambda predicted, target: predicted-target)
        model, data = Plant(spec), batch()
        first = model(data.states[0], data.actions[0], None, None)
        second = model(first, data.actions[1], None, None)
        expected = torch.stack([(first-data.states[1]).square().sum(-1),
                                (second-data.states[2]).square().sum(-1)]).mean()
        actual, _ = self.trainer(model).loss_on(data)
        torch.testing.assert_close(actual, expected)

    def test_contact_bce_keeps_its_separate_coefficient(self):
        class ContactPlant(Plant):
            def low_step(self, state, action, context, dt, return_logits=False):
                predicted = self(state, action, context, dt)
                logits = self.gain[:1].expand(len(state), 4)
                return predicted, logits

        model, data = ContactPlant(), batch()
        model.spec.get = lambda state, name: state.new_ones(len(state), 4)
        loss, metrics = self.trainer(model, contact_weight=.1).loss_on(data)
        expected_bce = torch.nn.functional.softplus(-model.gain[0])
        torch.testing.assert_close(metrics["contact_bce"], expected_bce)
        torch.testing.assert_close(loss, metrics["state_loss"] + .1 * expected_bce)

    def test_configuration_rejects_invalid_reduction_or_auxiliary_weight(self):
        for invalid in ({"state_loss_reduction": "bad"}, {"auxiliary_weight": -1.},
                        {"auxiliary_weight": math.inf}, {"auxiliary_weight": math.nan},
                        {"auxiliary_weight": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.trainer(Plant(), **invalid)


if __name__ == "__main__":
    unittest.main()
