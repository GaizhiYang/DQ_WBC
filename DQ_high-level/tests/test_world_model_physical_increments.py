"""Physical labels and source loss must agree before model-based control is useful."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.world_model import HierarchicalDynamics, WMTrainerConfig, WorldModelTrainer
from utils.world_model.physical_increments import (
    physical_increment_targets, physical_endpoint_error, integrate_physical_increments,
    physical_increment_metadata)
from utils.world_model.replay import WindowBatch
import test_world_model_manipulation_data as manipulation_data
from test_world_model_microbatch import FixedReplay, Plant


class PhysicalIncrementTests(unittest.TestCase):
    pair = manipulation_data.PhysicalNormalizationTests.physical_pair

    def test_raw_targets_integrate_real_pose_twist_and_joint_endpoint(self):
        spec, state, post, action, context, _ = self.pair()
        increments = physical_increment_targets(spec, state, post)
        rebuilt = integrate_physical_increments(spec, state, increments, action, context, .02)
        error = physical_endpoint_error(spec, rebuilt, post, state)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=3e-6, rtol=0)
        # Raw q increment differs from the old velocity-integrator correction.
        torch.testing.assert_close(increments[:, 24:43], spec.get(post, "dof_pos") - spec.get(state, "dof_pos"))
        torch.testing.assert_close(spec.get(rebuilt, "table"), spec.nominal_table_step(state, context))

    def test_quaternion_sign_and_left_rotation_heading_coordinates(self):
        from utils.wm_env import from_euler, from_rotvec, quat_apply, quat_mul, quat_normalize
        spec, state, _, action, context, _ = self.pair()
        increments = torch.zeros(len(state), 62)
        increments[:, 9:12] = torch.tensor([.17, -.23, .31])
        increments[:, 21:24] = torch.tensor([-.21, .08, .19])
        result = integrate_physical_increments(spec, state, increments, action, context, .02)
        _, heading = spec._heading(state)
        for i, name in enumerate(("robot", "object")):
            expected = quat_normalize(quat_mul(from_rotvec(quat_apply(heading, increments[:, i*12+9:i*12+12])),
                                                spec.get(state, name)[:, 3:7]))
            torch.testing.assert_close(spec.get(result, name)[:, 3:7], expected, atol=3e-6, rtol=1e-5)
        flipped = {name: spec.get(result, name).clone() for name in ("robot", "object")}
        for value in flipped.values():
            value[:, 3:7].neg_()
        torch.testing.assert_close(physical_increment_targets(spec, state, spec.replace(result, **flipped)),
                                   increments, atol=3e-6, rtol=1e-5)

    def test_zero_raw_increment_does_not_extrapolate_velocity(self):
        spec, state, _, action, context, _ = self.pair()
        result = integrate_physical_increments(spec, state, torch.zeros(len(state), 62), action, context, .02)
        torch.testing.assert_close(physical_endpoint_error(spec, result, state, state), torch.zeros(len(state), 62),
                                   atol=3e-6, rtol=0)

    def test_raw_output_statistics_frozen_forward_checkpoint_and_input_gradients(self):
        spec, state, post, action, context, _ = self.pair()
        kwargs = dict(hidden=(16, 16), contact_hidden=(8,), layer_norm=True,
                      running_normalization=True, dynamics_target="physical_increment")
        model = HierarchicalDynamics(spec, **kwargs)
        mask = torch.arange(len(state)) < 6
        model.observe_transition(state, action, post, context, .02, train_mask=mask)
        expected = physical_increment_targets(spec, state, post)[mask]
        torch.testing.assert_close(model.out_norm.mean, expected.mean(0))
        self.assertEqual(float(model.out_norm.count), 6)
        with torch.no_grad():
            model.dynamics.net[-1].weight.normal_(0, .01)
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        state, action = state.requires_grad_(True), action.requires_grad_(True)
        predicted = model.low_step(state, action, context)
        loss = model.reference_error_vector(predicted, post, state).square().sum()
        for gradient in torch.autograd.grad(loss, (state, action)):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)
        for name, value in model.named_buffers():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        restored = HierarchicalDynamics(spec, **kwargs)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.low_step(state, action, context), predicted)
        self.assertEqual(model.coordinate_metadata(), physical_increment_metadata())
        self.assertEqual(len(model.error_floor), 62)
        self.assertFalse(model.coordinate_metadata()["error_floor_is_accuracy_threshold"])

    def test_reject_residual_statistics_for_physical_loss(self):
        model = HierarchicalDynamics(Plant(), hidden=(8,), contact_hidden=(8,))
        with self.assertRaisesRegex(ValueError, "physical_increment"):
            WorldModelTrainer(model, WMTrainerConfig(state_loss_reduction="reference"))
        spec, *_ = self.pair()
        with self.assertRaisesRegex(ValueError, "running normalization"):
            HierarchicalDynamics(spec, hidden=(8,), dynamics_target="physical_increment")

    def test_increment_roundtrip_keeps_finite_nonzero_action_derivatives(self):
        spec, state, _, action, context, _ = self.pair()
        delta = (torch.randn(len(state), 62) * .01).requires_grad_(True)
        result = integrate_physical_increments(spec, state, delta, action, context, .02)
        recovered = physical_increment_targets(spec, state, result)
        weights = torch.randn_like(delta)
        gradient, = torch.autograd.grad((recovered * weights).sum(), delta)
        torch.testing.assert_close(gradient, weights, atol=1e-5, rtol=1e-5)

    def test_physical_model_masked_fit_matches_chunks_and_preserves_real_statistics(self):
        spec, state, _, action, context, residual = self.pair()
        model = HierarchicalDynamics(spec, hidden=(16, 16), contact_hidden=(8,), layer_norm=True,
            running_normalization=True, dynamics_target="physical_increment")
        states = [state]
        for _ in range(3):
            states.append(spec.integrate_low(states[-1], residual, action, context,
                                             state.new_full((len(state), 1), .02)))
        model.observe_transition(states[0], action, states[1], context, .02,
                                 train_mask=torch.ones(len(state), dtype=torch.bool))
        # Exercise hidden-layer derivatives, rather than only the zero head.
        with torch.no_grad():
            model.dynamics.net[-1].weight.normal_(0, .01)
        other = copy.deepcopy(model)
        count = len(state)
        mask = torch.arange(3)[:, None] < (torch.arange(count) % 4)[None]
        states = torch.stack(states)
        actions = action[None].repeat(3, 1, 1)
        contexts = context[None].repeat(3, 1, 1)
        durations = state.new_full((3, count, 1), .02)
        states[1:][~mask] = float("nan")
        for payload in (actions, contexts, durations):
            payload[~mask] = float("nan")
        zeros = state.new_zeros(3, count)
        ids = torch.arange(count)
        batch = WindowBatch(states, actions, contexts, durations, zeros, zeros.bool(), zeros.bool(),
                            zeros.bool(), ids % 3, torch.zeros_like(ids), ids, valid_mask=mask)
        config = dict(window=3, batches_per_update=1, batch_size=count,
                      state_loss_reduction="reference", auxiliary_weight=0, validation_batch_size=2)
        full = WorldModelTrainer(model, WMTrainerConfig(**config))
        chunks = WorldModelTrainer(other, WMTrainerConfig(**config, microbatch_size=5))
        before = {name: value.clone() for name, value in model.named_buffers()}
        a, b = full.update(FixedReplay(batch)), chunks.update(FixedReplay(batch))
        self.assertEqual(a["updates"], 1)
        for name in a:
            self.assertAlmostEqual(a[name], b[name], delta=max(abs(a[name]) * 2e-5, 2e-5), msg=name)
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=3e-4)
            torch.testing.assert_close(left, right, atol=3e-6, rtol=1e-5)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_bf16_only_applies_to_mlp_not_physical_geometry_or_statistics(self):
        spec, state, post, action, context, _ = self.pair()
        model = HierarchicalDynamics(spec, hidden=(16,), contact_hidden=(8,), layer_norm=True,
            running_normalization=True, dynamics_target="physical_increment", amp="bf16")
        model.observe_transition(state, action, post, context, .02,
                                 train_mask=torch.ones(len(state), dtype=torch.bool))
        result = model.low_step(state, action, context)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result).all()))
        for value in list(model.parameters()) + list(model.buffers()):
            self.assertEqual(value.dtype, torch.float32)


class _AnalyticIncrementModel(torch.nn.Module):
    """One scalar increment gives a hand-computable autoregressive reference loss."""
    dynamics_target = "physical_increment"
    term_head = None

    def __init__(self):
        super().__init__()
        self.step_increment = torch.nn.Parameter(torch.tensor(1.))
        self.spec = Plant()

    def low_step(self, state, action, context, dt, return_logits=False):
        output = torch.cat((state[:, :3] + self.step_increment, state[:, 3:]), -1)
        logits = torch.zeros_like(state[:, 3:])
        return (output, logits) if return_logits else output

    def reference_error_vector(self, predicted, target, pre):
        return predicted[:, :3] - target[:, :3]

    def reference_error_scale(self):
        return self.step_increment.new_ones(3)


def _batch(lengths):
    lengths = torch.tensor(lengths)
    width, steps = len(lengths), int(lengths.max())
    scalar = torch.zeros(steps, width)
    ids = torch.arange(width)
    mask = torch.arange(steps)[:, None] < lengths[None]
    states = torch.zeros(steps + 1, width, 7)
    actions = torch.zeros(steps, width, 20)
    context = torch.zeros(steps, width, 2)
    dt = torch.full((steps, width, 1), .02)
    states[1:][~mask] = float("nan")
    actions[~mask] = float("nan")
    context[~mask] = float("nan")
    dt[~mask] = float("nan")
    return WindowBatch(states, actions, context, dt, scalar, scalar.bool(), scalar.bool(), scalar.bool(),
                       ids % 3, torch.zeros_like(ids), ids, valid_mask=mask)


class ReferenceLossTests(unittest.TestCase):
    def trainer(self, model, chunk=0):
        return WorldModelTrainer(model, WMTrainerConfig(window=4, batch_size=5, batches_per_update=1,
            validation_batch_size=2, state_loss_reduction="reference", auxiliary_weight=0,
            contact_weight=0, microbatch_size=chunk, grad_clip=100.))

    def test_time_sum_batch_mean_hand_computed_loss_and_gradient(self):
        model = _AnalyticIncrementModel()
        trainer = self.trainer(model)
        loss, metrics = trainer.loss_on(_batch([2, 1]))
        # Surviving endpoints are 1, 2 for row0 and 1 for row1: (1+4+1)/B.
        self.assertEqual(float(loss), 3.)
        self.assertEqual(float(metrics["state_step_loss"]), 2.)
        self.assertEqual(float(metrics["valid_count"]), 3.)
        loss.backward()
        self.assertEqual(float(model.step_increment.grad), 6.)

    def test_contact_objective_uses_same_time_sum_with_mean_of_four_bits(self):
        trainer = self.trainer(_AnalyticIncrementModel())
        trainer.cfg.contact_weight = .2
        loss, metrics = trainer.loss_on(_batch([2, 1]))
        expected_contact = 1.5 * torch.log(torch.tensor(2.))
        torch.testing.assert_close(metrics["contact_bce"], expected_contact)
        torch.testing.assert_close(loss, 3 + .2 * expected_contact)

    def test_unequal_masked_microbatches_match_full_update_and_adam_moments(self):
        a = _AnalyticIncrementModel()
        b = copy.deepcopy(a)
        full, chunked = self.trainer(a), self.trainer(b, 2)
        replay = FixedReplay(_batch([4, 1, 0, 0, 3]))
        for _ in range(2):
            left, right = full.update(replay), chunked.update(replay)
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.assertAlmostEqual(left[name], right[name], delta=2e-5, msg=name)
            torch.testing.assert_close(a.step_increment, b.step_increment, rtol=0, atol=0)
            torch.testing.assert_close(a.step_increment.grad, b.step_increment.grad)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(full.optimizer.state[a.step_increment][key],
                                           chunked.optimizer.state[b.step_increment][key])

    def test_no_mask_matches_all_valid_mask(self):
        batch = _batch([3, 3])
        trainer = self.trainer(_AnalyticIncrementModel())
        expected, _ = trainer.loss_on(batch)
        batch.valid_mask = None
        actual, _ = trainer.loss_on(batch)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
