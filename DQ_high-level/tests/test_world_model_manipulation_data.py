"""Physical inversion, terminal-prefix fitting and real-only running statistics."""

import copy
from dataclasses import fields
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.world_model import HierarchicalDynamics, TransitionReplay, WMTrainerConfig, WorldModelTrainer
from utils.world_model.normalization import RunningNorm
from utils.world_model.residual_targets import residual_targets
from test_world_model_microbatch import MicrobatchChecks, FixedReplay, Plant


class ReplayPrefixTests(unittest.TestCase):
    def test_terminal_invalid_and_external_reset_have_distinct_prefixes(self):
        replay = TransitionReplay(4, 8, 3, 1, 1, "cpu", holdout_fraction=.25,
                                  holdout_mode="environment", allow_partial_windows=True)
        train_counts = []
        for step in range(4):
            state = torch.full((4, 3), float(step))
            post = state + 1
            valid = torch.ones(4, dtype=torch.bool)
            done = torch.zeros(4, dtype=torch.bool)
            episodes = torch.zeros(4, dtype=torch.long)
            if step == 1:
                done[0], valid[1], post[1] = True, False, float("nan")
            if step >= 2:
                episodes[[0, 2]] = 1
            mask = replay.add(state, torch.full((4, 1), float(step)), post,
                              torch.zeros(4, 1), .02, torch.zeros(4), done,
                              torch.zeros_like(done), torch.zeros(4, dtype=torch.long),
                              episode_ids=episodes, valid=valid)
            train_counts.append(int(mask.sum()))
            self.assertFalse(bool(mask[3]))
        self.assertEqual(train_counts, [3, 2, 3, 3])
        self.assertEqual(int(replay.valid_train_transitions), 11)
        self.assertFalse(replay.can_sample(5))  # no invented future rows
        with patch.object(replay, "_sample_indices", return_value=(torch.zeros(3, dtype=torch.long), torch.arange(3))):
            batch = replay.sample_windows(3, 4)
        torch.testing.assert_close(batch.valid_mask,
            torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool))
        self.assertEqual(float(batch.states[2, 0, 0]), 2.)  # terminal POST retained
        self.assertTrue(bool(replay.sample_starts(100, split="validation").env_ids.eq(3).all()))
        self.assertTrue(bool(replay.sample_starts(100, split="train").env_ids.lt(3).all()))

    def test_new_training_count_survives_ring_overwrite_and_episode_changes(self):
        replay = TransitionReplay(20, 3, 3, 1, 1, "cpu", holdout_fraction=.05,
                                  holdout_mode="environment", allow_partial_windows=True)
        for step in range(11):
            state = torch.zeros(20, 3)
            replay.add(state, state[:, :1], state, state[:, :1], .02, state[:, 0],
                       torch.ones(20, dtype=torch.bool), torch.zeros(20, dtype=torch.bool),
                       torch.zeros(20, dtype=torch.long))
        self.assertEqual(int(replay.valid_train_transitions), 11 * 19)
        self.assertEqual(replay.size, 3 * 20)
        self.assertEqual(replay.holdout_envs, 1)
        self.assertTrue(bool(replay.holdout[:, -1].all()))
        self.assertFalse(bool(replay.holdout[:, :-1].any()))
        self.assertTrue(bool(replay.sample_windows(10, 3).valid_mask[0].all()))
        self.assertFalse(bool(replay.sample_windows(10, 3).valid_mask[1:].any()))


class MaskedFittingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(151)
        self.fixture = MicrobatchChecks()

    def batch(self):
        batch = self.fixture.batch(count=13, steps=5)
        lengths = torch.tensor([0, 1, 2, 3, 4, 5, 1, 4, 2, 0, 3, 5, 1])
        batch.valid_mask = torch.arange(5)[:, None] < lengths[None]
        # Invalid tails can contain arbitrary broken simulator data. These
        # must never enter network, quaternion, context or BCE operations.
        batch.states[0, lengths == 0] = float("nan")
        batch.states[1:][~batch.valid_mask] = float("nan")
        for name in ("actions", "context", "dt"):
            getattr(batch, name)[~batch.valid_mask] = float("nan")
        return batch, lengths

    def trainer(self, model, microbatch=0):
        return WorldModelTrainer(model, WMTrainerConfig(window=5, batch_size=13,
            batches_per_update=1, validation_batch_size=3, microbatch_size=microbatch,
            contact_weight=.3, auxiliary_weight=.7, state_loss_reduction="sum"))

    def test_masked_loss_and_gradients_equal_independent_unpadded_trajectories(self):
        model = self.fixture.model()
        trainer = self.trainer(model)
        batch, lengths = self.batch()
        actual, _ = trainer.loss_on(batch)
        expected = actual.new_zeros(())
        for row, length in enumerate(lengths.tolist()):
            if not length:
                continue
            values = {}
            for field in fields(batch):
                value = getattr(batch, field.name)
                if field.name == "valid_mask":
                    values[field.name] = None
                elif field.name == "states":
                    values[field.name] = value[:length+1, row:row+1]
                elif field.name in ("family", "episode_ids", "env_ids"):
                    values[field.name] = value[row:row+1]
                else:
                    values[field.name] = value[:length, row:row+1]
            part, _ = trainer.loss_on(type(batch)(**values))
            expected = expected + part * length / lengths.sum()
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        got = torch.autograd.grad(actual, tuple(model.parameters()))
        wanted = torch.autograd.grad(expected, tuple(model.parameters()))
        for left, right in zip(got, wanted):
            self.assertTrue(bool(torch.isfinite(left).all()))
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-5)

    def test_unequal_prefix_microbatches_match_parameters_gradients_and_metrics(self):
        model = self.fixture.model()
        full, chunked = self.trainer(model), self.trainer(copy.deepcopy(model), 4)
        replay = FixedReplay(self.batch()[0])
        for _ in range(2):
            expected, actual = full.update(replay), chunked.update(replay)
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assertAlmostEqual(expected[name], actual[name], delta=3e-6, msg=name)
            for a, b in zip(full.model.parameters(), chunked.model.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)

    def test_empty_prefixes_skip_optimizer_and_active_nonfinite_rejects(self):
        trainer = self.trainer(self.fixture.model(), 4)
        batch, _ = self.batch()
        batch.valid_mask.zero_()
        result = trainer.update(FixedReplay(batch), batches=2)
        self.assertEqual(result["skipped_no_valid_steps"], 2)
        self.assertFalse(trainer.optimizer.state)
        batch, _ = self.batch()
        batch.states[1, 1, 0] = float("nan")  # this POST is explicitly valid
        result = trainer.update(FixedReplay(batch))
        self.assertEqual(result["skipped_nonfinite"], 1)
        self.assertEqual(result["updates"], 0)
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))

    def test_full_mask_preserves_legacy_loss(self):
        trainer = self.trainer(self.fixture.model())
        batch = self.fixture.batch(count=13, steps=5)
        expected, _ = trainer.loss_on(batch)
        batch.valid_mask = torch.ones(5, 13, dtype=torch.bool)
        actual, _ = trainer.loss_on(batch)
        torch.testing.assert_close(actual, expected)

    def test_explicit_update_budget_reports_progress_and_validates_once(self):
        trainer = self.trainer(self.fixture.model())
        replay = FixedReplay(self.batch()[0])
        progress = []
        with patch.object(trainer, "validate", wraps=trainer.validate) as validate:
            result = trainer.update(replay, batches=3, progress_interval=2,
                                    progress_callback=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress, [(2, 3), (3, 3)])
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(trainer.cfg.batches_per_update, 1)
        self.assertEqual(result["updates"], 3)


class PhysicalNormalizationTests(unittest.TestCase):
    def physical_pair(self):
        from utils.wm_env import from_euler
        from test_world_model_fixed8_ppo import make_fixed8_agent
        torch.manual_seed(191)
        spec = make_fixed8_agent("cpu").wm_adapter
        state, context = spec.capture()
        robot = spec.get(state, "robot").clone()
        robot[:, 3:7] = from_euler(torch.randn(len(state), 3) * .4)
        robot[:, 7:] = torch.randn(len(state), 6) * .3
        state = spec.replace(state, robot=robot,
                             dof_vel=torch.randn(len(state), 19) * .2)
        action = torch.cat((torch.zeros(len(state), 12), spec.get(state, "held_ee_target"),
                            spec.get(state, "held_gripper_target")), -1)
        delta = torch.randn(len(state), 62) * .2
        post = spec.integrate_low(state, delta, action, context, state.new_full((len(state), 1), .02))
        return spec, state, post, action, context, delta

    def test_inverse_residual_roundtrip_includes_heading_com_and_rotation_sign(self):
        spec, state, post, action, context, delta = self.physical_pair()
        got = residual_targets(spec, state, post, .02)
        torch.testing.assert_close(got, delta, atol=2e-5, rtol=2e-4)
        rebuilt = spec.integrate_low(state, got, action, context, state.new_full((len(state), 1), .02))
        torch.testing.assert_close(spec.physical_error_vector(rebuilt, post), torch.zeros(len(state), 62),
                                   atol=2e-5, rtol=0)
        flipped = {}
        for name in ("robot", "object"):
            body = spec.get(post, name).clone()
            body[:, 3:7].neg_()
            flipped[name] = body
        torch.testing.assert_close(residual_targets(spec, state, spec.replace(post, **flipped), .02), got)

    def test_real_train_mask_statistics_forward_frozen_gradients_and_checkpoint(self):
        spec, state, post, action, context, delta = self.physical_pair()
        model = HierarchicalDynamics(spec, hidden=(16, 16), contact_hidden=(8,),
                                      running_normalization=True, layer_norm=True)
        mask = torch.arange(len(state)) < 6
        invalid = state.clone()
        invalid[~mask] = float("nan")
        model.observe_transition(invalid, action, post, context, .02, train_mask=mask)
        self.assertEqual(float(model.in_norm.count), 6)
        self.assertEqual(float(model.out_norm.count), 6)
        torch.testing.assert_close(model.out_norm.mean, delta[mask].mean(0), atol=1e-5, rtol=2e-4)
        before = {name: x.clone() for name, x in model.named_buffers()}
        with torch.no_grad():
            model.dynamics.net[-1].weight.normal_(0, .01)
        differentiable_state = state.detach().requires_grad_(True)
        differentiable_action = action.detach().requires_grad_(True)
        result = model.low_step(differentiable_state, differentiable_action, context)
        gradients = torch.autograd.grad(spec.physical_error_vector(result, post).square().sum(),
                                        (differentiable_state, differentiable_action))
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)
        model.eval()
        model.low_step(state, action, context)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        restored = HierarchicalDynamics(spec, hidden=(16, 16), contact_hidden=(8,),
                                         running_normalization=True, layer_norm=True)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.low_step(state, action, context), model.low_step(state, action, context))
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in model.dynamics.modules()), 2)
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in model.contact_head.modules()), 0)

    def test_running_moments_match_direct_population_statistics(self):
        torch.manual_seed(5)
        values = torch.randn(113, 7) * 3 + 2
        norm = RunningNorm(7)
        norm.update(values[:17])
        norm.update(values[17:])
        torch.testing.assert_close(norm.mean, values.mean(0))
        torch.testing.assert_close(norm.var, values.var(0, unbiased=False))
        count = norm.count.clone()
        x = values[:3].requires_grad_(True)
        output = norm.denormalize(norm(x))
        torch.testing.assert_close(output, x)
        torch.testing.assert_close(torch.autograd.grad(output.sum(), x)[0], torch.ones_like(x))
        torch.testing.assert_close(count, norm.count)

    def test_training_and_heldout_validation_do_not_reobserve_real_statistics(self):
        spec, state, _, action, context, delta = self.physical_pair()
        model = HierarchicalDynamics(spec, hidden=(16,), contact_hidden=(8,),
                                     running_normalization=True)
        replay = TransitionReplay(len(state), 8, state.shape[-1], 20, context.shape[-1], "cpu",
                                  holdout_mode="environment", holdout_fraction=.25,
                                  allow_partial_windows=True)
        scalar = torch.zeros(len(state))
        for _ in range(4):
            post = spec.integrate_low(state, delta, action, context, state.new_full((len(state), 1), .02))
            mask = replay.add(state, action, post, context, .02, scalar,
                              scalar.bool(), scalar.bool(), scalar.long())
            model.observe_transition(state, action, post, context, .02, train_mask=mask)
            state = post
        self.assertEqual(float(model.in_norm.count), 4 * 9)
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        trainer = WorldModelTrainer(model, WMTrainerConfig(window=3, batches_per_update=1,
            batch_size=4, validation_batch_size=4, state_loss_reduction="sum", auxiliary_weight=0))
        metrics = trainer.update(replay)
        self.assertEqual(metrics["validation_available"], 1)
        self.assertEqual(metrics["updates"], 1)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

    def test_bf16_is_local_to_mlp_and_returns_fp32_for_integrator(self):
        model = HierarchicalDynamics(Plant(), hidden=(16,), contact_hidden=(8,), amp="bf16")
        state, action, context = torch.zeros(4, 7), torch.ones(4, 20), torch.zeros(4, 2)
        dtypes = []
        hook = model.dynamics.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
        result = model.low_step(state, action, context)
        hook.remove()
        self.assertEqual(dtypes, [torch.bfloat16])
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))


class ProbeFactoryTests(unittest.TestCase):
    def test_probe_metadata_matches_numeric_contract_and_legacy_defaults(self):
        from probe_teacher_world_model import _verify_probe_metadata

        class Adapter:
            def metadata(self):
                return {"schema": "fixture", "task_contract": "legacy", "task_aux_alpha": 0.,
                        "task_objective": None, "raw_action_bound_weight": 0.,
                        "raw_gripper_latent_bound": 3., "reward_contract": "legacy",
                        "critic_memory": False, "small_commands_zero": True,
                        "control": {"small_commands": True}}

        spec = Adapter()
        # A pre-revision candidate lacks only explicit legacy defaults.
        old = {"schema": "fixture", "control": {"small_commands": True}}
        _verify_probe_metadata(spec, dict(config={}, metadata=old))
        config = dict(recipe="manipulation", layer_norm=True, running_normalization=True, amp="bf16")
        metadata = dict(spec.metadata(), model_execution=dict(config))
        _verify_probe_metadata(spec, dict(config=config, metadata=metadata))
        for changed in (dict(config, amp="none"), dict(config, running_normalization=False)):
            with self.assertRaises(ValueError):
                _verify_probe_metadata(spec, dict(config=changed, metadata=metadata))
        bad = dict(metadata, task_aux_alpha=.1)
        with self.assertRaises(ValueError):
            _verify_probe_metadata(spec, dict(config=config, metadata=bad))

    def test_probe_native_effects_exclude_opposing_imagined_auxiliary(self):
        from utils.world_model.probes import predicted_task_reward

        class Adapter:
            def task_reward(self, previous, future, action, context):
                return future[:, 0]

            def reward(self, previous, future, action, context):
                return -10 * future[:, 0]  # deliberately opposing auxiliary

        spec = Adapter()
        state, action, context = torch.zeros(3, 7), torch.zeros(3, 9), torch.zeros(3, 2)
        post = state.clone()
        post[:, 0] = torch.tensor([1., 2., 3.])
        torch.testing.assert_close(predicted_task_reward(spec, state, post, action, context), post[:, 0])

        class LegacyAdapter:
            reward = Adapter.reward

        torch.testing.assert_close(predicted_task_reward(LegacyAdapter(), state, post, action, context),
                                   -10 * post[:, 0])

    def test_fixed8_candidate_restores_layernorm_running_statistics_and_precision(self):
        from probe_teacher_world_model import _load_probe_model
        for amp in ("none", "bf16"):
            spec = Plant()
            model = HierarchicalDynamics(spec, hidden=(16, 16), contact_hidden=(8,),
                                        layer_norm=True, running_normalization=True, amp=amp)
            model.in_norm.update(torch.randn(11, model.in_norm.mean.numel()))
            model.out_norm.update(torch.randn(11, model.out_norm.mean.numel()) * .1)
            with torch.no_grad():
                model.dynamics.net[-1].weight.normal_(0, .01)
            config = dict(fixed8=True, hidden=[16, 16], contact_hidden=[8],
                          layer_norm=True, running_normalization=True, amp=amp)
            bundle = dict(config=config, model=model.state_dict())
            restored = _load_probe_model(spec, bundle, "cpu")
            self.assertEqual(restored.amp, amp)
            self.assertFalse(restored.training)
            self.assertEqual(float(restored.out_norm.count), 11)
            state, action, context = torch.zeros(3, 7), torch.ones(3, 20), torch.zeros(3, 2)
            torch.testing.assert_close(restored.low_step(state, action, context),
                                       model.low_step(state, action, context))
            mismatched = dict(bundle, config=dict(config, running_normalization=False))
            with self.assertRaises(RuntimeError):
                _load_probe_model(spec, mismatched, "cpu")

    def test_legacy_probe_snapshot_needs_no_new_config_keys(self):
        from probe_teacher_world_model import _load_probe_model
        spec = Plant()
        model = HierarchicalDynamics(spec, hidden=(8,), contact_hidden=(8,))
        bundle = dict(config=dict(fixed8=True, hidden=[8], contact_hidden=[8]), model=model.state_dict())
        restored = _load_probe_model(spec, bundle, "cpu")
        self.assertIsNone(restored.in_norm)
        self.assertEqual(restored.amp, "none")
        self.assertEqual(set(restored.state_dict()), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
