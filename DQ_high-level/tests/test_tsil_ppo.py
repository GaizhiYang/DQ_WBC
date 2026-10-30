"""CPU algorithm checks for the joint SGS/EKF/TSIL teacher (no Isaac Gym)."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party/skrl"))

from skrl.memories.torch import RandomMemory
from utils.tsil import ReplaySelection, TemporalState
from utils.tsil_checkpoint import TemporalCheckpointMetadata
from utils.tsil_normalization import FrozenObservationScaler, RewardTraceNormalizer, ValueScaler
from utils.tsil_ppo import TSILPPO, clipped_value_loss, compute_gae, positive_gap_losses


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device, self.num_actions = torch.device("cpu"), 1
        self.linear = torch.nn.Linear(4, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1))

    def act(self, inputs, role="policy"):
        mean = self.linear(inputs["states"])
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
        action = self.distribution.sample()
        logp = self.distribution.log_prob(inputs.get("taken_actions", action))
        return action, logp, {"mean_actions": mean}

    def get_entropy(self, role="policy"):
        return self.distribution.entropy()

    def set_mode(self, mode):
        self.train(mode == "train")


class TinyValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def act(self, inputs, role="value"):
        return self.linear(inputs["states"]), None, {}

    def set_mode(self, mode):
        self.train(mode == "train")


def make_agent(num_envs=4, memory=True, micro_batch_size=3):
    state = TemporalState(4, 16., family_ids=[0, 1, 2, 0])
    env = SimpleNamespace(tsil_state=state, num_envs=num_envs, max_episode_length=3,
                          tsil_time_reference=16.)
    cfg = dict(rollouts=2, learning_epochs=3, mini_batches=2, learning_rate=1e-3,
               grad_norm_clip=1., clip_predicted_values=True, value_clip=.2,
               experiment=dict(write_interval=0, checkpoint_interval=0),
               tsil=dict(time_indices=(2, 3), static_prefix_size=0, samples_per_trajectory=2,
                         micro_batch_size=micro_batch_size, screen_batch_size=8))
    agent = TSILPPO(models={"policy": TinyPolicy(), "value": TinyValue()},
                    memory=RandomMemory(2, num_envs, device="cpu") if memory else None,
                    cfg=cfg, observation_space=4, action_space=1, device="cpu", tsil_env=env)
    agent.init()
    return agent


class NormalizationTests(unittest.TestCase):
    def test_observation_snapshot_freezes_likelihood_and_exempts_time(self):
        scaler = FrozenObservationScaler(4, time_indices=(2, 3), device="cpu")
        states = torch.tensor([[2., 4., .25, 7.], [6., 8., .5, 12.]])
        before = scaler(states)
        scaler.observe(states)
        torch.testing.assert_close(scaler(states), before)
        self.assertEqual(scaler.current_count.item(), 1)
        scaler.commit()
        self.assertEqual(scaler.current_count.item(), 3)
        self.assertFalse(torch.equal(scaler(states)[:, :2], before[:, :2]))
        torch.testing.assert_close(scaler(states)[:, 2:], states[:, 2:])
        self.assertEqual(scaler.pending_count.item(), 0)
        with self.assertRaisesRegex(ValueError, "optimization"):
            scaler(states, train=True)

    def test_reward_traces_are_per_env_terminal_is_observed_then_cleared(self):
        normalizer = RewardTraceNormalizer(2, gamma=.5)
        normalizer.observe(torch.tensor([1., 3.]), torch.tensor([False, True]))
        torch.testing.assert_close(normalizer.trace, torch.tensor([1., 0.], dtype=torch.float64))
        normalizer.observe(torch.tensor([2., 4.]), torch.tensor([False, False]))
        sample_traces = torch.tensor([1., 3., 2.5, 4.], dtype=torch.float64)
        torch.testing.assert_close(normalizer.running_mean, sample_traces.mean())
        torch.testing.assert_close(normalizer.running_variance, sample_traces.var(unbiased=False))
        self.assertEqual(normalizer.active_scale.item(), 1.)
        ratio = normalizer.commit()
        self.assertAlmostEqual(ratio, 1 / sample_traces.std(unbiased=False).item())
        normalizer.observe(torch.tensor([7., 1.]), torch.zeros(2, dtype=torch.bool),
                           episode_start=torch.tensor([True, False]))
        torch.testing.assert_close(normalizer.trace, torch.tensor([7., 3.], dtype=torch.float64))

    def test_value_statistics_updated_only_from_returns_and_unit_conversion(self):
        scaler = ValueScaler(size=1, device="cpu")
        scaler.update_returns(torch.tensor([[[2.], [4.]], [[6.], [8.]]]))
        self.assertEqual(scaler.current_count.item(), 5.)
        mean, var = scaler.running_mean.clone(), scaler.running_variance.clone()
        normalized = torch.tensor([[.2], [-.3]])
        raw = scaler(normalized, inverse=True)
        scaler.rescale_units(.25)
        torch.testing.assert_close(scaler.running_mean, mean * .25)
        torch.testing.assert_close(scaler.running_variance, var * .25 ** 2)
        torch.testing.assert_close(scaler(normalized, inverse=True), raw * .25)
        self.assertEqual(scaler.current_count.item(), 5.)

    def test_value_adapter_preserves_raw_predictions_and_invertibility_outside_clip_range(self):
        scaler = ValueScaler(size=1, device="cpu")
        network = torch.tensor([[-20.], [30.]], dtype=torch.float64, requires_grad=True)
        raw_before = scaler(scaler.normalize_prediction(network), inverse=True)
        torch.testing.assert_close(raw_before, network.detach())
        scaler.update_returns(torch.tensor([[-1000.], [3000.]], dtype=torch.float64))
        current = scaler.normalize_prediction(network)
        torch.testing.assert_close(scaler(current, inverse=True), raw_before)
        torch.testing.assert_close(current, scaler(raw_before))
        extreme = torch.tensor([[-50.], [100.]], dtype=torch.float64)
        torch.testing.assert_close(scaler(scaler(extreme, inverse=True)), extreme)
        current.sum().backward()
        torch.testing.assert_close(network.grad, scaler.prediction_scale.expand_as(network))
        self.assertEqual(list(scaler.parameters()), [])
        with self.assertRaisesRegex(ValueError, "update_returns"):
            scaler(raw_before, train=True)

    def test_value_stat_update_leaves_warmed_adam_and_model_parameters_untouched(self):
        model = torch.nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scaler = ValueScaler(size=1, device="cpu")
        inputs = torch.randn(5, 4)
        for _ in range(3):
            optimizer.zero_grad()
            scaler.normalize_prediction(model(inputs)).square().mean().backward()
            optimizer.step()
        parameters = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(optimizer.state_dict())
        raw_before = scaler(scaler.normalize_prediction(model(inputs)), inverse=True)
        for returns in (torch.full((10, 1), 200.), torch.full((20, 1), -100.)):
            scaler.update_returns(returns)
            torch.testing.assert_close(
                scaler(scaler.normalize_prediction(model(inputs)), inverse=True),
                raw_before, atol=2e-5, rtol=1e-4)
        for name, expected in parameters.items():
            torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
        actual_moments = optimizer.state_dict()
        self.assertEqual(actual_moments["param_groups"], moments["param_groups"])
        for parameter, expected in moments["state"].items():
            for name, value in expected.items():
                torch.testing.assert_close(actual_moments["state"][parameter][name], value, atol=0, rtol=0)

    def test_value_unit_conversion_preserves_adapter_even_below_initial_epsilon(self):
        scaler = ValueScaler(size=1, device="cpu")
        scaler.running_variance.zero_()
        network = torch.tensor([[-20.], [30.]], dtype=torch.float64)
        expected_raw = scaler(scaler.normalize_prediction(network), inverse=True)
        initial_floor = scaler.standard_deviation_floor.clone()
        for ratio in (1e-8, .5, 100.):
            adapter = (scaler.prediction_scale.clone(), scaler.prediction_bias.clone())
            scaler.rescale_units(ratio)
            expected_raw *= ratio
            torch.testing.assert_close(
                scaler(scaler.normalize_prediction(network), inverse=True), expected_raw,
                rtol=1e-12, atol=1e-25)
            torch.testing.assert_close(scaler.prediction_scale, adapter[0], atol=0, rtol=0)
            torch.testing.assert_close(scaler.prediction_bias, adapter[1], atol=0, rtol=0)
            scaler.update_returns(torch.tensor([[1e-15], [-2e-15]], dtype=torch.float64))
            torch.testing.assert_close(
                scaler(scaler.normalize_prediction(network), inverse=True), expected_raw,
                rtol=1e-12, atol=1e-25)
        torch.testing.assert_close(scaler.standard_deviation_floor, initial_floor * 5e-7,
                                   rtol=1e-12, atol=0)
        for ratio in (0., -1., float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "positive ratio"):
                scaler.rescale_units(ratio)

    def test_reward_checkpoint_can_evaluate_different_env_count(self):
        training = RewardTraceNormalizer(4)
        training.observe(torch.arange(4.), torch.zeros(4, dtype=torch.bool))
        training.commit()
        evaluation = RewardTraceNormalizer(2)
        evaluation.load_state_dict(training.state_dict())
        self.assertEqual(evaluation.trace.shape, (2,))
        self.assertEqual(evaluation.trace.sum().item(), 0.)
        torch.testing.assert_close(evaluation.active_scale, training.active_scale)


class JointLossTests(unittest.TestCase):
    def test_value_clipping_keeps_unclipped_penalty_for_worsening_prediction(self):
        values = torch.tensor([[1.]], requires_grad=True)
        old = torch.zeros_like(values)
        loss = clipped_value_loss(values, old, torch.full_like(values, -1.), True, .2)
        self.assertAlmostEqual(loss.item(), 4.)
        loss.backward()
        self.assertAlmostEqual(values.grad.item(), 4.)
        # An improvement beyond the trust region is legitimately capped.
        values.grad.zero_()
        loss = clipped_value_loss(values, old, torch.full_like(values, 10.), True, .2)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 9.8 ** 2, places=4)
        self.assertEqual(values.grad.item(), 0.)

    def test_critic_gradient_and_actor_detachment_obey_positive_gate(self):
        values = torch.tensor([[1.], [3.], [-1.]], requires_grad=True)
        logp = torch.tensor([[-2.], [-4.], [-1.]], requires_grad=True)
        target = torch.tensor([[2.], [1.], [-1.]])
        efficiency = torch.tensor([[2.], [1.], [1.]])
        actor, critic, gap, weights = positive_gap_losses(logp, values, target, efficiency)
        actor.backward(retain_graph=True)
        self.assertIsNone(values.grad)
        torch.testing.assert_close(logp.grad, torch.tensor([[-1 / 3], [0.], [0.]]))
        critic.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[-1 / 3], [0.], [0.]]))
        self.assertTrue(torch.isfinite(actor))  # Singleton positive weight std=0 is safe.
        self.assertAlmostEqual(critic.item(), 1 / 6)

    def test_no_positive_gap_has_zero_gradients_and_no_nans(self):
        values = torch.ones((3, 1), requires_grad=True)
        logp = torch.full((3, 1), -2., requires_grad=True)
        actor, critic, _, _ = positive_gap_losses(logp, values, torch.zeros(3, 1), torch.ones(3, 1))
        (actor + critic).backward()
        self.assertEqual(values.grad.abs().sum().item(), 0.)
        self.assertEqual(logp.grad.abs().sum().item(), 0.)
        self.assertEqual((actor + critic).item(), 0.)

    def test_microbatches_match_full_logical_batch_gradient(self):
        torch.manual_seed(7)
        full = make_agent(micro_batch_size=100)
        micro = make_agent(micro_batch_size=2)
        micro.policy.load_state_dict(full.policy.state_dict())
        micro.value.load_state_dict(full.value.state_dict())
        count = 7
        states = torch.randn(count, 4)
        actions = torch.randn(count, 1)
        returns = torch.tensor([[-2.], [.02], [.3], [1.], [2.], [3.], [4.]])
        selection = ReplaySelection(states, actions, returns,
                                    torch.tensor([[1.], [2.], [1.5], [1.], [1.8], [2.], [1.]]),
                                    torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.bool),
                                    torch.ones(count), torch.zeros(count, 3, dtype=torch.long), None, {})
        indices = torch.arange(count)
        full._backward_replay(selection, indices)
        micro._backward_replay(selection, indices)
        for model_name in ("policy", "value"):
            for a, b in zip(getattr(full, model_name).parameters(), getattr(micro, model_name).parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7)

    def test_replay_screen_and_losses_use_compensated_value_and_preserve_gradients(self):
        agent = make_agent()
        agent._value_preprocessor.update_returns(torch.full((8, 1), 10.))
        states = torch.zeros(4, 4)
        returns = torch.tensor([[1.], [2.], [3.], [4.]])
        selection = ReplaySelection(states, torch.full((4, 1), .25), returns,
                                    torch.ones(4, 1), torch.zeros(4, dtype=torch.long),
                                    torch.ones(4, dtype=torch.bool), torch.ones(4),
                                    torch.zeros(4, 3, dtype=torch.long), None, {})
        expected_gap = returns / agent._value_preprocessor._standard_deviation().float()
        torch.testing.assert_close(agent._screen_gap(states, returns), expected_gap.flatten())
        indices = torch.arange(4)
        torch.testing.assert_close(agent._logical_weight_std(selection, indices),
                                   expected_gap.flatten().std(unbiased=False))
        agent._backward_replay(selection, indices)
        for model in (agent.value, agent.policy):
            gradients = [parameter.grad.flatten() for parameter in model.parameters()
                         if parameter.grad is not None]
            self.assertTrue(torch.isfinite(torch.cat(gradients)).all())
            self.assertGreater(torch.cat(gradients).norm().item(), 0.)

    def test_task_terminal_zero_tail_and_rollout_boundary_bootstrap(self):
        rewards = torch.tensor([[[1.], [1.]], [[2.], [2.]]])
        dones = torch.tensor([[[False], [False]], [[True], [False]]])
        returns, _ = compute_gae(rewards, dones, torch.zeros_like(rewards),
                                torch.tensor([[100.], [10.]]), gamma=.5, lam=1.)
        torch.testing.assert_close(returns, torch.tensor([[[2.], [4.5]], [[2.], [7.]]]))


class PPOIntegrationTests(unittest.TestCase):
    def test_empty_replay_critic_updates_with_production_clipping_and_rms_shift(self):
        torch.manual_seed(20261030)
        agent = make_agent()
        self.assertTrue(agent._clip_predicted_values)
        initial = copy.deepcopy(agent.value.state_dict())
        original_update = agent._value_preprocessor.update_returns
        original_step = agent.optimizer.step
        gradients, coordinate_errors = [], []
        states = torch.zeros(4, 4)
        states[:, 3] = 1.

        def update_returns(returns):
            normalized = agent._state_preprocessor(states)
            old_raw = agent._value_preprocessor(agent._predict_values(normalized), inverse=True)
            original_update(returns)
            current = agent._predict_values(normalized)
            cached = agent._value_preprocessor(old_raw)
            coordinate_errors.append((current - cached).abs().max().item())

        def step(*args, **kwargs):
            gradients.append(torch.cat([parameter.grad.flatten() for parameter in agent.value.parameters()
                                        if parameter.grad is not None]).norm().item())
            return original_step(*args, **kwargs)

        agent._value_preprocessor.update_returns = update_returns
        agent.optimizer.step = step
        for index in range(2):
            elapsed = torch.full((4,), .1 * (index + 1))
            info = dict(config_id=torch.arange(4), episode_id=torch.ones(4, dtype=torch.long),
                        elapsed=elapsed, elapsed_before=elapsed - .1,
                        success=torch.zeros(4, dtype=torch.bool), reward_fixed=torch.ones(4),
                        reward_command_raw=torch.zeros(4), command_gate=False)
            done = torch.zeros(4, 1, dtype=torch.bool)
            with torch.no_grad():
                actions, _, _ = agent.act(states, index, 20)
                agent.record_transition(states, actions, torch.ones(4, 1), states, done, done,
                                        {"tsil": info}, index, 20)
            agent.post_interaction(index, 20)
        self.assertEqual(len(agent._last_replay_selection), 0)
        self.assertEqual(len(gradients), 6)
        self.assertTrue(all(gradient > 0 and torch.isfinite(torch.tensor(gradient))
                            for gradient in gradients))
        self.assertLess(max(coordinate_errors), 1e-6)
        self.assertGreater(max((parameter - initial[name]).abs().max().item()
                               for name, parameter in agent.value.state_dict().items()), 0.)

    def test_rollout_cache_and_bootstrap_apply_adapter_after_prior_rms_change(self):
        agent = make_agent()
        agent._value_preprocessor.update_returns(torch.full((8, 1), 10.))
        states = torch.zeros(4, 4)
        original_update = agent._value_preprocessor.update_returns
        captured_returns = []

        def update_returns(returns):
            captured_returns.append(returns.clone())
            original_update(returns)

        agent._value_preprocessor.update_returns = update_returns
        for index in range(2):
            elapsed = torch.full((4,), .1 * (index + 1))
            info = dict(config_id=torch.arange(4), episode_id=torch.ones(4, dtype=torch.long),
                        elapsed=elapsed, elapsed_before=elapsed - .1,
                        success=torch.zeros(4, dtype=torch.bool), reward_fixed=torch.ones(4),
                        reward_command_raw=torch.zeros(4), command_gate=False)
            done = torch.zeros(4, 1, dtype=torch.bool)
            with torch.no_grad():
                actions, _, _ = agent.act(states, index, 20)
                agent.record_transition(states, actions, torch.ones(4, 1), states, done, done,
                                        {"tsil": info}, index, 20)
            cached = agent.memory.get_tensor_by_name("values")[index]
            torch.testing.assert_close(cached, torch.zeros_like(cached), atol=1e-6, rtol=0)
            agent.post_interaction(index, 20)
        expected = torch.ones(2, 4, 1)
        expected[0] += agent._discount_factor * agent._lambda
        torch.testing.assert_close(captured_returns[0], expected, atol=2e-6, rtol=1e-6)

    def test_full_joint_update_keeps_replay_set_and_commits_only_after_optimization(self):
        torch.manual_seed(11)
        agent = make_agent()
        original_step = agent.optimizer.step
        optimizer_targets = []
        def step(*args, **kwargs):
            optimizer_targets.append(agent.tsil_state.targets.clone())
            return original_step(*args, **kwargs)
        agent.optimizer.step = step
        original_replay = agent._backward_replay
        replay_batches = []
        def backward(selection, indices):
            replay_batches.append((len(selection), indices.clone()))
            return original_replay(selection, indices)
        agent._backward_replay = backward
        states = torch.zeros(4, 4)
        states[:, 0] = torch.arange(4) * .1
        states[:, 3] = 1.
        episodes = torch.ones(4, dtype=torch.long)
        for step_id in range(4):
            within = step_id % 3
            states[:, 2] = within * .1 / 16
            elapsed = torch.full((4,), (within + 1) * .1)
            done = torch.full((4, 1), within == 2)
            success = (torch.arange(4) % 2 == 0) & done.reshape(-1)
            rewards = 1. + success[:, None].float() * 100
            info = dict(config_id=torch.arange(4), episode_id=episodes.clone(), elapsed=elapsed,
                        elapsed_before=torch.full((4,), within * .1), success=success,
                        reward_fixed=torch.ones(4), reward_command_raw=torch.zeros(4), command_gate=False)
            with torch.no_grad():
                actions, old_logp, _ = agent.act(states, step_id, 20)
                next_states = states.clone()
                next_states[:, 2] = elapsed / 16
                agent.record_transition(states.clone(), actions, rewards, next_states, done,
                                        torch.zeros_like(done), {"tsil": info}, step_id, 20)
            if step_id == 2:
                self.assertTrue(torch.equal(agent.tsil_state.targets, torch.full((4,), 16.)))
                self.assertEqual(agent.replay.completed_episodes, 4)
            agent.post_interaction(step_id, 20)
            if within == 2:
                episodes += 1
        self.assertEqual(len(optimizer_targets), 12)  # 2 updates x 3 epochs x 2 PPO batches.
        self.assertTrue(all(torch.equal(target, torch.full((4,), 16.)) for target in optimizer_targets))
        torch.testing.assert_close(agent.tsil_state.targets, torch.tensor([.3, 16., .3, 16.]))
        self.assertEqual(agent._value_preprocessor.current_count.item(), 17.)
        self.assertEqual(agent._state_preprocessor.current_count.item(), 17.)
        self.assertEqual(agent.reward_normalizer.current_count.item(), 16.)
        self.assertGreater(len(agent._last_replay_selection), 0)
        for start in range(6, 12, 2):
            n, first = replay_batches[start]
            _, second = replay_batches[start + 1]
            torch.testing.assert_close(torch.cat((first, second)).sort().values, torch.arange(n))
        self.assertIn("TSIL PPO / full_task/F_success", agent.tracking_data)
        self.assertIn("TSIL Reference / logp_change", agent.tracking_data)
        # Completed episodes were not split into two 2-step fragments.
        self.assertTrue(all(len(traj) == 3 for traj in agent.replay.iter_selected()))

    def test_checkpoint_load_requires_temporal_state_and_supports_new_eval_env_count(self):
        training = make_agent()
        training._value_preprocessor.update_returns(torch.full((10, 1), 7.))
        training._value_preprocessor.rescale_units(.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            training.save(str(path))
            evaluation = make_agent(num_envs=2, memory=False)
            evaluation.load(str(path))
            self.assertIsNone(evaluation.replay)
            self.assertEqual(evaluation.reward_normalizer.trace.shape, (2,))
            states = torch.randn(2, 4)
            torch.testing.assert_close(evaluation._predict_values(states), training._predict_values(states))
            for name in ValueScaler.ADAPTER_STATE_KEYS:
                torch.testing.assert_close(evaluation._value_preprocessor.state_dict()[name],
                                           training._value_preprocessor.state_dict()[name])
            state = torch.load(path, weights_only=False)
            del state["tsil_state"]
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "Incomplete TSIL"):
                evaluation.load(str(path))

    def test_legacy_checkpoint_requires_explicit_actor_only_evaluation(self):
        training = make_agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            training.save(str(path))
            legacy = torch.load(path, weights_only=False)
            legacy["tsil_metadata"] = {"version": 1}
            for name in ValueScaler.ADAPTER_STATE_KEYS:
                del legacy["value_preprocessor"][name]
            torch.save(legacy, path)
            evaluation = make_agent(num_envs=2, memory=False)
            with self.assertWarnsRegex(RuntimeWarning, "policy evaluation only"):
                evaluation.load(str(path))
            states = torch.randn(2, 4)
            with torch.no_grad():
                expected = training.act(states, 0, 1)[2]["mean_actions"]
                actual = evaluation.act(states, 0, 1)[2]["mean_actions"]
            torch.testing.assert_close(actual, expected)
            with self.assertRaisesRegex(RuntimeError, "Critic use is unavailable"):
                evaluation._predict_values(states)
            with self.assertRaisesRegex(ValueError, "policy evaluation only"):
                training.load(str(path))
            # Neither partial new state nor new metadata can masquerade as the
            # legacy format and silently reinitialize a learned adapter.
            partial = copy.deepcopy(legacy)
            partial["value_preprocessor"]["prediction_scale"] = torch.ones(1, dtype=torch.float64)
            torch.save(partial, path)
            with self.assertRaisesRegex(ValueError, "Incomplete TSIL value adapter"):
                evaluation.load(str(path))
            for metadata in ({"version": 2}, {}):
                invalid = copy.deepcopy(legacy)
                invalid["tsil_metadata"] = metadata
                torch.save(invalid, path)
                with self.assertRaisesRegex(ValueError, "Incomplete TSIL value adapter"):
                    evaluation.load(str(path))

    def test_value_scaler_direct_load_is_strict_about_adapter_state(self):
        scaler = ValueScaler(size=1, device="cpu")
        state = scaler.state_dict()
        del state["prediction_bias"]
        with self.assertRaisesRegex(RuntimeError, "prediction_bias"):
            scaler.load_state_dict(state)

    def test_legacy_actor_only_restriction_survives_checkpoint_resave(self):
        training = make_agent()
        metadata = TemporalCheckpointMetadata({"env": {}}, SimpleNamespace(), 4, 16.)
        training.checkpoint_modules["tsil_metadata"] = metadata
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            migrated_path = Path(directory) / "migrated.pt"
            training.save(str(path))
            payload = torch.load(path, weights_only=False)
            payload["tsil_metadata"]["version"] = 1
            for name in ValueScaler.ADAPTER_STATE_KEYS:
                del payload["value_preprocessor"][name]
            torch.save(payload, path)
            evaluation = make_agent(num_envs=2, memory=False)
            evaluation.checkpoint_modules["tsil_metadata"] = copy.deepcopy(metadata)
            with self.assertWarnsRegex(RuntimeWarning, "policy evaluation only"):
                evaluation.load(str(path))
            evaluation.save(str(migrated_path))
            saved = torch.load(migrated_path, weights_only=False)
            self.assertEqual(saved["tsil_metadata"]["migrated_from_version"], 1)
            evaluation_again = make_agent(num_envs=2, memory=False)
            evaluation_again.checkpoint_modules["tsil_metadata"] = copy.deepcopy(metadata)
            with self.assertWarnsRegex(RuntimeWarning, "policy evaluation only"):
                evaluation_again.load(str(migrated_path))
            with self.assertRaisesRegex(RuntimeError, "Critic use is unavailable"):
                evaluation_again._predict_values(torch.zeros(2, 4))
            with self.assertRaisesRegex(ValueError, "policy evaluation only"):
                training.load(str(migrated_path))


if __name__ == "__main__":
    unittest.main()
