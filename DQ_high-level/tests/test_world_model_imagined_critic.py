"""Independent imagined value targets, ownership, endpoints and GPU BPTT."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.gradients import freeze_parameters
from utils.world_model.imagined_critic import (ImaginedCritic, ImaginedCriticTrainer,
                                              lambda_returns)
from utils.world_model.imagination import (ImaginationConfig, ImaginationTrainer,
                                          imagination_objective)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((9,), .1))
        self.bias = nn.Parameter(torch.zeros(9))
        self.log_std = nn.Parameter(torch.full((9,), -2.))

    def compute(self, inputs, role):
        return inputs["states"] * self.weight + self.bias, self.log_std, {}

    def mean_to_action(self, mean):
        return torch.cat((mean[..., :6], mean[..., 6:7].sigmoid(), mean[..., 7:]), -1)


class PPOCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(101.))

    def compute(self, inputs, role):
        raise AssertionError("Independent imagined value learning must not query PPO Critic")


class Model(nn.Module):
    term_head = None

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.))

    def step(self, state, action, context, dt):
        return torch.cat((state[:, :1] + self.gain * action[:, :1], state[:, 1:] + 1.), -1)


class Adapter:
    hard_absorbing = True
    critic_feature_dim = 3

    def __init__(self, endpoint=None):
        self.endpoint = endpoint

    def observation(self, state, context):
        return state[:, :1]

    def critic_features(self, state, context):
        return torch.cat((state, context), -1)

    def reward(self, previous, next_state, action, context):
        return 2. + action[:, 0] * 0.

    def timeout(self, state, context):
        return (state[:, 1] >= 1.) & (self.endpoint == "timeout")

    def task_outcome(self, previous, next_state, action, context):
        boundary = next_state[:, 1] >= 1.
        return {"failed": boundary & (self.endpoint == "failure"),
                "completed": boundary & (self.endpoint == "success"),
                "timeout": self.timeout(next_state, context)}


def tree_equal(test, first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            tree_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            tree_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class ImaginedCriticChecks:
    device = "cpu"

    def make(self, count=5, adapter=None, **overrides):
        settings = dict(value_source="imagined", horizon=3, gamma=.9, lr=2e-4,
                        strength=1., max_grad_norm=1., actor_betas=(.9, .95),
                        grad_clip_value=.1, kl_rollback=False, critic_hidden=(),
                        critic_epochs=5, critic_minibatches=4)
        settings.update(overrides)
        trainer = ImaginationTrainer(Model().to(self.device), adapter or Adapter(),
                                     Actor().to(self.device), PPOCritic().to(self.device),
                                     ImaginationConfig(**settings))
        with torch.no_grad():
            head = trainer.imagined_critic.mlp.net[-1]
            head.weight.copy_(torch.tensor([[1., 0., 0.]], device=self.device))
            head.bias.zero_()
        state = torch.stack((torch.linspace(.5, 1.5, count, device=self.device),
                             torch.zeros(count, device=self.device)), -1)
        context = torch.ones(count, 1, device=self.device)
        return trainer, state, context

    def test_td_lambda_targets_match_backward_recursion_and_endpoint_semantics(self):
        rewards = torch.tensor([[1., 1., 1.], [2., 2., 2.]], device=self.device)
        done = torch.tensor([[0., 1., 0.], [0., 0., 0.]], device=self.device)
        timeouts = torch.tensor([[0., 0., 1.], [0., 0., 0.]], device=self.device)
        values = torch.tensor([[4., 4., 4.], [5., 5., 5.], [6., 6., 6.]],
                              device=self.device, requires_grad=True)
        targets = lambda_returns(rewards, done, timeouts, values, gamma=.9, lam=.95)
        expected_last = 2. + .9 * 6.
        expected_first = 1. + .9 * (.05 * 5. + .95 * expected_last)
        torch.testing.assert_close(targets[0], targets.new_tensor([expected_first, 1., 1.]))
        self.assertFalse(targets.requires_grad)
        bootstrapped = lambda_returns(rewards, done, timeouts, values, gamma=.9, lam=.95,
                                      bootstrap_timeouts=True)
        torch.testing.assert_close(bootstrapped[0], targets.new_tensor([expected_first, 1., 5.5]))

    def test_privileged_tail_preserves_input_jacobian_without_ppo_value_scaler(self):
        trainer, state, context = self.make()

        class PoisonScaler(nn.Module):
            def forward(self, value, **kwargs):
                raise AssertionError("Imagined value is already in raw reward units")

        trainer.value_preprocessor = PoisonScaler()
        state.requires_grad_(True)
        with freeze_parameters(trainer.model, trainer.critic, trainer.imagined_critic):
            rollout = trainer.imagine(state, context)
            imagination_objective(rollout, trainer.cfg.gamma).backward()
        expected = .9 ** 3 * 3 * 1.1 ** 2 * state[:, 0].mean()
        torch.testing.assert_close(trainer.actor.weight.grad[0], expected)
        self.assertIsNone(state.grad)
        self.assertIsNone(trainer.model.gain.grad)
        self.assertIsNone(trainer.critic.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.imagined_critic.parameters()))
        self.assertIsNone(trainer.actor.log_std.grad)

    def test_fixed_targets_do_not_change_during_multiple_critic_epochs(self):
        trainer, state, context = self.make()
        with freeze_parameters(trainer.model, trainer.critic, trainer.imagined_critic):
            rollout = trainer.imagine(state, context)
            batch = trainer.imagined_critic_trainer.prepare(rollout, context)
        targets = batch.targets.clone()
        before = deepcopy(trainer.imagined_critic.state_dict())
        original_prepare = trainer.imagined_critic_trainer.prepare
        trainer.imagined_critic_trainer.prepare = lambda *args: self.fail("Targets recomputed during fit")
        metrics = trainer.imagined_critic_trainer.fit(batch)
        trainer.imagined_critic_trainer.prepare = original_prepare
        torch.testing.assert_close(batch.targets, targets, rtol=0, atol=0)
        self.assertEqual(metrics["critic_updates"], 20.)
        self.assertEqual(metrics["critic_samples"], 15.)
        self.assertFalse(batch.features.requires_grad)
        self.assertFalse(batch.targets.requires_grad)
        self.assertTrue(any(not torch.equal(before[key], value)
                            for key, value in trainer.imagined_critic.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.actor.parameters()))

    def test_absorbing_padding_excluded_and_endpoint_reward_retained(self):
        for endpoint in ("success", "failure", "timeout"):
            with self.subTest(endpoint=endpoint):
                trainer, state, context = self.make(adapter=Adapter(endpoint))
                with freeze_parameters(trainer.model, trainer.critic, trainer.imagined_critic):
                    rollout = trainer.imagine(state, context)
                    batch = trainer.imagined_critic_trainer.prepare(rollout, context)
                self.assertEqual(len(batch.targets), len(state))
                torch.testing.assert_close(batch.targets, torch.full_like(batch.targets, 2.))
                torch.testing.assert_close(rollout.states[1], rollout.states[-1])
                self.assertTrue(bool((rollout.survival[1:] == 0).all()))
                torch.testing.assert_close(imagination_objective(rollout, .9), state.new_tensor(2.))

    def test_independent_updates_preserve_real_critic_model_and_actor_std(self):
        trainer, state, context = self.make(chunk_size=2)
        before_model = deepcopy(trainer.model.state_dict())
        before_ppo = deepcopy(trainer.critic.state_dict())
        before_std = trainer.actor.log_std.detach().clone()
        before_actor = trainer.actor.weight.detach().clone()
        before_critic = deepcopy(trainer.imagined_critic.state_dict())
        # Preserve any real-training gradients, not just weights.
        trainer.critic.weight.grad = torch.full_like(trainer.critic.weight, 7.)
        trainer.model.gain.grad = torch.full_like(trainer.model.gain, 11.)
        trainer.actor.log_std.grad = torch.ones_like(trainer.actor.log_std)
        order = []
        actor_step = trainer.optimizer.step
        critic_step = trainer.imagined_critic_trainer.optimizer.step

        def tracked_actor_step(*args, **kwargs):
            order.append("actor")
            return actor_step(*args, **kwargs)

        def tracked_critic_step(*args, **kwargs):
            order.append("critic")
            return critic_step(*args, **kwargs)

        trainer.optimizer.step = tracked_actor_step
        trainer.imagined_critic_trainer.optimizer.step = tracked_critic_step
        metrics = trainer.update(state, context)
        self.assertEqual(metrics["accepted"], 1.)
        self.assertEqual(metrics["actor_chunks"], 3.)
        self.assertEqual(order, ["actor"] + ["critic"] * 20)
        self.assertFalse(torch.equal(before_actor, trainer.actor.weight))
        self.assertTrue(any(not torch.equal(before_critic[key], value)
                            for key, value in trainer.imagined_critic.state_dict().items()))
        tree_equal(self, before_model, trainer.model.state_dict())
        tree_equal(self, before_ppo, trainer.critic.state_dict())
        torch.testing.assert_close(before_std, trainer.actor.log_std, rtol=0, atol=0)
        self.assertIsNone(trainer.actor.log_std.grad)
        self.assertNotIn(trainer.actor.log_std, trainer.optimizer.state)
        self.assertEqual(float(trainer.critic.weight.grad), 7.)
        self.assertEqual(float(trainer.model.gain.grad), 11.)
        self.assertLessEqual(float(trainer.actor.weight.grad.abs().max()), .100001)
        self.assertEqual(trainer.optimizer.param_groups[0]["betas"], (.9, .95))
        for item in trainer.imagined_critic_trainer.optimizer.state.values():
            self.assertEqual(item["exp_avg"].device, state.device)

    def test_chunked_actor_gradient_equals_single_batch_with_sample_weights_and_dt(self):
        whole, state, context = self.make(chunk_size=0)
        chunked, _, _ = self.make(chunk_size=2)
        whole.actor.double()
        whole.model.double()
        whole.imagined_critic.double()
        chunked.actor.double()
        chunked.model.double()
        chunked.imagined_critic.double()
        state, context = state.double(), context.double()
        weights = torch.linspace(.1, 1., len(state), device=self.device, dtype=state.dtype)
        dt = torch.full((3, len(state), 1), .16, device=self.device, dtype=state.dtype)
        first = whole.update(state, context, weights, dt=dt)
        second = chunked.update(state, context, weights, dt=dt)
        torch.testing.assert_close(whole.actor.weight, chunked.actor.weight, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(whole.actor.bias, chunked.actor.bias, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(first["objective"], second["objective"], places=12)
        self.assertEqual(len(whole.optimizer.state), len(chunked.optimizer.state))

    def test_kl_disabled_accepts_finite_update_and_nonfinite_step_still_rolls_back(self):
        trainer, state, context = self.make(lr=.1, max_kl=1e-20, kl_rollback=False)
        metrics = trainer.update(state, context)
        self.assertGreater(metrics["kl"], trainer.cfg.max_kl)
        self.assertEqual(metrics["accepted"], 1.)
        before_actor = deepcopy(trainer.actor.state_dict())
        before_optimizer = deepcopy(trainer.optimizer.state_dict())
        old_step = trainer.optimizer.step

        def bad_step(*args, **kwargs):
            old_step(*args, **kwargs)
            with torch.no_grad():
                trainer.actor.weight[0] = float("nan")

        trainer.optimizer.step = bad_step
        rejected = trainer.update(state, context)
        self.assertEqual(rejected["accepted"], 0.)
        self.assertEqual(rejected["rollback"], 1.)
        tree_equal(self, before_actor, trainer.actor.state_dict())
        tree_equal(self, before_optimizer, trainer.optimizer.state_dict())

    def test_checkpoint_round_trip_keeps_independent_critic_optimizer_and_lr(self):
        trainer, state, context = self.make()
        trainer.update(state, context)
        trainer.set_lr(3e-5, 5e-5)
        checkpoint = deepcopy(trainer.state_dict())
        replacement, _, _ = self.make()
        replacement.load_state_dict(checkpoint)
        tree_equal(self, checkpoint, replacement.state_dict())
        self.assertEqual(replacement.optimizer.param_groups[0]["lr"], 3e-5)
        self.assertEqual(replacement.imagined_critic_trainer.optimizer.param_groups[0]["lr"], 5e-5)

    def test_corrupt_actor_is_detected_before_imagining(self):
        trainer, state, context = self.make(kl_rollback=False)
        with torch.no_grad():
            trainer.actor.weight[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "before its update"):
            trainer.update(state, context)

    def test_corrupt_initial_adam_is_detected_before_imagining(self):
        trainer, state, context = self.make(kl_rollback=False)
        self.assertEqual(trainer.update(state, context)["accepted"], 1.)
        next(iter(trainer.optimizer.state.values()))["exp_avg_sq"].fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "before its update"):
            trainer.update(state, context)

    def test_nonfinite_adam_moment_rolls_back_even_when_actor_is_finite(self):
        trainer, state, context = self.make(kl_rollback=False)
        self.assertEqual(trainer.update(state, context)["accepted"], 1.)
        before_actor = deepcopy(trainer.actor.state_dict())
        before_optimizer = deepcopy(trainer.optimizer.state_dict())
        old_step = trainer.optimizer.step

        def bad_step(*args, **kwargs):
            old_step(*args, **kwargs)
            next(iter(trainer.optimizer.state.values()))["exp_avg"].fill_(float("inf"))

        trainer.optimizer.step = bad_step
        result = trainer.update(state, context)
        self.assertEqual(result["nonfinite_optimizer"], 1.)
        self.assertEqual(result["nonfinite_parameter"], 0.)
        self.assertEqual(result["accepted"], 0.)
        tree_equal(self, before_actor, trainer.actor.state_dict())
        tree_equal(self, before_optimizer, trainer.optimizer.state_dict())

    def test_policy_observation_hook_does_not_pollute_privileged_features(self):
        class NoisyAdapter(Adapter):
            def policy_observation(self, state, context):
                return self.observation(state, context) + .25

        trainer, state, context = self.make(adapter=NoisyAdapter())
        rollout = trainer.imagine(state, context)
        torch.testing.assert_close(rollout.actions[0, :, 0], .1 * (state[:, 0] + .25))
        features = trainer.adapter.critic_features(state, context)
        torch.testing.assert_close(features[:, :2], state)


class CPUChecks(ImaginedCriticChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAChecks(ImaginedCriticChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
