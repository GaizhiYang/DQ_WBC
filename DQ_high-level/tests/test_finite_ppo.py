"""Numerical rejection and ordinary-update parity with the vendored PPO (CPU)."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from utils.gripper_policy import ContinuousGripperGaussianMixin
from utils.finite_ppo import FinitePPO, PPONumericalError, clipped_policy_loss


class Policy(ContinuousGripperGaussianMixin, Model):
    def __init__(self):
        Model.__init__(self, 4, 9, device="cpu")
        ContinuousGripperGaussianMixin.__init__(self)
        self.linear = torch.nn.Linear(4, 9)
        self.log_std = torch.nn.Parameter(torch.full((9,), -.5))

    def compute(self, inputs, role="policy"):
        return self.linear(inputs["states"]), self.log_std, {}


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = torch.nn.Linear(4, 1)

    def act(self, inputs, role="value"):
        return self.linear(inputs["states"]), None, {}

    def set_mode(self, mode):
        self.train(mode == "train")


def make_agent(cls=FinitePPO, preprocess=False, policy_cls=Policy, **overrides):
    torch.manual_seed(12)
    config = dict(rollouts=3, learning_epochs=1, mini_batches=1, learning_rate=1e-3,
                  clip_predicted_values=True, entropy_loss_scale=.001,
                  experiment={"write_interval": 0, "checkpoint_interval": 0})
    if preprocess:
        config.update(state_preprocessor=RunningStandardScaler,
                      state_preprocessor_kwargs={"size": 4, "device": "cpu"},
                      value_preprocessor=RunningStandardScaler,
                      value_preprocessor_kwargs={"size": 1, "device": "cpu"})
    config.update(overrides)
    agent = cls(models={"policy": policy_cls(), "value": Value()},
                memory=RandomMemory(3, 8, device="cpu"),
                observation_space=4, action_space=9, device="cpu", cfg=config)
    agent.init()
    with torch.no_grad():
        for step in range(3):
            states, future = torch.randn(8, 4), torch.randn(8, 4)
            actions, _, _ = agent.act(states, step, 100)
            reward = torch.linspace(-.2, .4, 8)[:, None] + .05 * step
            done = torch.zeros(8, 1, dtype=torch.bool)
            done[step] = True
            agent.record_transition(states, actions, reward, future, done,
                                    torch.zeros_like(done), {}, step, 100)
    return agent


class FinitePPOTests(unittest.TestCase):
    def assert_nested_close(self, first, second, *, exact=False):
        self.assertEqual(type(first), type(second))
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0 if exact else 2e-5,
                                       atol=0 if exact else 2e-7)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_close(first[key], second[key], exact=exact)
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_close(a, b, exact=exact)
        else:
            self.assertEqual(first, second)

    def snapshot(self, agent):
        return deepcopy((agent.policy.state_dict(), agent.value.state_dict(), agent.optimizer.state_dict()))

    def test_surrogate_value_and_gradient_match_both_advantage_signs(self):
        torch.manual_seed(29)
        log_ratio = (torch.randn(101, dtype=torch.float64) * 2).requires_grad_()
        advantages = torch.randn_like(log_ratio)
        advantages[0] = 0
        ratio = log_ratio.exp()
        expected = -torch.minimum(advantages * ratio, advantages * ratio.clamp(.8, 1.2)).mean()
        expected_gradient, = torch.autograd.grad(expected, log_ratio)
        actual = clipped_policy_loss(log_ratio, advantages, .2)
        actual_gradient, = torch.autograd.grad(actual, log_ratio)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-13)

    def test_finite_loss_nan_gradient_overflow_is_removed_without_capping_negative_branch(self):
        log_ratio = torch.tensor([100.], requires_grad=True)
        ratio = log_ratio.exp()
        old_loss = -torch.minimum(ratio, ratio.clamp(.8, 1.2)).mean()
        self.assertTrue(bool(torch.isfinite(old_loss)))
        old_gradient, = torch.autograd.grad(old_loss, log_ratio)
        self.assertTrue(bool(torch.isnan(old_gradient).all()))
        loss = clipped_policy_loss(log_ratio, torch.ones_like(log_ratio), .2)
        gradient, = torch.autograd.grad(loss, log_ratio)
        torch.testing.assert_close(loss, torch.tensor(-1.2))
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
        with self.assertRaisesRegex(PPONumericalError, "surrogate_overflow"):
            clipped_policy_loss(log_ratio, -torch.ones_like(log_ratio), .2)

    def test_normal_updates_match_original_gae_scalers_adam_and_scheduler(self):
        for preprocess in (False, True):
            with self.subTest(preprocess=preprocess):
                options = dict(learning_epochs=3, mini_batches=2,
                               learning_rate_scheduler=KLAdaptiveLR,
                               learning_rate_scheduler_kwargs={"kl_threshold": .008})
                reference = make_agent(PPO, preprocess, **options)
                actual = make_agent(FinitePPO, preprocess, **options)
                torch.manual_seed(38)
                reference._update(3, 100)
                torch.manual_seed(38)
                actual._update(3, 100)
                self.assert_nested_close(self.snapshot(actual), self.snapshot(reference))
                for name in ("values", "returns", "advantages"):
                    torch.testing.assert_close(actual.memory.get_tensor_by_name(name),
                                               reference.memory.get_tensor_by_name(name), rtol=0, atol=0)
                if preprocess:
                    for name in ("_state_preprocessor", "_value_preprocessor"):
                        self.assert_nested_close(getattr(actual, name).state_dict(),
                                                 getattr(reference, name).state_dict(), exact=True)
                self.assertEqual(actual.scheduler.get_last_lr(), reference.scheduler.get_last_lr())
                self.assertEqual(actual.last_ppo_numerics["accepted_minibatches"], 6)
                self.assertEqual(actual.last_ppo_numerics["rejected_minibatches"], 0)

    def test_bad_value_backward_cannot_poison_actor_or_adam(self):
        agent = make_agent()
        before = self.snapshot(agent)
        messages = []
        agent._report_progress = messages.append
        hook = agent.value.linear.weight.register_hook(lambda gradient: gradient * float("nan"))
        try:
            agent._update(3, 100)
        finally:
            hook.remove()
        self.assert_nested_close(self.snapshot(agent), before, exact=True)
        self.assertEqual(agent.last_ppo_numerics["reason_counts"], {"nonfinite_gradient": 1})
        self.assertTrue(all(parameter.grad is None for model in (agent.policy, agent.value)
                            for parameter in model.parameters()))
        self.assertIn("nonfinite_gradient", messages[0])

    def test_positive_extreme_ratio_updates_and_negative_extreme_ratio_rejects(self):
        for sign in (1., -1.):
            with self.subTest(advantage_sign=sign):
                agent = make_agent()
                before = self.snapshot(agent)
                sample = agent.memory.sample_all

                def extreme_batches(*args, **kwargs):
                    return [list(batch[:2]) + [batch[2] - 100.] + list(batch[3:5]) +
                            [torch.full_like(batch[5], sign)] + list(batch[6:])
                            for batch in sample(*args, **kwargs)]

                with patch.object(agent.memory, "sample_all", extreme_batches):
                    agent._update(3, 100)
                stats = agent.last_ppo_numerics
                self.assertEqual(stats["nonfinite_kl_diagnostics"], 1)
                if sign > 0:
                    self.assertEqual(stats["accepted_minibatches"], 1)
                    self.assertTrue(all(bool(torch.isfinite(parameter).all())
                                        for parameter in agent.policy.parameters()))
                else:
                    self.assertEqual(stats["reason_counts"], {"surrogate_overflow": 1})
                    self.assert_nested_close(self.snapshot(agent), before, exact=True)

    def test_finite_gradients_with_overflowing_joint_norm_are_rejected(self):
        agent = make_agent()
        before = self.snapshot(agent)
        hook = agent.value.linear.weight.register_hook(lambda gradient: torch.full_like(gradient, 1e30))
        try:
            agent._update(3, 100)
        finally:
            hook.remove()
        self.assert_nested_close(self.snapshot(agent), before, exact=True)
        self.assertEqual(agent.last_ppo_numerics["reason_counts"], {"nonfinite_gradient_norm": 1})

    def test_invalid_optimizer_step_restores_parameters_and_existing_moments(self):
        for corrupt_parameters in (False, True):
            with self.subTest(corrupt_parameters=corrupt_parameters):
                agent = make_agent()
                agent._update(3, 100)  # nonzero Adam step, exp_avg and exp_avg_sq
                before = self.snapshot(agent)
                original_step = agent.optimizer.step

                def corrupt_step(*args, **kwargs):
                    result = original_step(*args, **kwargs)
                    parameter = next(agent.policy.parameters())
                    with torch.no_grad():
                        agent.optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
                        if corrupt_parameters:
                            parameter.fill_(float("inf"))
                    return result

                with patch.object(agent.optimizer, "step", corrupt_step):
                    agent._update(6, 100)
                self.assert_nested_close(self.snapshot(agent), before, exact=True)
                reason = ("nonfinite_parameter_after_step" if corrupt_parameters else
                          "nonfinite_optimizer_after_step")
                self.assertEqual(agent.last_ppo_numerics["reason_counts"], {reason: 1})
                self.assertEqual(agent.last_ppo_numerics["rolled_back_steps"], 1)

    def test_latent_memory_keeps_actual_sample_and_public_execution_log_probability(self):
        agent = make_agent()
        self.assertEqual(agent._tensors_names[-1], "latent_actions")
        states = agent.memory.get_tensor_by_name("states")
        actions = agent.memory.get_tensor_by_name("actions")
        latent = agent.memory.get_tensor_by_name("latent_actions")
        old_logp = agent.memory.get_tensor_by_name("log_prob")
        torch.testing.assert_close(actions, agent.policy.mean_to_action(latent), rtol=0, atol=0)
        self.assertFalse(latent.requires_grad)
        _, _, replay = agent.policy.act({"states": states.flatten(0, 1),
            "taken_actions": actions.flatten(0, 1), "taken_latent_actions": latent.flatten(0, 1),
            "latent_ppo": True})
        torch.testing.assert_close(replay["latent_log_prob"].reshape_as(old_logp), old_logp, rtol=0, atol=0)
        with torch.no_grad():
            executed, public_logp, outputs = agent.act(torch.zeros(8, 4), 4, 100)
        _, expected_public, _ = agent.policy.act({"states": torch.zeros(8, 4), "taken_actions": executed})
        torch.testing.assert_close(public_logp, expected_public, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(agent._current_log_prob, outputs["latent_log_prob"], rtol=0, atol=0)

    def test_saturated_policy_replays_gaussian_and_updates_std(self):
        for mean in (100., 1000.):
            with self.subTest(mean=mean):
                class SaturatedPolicy(Policy):
                    def __init__(self):
                        super().__init__()
                        with torch.no_grad():
                            self.linear.weight[6].zero_()
                            self.linear.bias[6] = mean
                agent = make_agent(policy_cls=SaturatedPolicy, entropy_loss_scale=0)
                latent = agent.memory.get_tensor_by_name("latent_actions")
                actions = agent.memory.get_tensor_by_name("actions")
                self.assertGreater(len(latent[..., 6].unique()), 1)
                self.assertEqual(len(actions[..., 6].unique()), 1)
                self.assertLess(float((latent[..., 6] - mean).abs().max()), 3.)
                old_std = agent.policy.log_std.detach().clone()
                agent._update(3, 100)
                self.assertEqual(agent.last_ppo_numerics["accepted_minibatches"], 1)
                self.assertEqual(agent.last_ppo_numerics["rejected_minibatches"], 0)
                # Collection and replay use different GEMM batch sizes; allow
                # float32 roundoff in the other eight Linear outputs.
                self.assertLessEqual(agent.last_ppo_numerics["log_ratio_abs_max"], 2e-6)
                self.assertNotEqual(float(agent.policy.log_std[6]), float(old_std[6]))

    def test_capability_opt_in_leaves_default_policy_and_ordinary_ppo_unchanged(self):
        class LegacyPolicy(Policy):
            supports_latent_ppo = False
        reference = make_agent(PPO, policy_cls=LegacyPolicy)
        actual = make_agent(policy_cls=LegacyPolicy)
        self.assertFalse(actual._latent_replay)
        self.assertEqual(len(actual._tensors_names), 6)
        self.assertNotIn("latent_actions", actual.memory.tensors)
        with torch.no_grad():
            _, _, outputs = actual.policy.act({"states": torch.zeros(8, 4)})
        self.assertNotIn("latent_actions", outputs)
        torch.manual_seed(34)
        reference._update(3, 100)
        torch.manual_seed(34)
        actual._update(3, 100)
        self.assert_nested_close(self.snapshot(actual), self.snapshot(reference))

    def test_latent_warmup_and_missing_collection_sample_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "random_timesteps=0"):
            make_agent(random_timesteps=1)
        agent = make_agent()
        with self.assertRaisesRegex(ValueError, "must follow act"):
            agent.record_transition(torch.zeros(8, 4), torch.zeros(8, 9), torch.zeros(8, 1),
                torch.zeros(8, 4), torch.zeros(8, 1, dtype=torch.bool),
                torch.zeros(8, 1, dtype=torch.bool), {}, 4, 100)

    def test_secondary_memory_single_write_reward_shaping_and_timeout_bootstrap(self):
        primary, secondary = (RandomMemory(3, 8, device="cpu") for _ in range(2))
        for name, size, dtype in (("states", 4, torch.float32), ("actions", 9, torch.float32),
            ("rewards", 1, torch.float32), ("terminated", 1, torch.bool),
            ("truncated", 1, torch.bool), ("log_prob", 1, torch.float32), ("values", 1, torch.float32)):
            secondary.create_tensor(name=name, size=size, dtype=dtype)
        agent = FinitePPO(models={"policy": Policy(), "value": Value()}, memory=[primary, secondary],
            observation_space=4, action_space=9, device="cpu", cfg={"rollouts": 3,
                "time_limit_bootstrap": True, "discount_factor": .9,
                "rewards_shaper": lambda reward, timestep, timesteps: reward * 2,
                "experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        agent.init()
        with torch.no_grad():
            for step in range(3):
                states = torch.full((8, 4), float(step))
                actions, _, output = agent.act(states, step, 100)
                values = agent.value.act({"states": states})[0]
                truncated = torch.zeros(8, 1, dtype=torch.bool)
                truncated[step] = True
                rewards = torch.ones(8, 1)
                expected = rewards * 2 + .9 * values * truncated
                agent.record_transition(states, actions, rewards, states + 1,
                    torch.zeros_like(truncated), truncated, {}, step, 100)
                for memory in (primary, secondary):
                    self.assertEqual(memory.memory_index, (step + 1) % 3)
                    torch.testing.assert_close(memory.get_tensor_by_name("rewards")[step], expected)
                    torch.testing.assert_close(memory.get_tensor_by_name("latent_actions")[step],
                                               output["latent_actions"], rtol=0, atol=0)
                    torch.testing.assert_close(memory.get_tensor_by_name("log_prob")[step],
                                               output["latent_log_prob"], rtol=0, atol=0)
                self.assertIsNone(agent._current_latent_actions)
        for name in ("states", "actions", "latent_actions", "log_prob", "rewards", "values"):
            torch.testing.assert_close(primary.get_tensor_by_name(name),
                                       secondary.get_tensor_by_name(name), rtol=0, atol=0)

    def test_initial_bad_parameters_or_adam_raise_instead_of_silent_recovery(self):
        for corrupt_parameters in (False, True):
            with self.subTest(corrupt_parameters=corrupt_parameters):
                agent = make_agent()
                agent._update(3, 100)
                parameter = next(agent.policy.parameters())
                with torch.no_grad():
                    if corrupt_parameters:
                        parameter.fill_(float("nan"))
                    else:
                        agent.optimizer.state[parameter]["exp_avg_sq"].fill_(float("inf"))
                with self.assertRaisesRegex(PPONumericalError, "entered with nonfinite"):
                    agent._update(6, 100)

    def test_invalid_input_is_rejected_before_state_scaler_statistics_change(self):
        agent = make_agent(preprocess=True)
        before = self.snapshot(agent)
        scaler_before = deepcopy(agent._state_preprocessor.state_dict())
        agent.memory.get_tensor_by_name("states")[0, 0, 0] = float("nan")
        agent._update(3, 100)
        self.assert_nested_close(self.snapshot(agent), before, exact=True)
        self.assert_nested_close(agent._state_preprocessor.state_dict(), scaler_before, exact=True)
        self.assertEqual(agent.last_ppo_numerics["reason_counts"], {"nonfinite_minibatch_input": 1})

    def test_nonfinite_preprocessor_update_restores_previous_scaler_state(self):
        for name in ("_state_preprocessor", "_value_preprocessor"):
            with self.subTest(preprocessor=name):
                agent = make_agent(preprocess=True)
                before = self.snapshot(agent)
                module = getattr(agent, name)
                saved = deepcopy(module.state_dict())
                forward = module.forward

                def corrupt_statistics(x, train=False, **kwargs):
                    result = forward(x, train=train, **kwargs)
                    if train:
                        module.running_variance.fill_(float("inf"))
                    return result

                with patch.object(module, "forward", corrupt_statistics):
                    agent._update(3, 100)
                self.assert_nested_close(self.snapshot(agent), before, exact=True)
                self.assert_nested_close(module.state_dict(), saved, exact=True)
                reason = "nonfinite_" + name.lstrip("_").replace("preprocessor", "preprocessing")
                self.assertEqual(agent.last_ppo_numerics["reason_counts"], {reason: 1})


if __name__ == "__main__":
    unittest.main()
