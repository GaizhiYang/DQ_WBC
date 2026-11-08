"""Compare shared mathematics against the actual local SuperMimic source.

Only small pure definitions are loaded through AST, so importing these tests
does not import MuJoCo, IsaacGym or Warp. Set SUPERMIMIC_SOURCE_ROOT when the
reference checkout is elsewhere. Fixed critic-target ownership is additionally
covered by test_world_model_imagined_critic's multi-epoch regression test.
"""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_world_model_fixed8_ppo import make_fixed8_agent
from utils.world_model.dynamics import MLP
from utils.world_model.gradients import freeze_parameters
from utils.world_model.imagined_critic import ImaginedCritic, lambda_returns
from utils.world_model.imagination import ImaginedRollout, imagination_objective


SOURCE_ROOT = Path(os.environ.get(
    "SUPERMIMIC_SOURCE_ROOT", str(Path(__file__).resolve().parents[3] / "SuperMimic-Reproduced")))


def source_definitions(relative_path, names):
    """Execute the reference definitions verbatim, without its simulator imports."""
    path = SOURCE_ROOT / "src" / "supermimic" / relative_path
    if not path.is_file():
        raise unittest.SkipTest("Local SuperMimic reference source unavailable: {}".format(path))
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in source.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("Reference definitions changed in {}".format(path))
    # Preserve annotations without resolving the reference rollout dataclass.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {"torch": torch, "nn": nn}
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class SuperMimicSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objective = source_definitions("imagination/objective.py", ("policy_objective", "gae_targets"))
        cls.networks = source_definitions("world_model/dynamics.py", ("MLP",))

    def test_soft_survival_objective_and_all_input_gradients_match_source(self):
        generator = torch.Generator().manual_seed(43)
        for horizon in (1, 4, 32):
            for gamma, use_tail in ((.99, True), (1., True), (.9, False)):
                with self.subTest(horizon=horizon, gamma=gamma, tail=use_tail):
                    rewards = torch.randn(horizon, 5, generator=generator, dtype=torch.float64,
                                          requires_grad=True)
                    logits = torch.randn(horizon, 5, generator=generator, dtype=torch.float64,
                                         requires_grad=True)
                    tail = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)
                    termination = logits.sigmoid()
                    survival = torch.cat((torch.ones_like(termination[:1]),
                                          (1. - termination).cumprod(0)), 0)
                    reference = SimpleNamespace(rewards=rewards, survival=survival)
                    current = ImaginedRollout(
                        states=rewards.new_zeros(horizon + 1, 5, 1),
                        actions=rewards.new_zeros(horizon, 5, 9), rewards=rewards,
                        termination=termination, survival=survival,
                        timeouts=torch.zeros_like(rewards), timeout_values=torch.zeros_like(rewards),
                        tail_values=tail if use_tail else torch.zeros_like(tail))
                    expected = self.objective.policy_objective(reference, tail if use_tail else None, gamma)
                    actual = imagination_objective(current, gamma)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    inputs = (rewards, logits, tail) if use_tail else (rewards, logits)
                    expected_gradients = torch.autograd.grad(expected, inputs, retain_graph=True)
                    actual_gradients = torch.autograd.grad(actual, inputs)
                    for got, wanted in zip(actual_gradients, expected_gradients):
                        torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    def test_fixed_lambda_targets_match_source_for_soft_nonzero_termination(self):
        generator = torch.Generator().manual_seed(44)
        rewards = torch.randn(32, 7, generator=generator, dtype=torch.float64, requires_grad=True)
        termination = torch.rand(32, 7, generator=generator, dtype=torch.float64, requires_grad=True)
        values = torch.randn(33, 7, generator=generator, dtype=torch.float64, requires_grad=True)
        for gamma in (.9, .99, 1.):
            for lam in (0., .95, 1.):
                with self.subTest(gamma=gamma, lam=lam):
                    expected = self.objective.gae_targets(rewards, termination, values, gamma, lam)
                    actual = lambda_returns(rewards, termination, torch.zeros_like(rewards), values,
                                            gamma=gamma, lam=lam)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertFalse(actual.requires_grad)
                    self.assertFalse(expected.requires_grad)
        self.assertIsNone(rewards.grad)
        self.assertIsNone(termination.grad)
        self.assertIsNone(values.grad)

    def test_mlp_default_and_zero_residual_initialization_match_source(self):
        for zero_last in (False, True):
            with self.subTest(zero_last=zero_last), torch.random.fork_rng(devices=[]):
                torch.manual_seed(45)
                reference = self.networks.MLP(9, (17, 11), 5, zero_last=zero_last).double()
                torch.manual_seed(45)
                current = MLP(9, (17, 11), 5, zero_last=zero_last).double()
                for got, expected in zip(current.parameters(), reference.parameters()):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                inputs = torch.linspace(-1., 1., 36, dtype=torch.float64).reshape(4, 9).requires_grad_(True)
                expected, actual = reference(inputs), current(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_gradient, = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
                gradient, = torch.autograd.grad(actual.sum(), inputs)
                torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


class CompactCriticGradientContractTests(unittest.TestCase):
    def test_same_physics_different_raw_history_has_same_value_and_state_jacobian(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(46)
            spec = make_fixed8_agent("cpu").wm_adapter
            critic = ImaginedCritic(spec.critic_feature_dim, hidden=(17, 11))
        state, context = spec.capture()
        excluded = ("prev_action", "low_history", "low_actions", "ekf_poses", "ekf_ages", "ekf_count")
        changed = spec.replace(state, **{name: state[:, spec.slices[name]] + 1e4 for name in excluded})

        def value_and_jacobian(start):
            start = start.detach().requires_grad_(True)
            with freeze_parameters(critic):
                value = critic(spec.critic_features(start, context))
                gradient, = torch.autograd.grad(value.sum(), start)
            return value.detach(), gradient

        expected_value, expected_gradient = value_and_jacobian(state)
        value, gradient = value_and_jacobian(changed)
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        for name in excluded:
            self.assertEqual(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)
        for name in ("robot", "object", "dof_pos", "dof_vel", "goal_pos", "commands"):
            self.assertGreater(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))

    def test_saturated_commands_have_no_direct_history_to_value_gradient(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            agent = make_fixed8_agent("cpu")
            critic = ImaginedCritic(agent.wm_adapter.critic_feature_dim, hidden=(17,))
        spec, model = agent.wm_adapter, agent.wm_candidate
        state, context = spec.capture()
        state = spec.replace(state, goal_pos=state.new_tensor([.3, 0., 0.]).expand(len(state), -1))
        spec.clip_actions = float("inf")  # This regression must not pass due to an outer clip.
        spec.use_tanh = False
        indices = [0, 1, 2, 3, 4, 5, 7, 8]
        for coordinates in ("physical", "normalized_commands"):
            with self.subTest(coordinates=coordinates):
                spec.action_coordinates = coordinates
                interior = state.new_zeros(len(state), 9)
                interior[:, 0] = .5
                _, goal, _, _ = spec._control(state, interior)
                expected_dx = .01 if coordinates == "normalized_commands" else .02
                torch.testing.assert_close(goal[:, 0] - spec.get(state, "goal_pos")[:, 0],
                                           state.new_full((len(state),), expected_dx), rtol=0, atol=2e-8)
                action = state.new_full((len(state), 9), 2.)
                action[:, 6] = .4
                alternative = action.clone()
                alternative[:, indices] = 1000.
                action.requires_grad_(True)
                with freeze_parameters(model, critic, spec.low_actor):
                    predicted = model.step(state, action, context, .16)
                    other = model.step(state, alternative, context, .16)
                    value = critic(spec.critic_features(predicted, context))
                    other_value = critic(spec.critic_features(other, context))
                    torch.testing.assert_close(value, other_value, rtol=0, atol=0)
                    gradient, = torch.autograd.grad(value.sum(), action)
                # Physical limits identify these commands. An unbounded V(a)
                # shortcut would make either the values or gradients differ.
                torch.testing.assert_close(gradient[:, indices], torch.zeros_like(gradient[:, indices]),
                                           rtol=0, atol=0)
                self.assertGreater(float(gradient[:, 6].abs().sum()), 0.)
                self.assertFalse(torch.equal(spec.get(predicted, "prev_action"), spec.get(other, "prev_action")))
                for name in ("robot", "object", "dof_pos", "dof_vel", "goal_pos", "goal_rpy", "commands"):
                    torch.testing.assert_close(spec.get(predicted, name), spec.get(other, name), rtol=0, atol=0)

    def test_frozen_low_controller_and_dynamics_keep_state_and_action_jacobians(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(48)
            agent = make_fixed8_agent("cpu")
            # A trained residual model has nonzero action sensitivity; the
            # deliberately zero initial residual head cannot exercise it.
            with torch.no_grad():
                agent.wm_candidate.dynamics.net[-1].weight.normal_(mean=0., std=.02)
        spec, model = agent.wm_adapter, agent.wm_candidate
        initial, context = spec.capture()
        state = initial.detach().requires_grad_(True)
        with freeze_parameters(model, spec.low_actor):
            prepared, actuator = spec.low_control(state, context)
            actuator.retain_grad()
            predicted = model.low_step(prepared, actuator, context, .02)
            spec.get(predicted, "dof_pos").sum().backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))
        self.assertGreater(float(state.grad[:, spec.slices["low_history"]].abs().sum()), 0.)
        self.assertGreater(float(state.grad[:, spec.slices["dof_pos"]].abs().sum()), 0.)
        self.assertGreater(float(actuator.grad[:, :12].abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in spec.low_actor.parameters()))


if __name__ == "__main__":
    unittest.main()
