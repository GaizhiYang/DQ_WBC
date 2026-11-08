"""Source action coordinates/initialization shared by simulator and BPTT (CPU)."""

import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skrl.models.torch import Model
from modules.predictattention import PredictAttentionSelector
from test_world_model_action_bounds import production_controller
from test_world_model_env import production_function, reference_namespace
from test_world_model_fixed8_ppo import make_fixed8_agent
from utils.gripper_policy import ContinuousGripperGaussianMixin
from utils.world_model.action_contract import (action_rate_cost, high_level_command_components,
                                               initialize_actor_output, validate_action_contract)


def teacher_policy(**options):
    path = ROOT / "train_multistate_DQ_teacher.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Policy")
    namespace = dict(torch=torch, nn=torch.nn, Model=Model,
        ContinuousGripperGaussianMixin=ContinuousGripperGaussianMixin,
        PredictAttentionSelector=PredictAttentionSelector, initialize_actor_output=initialize_actor_output,
        cprint=lambda *_: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Policy"](1284, 9, "cpu", 1024, 128, None, None, None, **options)


class ActionContractTests(unittest.TestCase):
    def test_source_output_initialization_and_legacy_defaults(self):
        policy = teacher_policy(initial_action_std=.1, actor_last_gain=.01)
        output = policy.net[-1]
        torch.testing.assert_close(output.weight @ output.weight.T, torch.eye(9) * .01 ** 2,
                                   rtol=2e-6, atol=1e-10)
        torch.testing.assert_close(output.bias, torch.zeros(9), rtol=0, atol=0)
        torch.testing.assert_close(policy.log_std_parameter.exp(), torch.full((9,), .1))
        torch.manual_seed(34)
        first = teacher_policy()
        torch.manual_seed(34)
        explicit_none = teacher_policy(initial_action_std=None, actor_last_gain=None)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, explicit_none.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(first.log_std_parameter, torch.zeros(9), rtol=0, atol=0)
        # Restoring learned weights/std does not trigger fresh initialization.
        with torch.no_grad():
            policy.net[-1].bias.fill_(.7)
            policy.log_std_parameter.fill_(-1.3)
        restored = teacher_policy(initial_action_std=.1, actor_last_gain=.01)
        restored.load_state_dict(policy.state_dict())
        torch.testing.assert_close(restored.net[-1].bias, policy.net[-1].bias, rtol=0, atol=0)
        torch.testing.assert_close(restored.log_std_parameter, policy.log_std_parameter, rtol=0, atol=0)

    def test_normalized_scale_has_correct_interior_derivatives_and_original_physical_limits(self):
        action = torch.full((4, 9), .1, requires_grad=True)
        parts = high_level_command_components(action, .45, action_coordinates="normalized_commands")
        for actual, expected in zip(parts, (.002, .006, .045, .06)):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))
        sum(part.sum() for part in parts).backward()
        expected = torch.tensor([.02] * 3 + [.06] * 3 + [0., .45, .6]).expand_as(action)
        torch.testing.assert_close(action.grad, expected, rtol=0, atol=0)
        saturated = torch.full((2, 9), 100., requires_grad=True)
        parts = high_level_command_components(saturated, .4, action_coordinates="normalized_commands")
        for actual, expected in zip(parts, (.02, .06, .4, .6)):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))
        sum(part.sum() for part in parts).backward()
        torch.testing.assert_close(saturated.grad, torch.zeros_like(saturated), rtol=0, atol=0)
        physical = high_level_command_components(action.detach(), .45)
        torch.testing.assert_close(physical[0], torch.full((4, 3), .02))
        torch.testing.assert_close(physical[1], torch.full((4, 3), .06))

    def test_real_and_imagined_mapping_preserve_raw_history_and_match_all_commands(self):
        agent = make_fixed8_agent("cpu")
        adapter = agent.wm_adapter
        adapter.action_coordinates = "normalized_commands"
        adapter.clip_actions = float("inf")
        state, context = adapter.capture()
        action = torch.tensor([
            [.1, -.2, .3, -.1, .2, -.3, .2, .1, -.2],
            [2., -5., 3., -4., 6., -8., .8, 9., -10.],
            [-.1, .2, -.3, .1, -.2, .3, .5, -.4, .7],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
        ]).repeat(3, 1)
        original = action.clone()
        for stop_pick in (False, True):
            for small_commands in (False, True):
                with self.subTest(stop_pick=stop_pick, small_commands=small_commands):
                    env = deepcopy(agent.wm_env)
                    env.action_coordinates = "normalized_commands"
                    env.stop_pick, adapter.stop_pick = stop_pick, stop_pick
                    env.cfg["env"]["smallValueSetZero"] = small_commands
                    adapter.small_commands = small_commands
                    control = production_controller(env)
                    control(env, action)
                    imagined = adapter.prepare_high(state, action)
                    for field, attr in (("goal_pos", "curr_ee_goal_cart"),
                                        ("goal_rpy", "curr_ee_goal_orn_rpy"),
                                        ("commands", "commands"), ("prev_action", "actions"),
                                        ("held_gripper_target", "gripper_dof_pos")):
                        torch.testing.assert_close(adapter.get(imagined, field), getattr(env, attr),
                                                   rtol=0, atol=1e-6)
                    torch.testing.assert_close(env.actions, original, rtol=0, atol=0)
                    torch.testing.assert_close(env.action_history_buf[:, -1], original, rtol=0, atol=0)
                    self.assertGreater(float(env.actions.abs().max()), 1.)
        torch.testing.assert_close(action, original, rtol=0, atol=0)

    def test_real_and_imagined_squared_rate_use_all_stored_coordinates_and_same_weight(self):
        agent = make_fixed8_agent("cpu")
        adapter = agent.wm_adapter
        adapter.action_coordinates, adapter.action_rate_mode = "normalized_commands", "squared_all"
        adapter.clip_actions = float("inf")
        previous, context = adapter.capture()
        action = torch.linspace(-2., 2., len(previous) * 9).reshape(len(previous), 9)
        action[:, 6] = .7
        before = action * .5
        previous = adapter.replace(previous, prev_action=before)
        state = adapter.prepare_high(previous, action)
        env = SimpleNamespace(actions=action, last_actions=before, action_rate_mode="squared_all")
        function = production_function(ROOT / "envs/reward_vec_task.py", "_reward_action_rate",
                                       reference_namespace(), "RewardVecTask")
        real, diagnostic = function(env)
        expected = (action - before).square().sum(-1)
        torch.testing.assert_close(real, expected, rtol=0, atol=0)
        torch.testing.assert_close(diagnostic, expected, rtol=0, atol=0)
        adapter.reward_scales = {"action_rate": -.002}
        torch.testing.assert_close(adapter.reward(previous, state, action, context), real * -.002,
                                   rtol=0, atol=0)
        env.action_rate_mode = "base_l2"
        legacy = function(env)[0]
        torch.testing.assert_close(legacy, (action[:, 7:9] - before[:, 7:9]).norm(dim=-1))
        # A pure EE-coordinate change has a nonzero source rate penalty, whereas
        # the legacy base-only penalty deliberately ignores it.
        only_arm = torch.zeros(2, 9)
        only_arm[:, 0] = .5
        torch.testing.assert_close(action_rate_cost(only_arm, torch.zeros_like(only_arm), "squared_all"),
                                   torch.full((2,), .25))
        torch.testing.assert_close(action_rate_cost(only_arm, torch.zeros_like(only_arm)), torch.zeros(2))

    def test_fresh_gaussian_remains_stochastic_and_mean_physical_gradient_excludes_std(self):
        policy = teacher_policy(initial_action_std=.1, actor_last_gain=.01)
        states = torch.randn(32, 1284) * .02
        with torch.no_grad():
            first, _, outputs = policy.act({"states": states, "latent_ppo": True})
            second, _, _ = policy.act({"states": states, "latent_ppo": True})
        self.assertFalse(torch.equal(first, second))
        mean, _, _ = policy.compute({"states": states}, role="policy")
        mean.retain_grad()
        mapped = policy.mean_to_action(mean)
        physical = high_level_command_components(mapped, .45, action_coordinates="normalized_commands")
        loss = sum(part.sum() for part in physical) + mapped[:, 6].sum()
        policy.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNone(policy.log_std_parameter.grad)
        self.assertTrue(bool((mean.grad.abs() > 0).all()))
        self.assertGreater(float(policy.net[-1].weight.grad.abs().sum()), 0.)
        # The actual PPO score still differentiates all Gaussian standard deviations.
        policy.zero_grad(set_to_none=True)
        _, _, replay = policy.act({"states": states, "taken_actions": first,
            "taken_latent_actions": outputs["latent_actions"], "latent_ppo": True})
        replay["latent_log_prob"].sum().backward()
        self.assertTrue(bool(torch.isfinite(policy.log_std_parameter.grad).all()))
        self.assertTrue(bool((policy.log_std_parameter.grad.abs() > 0).all()))

    def test_metadata_distinguishes_action_and_reward_contracts(self):
        adapter = make_fixed8_agent("cpu").wm_adapter
        original = adapter.metadata()
        adapter.action_coordinates, adapter.action_rate_mode = "normalized_commands", "squared_all"
        changed = adapter.metadata()
        self.assertEqual(changed["action_coordinates"], "normalized_commands")
        self.assertEqual(changed["action_rate_mode"], "squared_all")
        self.assertNotEqual(original, changed)

    def test_invalid_or_double_transformed_contracts_fail_explicitly(self):
        for coords, rate, tanh in (("unknown", "base_l2", False),
                                  ("physical", "unknown", False),
                                  ("normalized_commands", "squared_all", True)):
            with self.assertRaises(ValueError):
                validate_action_contract(coords, rate, tanh)


if __name__ == "__main__":
    unittest.main()
