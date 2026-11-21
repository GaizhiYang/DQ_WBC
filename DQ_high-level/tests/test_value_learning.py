"""T2 value learning: critic reward state, potential shaping, command schedule, PPO flags.

Every switch defaults to the original behavior. Runs on CPU without Isaac Gym,
through the production methods (AST extraction, as in the other teacher tests).
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

# Module imports keep these modules' test cases from being collected again here.
import test_critic_time_obs as critic_time
from test_sgs_env import NAMESPACE, method_node, source_method
from test_teacher_gfm import Policy, Value
from modules.predictattention import teacher_gfm_kwargs
from utils import config as config_utils


ROOT = Path(__file__).resolve().parents[1]
ACTIONS, GRASPS, FUTURE, TEMPORAL = 9, 180, 13, 2


def pick_constant(name):
    tree = ast.parse((ROOT / "envs" / "b1z1_pickmulti.py").read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == name for target in node.targets))


STATE_NAMES = pick_constant("CRITIC_REWARD_STATE_NAMES")
WIDTH = len(STATE_NAMES)
CONSTANTS = {name: pick_constant(name) for name in
             ("CRITIC_REWARD_STATE_NAMES", "SHAPING_SLOPE_PER_M", "APPROACH_POTENTIAL_RANGE_M")}


def pick_method(name):
    """A B1Z1PickMulti method that may read the module constants above."""
    namespace = dict(NAMESPACE, **CONSTANTS)
    module = ast.Module(body=[method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "envs" / "b1z1_pickmulti.py"), "exec"),
         namespace)
    return namespace[name]


COMMAND_FACTOR = source_method("reward_vec_task.py", "RewardVecTask", "_command_reward_factor")


class ConfigurationTests(unittest.TestCase):
    def test_defaults_mean_the_original_rewards_and_repository_yaml_keeps_them(self):
        defaults = config_utils.REWARD_OPTION_DEFAULTS
        self.assertEqual(config_utils.validate_reward_options({}), defaults)
        cfg = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertEqual({name: cfg["reward"][name] for name in defaults}, defaults)
        self.assertIs(cfg["env"]["criticRewardStateObs"], False)

    def test_invalid_options_are_rejected(self):
        for bad in ({"commandRewardSchedule": "late"}, {"commandRewardRampFraction": 0},
                    {"commandRewardRampFraction": 1.5}, {"commandRewardRampFraction": True},
                    {"approachShaping": "tanh"}, {"shapingGamma": 0}, {"shapingGamma": 1.2},
                    {"approachShaping": "potential", "only_positive_rewards": True}):
            with self.subTest(options=bad), self.assertRaises(ValueError):
                config_utils.validate_reward_options(bad)

    def test_cli_overrides_yaml_and_absent_cli_keeps_yaml(self):
        parse = config_utils.get_params
        cfg = {"reward": {"approachShaping": "potential", "commandRewardSchedule": "ramp"}}
        options = config_utils.apply_reward_options(cfg, parse([]))
        self.assertEqual((options["approachShaping"], options["commandRewardSchedule"]),
                         ("potential", "ramp"))
        options = config_utils.apply_reward_options(
            cfg, parse(["--approach_shaping", "record", "--command_reward_schedule", "constant"]))
        self.assertEqual((cfg["reward"]["approachShaping"], cfg["reward"]["commandRewardSchedule"]),
                         ("record", "constant"))
        self.assertEqual(options["shapingGamma"], 0.99)

    def test_new_flags_default_to_none_and_last_flag_wins(self):
        parse = config_utils.get_params
        args = parse([])
        for name in ("critic_reward_state", "approach_shaping", "command_reward_schedule",
                     "value_clip", "gae_lambda"):
            self.assertIsNone(getattr(args, name), name)
        self.assertTrue(parse(["--critic_reward_state"]).critic_reward_state)
        self.assertFalse(parse(["--critic_reward_state", "--no_critic_reward_state"]).critic_reward_state)
        self.assertFalse(parse(["--no_value_clip"]).value_clip)
        self.assertEqual(parse(["--gae_lambda", "0.9"]).gae_lambda, 0.9)

    def test_eval_and_temporal_checkpoints_restore_the_layout_flag(self):
        source = (ROOT / "train_multistate_DQ_teacher.py").read_text()
        restore = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                       and node.name == "restore_eval_arguments")
        self.assertIn("critic_reward_state",
                      {node.value for node in ast.walk(restore) if isinstance(node, ast.Constant)})
        tree = ast.parse((ROOT / "utils" / "tsil_checkpoint.py").read_text())
        names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(getattr(t, "id", None) == "OBSERVATION_ARGUMENTS" for t in node.targets))
        self.assertIn("critic_reward_state", names)


class LayoutTests(unittest.TestCase):
    def setup_task(self, state=None, time=True, last_commands=False, prediction=True):
        task = critic_time.task_with("_setup_obs_and_action_info", base=critic_time.SetupBase)
        type(task)._setup_obs_and_action_info.__globals__.update(CONSTANTS)
        task.no_feature = False
        task.num_features = 1024
        env = {"noVelocityObs": True, "lastCommands": last_commands, "tsil": {"enabled": False},
               "posePrediction": {"enabled": prediction, "horizons": [.6, .9]}, "criticTimeObs": time}
        if state is not None:
            env["criticRewardStateObs"] = state
        task.cfg = {"env": env}
        task._setup_obs_and_action_info()
        return task

    def test_missing_or_disabled_flag_keeps_the_previous_layout(self):
        reference = self.setup_task()
        for flag in (None, False):
            with self.subTest(flag=flag):
                task = self.setup_task(flag)
                self.assertEqual(task.num_critic_state_obs, 0)
                self.assertIsNone(task.critic_state_obs_index)
                self.assertEqual(task.dimensions, reference.dimensions)

    def test_enabled_block_sits_before_the_time_channel_without_moving_it(self):
        for time in (True, False):
            with self.subTest(time=time):
                reference, task = self.setup_task(time=time), self.setup_task(True, time=time)
                self.assertEqual(task.num_critic_state_obs, WIDTH)
                self.assertEqual(task.critic_state_obs_index,
                                 -(ACTIONS + GRASPS + FUTURE + int(time) + WIDTH))
                self.assertEqual(task.critic_time_obs_index, reference.critic_time_obs_index)
                for key in ("removed_dim", "num_obs"):
                    self.assertEqual(task.dimensions[key], reference.dimensions[key] + WIDTH)

    def test_invalid_flag_or_last_commands_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "criticRewardStateObs"):
            self.setup_task(1)
        # Without forecasts and the time channel, only the reward-state check applies.
        with self.assertRaisesRegex(ValueError, "Critic reward-state.*lastCommands"):
            self.setup_task(True, time=False, last_commands=True, prediction=False)

    def test_observation_writes_the_block_at_the_published_index(self):
        task = critic_time.EnvironmentLayoutTests.observation_task([0, 50])
        type(task)._compute_observations.__globals__.update(CONSTANTS)
        task.num_critic_state_obs = WIDTH
        task.obs_buf = torch.zeros(2, 1286 + 1 + WIDTH)
        values = torch.arange(WIDTH, dtype=torch.float) + 200.
        task._critic_reward_state = lambda ids: values.expand(len(ids), -1)
        task._compute_observations(torch.arange(2))
        start = -(ACTIONS + GRASPS + TEMPORAL + FUTURE + 1 + WIDTH)
        torch.testing.assert_close(task.obs_buf[:, start:start + WIDTH], values.expand(2, -1))
        torch.testing.assert_close(task.obs_buf[:, start - 1], torch.full((2,), 57.))  # robot state
        torch.testing.assert_close(task.obs_buf[:, -205], torch.tensor([0., .5]))  # time unchanged
        torch.testing.assert_close(task.obs_buf[:, -204:-191], task.ekf.expand(2, -1))


class RewardStateTests(unittest.TestCase):
    def task(self, schedule="constant"):
        task = SimpleNamespace(
            closest_dist=torch.tensor([.5, .2]), curr_dist=torch.tensor([.8, .1]),
            highest_object=torch.tensor([-1., .10]), curr_height=torch.tensor([0., .04]),
            pick_counter=torch.tensor([0, 5]), best_pick_counter=torch.tensor([3, 5]), hold_steps=10,
            lifted_object=torch.tensor([False, True]),
            cfg={"reward": {"commandRewardSchedule": schedule}}, global_step_counter=0,
            train_reward_strict=10, total_timesteps=20)
        task._command_reward_factor = lambda: COMMAND_FACTOR(task)
        return task

    def test_channels_follow_the_published_names(self):
        task = self.task()
        state = pick_method("_critic_reward_state")(task, torch.arange(2))
        self.assertEqual(state.shape, (2, WIDTH))
        expected = {
            "closest_dist": [.5, .2], "approach_gap": [.3, 0.],  # 0: curr is a new record
            "lift_gap": [0., .06],  # -1 means no height record yet
            "hold_progress": [0., .5], "best_hold_progress": [.3, .5], "lifted": [0., 1.],
            "command_reward_factor": [1., 1.]}
        self.assertEqual(tuple(expected), STATE_NAMES)
        for column, name in enumerate(STATE_NAMES):
            with self.subTest(channel=name):
                torch.testing.assert_close(state[:, column], torch.tensor(expected[name]))

    def test_command_factor_tracks_the_schedule(self):
        task = self.task("step")
        state = pick_method("_critic_reward_state")
        torch.testing.assert_close(state(task, torch.arange(2))[:, -1], torch.zeros(2))
        task.global_step_counter = 10
        torch.testing.assert_close(state(task, torch.arange(2))[:, -1], torch.ones(2))


class CommandScheduleTests(unittest.TestCase):
    def factor(self, schedule, step, fraction=None):
        reward = {} if schedule is None else {"commandRewardSchedule": schedule}
        if fraction is not None:
            reward["commandRewardRampFraction"] = fraction
        env = SimpleNamespace(cfg={"reward": reward}, global_step_counter=step,
                              train_reward_strict=50, total_timesteps=100)
        return COMMAND_FACTOR(env)

    def test_step_keeps_the_original_half_training_switch(self):
        for schedule in (None, "step"):
            self.assertEqual([self.factor(schedule, step) for step in (0, 49, 50, 99)], [0., 0., 1., 1.])

    def test_constant_and_ramp(self):
        self.assertEqual([self.factor("constant", step) for step in (0, 99)], [1., 1.])
        self.assertEqual([self.factor("ramp", step) for step in (0, 10, 20, 90)], [0., .5, 1., 1.])
        self.assertEqual(self.factor("ramp", 25, fraction=.5), .5)

    def test_tsil_requires_the_original_schedule(self):
        env = SimpleNamespace(tsil_enabled=True, cfg={"reward": {"only_positive_rewards": False}},
                              reward_options={"commandRewardSchedule": "ramp"})
        with self.assertRaisesRegex(ValueError, "commandRewardSchedule"):
            pick_method("_init_tsil")(env)


class PotentialShapingTests(unittest.TestCase):
    gamma = .99

    def env(self, distance, height):
        env = SimpleNamespace(curr_dist=torch.tensor([distance]), curr_height=torch.tensor([height]),
                              lifted_success_threshold=.2, reset_buf=torch.zeros(1, dtype=torch.long),
                              reward_options={"shapingGamma": self.gamma}, approach_shaping="potential",
                              shaping_potential_prev=torch.zeros(1, 2))
        for name in ("_shaping_potential", "_reset_shaping_potential", "_potential_difference",
                     "_reward_approaching", "_reward_lifting"):
            method = pick_method(name)
            setattr(env, name, lambda *args, method=method: method(env, *args))
        env._reset_shaping_potential(torch.tensor([0]))
        return env

    def test_potentials_are_bounded_and_non_negative(self):
        env = self.env(5., .5)
        torch.testing.assert_close(env._shaping_potential(), torch.tensor([[0., 2.]]))
        env.curr_dist[:], env.curr_height[:] = 0., -.1
        torch.testing.assert_close(env._shaping_potential(), torch.tensor([[30., 0.]]))

    def test_step_reward_is_the_discounted_potential_difference(self):
        env = self.env(2., 0.)
        env.curr_dist[:], env.curr_height[:] = 1.9, .05
        approaching, _ = env._reward_approaching()
        lifting, _ = env._reward_lifting()
        torch.testing.assert_close(approaching, torch.tensor([self.gamma * 11. - 10.]))
        torch.testing.assert_close(lifting, torch.tensor([self.gamma * .5]))
        env.curr_dist[:] = 2.1  # the object moved away: negative shaping
        self.assertLess(float(env._reward_approaching()[0]), 0.)

    def test_discounted_episode_shaping_telescopes_to_minus_initial_potential(self):
        distances, heights = [1.8, 1.2, .7, .3, .05, .02], [0., 0., .01, .05, .15, .2]
        env = self.env(2., 0.)
        initial = env._shaping_potential().clone()
        total = torch.zeros(1, 2)
        for step, (distance, height) in enumerate(zip(distances, heights)):
            env.curr_dist[:], env.curr_height[:] = distance, height
            env.reset_buf[:] = int(step == len(distances) - 1)  # true terminal on the last step
            rewards = torch.stack((env._reward_approaching()[0], env._reward_lifting()[0]), dim=-1)
            total += self.gamma ** step * rewards
        torch.testing.assert_close(total, -initial)


class TeacherModelTests(unittest.TestCase):
    features, body = 16, 24
    time_index = -(ACTIONS + GRASPS + TEMPORAL + FUTURE + 1)
    state_index = time_index - WIDTH

    def parts(self, state_values=None, time_value=None, batch=3):
        generator = torch.Generator().manual_seed(7)
        features, body = (torch.randn(batch, width, generator=generator) * .3
                          for width in (self.features, self.body))
        tail = [torch.randn(batch, width, generator=generator) * .3
                for width in (FUTURE, TEMPORAL, GRASPS, ACTIONS)]
        middle = []
        if state_values is not None:
            middle.append(torch.full((batch, WIDTH), float(state_values)))
        if time_value is not None:
            middle.append(torch.full((batch, 1), float(time_value)))
        return torch.cat([features, body, *middle, *tail], dim=-1)

    def model(self, cls, width, **kwargs):
        torch.manual_seed(5)
        return cls(width, ACTIONS, "cpu", self.features, 128, None, None, None,
                   num_future_pose_obs=FUTURE, num_tsil_obs=TEMPORAL, **kwargs)

    def test_actor_ignores_both_critic_only_blocks(self):
        reference = self.model(Policy, self.parts().shape[-1])
        cases = {"state and time": dict(time=0., kwargs=dict(
                     critic_time_index=self.time_index, critic_state_index=self.state_index,
                     critic_state_width=WIDTH)),
                 "state only": dict(time=None, kwargs=dict(
                     critic_state_index=self.state_index + 1, critic_state_width=WIDTH))}
        expected = reference.compute({"states": self.parts()}, role="policy")[0]
        for name, case in cases.items():
            with self.subTest(case=name):
                width = self.parts(0., case["time"]).shape[-1]
                policy = self.model(Policy, width, **case["kwargs"])
                self.assertEqual(policy.net[0].in_features, reference.net[0].in_features)
                policy.load_state_dict(reference.state_dict())
                for value in (0., 3., -7.):
                    time = None if case["time"] is None else value
                    actions = policy.compute({"states": self.parts(value, time)}, role="policy")[0]
                    torch.testing.assert_close(actions, expected)

    def test_critic_receives_the_block(self):
        reference = self.model(Value, self.parts().shape[-1])
        value = self.model(Value, self.parts(0., 0.).shape[-1], critic_time_index=self.time_index,
                           critic_state_index=self.state_index, critic_state_width=WIDTH)
        self.assertEqual(value.net[0].in_features, reference.net[0].in_features + WIDTH + 1)

    def test_invalid_or_overlapping_blocks_are_rejected(self):
        width = self.parts(0., 0.).shape[-1]
        for kwargs in (dict(critic_state_index=self.state_index, critic_state_width=0),
                       # Starts one channel before the grasps: runs past the state block.
                       dict(critic_state_index=-(ACTIONS + GRASPS + 1), critic_state_width=WIDTH),
                       dict(critic_time_index=self.time_index, critic_state_index=self.state_index + 1,
                            critic_state_width=WIDTH)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(Policy, width, **kwargs)

    def test_layout_kwargs_publish_the_block(self):
        env = SimpleNamespace(cfg={"env": {}}, critic_time_obs_index=self.time_index,
                              critic_state_obs_index=self.state_index, num_critic_state_obs=WIDTH)
        kwargs = teacher_gfm_kwargs(env)
        self.assertEqual((kwargs["critic_state_index"], kwargs["critic_state_width"]),
                         (self.state_index, WIDTH))
        env.critic_state_obs_index = None
        self.assertNotIn("critic_state_index", teacher_gfm_kwargs(env))


if __name__ == "__main__":
    unittest.main()
