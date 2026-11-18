"""Critic-only episode-progress input: environment layout and teacher models.

Timeouts end episodes as true terminals, so the critic needs the elapsed step
count. The actor must ignore the channel. Runs on CPU without Isaac Gym.
"""

import ast
import unittest

import torch
import yaml

from modules.predictattention import teacher_gfm_kwargs
from test_sgs_env import NAMESPACE, method_node
from test_teacher_gfm import ROOT, Policy, Value
from utils import config as config_utils


ACTIONS, GRASPS = 9, 180


def task_with(*methods, base=None):
    """Production B1Z1PickMulti methods on a stand-in base class."""
    node = ast.ClassDef(name="Task", bases=[ast.Name(id="Base", ctx=ast.Load())],
                        keywords=[], decorator_list=[],
                        body=[method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name)
                              for name in methods])
    namespace = dict(NAMESPACE, Base=base or object,
                     quat_conjugate=lambda q: q * torch.tensor([-1., -1., -1., 1.]),
                     quat_to_euler_zyx=lambda q: torch.zeros_like(q[..., :3]),
                     quat_rotate_inverse_batched=NAMESPACE["quat_rotate_inverse"])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 __file__, "exec"), namespace)
    return namespace["Task"]()


class SetupBase:
    def _setup_obs_and_action_info(self, **kwargs):
        self.dimensions = kwargs
        self.cfg["env"]["numActions"] = kwargs["num_action"]


class EnvironmentLayoutTests(unittest.TestCase):
    def setup_task(self, critic_time=None, tsil=False, prediction=True, last_commands=False):
        task = task_with("_setup_obs_and_action_info", base=SetupBase)
        task.no_feature = False
        task.num_features = 1024
        env = {"noVelocityObs": True, "lastCommands": last_commands, "tsil": {"enabled": tsil},
               "posePrediction": {"enabled": prediction, "horizons": [.6, .9]}}
        if critic_time is not None:
            env["criticTimeObs"] = critic_time
        task.cfg = {"env": env}
        task._setup_obs_and_action_info()
        return task

    def test_disabled_or_missing_flag_keeps_previous_layout(self):
        for flag in (None, False):
            with self.subTest(flag=flag):
                task = self.setup_task(flag)
                self.assertEqual(task.num_critic_time_obs, 0)
                self.assertIsNone(task.critic_time_obs_index)
                # Same sizes as the existing SGS+EKF teacher (see test_tsil_env).
                self.assertEqual(task.dimensions["num_obs"] + ACTIONS + 24, 1284)

    def test_enabled_flag_adds_one_teacher_only_channel_before_forecasts(self):
        for tsil in (False, True):
            for prediction in (False, True):
                with self.subTest(tsil=tsil, prediction=prediction):
                    old = self.setup_task(False, tsil=tsil, prediction=prediction).dimensions
                    task = self.setup_task(True, tsil=tsil, prediction=prediction)
                    self.assertEqual(task.dimensions["num_obs"], old["num_obs"] + 1)
                    # removed_dim grows too, so the student state size is unchanged.
                    self.assertEqual(task.dimensions["removed_dim"], old["removed_dim"] + 1)
                    future = 13 if prediction else 0
                    temporal = 2 if tsil else 0
                    self.assertEqual(task.critic_time_obs_index,
                                     -(ACTIONS + GRASPS + temporal + future + 1))
                    if tsil:
                        self.assertEqual(task.tsil_time_indices, (-191, -190))

    def test_flag_must_be_boolean_and_needs_the_feature_layout(self):
        for flag in ("true", 1, 0, []):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "criticTimeObs"):
                self.setup_task(flag)
        with self.assertRaisesRegex(ValueError, "lastCommands"):
            self.setup_task(True, last_commands=True)

    @staticmethod
    def observation_task(progress, enabled=True):
        class Base:
            def _compute_observations(self, ids):
                return torch.arange(64, dtype=torch.float).expand(len(ids), -1)

        num_envs = len(progress)
        task = task_with("_compute_observations", "_grasp_candidate_observations", base=Base)
        task.no_feature = False
        task.cfg = {"env": {}}
        task.num_critic_time_obs = 1 if enabled else 0
        task.max_episode_length = 100
        task.progress_buf = torch.tensor(progress)
        task._robot_root_states = torch.zeros(num_envs, 13)
        task._robot_root_states[:, 6] = 1.
        task.arm_base = torch.zeros(num_envs, 3)
        task.feature_obs = torch.arange(1024, dtype=torch.float).expand(num_envs, -1)
        task.obs_buf = torch.zeros(num_envs, 1286 + task.num_critic_time_obs)
        task.action_history_buf = torch.arange(9, dtype=torch.float).expand(num_envs, 1, -1)
        task.get_grasp_world = lambda pose, ids: (torch.zeros(len(ids), 30, 3),
                                                  torch.zeros(len(ids), 30, 3))
        task.ekf = torch.arange(13, dtype=torch.float)[None] + 100.
        task._future_object_observations = lambda ids, pose: task.ekf.expand(len(ids), -1)
        task._tsil_observations = lambda ids: torch.tensor([[.25, .75]]).expand(len(ids), -1)
        return task

    def test_observation_places_elapsed_fraction_at_the_published_index(self):
        task = self.observation_task([0, 37, 99, 100])
        task._compute_observations(torch.arange(4))
        index = -(ACTIONS + GRASPS + 2 + 13 + 1)
        self.assertEqual(index, -205)
        # 0 right after reset, 1 at the timeout step.
        torch.testing.assert_close(task.obs_buf[:, index], torch.tensor([0., .37, .99, 1.]))
        # Robot state ends just before it; forecast, TSIL and tail keep their positions.
        torch.testing.assert_close(task.obs_buf[:, index - 1], torch.full((4,), 57.))
        torch.testing.assert_close(task.obs_buf[:, -204:-191], task.ekf.expand(4, -1))
        torch.testing.assert_close(task.obs_buf[:, -191:-189], torch.tensor([[.25, .75]]).expand(4, -1))
        torch.testing.assert_close(task.obs_buf[:, -9:], task.action_history_buf[:, -1])

    def test_partial_reset_writes_only_the_selected_environments(self):
        task = self.observation_task([10, 20, 30])
        task._compute_observations(torch.tensor([1]))
        torch.testing.assert_close(task.obs_buf[:, -205], torch.tensor([0., .2, 0.]))

    def test_disabled_channel_leaves_the_previous_observation_layout(self):
        task = self.observation_task([42], enabled=False)
        task._compute_observations(torch.tensor([0]))
        torch.testing.assert_close(task.obs_buf[:, -205], torch.tensor([57.]))
        torch.testing.assert_close(task.obs_buf[:, -204:-191], task.ekf)


class TeacherModelTests(unittest.TestCase):
    features, body, future, temporal = 16, 24, 13, 2

    def state(self, time_value=None, batch=3):
        generator = torch.Generator().manual_seed(7)
        parts = [torch.randn(batch, width, generator=generator) * .3
                 for width in (self.features, self.body)]
        if time_value is not None:
            parts.append(torch.full((batch, 1), float(time_value)))
        parts += [torch.randn(batch, width, generator=generator) * .3
                  for width in (self.future, self.temporal, GRASPS, ACTIONS)]
        return torch.cat(parts, dim=-1)

    def model(self, cls, width, critic_time_index=None):
        torch.manual_seed(5)
        kwargs = {"num_future_pose_obs": self.future, "num_tsil_obs": self.temporal}
        if critic_time_index is not None:
            kwargs["critic_time_index"] = critic_time_index
        return cls(width, ACTIONS, "cpu", self.features, 128, None, None, None, **kwargs)

    @property
    def index(self):
        return -(ACTIONS + GRASPS + self.temporal + self.future + 1)

    def test_actor_matches_a_model_without_the_channel_and_ignores_its_value(self):
        reference = self.model(Policy, self.state().shape[-1])
        policy = self.model(Policy, self.state(0.).shape[-1], self.index)
        self.assertEqual(policy.net[0].in_features, reference.net[0].in_features)
        policy.load_state_dict(reference.state_dict())
        expected = reference.compute({"states": self.state()}, role="policy")[0]
        for time_value in (0., .5, 1.):
            with self.subTest(time=time_value):
                actions = policy.compute({"states": self.state(time_value)}, role="policy")[0]
                torch.testing.assert_close(actions, expected)

    def test_critic_receives_the_channel(self):
        width = self.state(0.).shape[-1]
        value = self.model(Value, width, self.index)
        self.assertEqual(value.net[0].in_features, width - self.features + 128 - GRASPS + 6)
        early = value.compute({"states": self.state(0.)}, role="value")[0]
        late = value.compute({"states": self.state(1.)}, role="value")[0]
        self.assertGreater((early - late).abs().max().item(), 1e-6)
        state = self.state(.5).requires_grad_(True)
        value.compute({"states": state}, role="value")[0].sum().backward()
        self.assertGreater(state.grad[:, self.index].abs().sum().item(), 0.)

    def test_index_outside_the_state_block_is_rejected(self):
        width = self.state(0.).shape[-1]
        for index in (-(ACTIONS + GRASPS), -(width - self.features - 5), 3):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "critic_time_index"):
                self.model(Policy, width, index)


class ConfigurationTests(unittest.TestCase):
    def test_kwargs_publish_index_only_when_the_channel_exists(self):
        cfg = {"env": {"posePrediction": {"gfmFuturePoses": True}}}
        env = type("Env", (), dict(cfg=cfg, num_future_pose_obs=13, num_tsil_obs=0))()
        self.assertEqual(teacher_gfm_kwargs(env), {"num_future_pose_obs": 13, "num_tsil_obs": 0})
        env.critic_time_obs_index = None
        self.assertNotIn("critic_time_index", teacher_gfm_kwargs(env))
        env.critic_time_obs_index = -203
        self.assertEqual(teacher_gfm_kwargs(env)["critic_time_index"], -203)

    def test_cli_defaults_to_yaml_and_last_flag_wins(self):
        self.assertIsNone(config_utils.get_params([]).critic_time_obs)
        self.assertTrue(config_utils.get_params(["--critic_time_obs"]).critic_time_obs)
        self.assertFalse(config_utils.get_params(
            ["--critic_time_obs", "--no_critic_time_obs"]).critic_time_obs)

    def test_repository_yaml_keeps_it_off_and_launcher_turns_it_on(self):
        cfg = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertIs(cfg["env"]["criticTimeObs"], False)
        self.assertIn("--critic_time_obs", (ROOT / "train_teacher_sgs_ekf.sh").read_text())


if __name__ == "__main__":
    unittest.main()
