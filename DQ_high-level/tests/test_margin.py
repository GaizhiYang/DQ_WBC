"""Main line A: time-to-reach T, interception margin, shaping, tokens, dynamic SGS.

No Isaac Gym: core functions run directly, environment hooks run on stand-ins
(the InterceptionMarginMixin is pure torch), and pick-task methods are
extracted from source like the other environment tests.
"""

import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml

import test_critic_time_obs as critic_time
from test_teacher_gfm import Policy, Value
from test_value_learning import CONSTANTS
from utils.margin_task import InterceptionMarginMixin, MARGIN_SCALAR_NAMES, MARGIN_TOKEN_DIM
from modules.predictattention import teacher_gfm_kwargs
from modules import reachability as R
from utils import config as config_utils
from utils import margin_curriculum as MC
from utils.sgs import SGSSampler

ROOT = Path(__file__).resolve().parents[1]
ACTIONS, GRASPS, FUTURE, TEMPORAL = 9, 180, 13, 2
STATE_DIM = R.reach_state_dim(1)


def yaw_quat(yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    return torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)), -1)


def repository_config():
    return yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())


# ---------------------------------------------------------------------------- core
class CarrierFutureTests(unittest.TestCase):
    def integrate(self, body, carrier, carrier_yaw, velocity, yaw_rate, horizon, steps=20000):
        """Explicit integration of the carrier twist; the body follows rigidly."""
        dt = horizon / steps
        position, yaw = carrier.clone(), float(carrier_yaw)
        lever = body[:2] - carrier[:2]
        for _ in range(steps):
            c, s = math.cos(yaw), math.sin(yaw)
            position[0] += dt * (c * velocity[0] - s * velocity[1])
            position[1] += dt * (s * velocity[0] + c * velocity[1])
            yaw += dt * yaw_rate
        turn = yaw - carrier_yaw
        c, s = math.cos(turn), math.sin(turn)
        return torch.tensor([position[0] + c * lever[0] - s * lever[1],
                             position[1] + s * lever[0] + c * lever[1], body[2]]), yaw

    def test_matches_numerical_integration_for_lines_and_circles(self):
        body = torch.tensor([[1.1, 2.05, .6]])
        carrier = torch.tensor([[1.0, 2.0, .55]])
        horizons = torch.tensor([0., .9, 2.4])
        for velocity, yaw_rate, carrier_yaw in (((0., .3), 0., .4), ((0., -.25), .3, -1.2),
                                                ((.1, .2), -.28, 2.0), ((0., .2), 1e-9, 0.)):
            with self.subTest(velocity=velocity, yaw_rate=yaw_rate):
                pos, quat = R.planar_carrier_future(
                    body, yaw_quat([0.3]), carrier, torch.tensor([carrier_yaw]),
                    torch.tensor([velocity]), torch.tensor([yaw_rate]), horizons)
                for k, h in enumerate(horizons.tolist()):
                    expected, _ = self.integrate(body[0], carrier[0], carrier_yaw, velocity, yaw_rate, h)
                    torch.testing.assert_close(pos[0, k], expected, atol=2e-4, rtol=0)
                    yaw = R.yaw_from_quat(quat[0, k])
                    expected_yaw = torch.tensor(0.3 + yaw_rate * h)
                    torch.testing.assert_close(torch.atan2(torch.sin(yaw - expected_yaw),
                                                           torch.cos(yaw - expected_yaw)),
                                               torch.tensor(0.), atol=1e-5, rtol=0)


class MarginTests(unittest.TestCase):
    def test_margin_takes_the_best_candidate_and_valid_horizon(self):
        horizons = torch.tensor([0., 1., 2., 3.])
        times = torch.tensor([[[2., 3.], [1.5, 2.], [1., 4.], [.5, .5]]])  # [1, H, C]
        result = R.interception_margin(times.expand(2, -1, -1), horizons, torch.tensor([10., 1.5]))
        torch.testing.assert_close(result["per_horizon"][0], torch.tensor([-2., -.5, 1., 2.5]))
        torch.testing.assert_close(result["margin"], torch.tensor([2.5, -.5]))
        self.assertEqual(result["horizon"].tolist(), [3, 1])
        self.assertEqual(result["candidate"][0].tolist(), [0, 0, 0, 0])
        self.assertTrue(math.isinf(result["per_horizon"][1, 2].item()))

    def test_best_ignores_the_remaining_time(self):
        times = torch.tensor([[[2., 3.], [1.5, 2.], [1., 4.], [.5, .5]]])
        result = R.interception_margin(times, torch.tensor([0., 1., 2., 3.]), torch.tensor([1.5]))
        torch.testing.assert_close(result["best"][0], torch.tensor([-2., -.5, 1., 2.5]))

    def test_intercept_time_of_a_static_goal_is_its_reach_time(self):
        horizons = torch.tensor([0., .3, .6, .9])
        for reach in (.45, .3, 2.):
            with self.subTest(reach=reach):
                margins = (horizons - reach).view(1, -1)
                tau = R.intercept_time(margins, horizons, 8.)
                torch.testing.assert_close(tau, torch.tensor([reach]))

    def test_intercept_time_takes_the_first_crossing_and_clamps(self):
        horizons = torch.tensor([0., 1., 2., 3.])
        margins = torch.tensor([[0., -1., 1., 2.],      # at the goal
                                [-2., -.5, 1., 2.5],    # crosses between 1 and 2 at 1 + .5 / 1.5
                                [-1., .5, -1., 3.],     # the earliest crossing wins
                                [-9., -9., -9., -9.]])  # beyond: 3 + 9 = 12, clamped
        tau = R.intercept_time(margins, horizons, 8.)
        torch.testing.assert_close(tau, torch.tensor([0., 1. + 1. / 3, 2. / 3, 8.]))
        with self.assertRaises(ValueError):
            R.intercept_time(margins[:, :3], horizons, 8.)

    def test_intercept_time_is_continuous_at_the_last_horizon(self):
        horizons = torch.tensor([0., .3, .6, .9])
        below = R.intercept_time(torch.tensor([[-.9, -.6, -.3, -1e-4]]), horizons, 8.)
        at = R.intercept_time(torch.tensor([[-.9, -.6, -.3, 0.]]), horizons, 8.)
        torch.testing.assert_close(below, at, atol=2e-4, rtol=0)
        torch.testing.assert_close(at, torch.tensor([.9]))

    def test_horizon_zero_is_valid_even_without_remaining_time(self):
        result = R.interception_margin(torch.ones(1, 2, 3), torch.tensor([0., .3]), torch.tensor([0.]))
        torch.testing.assert_close(result["margin"], torch.tensor([-1.]))

    def test_horizon_helper_requires_zero_start(self):
        self.assertEqual(R.margin_horizons(0.0, .3, 4), [0.0, .3, .6, .9])
        for args in ((.1, .3, 4), (0., 0., 4), (0., .3, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                R.margin_horizons(*args)


def random_states(n, generator=None):
    states = torch.randn(n, STATE_DIM, generator=generator) * .3
    return states


class TimeToReachTests(unittest.TestCase):
    def test_every_kind_respects_the_physical_floor_and_clamp(self):
        generator = torch.Generator().manual_seed(0)
        states = random_states(64, generator)
        goals = torch.randn(64, 5, 3, generator=generator) * 2.
        floor = R.physical_lower_bound(states[:, :3], goals, R.ReachLimits())
        for kind in R.REACH_MODEL_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(1)
                model = R.TimeToReach(STATE_DIM, kind=kind)
                times = model(states, goals)
                self.assertEqual(times.shape, (64, 5))
                self.assertTrue(bool((times >= floor - 1e-6).all()))
                self.assertTrue(bool((times <= model.limits.max_time + 1e-6).all()))

    def test_floor_is_zero_at_the_end_effector_and_grows_with_distance(self):
        ee = torch.tensor([[.5, 0., .4]])
        goals = torch.tensor([[[.5, 0., .4], [1.5, 0., .4], [3.5, 0., .4]]])
        floor = R.physical_lower_bound(ee, goals, R.ReachLimits())[0]
        self.assertAlmostEqual(floor[0].item(), 0., places=6)
        self.assertTrue(bool((floor[1:] > floor[:-1]).all()))
        estimate = R.analytic_time_estimate(ee, goals, R.ReachLimits())[0]
        self.assertTrue(bool((estimate >= floor).all()))

    def test_chunked_evaluation_save_and_load(self):
        torch.manual_seed(2)
        model = R.TimeToReach(STATE_DIM, kind="quasimetric")
        model.update_normalizer(random_states(100))
        states, goals = random_states(10), torch.randn(10, 7, 3)
        torch.testing.assert_close(model.evaluate(states, goals, chunk=3), model(states, goals).detach())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reach.pt"
            R.save_reach_model(path, model, {"gripper_body_index": 23})
            loaded, metadata = R.load_reach_model(path)
        self.assertEqual(metadata["gripper_body_index"], 23)
        torch.testing.assert_close(loaded(states, goals), model(states, goals))

    def test_normalizer_matches_batch_statistics(self):
        model = R.TimeToReach(STATE_DIM)
        data = torch.randn(300, STATE_DIM) * 2. + 1.
        for part in data.split(70):
            model.update_normalizer(part)
        torch.testing.assert_close(model.state_mean, data.mean(0), atol=1e-5, rtol=0)
        torch.testing.assert_close(model.state_var, data.var(0, unbiased=False), atol=1e-4, rtol=0)

    def test_invalid_shapes_and_kinds_are_rejected(self):
        model = R.TimeToReach(STATE_DIM)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, STATE_DIM), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            R.TimeToReach(STATE_DIM, kind="table")

    def test_expectile_weights_overestimates_more_for_a_low_tau(self):
        target = torch.zeros(1)
        over = R.expectile_loss(torch.ones(1), target, .1)
        under = R.expectile_loss(-torch.ones(1), target, .1)
        self.assertAlmostEqual(over.item() / under.item(), 9., places=5)


class HindsightBufferTests(unittest.TestCase):
    def filled_buffer(self):
        buffer = R.HindsightBuffer(2, window=6, state_dim=STATE_DIM, device="cpu")
        for step in range(5):
            state = torch.zeros(2, STATE_DIM)
            state[:, 0] = step
            # Environment 1 starts a new episode at step 3.
            episode = torch.tensor([0, 0 if step < 3 else 1])
            state[:, 1] = episode
            ee = torch.tensor([[float(step), 0., 0.], [float(step), 1., 0.]])
            buffer.push(state, ee, torch.zeros(2, 3), torch.full((2,), math.pi / 2),
                        torch.tensor([.15 * step, .15 * (step if step < 3 else step - 3)]), episode)
        return buffer

    def test_pairs_stay_within_one_episode_with_exact_labels_and_frames(self):
        buffer = self.filled_buffer()
        generator = torch.Generator().manual_seed(3)
        states, goals, labels, steps = buffer.sample(4000, 4, torch.arange(2), generator)
        self.assertGreater(labels.numel(), 500)
        start = states[:, 0]
        end = start + steps
        torch.testing.assert_close(labels, .15 * steps.float(), atol=1e-5, rtol=0)
        env1 = states[:, 1] == 1
        self.assertTrue(bool((start[env1] >= 3).all()))
        # Yaw pi/2 maps environment 1's world y = 1 to query x = 1. Its episode 0
        # holds steps 0-2 only: no such pair may end at step 3 or later.
        early = (states[:, 1] == 0) & (goals[:, 0] > .5)
        self.assertTrue(bool(early.any()))
        self.assertTrue(bool((end[early] <= 2).all()))
        # Query frame yaw pi/2 maps world (x, y) to (y, -x).
        env0 = goals[:, 0].abs() < 1e-5
        torch.testing.assert_close(goals[env0, 1], -end[env0].float(), atol=1e-5, rtol=0)

    def test_empty_or_unfilled_buffer_returns_no_pairs(self):
        buffer = R.HindsightBuffer(2, window=4, state_dim=STATE_DIM, device="cpu")
        self.assertEqual(buffer.sample(10, 3, torch.arange(2))[2].numel(), 0)


class ReachLearnerTests(unittest.TestCase):
    def test_learns_constant_speed_time_and_beats_the_analytic_estimate(self):
        """A robot moving its EE straight ahead at 0.2 m/s: T(s, g) = |g| / 0.2."""
        torch.manual_seed(4)
        num, speed, dt = 64, .2, .15
        model = R.TimeToReach(STATE_DIM, kind="mlp", hidden=64, latent=32)
        learner = R.ReachLearner(model, num, "cpu", window=24, max_steps=16, batch_size=1024,
                                 learning_rate=3e-3, expectile=.3, ema_decay=.9, validation_every=8)
        base = torch.randn(num, STATE_DIM) * .1
        for step in range(24):
            position = torch.zeros(num, 3)
            position[:, 0] = speed * dt * step
            state = base.clone()
            state[:, :3] = 0.  # the EE sits at the query-frame origin of each step
            learner.observe(state, position, position, torch.zeros(num), torch.full((num,), dt * step),
                            torch.zeros(num, dtype=torch.long))
        for _ in range(400):
            learner.update(1)
        quality = learner.validate(4096)
        self.assertLess(quality["learned_mae"], .25)
        self.assertLess(quality["learned_mae"], quality["analytic_mae"])
        restored = R.ReachLearner(R.TimeToReach(STATE_DIM, kind="mlp", hidden=64, latent=32), num, "cpu",
                                  window=24, max_steps=16)
        restored.load_state_dict(learner.state_dict())
        self.assertEqual(restored.updates, learner.updates)

    def test_analytic_model_cannot_be_trained(self):
        with self.assertRaises(ValueError):
            R.ReachLearner(R.TimeToReach(STATE_DIM, kind="analytic"), 4, "cpu")


# ---------------------------------------------------------------------- curriculum
def bank_state(object_x, table_vy, table_yaw=math.pi / 2, rotation=False, yaw_rate=0.):
    """Bank-format initial states: robot at the origin facing +x, EE ahead of the arm base."""
    n = len(object_x)
    robot = torch.zeros(n, 13)
    robot[:, 2], robot[:, 6] = .55, 1.
    bodies = torch.zeros(n, 3, 13)
    bodies[:, :, 6] = 1.
    bodies[:, 2, :3] = torch.tensor([.6, 0., .5])
    obj = torch.zeros(n, 13)
    obj[:, 0] = torch.tensor(object_x)
    obj[:, 2], obj[:, 6] = .5, 1.
    table = obj.clone()
    table[:, 3:7] = yaw_quat([table_yaw] * n)
    return {"robot_root": robot, "rigid_body_state": bodies, "object_root": obj, "table_root": table,
            "dof_pos": torch.zeros(n, 19), "dof_vel": torch.zeros(n, 19),
            "ee_goal_cart": torch.tensor([[.3, 0., .4]]).expand(n, -1).clone(),
            "ee_goal_orn_rpy": torch.zeros(n, 3), "table_vy": torch.tensor(table_vy),
            "table_yaw_rate": torch.full((n,), float(yaw_rate)),
            "rotation_enabled": torch.full((n,), rotation, dtype=torch.bool),
            "family_id": torch.zeros(n, dtype=torch.long)}


class CurriculumTests(unittest.TestCase):
    analytic = R.TimeToReach(STATE_DIM, kind="analytic")
    grasp = torch.eye(4).expand(2, 4, 4)
    horizons = R.margin_horizons(0., .3, 16)

    def test_object_motion_orders_the_initial_margin(self):
        # Table yaw pi/2: positive local vy moves the object toward the robot (-x).
        state = bank_state([2.5, 2.5, 2.5], [.3, 0., -.3])
        m0 = MC.initial_margins(self.analytic, state, self.grasp, self.horizons, 15., 2, 1)[:, 0]
        self.assertGreater(m0[0].item(), m0[1].item())
        self.assertGreater(m0[1].item(), m0[2].item())

    def test_faster_receding_tables_lower_the_margin(self):
        state = bank_state([1.8], [-.3])
        scales = torch.linspace(1., 1.6, 4)
        m0 = MC.initial_margins(self.analytic, state, self.grasp, self.horizons, 15., 2, 1, scales)[0]
        self.assertTrue(bool((m0[1:] <= m0[:-1] + 1e-6).all()))

    def test_frontier_choice_delta_schedule_and_prior(self):
        margins = torch.tensor([[1., .5, .1, -.2], [-1., -2., -3., -4.], [2., 2., 2., 2.]])
        self.assertEqual(MC.choose_frontier(margins, 0.).tolist(), [2, 0, 3])
        self.assertEqual(MC.margin_delta(0, 100, .5, 0.), .5)
        self.assertAlmostEqual(MC.margin_delta(50, 100, .5, 0.), .25)
        self.assertEqual(MC.margin_delta(500, 100, .5, 0.), 0.)
        rates = MC.prior_success_rates(torch.tensor([-5., 0., 5.]), 0., .5)
        self.assertLess(rates[0].item(), .01)
        self.assertAlmostEqual(rates[1].item(), .5)

    def test_auc_and_spearman(self):
        scores, labels = torch.tensor([.1, .4, .35, .8]), torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(MC.roc_auc(scores, labels), .75)
        self.assertAlmostEqual(MC.roc_auc(torch.tensor([1., 1.]), torch.tensor([0, 1])), .5)
        self.assertIsNone(MC.roc_auc(scores, torch.ones(4)))
        self.assertAlmostEqual(MC.spearman(torch.arange(5.), torch.arange(5.) ** 3), 1.)

    def test_sampler_prior_affects_sampling_only(self):
        sampler = SGSSampler(4, history_size=10, mode="sgs")
        sampler.observe(torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0]))
        observed = sampler.success_rates().clone()
        plain = sampler.probabilities().clone()
        sampler.set_prior(torch.tensor([.5, .5, .5, .0]), 2.)
        torch.testing.assert_close(sampler.success_rates(), observed)
        # (successes + 2 * prior) / (outcomes + 2)
        torch.testing.assert_close(sampler.sampling_rates(), torch.tensor([.5, 1. / 3, .5, 0.]))
        probabilities = sampler.probabilities()
        # Both unvisited: equal without the prior; the feasible one (prior 0.5) wins with it.
        torch.testing.assert_close(plain[2], plain[3])
        self.assertGreater(probabilities[2].item(), 5 * probabilities[3].item())
        sampler.set_prior(None, 0.)
        torch.testing.assert_close(sampler.probabilities(), plain)
        with self.assertRaises(ValueError):
            sampler.set_prior(torch.zeros(3), 1.)


# ---------------------------------------------------------------------- environment
class FakeEnv(InterceptionMarginMixin):
    """Simulator state of one B1Z1PickMulti with the margin mixin."""

    def __init__(self, num=2, settings=None, level="Level02", reward=False, curriculum=False):
        cfg = repository_config()
        margin = dict(reachModel="analytic", tokens=True)
        margin.update(settings or {})
        cfg["env"]["margin"] = margin
        cfg["reward"]["scales"]["margin_shaping"] = 10. if margin.get("reward") else 0.
        self.cfg = cfg
        self.no_feature = False
        self._setup_margin_obs()
        self.device, self.num_envs = "cpu", num
        self.tsil_enabled = self.floating_base = self.enable_camera = self.eval = False
        self.control_freq_low_init, self.control_freq_low, self.dt = 8, 8, .02
        self.rand_control = True
        self.T_grasp_in_obj = torch.eye(4).expand(num, 30, 4, 4).clone()
        self.Rx_rot = self.Rz_rot = torch.eye(3)
        self.num_gripper_dof, self.gripper_idx = 1, 2
        self.total_timesteps = 1000
        self.global_step_counter = 0
        self.pose_prediction_cfg = {}
        self.sgs_enabled = curriculum
        self.sgs_family_names = ("full_task", "pregrasp", "near_goal")
        self.table_level = level
        self.max_episode_length = 100
        self.progress_buf = torch.zeros(num, dtype=torch.long)
        self.reward_options = config_utils.validate_reward_options(cfg["reward"])
        self.extras = {}
        robot = torch.zeros(num, 13)
        robot[:, 2], robot[:, 6] = .55, 1.
        self._robot_root_states = robot
        self.ee_pos = torch.tensor([[.6, 0., .5]]).repeat(num, 1)
        self.ee_orn = torch.tensor([[0., 0., 0., 1.]]).repeat(num, 1)
        self._dof_pos = torch.zeros(num, 19)
        self._dof_vel = torch.zeros(num, 19)
        self.commands = torch.zeros(num, 3)
        self.curr_ee_goal_cart = torch.tensor([[.3, 0., .4]]).repeat(num, 1)
        self.curr_ee_goal_orn_rpy = torch.zeros(num, 3)
        obj = torch.zeros(num, 13)
        obj[:, 0], obj[:, 2], obj[:, 6] = .6, .5, 1.
        self._cube_root_states = obj
        table = obj.clone()
        table[:, 3:7] = yaw_quat([math.pi / 2] * num)
        self._table_root_states = table
        self.vel_line_forward_y_level02 = torch.zeros(num)
        self.lin_circle_select = torch.zeros(num, dtype=torch.bool)
        self.vel_angular_yaw = torch.zeros(num)
        self.lifted_object = torch.zeros(num, dtype=torch.bool)
        self.reset_buf = torch.zeros(num, dtype=torch.long)
        self.time = 0.
        if curriculum:
            self.sgs_bank = SimpleNamespace(tensors=bank_state([2.0, 2.5], [.3, 0.]))
            self.sgs_sampler = SGSSampler(2)
        self._init_margin()

    def _margin_sim_time(self):
        self.time += .15
        return self.time


class EnvironmentMarginTests(unittest.TestCase):
    def test_static_object_at_the_gripper_uses_the_largest_valid_horizon(self):
        env = FakeEnv()
        env.progress_buf[:] = torch.tensor([0, 94])
        env._update_margins()
        torch.testing.assert_close(env.margin_priv, torch.tensor([4.5, .9]))
        torch.testing.assert_close(env.margin_priv_horizon, torch.tensor([4.5, .9]))
        # tau* ignores the remaining time: the object is already at the gripper.
        torch.testing.assert_close(env.margin_priv_intercept, torch.zeros(2))

    def test_far_static_object_has_a_negative_margin(self):
        env = FakeEnv(1)
        env._cube_root_states[:, 0] = 3.3
        env._update_margins()
        # The repository YAML's 0.05 m EE increment gives arm_speed 0.25 * 0.05 / 0.02.
        arm_speed = R.reach_limits_for(config_utils.ee_delta_limits(env.cfg["env"])[0]).arm_speed
        self.assertAlmostEqual(env.margin_model.limits.arm_speed, arm_speed)
        expected = 2.4 / .45 + .6 / arm_speed
        self.assertAlmostEqual(env.margin_priv.item(), 4.5 - expected, places=4)
        # No crossing within 4.5 s: tau* continues as T of the static object.
        self.assertAlmostEqual(env.margin_priv_intercept.item(), expected, places=4)

    def test_reach_limits_follow_the_ee_increment(self):
        self.assertAlmostEqual(R.reach_limits_for(0.02).arm_speed, R.ReachLimits().arm_speed)
        self.assertAlmostEqual(R.reach_limits_for(0.05).arm_speed, 0.625)

    def test_approaching_object_has_a_larger_margin_than_a_receding_one(self):
        env = FakeEnv(3)
        env._cube_root_states[:, 0] = 2.5
        env.vel_line_forward_y_level02[:] = torch.tensor([.3, 0., -.3])  # table yaw pi/2: +vy -> -x
        env._update_margins()
        self.assertGreater(env.margin_priv[0].item(), env.margin_priv[1].item())
        self.assertGreater(env.margin_priv[1].item(), env.margin_priv[2].item())
        self.assertLess(env.margin_priv_intercept[0].item(), env.margin_priv_intercept[1].item())
        self.assertLess(env.margin_priv_intercept[1].item(), env.margin_priv_intercept[2].item())

    def test_token_block_layout_at_reset(self):
        env = FakeEnv(2)
        env.progress_buf[:] = torch.tensor([0, 97])
        env._reset_margin_state(torch.arange(2))
        block = env._margin_observations(torch.arange(2), torch.zeros(2, 1))
        self.assertEqual(block.shape, (2, len(MARGIN_SCALAR_NAMES) + 16 * MARGIN_TOKEN_DIM))
        tokens = block[:, 2:].view(2, 16, MARGIN_TOKEN_DIM)
        # Query frame: object 0.3 m ahead of the arm base (z 0.64 m), at z 0.5 m.
        torch.testing.assert_close(tokens[0, 0, :3], torch.tensor([.3, 0., -.14]), atol=1e-5, rtol=0)
        torch.testing.assert_close(tokens[0, :, 9], env.margin_horizons)
        # The EKF has one sample after a reset: only h = 0 is a valid token.
        self.assertEqual(tokens[0, :, 11].tolist(), [1.] + [0.] * 15)
        # Remaining 3 steps (0.45 s): later horizons carry the clipped margin.
        torch.testing.assert_close(tokens[1, 2:, 10], torch.full((14,), -2.))
        # Scalars: clipped M (static object at the gripper: the largest valid horizon), its horizon.
        torch.testing.assert_close(block[:, 0], torch.tensor([2., .3]), atol=1e-5, rtol=0)
        torch.testing.assert_close(block[:, 1], torch.tensor([4.5, .3]), atol=1e-5, rtol=0)

    def test_step_update_records_pairs_and_logs(self):
        env = FakeEnv(2, {"reachModel": "new:mlp", "online": True, "onlineInterval": 1,
                          "onlineUpdates": 1, "onlineBatch": 64, "window": 8, "maxPairSteps": 4})
        env._reset_margin_state(torch.arange(2))
        for step in range(1, 5):
            env.global_step_counter = step
            env.progress_buf += 1
            env.ee_pos[:, 0] += .02
            env._margin_step_update()
        self.assertEqual(env.margin_learner.buffer.filled, 4)
        self.assertGreater(env.margin_learner.updates, 0)
        self.assertIn("Margin / privileged mean (s)", env.extras["diagnostics"])
        self.assertIn("Margin / privileged intercept time mean (s)", env.extras["diagnostics"])
        self.assertIn("Reach / loss", env.extras["diagnostics"])
        torch.testing.assert_close(env.margin_episode_time, torch.full((2,), .64))

    def test_margin_shaping_telescopes_and_never_pays_for_termination(self):
        # Values per step: tau* (s) for intercept, M (s) for slack.
        for potential, buffer, values in (("intercept", "margin_priv_intercept", (3., 8., .2)),
                                          ("slack", "margin_priv", (.5, -1., 3.))):
            with self.subTest(potential=potential):
                env = FakeEnv(1, {"reward": True, "tokens": False, "potential": potential})
                env._reset_margin_state(torch.arange(1))
                phi0 = env.margin_phi_prev.clone()
                gamma = env.reward_options["shapingGamma"]
                total, discount = 0., 1.
                for step, value in enumerate(values):
                    getattr(env, buffer)[:] = value
                    env.reset_buf[:] = int(step == 2)
                    reward, phi = env._reward_margin_shaping()
                    self.assertTrue(0. <= phi.item() <= 1.)
                    if step == 2:
                        self.assertLessEqual(reward.item(), 0.)
                    total += discount * reward.item()
                    discount *= gamma
                self.assertAlmostEqual(total, -phi0.item(), places=5)

    def test_intercept_potential_rises_as_the_intercept_time_falls(self):
        env = FakeEnv(3, {"reward": True, "tokens": False})
        self.assertEqual(env.margin_potential_kind, "intercept")
        env.margin_priv_intercept[:] = torch.tensor([8., 4., 0.])
        torch.testing.assert_close(env._margin_potential(), torch.tensor([0., .5, 1.]))

    def test_lifted_object_has_the_maximal_potential(self):
        for potential in ("intercept", "slack"):
            with self.subTest(potential=potential):
                env = FakeEnv(2, {"reward": True, "tokens": False, "potential": potential})
                env.margin_priv[:] = -5.
                env.margin_priv_intercept[:] = 8.
                env.lifted_object[:] = torch.tensor([True, False])
                torch.testing.assert_close(env._margin_potential(), torch.tensor([1., 0.]))

    def test_shaping_requires_the_margin_reward(self):
        with self.assertRaisesRegex(ValueError, "margin.reward"):
            FakeEnv(1)._reward_margin_shaping()

    def test_frontier_speeds_up_only_offered_full_task_resets(self):
        env = FakeEnv(4, {"curriculum": "frontier", "frontierProbability": 1.,
                          "frontierCandidates": 4, "tokens": False}, curriculum=True)
        env.margin_curriculum = "frontier"
        env._margin_delta = lambda: 0.
        env._margin_initial_margins = lambda state, scales: torch.tensor(
            [[1., .5, .1, -.2], [1., .5, .1, -.2], [-1., -2., -3., -4.], [2., 2., 2., 2.]])
        state = {"family_id": torch.tensor([0, 1, 0, 0]), "table_vy": torch.full((4,), .2),
                 "table_yaw_rate": torch.full((4,), .1)}
        state = env._margin_apply_frontier(torch.arange(4), state)
        torch.testing.assert_close(state["table_vy"], torch.tensor([.2 * 1.4, .2, .2, .2 * 1.6]))
        self.assertEqual(env.margin_frontier_active.tolist(), [True, False, False, True])
        env.episode_success = torch.tensor([True, False, False, False])
        frontier = env._margin_record_frontier(torch.arange(4), torch.ones(4, dtype=torch.bool))
        self.assertEqual(frontier.tolist(), [True, False, False, True])
        snapshots = {}
        env._margin_sgs_snapshots(snapshots)
        self.assertEqual(snapshots["SuccessRate / frontier"], .5)

    def test_curriculum_requires_sgs(self):
        with self.assertRaisesRegex(ValueError, "SGS"):
            FakeEnv(1, {"curriculum": "prior"})


# ---------------------------------------------------------------- layout and models
class LayoutTests(unittest.TestCase):
    def setup_task(self, margin=None, state=False, time=True):
        class Base(critic_time.SetupBase, InterceptionMarginMixin):
            pass
        task = critic_time.task_with("_setup_obs_and_action_info", base=Base)
        type(task)._setup_obs_and_action_info.__globals__.update(CONSTANTS)
        task.no_feature = False
        task.num_features = 1024
        env = {"noVelocityObs": True, "lastCommands": False, "tsil": {"enabled": False},
               "posePrediction": {"enabled": True, "horizons": [.6, .9]}, "criticTimeObs": time,
               "criticRewardStateObs": state}
        if margin is not None:
            env["margin"] = margin
        task.cfg = {"env": env, "reward": {"scales": {}}}
        task._setup_obs_and_action_info()
        return task

    def test_tokens_sit_before_the_critic_blocks_without_moving_them(self):
        reference = self.setup_task(state=True)
        self.assertIsNone(reference.margin_obs_index)
        task = self.setup_task({"reachModel": "analytic", "tokens": True}, state=True)
        width = len(MARGIN_SCALAR_NAMES) + 16 * MARGIN_TOKEN_DIM
        self.assertEqual(task.num_margin_obs, width)
        self.assertEqual(task.critic_state_obs_index, reference.critic_state_obs_index)
        self.assertEqual(task.critic_time_obs_index, reference.critic_time_obs_index)
        self.assertEqual(task.margin_obs_index, reference.critic_state_obs_index - width)
        self.assertEqual(task.dimensions["num_obs"], reference.dimensions["num_obs"] + width)
        self.assertEqual(task.dimensions["removed_dim"], reference.dimensions["removed_dim"] + width)

    def test_margin_without_tokens_keeps_the_layout(self):
        reference = self.setup_task()
        task = self.setup_task({"reachModel": "analytic"})
        self.assertEqual(task.num_margin_obs, 0)
        self.assertEqual(task.dimensions, reference.dimensions)

    def test_observation_writes_the_block_at_the_published_index(self):
        task = critic_time.EnvironmentLayoutTests.observation_task([0, 50])
        width = 5
        task.num_margin_obs = width
        task.obs_buf = torch.zeros(2, 1286 + 1 + width)
        values = torch.arange(width, dtype=torch.float) + 300.
        task._margin_observations = lambda ids, like: values.expand(len(ids), -1)
        task._compute_observations(torch.arange(2))
        start = -(ACTIONS + GRASPS + TEMPORAL + FUTURE + 1 + width)
        torch.testing.assert_close(task.obs_buf[:, start:start + width], values.expand(2, -1))
        torch.testing.assert_close(task.obs_buf[:, start - 1], torch.full((2,), 57.))
        torch.testing.assert_close(task.obs_buf[:, -205], torch.tensor([0., .5]))


class TeacherModelTests(unittest.TestCase):
    features, body, tokens = 16, 24, 4
    width = len(MARGIN_SCALAR_NAMES) + tokens * MARGIN_TOKEN_DIM
    index = -(ACTIONS + GRASPS + TEMPORAL + FUTURE + width)

    def parts(self, token_value=None, batch=3):
        generator = torch.Generator().manual_seed(7)
        head = [torch.randn(batch, w, generator=generator) * .3 for w in (self.features, self.body)]
        tail = [torch.randn(batch, w, generator=generator) * .3 for w in (FUTURE, TEMPORAL, GRASPS, ACTIONS)]
        middle = []
        if token_value is not None:
            scalars = torch.full((batch, len(MARGIN_SCALAR_NAMES)), .5)
            middle = [scalars, torch.full((batch, self.tokens * MARGIN_TOKEN_DIM), float(token_value))]
        return torch.cat(head + middle + tail, dim=-1)

    def model(self, cls, width, margin=True):
        torch.manual_seed(5)
        kwargs = dict(num_future_pose_obs=FUTURE, num_tsil_obs=TEMPORAL)
        if margin:
            kwargs.update(margin_index=self.index, margin_tokens=self.tokens,
                          margin_token_dim=MARGIN_TOKEN_DIM, margin_scalars=len(MARGIN_SCALAR_NAMES))
        return cls(width, ACTIONS, "cpu", self.features, 128, None, None, None, **kwargs)

    def test_actor_mlp_keeps_the_scalars_and_routes_tokens_to_the_gfm(self):
        reference = self.model(Policy, self.parts().shape[-1], margin=False)
        policy = self.model(Policy, self.parts(0.).shape[-1])
        self.assertEqual(policy.net[0].in_features,
                         reference.net[0].in_features + len(MARGIN_SCALAR_NAMES))
        first = policy.compute({"states": self.parts(0.)}, role="policy")[0]
        second = policy.compute({"states": self.parts(3.)}, role="policy")[0]
        self.assertGreater((first - second).abs().max().item(), 1e-7)
        with torch.no_grad():
            for module in (policy.token_key_proj, policy.token_value_proj):
                module.weight.zero_()
                module.bias.zero_()
        first = policy.compute({"states": self.parts(0.)}, role="policy")[0]
        second = policy.compute({"states": self.parts(3.)}, role="policy")[0]
        torch.testing.assert_close(first, second)

    def test_critic_mlp_and_gfm_receive_the_block(self):
        reference = self.model(Value, self.parts().shape[-1], margin=False)
        value = self.model(Value, self.parts(0.).shape[-1])
        self.assertEqual(value.net[0].in_features, reference.net[0].in_features + self.width)
        self.assertTrue(hasattr(value, "token_slot"))
        value.compute({"states": self.parts(1.)}, role="value")

    def test_index_outside_the_state_block_is_rejected(self):
        width = self.parts(0.).shape[-1]
        with self.assertRaisesRegex(ValueError, "margin_index"):
            torch.manual_seed(5)
            Policy(width, ACTIONS, "cpu", self.features, 128, None, None, None,
                   num_future_pose_obs=FUTURE, num_tsil_obs=TEMPORAL, margin_index=-(ACTIONS + GRASPS + 1),
                   margin_tokens=self.tokens, margin_token_dim=MARGIN_TOKEN_DIM, margin_scalars=2)

    def test_layout_kwargs_publish_the_block(self):
        env = SimpleNamespace(cfg={"env": {}}, margin_obs_index=self.index, num_margin_tokens=self.tokens)
        kwargs = teacher_gfm_kwargs(env)
        self.assertEqual((kwargs["margin_index"], kwargs["margin_tokens"], kwargs["margin_token_dim"],
                          kwargs["margin_scalars"]),
                         (self.index, self.tokens, MARGIN_TOKEN_DIM, len(MARGIN_SCALAR_NAMES)))
        env.margin_obs_index = None
        self.assertNotIn("margin_index", teacher_gfm_kwargs(env))


class ConfigurationTests(unittest.TestCase):
    def test_repository_yaml_keeps_main_line_a_off(self):
        cfg = repository_config()
        settings = config_utils.validate_margin_config(cfg["env"], cfg["reward"])
        self.assertEqual(settings["reachModel"], "")
        self.assertEqual(cfg["reward"]["scales"]["margin_shaping"], 0.)
        args = config_utils.get_params([])
        for name in ("reach_model", "margin_reward", "margin_tokens", "margin_curriculum",
                     "reach_online", "margin_keep_tracking", "margin_reward_scale", "margin_potential"):
            self.assertIsNone(getattr(args, name))

    def test_margin_reward_keeps_tracking_rewards_unless_replaced(self):
        for extra, expected in (([], .5), (["--margin_keep_tracking"], .5), (["--no_margin_keep_tracking"], 0.)):
            with self.subTest(extra=extra):
                cfg = repository_config()
                argv = ["--reach_model", "analytic", "--margin_reward"] + extra
                settings = config_utils.apply_margin_config(cfg, config_utils.get_params(argv))
                self.assertTrue(settings["reward"])
                self.assertEqual(settings["potential"], "intercept")
                self.assertEqual(cfg["reward"]["scales"]["margin_shaping"], 1.)
                self.assertEqual(cfg["reward"]["scales"]["approaching"], expected)

    def test_margin_potential_and_scale_flags(self):
        cfg = repository_config()
        argv = ["--reach_model", "analytic", "--margin_reward", "--margin_potential", "slack",
                "--margin_reward_scale", "2.5"]
        settings = config_utils.apply_margin_config(cfg, config_utils.get_params(argv))
        self.assertEqual(settings["potential"], "slack")
        self.assertEqual(cfg["reward"]["scales"]["margin_shaping"], 2.5)
        cfg = repository_config()
        cfg["env"]["margin"]["potential"] = "distance"
        with self.assertRaisesRegex(ValueError, "potential"):
            config_utils.validate_margin_config(cfg["env"], cfg["reward"])

    def test_inconsistent_settings_are_rejected(self):
        for argv in (["--margin_reward"], ["--margin_tokens"], ["--reach_model", "new:mlp"],
                     ["--reach_model", "analytic", "--reach_online"], ["--reach_model", "missing.pt"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                config_utils.apply_margin_config(repository_config(), config_utils.get_params(argv))
        cfg = repository_config()
        cfg["env"]["margin"]["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            config_utils.validate_margin_config(cfg["env"], cfg["reward"])


if __name__ == "__main__":
    unittest.main()
