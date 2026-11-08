"""CPU regression for bounded executed-action memory, retaining raw PPO actions.

Use the production controller/reward functions without loading Isaac Gym, and
the production Fixed8 adapter with the existing tensor-only robot fixture.
"""

from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_world_model_env import production_function, reference_namespace
from test_world_model_fixed8_ppo import make_fixed8_agent
from utils.gripper_control import closing_command, opening_to_joint
from utils.wm_env import quat_apply


def production_controller(env):
    """Supply non-physics bookkeeping to the actual high-level controller."""
    n = env.num_envs
    env.cfg["env"]["useTanh"] = False
    env.continuous_gripper = True
    env.stu_distill = env.arm_delay = env.pitch_control = env.floating_base = False
    env.rand_depth_clip = False
    env.total_timesteps = 40000
    env.actions = torch.zeros(n, 9)
    env.clipped_actions = torch.zeros(n, 9)
    env.command_history_buf = torch.zeros(n, 3, 9)
    env.gripper_dof_pos = torch.zeros(n, 1)
    env.arm_base = env._robot_root_states[:, :3] + quat_apply(
        env._robot_root_states[:, 3:7], torch.tensor([.3, 0., .09]).expand(n, -1))
    namespace = dict(reference_namespace(), opening_to_joint=opening_to_joint,
                     closing_command=closing_command, LIN_VEL_X_CLIP=.15, ANG_VEL_YAW_CLIP=.35)
    path = ROOT / "envs/b1z1_base.py"
    gripper = production_function(path, "set_gripper", namespace, "B1Z1Base")
    commands = production_function(path, "clip_commands", namespace, "B1Z1Base")
    env.set_gripper = lambda: gripper(env)
    env.clip_commands = lambda: commands(env)
    return production_function(path, "pre_physics_step", namespace, "B1Z1Base")


class ActionBoundTests(unittest.TestCase):
    def setUp(self):
        self.agent = make_fixed8_agent("cpu")
        self.adapter = self.agent.wm_adapter
        self.state, self.context = self.adapter.capture()

    def actions(self):
        rows = torch.tensor([
            [1000., -1000., 400., -100., 50., -9000., .2, 50., -70.],
            [-5., 10., -3., 8., -9., 2., .8, -80., 90.],
            [.01, -.03, .5, -.01, .04, -.08, .5, .2, .1],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
        ])
        return rows.repeat(self.agent.wm_env.num_envs // len(rows), 1)

    def test_explicit_outer_bound_preserves_legacy_physical_commands(self):
        raw = self.actions()
        original = raw.clone()
        for stop_pick in (False, True):
            with self.subTest(stop_pick=stop_pick):
                self.adapter.stop_pick = stop_pick
                prepared, real = {}, {}
                for limit in (float("inf"), 1.):
                    self.adapter.clip_actions = limit
                    prepared[limit] = self.adapter.prepare_high(self.state, raw)
                    env = deepcopy(self.agent.wm_env)
                    env.stop_pick, env.clip_actions = stop_pick, limit
                    control = production_controller(env)
                    control(env, raw.clamp(-limit, limit))
                    real[limit] = env
                    for field, attribute in (("goal_pos", "curr_ee_goal_cart"),
                                             ("goal_rpy", "curr_ee_goal_orn_rpy"),
                                             ("commands", "commands"), ("prev_action", "actions"),
                                             ("held_gripper_target", "gripper_dof_pos")):
                        torch.testing.assert_close(self.adapter.get(prepared[limit], field),
                                                   getattr(env, attribute), rtol=0, atol=1e-6)
                    target = self.adapter.get(prepared[limit], "held_ee_target")
                    # Fixture roots are spaced up to 88 m; adding the origin
                    # back can differ by one float32 ULP at those coordinates.
                    torch.testing.assert_close(target[:, :3] + self.adapter.origin,
                                               env.ee_goal_global_cart, rtol=0, atol=1e-5)
                    torch.testing.assert_close(target[:, 3:], env.ee_goal_global_orn_quat,
                                               rtol=0, atol=2e-6)
                for field in ("goal_pos", "goal_rpy", "commands", "held_ee_target", "held_gripper_target"):
                    torch.testing.assert_close(self.adapter.get(prepared[1.], field),
                                               self.adapter.get(prepared[float("inf")], field), rtol=0, atol=0)
                for attribute in ("curr_ee_goal_cart", "curr_ee_goal_orn_rpy", "commands", "gripper_dof_pos"):
                    torch.testing.assert_close(getattr(real[1.], attribute),
                                               getattr(real[float("inf")], attribute), rtol=0, atol=0)
                self.assertLessEqual(float(real[1.].action_history_buf.abs().max()), 1.)
                torch.testing.assert_close(self.adapter.get(prepared[1.], "prev_action"), raw.clamp(-1., 1.))
        torch.testing.assert_close(raw, original, rtol=0, atol=0)

    def test_saturated_history_cannot_send_unbounded_action_gradients(self):
        self.adapter.clip_actions = 1.
        action = self.actions().requires_grad_()
        state = self.adapter.prepare_high(self.state, action)
        history = self.adapter.get(state, "prev_action")
        history.sum().backward()
        self.assertLessEqual(float(history.abs().max()), 1.)
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertTrue(bool((action.grad[action.detach().abs() > 1.] == 0).all()))
        interior = action.detach().abs() < 1.
        torch.testing.assert_close(action.grad[interior], torch.ones_like(action.grad[interior]))
        # In particular, the continuously open/close gripper retains its local
        # action derivative; only outer-bound saturation loses its derivative.
        torch.testing.assert_close(action.grad[:3, 6], torch.ones(3))

    def test_ppo_keeps_raw_actions_while_world_replay_keeps_executed_actions(self):
        agent, adapter = self.agent, self.adapter
        observation = adapter.observation(self.state, self.context)
        with torch.no_grad():
            agent.policy.linear.weight.zero_()
            agent.policy.linear.bias.copy_(self.actions()[0])
            raw, _, outputs = agent.act(observation, timestep=0, timesteps=100)
        latent = outputs["latent_actions"].clone()
        self.assertGreater(float(raw.abs().max()), 100.)
        executed = raw.clamp(-1., 1.)
        after = adapter.prepare_high(self.state, raw)
        next_observation = adapter.observation(after, self.context)
        zero = torch.zeros(len(raw))

        def transition(actions, dt):
            return dict(state=self.state, next_state=after, context=self.context,
                        actions=actions, dt=torch.full((len(raw), 1), dt), reward=zero,
                        terminated=zero.bool(), truncated=zero.bool(), success=zero.bool(),
                        family=zero.long(), episode_id=zero.long())

        high = transition(executed, .16)
        high["low_steps"] = [transition(torch.zeros(len(raw), 20), .02) for _ in range(8)]
        with torch.no_grad():
            agent.record_transition(observation, raw, zero[:, None], next_observation,
                                    zero[:, None].bool(), zero[:, None].bool(),
                                    {"world_model": high}, timestep=0, timesteps=100)
        torch.testing.assert_close(agent.memory.get_tensor_by_name("actions")[0], raw, rtol=0, atol=0)
        torch.testing.assert_close(agent.memory.get_tensor_by_name("latent_actions")[0], latent, rtol=0, atol=0)
        torch.testing.assert_close(agent.wm_replay.actions[0], executed, rtol=0, atol=0)

    def test_real_rewards_match_imagination_with_bounded_current_and_previous_actions(self):
        adapter, env = self.adapter, self.agent.wm_env
        raw = self.actions()
        previous = adapter.prepare_high(self.state, -raw)
        state = adapter.prepare_high(previous, raw)
        executed = adapter.get(state, "prev_action")
        outcome = adapter.task_outcome(previous, state, executed, self.context)
        robot, obj, ee, base, angles, heading, arm = adapter._geometry(state)
        standin = SimpleNamespace(num_envs=len(state), device="cpu",
            closest_dist=adapter.get(previous, "closest")[:, 0].clone(), curr_dist=outcome["distance"],
            highest_object=adapter.get(previous, "highest")[:, 0].clone(), curr_height=outcome["height"],
            lifted_object=outcome["lifted"], pick_success=outcome["success"],
            pick_hold_progress=outcome["hold_progress"], _dof_vel=adapter.get(state, "dof_vel"),
            _last_dof_vel=adapter.get(state, "last_dof_vel"), num_gripper_joints=1, dt=.02,
            commands=adapter.get(state, "commands"), actions=executed,
            last_actions=adapter.get(previous, "prev_action"), arm_base=arm,
            ee_pos=ee[:, :3], ee_orn=ee[:, 3:7], _robot_root_states=robot, _cube_root_states=obj,
            base_yaw_quat=heading, base_object_distace_threshold=.6,
            curr_ee_goal_cart=adapter.get(state, "goal_pos"), cfg=env.cfg,
            global_step_counter=100, train_reward_strict=50)
        namespace = reference_namespace()
        expected = torch.zeros(len(state))
        # Include inactive optional action penalties as well as the full task
        # reward, since they must obey the same execution/history convention.
        scales = dict(env.reward_scales, gripper_rate=-.001, action_penalty=-.001)
        adapter.reward_scales = scales
        terms = adapter.reward_terms(previous, state, raw, self.context)
        for name, scale in scales.items():
            if name == "grasp_base_height":
                function = production_function(ROOT / "envs/reward_vec_task.py", "_reward_base_height",
                                               namespace, "RewardVecTask")
                value = function(standin)[0] * outcome["hold_progress"]
            elif name == "limit_yaw_rotation_penalty":
                standin._yaw_deviation = lambda: (angles[:, 2] - self.context[:, 1]).abs()
                function = production_function(ROOT / "envs/b1z1_pickmulti.py", "_reward_" + name,
                    dict(namespace, np=SimpleNamespace(deg2rad=lambda v: v * math.pi / 180)), "B1Z1PickMulti")
                value = function(standin)[0]
            else:
                function = production_function(ROOT / "envs/reward_vec_task.py", "_reward_" + name,
                                               namespace, "RewardVecTask")
                value = (function(standin, obj[:, :3])[0]
                         if name in ("command_reward", "ee_orn", "base_dir", "base_approaching")
                         else function(standin)[0])
                if name == "approaching":
                    value = value * ~outcome["lifted"]
            torch.testing.assert_close(terms[name], value, atol=1e-6, rtol=1e-5)
            expected += value * scale
        torch.testing.assert_close(adapter.reward(previous, state, raw, self.context),
                                   expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
