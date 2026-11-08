"""Native-forward / surrogate-gradient contracts of the reference pick MDP."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from utils.pick_success import advance_pick_hold, lifted_grasp_mask
from utils.world_model.reference_task import ReferencePickTask, resolve_reference_task_config
from utils.wm_fixed8 import Fixed8TeacherWorldModelAdapter
from test_world_model_fixed8_ppo import make_fixed8_agent


def task():
    return ReferencePickTask(hold_steps=10, lift_threshold=.2, success_reward=50., high_dt=.16)


def outcomes(mdp, height, distance, opening, count=None, best=None, completed=None, failed=None, timeout=None):
    n = len(height)
    count = height.new_zeros(n) if count is None else count
    best = height.new_zeros(n) if best is None else best
    completed = torch.zeros(n, dtype=torch.bool) if completed is None else completed
    failed = torch.zeros(n, dtype=torch.bool) if failed is None else failed
    timeout = torch.zeros(n, dtype=torch.bool) if timeout is None else timeout
    lifted = lifted_grasp_mask(height, height.new_full((n,), .2), height.new_full((n,), .05),
                               distance, opening, .2)
    c, b, progress, success = advance_pick_hold(lifted, failed, count, best, completed, 10)
    native = dict(lifted=lifted, hold=c, best_hold=b, hold_progress=progress,
                  success=success, failed=failed, timeout=timeout & ~success & ~failed)
    augmented = mdp.outcomes(outcome=native, previous_count=count, previous_best=best,
        previous_completed=completed, object_height=height, target_height=height.new_full((n,), .45),
        distance=distance, opening=opening)
    return native, {**native, **augmented}


class ReferenceTaskTests(unittest.TestCase):
    @staticmethod
    def adapter():
        agent = make_fixed8_agent("cpu")
        env = agent.wm_env
        env.cfg["env"]["world_model"] = dict(task_contract="manipulation_v2", task_aux_alpha=0.,
            imagined_reward_mode="reference_task", critic_features_mode="reference_full")
        env._low_entry_dof_vel = env._dof_vel.clone()
        return agent, Fixed8TeacherWorldModelAdapter(env)

    def test_hard_candidate_equality_boundaries_and_failure_precedence(self):
        height = torch.tensor([.45, .4499, .46, .46, .46, .46], requires_grad=True)
        distance = torch.tensor([.1, .1, .2, .1, .1, .1], requires_grad=True)
        opening = torch.tensor([.1, .1, .1, .5, .1, .1], requires_grad=True)
        failed = torch.tensor([False, False, False, False, True, False])
        completed = torch.tensor([False, False, False, False, False, True])
        count = torch.full((6,), 9.)
        native, result = outcomes(task(), height, distance, opening, count=count,
                                  completed=completed, failed=failed)
        for name in ("hold", "best_hold"):
            torch.testing.assert_close(result[name][:-1], native[name][:-1], rtol=0, atol=0)
        torch.testing.assert_close(result["candidate_surrogate"], native["lifted"].float(), rtol=0, atol=0)
        torch.testing.assert_close(result["success_surrogate"][:-1], native["success"][:-1].float(), rtol=0, atol=0)
        torch.testing.assert_close(result["imagined_done"][:-1], (native["success"] | failed)[:-1].float(), rtol=0, atol=0)
        self.assertEqual(result["success_surrogate"][-1].item(), 1.)  # source hypothetical continuation
        gradient = torch.autograd.grad(result["success_surrogate"].sum(), (height, distance, opening))
        for value in gradient:
            self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(gradient[0][0].item(), 0.)
        self.assertLess(gradient[1][0].item(), 0.)
        self.assertLess(gradient[2][0].item(), 0.)
        self.assertEqual(gradient[0][4].item(), 0.)

    def test_sixteen_steps_terminal_forward_and_count_gradient_survival(self):
        height = torch.tensor([.455], requires_grad=True)
        distance, opening = torch.tensor([.19]), torch.tensor([.45])
        count = best = torch.zeros(1)
        completed = torch.zeros(1, dtype=torch.bool)
        dones, rewards = [], []
        for index in range(16):
            native, result = outcomes(task(), height, distance, opening, count=count,
                best=best, completed=completed)
            if index < 10:  # Every reachable transition matches native rules.
                torch.testing.assert_close(result["hold"], native["hold"], rtol=0, atol=0)
            dones.append(result["imagined_done"])
            rewards.append(50*result["success_surrogate"])
            count, best = result["hold"], result["best_hold"]
            completed = completed | native["success"]
        dones = torch.stack(dones)
        rewards = torch.stack(rewards)
        alive = torch.cumprod(1-dones, dim=0)
        before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
        self.assertEqual(dones[:9].sum().item(), 0.)
        self.assertEqual(dones[9].item(), 1.)
        torch.testing.assert_close(dones[9:], torch.ones_like(dones[9:]))
        self.assertEqual(count.item(), 16.)
        self.assertEqual(before[10:].sum().item(), 0.)
        self.assertEqual((before*rewards).sum().item(), 50.)
        discounts = .99**torch.arange(16, dtype=height.dtype)[:, None]
        gradient, = torch.autograd.grad((discounts*before*rewards).sum(), height, retain_graph=True)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.item(), 0.)
        survival_gradient, = torch.autograd.grad(alive[9].sum(), height)
        self.assertLess(survival_gradient.item(), 0.)

    def test_hypothetical_post_success_candidate_loss_matches_source_gradients(self):
        # The world model still evaluates this unreachable suffix. A native
        # completed latch must not force its counter/done/reward to constants.
        height = torch.tensor([.455, .445], requires_grad=True)
        mdp = task()
        count, done, raw_success = torch.tensor([9.]), [], []
        source_count, source_done, source_success = torch.tensor([9.]), [], []
        for index in range(2):
            _, result = outcomes(mdp, height[index:index+1], torch.tensor([.19]), torch.tensor([.45]),
                count=count, completed=torch.tensor([index == 1]))
            count = result["hold"]
            done.append(result["imagined_done"])
            raw_success.append(result["success_surrogate"])
            cfg = mdp.config
            soft = (torch.sigmoid((height[index:index+1]-.45)/cfg["height_temperature_m"])
                    * torch.sigmoid(torch.tensor([.01])/cfg["distance_temperature_m"])
                    * torch.sigmoid(torch.tensor([.05])/cfg["opening_temperature"]))
            candidate = (height[index:index+1] >= .45).float()+(soft-soft.detach())
            source_count = candidate*(source_count+1)
            soft_success = torch.sigmoid((source_count-10)/.5)
            success = (source_count >= 10).float()+(soft_success-soft_success.detach())
            success = success.detach()+.02*(success-success.detach())
            source_done.append(success)
            source_success.append(success)
        torch.testing.assert_close(torch.stack(done), torch.tensor([[1.], [0.]]))
        self.assertEqual(count.item(), 0.)
        # Include nonzero raw tail shaping: survival's ST derivative must see it.
        def objective(d, reward):
            d = torch.stack(d); reward = 50*torch.stack(reward)+torch.tensor([[.1], [.3]])
            alive = torch.cumprod(1-d, 0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
            return (before*reward).sum()+alive[-1].sum()*2
        actual = objective(done, raw_success)
        expected = objective(source_done, source_success)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_gradient, = torch.autograd.grad(actual, height, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, height)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)

    def test_broken_grasp_resets_count_and_timeout_is_absorbing_forward(self):
        native, result = outcomes(task(), torch.tensor([.449, .46]), torch.tensor([.1, .1]),
                                  torch.tensor([.1, .1]), count=torch.tensor([9., 2.]),
                                  timeout=torch.tensor([False, True]))
        torch.testing.assert_close(result["hold"], torch.tensor([0., 3.]))
        torch.testing.assert_close(result["imagined_done"], torch.tensor([0., 1.]))

    def test_relative_hold_quality_allows_common_world_translation(self):
        mdp = task()
        obj = torch.zeros(2, 13); obj[:, 6] = 1.; obj[:, :3] = torch.tensor([.1, 0., .46])
        ee = obj.clone(); ee[:, 0] -= .1
        opening = torch.tensor([.2, .2])
        _, result = outcomes(mdp, obj[:, 2], torch.tensor([.1, .1]), opening)
        action = torch.zeros(2, 9); action[:, 6] = opening
        inputs = dict(height=torch.tensor([.21, .21]), distance=torch.tensor([.1, .1]),
            action=action, previous_action=action, arm_velocity=torch.zeros(2, 6), outcome=result)
        stationary = mdp.reward_components(obj=obj, ee=ee, **inputs)
        obj[:, 7:10] = torch.tensor([1., -.5, .2]); ee[:, 7:10] = obj[:, 7:10]
        moving = mdp.reward_components(obj=obj, ee=ee, **inputs)
        torch.testing.assert_close(moving["reference_holding"], stationary["reference_holding"], rtol=0, atol=0)
        torch.testing.assert_close(moving["total"], stationary["total"], rtol=0, atol=0)
        obj[1, 7] += .3
        slipping = mdp.reward_components(obj=obj, ee=ee, **inputs)
        self.assertLess(slipping["reference_holding"][1].item(), moving["reference_holding"][1].item())

    def test_reference_adapter_keeps_real_reward_and_exposes_all_causal_inputs(self):
        legacy_agent, adapter = self.adapter()
        self.assertIsNone(adapter.task_objective)
        state, context = adapter.capture()
        variable = state.clone().requires_grad_()
        features = adapter.critic_features(variable, context)
        self.assertEqual(features.shape[-1], adapter.critic_feature_dim)
        self.assertEqual(adapter.observation(state, context).shape[-1], adapter.policy_observation_dim)
        # The compatibility EE state is deliberately stale after low steps.
        # FK supplies all current EE features; the cache must not reach V.
        changed = adapter.replace(state, ee=adapter.get(state, "ee")+100.)
        torch.testing.assert_close(adapter.critic_features(changed, context), features, rtol=0, atol=0)
        gradient, = torch.autograd.grad(features.sum(), variable)
        for name in ("prev_action", "low_history", "low_actions", "ekf_poses", "last_dof_vel"):
            self.assertTrue(torch.isfinite(gradient[:, adapter.slices[name]]).all())
            self.assertGreater(gradient[:, adapter.slices[name]].abs().sum().item(), 0.)
        action = adapter.get(state, "prev_action")
        parts = adapter.reward_components(state, state, action, context)
        torch.testing.assert_close(parts["task"], adapter.task_reward(state, state, action, context))
        torch.testing.assert_close(parts["total"], parts["reference_dense"]+parts["reference_success"])
        metadata = adapter.metadata()
        self.assertEqual(metadata["imagined_reward_mode"], "reference_task")
        self.assertEqual(metadata["critic_features_mode"], "reference_full")
        with self.assertRaisesRegex(ValueError, "schema/task"):
            legacy_agent._check_metadata(metadata)

    def test_fixed8_memory_and_ekf_advance_from_imagined_state_only(self):
        _, adapter = self.adapter()
        state, context = adapter.capture()
        state = adapter.replace(state, progress=torch.full_like(adapter.get(state, "progress"), 5.))
        action = state.new_zeros(len(state), 9); action[:, 6] = .4
        current = adapter.prepare_high(state, action)
        old_history = adapter.get(current, "low_history").reshape(-1, 10, 71)
        frames = []
        for index in range(8):
            observation, current = adapter.low_observation(current, context)
            frames.append(observation[:, :71])
            expected = torch.cat((old_history[:, index+1:], torch.stack(frames, 1)), 1)
            torch.testing.assert_close(adapter.get(current, "low_history").reshape(-1, 10, 71), expected)
            # Synthetic predicted q/object changes stand in for model outputs.
            # Neither next observation nor EKF is fetched from the simulator.
            obj = adapter.get(current, "object").clone(); obj[:, 0] += .001
            current = adapter.replace(current, object=obj, dof_pos=adapter.get(current, "dof_pos")+.001)
        finished = adapter.finish_high(state, current, action, context)
        poses = adapter.get(finished, "ekf_poses").reshape(-1, adapter.history, 7)
        torch.testing.assert_close(poses[:, -1], adapter.get(current, "object")[:, :7])
        torch.testing.assert_close(adapter.get(finished, "progress"), adapter.get(state, "progress")+1)
        torch.testing.assert_close(adapter.get(finished, "ekf_count"), (adapter.get(state, "ekf_count")+1).clamp_max(adapter.history))
        torch.testing.assert_close(adapter.get(finished, "prev_action"), action)
        high_before = adapter.observation(finished, context)
        low_before = adapter.low_observation(finished, context)[0]
        for name in ("_cube_root_states", "_robot_root_states", "_dof_pos", "_dof_vel", "low_obs_history_buf"):
            getattr(adapter.env, name).add_(100.)
        torch.testing.assert_close(adapter.observation(finished, context), high_before, rtol=0, atol=0)
        torch.testing.assert_close(adapter.low_observation(finished, context)[0], low_before, rtol=0, atol=0)

    def test_configuration_rejects_unknown_and_nonfinite_scales(self):
        for config in ({"invented": 1}, {"opening_temperature": 0}, {"success_gradient_scale": float("nan")}):
            with self.assertRaises(ValueError):
                resolve_reference_task_config(config)


if __name__ == "__main__":
    unittest.main()
