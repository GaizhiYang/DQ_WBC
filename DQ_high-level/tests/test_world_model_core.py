"""Physics-independent invariants, also exercised on a real CUDA device.

These tests target reset leakage, ring overwrite, held-out episode ownership,
autoregressive gradients and optimizer ownership rather than MLP implementation
details. Run with the dqwbc Python; Isaac Gym is not needed for these checks.
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model import (ResidualDynamics, TransitionReplay,
                               WMTrainerConfig, WorldModelTrainer,
                               freeze_parameters)


class ToySpec:
    state_dim = feature_dim = 3
    action_dim = context_dim = 1
    delta_dim = 2
    termination_feature_dim = 1

    def features(self, state):
        return state

    def integrate(self, state, delta, action, context, dt):
        velocity = state[..., 1:2] + delta[..., :1]
        position = state[..., :1] + dt * velocity + delta[..., 1:2]
        return torch.cat((position, velocity, action), -1)

    def error_vector(self, predicted, target):
        return predicted - target

    def termination_features(self, state, context):
        return state[..., :1]


class CoreChecks:
    device = "cpu"

    def new_replay(self, envs=20, capacity=8, holdout=0.25):
        return TransitionReplay(envs, capacity, 3, 1, 1, self.device,
                                holdout_fraction=holdout)

    def populate(self, replay, steps, invalidate_at=None):
        n = replay.num_envs
        episodes = torch.zeros(n, device=self.device, dtype=torch.long)
        progress = torch.zeros(n, device=self.device)
        envs = torch.arange(n, device=self.device)
        for step in range(steps):
            states = torch.stack((episodes * 1000 + progress,
                                  envs.float(), torch.ones_like(progress)), -1)
            next_states = states.clone()
            next_states[:, 0] += 1
            done = (progress + 1 >= (envs % 4 + 2))
            valid = torch.ones(n, device=self.device, dtype=torch.bool)
            if step == invalidate_at:
                valid[1] = False
            replay.add(states, torch.full((n, 1), float(step), device=self.device),
                       next_states, envs[:, None].float(), 0.15,
                       torch.ones(n, device=self.device), done,
                       torch.zeros_like(done), envs % 3, episode_ids=episodes,
                       valid=valid, success=done & (envs % 2 == 0))
            # Mimic simulator buffer reuse: replay must own its copied values.
            states.fill_(-999)
            next_states.fill_(-999)
            progress += 1
            progress[done] = 0
            episodes += done.long()

    def test_windows_keep_terminal_states_and_never_cross_episode_or_ring(self):
        replay = self.new_replay()
        self.populate(replay, 17, invalidate_at=14)
        windows = replay.sample_windows(2048, 2, split="all")
        self.assertEqual(windows.states.device.type, torch.device(self.device).type)
        torch.testing.assert_close(windows.states[1:, :, 0] - windows.states[:-1, :, 0],
                                   torch.ones(2, 2048, device=self.device))
        self.assertFalse(bool(windows.terminated[:-1].any()))
        self.assertTrue(bool(windows.terminated[-1].any()))
        torch.testing.assert_close(windows.actions[1] - windows.actions[0],
                                   torch.ones(2048, 1, device=self.device))
        # The excluded row cannot appear at either end of a two-step window.
        bad = ((windows.env_ids == 1)[None]
               & (windows.actions[..., 0] == 14))
        self.assertFalse(bool(bad.any()))

    def test_validation_is_by_whole_episode_and_payloads_stay_on_device(self):
        replay = self.new_replay()
        self.populate(replay, 12)
        train = replay.sample_starts(4000, split="train")
        validation = replay.sample_starts(4000, split="validation")
        train_ids = set(zip(train.env_ids.cpu().tolist(), train.episode_ids.cpu().tolist()))
        validation_ids = set(zip(validation.env_ids.cpu().tolist(),
                                 validation.episode_ids.cpu().tolist()))
        self.assertTrue(train_ids)
        self.assertTrue(validation_ids)
        self.assertFalse(train_ids & validation_ids)
        for batch in (train, validation):
            for value in vars(batch).values():
                self.assertEqual(value.device.type, torch.device(self.device).type)

    def test_family_balancing_does_not_follow_family_population(self):
        replay = self.new_replay(envs=20, holdout=0.)
        states = torch.zeros(20, 3, device=self.device)
        family = torch.tensor([0] * 18 + [1, 2], device=self.device)
        replay.add(states, states[:, :1], states, states[:, :1], 0.15,
                   states[:, 0], states[:, 0].bool(), states[:, 0].bool(), family)
        starts = replay.sample_starts(12000)
        proportion = torch.bincount(starts.family, minlength=3).float() / 12000
        self.assertTrue(bool(((proportion - 1 / 3).abs() < 0.035).all()))

    def test_freezing_keeps_action_and_tail_value_gradients(self):
        torch.manual_seed(7)
        model = ResidualDynamics(ToySpec(), hidden=(16, 16), device=self.device)
        actor = torch.nn.Linear(3, 1).to(self.device)
        critic = torch.nn.Linear(3, 1).to(self.device)
        with torch.no_grad():
            # A fitted model generally has nonzero input Jacobians. Start this
            # test away from the deliberate zero-residual initialization.
            model.dynamics.net[-1].weight.fill_(0.2)
            critic.weight.fill_(0.3)
        model.dynamics.eval()  # mixed submodule mode must survive freezing
        state = torch.ones(16, 3, device=self.device)
        context = torch.zeros(16, 1, device=self.device)
        snapshot = [p.clone() for p in model.parameters()]
        with freeze_parameters(model, critic):
            first = model(state, actor(state), context, 0.15)
            second = model(first, actor(first), context, 0.15)
            # Reward-free objective isolates the entire tail value path.
            objective = critic(second).mean()
            objective.backward()
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(model.training)
        self.assertFalse(model.dynamics.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertGreater(float(actor.weight.grad.abs().sum()), 1e-5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        for before, after in zip(snapshot, model.parameters()):
            torch.testing.assert_close(before, after)

    def test_fitting_reduces_true_heldout_multistep_error(self):
        torch.manual_seed(12)
        replay = self.new_replay(envs=64, capacity=12)
        states = torch.randn(64, 3, device=self.device) * 0.1
        context = torch.zeros(64, 1, device=self.device)
        for _ in range(10):
            action = torch.randn(64, 1, device=self.device) * 0.3
            velocity = states[:, 1:2] + 0.4 * action
            future = torch.cat((states[:, :1] + 0.15 * velocity, velocity, action), -1)
            replay.add(states, action, future, context, 0.15,
                       torch.zeros(64, device=self.device),
                       torch.zeros(64, device=self.device, dtype=torch.bool),
                       torch.zeros(64, device=self.device, dtype=torch.bool),
                       torch.arange(64, device=self.device) % 3)
            states = future
        model = ResidualDynamics(ToySpec(), hidden=(32, 32), device=self.device)
        trainer = WorldModelTrainer(model, WMTrainerConfig(window=3, batches_per_update=80,
                                    batch_size=64, learning_rate=3e-3,
                                    termination_weight=0., validation_batch_size=1024))
        before = trainer.validate(replay)["validation_nse_kstep"]
        metrics = trainer.update(replay)
        self.assertLess(metrics["validation_nse_kstep"], before * 0.35)
        self.assertEqual(metrics["updates"], 80.)
        batch = replay.sample_windows(64, 3, split="validation")
        # Perturb a target at t=1: predictions must remain autoregressive.
        predictions = []
        hook = model.register_forward_hook(lambda module, inputs, output:
                                           predictions.append(output.detach().clone()))
        loss, _ = trainer.loss_on(batch)
        self.assertTrue(torch.isfinite(loss))
        baseline_predictions = predictions[:]
        predictions.clear()
        batch.states[1].add_(100.)
        trainer.loss_on(batch)
        hook.remove()
        self.assertEqual(len(predictions), 3)
        for baseline, changed in zip(baseline_predictions, predictions):
            torch.testing.assert_close(baseline, changed)
        original_state_dict = trainer.state_dict()
        replacement = WorldModelTrainer(ResidualDynamics(ToySpec(), hidden=(32, 32),
                                                         device=self.device), trainer.cfg)
        replacement.load_state_dict(original_state_dict)
        self.assertEqual(replacement.updates, trainer.updates)


class CPUWorldModelTests(CoreChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAWorldModelTests(CoreChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
