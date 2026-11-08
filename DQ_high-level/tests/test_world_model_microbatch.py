"""Unequal microbatches must implement the same model-fitting update."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model import HierarchicalDynamics, WMTrainerConfig, WorldModelTrainer
from utils.world_model.replay import WindowBatch


class Plant:
    low_dt = .02
    low_feature_dim = 3
    low_action_dim = 20
    context_dim = context_feature_dim = 2
    delta_dim = 3

    def low_features(self, state):
        return state[..., :3]

    def low_action_features(self, state, action):
        return action

    def normalize_context(self, context):
        return context

    def integrate_low(self, state, delta, action, context, dt):
        return torch.cat((state[..., :3] + dt * (delta + action[..., :3]), state[..., 3:]), -1)

    def with_contacts(self, state, contacts):
        return torch.cat((state[..., :3], contacts), -1)

    def get(self, state, name):
        assert name == "foot_contacts"
        return state[..., 3:]

    def error_vector(self, prediction, target):
        return prediction[..., :3] - target[..., :3]

    def auxiliary_loss(self, prediction, target):
        return (prediction[..., 0] + prediction[..., 1] - target[..., :2].sum(-1)).square()


class FixedReplay:
    def __init__(self, batch):
        self.batch = batch

    def can_sample(self, window, split):
        return split == "train"

    def sample_windows(self, *args, **kwargs):
        return self.batch


class MicrobatchChecks:
    device = "cpu"

    def batch(self, count=37, steps=5):
        states = torch.randn(steps + 1, count, 7, device=self.device) * .1
        states[..., 3:] = torch.randint(0, 2, (steps + 1, count, 4), device=self.device).float()
        actions = torch.randn(steps, count, 20, device=self.device) * .2
        scalar = torch.zeros(steps, count, device=self.device)
        ids = torch.arange(count, device=self.device)
        return WindowBatch(states, actions, torch.randn(steps, count, 2, device=self.device),
                           torch.full((steps, count, 1), .02, device=self.device),
                           scalar, scalar.bool(), scalar.bool(), scalar.bool(),
                           ids % 3, torch.zeros_like(ids), ids)

    def model(self):
        model = HierarchicalDynamics(Plant(), hidden=(32, 32), contact_hidden=(16,), device=self.device)
        # Exercise gradients throughout the main network on the first update,
        # rather than only testing the deliberately zero-initialized last layer.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * .01)
        return model

    def test_unequal_microbatches_match_full_batch_gradients_optimizer_and_metrics(self):
        self.compare_microbatches("mean")

    def test_sum_loss_unequal_microbatches_match_full_batch_update(self):
        self.compare_microbatches("sum")

    def compare_microbatches(self, reduction):
        torch.manual_seed(731)
        model = self.model()
        second = copy.deepcopy(model)
        common = dict(window=5, batch_size=37, batches_per_update=1, validation_batch_size=7,
                      grad_clip=.2, contact_weight=.3, auxiliary_weight=.7,
                      state_loss_reduction=reduction)
        full = WorldModelTrainer(model, WMTrainerConfig(**common, microbatch_size=0))
        chunked = WorldModelTrainer(second, WMTrainerConfig(**common, microbatch_size=8))
        replay = FixedReplay(self.batch())
        for _ in range(2):
            expected = full.update(replay)
            actual = chunked.update(replay)
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assertAlmostEqual(expected[name], actual[name], delta=2e-6, msg=name)
            for left, right in zip(full.model.parameters(), chunked.model.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(left.grad, right.grad, atol=2e-7, rtol=2e-5)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(full.optimizer.state[left][key],
                                               chunked.optimizer.state[right][key],
                                               atol=2e-8, rtol=2e-5)

    def test_late_nonfinite_microbatch_discards_earlier_partial_gradients(self):
        model = self.model()
        trainer = WorldModelTrainer(model, WMTrainerConfig(window=5, batch_size=37,
                    batches_per_update=1, microbatch_size=8, validation_batch_size=7))
        batch = self.batch()
        batch.states[-1, -1, 0] = float("nan")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        metrics = trainer.update(FixedReplay(batch))
        self.assertEqual(metrics["updates"], 0.)
        self.assertEqual(metrics["skipped_nonfinite"], 1.)
        self.assertFalse(trainer.optimizer.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


class CPUMicrobatchTests(MicrobatchChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAMicrobatchTests(MicrobatchChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
