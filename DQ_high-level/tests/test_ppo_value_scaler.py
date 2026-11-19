"""PPO value-scaler statistics (skrl, DQ_WBC change) on CPU.

Upstream skrl fits the value scaler to the critic's own predictions and to the
returns. When the critic's normalized predictions spread wider than the returns,
the variance grows without bound and the normalized targets collapse, which is
what the first direction-8 runs showed (value loss falling to ~1e-6).
"""

from types import SimpleNamespace
import unittest

import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler


def agent(train_on_values):
    return SimpleNamespace(_value_preprocessor=RunningStandardScaler(1, device="cpu"),
                           _value_preprocessor_train_on_values=train_on_values)


class ValueScalerTargetTests(unittest.TestCase):
    def test_default_keeps_upstream_behaviour(self):
        self.assertTrue(PPO_DEFAULT_CONFIG["value_preprocessor_train_on_values"])
        generator = torch.Generator().manual_seed(0)
        values = 100.0 * torch.randn(512, 1, generator=generator)
        returns = 10.0 + 5.0 * torch.randn(512, 1, generator=generator)
        upstream = agent(True)
        normalized_values, normalized_returns = PPO._normalize_value_targets(upstream, values, returns)
        reference = RunningStandardScaler(1, device="cpu")
        torch.testing.assert_close(normalized_values, reference(values, train=True))
        torch.testing.assert_close(normalized_returns, reference(returns, train=True))
        torch.testing.assert_close(upstream._value_preprocessor.running_variance, reference.running_variance)

    def test_returns_only_statistics_and_one_shared_scale(self):
        generator = torch.Generator().manual_seed(1)
        values = 100.0 * torch.randn(512, 1, generator=generator)
        returns = 10.0 + 5.0 * torch.randn(512, 1, generator=generator)
        fixed = agent(False)
        normalized_values, normalized_returns = PPO._normalize_value_targets(fixed, values, returns)
        reference = RunningStandardScaler(1, device="cpu")
        torch.testing.assert_close(normalized_returns, reference(returns, train=True))
        torch.testing.assert_close(fixed._value_preprocessor.running_variance, reference.running_variance)
        torch.testing.assert_close(normalized_values, reference(values))

    def test_overdispersed_critic_inflates_only_the_upstream_statistics(self):
        """Critic predictions with 3x the spread of the targets, 50 updates."""
        final_std = {}
        for train_on_values in (True, False):
            generator = torch.Generator().manual_seed(2)
            state = agent(train_on_values)
            for _ in range(50):
                prediction = 3.0 * torch.randn(1000, 1, generator=generator)
                values = state._value_preprocessor(prediction, inverse=True)
                returns = 10.0 + 5.0 * torch.randn(1000, 1, generator=generator)
                PPO._normalize_value_targets(state, values, returns)
            final_std[train_on_values] = state._value_preprocessor.running_variance.sqrt().item()
        self.assertGreater(final_std[True], 50.0)
        self.assertLess(abs(final_std[False] - 5.0), 0.5)


class ValueDiagnosticsTests(unittest.TestCase):
    def test_explained_variance_and_scaler_statistics(self):
        returns = torch.linspace(-3.0, 3.0, 64)[:, None]
        for values, expected in ((returns.clone(), 1.0), (torch.zeros_like(returns), 0.0)):
            logged = {}
            state = agent(False)
            state.track_data = lambda tag, value: logged.__setitem__(tag, value)
            PPO._track_value_diagnostics(state, values, returns)
            self.assertAlmostEqual(logged["Value / explained variance"], expected, places=5)
            self.assertAlmostEqual(logged["Value / scaler mean"], 0.0)
            self.assertAlmostEqual(logged["Value / scaler std"], 1.0)
        logged = {}
        state = SimpleNamespace(_value_preprocessor=lambda x, **kwargs: x,
                                track_data=lambda tag, value: logged.__setitem__(tag, value))
        PPO._track_value_diagnostics(state, returns, returns)
        self.assertEqual(set(logged), {"Value / explained variance"})


if __name__ == "__main__":
    unittest.main()
