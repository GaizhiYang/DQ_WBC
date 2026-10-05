"""CPU-only observation/preprocessor checks; no Isaac Gym import is needed.

Run with: python -m unittest discover -s tests -p test_teacher_vision_io.py
"""

from pathlib import Path
import sys
import unittest

import gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DQ_high-level"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "skrl"))

from skrl.resources.preprocessors.torch import RunningStandardScaler
from utils.teacher_vision_preprocessor import FrozenRunningStandardScaler, PrefixRunningStandardScaler
from utils.teacher_vision_wrapper import TeacherVisionWrapper


class FakeCameraEnv:
    """Reuse the same tensors and dictionary just like simulator buffers."""

    def __init__(self):
        self.device = "cpu"
        self.num_envs = 2
        self.num_states = 62269
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1276,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (9,), np.float32)
        self.observations = {"obs": torch.zeros(2, 1276), "states": torch.zeros(2, 62269)}
        self.calls = 0
        self.reset_calls = 0
        self.last_actions = None
        self.info = {"time_outs": torch.tensor([False, True])}

    def _fill(self):
        self.calls += 1
        self.observations["obs"].fill_(self.calls)
        self.observations["states"][:, :62208].fill_(self.calls + 10)
        self.observations["states"][:, 62208:].fill_(999)

    def reset(self):
        self.reset_calls += 1
        self._fill()
        return self.observations

    def step(self, actions):
        self.last_actions = actions
        self._fill()
        return self.observations, torch.tensor([1., 2.]), torch.tensor([False, True]), self.info


class TeacherVisionWrapperTests(unittest.TestCase):
    def test_dimensions_fresh_storage_and_original_step_semantics(self):
        env = FakeCameraEnv()
        wrapped = TeacherVisionWrapper(env)
        first, info = wrapped.reset()
        self.assertEqual(first.shape, (2, 63484))
        self.assertEqual(wrapped.observation_space.shape, (63484,))
        self.assertEqual(info, {})
        self.assertIs(wrapped.action_space, env.action_space)
        torch.testing.assert_close(first[:, :1276], env.observations["obs"])
        torch.testing.assert_close(first[:, 1276:], env.observations["states"][:, :62208])
        saved = first.clone()
        actions = torch.zeros(2, 9)
        second, rewards, terminated, truncated, returned_info = wrapped.step(actions)
        self.assertIs(env.last_actions, actions)
        self.assertIs(returned_info, env.info)
        self.assertEqual(rewards.shape, (2, 1))
        torch.testing.assert_close(terminated, torch.tensor([[False], [True]]))
        torch.testing.assert_close(truncated, torch.zeros(2, 1, dtype=torch.bool))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, saved)
        second_saved = second.clone()
        third, _ = wrapped.reset()
        self.assertEqual(env.reset_calls, 2)
        self.assertNotEqual(second.data_ptr(), third.data_ptr())
        torch.testing.assert_close(first, saved)
        torch.testing.assert_close(second, second_saved)
        env.observations["obs"].zero_()
        env.observations["states"].zero_()
        torch.testing.assert_close(first, saved)
        torch.testing.assert_close(second, second_saved)

    def test_rejects_missing_camera_or_wrong_dimensions(self):
        env = FakeCameraEnv()
        env.num_states = 0
        with self.assertRaisesRegex(ValueError, "full camera"):
            TeacherVisionWrapper(env)
        env = FakeCameraEnv()
        wrapped = TeacherVisionWrapper(env)
        env.observations["states"] = torch.zeros(2, 62268)
        with self.assertRaisesRegex(ValueError, "states shape"):
            wrapped.reset()
        env = FakeCameraEnv()
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, (1275,), np.float32)
        with self.assertRaisesRegex(ValueError, "privileged observation"):
            TeacherVisionWrapper(env)


class TeacherVisionPreprocessorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_state_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)

    def test_original_checkpoint_keys_load_strict_and_freeze_explicit_train(self):
        original = RunningStandardScaler(size=1276, device="cpu")
        original(torch.randn(8, 1276) * 2 + 3, train=True)
        scaler = PrefixRunningStandardScaler(device="cpu")
        scaler.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(set(scaler.state_dict()), {"running_mean", "running_variance", "current_count"})
        self.assertEqual(scaler.running_mean.shape, (1276,))
        checkpoint = {name: value.clone() for name, value in scaler.state_dict().items()}
        privileged = torch.randn(4, 1276)
        # Values outside scaler clipping range detect accidental pixel scaling.
        images = torch.linspace(-20, 20, 62208).repeat(4, 1)
        packed = torch.cat((privileged, images), dim=-1)
        before = packed.clone()
        scaler.train()
        normalized = scaler(packed, train=True)
        torch.testing.assert_close(normalized[:, :1276], original(privileged))
        torch.testing.assert_close(normalized[:, 1276:], images, rtol=0, atol=0)
        torch.testing.assert_close(packed, before, rtol=0, atol=0)
        self.assert_state_equal(checkpoint, scaler.state_dict())
        inverse = scaler(normalized, inverse=True, train=True)
        torch.testing.assert_close(inverse[:, :1276], original(normalized[:, :1276], inverse=True))
        torch.testing.assert_close(inverse[:, 1276:], images, rtol=0, atol=0)
        self.assert_state_equal(checkpoint, scaler.state_dict())

    def test_unfreeze_updates_only_prefix_like_original_including_sequences(self):
        scaler = PrefixRunningStandardScaler(size=9, privileged_dim=3, device="cpu", freeze=False)
        original = RunningStandardScaler(size=3, device="cpu")
        packed = torch.randn(2, 4, 9)
        normalized = scaler(packed, train=True)
        expected = original(packed[..., :3], train=True)
        torch.testing.assert_close(normalized[..., :3], expected)
        torch.testing.assert_close(normalized[..., 3:], packed[..., 3:], rtol=0, atol=0)
        self.assert_state_equal(scaler.state_dict(), original.state_dict())
        self.assertGreater(scaler.current_count.item(), 1)
        scaler.freeze = True
        before = {name: value.clone() for name, value in scaler.state_dict().items()}
        scaler(packed * 10, train=True)
        self.assert_state_equal(before, scaler.state_dict())

    def test_value_scaler_checkpoint_and_runtime_unfreeze(self):
        original = RunningStandardScaler(size=1, device="cpu")
        original(torch.tensor([[1.], [3.], [5.]]), train=True)
        scaler = FrozenRunningStandardScaler(size=1, device="cpu")
        scaler.load_state_dict(original.state_dict(), strict=True)
        values = torch.tensor([[2.], [4.], [6.]])
        normalized = scaler(values, train=True)
        torch.testing.assert_close(normalized, original(values))
        self.assert_state_equal(scaler.state_dict(), original.state_dict())
        scaler.freeze = False
        torch.testing.assert_close(scaler(values, train=True), original(values, train=True))
        self.assert_state_equal(scaler.state_dict(), original.state_dict())

    def test_gradient_option_and_input_validation(self):
        scaler = PrefixRunningStandardScaler(size=9, privileged_dim=3, device="cpu")
        packed = torch.randn(4, 9, requires_grad=True)
        self.assertFalse(scaler(packed).requires_grad)
        scaler(packed, no_grad=False).sum().backward()
        torch.testing.assert_close(packed.grad[:, 3:], torch.ones(4, 6))
        with self.assertRaisesRegex(ValueError, "final dimension"):
            scaler(torch.zeros(4, 8))
        with self.assertRaisesRegex(ValueError, "image suffix"):
            PrefixRunningStandardScaler(size=3, privileged_dim=3, device="cpu")


if __name__ == "__main__":
    unittest.main()
