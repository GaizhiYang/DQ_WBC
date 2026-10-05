"""CPU integration of task metadata, camera perception and episode resets."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.teacher_vision import PACKED_OBS_DIM, PRIVILEGED_OBS_DIM
from utils.asymmetric_teacher_wrapper import AsymmetricTeacherWrapper
from test_teacher_vision_training import ReusingCameraEnv


class TaskCameraEnv(ReusingCameraEnv):
    """Real wrapper contract with synthetic cameras and independent GT."""

    def __init__(self):
        super().__init__()
        self.progress_buf = torch.tensor([0, 5])
        self.reset_buf = torch.zeros(2, dtype=torch.bool)
        self.max_episode_length = 10
        self.closest_dist = torch.tensor([-1., .8])
        self.curr_dist = torch.tensor([.6, .2])
        self.highest_object = torch.tensor([-1., .7])
        self.curr_height = torch.tensor([.1, .2])
        self.pick_counter = torch.tensor([0., 15.])
        self.hold_steps = 25
        self.lifted_object = torch.tensor([False, True])
        self.eval = False
        self.sensor_times = torch.zeros(2, dtype=torch.float64)
        self.sensor_images = torch.zeros(2, 2, 2, 54, 96)
        self.sensor_images[:, :, 0, 22:32, 43:53] = 1
        self.sensor_images[:, 0, 1] = 1
        self.sensor_images[:, 1, 1] = 1.5
        self.camera_transforms = torch.eye(4).repeat(2, 2, 1, 1)
        self.camera_intrinsics = torch.tensor([[48., 0, 48.], [0, 48., 27.], [0, 0, 1.]]).repeat(2, 2, 1, 1)
        self.base_transform = torch.eye(4).repeat(2, 1, 1)
        self._cube_root_states = torch.zeros(2, 13)
        self._cube_root_states[:, 6] = 1
        self._m2_reward_capture_object_states = self._cube_root_states.clone()

    def get_asymmetric_sensor_packet(self):
        return {
            "images": self.sensor_images.clone(),
            "T_world_camera": self.camera_transforms.clone(),
            "intrinsics": self.camera_intrinsics.clone(),
            "timestamps": self.sensor_times[:, None].expand(-1, 2).clone(),
            "now": self.sensor_times.clone(),
            "T_world_base": self.base_transform.clone(),
        }

    def reset_idx(self, env_ids=None):
        ids = torch.arange(2) if env_ids is None else env_ids
        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.sensor_times[ids] = 0

    def reset(self):
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        return self._observe()

    def step(self, actions):
        self.sensor_times += .16
        self.progress_buf += 1
        observations, rewards, _, info = super().step(actions)
        return observations, rewards, self.reset_buf.clone(), info


def camera_settings(mode="m2", **reward):
    return {
        "mode": mode, "seed": 43,
        "perception": {
            "camera_delay_frames": [0, 0],
            "depth_noise_std_m": 0., "depth_noise_distance_scale": 0., "depth_noise_angular_scale": 0.,
            "pixel_dropout_prob": 0., "frame_dropout_prob": 0.,
            "frame_dropout_distance_scale": 0., "frame_dropout_angular_scale": 0.,
            "small_target_dropout_scale": 0.,
        },
        "perception_reward": dict({"weight": .02, "sigma_m": .1, "prediction_horizon_s": .15}, **reward),
    }


class AsymmetricTeacherWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_m0_uses_original_fresh_packed_observation_and_reward(self):
        raw = ReusingCameraEnv()
        env = AsymmetricTeacherWrapper(raw)
        states, _ = env.reset()
        before = states.clone()
        next_states, rewards, terminated, truncated, _ = env.step(torch.zeros(2, 9))
        self.assertEqual(env.observation_space.shape, (PACKED_OBS_DIM,))
        torch.testing.assert_close(states, before, atol=0, rtol=0)
        torch.testing.assert_close(next_states, raw.snapshots[-1], atol=0, rtol=0)
        torch.testing.assert_close(rewards[:, 0], torch.tensor([.04, .08]))
        self.assertFalse(terminated.any() or truncated.any())

    def test_m1_appends_normalized_task_state_with_sentinel_fallbacks(self):
        raw = TaskCameraEnv()
        env = AsymmetricTeacherWrapper(raw, camera_settings("m1"))
        states, _ = env.reset()
        self.assertEqual(env.observation_space.shape, (PACKED_OBS_DIM + 5,))
        expected = torch.tensor([[1., .3, .1, 0., 0.], [.5, .4, .7, .6, 1.]])
        torch.testing.assert_close(states[:, -5:], expected)
        torch.testing.assert_close(states[:, :PACKED_OBS_DIM], raw.snapshots[-1], atol=0, rtol=0)
        before = states.clone()
        raw.closest_dist.fill_(9)
        raw.highest_object.fill_(1.5)
        raw.pick_counter.fill_(100)
        raw.progress_buf.fill_(20)
        updated, _ = env.reset()
        torch.testing.assert_close(states, before, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(updated).all())
        torch.testing.assert_close(updated[:, -5], torch.zeros(2))
        torch.testing.assert_close(updated[:, -2], torch.ones(2))

    def test_m2_camera_order_depth_units_and_belief_are_packed_in_fresh_storage(self):
        raw = TaskCameraEnv()
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        states, _ = env.reset()
        self.assertEqual(env.observation_space.shape, (PACKED_OBS_DIM + 21,))
        self.assertTrue(torch.isfinite(states).all())
        history = states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM].reshape(2, 3, 4, 54, 96)
        mask_base, mask_wrist = raw.sensor_images[:, 0, 0], raw.sensor_images[:, 1, 0]
        expected = torch.stack((mask_base, mask_wrist, mask_base / 3, mask_wrist / 2), dim=1)
        for frame in range(3):
            torch.testing.assert_close(history[:, frame], expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(states[:, -3], torch.ones(2)))  # belief initialized
        before = states.clone()
        env.step(torch.zeros(2, 9))
        torch.testing.assert_close(states, before, atol=0, rtol=0)

    def test_task_and_object_gt_only_change_task_metadata_and_training_reward(self):
        first_raw, second_raw = TaskCameraEnv(), TaskCameraEnv()
        second_raw.closest_dist += 1
        second_raw.highest_object += 1
        second_raw._cube_root_states[:, :3] += 10
        first = AsymmetricTeacherWrapper(first_raw, camera_settings())
        second = AsymmetricTeacherWrapper(second_raw, camera_settings())
        first.reset()
        second.reset()
        left, left_reward, _, _, _ = first.step(torch.zeros(2, 9))
        right, right_reward, _, _, _ = second.step(torch.zeros(2, 9))
        torch.testing.assert_close(left[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM],
                                   right[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM], atol=0, rtol=0)
        torch.testing.assert_close(left[:, -16:], right[:, -16:], atol=0, rtol=0)
        self.assertFalse(torch.equal(left[:, PACKED_OBS_DIM:PACKED_OBS_DIM + 5],
                                     right[:, PACKED_OBS_DIM:PACKED_OBS_DIM + 5]))
        self.assertTrue((left_reward > right_reward).all())

    def test_perception_bonus_only_affects_training_reward(self):
        training_raw, evaluation_raw = TaskCameraEnv(), TaskCameraEnv()
        evaluation_raw.eval = True
        training = AsymmetricTeacherWrapper(training_raw, camera_settings(weight=.5))
        evaluation = AsymmetricTeacherWrapper(evaluation_raw, camera_settings(weight=.5))
        training.reset()
        evaluation.reset()
        train_state, train_reward, _, _, train_info = training.step(torch.zeros(2, 9))
        eval_state, eval_reward, _, _, eval_info = evaluation.step(torch.zeros(2, 9))
        torch.testing.assert_close(train_state, eval_state, atol=0, rtol=0)
        torch.testing.assert_close(eval_reward[:, 0], torch.tensor([.04, .08]))
        self.assertTrue((train_reward > eval_reward).all())
        self.assertGreater(train_info["perception"]["reward_mean"], 0)
        self.assertEqual(train_info["perception"], eval_info["perception"])

    def test_repeated_reset_does_not_refilter_or_advance_camera_history(self):
        env = AsymmetricTeacherWrapper(TaskCameraEnv(), camera_settings())
        env.reset()
        stepped, *_ = env.step(torch.zeros(2, 9))
        with mock.patch.object(env.perception, "step", wraps=env.perception.step) as update:
            repeated, _ = env.reset()
        update.assert_not_called()
        torch.testing.assert_close(repeated, stepped, atol=0, rtol=0)

    def test_partial_reset_clears_only_ended_robot_and_full_reset_clears_both(self):
        raw = TaskCameraEnv()
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        env.reset()
        for _ in range(3):
            previous, *_ = env.step(torch.zeros(2, 9))
        raw.reset_buf[0] = True
        with mock.patch.object(env.perception, "reset", wraps=env.perception.reset) as reset:
            current, _ = env.reset()
        self.assertEqual(reset.call_count, 1)
        torch.testing.assert_close(reset.call_args[0][0], torch.tensor([0]))
        torch.testing.assert_close(current[1], previous[1], atol=0, rtol=0)
        self.assertEqual(raw.sensor_times[0], 0)
        self.assertGreater(raw.sensor_times[1], 0)
        raw.reset_idx()
        env.reset_runtime()
        restarted, _ = env.reset()
        clean = AsymmetricTeacherWrapper(TaskCameraEnv(), camera_settings())
        expected, _ = clean.reset()
        # Original privileged state is intentionally independent of sensor
        # history in this fake; compare the full visual+belief deployment input.
        torch.testing.assert_close(restarted[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM],
                                   expected[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM], atol=0, rtol=0)
        torch.testing.assert_close(restarted[:, -16:], expected[:, -16:], atol=0, rtol=0)

    def test_missing_target_depth_yields_finite_uninitialized_belief_and_no_bonus(self):
        raw = TaskCameraEnv()
        raw.sensor_images.zero_()
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        env.reset()
        states, reward, _, _, info = env.step(torch.zeros(2, 9))
        self.assertTrue(torch.isfinite(states).all())
        self.assertEqual(torch.count_nonzero(states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM]).item(), 0)
        self.assertEqual(torch.count_nonzero(states[:, -3]).item(), 0)
        torch.testing.assert_close(reward[:, 0], torch.tensor([.04, .08]))
        self.assertEqual(info["perception"]["reward_mean"], 0.)

    def test_perception_rng_snapshot_and_seed_are_independent_of_global_rng(self):
        env = AsymmetricTeacherWrapper(TaskCameraEnv(), camera_settings())
        original = env.get_runtime_rng_state().clone()
        expected = torch.rand(4, generator=env.perception.generator)
        env.set_runtime_rng_state(original)
        torch.manual_seed(999)
        torch.testing.assert_close(torch.rand(4, generator=env.perception.generator), expected, atol=0, rtol=0)
        env.seed_runtime(123)
        first = torch.rand(4, generator=env.perception.generator)
        env.seed_runtime(123)
        torch.testing.assert_close(torch.rand(4, generator=env.perception.generator), first, atol=0, rtol=0)

    def test_float32_sensor_clocks_are_accepted(self):
        raw = TaskCameraEnv()
        raw.sensor_times = raw.sensor_times.float()
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        states, _ = env.reset()
        self.assertTrue(torch.isfinite(states).all())
        stepped, *_ = env.step(torch.zeros(2, 9))
        self.assertTrue(torch.isfinite(stepped).all())

    def test_camera_world_and_current_base_transforms_reach_belief_in_metres(self):
        raw = TaskCameraEnv()
        raw.sensor_images[:, :, 1] = 1
        raw.camera_transforms[:, :, 0, 3] = 2
        raw.base_transform[:, 0, 3] = 1
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        states, _ = env.reset()
        offset = -.5 / 48
        expected = torch.tensor([1 + offset, offset, 1.]).repeat(2, 1)
        torch.testing.assert_close(states[:, -16:-13], expected, atol=1e-6, rtol=1e-6)
        raw.base_transform[:, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated, *_ = env.step(torch.zeros(2, 9))
        expected_rotated = torch.stack((expected[:, 1], -expected[:, 0], expected[:, 2]), dim=-1)
        torch.testing.assert_close(rotated[:, -16:-13], expected_rotated, atol=1e-6, rtol=1e-6)

    def test_repeated_capture_timestamp_is_not_presented_as_a_new_visible_frame(self):
        raw = TaskCameraEnv()
        original_packet = raw.get_asymmetric_sensor_packet

        def repeated_capture():
            packet = original_packet()
            packet["timestamps"].zero_()
            return packet

        raw.get_asymmetric_sensor_packet = repeated_capture
        env = AsymmetricTeacherWrapper(raw, camera_settings())
        env.reset()
        states, *_ = env.step(torch.zeros(2, 9))
        belief = states[:, -16:]
        torch.testing.assert_close(belief[:, 9:11], torch.zeros(2, 2))
        torch.testing.assert_close(belief[:, 11:13], torch.full((2, 2), .16))
        self.assertTrue((belief[:, 13] == 1).all())
        newest = states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM].reshape(2, 3, 4, 54, 96)[:, -1]
        self.assertEqual(torch.count_nonzero(newest).item(), 0)


if __name__ == "__main__":
    unittest.main()
