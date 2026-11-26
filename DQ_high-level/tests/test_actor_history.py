"""Teacher history/reset/PPO sample semantics without importing Isaac Gym."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gym
import numpy as np
import torch
import yaml

from test_teacher_gfm import Policy, Value
from test_teacher_entry import entry_function
from utils.config import apply_actor_history_config, apply_critic_history_config, get_params
from utils.wrapper import IsaacGymPreview3Wrapper
from skrl.memories.torch import RandomMemory
from utils.tsil import TemporalReplay, TemporalState, Trajectory
from utils.tsil_ppo import TSILPPO
from utils.tsil_replay_gpu import GPUTemporalReplay


ROOT = Path(__file__).resolve().parents[1]


class FakeTask:
    """DQ resets explicitly after terminal transitions, never inside step()."""
    device = "cpu"
    num_envs, num_states = 2, 0
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.obs = torch.zeros(2, 2)
        self.progress_buf = torch.zeros(2, dtype=torch.long)
        self.reset_buf = torch.ones(2, dtype=torch.long)
        self.episodes = torch.zeros(2)

    def reset(self):
        ids = self.reset_buf.bool()
        self.episodes[ids] += 1
        self.obs[ids] = self.episodes[ids, None] * 10
        self.progress_buf[ids] = 0
        self.reset_buf.zero_()
        return {"obs": self.obs}

    def step(self, actions):
        self.obs.add_(1)
        self.progress_buf.add_(1)
        return {"obs": self.obs}, torch.zeros(2), self.reset_buf, {}


class WrapperTests(unittest.TestCase):
    def test_order_partial_reset_and_no_aliasing(self):
        task = FakeTask()
        env = IsaacGymPreview3Wrapper(task, actor_history_frames=4)
        self.assertEqual(env.observation_space.shape, (8,))
        initial, _ = env.reset()
        snapshot = initial.clone()
        for _ in range(3):
            observation = env.step(None)[0]
        torch.testing.assert_close(observation.reshape(2, 4, 2)[0, :, 0],
                                   torch.tensor([10., 11., 12., 13.]))
        torch.testing.assert_close(initial, snapshot)
        terminal = observation.clone()
        task.reset_buf[0] = 1
        reset, _ = env.reset()
        torch.testing.assert_close(reset[0], torch.full((8,), 20.))
        torch.testing.assert_close(reset[1], terminal[1])
        torch.testing.assert_close(observation, terminal)
        # Re-reading after reset must not shift the continuing environment.
        torch.testing.assert_close(env.reset()[0], reset)
        following = env.step(None)[0].reshape(2, 4, 2)
        torch.testing.assert_close(following[0, :, 0], torch.tensor([20., 20., 20., 21.]))
        torch.testing.assert_close(following[1, :, 0], torch.tensor([11., 12., 13., 14.]))

    def test_single_frame_legacy_layout_and_invalid_configuration(self):
        env = IsaacGymPreview3Wrapper(FakeTask())
        self.assertEqual(env.observation_space.shape, (2,))
        self.assertEqual(env.reset()[0].shape, (2, 2))
        self.assertIsNone(env._actor_history.buffer)
        for frames in (0, -1, 1.5, True):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                IsaacGymPreview3Wrapper(FakeTask(), frames)

    def test_stack_holds_the_larger_of_actor_and_critic_frames(self):
        env = IsaacGymPreview3Wrapper(FakeTask(), actor_history_frames=1, critic_history_frames=4)
        self.assertEqual((env.history_frames, env.observation_space.shape), (4, (8,)))
        self.assertEqual(env.reset()[0].shape, (2, 8))
        for frames in (0, 1.5, True):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                IsaacGymPreview3Wrapper(FakeTask(), 1, frames)


class ModelTests(unittest.TestCase):
    @staticmethod
    def model(cls, frames, critic_frames=1):
        return cls(1285 * max(frames, critic_frames), 9, "cpu", 1024, 128, None, None, None,
                   num_future_pose_obs=13, critic_time_index=-203,
                   actor_history_frames=frames, critic_history_frames=critic_frames)

    def test_actor_encodes_all_four_frames_in_order_and_ignores_every_critic_clock(self):
        torch.manual_seed(19)
        actor = self.model(Policy, 4)
        single = self.model(Policy, 1)
        single.load_state_dict({k: (v if k != "net.0.weight" else v[:, -214:])
                                for k, v in actor.state_dict().items()})
        self.assertEqual(actor.net[0].in_features, 856)
        captures = {}
        actor.net.register_forward_pre_hook(lambda m, x: captures.update(stacked=x[0].detach()))
        single.net.register_forward_pre_hook(lambda m, x: captures.update(single=x[0].detach()))
        frames = torch.randn(3, 4, 1285) * .2
        actions = actor.compute({"states": frames.flatten(1)}, "policy")[0]
        single.compute({"states": frames.reshape(-1, 1285)}, "policy")
        torch.testing.assert_close(captures["stacked"], captures["single"].reshape(3, 856))
        modified = frames.clone()
        modified[:, :, -203] += 100
        torch.testing.assert_close(actor.compute({"states": modified.flatten(1)}, "policy")[0], actions)
        # PPO shuffles examples, but the four frames within each example stay intact.
        order = torch.tensor([2, 0, 1])
        shuffled = actor.compute({"states": frames[order].flatten(1)}, "policy")[0]
        torch.testing.assert_close(shuffled, actions[order])
        frames.requires_grad_()
        actor.compute({"states": frames.flatten(1)}, "policy")[0].sum().backward()
        for lag in range(4):
            self.assertGreater(frames.grad[:, lag, 1024:1030].abs().sum().item(), 0)
            self.assertEqual(frames.grad[:, lag, -203].abs().sum().item(), 0)

    def test_critic_is_identical_to_single_frame_and_checkpoint_roundtrip(self):
        critic, reference = self.model(Value, 4), self.model(Value, 1)
        reference.load_state_dict(critic.state_dict())
        self.assertEqual(critic.net[0].in_features, 215)
        frames = torch.randn(3, 4, 1285) * .2
        expected = reference.compute({"states": frames[:, -1]}, "value")[0]
        torch.testing.assert_close(critic.compute({"states": frames.flatten(1)}, "value")[0], expected)
        frames[:, :-1] *= 50
        torch.testing.assert_close(critic.compute({"states": frames.flatten(1)}, "value")[0], expected)
        actor, restored = self.model(Policy, 4), self.model(Policy, 4)
        restored.load_state_dict(actor.state_dict())
        torch.testing.assert_close(actor.compute({"states": frames.flatten(1)}, "policy")[0],
                                   restored.compute({"states": frames.flatten(1)}, "policy")[0])


    def test_critic_can_encode_all_four_frames_in_order(self):
        torch.manual_seed(23)
        critic, single = self.model(Value, 4, critic_frames=4), self.model(Value, 1)
        single.load_state_dict({k: (v if k != "net.0.weight" else v[:, -215:])
                                for k, v in critic.state_dict().items()})
        self.assertEqual(critic.net[0].in_features, 860)
        captures = {}
        critic.net.register_forward_pre_hook(lambda m, x: captures.update(stacked=x[0].detach()))
        single.net.register_forward_pre_hook(lambda m, x: captures.update(single=x[0].detach()))
        frames = torch.randn(3, 4, 1285) * .2
        critic.compute({"states": frames.flatten(1)}, "value")
        single.compute({"states": frames.reshape(-1, 1285)}, "value")
        torch.testing.assert_close(captures["stacked"], captures["single"].reshape(3, 860))
        frames.requires_grad_()
        critic.compute({"states": frames.flatten(1)}, "value")[0].sum().backward()
        for lag in range(4):
            self.assertGreater(frames.grad[:, lag, 1024:1030].abs().sum().item(), 0)
            # Unlike the actor, the critic reads its clock in every frame.
            self.assertGreater(frames.grad[:, lag, -203].abs().sum().item(), 0)

    def test_single_frame_actor_reads_only_the_newest_frame_of_a_critic_stack(self):
        actor, reference = self.model(Policy, 1, critic_frames=4), self.model(Policy, 1)
        reference.load_state_dict(actor.state_dict())
        self.assertEqual(actor.net[0].in_features, 214)
        frames = torch.randn(3, 4, 1285) * .2
        expected = reference.compute({"states": frames[:, -1]}, "policy")[0]
        torch.testing.assert_close(actor.compute({"states": frames.flatten(1)}, "policy")[0], expected)
        frames[:, :-1] *= 50
        torch.testing.assert_close(actor.compute({"states": frames.flatten(1)}, "policy")[0], expected)


class ConfigTests(unittest.TestCase):
    def test_saved_frame_count_and_explicit_cli_override(self):
        self.assertIsNone(get_params([]).actor_history_frames)
        self.assertEqual(apply_actor_history_config({"env": {}}, get_params([])), 1)
        cfg = {"env": {"actorHistoryFrames": 4}}
        self.assertEqual(apply_actor_history_config(cfg, get_params([])), 4)
        self.assertEqual(apply_actor_history_config(cfg, get_params(["--actor_history_frames", "1"])), 1)
        for value in (0, -3, True, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_actor_history_config({"env": {"actorHistoryFrames": value}}, get_params([]))

    def test_evaluation_restores_saved_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "checkpoints/agent_24.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            (run / "teacher_run.json").write_text(json.dumps({"arguments": {"actor_history_frames": 4}}))
            args = get_params(["--checkpoint", str(checkpoint)])
            entry_function("restore_eval_arguments")(args)
            self.assertEqual(args.actor_history_frames, 4)

    def test_critic_frames_default_to_one_and_follow_yaml_or_cli(self):
        self.assertIsNone(get_params([]).critic_history_frames)
        self.assertEqual(apply_critic_history_config({"env": {}}, get_params([])), 1)
        cfg = {"env": {"criticHistoryFrames": 4}}
        self.assertEqual(apply_critic_history_config(cfg, get_params([])), 4)
        self.assertEqual(apply_critic_history_config(cfg, get_params(["--critic_history_frames", "1"])), 1)
        for value in (0, -3, True, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_critic_history_config({"env": {"criticHistoryFrames": value}}, get_params([]))
        repository = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())
        self.assertEqual((apply_actor_history_config(repository, get_params([])),
                          apply_critic_history_config(repository, get_params([]))), (4, 4))

    def test_evaluation_takes_frames_from_the_run_not_the_passed_yaml(self):
        # Older runs: CLI values absent, the saved YAML has one actor frame and no critic key.
        for arguments, saved_env, expected in (
                ({"actor_history_frames": None}, {"actorHistoryFrames": 1}, (1, 1)),
                ({"actor_history_frames": 4, "critic_history_frames": 4}, {}, (4, 4))):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                checkpoint = run / "checkpoints/agent_24.pt"
                checkpoint.parent.mkdir()
                checkpoint.touch()
                (run / "teacher_run.json").write_text(json.dumps(
                    {"arguments": arguments, "config_file": "data/cfg/DQ_teacher.yaml"}))
                (run / "DQ_teacher.yaml").write_text(yaml.safe_dump({"env": saved_env}))
                args = get_params(["--checkpoint", str(checkpoint), "--critic_history_frames", "4"])
                entry_function("restore_eval_arguments")(args)
                self.assertEqual((args.actor_history_frames, args.critic_history_frames), expected)


class TemporalHistoryTests(unittest.TestCase):
    def test_normalizer_preserves_time_in_every_frame(self):
        width = 1287
        kwargs = dict(num_future_pose_obs=13, num_tsil_obs=2,
                      critic_time_index=-205, actor_history_frames=4)
        models = {name: cls(width * 4, 9, "cpu", 1024, 128, None, None, None, **kwargs)
                  for name, cls in (("policy", Policy), ("value", Value))}
        env = SimpleNamespace(tsil_state=TemporalState(1, 16.), num_envs=2, max_episode_length=4)
        agent = TSILPPO(models=models, memory=RandomMemory(2, 2, device="cpu"),
                        observation_space=width * 4, action_space=9, device="cpu", tsil_env=env,
                        cfg={"experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        states = torch.randn(2, width * 4)
        scaler = agent._state_preprocessor
        scaler.observe(states)
        scaler.commit()
        indices = [i - lag * width for lag in range(4) for i in (-191, -190)]
        torch.testing.assert_close(scaler(states)[:, indices], states[:, indices])
        self.assertEqual(agent.replay.target_indices,
                         tuple((i % (width * 4)) for i in indices[1::2]))

    def test_cpu_and_gpu_replay_relabel_all_historical_goals(self):
        # Four tiny frames: [physical value, elapsed fraction, episode goal].
        observations = torch.arange(24, dtype=torch.float).reshape(2, 12)
        original = observations.clone()
        sample = Trajectory((0, 1), 0, observations, torch.zeros(2, 1),
                            torch.ones(2), torch.zeros(2), torch.ones(2), False, 2., 2.)
        targets = [2, 5, 8, 11]
        for cls in (TemporalReplay, GPUTemporalReplay):
            with self.subTest(backend=cls.__name__):
                state = TemporalState(1, 10.)
                replay = cls(state, num_envs=1, observation_size=12, action_size=1,
                             time_indices=(10, 11), target_indices=targets,
                             static_prefix_size=0, max_episode_steps=4)
                replay.insert(sample)
                state.targets.fill_(5.)

                def screen(states, returns):
                    torch.testing.assert_close(states[:, targets], torch.full((len(states), 4), .5))
                    return torch.ones_like(returns)

                selection = replay.prepare(screen, transitions_per_trajectory=2)
                self.assertGreater(len(selection), 0)
                batch = selection.get_batch(torch.arange(len(selection)))
                torch.testing.assert_close(batch["states"][:, targets], torch.full((len(selection), 4), .5))
                torch.testing.assert_close(sample.observations, original)


if __name__ == "__main__":
    unittest.main()
