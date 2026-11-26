"""Real SGS+EKF rollout, PPO update and checkpoint roundtrip for actor/critic history.

Run from DQ_high-level with the launcher's teacher arguments, --num_envs 8,
--stop_after 24 (or 48), --headless and a temporary --experiment_dir; no --wandb.
This validates training mechanics, not convergence or task success.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaacgym import gymapi  # noqa: F401; must precede torch.
import torch

from train_multistate_DQ_teacher import get_trainer, finalize_teacher_logging


def main():
    trainer = get_trainer()
    env, agent = trainer.env, trainer.agents
    raw = env._env
    assert not trainer.teacher_args.wandb
    assert trainer.teacher_args.stop_after in (24, 48)
    assert env.actor_history_frames == 4 and raw.num_obs == 1285
    assert agent.policy.net[0].in_features == 856
    # The repository YAML gives the critic 4 frames too; --critic_history_frames 1 keeps one.
    critic_frames = env.critic_history_frames
    assert critic_frames in (1, 4) and env.history_frames == 4
    assert agent.value.net[0].in_features == 215 * critic_frames
    assert raw.sgs_enabled and raw.object_pose_predictor.horizons == (.6, .9)
    assert env.observation_space.shape == (5140,)
    counts = dict(steps=0, updates=0, resets=0)
    try:
        initial, _ = env.reset()
        frames = initial.reshape(raw.num_envs, 4, -1)
        torch.testing.assert_close(frames, frames[:, -1:].expand_as(frames))
        for _ in range(3):
            env.step(torch.zeros(raw.num_envs, 9, device=env.rl_device))
            if raw.reset_buf.any():
                env.reset()
        previous = env._actor_history.buffer.clone()
        raw.reset_idx(torch.tensor([0], device=raw.device))
        reset, _ = env.reset()
        reset_frames = reset.reshape(raw.num_envs, 4, -1)
        torch.testing.assert_close(reset_frames[0], reset_frames[0, -1:].expand(4, -1))
        torch.testing.assert_close(reset_frames[1:], previous[1:])

        weights = agent.policy.net[0].weight.detach().clone()
        original_step, original_reset, original_update = env.step, env.reset, agent._update

        def checked_step(actions):
            old = env._actor_history.buffer
            snapshot = old.clone()
            result = original_step(actions)
            current = result[0].reshape(raw.num_envs, 4, -1)
            torch.testing.assert_close(old, snapshot)
            torch.testing.assert_close(current[:, :-1], snapshot[:, 1:])
            assert torch.isfinite(current).all()
            counts["steps"] += 1
            return result

        def checked_reset():
            ids = raw.reset_buf.bool().clone()
            before = env._actor_history.buffer.clone()
            result = original_reset()
            current = result[0].reshape(raw.num_envs, 4, -1)
            torch.testing.assert_close(current[~ids], before[~ids])
            torch.testing.assert_close(current[ids], current[ids, -1:].expand(-1, 4, -1))
            counts["resets"] += int(ids.sum())
            return result

        def checked_update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            counts["updates"] += 1
            return result

        env.step, env.reset, agent._update = checked_step, checked_reset, checked_update
        trainer.train()
        finalize_teacher_logging(trainer)
        assert counts["updates"] == trainer.teacher_args.stop_after // 24
        assert not torch.equal(weights, agent.policy.net[0].weight)
        assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
        checkpoint = Path(agent.experiment_dir) / "checkpoints" / f"agent_{counts['steps']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint))
        states = env._actor_history.buffer.flatten(1)
        with torch.no_grad():
            expected = agent.policy.compute({"states": agent._state_preprocessor(states)}, "policy")[0]
            agent.load(str(checkpoint))
            actual = agent.policy.compute({"states": agent._state_preprocessor(states)}, "policy")[0]
        torch.testing.assert_close(actual, expected)
        report = dict(passed=True, actor_frames=4, actor_input=856, critic_frames=critic_frames,
                      critic_input=215 * critic_frames,
                      raw_frame=1285, stacked_observation=5140, partial_reset_checked=True,
                      checkpoint=str(checkpoint), **counts)
        (Path(agent.experiment_dir) / "actor_history_smoke.json").write_text(json.dumps(report, indent=2))
        print("ACTOR_HISTORY_SMOKE " + json.dumps(report))
    finally:
        if raw.viewer is not None:
            raw.gym.destroy_viewer(raw.viewer)
        raw.gym.destroy_sim(raw.sim)


if __name__ == "__main__":
    main()
