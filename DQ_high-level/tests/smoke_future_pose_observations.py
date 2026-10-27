"""Small real Isaac Gym rollout; run from DQ_high-level in the dqwbc env.

python tests/smoke_future_pose_observations.py
This performs no long training run and uses no W&B connection.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaacgym import gymapi
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler
from train_multistate_DQ_teacher import create_env, get_predict_point, Policy, Value
from envs.b1z1_base import B1Z1Base
from utils.config import get_params, load_cfg


def main():
    args = get_params([
        "--task", "B1Z1PickMulti", "--object_name", "sugar_box", "--num_envs", "4",
        "--headless", "--roboinfo", "--observe_gait_commands", "--small_value_set_zero",
        "--no_velocity_obs", "--future_pose_obs", "--rand_control",
    ])
    args.eval = False
    cfg = load_cfg("data/cfg/DQ_teacher.yaml")
    asset = cfg["env"]["asset"]["asset_multi"]["sugar_box"]
    cfg["env"].update(numEnvs=4, maxEpisodeLength=12, asset={**cfg["env"]["asset"], "asset_multi": {"sugar_box": asset}},
                      wandb=False, useTanh=False, noVelocityObs=True, totalTimesteps=10000)
    cfg["sensor"]["enableCamera"] = False
    camera, grasps, initial = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul", cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=4, intervel=23, delta_height=.1, object_indices=[asset["dict_idx"]])
    env = create_env(cfg, args, camera, grasps, initial)
    raw = env._env
    try:
        raw.reset_idx()
        obs, _ = env.reset()
        assert obs.shape == (4, 1284), obs.shape
        assert raw.num_future_pose_obs == 13
        assert raw.object_pose_predictor.horizons == (.3, .6)
        assert not raw.object_pose_predictor.valid.any()
        policy = Policy(env.observation_space, env.action_space, env.rl_device, 1024, 128, camera, grasps, initial).to(env.rl_device)
        value = Value(env.observation_space, env.action_space, env.rl_device, 1024, 128, camera, grasps, initial).to(env.rl_device)
        scaler = RunningStandardScaler(size=env.observation_space, device=env.rl_device)
        valid_samples = 0
        resets = 0
        for step in range(30):
            with torch.no_grad():
                obs, _, done, _, _ = env.step(torch.zeros((4, 9), device=env.rl_device))
                prediction = obs[:, -202:-189]
                assert prediction.shape == (4, 13)
                torch.testing.assert_close(prediction, raw.future_object_obs)
                assert torch.isfinite(obs).all()
                valid = prediction[:, -1].bool()
                valid_samples += int(valid.sum())
                if valid.any():
                    # Both z channels are predicted WORLD heights, not relative z.
                    torch.testing.assert_close(prediction[valid][:, [2, 8]],
                                               raw.object_pose_predictor.future_poses[valid, :, 2])
                if (~valid).any():
                    torch.testing.assert_close(prediction[~valid, :12].reshape(-1, 2, 6),
                                               obs[~valid, 1024:1030].unsqueeze(1).expand(-1, 2, -1))
                count = raw.object_pose_predictor.count.clone()
                raw._compute_observations()  # Reading observations is side-effect free for history.
                torch.testing.assert_close(raw.object_pose_predictor.count, count)
                normalized = scaler(obs, train=True)
                actions, _, _ = policy.compute({"states": normalized}, "policy")
                values, _ = value.compute({"states": normalized}, "value")
                assert actions.shape == (4, 9) and values.shape == (4, 1)
                assert torch.isfinite(actions).all() and torch.isfinite(values).all()
                if step == 6:
                    # An asynchronous reset must not clear another env's history.
                    other_history = raw.object_pose_predictor.poses[1:].clone()
                    raw.reset_idx(torch.tensor([0], device=raw.device))
                    assert raw.object_pose_predictor.count[0] == 1
                    assert raw.future_object_obs[0, -1] == 0
                    torch.testing.assert_close(raw.object_pose_predictor.poses[1:], other_history)
                    resets += 1
                if done.any():
                    ids = raw.reset_buf.nonzero().flatten()
                    resets += int(ids.numel())
                    obs, _ = env.reset()
                    assert (raw.object_pose_predictor.count[ids] == 1).all()
                    assert (raw.future_object_obs[ids, -1] == 0).all()
        assert valid_samples > 0 and resets > 1
        # Exercise the actual student slicing method without allocating cameras.
        camera_history = torch.zeros((4, 3, 2), device=raw.device)
        expected_student = torch.cat((camera_history.flatten(1), raw.obs_buf[:, 1030:-202], raw.obs_buf[:, -9:]), dim=-1)
        proxy = SimpleNamespace(num_actions=9, num_velocity_obs=0, num_future_pose_obs=13,
                                num_features=1024, num_envs=4, camera_history_buf=camera_history,
                                obs_buf=raw.obs_buf, states_buf=torch.empty_like(expected_student))
        B1Z1Base._compute_states_buf(proxy)
        torch.testing.assert_close(proxy.states_buf, expected_student)
        print("FUTURE_POSE_SMOKE " + json.dumps(dict(observation_dim=obs.shape[1], horizons=[.3, .6],
              valid_forecast_samples=valid_samples, resets=resets, policy_input=policy.net[0].in_features,
              student_proprioception_dim=expected_student.shape[1]-6, finite=True)))
    finally:
        if raw.viewer is not None:
            raw.gym.destroy_viewer(raw.viewer)
        raw.gym.destroy_sim(raw.sim)


if __name__ == "__main__":
    main()
