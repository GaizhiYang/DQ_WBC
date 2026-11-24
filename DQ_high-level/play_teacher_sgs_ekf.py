#!/usr/bin/env python3
"""Replay a saved SGS/EKF teacher on full_task and save one MP4 per episode.

From any directory, after conda activate dqwbc:
  bash DQ_high-level/play_teacher_sgs_ekf.sh
  bash DQ_high-level/play_teacher_sgs_ekf.sh --checkpoint best --episodes 3
  bash DQ_high-level/play_teacher_sgs_ekf.sh --checkpoint 85000 --viewer

full_task is the original Level02 task: ordinary resets at 2 m, no SGS bank
sampling, and the training hold-success rule. Saved YAML/teacher_run.json,
policy, observation scaler and online reach learner are restored together.
"""

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = SCRIPT_DIR / "DQ_teacher/b1-pick-sugar-box-sgs-ekf/r_online_s43_1123"


def resolve_checkpoint(run_dir, selection):
    """Select numbered checkpoints numerically; never confuse best with latest."""
    checkpoint_dir = run_dir / "checkpoints"
    if selection == "latest":
        numbered = []
        for path in checkpoint_dir.glob("agent_*.pt"):
            match = re.fullmatch(r"agent_(\d+)\.pt", path.name)
            if match:
                numbered.append((int(match.group(1)), path))
        if not numbered:
            raise ValueError("No numbered checkpoints found in {}".format(checkpoint_dir))
        path = max(numbered)[1]
    elif selection == "best":
        path = checkpoint_dir / "best_agent.pt"
    elif selection.isdigit():
        path = checkpoint_dir / "agent_{}.pt".format(selection)
    else:
        path = Path(selection).expanduser()
        if not path.is_file() and not path.is_absolute():
            path = checkpoint_dir / path
    if not path.is_file():
        raise ValueError("Checkpoint not found: {}".format(path))
    return path.resolve()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN, help="Training run containing teacher_run.json and checkpoints/")
    parser.add_argument("--checkpoint", default="latest", help="latest (default), best, step number, filename, or checkpoint path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of complete episodes (default: 5)")
    parser.add_argument("--seed", type=int, default=43, help="Each episode uses seed + episode index (default: 43)")
    parser.add_argument("--device", default="cuda:0", help="Policy, simulation and rendering GPU (default: cuda:0)")
    parser.add_argument("--viewer", action="store_true", help="Also open the Isaac Gym viewer; default is offscreen recording")
    parser.add_argument("--fps", type=int, choices=(10, 25, 50), default=25, help="Real-time video FPS (default: 25)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera", choices=("wide", "close"), default="wide", help="Follow robot + object, or focus on the grasp")
    parser.add_argument("--output_dir", type=Path, help="New output directory; default: RUN/videos/full_task_CHECKPOINT_TIMESTAMP")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved checkpoint/config/output without loading Isaac Gym")
    return parser


class EpisodeVideo:
    """CPU camera readback, scheduled on the simulator clock at low-level ticks."""

    def __init__(self, raw, args):
        from isaacgym import gymapi
        self.raw, self.args, self.gymapi = raw, args, gymapi
        props = gymapi.CameraProperties()
        props.width, props.height = args.width, args.height
        props.horizontal_fov = 58.0
        self.camera = raw.gym.create_camera_sensor(raw.envs[0], props)
        if self.camera < 0:
            raise RuntimeError("Could not create recording camera; check the GPU graphics driver")
        self.writer = None
        self.frames = 0

    def start(self, path, episode):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(str(path), fps=self.args.fps, codec="libx264",
                                       quality=8, macro_block_size=1, pixelformat="yuv420p")
        self.episode = episode
        self.frames = 0
        self.start_time = float(self.raw.gym.get_sim_time(self.raw.sim))
        self.capture()

    def frame(self, status=None):
        import cv2
        import numpy as np
        raw, api = self.raw, self.gymapi
        robot = raw._robot_root_states[0, :3].detach().cpu().numpy()
        cube = raw._cube_root_states[0, :3].detach().cpu().numpy()
        if self.args.camera == "wide":
            target = (robot + cube) / 2
            target[2] = 0.46
            # Pull back if the moving table gets farther from the robot.
            scale = max(1.0, float(np.linalg.norm(robot[:2] - cube[:2])) / 2.4)
            eye = target + np.array([0.25, -3.25, 1.75]) * scale
        else:
            target = cube.copy()
            eye = target + np.array([-0.9, -1.25, 0.65])
        raw.gym.fetch_results(raw.sim, True)
        raw.gym.set_camera_location(self.camera, raw.envs[0], api.Vec3(*eye), api.Vec3(*target))
        raw.gym.step_graphics(raw.sim)
        raw.gym.render_all_camera_sensors(raw.sim)
        rgba = raw.gym.get_camera_image(raw.sim, raw.envs[0], self.camera, api.IMAGE_COLOR)
        rgb = np.asarray(rgba).reshape(self.args.height, self.args.width, 4)[..., :3].copy()
        elapsed = float(raw.gym.get_sim_time(raw.sim)) - self.start_time
        label = "full_task | episode {} | {:.2f} s".format(self.episode, elapsed)
        if status:
            label += " | " + status
        cv2.rectangle(rgb, (0, 0), (self.args.width, 64), (25, 25, 25), -1)
        for text, y in ((label, 27), (self.args.checkpoint.name, 52)):
            cv2.putText(rgb, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return rgb

    def capture(self):
        if self.writer is None:
            return
        elapsed = float(self.raw.gym.get_sim_time(self.raw.sim)) - self.start_time
        if elapsed + 1e-6 < self.frames / self.args.fps:
            return
        self.writer.append_data(self.frame())
        self.frames += 1

    def finish(self, status):
        # A one-second terminal still makes both successes and failures readable.
        frame = self.frame(status)
        for _ in range(self.args.fps):
            self.writer.append_data(frame)
        self.frames += self.args.fps
        self.close_writer()

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):
        self.close_writer()
        self.raw.gym.destroy_camera_sensor(self.raw.sim, self.raw.envs[0], self.camera)


def record(args, planned_timesteps):
    # Isaac Gym must load before torch, including indirect torch imports.
    from train_multistate_DQ_teacher import get_trainer
    import torch
    from skrl.utils import set_seed

    teacher_argv = ["--checkpoint", str(args.checkpoint), "--task", "B1Z1PickMulti",
                    "--object_name", "sugar_box", "--num_envs", "1",
                    "--eval_episodes", str(args.episodes), "--seed", str(args.seed),
                    "--timesteps", str(planned_timesteps),
                    "--rl_device", args.device, "--sim_device", args.device,
                    "--graphics_device_id", args.device.split(":")[1]]
    if not args.viewer:
        teacher_argv.append("--headless")
    trainer = get_trainer(is_eval=True, strict_eval=True, argv=teacher_argv, enable_camera_sensors=True)
    env, agent = trainer.env, trainer.agents
    raw = env._env
    video = None
    summary = {"checkpoint": str(args.checkpoint), "task": "full_task", "task_level": raw.table_level,
               "start_distance": 2.0, "sampler_mode": "disabled", "success_mode": "hold",
               "hold_steps": raw.hold_steps, "deterministic_policy": True,
               "action_scaling": raw.action_scaling, "fps": args.fps,
               "terminal_still_seconds": 1.0, "episodes": []}
    try:
        if raw.sgs_enabled or not raw.eval:
            raise RuntimeError("Recording requires full-task evaluation with SGS disabled")
        agent.set_mode("eval")
        agent.policy.deterministic = True
        for name in ("_state_preprocessor", "_value_preprocessor", "reward_normalizer"):
            module = getattr(agent, name, None)
            if hasattr(module, "eval"):
                module.eval()
        video = EpisodeVideo(raw, args)
        import yaml
        (args.output_dir / "environment.yaml").write_text(yaml.safe_dump(trainer.teacher_config, sort_keys=False))
        with torch.no_grad():
            for episode in range(1, args.episodes + 1):
                seed = (args.seed + episode - 1) % (2 ** 32)
                set_seed(seed)
                raw.reset_buf.fill_(1)
                states, _ = env.reset()
                path = args.output_dir / "episode_{:03d}.mp4".format(episode)
                video.start(path, episode)
                raw._video_frame_callback = video.capture
                elapsed, ever_lifted = 0.0, False
                for step in range(1, raw.max_episode_length + 1):
                    actions = agent.act(states, timestep=step - 1, timesteps=trainer.timesteps)[0]
                    states, _, terminated, truncated, _ = env.step(actions)
                    elapsed += raw.control_freq_low * raw.dt
                    ever_lifted |= bool(raw.lifted_object[0].item())
                    if bool((terminated | truncated).any().item()):
                        break
                else:
                    raise RuntimeError("Episode exceeded the configured time limit")
                raw._video_frame_callback = None
                success = bool(raw.episode_success[0].item())
                status = "SUCCESS" if success else ("TIMEOUT" if raw.timeout_buf[0].item() else "FAILED")
                video.finish(status)
                result = {"episode": episode, "seed": seed, "success": success,
                          "ever_lifted": ever_lifted, "status": status, "control_steps": step,
                          "simulation_seconds": elapsed, "video": path.name, "video_frames": video.frames}
                summary["episodes"].append(result)
                print("Episode {}/{}: {} ({:.2f}s) -> {}".format(episode, args.episodes, status, elapsed, path), flush=True)
    except BaseException as exc:
        summary["error"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        raw._video_frame_callback = None
        completed = len(summary["episodes"])
        successes = sum(item["success"] for item in summary["episodes"])
        summary.update(completed_episodes=completed, successes=successes,
                       success_rate=successes / completed if completed else None)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        try:
            if video is not None:
                video.close()
        finally:
            if raw.viewer is not None:
                raw.gym.destroy_viewer(raw.viewer)
            raw.gym.destroy_sim(raw.sim)
    print("Saved {} episodes, hold success {}/{}: {}".format(completed, successes, completed, args.output_dir), flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must be a CUDA device such as cuda:0")
    if min(args.width, args.height) < 64 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be even integers >= 64 (H.264)")
    args.run_dir = args.run_dir.expanduser().resolve()
    try:
        args.checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)
        # An explicit checkpoint path determines which saved configuration to use.
        args.run_dir = args.checkpoint.parent.parent
        metadata = json.loads((args.run_dir / "teacher_run.json").read_text())
        config_path = args.run_dir / Path(metadata.get("config_file", "DQ_teacher.yaml")).name
        if not config_path.is_file():
            raise ValueError("Saved teacher configuration not found: {}".format(config_path))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output_dir is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = args.run_dir / "videos" / "full_task_{}_{}".format(args.checkpoint.stem, stamp)
    args.output_dir = args.output_dir.expanduser().resolve()
    print("Checkpoint: {}\nSaved config: {}\nTask: full_task (Level02, 2 m, hold success)\nOutput: {}".format(
        args.checkpoint, config_path, args.output_dir), flush=True)
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Existing asset/low-level model paths are relative to DQ_high-level.
    os.chdir(str(SCRIPT_DIR))
    record(args, metadata.get("planned_timesteps", 120000))


if __name__ == "__main__":
    main()
