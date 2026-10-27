"""Write independent, timestamped object trajectories before environment resets."""

import csv
import json
import math
from pathlib import Path


class PoseTrajectoryRecorder:
    fields = ("timestamp", "x", "y", "z", "qx", "qy", "qz", "qw",
              "env_id", "episode_id", "sim_timestamp", "step_dt", "step",
              "terminated", "lifted")

    def __init__(self, output, num_envs):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.episode_ids = [0] * num_envs
        self.active = [None] * num_envs
        self.episodes = []
        self.last_time = None

    def append(self, sim_time, step_dt, poses, terminated, lifted, step):
        """Record post-physics/pre-reset poses. End flags close only their env."""
        if not math.isfinite(sim_time) or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Simulation time must be finite and step_dt positive")
        if self.last_time is not None and sim_time <= self.last_time:
            raise ValueError("Simulation time must strictly increase")
        if not len(poses) == len(terminated) == len(lifted) == len(self.active):
            raise ValueError("Batch size does not match the number of environments")
        self.last_time = sim_time
        for env_id, pose in enumerate(poses):
            pose = [float(x) for x in pose]
            if len(pose) != 7 or not all(math.isfinite(x) for x in pose):
                raise ValueError("Expected finite [x,y,z,qx,qy,qz,qw] poses")
            norm = math.sqrt(sum(x*x for x in pose[3:]))
            if norm < 1e-8:
                raise ValueError("Object has a zero quaternion")
            pose[3:] = [x / norm for x in pose[3:]]
            current = self.active[env_id]
            if current is None:
                episode_id = self.episode_ids[env_id]
                path = self.output / "env_{:04d}_episode_{:06d}.csv".format(env_id, episode_id)
                stream = path.open("x", newline="")
                writer = csv.writer(stream)
                writer.writerow(self.fields)
                current = dict(stream=stream, writer=writer, file=path.name, env_id=env_id,
                               episode_id=episode_id, start=sim_time, end=sim_time, samples=0)
                self.active[env_id] = current
            current["writer"].writerow([sim_time-current["start"]] + pose + [
                env_id, current["episode_id"], sim_time, step_dt, step,
                int(bool(terminated[env_id])), int(bool(lifted[env_id]))])
            current["stream"].flush()
            current["end"] = sim_time
            current["samples"] += 1
            if terminated[env_id]:
                self._finish(env_id, "terminated")

    def _finish(self, env_id, reason):
        current = self.active[env_id]
        if current is None:
            return
        current["stream"].close()
        self.episodes.append(dict(file=current["file"], env_id=env_id,
            episode_id=current["episode_id"], samples=current["samples"],
            duration_s=current["end"]-current["start"], end_reason=reason))
        self.episode_ids[env_id] += 1
        self.active[env_id] = None

    def close(self, reason="capture_end"):
        for env_id in range(len(self.active)):
            self._finish(env_id, reason)
        (self.output / "episodes.json").write_text(json.dumps(self.episodes, indent=2) + "\n")
