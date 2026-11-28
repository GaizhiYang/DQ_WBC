#!/usr/bin/env python3
"""Capture, evaluate and summarize the current Level02/Level03 online EKF.

Run with the dqwbc Python environment. Each simulator runs sequentially on the
selected device. Capture/evaluation data is never overwritten; --skip-collect
reuses complete captures and evaluations and regenerates the summary files.

Example (from DQ_high-level):
  conda activate dqwbc
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python run_online_ekf_check.py --output outputs/ekf_check --device cuda:0
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def _simulator_environment():
    environment = os.environ.copy()
    # Absolute interpreter invocation need not imply an activated Conda PATH.
    # Isaac Gym's torch extension build must find that interpreter's ninja.
    environment["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + environment.get("PATH", "")
    return environment


def _complete_capture(directory, level, seed, num_envs, steps):
    metadata = json.loads((directory / "metadata.json").read_text())
    expected = dict(motion_level=level, seed=seed, num_envs=num_envs,
                    requested_steps=steps, completed_steps=steps, error=None)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Capture settings mismatch or incomplete run: " + str(directory))
    if not (directory / "capture.npz").is_file():
        raise ValueError("Missing capture.npz: " + str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 47])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.num_envs < 1 or args.steps < 12 or args.bootstrap_samples < 2:
        parser.error("num-envs >= 1, steps >= 12 and bootstrap-samples >= 2 are required")
    output = args.output.expanduser().resolve()
    if not args.skip_collect:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        parser.error("--skip-collect requires an existing output directory")
    # Import torch-dependent diagnostics only after simulator subprocesses;
    # the parent never imports Isaac Gym or consumes a simulation GPU context.
    directories = []
    for level in ("Level02", "Level03"):
        for seed in args.seeds:
            directory = output / (level.lower() + "_seed" + str(seed))
            if not args.skip_collect:
                command = [sys.executable, str(ROOT / "collect_online_ekf.py"),
                    "--level", level, "--seed", str(seed), "--num_envs", str(args.num_envs),
                    "--steps", str(args.steps), "--device", args.device, "--output", str(directory)]
                print("Collecting {} seed {} (log: {})".format(level, seed, output / (directory.name + ".log")),
                      flush=True)
                with (output / (directory.name + ".log")).open("x") as log:
                    subprocess.run(command, cwd=str(ROOT), env=_simulator_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            _complete_capture(directory, level, seed, args.num_envs, args.steps)
            directories.append(directory)
    from utils.ekf_motion_evaluation import evaluate_capture
    from utils.ekf_world_diagnostic import evaluate_world_capture
    from summarize_online_ekf import summarize
    for directory in directories:
        relative, world = directory / "relative_evaluation", directory / "world_diagnostic"
        if not relative.exists():
            evaluate_capture(directory, relative)
        elif not args.skip_collect or not all((relative / name).is_file() for name in
                ("summary.json", "raw_errors.npz", "metrics.csv", "episode_metrics.csv")):
            raise ValueError("Existing or incomplete evaluation: " + str(relative))
        if not world.exists():
            evaluate_world_capture(directory, world, device=args.device)
        elif not args.skip_collect or not all((world / name).is_file() for name in
                ("summary.json", "online_replay.json", "online_replay_errors.npz",
                 "evaluation/summary.json", "evaluation/raw_errors.npz",
                 "evaluation/metrics.csv", "evaluation/episode_metrics.csv")):
            raise ValueError("Existing or incomplete world diagnostic: " + str(world))
        print("Evaluated " + directory.name, flush=True)
    summary = summarize(output, args.bootstrap_samples, seeds=args.seeds)
    print(json.dumps(summary["counts"], indent=2))
    if not summary["all_replays_match"]:
        raise RuntimeError("Production EKF replay mismatch; inspect online_replay.json before interpreting metrics")
    print("Results: " + str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
