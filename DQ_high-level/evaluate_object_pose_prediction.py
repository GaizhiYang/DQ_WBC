#!/usr/bin/env python3
"""Evaluate recorded episodes with the existing Paparazzo benchmark.

CPU-only replay (does not import Isaac Gym or PyTorch):
  python evaluate_object_pose_prediction.py --input outputs/sugar_box_pose

Redraw saved predictions without recomputing them (backs up replaced PNGs):
  python evaluate_object_pose_prediction.py \
      --input outputs/sugar_box_pose/evaluation --replot_only
"""

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def poses_xyzw(transforms):
    """Convert [..., 4, 4] transforms into [..., 7] position + xyzw quaternion."""
    rotations = transforms[..., :3, :3].reshape(-1, 3, 3)
    quaternions = Rotation.from_matrix(rotations).as_quat().reshape(transforms.shape[:-2] + (4,))
    return np.concatenate((transforms[..., :3, 3], quaternions), axis=-1)


def add_evaluation_arguments(parser):
    parser.add_argument("--paparazzo_dir", type=Path, default=Path.home() / "Paparazzo")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--prediction_dt", type=float, default=0.15)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--velocity_window", type=int, default=5)
    parser.add_argument("--pos_noise", type=float, default=0.0, help="Observation noise std, meters per axis")
    parser.add_argument("--rot_noise_deg", type=float, default=0.0)
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--max_plots", type=int, default=3)
    parser.add_argument("--pre_lift_only", action="store_true", help="Evaluate only the prefix before first lifted=1")


def load_benchmark(directory):
    path = directory / "scripts" / "evaluate_pose_prediction.py"
    if not path.is_file():
        raise FileNotFoundError("Paparazzo benchmark not found: {}".format(path))
    spec = importlib.util.spec_from_file_location("paparazzo_pose_benchmark", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_evaluation_arguments(args):
    args.steps = sorted(set(args.steps))
    if args.history < 2 or args.velocity_window < 2 or args.stride < 1 or args.steps[0] < 1 or args.max_plots < 0:
        raise ValueError("history/velocity_window >=2; steps/stride >=1; max_plots >=0 required")
    if not all(np.isfinite(x) and x > 0 for x in (args.prediction_dt, args.q_scale, args.r_scale)):
        raise ValueError("Prediction interval and covariance scales must be positive and finite")
    if not all(np.isfinite(x) and x >= 0 for x in (args.pos_noise, args.rot_noise_deg)):
        raise ValueError("Noise standard deviations must be nonnegative and finite")


def metric_rows(position_error, rotation_error, args, methods, episode=None):
    rows = []
    for m, method in enumerate(methods):
        for h, step in enumerate(args.steps):
            p, r = position_error[:, m, h] * 100, rotation_error[:, m, h]
            row = dict(method=method, horizon_s=step*args.prediction_dt, samples=len(p),
                position_mean_cm=float(p.mean()), position_rmse_cm=float(np.sqrt(np.mean(p*p))),
                position_p95_cm=float(np.percentile(p, 95)), rotation_mean_deg=float(r.mean()),
                rotation_rmse_deg=float(np.sqrt(np.mean(r*r))), rotation_p95_deg=float(np.percentile(r, 95)))
            if episode is not None:
                row = dict(episode=episode, **row)
            rows.append(row)
    return rows


def write_metrics(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_recordings(input_dir, output_dir, args):
    validate_evaluation_arguments(args)
    benchmark = load_benchmark(args.paparazzo_dir)
    trajectory_dir = input_dir / "trajectories"
    paths = sorted(trajectory_dir.glob("env_*_episode_*.csv"))
    if not paths:
        raise ValueError("No recorded episodes in {}".format(trajectory_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir()
    rng = np.random.default_rng(args.seed)
    positions, rotations, episode_metrics, skipped = [], [], [], []
    evaluated = 0
    for path in paths:
        times, truth = benchmark.read_trajectory(path)
        if args.pre_lift_only:
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if "lifted" not in rows[0]:
                raise ValueError("--pre_lift_only requires a lifted column")
            stop = next((i for i, row in enumerate(rows) if int(row["lifted"])), len(rows))
            times, truth = times[:stop], truth[:stop]
        horizon = args.steps[-1] * args.prediction_dt
        if len(times) < args.history or times[-1] - times[args.history-1] < horizon - 1e-10:
            skipped.append(dict(file=path.name, samples=len(times), reason="too_short_for_history_and_horizon"))
            continue
        observations = benchmark.noisy_observations(truth, args.pos_noise, args.rot_noise_deg, rng)
        result = benchmark.evaluate(times, truth, observations, args)
        origins, target_times, targets, predictions, pe, re = result
        positions.append(pe)
        rotations.append(re)
        episode_metrics.extend(metric_rows(pe, re, args, benchmark.METHODS, path.stem))
        np.savez_compressed(predictions_dir / (path.stem + ".npz"), timestamps=times,
            truth=truth, observations=observations, origin_indices=origins,
            target_timestamps=target_times, targets=targets, predictions=predictions,
            predicted_poses=poses_xyzw(predictions), target_poses=poses_xyzw(targets),
            methods=np.array(benchmark.METHODS), horizons_s=np.array(args.steps)*args.prediction_dt,
            position_error_m=pe, rotation_error_deg=re)
        if evaluated < args.max_plots:
            benchmark.plot_result(output_dir / (path.stem + ".png"), path.stem, times, truth, observations, result, args)
        evaluated += 1
    summary = dict(input=str(input_dir.resolve()), evaluated_episodes=evaluated, skipped=skipped,
                   settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   aggregation="pooled forecast origins; histories never cross environments or episode boundaries",
                   prediction_axes=["forecast_origin", "method", "horizon", "row", "column"],
                   predicted_poses_axes=["forecast_origin", "method", "horizon", "pose_component"],
                   pose_components=["x", "y", "z", "qx", "qy", "qz", "qw"],
                   methods=list(benchmark.METHODS))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not evaluated:
        raise ValueError("All episodes are too short. Increase collection/episode length or reduce history/horizon; see summary.json")
    rows = metric_rows(np.concatenate(positions), np.concatenate(rotations), args, benchmark.METHODS)
    write_metrics(output_dir / "metrics.csv", rows)
    write_metrics(output_dir / "episode_metrics.csv", episode_metrics)
    print("\nEvaluated {} episodes; skipped {} short episodes.".format(evaluated, len(skipped)))
    print("method                horizon(s)  position(cm)  rotation(deg)")
    for row in rows:
        print("{method:<22}{horizon_s:>10.2f}{position_mean_cm:>14.4f}{rotation_mean_deg:>15.4f}".format(**row))
    print("Results:", output_dir.resolve())
    return rows


def replot_recordings(input_dir, output_dir, paparazzo_dir, max_plots):
    """Redraw saved results; CSV/NPZ values and evaluation settings are untouched."""
    if max_plots < 0:
        raise ValueError("max_plots must be nonnegative")
    summary = json.loads((input_dir / "summary.json").read_text())
    plot_args = argparse.Namespace(**summary["settings"])
    benchmark = load_benchmark(paparazzo_dir)
    paths = sorted((input_dir / "predictions").glob("env_*_episode_*.npz"))
    if not paths:
        raise ValueError("No saved predictions in " + str(input_dir / "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[:max_plots]:
        with np.load(path, allow_pickle=False) as data:
            ekf_index = list(data["methods"]).index("ekf")
            # The plotter consumes only EKF, placed at index zero as in the
            # original benchmark. No other baseline curves are drawn.
            result = (data["origin_indices"], data["target_timestamps"], data["targets"],
                      data["predictions"][:, ekf_index:ekf_index + 1], None, None)
            destination = output_dir / (path.stem + ".png")
            if destination.exists():
                backup = output_dir / "plots_before_replot" / destination.name
                backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    shutil.copy2(destination, backup)
            benchmark.plot_result(destination, path.stem, data["timestamps"], data["truth"],
                                  data["observations"], result, plot_args)
    print("Redrew {} plot(s): {}".format(min(len(paths), max_plots), output_dir.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Collection directory; with --replot_only, an existing evaluation directory")
    parser.add_argument("--output", type=Path, help="New output directory (default: INPUT/evaluation)")
    parser.add_argument("--replot_only", action="store_true", help="Redraw saved predictions; default output is INPUT and replaced PNGs are backed up")
    parser.add_argument("--seed", type=int, default=43)
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    if args.replot_only:
        replot_recordings(args.input, args.output or args.input, args.paparazzo_dir, args.max_plots)
    else:
        evaluate_recordings(args.input, args.output or args.input / "evaluation", args)


if __name__ == "__main__":
    main()
