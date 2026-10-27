#!/usr/bin/env python3
"""Tune pose prediction on earlier episodes; report later episodes separately.

CPU-only example:
  python tune_object_pose_prediction.py --input outputs/sugar_box_pose \
      --output outputs/sugar_box_pose/tuning_short

Default objective: mean(position error / 1 cm + rotation error / 1 degree)
over 0.15 and 0.30 second horizons. Validation never selects parameters.
All comparisons use the same forecast origins, including when history varies.
The original Paparazzo equations and recorded ground truth remain unchanged.
"""

import argparse
import csv
import itertools
import json
import re
from copy import copy
from pathlib import Path

import numpy as np

from evaluate_object_pose_prediction import (
    add_evaluation_arguments, load_benchmark, metric_rows, poses_xyzw,
    validate_evaluation_arguments, write_metrics,
)


def split_episodes(paths):
    """Keep whole episodes apart, sharing the chronological split across envs."""
    episodes = {}
    for path in paths:
        match = re.search(r"episode_(\d+)$", path.stem)
        if match is None:
            raise ValueError("Unrecognized episode filename: " + path.name)
        episodes[path] = int(match.group(1))
    ids = sorted(set(episodes.values()))
    if len(ids) < 2:
        raise ValueError("Need at least two episode IDs; collect more steps across resets")
    calibration_ids = set(ids[:max(1, len(ids) // 2)])
    return ([p for p in paths if episodes[p] in calibration_ids],
            [p for p in paths if episodes[p] not in calibration_ids])


def short_horizon_score(position, rotation, horizons, max_horizon, pos_scale_cm, rot_scale_deg):
    selected = np.asarray(horizons) <= max_horizon + 1e-10
    if not selected.any():
        raise ValueError("No prediction horizons within the optimization horizon")
    return float(np.mean(position[:, selected] * 100 / pos_scale_cm
                         + rotation[:, selected] / rot_scale_deg))


def read_episodes(paths, benchmark, args, common_history, rng):
    records, skipped = [], []
    for path in paths:
        times, truth = benchmark.read_trajectory(path)
        if args.pre_lift_only:
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if "lifted" not in rows[0]:
                raise ValueError("--pre_lift_only requires a lifted column")
            stop = next((i for i, row in enumerate(rows) if int(row["lifted"])), len(rows))
            times, truth = times[:stop], truth[:stop]
        if (len(times) < common_history or
                times[-1] - times[common_history - 1] < max(args.steps) * args.prediction_dt - 1e-10):
            skipped.append(path.name)
            continue
        observations = benchmark.noisy_observations(truth, args.pos_noise, args.rot_noise_deg, rng)
        records.append(dict(path=path, times=times, truth=truth, observations=observations))
    if not records:
        raise ValueError("No episodes long enough for the common history and forecast horizons")
    return records, skipped


def evaluate_setting(records, benchmark, args, common_history, method):
    """Filter to shared origins after prediction, without changing its history."""
    positions, rotations, results = [], [], []
    method_index = benchmark.METHODS.index(method)
    # Apply stride after the common-origin cutoff; candidate histories cannot
    # shift the evaluated timestamps when --stride > 1.
    evaluation_args = copy(args)
    evaluation_args.stride = 1
    for record in records:
        result = benchmark.evaluate(record["times"], record["truth"], record["observations"], evaluation_args)
        origins = result[0]
        keep = (origins >= common_history - 1) & ((origins - common_history + 1) % args.stride == 0)
        result = tuple(value[keep] for value in result)
        positions.append(result[-2][:, method_index])
        rotations.append(result[-1][:, method_index])
        results.append(result)
    return np.concatenate(positions), np.concatenate(rotations), results


def plot_validation(path, rows, labels, optimize_horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = [("tab:blue", "--"), ("tab:blue", "-"), ("gray", ":"),
              ("tab:orange", "--"), ("tab:orange", "-")]
    for label, (color, style) in zip(labels, styles):
        selected = [row for row in rows if row["method"] == label]
        horizons = [row["horizon_s"] for row in selected]
        for ax, key in zip(axes, ("position_mean_cm", "rotation_mean_deg")):
            ax.plot(horizons, [row[key] for row in selected], marker="o",
                    color=color, linestyle=style, label=label)
    for ax, ylabel in zip(axes, ("Mean position error (cm)", "Mean rotation error (deg)")):
        ax.axvline(optimize_horizon, color="black", linestyle=":", alpha=.4)
        ax.set(xlabel="Forecast horizon (s)", ylabel=ylabel, ylim=(0, None))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Held-out episodes, identical forecast origins; selection used calibration episodes only")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tune(args):
    validate_evaluation_arguments(args)
    for name in ("history_grid", "velocity_window_grid"):
        values = sorted(set(getattr(args, name)))
        if min(values) < 2:
            raise ValueError(name + " must contain values >= 2")
        setattr(args, name, values)
    positive = args.q_scale_grid + args.r_scale_grid + [args.optimize_horizon, args.position_scale_cm, args.rotation_scale_deg]
    if not all(np.isfinite(x) and x > 0 for x in positive):
        raise ValueError("Scales and optimization horizon must be positive and finite")
    horizons = np.asarray(args.steps) * args.prediction_dt
    if not np.any(horizons <= args.optimize_horizon + 1e-10):
        raise ValueError("No prediction steps within --optimize_horizon")
    benchmark = load_benchmark(args.paparazzo_dir)
    paths = sorted((args.input / "trajectories").glob("env_*_episode_*.csv"))
    calibration_paths, validation_paths = split_episodes(paths)
    common_history = max(args.history, max(args.history_grid), max(args.velocity_window_grid))
    seeds = np.random.SeedSequence(args.seed).spawn(2)
    calibration, skipped_cal = read_episodes(calibration_paths, benchmark, args, common_history, np.random.default_rng(seeds[0]))
    validation, skipped_val = read_episodes(validation_paths, benchmark, args, common_history, np.random.default_rng(seeds[1]))
    args.output.mkdir(parents=True, exist_ok=False)
    split = dict(calibration=[r["path"].name for r in calibration],
                 validation=[r["path"].name for r in validation],
                 skipped_calibration=skipped_cal, skipped_validation=skipped_val,
                 common_origin_start_index=common_history - 1,
                 note="Chronological whole-episode split within one collection; not independent-seed testing")
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print("Calibration: {} episodes; validation: {} episodes; common start index: {}".format(
        len(calibration), len(validation), common_history - 1), flush=True)

    def score(p, r):
        return short_horizon_score(p, r, horizons, args.optimize_horizon,
                                   args.position_scale_cm, args.rotation_scale_deg)

    # Search only calibration episodes. Store all candidates, not only winners.
    trials, winners = [], {}
    ekf_candidates = list(itertools.product(args.history_grid, args.q_scale_grid, args.r_scale_grid))
    settings = [("ekf", dict(history=h, q_scale=q, r_scale=r)) for h, q, r in ekf_candidates]
    settings += [("constant_velocity", dict(history=max(args.history, w), velocity_window=w))
                 for w in args.velocity_window_grid]
    for index, (method, changes) in enumerate(settings):
        candidate = copy(args)
        for key, value in changes.items():
            setattr(candidate, key, value)
        p, r, _ = evaluate_setting(calibration, benchmark, candidate, common_history, method)
        loss = score(p, r)
        trial = dict(method=method, history=candidate.history, q_scale=candidate.q_scale,
                     r_scale=candidate.r_scale, velocity_window=candidate.velocity_window,
                     calibration_score=loss)
        trials.append(trial)
        if method not in winners or loss < winners[method][0]:
            winners[method] = (loss, candidate)
        if (index + 1) % 5 == 0 or index + 1 == len(settings):
            print("Scored {}/{} candidates on calibration episodes".format(index + 1, len(settings)), flush=True)
    write_metrics(args.output / "calibration_search.csv", trials)
    selected = {method: dict(history=winner.history, q_scale=winner.q_scale,
                r_scale=winner.r_scale, velocity_window=winner.velocity_window)
                for method, (_, winner) in winners.items()}
    (args.output / "selected_parameters.json").write_text(json.dumps(selected, indent=2) + "\n")

    comparisons = [("ekf_default", "ekf", args), ("ekf_tuned", "ekf", winners["ekf"][1]),
                   ("hold", "hold", args), ("constant_velocity_default", "constant_velocity", args),
                   ("constant_velocity_tuned", "constant_velocity", winners["constant_velocity"][1])]
    rows, scores, forecast_sets = [], [], []
    references = None
    for label, method, candidate in comparisons:
        p, r, results = evaluate_setting(validation, benchmark, candidate, common_history, method)
        if references is None:
            references = [result[:3] for result in results]
        else:
            for reference, result in zip(references, results):
                for expected, actual in zip(reference, result[:3]):
                    if not np.array_equal(expected, actual):
                        raise ValueError("Comparison must use identical origins and future references")
        rows.extend(metric_rows(p[:, None, :], r[:, None, :], args, [label]))
        scores.append(dict(method=label, validation_score=score(p, r)))
        method_index = benchmark.METHODS.index(method)
        forecast_sets.append([result[3][:, method_index] for result in results])
    write_metrics(args.output / "validation_metrics.csv", rows)
    write_metrics(args.output / "validation_scores.csv", scores)
    prediction_dir = args.output / "validation_predictions"
    prediction_dir.mkdir()
    labels = [item[0] for item in comparisons]
    for i, record in enumerate(validation):
        origins, target_times, targets = references[i]
        predictions = np.stack([setting[i] for setting in forecast_sets], axis=1)
        np.savez_compressed(prediction_dir / (record["path"].stem + ".npz"),
            timestamps=record["times"], truth=record["truth"], observations=record["observations"],
            origin_indices=origins, target_timestamps=target_times, targets=targets, predictions=predictions,
            predicted_poses=poses_xyzw(predictions), target_poses=poses_xyzw(targets),
            methods=np.array(labels), horizons_s=horizons)
    if args.max_plots:
        plot_validation(args.output / "validation_comparison.png", rows, labels, args.optimize_horizon)
    summary = dict(settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   split=split, selected_parameters=selected,
                   selection="Calibration only; weighted position and rotation errors over short horizons",
                   samples_per_horizon=rows[0]["samples"],
                   caveat="Validation is from one collection. Recheck on new seeds, policies, and observation noise before deployment.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Selected parameters:", json.dumps(selected, indent=2))
    print("Validation only: method                   horizon(s)   position(cm)   rotation(deg)")
    for row in rows:
        print("{method:<32}{horizon_s:>8.2f}{position_mean_cm:>15.4f}{rotation_mean_deg:>16.4f}".format(**row))
    print("Results:", args.output.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--seed", type=int, default=43)
    add_evaluation_arguments(parser)
    parser.add_argument("--history_grid", type=int, nargs="+", default=[3, 5, 8, 12, 20])
    parser.add_argument("--q_scale_grid", type=float, nargs="+", default=[1., 10.])
    parser.add_argument("--r_scale_grid", type=float, nargs="+", default=[0.01, 0.1, 1.])
    parser.add_argument("--velocity_window_grid", type=int, nargs="+", default=[2, 3, 5, 8, 12])
    parser.add_argument("--optimize_horizon", type=float, default=0.30)
    parser.add_argument("--position_scale_cm", type=float, default=1., help="Position-error normalization in selection score")
    parser.add_argument("--rotation_scale_deg", type=float, default=1., help="Rotation-error normalization in selection score")
    args = parser.parse_args()
    tune(args)


if __name__ == "__main__":
    main()
