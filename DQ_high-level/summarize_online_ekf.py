#!/usr/bin/env python3
"""Pool captured Level02/03 forecast evaluations and draw a comparison figure.

Usage: python summarize_online_ekf.py outputs/level03_ekf_check_20261127
Expects level02_seed*/level03_seed* directories with relative_evaluation and
world_diagnostic/evaluation results. Does not run or modify the simulator.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_episode_bootstrap(position_cm, rotation_deg, episode_keys, methods, horizons,
                             samples=2000, seed=2026):
    """Paired EKF-minus-baseline means, weighting each sampled episode equally.

    An episode key is (capture, environment, episode). Overlapping origins are
    averaged within the episode before resampling. These intervals describe the
    sampled episodes, not uncertainty across new policies, sensors or seeds.
    """
    if samples < 2:
        raise ValueError("bootstrap samples must be at least 2")
    _, group = np.unique(episode_keys, axis=0, return_inverse=True)
    episodes = int(group.max()) + 1
    errors = np.stack((position_cm, rotation_deg), axis=-1)
    means = np.stack([errors[group == index].mean(axis=0) for index in range(episodes)])
    delta = means[:, :1] - means[:, 1:]
    rng = np.random.default_rng(seed)
    resampled = np.empty((samples,) + delta.shape[1:])
    for start in range(0, samples, 128):
        end = min(start + 128, samples)
        indices = rng.integers(0, episodes, size=(end - start, episodes))
        resampled[start:end] = delta[indices].mean(axis=1)
    lower, upper = np.percentile(resampled, [2.5, 97.5], axis=0)
    average = delta.mean(axis=0)
    rows = []
    for baseline_index, baseline in enumerate(methods[1:]):
        for h_index, horizon in enumerate(horizons):
            for metric_index, metric in enumerate(("position_cm", "rotation_deg")):
                rows.append(dict(method=str(methods[0]), baseline=str(baseline),
                    horizon_s=float(horizon), metric=metric, episodes=episodes,
                    mean_difference=float(average[baseline_index, h_index, metric_index]),
                    ci95_lower=(float(lower[baseline_index, h_index, metric_index]) if episodes > 1 else None),
                    ci95_upper=(float(upper[baseline_index, h_index, metric_index]) if episodes > 1 else None)))
    return rows


def _metric_rows(position, rotation, methods, horizons, **identifiers):
    rows = []
    for method_index, method in enumerate(methods):
        for h_index, horizon in enumerate(horizons):
            p, r = position[:, method_index, h_index], rotation[:, method_index, h_index]
            rows.append(dict(identifiers, method=str(method), horizon_s=float(horizon), samples=len(p),
                position_mean_cm=float(p.mean()), position_rmse_cm=float(np.sqrt(np.mean(p*p))),
                position_p95_cm=float(np.percentile(p, 95)), within_2cm_fraction=float(np.mean(p <= 2)),
                rotation_mean_deg=float(r.mean()), rotation_p95_deg=float(np.percentile(r, 95))))
    return rows


def summarize(directory, bootstrap_samples=2000, plot=True, seeds=None):
    """Summarize all matching captures, or exactly the requested seed set."""
    root = Path(directory).resolve()
    if seeds is not None:
        seeds = tuple(seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Selected seeds must be nonempty and unique")
    rows, seed_rows, comparisons, captures, counts, replay = [], [], [], [], {}, []
    pooled = {}
    truth_intervals = []
    for level in ("level02", "level03"):
        if seeds is None:
            directories = sorted(path for path in root.glob(level + "_seed*") if path.is_dir())
        else:
            directories = [root / (level + "_seed" + str(seed)) for seed in seeds]
            for path in directories:
                if not path.is_dir():
                    raise ValueError("Missing selected capture: " + str(path))
        if not directories:
            raise ValueError("No captures for " + level)
        metadata_by_directory = {}
        for directory in directories:
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["error"] or metadata["completed_steps"] != metadata["requested_steps"]:
                raise ValueError("Incomplete capture: " + str(directory))
            if metadata["motion_level"].lower() != level:
                raise ValueError("Capture motion level disagrees with directory: " + str(directory))
            if seeds is not None and directory.name != level + "_seed" + str(metadata["seed"]):
                raise ValueError("Capture seed disagrees with selected directory: " + str(directory))
            metadata_by_directory[directory] = metadata
            captures.append(dict(path=str(directory), **metadata))
            replay.append(dict(path=str(directory), **json.loads(
                (directory / "world_diagnostic/online_replay.json").read_text())))
        for frame, relative in (("arm_relative", "relative_evaluation"),
                                ("world_diagnostic", "world_diagnostic/evaluation")):
            items = []
            for capture_index, directory in enumerate(directories):
                with np.load(directory / relative / "raw_errors.npz", allow_pickle=False) as archive:
                    item = {key: archive[key] for key in archive.files}
                with np.load(directory / "capture.npz", allow_pickle=False) as capture:
                    dense_times = capture["dense_truth_times"]
                    dense_episodes = capture["dense_truth_episode_ids"]
                if frame == "arm_relative":
                    truth_intervals.extend(np.diff(dense_times).tolist())
                starts = {}
                for env_id in range(dense_episodes.shape[1]):
                    indices = np.r_[0, np.flatnonzero(np.diff(dense_episodes[:, env_id]) != 0) + 1]
                    starts.update({(env_id, int(dense_episodes[index, env_id])): dense_times[index]
                                   for index in indices})
                item["episode_age_s"] = item["origin_times"] - np.array([
                    starts[int(env_id), int(episode_id)] for env_id, episode_id
                    in zip(item["env_ids"], item["episode_ids"])])
                item["episode_keys"] = np.column_stack((
                    np.full(len(item["env_ids"]), capture_index), item["env_ids"], item["episode_ids"]))
                item["seed"] = metadata_by_directory[directory]["seed"]
                items.append(item)
            methods, horizons = items[0]["methods"], items[0]["horizons"]
            if not all(np.array_equal(item["methods"], methods)
                       and np.array_equal(item["horizons"], horizons) for item in items):
                raise ValueError("Cannot pool mismatched methods or horizons: " + level + "/" + frame)
            position = np.concatenate([item["position_error_m"] for item in items]) * 100
            rotation = np.concatenate([item["rotation_error_deg"] for item in items])
            phases = np.concatenate([item["current_phase"] for item in items])
            changed = np.concatenate([item["future_phase_changed"] for item in items])
            ages = np.concatenate([item["episode_age_s"] for item in items])
            keys = np.concatenate([item["episode_keys"] for item in items])
            pooled[level, frame] = dict(position=position, rotation=rotation,
                                         methods=methods, horizons=horizons)
            if frame == "arm_relative":
                counts[level] = dict(seeds=[item["seed"] for item in items],
                    environments_per_capture=[metadata_by_directory[path]["num_envs"] for path in directories],
                    evaluated_origins=len(position), evaluated_episodes=len(np.unique(keys, axis=0)))
            scopes = {"all": np.ones(len(position), dtype=bool),
                      "drive": phases == 0, "brake": phases == 1, "pause": phases == 2,
                      "phase_changes_within_{:g}s".format(horizons[-1]): changed[:, -1],
                      "first_2s_after_reset": ages < 2., "after_2s_from_reset": ages >= 2.}
            if all("origin_speed_change_m_s2" in item for item in items):
                speed_change = np.concatenate([item["origin_speed_change_m_s2"] for item in items])
                scopes.update(speeding_up=speed_change > .02,
                              slowing_down=speed_change < -.02,
                              steady_speed=np.isfinite(speed_change) & (np.abs(speed_change) <= .02))
            for scope, mask in scopes.items():
                if mask.any():
                    rows.extend(_metric_rows(position[mask], rotation[mask], methods, horizons,
                                             level=level, frame=frame, scope=scope))
            # The event mask belongs to the evaluated horizon. A phase change
            # at +0.8 s must not classify a +0.3 s prediction as a transition.
            for h_index in range(len(horizons)):
                for scope, mask in (("phase_change_within_horizon", changed[:, h_index]),
                                    ("no_phase_change_within_horizon", ~changed[:, h_index])):
                    if mask.any():
                        rows.extend(_metric_rows(position[mask, :, h_index:h_index+1],
                            rotation[mask, :, h_index:h_index+1], methods, horizons[h_index:h_index+1],
                            level=level, frame=frame, scope=scope))
            for item in items:
                seed_rows.extend(_metric_rows(item["position_error_m"] * 100, item["rotation_error_deg"],
                    methods, horizons, level=level, frame=frame, seed=int(item["seed"]), scope="all"))
            comparisons.extend(dict(level=level, frame=frame, scope="all", **row)
                for row in paired_episode_bootstrap(position, rotation, keys, methods, horizons,
                                                     samples=bootstrap_samples))
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "seed_metrics.csv", seed_rows)
    _write_csv(root / "paired_episode_comparisons.csv", comparisons)
    summary = dict(counts=counts, captures=captures, online_replay=replay,
        all_replays_match=all(r["matches_online_within_tolerance"] for r in replay),
        primary_frame="object relative to moving arm base, same as teacher online EKF",
        baselines=dict(hold="Current measured pose held fixed at every future horizon",
                       cv_last2="Constant linear/angular velocity from last two measurements",
                       cv_window5="Constant velocity fitted from the same five captured measurements"),
        protocol="All methods share causal origins, histories and dense future truth; no reset crossing",
        event_scopes=dict(phase_change_within_horizon="Any recorded driver phase change before this forecast's target time",
            speed_change_source="Causal change of dense table commanded planar speed magnitude over preceding 0.10 s; not measured object acceleration",
            speeding_up="origin_speed_change_m_s2 > 0.02", slowing_down="origin_speed_change_m_s2 < -0.02",
            steady_speed="finite abs(origin_speed_change_m_s2) <= 0.02; unknown speed changes excluded"),
        bootstrap=dict(samples=bootstrap_samples, seed=2026, confidence=.95,
            unit="whole sampled episode (capture, environment, episode)",
            statistic="equal-episode mean of paired EKF-minus-baseline mean errors; negative favors EKF",
            scope="Descriptive uncertainty over sampled episodes; not independent origins or new random seeds"),
        truth_sample_interval_s=dict(median=float(np.median(truth_intervals)),
                                     min=float(np.min(truth_intervals)), max=float(np.max(truth_intervals))),
        limits=["Zero high-level actions with original low-level standing controller; no grasp-success ablation",
                "Clean privileged root measurements, matching teacher; real sensor noise/delay not tested",
                "Level02 and Level03 have different motion distributions, not only different accelerations",
                "World-frame replay is a secondary diagnostic, not the teacher's online prediction",
                "Bootstrap weights episodes equally; pooled metrics weight forecast origins equally",
                "Episodes sharing an environment/seed may be dependent; intervals do not establish seed-level generalization"],
        metrics=rows, seed_metrics=seed_rows, paired_episode_comparisons=comparisons)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if plot:
        _plot(root, pooled, summary)
    return summary


def _plot(root, pooled, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("EKF (5 samples)", "Hold current pose", "CV (last 2)", "CV (5 samples)")
    colors = ("#1766a7", "#d56b35", "#328651", "#8b78a5")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    h = pooled["level03", "arm_relative"]["horizons"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        current = pooled["level03", "arm_relative"]
        axes[0, 0].plot(h, current["position"][:, index].mean(axis=0), "o-", label=label, color=color)
        axes[0, 1].plot(h, current["rotation"][:, index].mean(axis=0), "o-", label=label, color=color)
        axes[1, 1].plot(h, np.mean(current["position"][:, index] <= 2, axis=0)*100,
                        "o-", label=label, color=color)
    for level, color in (("level02", "#8a8a8a"), ("level03", "#1766a7")):
        for frame, style in (("arm_relative", "-"), ("world_diagnostic", "--")):
            current = pooled[level, frame]
            values = current["position"][:, 0].mean(axis=0)
            label = level.title() + (" / online relative" if frame == "arm_relative" else " / world replay")
            axes[1, 0].plot(current["horizons"], values, "o" + style, label=label, color=color)
    axes[0, 0].set(title="Level03: mean position error", ylabel="Position error (cm)")
    axes[0, 1].set(title="Level03: mean orientation error", ylabel="Geodesic rotation error (deg)")
    axes[1, 0].set(title="EKF: online frame vs world diagnostic", ylabel="Mean position error (cm)")
    axes[1, 1].set(title="Level03: fraction within 2 cm", ylabel="Forecasts (%)", ylim=(0, 100))
    for axis in axes.flat:
        axis.set_xlabel("Forecast horizon (s)")
        axis.set_xticks(h)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    counts = summary["counts"]["level03"]
    seeds = ", ".join(map(str, counts["seeds"]))
    environments = ", ".join(map(str, counts["environments_per_capture"]))
    fig.suptitle("Current Level03 EKF check | seeds {} | environments per capture: {}\n"
                 "Clean measurements, zero high-level actions".format(seeds, environments), fontsize=12)
    fig.tight_layout(rect=(0, .025, 1, .94))
    fig.text(.5, .012, "Same origins for all methods within each capture; median truth interval {:.4g} s. "
             "This evaluates forecasts, not grasp success.".format(summary["truth_sample_interval_s"]["median"]),
             ha="center", fontsize=9)
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Summarize exactly these seeds; default checks all matching capture directories")
    args = parser.parse_args()
    summary = summarize(args.directory, args.bootstrap_samples, seeds=args.seeds)
    print(json.dumps(summary["counts"], indent=2))
    print("Wrote metrics.csv, seed_metrics.csv, paired_episode_comparisons.csv, summary.json, comparison.png in",
          args.directory.resolve())


if __name__ == "__main__":
    main()
