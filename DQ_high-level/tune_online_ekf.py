#!/usr/bin/env python3
"""Tune the production rolling SE(3) EKF on fixed, causal forecast origins.

Selection uses seed43 only. The selected configuration is written to disk
before seed47 is read. All configurations retain the original five-sample
baseline's scoring origins, including warmup-adjacent origins and braking.
No simulator velocity, driver input, or future pose enters a prediction fit.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


BASELINE = dict(history=5, q_scale=1., r_scale=.01)
PHASES = {0: "drive", 1: "brake", 2: "pause"}


def config_id(config):
    return "h{}_q{:g}_r{:g}".format(config["history"], config["q_scale"], config["r_scale"])


def load_capture_case(capture_dir, evaluation_dir=None):
    """Load histories and dense targets already accepted by the primary evaluator.

    This deliberately does not construct extra origins for shorter histories:
    baseline and tuned scores therefore describe exactly the same futures.
    """
    capture_dir = Path(capture_dir)
    evaluation_dir = Path(evaluation_dir) if evaluation_dir else capture_dir / "relative_evaluation"
    with np.load(capture_dir / "capture.npz", allow_pickle=False) as archive:
        capture = {name: archive[name] for name in archive.files}
    with np.load(evaluation_dir / "raw_errors.npz", allow_pickle=False) as archive:
        errors = {name: archive[name] for name in archive.files}
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    if metadata.get("error"):
        raise ValueError("Capture did not complete successfully")
    indices, envs = errors["origin_indices"], errors["env_ids"]
    histories = capture["history_poses"][indices, envs]
    history_times = capture["history_times"][indices, envs]
    if histories.shape[1] != 5 or np.any(np.diff(history_times, axis=1) <= 0):
        raise ValueError("Expected valid, strictly causal five-sample captured histories")
    if not np.allclose(history_times[:, -1], errors["origin_times"], rtol=0, atol=1e-7):
        raise ValueError("History endpoint differs from forecast origin")
    if not np.array_equal(capture["episode_ids"][indices, envs], errors["episode_ids"]):
        raise ValueError("Forecast origins are not aligned with capture episodes")
    if not np.all(capture["valid"][indices, envs]):
        raise ValueError("Scoring origins include invalid baseline predictions")
    if not np.array_equal(capture["horizons"], errors["horizons"]):
        raise ValueError("Capture and evaluation horizons differ")
    starts = {}
    episodes = capture["dense_truth_episode_ids"]
    for env_id in range(episodes.shape[1]):
        cuts = np.r_[0, np.flatnonzero(np.diff(episodes[:, env_id]) != 0) + 1]
        for index in cuts:
            starts[env_id, int(episodes[index, env_id])] = capture["dense_truth_times"][index]
    ages = errors["origin_times"] - np.asarray([
        starts[int(env), int(episode)] for env, episode in zip(envs, errors["episode_ids"])])
    return dict(capture_dir=str(capture_dir.resolve()), metadata=metadata,
        histories=histories, history_times=history_times, horizons=errors["horizons"],
        targets=errors["target_poses"], online=capture["predictions"][indices, envs],
        origin_indices=indices, origin_times=errors["origin_times"], env_ids=envs,
        episode_ids=errors["episode_ids"], phase=errors["current_phase"], age_s=ages,
        future_phase_changed=errors["future_phase_changed"])


def predict_case(case, config, device="cuda:0", batch_size=4096):
    """Run the exact production predictor on a causal suffix of each history."""
    import torch
    from modules.pose_prediction import RollingSE3Predictor

    history = config["history"]
    if type(history) is not int or not 2 <= history <= case["histories"].shape[1]:
        raise ValueError("history must be an integer within available causal samples")
    predictions = []
    for start in range(0, len(case["histories"]), batch_size):
        end = min(start + batch_size, len(case["histories"]))
        predictor = RollingSE3Predictor(end-start, device, history=history,
            horizons=tuple(float(h) for h in case["horizons"]),
            q_scale=config["q_scale"], r_scale=config["r_scale"])
        predictor.poses.copy_(torch.as_tensor(case["histories"][start:end, -history:],
                                              dtype=torch.float32, device=device))
        predictor.times.copy_(torch.as_tensor(case["history_times"][start:end, -history:],
                                              dtype=torch.float64, device=device))
        predictor.count.fill_(history)
        prediction, valid = predictor.predict()
        if not bool(valid.all()):
            raise ValueError("Invalid prediction for " + config_id(config))
        predictions.append(prediction.cpu().numpy().copy())
    return np.concatenate(predictions)


def prediction_errors(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or predictions.shape[-1] != 7:
        raise ValueError("Predictions and targets must be matching xyz/xyzw arrays")
    position = np.linalg.norm(predictions[..., :3] - targets[..., :3], axis=-1)
    rotation = np.rad2deg((Rotation.from_quat(targets[..., 3:].reshape(-1, 4)).inv()
        * Rotation.from_quat(predictions[..., 3:].reshape(-1, 4))).magnitude()).reshape(position.shape)
    return position, rotation


def score_predictions(case, predictions, config, role="candidate"):
    position, rotation = prediction_errors(predictions, case["targets"])
    scopes = dict(all=np.ones(len(position), dtype=bool),
        first_2s_after_reset=case["age_s"] < 2., after_2s_from_reset=case["age_s"] >= 2.,
        phase_changes_within_0_9s=case["future_phase_changed"][:, -1])
    scopes.update({name: case["phase"] == phase for phase, name in PHASES.items()})
    rows = []
    for scope, mask in scopes.items():
        if not mask.any():
            continue
        for index, horizon in enumerate(case["horizons"]):
            p, r = position[mask, index]*100, rotation[mask, index]
            rows.append(dict(config_id=config_id(config), role=role,
                history=config["history"], q_scale=config["q_scale"], r_scale=config["r_scale"],
                seed=case["metadata"].get("seed"), scope=scope, horizon_s=float(horizon), samples=len(p),
                position_mean_cm=float(p.mean()), position_rmse_cm=float(np.sqrt(np.mean(p*p))),
                position_p95_cm=float(np.percentile(p, 95)), within_2cm_fraction=float(np.mean(p <= 2)),
                rotation_mean_deg=float(r.mean()), rotation_p95_deg=float(np.percentile(r, 95))))
    return rows, position, rotation


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_errors(path, case, config, predictions, position, rotation):
    np.savez_compressed(path, predictions=predictions, target_poses=case["targets"],
        position_error_m=position, rotation_error_deg=rotation, horizons=case["horizons"],
        origin_indices=case["origin_indices"], origin_times=case["origin_times"],
        env_ids=case["env_ids"], episode_ids=case["episode_ids"], driver_phase=case["phase"],
        episode_age_s=case["age_s"], configuration=np.asarray(json.dumps(config, sort_keys=True)))


def replay_check(case, device):
    config = dict(case["metadata"]["ekf"])
    config = {name: config[name] for name in BASELINE}
    prediction = predict_case(case, config, device)
    position, rotation = prediction_errors(prediction, case["online"])
    summary = dict(origins=len(position), position_max_m=float(position.max()),
                   rotation_max_deg=float(rotation.max()),
                   matches_online=bool(position.max() <= 1e-5 and rotation.max() <= .001),
                   configuration=config)
    if not summary["matches_online"]:
        raise AssertionError("Production baseline replay differs from captured online forecasts: " + json.dumps(summary))
    return summary


def evaluate_configs(capture_dir, output_dir, configs, device="cuda:0", evaluation_dir=None):
    """Evaluate preselected configs on a new capture, without selection or tuning.

    The capture must retain five-sample histories for comparison. ``configs``
    is a list of dictionaries with history/q_scale/r_scale; baseline is added
    if absent. Output includes predictions and paired errors for every config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    case = load_capture_case(capture_dir, evaluation_dir)
    replay = replay_check(case, device)
    unique = {config_id(BASELINE): BASELINE}
    unique.update({config_id(config): config for config in configs})
    rows = []
    for key, config in unique.items():
        predictions = predict_case(case, config, device)
        result, position, rotation = score_predictions(case, predictions, config,
            role="baseline" if config == BASELINE else "preselected")
        rows.extend(result)
        _save_errors(output_dir / (key + ".npz"), case, config, predictions, position, rotation)
    _write_csv(output_dir / "metrics.csv", rows)
    summary = dict(capture=case["capture_dir"], metadata=case["metadata"], origins=len(case["histories"]),
        configurations=list(unique.values()), replay=replay, selection="none; preselected configurations only",
        method="Exact production predictor; five-sample scoring origins retained for all history lengths")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return dict(rows=rows, summary=summary)


def rank_configs(rows, rotation_p95_guardrail=False):
    """Apply the predeclared seed43 objective and guardrails, without validation."""
    configs = {}
    for row in rows:
        configs.setdefault(row["config_id"], {})[row["scope"], row["horizon_s"]] = row
    base = configs[config_id(BASELINE)]
    ranked = []
    for key, values in configs.items():
        objective, failures = [], []
        for horizon in (.6, .9):
            current, reference = values["all", horizon], base["all", horizon]
            objective.append(current["position_mean_cm"] / reference["position_mean_cm"])
            guarded_metrics = ["position_p95_cm", "rotation_mean_deg"]
            if rotation_p95_guardrail:
                guarded_metrics.append("rotation_p95_deg")
            for metric in guarded_metrics:
                if current[metric] > reference[metric]*1.05 + 1e-10:
                    failures.append("{}@{}".format(metric, horizon))
            if ("brake", horizon) in base and values["brake", horizon]["position_mean_cm"] > base["brake", horizon]["position_mean_cm"] + 1e-10:
                failures.append("brake_position_mean_cm@{}".format(horizon))
        row = values["all", .6]
        ranked.append(dict(config_id=key, configuration={name: row[name] for name in BASELINE},
            objective=float(np.mean(objective)), guardrails_pass=not failures, guardrail_failures=failures))
    return sorted(ranked, key=lambda item: (not item["guardrails_pass"], item["objective"], item["config_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("outputs/level03_ekf_check_20261127"))
    parser.add_argument("--output", type=Path, default=Path("outputs/level03_ekf_tuning_20261127"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rotation-p95-guardrail", action="store_true",
                        help="Also reject rotation P95 exceeding 105%% of baseline at either primary horizon")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(selection_seed=43, validation_seed=47, histories=[2, 3, 4, 5],
        q_scales=[.001, .01, .1, 1., 10., 100.], r_scales=[.0001, .001, .01, .1, 1., 10.],
        objective="Mean of position-mean ratios to baseline at .6 and .9 seconds",
        guardrails="At each primary horizon: position P95 and rotation mean <=105% of baseline; brake position mean <=baseline"
            + ("; rotation P95 <=105% of baseline" if args.rotation_p95_guardrail else ""),
        rotation_p95_guardrail=args.rotation_p95_guardrail,
        common_origins="Original h5 baseline evaluation origins, including reset early and braking; causal suffix for shorter histories",
        limitations="Clean privileged poses; parameter tuning does not change the constant-twist model or establish grasp-policy success",
        baseline=BASELINE)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    case = load_capture_case(args.captures / "level03_seed43")
    replay = replay_check(case, args.device)
    (args.output / "selection_replay.json").write_text(json.dumps(replay, indent=2) + "\n")
    rows = []
    grid = list(itertools.product(protocol["histories"], protocol["q_scales"], protocol["r_scales"]))
    for index, (history, q_scale, r_scale) in enumerate(grid):
        config = dict(history=history, q_scale=q_scale, r_scale=r_scale)
        prediction = predict_case(case, config, args.device)
        result, _, _ = score_predictions(case, prediction, config,
            role="baseline" if config == BASELINE else "candidate")
        rows.extend(result)
        if (index+1) % 12 == 0:
            print("Selection seed43: {}/{} configs, {:.1f}s".format(index+1, len(grid), time.monotonic()-started), flush=True)
    _write_csv(args.output / "selection_grid_metrics.csv", rows)
    ranked = rank_configs(rows, rotation_p95_guardrail=args.rotation_p95_guardrail)
    winner = ranked[0]
    sensitivity = dict(history_only=[row for row in ranked
        if row["configuration"]["q_scale"] == 1. and row["configuration"]["r_scale"] == .01],
        best_qr_only=next(row for row in ranked if row["configuration"]["history"] == 5))
    selection = dict(selected=winner, sensitivity=sensitivity, ranked=ranked,
        selected_using_seed=43, validation_seed_not_loaded_at_selection=True,
        elapsed_seconds=time.monotonic()-started, protocol=protocol)
    # Freeze the winner before reading held-out data. This is also a progress
    # artifact for a caller launching an independent fresh simulator capture.
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("SELECTED " + json.dumps(winner), flush=True)
    chosen = winner["configuration"]
    evaluate_configs(args.captures / "level03_seed43", args.output / "selection_comparison",
                     [chosen], args.device)
    evaluate_configs(args.captures / "level03_seed47", args.output / "validation_seed47",
                     [chosen], args.device)
    (args.output / "run_summary.json").write_text(json.dumps(dict(selection=selection,
        elapsed_seconds=time.monotonic()-started, validation_directory="validation_seed47"), indent=2) + "\n")
    print("Completed tuning and held-out validation in {:.1f}s".format(time.monotonic()-started), flush=True)


if __name__ == "__main__":
    main()
