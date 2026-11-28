"""Secondary world-frame EKF scoring and captured-online numerical replay.

This offline utility never starts Isaac Gym. A caller that also imports Isaac
Gym should import it before invoking this module's torch-dependent functions.
"""

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from utils.ekf_motion_evaluation import TIME_TOLERANCE, evaluate_capture, interpolate_poses


def _predict_histories(poses, times, horizons, device, batch_size=4096, q_scale=1., r_scale=.01):
    # Lazy imports keep the NumPy evaluation module usable without torch and
    # allow simulator entrypoints to establish their required import order.
    import torch
    from modules.pose_prediction import RollingSE3Predictor

    predictions, validity = [], []
    for start in range(0, len(poses), batch_size):
        end = min(start + batch_size, len(poses))
        predictor = RollingSE3Predictor(end - start, device, history=5,
            horizons=tuple(float(h) for h in horizons), q_scale=q_scale, r_scale=r_scale)
        # Recorded production histories already contain normalized float32
        # measurements. Restore them exactly, avoiding an extra push/normalize.
        predictor.poses.copy_(torch.as_tensor(poses[start:end], dtype=torch.float32, device=device))
        predictor.times.copy_(torch.as_tensor(times[start:end], dtype=torch.float64, device=device))
        predictor.count.fill_(5)
        prediction, valid = predictor.predict()
        predictions.append(prediction.cpu().numpy().copy())
        validity.append(valid.cpu().numpy().copy())
    if not predictions:
        return np.empty((0, len(horizons), 7), np.float32), np.empty(0, dtype=bool)
    return np.concatenate(predictions), np.concatenate(validity)


def _history_validity(poses, times, origin_times):
    return (np.isfinite(poses).all(axis=(-1, -2)) & np.isfinite(times).all(axis=-1)
            & (np.diff(times, axis=-1) > 0).all(axis=-1)
            & (np.linalg.norm(poses[..., 3:], axis=-1) > 1e-10).all(axis=-1)
            & (np.abs(times[..., -1] - origin_times[:, None]) <= TIME_TOLERANCE))


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_world_capture(capture_path, outputdir, device="cuda:0"):
    """Return world evaluation plus production-online replay diagnostics.

    Uses captured metadata Q/R settings (legacy defaults Q=1, R=.01), five
    history samples and the capture's horizons. World histories are sampled at the captured history timestamps
    from dense, same-episode world truth. Insufficient reset warm-up coverage
    is excluded. No simulator state or future sample enters a history fit.

    ``outputdir`` must not exist. Outputs are evaluation/{metrics.csv,
    episode_metrics.csv,raw_errors.npz,summary.json}, online_replay_errors.npz,
    online_replay.json and summary.json. The world metric method is explicitly
    named world_ekf, since these are diagnostic replays, not online forecasts.
    """
    capture_path, outputdir = Path(capture_path), Path(outputdir)
    if capture_path.is_dir():
        capture_path = capture_path / "capture.npz"
    with np.load(capture_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = ("times", "episode_ids", "world_poses", "relative_poses", "predictions",
        "valid", "history_poses", "history_times", "horizons", "dense_truth_times",
        "dense_truth_episode_ids", "dense_truth_world_poses")
    missing = set(required) - set(data)
    if missing:
        raise ValueError("World diagnostic is missing fields: " + ", ".join(sorted(missing)))
    metadata_path = capture_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    settings = dict(history=5, q_scale=1., r_scale=.01)
    settings.update({name: metadata.get("ekf", {})[name] for name in settings if name in metadata.get("ekf", {})})
    if settings["history"] != 5:
        raise ValueError("World diagnostic requires a five-sample production predictor")
    for name in ("q_scale", "r_scale"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError("Captured " + name + " must be finite and positive")
    times = np.asarray(data["times"], dtype=np.float64)
    horizons = np.asarray(data["horizons"], dtype=np.float64)
    history_times = np.asarray(data["history_times"], dtype=np.float64)
    history_poses = data["history_poses"]
    length, num_envs = data["episode_ids"].shape
    if history_poses.shape != (length, num_envs, 5, 7) or history_times.shape != (length, num_envs, 5):
        raise ValueError("World diagnostic requires captured histories of exactly five samples")
    ready = np.asarray(data["valid"], dtype=bool) & _history_validity(history_poses, history_times, times)
    for env_id in range(num_envs):
        starts = np.flatnonzero(np.diff(data["episode_ids"][:, env_id]) != 0) + 1
        ends = np.r_[starts[1:], length]
        for start, end in zip(starts, ends):
            ready[start:end, env_id] &= history_times[start:end, env_id, 0] >= times[start - 1] - TIME_TOLERANCE
    replay_indices, replay_envs = np.nonzero(ready)
    replay, replay_valid = _predict_histories(history_poses[ready], history_times[ready], horizons, device,
        q_scale=settings["q_scale"], r_scale=settings["r_scale"])
    if not len(replay):
        raise ValueError("No valid captured online history can be replayed")
    online = data["predictions"][ready]
    replay_position = np.linalg.norm(replay[..., :3] - online[..., :3], axis=-1)
    replay_rotation = np.rad2deg((Rotation.from_quat(online[..., 3:].reshape(-1, 4)).inv()
        * Rotation.from_quat(replay[..., 3:].reshape(-1, 4))).magnitude()).reshape(len(replay), len(horizons))
    replay_summary = dict(origins=len(replay), horizons_s=horizons.tolist(),
        valid_replay_origins=int(replay_valid.sum()),
        position_max_m=float(replay_position.max()), position_mean_m=float(replay_position.mean()),
        rotation_max_deg=float(replay_rotation.max()), rotation_mean_deg=float(replay_rotation.mean()),
        position_tolerance_m=1e-5, rotation_tolerance_deg=.001,
        matches_online_within_tolerance=bool(replay_valid.all() and replay_position.max() <= 1e-5 and replay_rotation.max() <= .001),
        settings=settings, device=str(device),
        method="Production RollingSE3Predictor.predict on exact captured float32 pose/float64 time histories")

    dense_times = np.asarray(data["dense_truth_times"], dtype=np.float64)
    dense_episodes = data["dense_truth_episode_ids"]
    dense_world = data["dense_truth_world_poses"]
    if (dense_times.ndim != 1 or len(dense_times) < 2 or not np.isfinite(dense_times).all()
            or np.any(np.diff(dense_times) <= 0)):
        raise ValueError("Dense world truth times must be finite and strictly increasing")
    if dense_episodes.shape != (len(dense_times), num_envs) or dense_world.shape != (len(dense_times), num_envs, 7):
        raise ValueError("Dense world truth shapes must match times and environment count")
    if np.any(np.diff(dense_episodes, axis=0) < 0):
        raise ValueError("Dense world episode IDs must not decrease")
    world_history = np.zeros_like(history_poses)
    world_history[..., 6] = 1.
    covered = np.zeros_like(ready)
    for env_id in range(num_envs):
        cuts = np.r_[0, np.flatnonzero(np.diff(dense_episodes[:, env_id]) != 0) + 1, len(dense_times)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end - start < 2:
                continue
            episode_id = dense_episodes[start, env_id]
            eligible = (ready[:, env_id] & (data["episode_ids"][:, env_id] == episode_id)
                & (history_times[:, env_id, 0] >= dense_times[start] - TIME_TOLERANCE)
                & (history_times[:, env_id, -1] <= dense_times[end - 1] + TIME_TOLERANCE))
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            world_history[indices, env_id] = interpolate_poses(dense_times[start:end],
                dense_world[start:end, env_id], history_times[indices, env_id])
            covered[indices, env_id] = True
    world_prediction, world_valid = _predict_histories(world_history[covered], history_times[covered], horizons, device,
        q_scale=settings["q_scale"], r_scale=settings["r_scale"])
    if not len(world_prediction):
        raise ValueError("No complete world histories are covered by same-episode dense truth")
    world = {key: value for key, value in data.items()}
    world["history_poses"] = world_history
    world["relative_poses"] = data["world_poses"]
    world["dense_truth_relative_poses"] = dense_world
    world["predictions"] = np.zeros_like(data["predictions"])
    world["predictions"][..., 6] = 1.
    world["predictions"][covered] = world_prediction
    world["valid"] = np.zeros_like(ready)
    world["valid"][covered] = world_valid
    metadata = dict(metadata, diagnostic="secondary world-frame replay; not actual online teacher forecasts")
    outputdir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="world_replay_", dir=str(outputdir)) as temporary:
        temporary = Path(temporary)
        np.savez_compressed(temporary / "capture.npz", **world)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        evaluation = evaluate_capture(temporary, outputdir / "evaluation", device=device)
    for key in ("rows", "episode_rows", "event_rows"):
        for row in evaluation[key]:
            if row["method"] == "online_ekf":
                row["method"] = "world_ekf"
    summary = evaluation["summary"]
    summary.update(capture=str(capture_path.resolve()),
        diagnostic="secondary world-frame replay; not actual online teacher forecasts",
        methods=["world_ekf", "hold", "cv_last2", "cv_window5"],
        pose_frame="world coordinates", truth_source="dense_truth_world_poses",
        history_source="same-episode dense world poses sampled at captured predictor history_times",
        online_valid_origins=int(ready.sum()), world_history_covered_origins=int(covered.sum()),
        world_history_uncovered_origins=int(ready.sum() - covered.sum()),
        notes=["World fits and baselines share five causal samples and identical world-score origins.",
               "World-score origins can differ from the primary relative score due to reset warm-up coverage.",
               "Future lifting is flagged, not excluded, when the origin itself is unlifted."])
    with np.load(evaluation["raw_errors_path"], allow_pickle=False) as archive:
        errors = {name: archive[name] for name in archive.files}
    errors["methods"] = np.asarray(summary["methods"])
    np.savez_compressed(evaluation["raw_errors_path"], **errors)
    _write_csv(outputdir / "evaluation" / "metrics.csv", evaluation["rows"])
    _write_csv(outputdir / "evaluation" / "episode_metrics.csv", evaluation["episode_rows"])
    _write_csv(outputdir / "evaluation" / "event_metrics.csv", evaluation["event_rows"])
    (outputdir / "evaluation" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(outputdir / "online_replay_errors.npz", origin_indices=replay_indices,
        origin_times=times[replay_indices], env_ids=replay_envs,
        episode_ids=data["episode_ids"][ready], horizons=horizons, replay_valid=replay_valid,
        position_error_m=replay_position, rotation_error_deg=replay_rotation,
        replay_predictions=replay, online_predictions=online)
    (outputdir / "online_replay.json").write_text(json.dumps(replay_summary, indent=2) + "\n")
    overall = dict(world=summary, online_replay=replay_summary)
    (outputdir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    return dict(evaluation=evaluation, replay_summary=replay_summary, summary=overall,
                replay_errors_path=str(outputdir / "online_replay_errors.npz"))
