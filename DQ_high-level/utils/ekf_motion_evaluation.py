"""Evaluate captured online EKF forecasts in the teacher's moving arm frame.

All methods share causal, captured histories and the same forecast origins.
Ground truth is the recorded relative pose at each future sample, without
re-expressing it in the arm frame at the forecast origin. This module uses
NumPy/SciPy on CPU and does not instantiate a simulator or prediction filter.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


METHODS = ("online_ekf", "hold", "cv_last2", "cv_window5")
PHASES = {0: "drive", 1: "brake", 2: "pause"}
TIME_TOLERANCE = 1e-7


def _check_poses(poses, name):
    if poses.shape[-1] != 7 or not np.isfinite(poses).all():
        raise ValueError(name + " must contain finite xyz/xyzw poses")
    if np.any(np.linalg.norm(poses[..., 3:], axis=-1) < 1e-10):
        raise ValueError(name + " contains a zero quaternion")


def causal_baselines(history_poses, history_times, horizons):
    """Return hold/last-two/window predictions, using only supplied histories.

    Input shapes are [origins, history, 7] and [origins, history], or one
    unbatched history. Output is [origins, 3, horizons, 7], with the leading
    dimension omitted for an unbatched input. Horizons are relative to the
    final history timestamp. The window method uses all supplied history
    samples, which must be the five samples captured by the production EKF.
    """
    poses = np.asarray(history_poses, dtype=np.float64)
    times = np.asarray(history_times, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.float64)
    unbatched = poses.ndim == 2
    if unbatched:
        poses, times = poses[None], times[None]
    if poses.ndim != 3 or times.shape != poses.shape[:2] or poses.shape[1] < 2:
        raise ValueError("Expected matching pose/time histories with at least two samples")
    _check_poses(poses, "history_poses")
    if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("History timestamps must be finite and strictly increasing")
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or np.any(horizons <= 0):
        raise ValueError("horizons must be positive finite seconds")
    count, history = times.shape
    last = poses[:, -1].copy()
    last[:, 3:] /= np.linalg.norm(last[:, 3:], axis=-1, keepdims=True)
    result = np.broadcast_to(last[:, None, None], (count, 3, len(horizons), 7)).copy()

    dt = times[:, -1] - times[:, -2]
    last_velocity = (poses[:, -1, :3] - poses[:, -2, :3]) / dt[:, None]
    last_rotation = Rotation.from_quat(last[:, 3:])
    last_omega = (Rotation.from_quat(poses[:, -2, 3:]).inv() * last_rotation).as_rotvec() / dt[:, None]

    centered_time = times - times.mean(axis=1, keepdims=True)
    centered_position = poses[..., :3] - poses[..., :3].mean(axis=1, keepdims=True)
    window_velocity = (centered_time[..., None] * centered_position).sum(axis=1) / (centered_time**2).sum(axis=1)[:, None]
    relative_time = times - times[:, -1:]
    relative_rotation = (
        Rotation.from_quat(np.repeat(last[:, 3:], history, axis=0)).inv()
        * Rotation.from_quat(poses[..., 3:].reshape(-1, 4))
    ).as_rotvec().reshape(count, history, 3)
    window_omega = (relative_time[..., None] * relative_rotation).sum(axis=1) / (relative_time**2).sum(axis=1)[:, None]
    for method_index, velocity, omega in ((1, last_velocity, last_omega), (2, window_velocity, window_omega)):
        result[:, method_index, :, :3] = last[:, None, :3] + velocity[:, None] * horizons[None, :, None]
        rotation = (
            Rotation.from_quat(np.repeat(last[:, 3:], len(horizons), axis=0))
            * Rotation.from_rotvec((omega[:, None] * horizons[None, :, None]).reshape(-1, 3))
        ).as_quat().reshape(count, len(horizons), 4)
        result[:, method_index, :, 3:] = rotation
    return result[0] if unbatched else result


def interpolate_poses(times, poses, target_times):
    """Interpolate within one episode; never extrapolate or cross a reset."""
    times = np.asarray(times, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or poses.shape != (len(times), 7):
        raise ValueError("An interpolation episode needs at least two matching poses/times")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode times must be finite and strictly increasing")
    if (not np.isfinite(target_times).all()
            or np.any(target_times < times[0] - TIME_TOLERANCE)
            or np.any(target_times > times[-1] + TIME_TOLERANCE)):
        raise ValueError("Forecast ground truth would require extrapolation")
    _check_poses(poses, "relative_poses")
    # Arithmetic such as 0.1 + 0.2 must not reject an endpoint stored as 0.3.
    flat = np.clip(target_times.reshape(-1), times[0], times[-1])
    xyz = np.stack([np.interp(flat, times, poses[:, axis]) for axis in range(3)], axis=-1)
    quat = Slerp(times, Rotation.from_quat(poses[:, 3:]))(flat).as_quat()
    return np.concatenate((xyz, quat), axis=-1).reshape(target_times.shape + (7,))


def _metric_rows(position, rotation, horizons, scope, sample_mask=None, **identifiers):
    rows = []
    for method_index, method in enumerate(METHODS):
        for horizon_index, horizon in enumerate(horizons):
            mask = slice(None) if sample_mask is None else sample_mask[:, horizon_index]
            p = position[mask, method_index, horizon_index] * 100.
            r = rotation[mask, method_index, horizon_index]
            if not len(p):
                continue
            rows.append(dict(identifiers, scope=scope, method=method, horizon_s=float(horizon), samples=len(p),
                position_mean_cm=float(p.mean()), position_rmse_cm=float(np.sqrt(np.mean(p**2))),
                position_p95_cm=float(np.percentile(p, 95)),
                position_within_1cm_fraction=float(np.mean(p <= 1.)),
                position_within_2cm_fraction=float(np.mean(p <= 2.)),
                position_within_5cm_fraction=float(np.mean(p <= 5.)),
                rotation_mean_deg=float(r.mean()), rotation_rmse_deg=float(np.sqrt(np.mean(r**2))),
                rotation_p95_deg=float(np.percentile(r, 95)), rotation_within_5deg_fraction=float(np.mean(r <= 5.))))
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_capture(path, outputdir, device="cpu"):
    """Score a capture and write metrics.csv, episode_metrics.csv and raw_errors.npz.

    ``path`` may name capture.npz or its parent directory. ``outputdir`` must
    not exist. ``device`` is retained for caller compatibility; this primary
    evaluator always runs on CPU. Returns rows, episode_rows, summary and the
    raw-errors path. World-frame comparisons and filter replay are separate.

    Origins require valid online predictions, an unlifted object at the origin,
    and full future coverage through the longest horizon in the same episode.
    Future lifting is flagged in raw output rather than silently excluding it.
    Event metrics split each horizon by future phase transitions and lifting,
    with identical origins for every method within each split.
    Optional dense_truth_times/episode_ids/relative_poses supply more frequent
    ground truth; they never change the captured forecast histories or origins.
    """
    path, outputdir = Path(path), Path(outputdir)
    if path.is_dir():
        path = path / "capture.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = ("times", "episode_ids", "relative_poses", "predictions", "valid",
                    "history_poses", "history_times", "driver_phase", "lifted", "horizons")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError("Capture is missing fields: " + ", ".join(sorted(missing)))
        data = {key: archive[key] for key in required}
        dense_fields = ("dense_truth_times", "dense_truth_episode_ids", "dense_truth_relative_poses")
        dense_present = [key in archive.files for key in dense_fields]
        if any(dense_present) and not all(dense_present):
            raise ValueError("Dense truth requires all three dense_truth_* fields")
        if all(dense_present):
            data.update({key: archive[key] for key in dense_fields})
        for key in ("dense_truth_driver_phase", "dense_truth_lifted", "dense_truth_table_commands"):
            if key in archive.files:
                if not all(dense_present):
                    raise ValueError(key + " requires dense truth times/episodes/poses")
                data[key] = archive[key]
    times = np.asarray(data["times"], dtype=np.float64)
    horizons = np.asarray(data["horizons"], dtype=np.float64)
    episodes = data["episode_ids"]
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Capture times must be finite and strictly increasing")
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
        raise ValueError("Capture horizons must be finite, positive and strictly increasing")
    if episodes.ndim != 2 or episodes.shape[0] != len(times):
        raise ValueError("episode_ids must have shape [time, environment]")
    length, num_envs = episodes.shape
    expected_shapes = {"relative_poses": (length, num_envs, 7),
        "predictions": (length, num_envs, len(horizons), 7),
        "valid": (length, num_envs), "driver_phase": (length, num_envs),
        "lifted": (length, num_envs), "history_poses": (length, num_envs, 5, 7),
        "history_times": (length, num_envs, 5)}
    for key, shape in expected_shapes.items():
        if data[key].shape != shape:
            raise ValueError("{} shape {} != {}".format(key, data[key].shape, shape))
    _check_poses(data["relative_poses"], "relative_poses")
    if np.any(np.diff(episodes, axis=0) < 0):
        raise ValueError("Episode IDs must not decrease within an environment")
    dense = "dense_truth_times" in data
    truth_times = np.asarray(data["dense_truth_times"], dtype=np.float64) if dense else times
    truth_episodes = data["dense_truth_episode_ids"] if dense else episodes
    truth_poses = data["dense_truth_relative_poses"] if dense else data["relative_poses"]
    if truth_times.ndim != 1 or len(truth_times) < 2 or not np.isfinite(truth_times).all() or np.any(np.diff(truth_times) <= 0):
        raise ValueError("Truth times must be finite and strictly increasing")
    if truth_episodes.shape != (len(truth_times), num_envs) or truth_poses.shape != (len(truth_times), num_envs, 7):
        raise ValueError("Truth episode/pose shapes must match truth times and environment count")
    if np.any(np.diff(truth_episodes, axis=0) < 0):
        raise ValueError("Truth episode IDs must not decrease within an environment")
    _check_poses(truth_poses, "truth relative_poses")
    for key in ("dense_truth_driver_phase", "dense_truth_lifted"):
        if key in data and data[key].shape != (len(truth_times), num_envs):
            raise ValueError(key + " must match dense truth times and environment count")
    if "dense_truth_table_commands" in data:
        commands = data["dense_truth_table_commands"]
        if commands.shape != (len(truth_times), num_envs, 6) or not np.isfinite(commands).all():
            raise ValueError("dense_truth_table_commands must contain finite [time, environment, 6] commands")

    chunks = []
    skipped = dict(invalid_online=0, lifted_origin=0, insufficient_future=0, invalid_history=0)
    for env_id in range(num_envs):
        cuts = np.r_[0, np.flatnonzero(np.diff(episodes[:, env_id]) != 0) + 1, length]
        truth_cuts = np.r_[0, np.flatnonzero(np.diff(truth_episodes[:, env_id]) != 0) + 1, len(truth_times)]
        truth_segments = {int(truth_episodes[a, env_id]): (a, b) for a, b in zip(truth_cuts[:-1], truth_cuts[1:])}
        for start, end in zip(cuts[:-1], cuts[1:]):
            truth_start, truth_end = truth_segments.get(int(episodes[start, env_id]), (0, 0))
            candidates = []
            for index in range(start, end):
                if not bool(data["valid"][index, env_id]):
                    skipped["invalid_online"] += 1
                    continue
                if bool(data["lifted"][index, env_id]):
                    skipped["lifted_origin"] += 1
                    continue
                if (truth_end - truth_start < 2
                    or times[index] + horizons[0] < truth_times[truth_start] - TIME_TOLERANCE
                    or times[index] + horizons[-1] > truth_times[truth_end - 1] + TIME_TOLERANCE):
                    skipped["insufficient_future"] += 1
                    continue
                history_time = data["history_times"][index, env_id]
                history_pose = data["history_poses"][index, env_id]
                history_good = (np.isfinite(history_time).all() and np.all(np.diff(history_time) > 0)
                    and abs(history_time[-1] - times[index]) <= TIME_TOLERANCE
                    # A reset seed can share the preceding terminal timestamp,
                    # but an older sample cannot belong to this episode. The
                    # first recorded episode may legitimately have pre-capture
                    # warm-up history, so it has no observed lower bound.
                    and (start == 0 or history_time[0] >= times[start - 1] - TIME_TOLERANCE)
                    and np.isfinite(history_pose).all()
                    and np.all(np.linalg.norm(history_pose[:, 3:], axis=-1) > 1e-10))
                if not history_good:
                    skipped["invalid_history"] += 1
                    continue
                candidates.append(index)
            if not candidates:
                continue
            indices = np.asarray(candidates, dtype=np.int64)
            target_times = times[indices, None] + horizons[None]
            targets = interpolate_poses(truth_times[truth_start:truth_end], truth_poses[truth_start:truth_end, env_id], target_times)
            online = np.asarray(data["predictions"][indices, env_id], dtype=np.float64)
            _check_poses(online, "valid online predictions")
            baseline = causal_baselines(data["history_poses"][indices, env_id],
                                        data["history_times"][indices, env_id], horizons)
            predictions = np.concatenate((online[:, None], baseline), axis=1)
            phase = data["driver_phase"][indices, env_id]
            changed = np.zeros((len(indices), len(horizons)), dtype=bool)
            lifted_future = np.zeros_like(changed)
            for row, index in enumerate(indices):
                for h_index, target_time in enumerate(target_times[row]):
                    last = min(end, np.searchsorted(times, target_time + TIME_TOLERANCE, side="right"))
                    changed[row, h_index] = np.any(data["driver_phase"][index:last, env_id] != phase[row])
                    lifted_future[row, h_index] = np.any(data["lifted"][index:last, env_id])
                    dense_first = max(truth_start, np.searchsorted(truth_times, times[index], side="left"))
                    dense_last = min(truth_end, np.searchsorted(truth_times, target_time + TIME_TOLERANCE, side="right"))
                    if "dense_truth_driver_phase" in data:
                        changed[row, h_index] |= np.any(
                            data["dense_truth_driver_phase"][dense_first:dense_last, env_id] != phase[row])
                    if "dense_truth_lifted" in data:
                        lifted_future[row, h_index] |= np.any(data["dense_truth_lifted"][dense_first:dense_last, env_id])
            chunks.append(dict(origin_indices=indices, origin_times=times[indices],
                env_ids=np.full(len(indices), env_id, dtype=np.int64),
                episode_ids=episodes[indices, env_id], current_phase=phase,
                target_times=target_times, target_poses=targets, predictions=predictions,
                future_phase_changed=changed, future_lifted=lifted_future))
            if "dense_truth_table_commands" in data:
                acceleration = np.full(len(indices), np.nan)
                episode_times = truth_times[truth_start:truth_end]
                speeds = np.linalg.norm(data["dense_truth_table_commands"][truth_start:truth_end, env_id, :2], axis=-1)
                for row, index in enumerate(indices):
                    latest = np.searchsorted(episode_times, times[index] + TIME_TOLERANCE, side="right") - 1
                    if latest < 0:
                        continue
                    earlier = episode_times[latest] - .1
                    if earlier < episode_times[0] - TIME_TOLERANCE:
                        continue
                    previous_speed = np.interp(earlier, episode_times[:latest + 1], speeds[:latest + 1])
                    acceleration[row] = (speeds[latest] - previous_speed) / .1
                chunks[-1]["origin_speed_change_m_s2"] = acceleration
    if not chunks:
        raise ValueError("No eligible forecast origins; exclusions: " + json.dumps(skipped))
    raw = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
    # Keep output convenient for chronological plots despite per-episode processing.
    order = np.lexsort((raw["env_ids"], raw["origin_indices"]))
    raw = {key: value[order] for key, value in raw.items()}
    prediction, target = raw["predictions"], raw["target_poses"]
    position = np.linalg.norm(prediction[..., :3] - target[:, None, :, :3], axis=-1)
    truth_rotations = np.broadcast_to(target[:, None, :, 3:], prediction[..., 3:].shape)
    rotation = np.rad2deg((
        Rotation.from_quat(truth_rotations.reshape(-1, 4)).inv()
        * Rotation.from_quat(prediction[..., 3:].reshape(-1, 4))
    ).magnitude()).reshape(position.shape)
    raw.update(position_error_m=position, rotation_error_deg=rotation,
               methods=np.asarray(METHODS), horizons=horizons)
    rows = _metric_rows(position, rotation, horizons, "all")
    for phase, name in PHASES.items():
        mask = raw["current_phase"] == phase
        if mask.any():
            rows.extend(_metric_rows(position[mask], rotation[mask], horizons, name))
    event_rows = []
    for scope, mask in (("phase_unchanged", ~raw["future_phase_changed"]),
                        ("phase_changed", raw["future_phase_changed"]),
                        ("future_unlifted", ~raw["future_lifted"]),
                        ("future_lifted", raw["future_lifted"])):
        event_rows.extend(_metric_rows(position, rotation, horizons, scope, sample_mask=mask))
    if "origin_speed_change_m_s2" in raw:
        acceleration = raw["origin_speed_change_m_s2"]
        for scope, mask in (("speeding_up", acceleration > .02),
                            ("slowing_down", acceleration < -.02),
                            ("steady_speed", np.abs(acceleration) <= .02)):
            if mask.any():
                event_rows.extend(_metric_rows(position[mask], rotation[mask], horizons, scope))
    episode_rows = []
    pairs = np.unique(np.stack((raw["env_ids"], raw["episode_ids"]), axis=-1), axis=0)
    for env_id, episode_id in pairs:
        mask = (raw["env_ids"] == env_id) & (raw["episode_ids"] == episode_id)
        episode_rows.extend(_metric_rows(position[mask], rotation[mask], horizons, "all",
                                        env_id=int(env_id), episode_id=int(episode_id)))
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    summary = dict(capture=str(path.resolve()), metadata=metadata, methods=list(METHODS),
        horizons_s=horizons.tolist(), evaluated_origins=len(position), evaluated_episodes=len(pairs),
        exclusions=skipped, backend="NumPy/SciPy CPU", requested_device=str(device),
        pose_frame="per-sample arm-relative poses; future arm frame moves with the robot",
        eligibility="valid online; unlifted at origin; all horizons covered in the same episode; valid captured history",
        future_reference="linear xyz and SO(3) SLERP; no extrapolation and no reset crossing",
        truth_source="dense_truth_relative_poses" if dense else "relative_poses",
        truth_samples=len(truth_times),
        aggregation="pooled shared forecast origins, with current-driver-phase and episode metrics",
        history_samples=5, history_source="captured production predictor histories only",
        future_lifted_origins=int(raw["future_lifted"].any(axis=1).sum()),
        future_phase_changed_origins=int(raw["future_phase_changed"].any(axis=1).sum()),
        event_sampling={name: "dense truth plus high-level samples" if "dense_truth_" + name in data
                        else "high-level samples only" for name in ("driver_phase", "lifted")},
        speed_change_diagnostic=(dict(
            source="dense_truth_table_commands planar speed magnitude; labels only, never predictor inputs",
            window_s=.1, threshold_m_s2=.02,
            definition="Trailing speed difference over 0.10 s at latest dense timestamp <= origin, within the same episode",
            unavailable_origins=int(np.isnan(raw["origin_speed_change_m_s2"]).sum()))
            if "origin_speed_change_m_s2" in raw else None),
        raw_error_axes=["origin", "method", "horizon"],
        notes=["Future lifting is flagged, not excluded, when the origin itself is unlifted.",
               "World-frame secondary scores and production-filter replay are not performed here."])
    outputdir.mkdir(parents=True, exist_ok=False)
    _write_csv(outputdir / "metrics.csv", rows)
    _write_csv(outputdir / "episode_metrics.csv", episode_rows)
    _write_csv(outputdir / "event_metrics.csv", event_rows)
    raw_path = outputdir / "raw_errors.npz"
    np.savez_compressed(raw_path, **raw)
    (outputdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return dict(rows=rows, episode_rows=episode_rows, event_rows=event_rows,
                summary=summary, raw_errors_path=str(raw_path))
