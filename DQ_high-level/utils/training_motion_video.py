"""Render real training samples, separating intentional episode resets.

Only camera images and recorded telemetry are drawn. Episode boundaries are
never joined by either actual-velocity or target-input curves.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from utils.table_motion_video import (
    _BACKGROUND, _PANEL, _MUTED, _GRID, _CYAN, _GOLD, _PURPLE,
    _label, _text, _dashed, compose_frame,
)


def _episodes(samples):
    groups = []
    for sample in samples:
        if not groups or sample["episode"] != groups[-1][-1]["episode"]:
            groups.append([])
        groups[-1].append(sample)
    return groups


def _training_chart(draw, groups, duration, bounds, limit, series):
    left, top, right, bottom = bounds

    def point(time, value):
        return (left + (right-left) * max(0., min(float(time) / duration, 1.)),
                top + (bottom-top) * (limit-float(value)) / (2*limit))

    for fraction in (-1., -.5, 0., .5, 1.):
        y = point(0., fraction*limit)[1]
        draw.line((left, y, right, y), fill=_GRID if fraction else (90, 106, 126))
        _text(draw, (left-9, y), "{:+.2f}".format(fraction*limit) if fraction else "0",
              size=13, fill=_MUTED, anchor="rm")
    for index in range(5):
        time = duration*index/4
        x = point(time, 0.)[0]
        draw.line((x, top, x, bottom), fill=_GRID)
        _text(draw, (x, bottom+7), "{:g}s".format(round(time, 2)),
              size=13, fill=_MUTED, anchor="mt")
    for group in groups[1:]:
        sample = group[0]
        reset_time = sample.get("episode_start_time", sample["time"])
        x = point(reset_time, 0.)[0]
        _dashed(draw, [(x, top), (x, bottom)], fill=(166, 108, 62), dash=4., gap=4.)
        _text(draw, (min(x+3, right-36), top+4), "RESET", size=10, fill=_GOLD)
    for actual, target, color in series:
        for group in groups:
            # Separate calls are deliberate: no segment spans two episodes.
            target_points = [point(s["time"], s[target]) for s in group]
            _dashed(draw, target_points, fill=tuple(int(v*.58) for v in color))
            actual_points = [point(s["time"], s[actual]) for s in group]
            if len(actual_points) > 1:
                draw.line(actual_points, fill=color, width=2)
        if groups:
            current = groups[-1][-1]
            x, y = point(current["time"], current[actual])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)


def compose_training_frame(wide_rgb, close_rgb, samples, duration, linear_accel,
                           yaw_accel, env_id, max_episode_steps=100):
    """Return uint8 RGB [800,1280,3] from a causal prefix of training samples.

    Camera shapes and motion fields match ``table_motion_video.compose_frame``.
    Every sample additionally supplies ``episode`` (one based) and
    ``episode_step``. Optional ``episode_start_time`` is elapsed video time;
    optional ``reset_elapsed_s`` gives time since that episode reset directly.
    Samples can contain equal timestamps at a reset, but must not go backwards.
    ``reset_recent=True`` also forces the reset banner to appear.
    """
    # Reuse the established layout and camera shape/type validation. Empty
    # telemetry avoids drawing any curves that could connect across resets.
    canvas = Image.fromarray(compose_frame(wide_rgb, close_rgb, [], duration,
                                           linear_accel, yaw_accel))
    draw = ImageDraw.Draw(canvas)
    groups = _episodes(samples)
    if any(float(b["time"]) < float(a["time"]) for a, b in zip(samples, samples[1:])):
        raise ValueError("Training samples must have nondecreasing video times")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    current = samples[-1] if samples else {}
    elapsed = float(current.get("time", 0.))
    episode = int(current.get("episode", 1))
    step = int(current.get("episode_step", 0))

    draw.rectangle((0, 0, 1279, 63), fill=_BACKGROUND)
    _text(draw, (24, 15), _label("Level03 实际训练采样", "Level03 Training Rollout"),
          size=27, bold=True)
    subtitle = _label("PPO训练初期 · 环境 {} · 回合 {}", "Early PPO | Env {} | Episode {}")
    _text(draw, (390, 23), subtitle.format(env_id, episode), size=16, fill=_MUTED)
    _text(draw, (1118, 22), "{:05.2f} / {:g} s".format(elapsed, duration),
          size=19, anchor="rt")
    draw.rounded_rectangle((1144, 15, 1256, 48), radius=8, fill=(36, 59, 68))
    _text(draw, (1200, 23), _label("1 倍速", "1x speed"), size=17, fill=_CYAN, anchor="mt")
    for x, label, width in ((16, _label("机器人与桌面", "Robot and table"), 166),
                            (856, _label("物体近景", "Object close-up"), 166)):
        draw.rounded_rectangle((x, 80, x+width, 115), radius=7, fill=(18, 29, 40))
        _text(draw, (x+13, 88), label, size=18)
    if current:
        episode_start = current.get("episode_start_time", groups[-1][0]["time"])
        reset_elapsed = float(current.get("reset_elapsed_s", elapsed-episode_start))
        if current.get("reset_recent", False) or (episode > 1 and 0. <= reset_elapsed < .5):
            draw.rounded_rectangle((16, 126, 435, 169), radius=7, fill=(85, 54, 25))
            _text(draw, (29, 136), _label("新回合重置 · 从静止重新开始", "EPISODE RESET | Restart from rest"),
                  size=21, fill=_GOLD, bold=True)

    draw.rounded_rectangle((840, 408, 1279, 563), radius=10, fill=_PANEL)
    phase = int(current.get("driver_phase", -1))
    phase_label = {0: _label("行驶", "Driving"), 1: _label("减速停车", "Braking"),
                   2: _label("停车等待", "Paused")}.get(phase, "")
    _text(draw, (856, 420), _label("当前速度", "Velocity") + " · " + phase_label,
          size=16, fill=_MUTED)
    _text(draw, (1256, 420), _label("步数", "Step") + " {}/{}".format(step, max_episode_steps),
          size=15, fill=_MUTED, anchor="rt")
    if current:
        _text(draw, (856, 447), "vx  {:+.3f}".format(current["vx"]), size=21, fill=_CYAN)
        _text(draw, (1045, 447), "vy  {:+.3f} m/s".format(current["vy"]), size=21, fill=_GOLD)
        _text(draw, (856, 477), "yaw  {:+.3f}".format(current["yaw"]), size=19, fill=_PURPLE)
        _text(draw, (1256, 479), _label("目标", "Target") + " {:+.3f} rad/s".format(current["target_yaw"]),
              size=16, fill=_PURPLE, anchor="rt")
        _text(draw, (856, 507), _label("桌子", "Table") + " {:.3f} m/s".format(current["table_speed"]), size=17)
        _text(draw, (1060, 507), _label("物体", "Object") + " {:.3f} m/s".format(current["object_speed"]), size=17)
    _text(draw, (856, 538), _label("加速度上限", "Accel. limits")
          + " {:.2f} m/s² | {:.2f} rad/s²".format(linear_accel, yaw_accel), size=15, fill=_MUTED)

    draw.rectangle((0, 564, 1279, 799), fill=_BACKGROUND)
    _text(draw, (24, 583), _label("车身方向速度", "Body velocity") + " (m/s)", size=19, bold=True)
    _text(draw, (840, 583), _label("旋转角速度", "Yaw velocity") + " (rad/s)", size=19, bold=True)
    draw.line((342, 595, 365, 595), fill=_CYAN, width=3)
    _text(draw, (372, 585), "vx", size=16, fill=_CYAN)
    draw.line((412, 595, 435, 595), fill=_GOLD, width=3)
    _text(draw, (442, 585), "vy", size=16, fill=_GOLD)
    _dashed(draw, [(484, 595), (507, 595)], fill=_MUTED)
    _text(draw, (514, 585), _label("遥控输入", "RC input"), size=15, fill=_MUTED)
    _text(draw, (656, 585), _label("RESET：新回合", "RESET: new episode"), size=14, fill=_GOLD)
    _text(draw, (1256, 587), _label("实线：当前", "Solid: current"), size=14, fill=_MUTED, anchor="rt")
    linear_limit = max([.50] + [abs(float(s[k]))*1.06 for s in samples
                               for k in ("vx", "vy", "target_vx", "target_vy")])
    yaw_limit = max([.44] + [abs(float(s[k]))*1.06 for s in samples for k in ("yaw", "target_yaw")])
    if not math.isfinite(linear_limit+yaw_limit):
        raise ValueError("Training velocities must be finite")
    _training_chart(draw, groups, duration, (63, 627, 798, 764), linear_limit,
                    (("vx", "target_vx", _CYAN), ("vy", "target_vy", _GOLD)))
    _training_chart(draw, groups, duration, (892, 627, 1258, 764), yaw_limit,
                    (("yaw", "target_yaw", _PURPLE),))
    return np.array(canvas, dtype=np.uint8)
