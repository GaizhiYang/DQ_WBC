"""Compose real Level03 camera frames and recorded telemetry into a video frame.

The caller owns simulation, sampling and encoding. This module only draws the
supplied RGB camera images and samples; it does not filter or alter the data.
"""

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_CJK_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_CJK_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
_HAS_CJK = _CJK_REGULAR.is_file()
_BACKGROUND = (13, 20, 31)
_PANEL = (21, 31, 45)
_TEXT = (235, 241, 249)
_MUTED = (153, 170, 191)
_GRID = (49, 64, 84)
_CYAN = (72, 212, 231)
_GOLD = (255, 199, 92)
_PURPLE = (192, 163, 255)


@lru_cache(maxsize=16)
def _font(size, bold=False):
    if _HAS_CJK:
        path = _CJK_BOLD if bold and _CJK_BOLD.is_file() else _CJK_REGULAR
        # The Simplified Chinese face is index 2 in the Noto CJK collection.
        return ImageFont.truetype(str(path), size, index=2)
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size)


def _label(chinese, english):
    return chinese if _HAS_CJK else english


def _text(draw, xy, value, size=18, fill=_TEXT, bold=False, anchor="lt"):
    draw.text(xy, value, font=_font(size, bold), fill=fill, anchor=anchor)


def _dashed(draw, points, fill, width=1, dash=6., gap=5.):
    """Dash a polyline by screen distance, retaining vertical target changes."""
    phase = 0.
    period = dash + gap
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        offset = 0.
        while offset < length - 1e-9:
            on = phase < dash
            remaining = (dash if on else period) - phase
            distance = min(remaining, length - offset)
            if on:
                a = (start[0] + dx * offset / length,
                     start[1] + dy * offset / length)
                b = (start[0] + dx * (offset + distance) / length,
                     start[1] + dy * (offset + distance) / length)
                draw.line((a, b), fill=fill, width=width)
            offset += distance
            phase = (phase + distance) % period
            if period - phase < 1e-9:
                phase = 0.


def _chart(draw, samples, duration, bounds, limit, series, continuous_targets=False):
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top

    def position(time, value):
        return (left + width * max(0., min(float(time), duration)) / duration,
                top + height * (limit - float(value)) / (2 * limit))

    for fraction in (-1., -.5, 0., .5, 1.):
        y = position(0., fraction * limit)[1]
        draw.line((left, y, right, y), fill=_GRID if fraction else (90, 106, 126), width=1)
        _text(draw, (left - 9, y), "{:+.2f}".format(fraction * limit) if fraction else "0",
              size=13, fill=_MUTED, anchor="rm")
    for index in range(5):
        time = duration * index / 4
        x = position(time, 0.)[0]
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        _text(draw, (x, bottom + 7), "{:g}s".format(round(time, 2)),
              size=13, fill=_MUTED, anchor="mt")

    # Natural driver inputs vary continuously; preset boundary targets step.
    for actual_key, target_key, color in series:
        target_points = []
        previous = None
        for sample in samples:
            if previous is not None and not continuous_targets:
                target_points.append(position(sample["time"], previous))
            target_points.append(position(sample["time"], sample[target_key]))
            previous = sample[target_key]
        target_color = tuple(int(channel * .62) for channel in color)
        _dashed(draw, target_points, fill=target_color, width=1)

    if samples:
        now_x = position(samples[-1]["time"], 0.)[0]
        draw.line((now_x, top, now_x, bottom), fill=(96, 111, 131), width=1)
    for actual_key, _, color in series:
        points = [position(sample["time"], sample[actual_key]) for sample in samples]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        if points:
            x, y = points[-1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def compose_frame(wide_rgb, close_rgb, samples, duration, linear_accel, yaw_accel,
                  scenario="random", overview_follow=False):
    """Return a ``uint8`` RGB array with shape ``(800, 1280, 3)``.

    ``wide_rgb`` must be ``(500, 824, 3)`` and ``close_rgb`` must be
    ``(332, 440, 3)``. ``samples`` is the recorded prefix up to this frame,
    ordered by increasing elapsed ``time`` in seconds. Each dict contains
    ``vx``, ``vy``, ``yaw``, ``target_vx``, ``target_vy``, ``target_yaw``,
    ``object_speed`` and ``table_speed``. Speeds use m/s; yaw uses rad/s.
    The acceleration limits use m/s² and rad/s². Inputs are never mutated.
    ``scenario="yaw-limit"`` explicitly labels preset boundary targets;
    the default retains the production random-driving presentation.
    """
    wide_rgb = np.asarray(wide_rgb)
    close_rgb = np.asarray(close_rgb)
    if wide_rgb.shape != (500, 824, 3) or close_rgb.shape != (332, 440, 3):
        raise ValueError("Expected wide RGB (500, 824, 3) and close RGB (332, 440, 3)")
    if wide_rgb.dtype != np.uint8 or close_rgb.dtype != np.uint8:
        raise ValueError("Camera RGB arrays must have dtype uint8")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number of seconds")
    if scenario not in ("random", "yaw-limit"):
        raise ValueError("scenario must be random or yaw-limit")

    canvas = Image.new("RGB", (1280, 800), _BACKGROUND)
    canvas.paste(Image.fromarray(wide_rgb), (0, 64))
    canvas.paste(Image.fromarray(close_rgb), (840, 64))
    draw = ImageDraw.Draw(canvas)

    _text(draw, (24, 15), _label("Level03 小车式运动", "Level03 Car Motion"), size=27, bold=True)
    subtitle = (_label("上限验证 · 预设油门/转向目标", "Limit check | Preset throttle / steering")
                if scenario == "yaw-limit" else
                _label("连续遥控输入 · 平滑变速 / 转弯", "RC inputs | Smooth speed / Turn"))
    _text(draw, (379, 23), subtitle,
          size=16, fill=_MUTED)
    elapsed = float(samples[-1]["time"]) if samples else 0.
    _text(draw, (1118, 22), "{:05.2f} / {:g} s".format(elapsed, duration), size=19, anchor="rt")
    draw.rounded_rectangle((1144, 15, 1256, 48), radius=8, fill=(36, 59, 68))
    _text(draw, (1200, 23), _label("1 倍速", "1x speed"), size=17, fill=_CYAN, anchor="mt")

    wide_label = _label("跟随远景", "Tracking wide") if overview_follow else _label("固定视角", "Fixed view")
    for x, text in ((16, wide_label),
                    (856, _label("跟随近景", "Tracking close-up"))):
        label_width = 112 if _HAS_CJK else (132 if x == 16 else 200)
        draw.rounded_rectangle((x, 80, x + label_width, 115), radius=7, fill=(18, 29, 40))
        _text(draw, (x + 13, 88), text, size=18)

    draw.rounded_rectangle((840, 408, 1279, 563), radius=10, fill=_PANEL)
    phase = int(samples[-1].get("driver_phase", -1)) if samples else -1
    phase_label = {0: _label("行驶", "Driving"), 1: _label("减速停车", "Braking"),
                   2: _label("停车等待", "Paused")}.get(phase, "")
    panel_title = _label("当前速度", "Current velocity") + (" · " + phase_label if phase_label else "")
    _text(draw, (856, 420), panel_title, size=16, fill=_MUTED)
    if samples:
        current = samples[-1]
        _text(draw, (856, 447), "vx  {:+.3f}".format(current["vx"]), size=21, fill=_CYAN)
        _text(draw, (1045, 447), "vy  {:+.3f} m/s".format(current["vy"]), size=21, fill=_GOLD)
        _text(draw, (856, 477), "yaw  {:+.3f}".format(current["yaw"]), size=19, fill=_PURPLE)
        _text(draw, (1256, 479), _label("目标", "Target")
              + "  {:+.3f} rad/s".format(current["target_yaw"]),
              size=16, fill=_PURPLE, anchor="rt")
        _text(draw, (856, 507), _label("桌子", "Table") + "  {:.3f} m/s".format(current["table_speed"]),
              size=17)
        _text(draw, (1060, 507), _label("物体", "Object") + "  {:.3f} m/s".format(current["object_speed"]),
              size=17)
    else:
        _text(draw, (856, 465), _label("等待仿真采样", "Waiting for samples"), size=21, fill=_MUTED)
    _text(draw, (856, 538), _label("加速度上限", "Accel. limits")
          + "  {:.2f} m/s²  |  {:.2f} rad/s²".format(linear_accel, yaw_accel),
          size=15, fill=_MUTED)

    _text(draw, (24, 583), _label("车身方向速度", "Body-frame velocity") + " (m/s)", size=19, bold=True)
    _text(draw, (840, 583), _label("旋转角速度", "Yaw velocity") + " (rad/s)", size=19, bold=True)
    draw.line((411, 595, 438, 595), fill=_CYAN, width=3)
    _text(draw, (446, 585), "vx", size=16, fill=_CYAN)
    draw.line((492, 595, 519, 595), fill=_GOLD, width=3)
    _text(draw, (527, 585), "vy", size=16, fill=_GOLD)
    _dashed(draw, [(579, 595), (606, 595)], fill=_MUTED)
    target_label = (_label("预设目标", "Preset target") if scenario == "yaw-limit"
                    else _label("遥控输入", "RC input"))
    _text(draw, (614, 585), target_label, size=15, fill=_MUTED)
    _text(draw, (1256, 587), _label("实线：当前", "Solid: current"), size=14, fill=_MUTED, anchor="rt")

    # Stable axes include the signed 0.45 m/s car speed range. Expand only if supplied
    # telemetry exceeds them, so no recorded value is clipped or hidden.
    linear_limit = max([.50] + [abs(float(sample[key])) * 1.06 for sample in samples
                               for key in ("vx", "vy", "target_vx", "target_vy")])
    yaw_limit = max([.44] + [abs(float(sample[key])) * 1.06 for sample in samples
                           for key in ("yaw", "target_yaw")])
    _chart(draw, samples, duration, (63, 627, 798, 764), linear_limit,
           (("vx", "target_vx", _CYAN), ("vy", "target_vy", _GOLD)),
           continuous_targets=scenario == "random")
    _chart(draw, samples, duration, (892, 627, 1258, 764), yaw_limit,
           (("yaw", "target_yaw", _PURPLE),), continuous_targets=scenario == "random")
    return np.array(canvas, dtype=np.uint8)
