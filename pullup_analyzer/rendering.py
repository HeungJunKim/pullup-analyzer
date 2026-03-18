from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .state import (
    PullUpMetrics,
    PullUpState,
    PoseFrame,
    SCORE_GUIDE_VALUES,
    SCORE_LEVEL_ADVANCED,
    SCORE_LEVEL_BEGINNER,
    SCORE_LEVEL_GOD,
    SCORE_LEVEL_INTERMEDIATE,
    SCORE_LEVEL_MASTER,
    clamp,
)


PHASE_COLORS = {
    "Pull": (98, 211, 179),
    "Down": (247, 196, 88),
    "Deadhang": (96, 174, 255),
    "Ready": (181, 136, 255),
    "Stand": (132, 145, 166),
}

PHASE_LABELS = {
    "Pull": "PULL UP",
    "Down": "PULL DOWN",
    "Deadhang": "DEAD HANG",
    "Ready": "READY",
    "Stand": "STAND",
}

SCORE_LEVEL_COLORS = {
    SCORE_LEVEL_BEGINNER: (150, 162, 178),
    SCORE_LEVEL_INTERMEDIATE: (96, 174, 255),
    SCORE_LEVEL_ADVANCED: (98, 211, 179),
    SCORE_LEVEL_MASTER: (88, 196, 247),
    SCORE_LEVEL_GOD: (96, 126, 255),
}

SCORE_LEVEL_LABELS = {
    SCORE_LEVEL_BEGINNER: "BEGINNER",
    SCORE_LEVEL_INTERMEDIATE: "INTERMEDIATE",
    SCORE_LEVEL_ADVANCED: "ADVANCED",
    SCORE_LEVEL_MASTER: "MASTER",
    SCORE_LEVEL_GOD: "GOD",
}

SCORE_BAR_DISPLAY_MAX = 6000
SCORE_BAR_SEGMENTS = (
    (0, 1000, "BEGINNER", SCORE_LEVEL_BEGINNER),
    (1000, 2000, "INTERMEDIATE", SCORE_LEVEL_INTERMEDIATE),
    (2000, 3000, "ADVANCED", SCORE_LEVEL_ADVANCED),
    (3000, 5000, "MASTER", SCORE_LEVEL_MASTER),
    (5000, SCORE_BAR_DISPLAY_MAX, "GOD", SCORE_LEVEL_GOD),
)


def format_video_session_label(video_path: Path) -> str:
    stem = video_path.stem
    digits = "".join(character for character in stem if character.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return stem


def frame_ui_scale(frame_shape) -> float:
    height, width = frame_shape[:2]
    return clamp(min(width, height) / 1080.0, 0.60, 1.55)


def pose_ui_scale(pose: PoseFrame) -> float:
    return clamp(pose.body_scale / 220.0, 0.70, 1.60)


def as_point(point) -> tuple[int, int]:
    point = np.asarray(point)
    return tuple(np.round(point[:2]).astype(int))


def score_level_color(level: str) -> tuple[int, int, int]:
    return SCORE_LEVEL_COLORS.get(level, SCORE_LEVEL_COLORS[SCORE_LEVEL_BEGINNER])


def score_level_label(level: str) -> str:
    return SCORE_LEVEL_LABELS.get(level, level.upper())


def rep_score_color(rep_score: int) -> tuple[int, int, int]:
    if rep_score >= 130:
        return (94, 214, 255)
    if rep_score >= 105:
        return (98, 211, 179)
    if rep_score > 90:
        return (96, 174, 255)
    return (150, 162, 178)


def rep_grade_label(rep_score: int) -> str:
    if rep_score >= 130:
        return "EXCELLENT"
    if rep_score >= 105:
        return "GOOD"
    if rep_score > 90:
        return "NORMAL"
    return "BAD"


def score_axis_max(max_score: int) -> int:
    if max_score <= 400:
        return 400
    for guide_value in SCORE_GUIDE_VALUES:
        if max_score <= guide_value:
            return guide_value
    return int(math.ceil(max_score / 500.0) * 500.0)


def score_value_to_y(value: float, bar_top: int, bar_bottom: int, *, display_max: int = SCORE_BAR_DISPLAY_MAX) -> int:
    bar_height = max(1, bar_bottom - bar_top)
    ratio = clamp(value / max(1, display_max), 0.0, 1.0)
    return bar_bottom - int(round(bar_height * ratio))


def draw_rounded_panel(
    image,
    top_left,
    bottom_right,
    color,
    *,
    alpha: float = 0.75,
    radius: int = 24,
    border_color=None,
    border_thickness: int = 2,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right

    overlay = image.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if border_color is None:
        return

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), border_color, border_thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), border_color, border_thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), border_color, border_thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), border_color, border_thickness)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, border_thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, border_thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, border_thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, border_thickness)


def draw_metric_bar(image, origin, size, value, color) -> None:
    x, y = origin
    width, height = size
    value = clamp(value, 0.0, 1.0)

    cv2.rectangle(image, (x, y), (x + width, y + height), (56, 63, 79), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (105, 112, 130), 1)
    cv2.rectangle(image, (x, y), (x + int(width * value), y + height), color, -1)


def prepare_title_banner(title_image, frame_shape):
    if title_image is None:
        return None

    _, frame_width = frame_shape[:2]
    image_height, image_width = title_image.shape[:2]
    if frame_width <= 0 or image_width <= 0 or image_height <= 0:
        return None

    resized_height = max(1, int(round(image_height * (frame_width / image_width))))
    return cv2.resize(title_image, (frame_width, resized_height), interpolation=cv2.INTER_CUBIC)


def overlay_title_banner(image, title_banner) -> None:
    if title_banner is None:
        return

    banner_height, banner_width = title_banner.shape[:2]
    if banner_height <= 0 or banner_width <= 0:
        return

    visible_height = min(banner_height, image.shape[0])
    visible_width = min(banner_width, image.shape[1])
    roi = image[:visible_height, :visible_width]
    banner = title_banner[:visible_height, :visible_width]
    if banner.shape[2] == 4:
        alpha = banner[:, :, 3:4].astype(np.float32) / 255.0
        banner_rgb = banner[:, :, :3].astype(np.float32)
        roi[:] = np.round(banner_rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return

    roi[:] = banner[:, :, :3]


def draw_center_trajectory(image, trace, *, fill_color, edge_color, scale: float) -> None:
    if not trace:
        return

    base_radius = max(3, int(round(5 * scale)))
    total_points = len(trace)
    for index, point in enumerate(trace):
        center = as_point(point)
        radius = base_radius + (1 if index == total_points - 1 else 0)
        cv2.circle(image, center, radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius + 1, edge_color, 1, cv2.LINE_AA)


def draw_center_trajectories(image, pullup_state: PullUpState, frame_shape) -> None:
    scale = frame_ui_scale(frame_shape)
    draw_center_trajectory(
        image,
        pullup_state.body_center_trace,
        fill_color=(255, 188, 92),
        edge_color=(255, 247, 228),
        scale=scale,
    )


def format_graph_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def draw_score_graph(image, trace, bounds, *, scale: float, line_color) -> None:
    if len(trace) < 2:
        return

    graph_left, graph_top, graph_right, graph_bottom = bounds
    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top
    if graph_width <= 0 or graph_height <= 0:
        return

    max_score = int(max(score for _, score in trace))
    score_limit = score_axis_max(max_score)

    for guide_score in SCORE_GUIDE_VALUES:
        if guide_score > score_limit:
            continue
        guide_y = graph_bottom - int(round(graph_height * (guide_score / score_limit)))
        cv2.line(image, (graph_left, guide_y), (graph_right, guide_y), (64, 74, 92), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{int(guide_score):,d}",
            (graph_left, guide_y - int(round(3 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32 * scale,
            (138, 148, 166),
            1,
            cv2.LINE_AA,
        )

    start_time = trace[0][0]
    end_time = trace[-1][0]
    time_span = max(1e-6, end_time - start_time)
    points = []
    for timestamp, score in trace:
        x = graph_left + int(round(((timestamp - start_time) / time_span) * graph_width))
        y = graph_bottom - int(round(graph_height * (clamp(score, 0.0, score_limit) / score_limit)))
        points.append((x, y))

    if len(points) >= 2:
        cv2.polylines(
            image,
            [np.array(points, dtype=np.int32)],
            False,
            line_color,
            max(2, int(round(2 * scale))),
            cv2.LINE_AA,
        )

    last_point = points[-1]
    cv2.circle(image, last_point, max(3, int(round(4 * scale))), line_color, -1, cv2.LINE_AA)
    cv2.circle(image, last_point, max(4, int(round(5 * scale))), (236, 246, 255), 1, cv2.LINE_AA)

    cv2.putText(
        image,
        "0",
        (graph_left, graph_bottom + int(round(10 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        (138, 148, 166),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{int(score_limit):,d}",
        (graph_left, graph_top - int(round(2 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        (138, 148, 166),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "0:00",
        (graph_left, graph_bottom + int(round(24 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        (138, 148, 166),
        1,
        cv2.LINE_AA,
    )
    elapsed_label = format_graph_time(end_time - start_time)
    label_size, _ = cv2.getTextSize(elapsed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, 1)
    cv2.putText(
        image,
        elapsed_label,
        (graph_right - label_size[0], graph_bottom + int(round(24 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        (138, 148, 166),
        1,
        cv2.LINE_AA,
    )


def draw_score_gain_popup(image, pullup_state: PullUpState, frame_shape, *, anchor_x: int, anchor_y: int, scale: float) -> None:
    if pullup_state.last_rep_score_value is None or pullup_state.last_rep_score_frame is None:
        return

    popup_duration_frames = max(8, int(round(pullup_state.fps * 0.70)))
    popup_age = pullup_state.frame_index - pullup_state.last_rep_score_frame
    if popup_age < 0 or popup_age > popup_duration_frames:
        return

    progress = popup_age / popup_duration_frames
    alpha = clamp(1.0 - progress, 0.0, 1.0)
    text = f"+{pullup_state.last_rep_score_value:d}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.12 * scale * (1.0 + (1.0 - progress) * 0.08)
    thickness = max(2, int(round(2.5 * scale)))
    text_color = rep_score_color(pullup_state.last_rep_score_value)

    float_y = anchor_y - int(round(progress * 18 * scale))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x = max(14, int(round(18 * scale)))
    padding_y = max(10, int(round(12 * scale)))
    x1 = max(8, anchor_x)
    x2 = min(frame_shape[1] - 8, x1 + text_width + padding_x * 2)
    y1 = max(8, float_y - text_height - padding_y)
    y2 = min(frame_shape[0] - 8, float_y + baseline + padding_y)

    popup = image.copy()
    draw_rounded_panel(
        popup,
        (x1, y1),
        (x2, y2),
        (10, 16, 24),
        alpha=0.74,
        radius=max(12, int(round(16 * scale))),
        border_color=text_color,
        border_thickness=max(1, int(round(2 * scale))),
    )
    text_origin = (x1 + padding_x, y2 - baseline - padding_y + max(1, int(round(1 * scale))))
    cv2.putText(
        popup,
        text,
        text_origin,
        font,
        font_scale,
        (12, 18, 26),
        thickness + max(2, int(round(2 * scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        popup,
        text,
        text_origin,
        font,
        font_scale,
        (245, 249, 252),
        thickness,
        cv2.LINE_AA,
    )
    cv2.addWeighted(popup, alpha, image, 1.0 - alpha, 0, image)


def draw_score_level_bar(image, metrics: PullUpMetrics, pullup_state: PullUpState, frame_shape, *, title_banner=None) -> None:
    height, width = frame_shape[:2]
    scale = frame_ui_scale(frame_shape)
    level_color = score_level_color(metrics.score_level)
    banner_height = title_banner.shape[0] if title_banner is not None else 0

    panel_left = max(12, int(round(18 * scale)))
    panel_top = max(banner_height + int(round(16 * scale)), int(round(height * 0.24)))
    panel_bottom = min(int(round(height * 0.58)), height - int(round(44 * scale)))
    panel_width = max(144, int(round(190 * scale)))
    min_panel_height = int(round(126 * scale))
    if panel_bottom - panel_top < min_panel_height:
        panel_top = max(banner_height + int(round(18 * scale)), panel_bottom - min_panel_height)
    panel_right = min(width - 12, panel_left + panel_width)

    draw_rounded_panel(
        image,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (12, 18, 28),
        alpha=0.54,
        radius=max(14, int(round(18 * scale))),
        border_color=(62, 76, 98),
        border_thickness=max(1, int(round(2 * scale))),
    )

    title_x = panel_left + int(round(16 * scale))
    bar_left = panel_left + int(round(20 * scale))
    bar_right = bar_left + max(18, int(round(28 * scale)))
    bar_top = panel_top + int(round(128 * scale))
    bar_bottom = panel_bottom - int(round(18 * scale))
    label_x = bar_right + int(round(22 * scale))
    score_y = panel_top + (bar_top - panel_top) // 2 + int(round(4 * scale))

    cv2.putText(
        image,
        f"{metrics.total_score:,d}",
        (title_x, score_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1.26 * scale,
        level_color,
        max(3, int(round(4 * scale))),
        cv2.LINE_AA,
    )

    overlay = image.copy()
    for start_score, end_score, _, level in SCORE_BAR_SEGMENTS:
        segment_top = score_value_to_y(end_score, bar_top, bar_bottom)
        segment_bottom = score_value_to_y(start_score, bar_top, bar_bottom)
        cv2.rectangle(
            overlay,
            (bar_left, segment_top),
            (bar_right, segment_bottom),
            score_level_color(level),
            -1,
        )
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)

    cv2.rectangle(image, (bar_left, bar_top), (bar_right, bar_bottom), (96, 110, 132), 1)
    for score_value in SCORE_GUIDE_VALUES:
        marker_y = score_value_to_y(score_value, bar_top, bar_bottom)
        cv2.line(image, (bar_left - 4, marker_y), (bar_right + 4, marker_y), (112, 124, 146), 1, cv2.LINE_AA)

    filled_top = score_value_to_y(metrics.total_score, bar_top, bar_bottom)
    if metrics.total_score > 0:
        fill_overlay = image.copy()
        cv2.rectangle(
            fill_overlay,
            (bar_left + 1, filled_top),
            (bar_right - 1, bar_bottom - 1),
            level_color,
            -1,
        )
        cv2.addWeighted(fill_overlay, 0.86, image, 0.14, 0, image)
        cv2.line(image, (bar_left - 6, filled_top), (bar_right + 6, filled_top), level_color, max(1, int(round(2 * scale))), cv2.LINE_AA)
        cv2.circle(image, ((bar_left + bar_right) // 2, filled_top), max(3, int(round(4 * scale))), (244, 248, 252), -1, cv2.LINE_AA)

    for start_score, end_score, label, level in SCORE_BAR_SEGMENTS:
        segment_top = score_value_to_y(end_score, bar_top, bar_bottom)
        segment_bottom = score_value_to_y(start_score, bar_top, bar_bottom)
        label_y = segment_top + (segment_bottom - segment_top) // 2 + int(round(4 * scale))
        cv2.putText(
            image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50 * scale,
            score_level_color(level),
            max(1, int(round(1.5 * scale))),
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "5000+",
        (bar_left - int(round(4 * scale)), bar_top - int(round(8 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56 * scale,
        (144, 154, 172),
        max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )
    draw_score_gain_popup(
        image,
        pullup_state,
        frame_shape,
        anchor_x=label_x,
        anchor_y=max(bar_top + int(round(18 * scale)), filled_top + int(round(6 * scale))),
        scale=scale,
    )


def height_marker_bounds(metrics: PullUpMetrics, frame_shape) -> tuple[int, int] | None:
    if metrics.peak_right_shoulder_x is None or metrics.peak_right_wrist_x is None:
        return None

    _, width = frame_shape[:2]
    scale = frame_ui_scale(frame_shape)
    right_shoulder_x = float(metrics.peak_right_shoulder_x)
    right_wrist_x = float(metrics.peak_right_wrist_x)
    segment_center_x = int(round(right_shoulder_x))
    arm_span = abs(right_wrist_x - right_shoulder_x)
    segment_width = max(int(round(30 * scale)), int(round(arm_span * 0.42)))
    half_width = max(18, segment_width // 2)
    marker_x1 = max(14, segment_center_x - half_width)
    marker_x2 = min(width - 14, segment_center_x + half_width)
    return marker_x1, marker_x2


def draw_height_marker(image, *, marker_y: int, marker_x1: int, marker_x2: int, scale: float, label: str, fill_color, line_color, label_position: str = "above") -> None:
    height = image.shape[0]
    band_half_height = max(4, int(round(8 * scale)))

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (marker_x1, max(0, marker_y - band_half_height)),
        (marker_x2, min(height - 1, marker_y + band_half_height)),
        fill_color,
        -1,
    )
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
    cv2.line(image, (marker_x1, marker_y), (marker_x2, marker_y), line_color, max(2, int(round(3 * scale))), cv2.LINE_AA)

    if label_position == "below":
        label_y = min(height - 12, marker_y + int(round(24 * scale)))
    else:
        label_y = max(24, marker_y - int(round(16 * scale)))

    cv2.putText(
        image,
        label,
        (marker_x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56 * scale,
        (242, 224, 224),
        2,
        cv2.LINE_AA,
    )


def draw_peak_marker(image, metrics: PullUpMetrics, frame_shape) -> None:
    marker_bounds = height_marker_bounds(metrics, frame_shape)
    if marker_bounds is None:
        return

    if (
        metrics.peak_shoulder_y is None or
        metrics.best_height_score <= 0
    ):
        return

    scale = frame_ui_scale(frame_shape)
    marker_x1, marker_x2 = marker_bounds

    if metrics.height_target_y is not None:
        draw_height_marker(
            image,
            marker_y=int(round(metrics.height_target_y)),
            marker_x1=marker_x1,
            marker_x2=marker_x2,
            scale=scale,
            label="TARGET HEIGHT",
            fill_color=(54, 126, 214),
            line_color=(88, 184, 255),
            label_position="below",
        )

    draw_height_marker(
        image,
        marker_y=int(round(metrics.peak_shoulder_y)),
        marker_x1=marker_x1,
        marker_x2=marker_x2,
        scale=scale,
        label="MAX HEIGHT",
        fill_color=(58, 58, 220),
        line_color=(70, 70, 235),
    )


def draw_hud(image, metrics: PullUpMetrics, frame_shape, session_label: str, pullup_state: PullUpState) -> None:
    height, width = frame_shape[:2]
    phase_color = PHASE_COLORS.get(metrics.state, PHASE_COLORS["Stand"])
    level_color = score_level_color(metrics.score_level)
    ui_scale = frame_ui_scale(frame_shape)
    font_boost = 1.5

    margin = max(16, int(round(28 * ui_scale)))
    panel_height = int(round(296 * ui_scale))
    panel_left = margin
    panel_right = width - margin
    panel_top = int(round(height * 0.70))
    panel_bottom = min(height - margin, panel_top + panel_height)
    panel_top = max(margin, panel_bottom - panel_height)
    panel_width = panel_right - panel_left

    draw_rounded_panel(
        image,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (12, 18, 28),
        alpha=0.68,
        radius=max(18, int(round(26 * ui_scale))),
        border_color=(67, 80, 104),
        border_thickness=max(1, int(round(2 * ui_scale))),
    )

    title_x = panel_left + int(round(24 * ui_scale))
    cv2.putText(
        image,
        "PERFORMANCE",
        (title_x, panel_top + int(round(42 * ui_scale))),
        cv2.FONT_HERSHEY_DUPLEX,
        0.82 * ui_scale * font_boost,
        (232, 236, 244),
        max(1, int(round(2 * ui_scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        session_label,
        (title_x, panel_top + int(round(92 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56 * ui_scale * font_boost,
        (168, 180, 200),
        max(1, int(round(2 * ui_scale))),
        cv2.LINE_AA,
    )

    reps_x = title_x
    reps_y = panel_bottom - int(round(46 * ui_scale))
    reps_scale = 4.20 * ui_scale
    reps_text = f"{metrics.count:2d}"
    (reps_width, _), _ = cv2.getTextSize(reps_text, cv2.FONT_HERSHEY_DUPLEX, reps_scale, max(2, int(round(3 * ui_scale))))
    cv2.putText(
        image,
        reps_text,
        (reps_x, reps_y),
        cv2.FONT_HERSHEY_DUPLEX,
        reps_scale,
        (244, 247, 250),
        max(2, int(round(3 * ui_scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "REPS",
        (reps_x + reps_width + int(round(18 * ui_scale)), reps_y - int(round(8 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78 * ui_scale * font_boost,
        (128, 140, 161),
        max(1, int(round(2 * ui_scale))),
        cv2.LINE_AA,
    )

    info_x = panel_left + int(round(panel_width * 0.36))
    info_value_x = panel_left + int(round(panel_width * 0.54))
    row_height = int(round(56 * ui_scale))
    tempo_text = "--" if metrics.tempo_spm <= 0 else f"{metrics.tempo_spm:.1f} spm"
    rep_score_value = metrics.last_rep_score
    rep_result_color = rep_score_color(rep_score_value) if rep_score_value > 0 else (188, 198, 214)
    rep_grade_text = rep_grade_label(rep_score_value) if rep_score_value > 0 else "TRACKING"
    rep_score_text = "--" if rep_score_value <= 0 else f"{rep_score_value:d}"

    info_items = [
        ("Grip", metrics.grip, (234, 238, 244)),
        ("Tempo", tempo_text, (234, 238, 244)),
        ("Rep Grade", rep_grade_text, rep_result_color),
        ("Rep Score", rep_score_text, rep_result_color),
    ]
    for index, (label, value, value_color) in enumerate(info_items):
        base_y = panel_top + int(round(82 * ui_scale)) + index * row_height
        cv2.putText(
            image,
            label.upper(),
            (info_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * ui_scale * font_boost,
            (126, 137, 156),
            max(1, int(round(1.5 * ui_scale))),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            value,
            (info_value_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56 * ui_scale * font_boost,
            value_color,
            max(1, int(round(2 * ui_scale))),
            cv2.LINE_AA,
        )

    divider_x = panel_left + int(round(panel_width * 0.73))
    cv2.line(
        image,
        (divider_x, panel_top + int(round(26 * ui_scale))),
        (divider_x, panel_bottom - int(round(24 * ui_scale))),
        (50, 62, 80),
        max(1, int(round(1.5 * ui_scale))),
        cv2.LINE_AA,
    )

    score_block_x = divider_x + int(round(36 * ui_scale))
    score_block_y = panel_top + int(round(54 * ui_scale))
    cv2.putText(
        image,
        "SCORE",
        (score_block_x, score_block_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.44 * ui_scale * font_boost,
        (200, 210, 224),
        max(1, int(round(1.5 * ui_scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{metrics.total_score:,d}",
        (score_block_x, score_block_y + int(round(48 * ui_scale))),
        cv2.FONT_HERSHEY_DUPLEX,
        1.18 * ui_scale * 1.35,
        level_color,
        max(2, int(round(3 * ui_scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "LEVEL",
        (score_block_x, score_block_y + int(round(102 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 * ui_scale * font_boost,
        (126, 137, 156),
        max(1, int(round(1.5 * ui_scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        score_level_label(metrics.score_level),
        (score_block_x, score_block_y + int(round(142 * ui_scale))),
        cv2.FONT_HERSHEY_DUPLEX,
        0.54 * ui_scale * font_boost,
        level_color,
        max(1, int(round(2 * ui_scale))),
        cv2.LINE_AA,
    )

    angle_label = "ANGLE"
    bar_height = max(10, int(round(14 * ui_scale)))
    angle_bar_y = panel_bottom - int(round(48 * ui_scale))
    angle_block_left = score_block_x
    angle_block_right = panel_right - int(round(24 * ui_scale))

    label_size, _ = cv2.getTextSize(angle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * ui_scale * font_boost, max(1, int(round(1.5 * ui_scale))))
    label_y = angle_bar_y + bar_height - int(round(1 * ui_scale))
    cv2.putText(
        image,
        angle_label,
        (angle_block_left, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44 * ui_scale * font_boost,
        (126, 137, 156),
        max(1, int(round(1.5 * ui_scale))),
        cv2.LINE_AA,
    )

    angle_value = f"{int(round(metrics.current_angle_score * 100)):>3d}"
    value_size, _ = cv2.getTextSize(angle_value, cv2.FONT_HERSHEY_DUPLEX, 0.60 * ui_scale * font_boost, max(1, int(round(2 * ui_scale))))
    value_y = label_y + int(round(1 * ui_scale))
    cv2.putText(
        image,
        angle_value,
        (angle_block_right - value_size[0], value_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.60 * ui_scale * font_boost,
        (236, 242, 248),
        max(1, int(round(2 * ui_scale))),
        cv2.LINE_AA,
    )

    bar_x = angle_block_left + label_size[0] + int(round(14 * ui_scale))
    bar_width = max(40, angle_block_right - value_size[0] - int(round(12 * ui_scale)) - bar_x)
    draw_metric_bar(image, (bar_x, angle_bar_y), (bar_width, bar_height), metrics.current_angle_score, phase_color)


def draw_angle_badge(image, position, angle: float, scale: float) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.52 * scale
    thickness = max(1, int(round(2 * scale)))
    text = f"{int(angle):d}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    position_x = int(position[0].item() if isinstance(position[0], np.ndarray) else position[0])
    position_y = int(position[1].item() if isinstance(position[1], np.ndarray) else position[1])

    padding_x = max(10, int(round(14 * scale)))
    padding_y = max(8, int(round(10 * scale)))
    rect_x1 = max(8, position_x - (text_width // 2) - padding_x - max(6, int(round(8 * scale))))
    rect_y1 = max(8, position_y - (text_height // 2) - padding_y)
    rect_x2 = min(image.shape[1] - 8, rect_x1 + text_width + padding_x * 2 + max(18, int(round(24 * scale))))
    rect_y2 = min(image.shape[0] - 8, rect_y1 + text_height + padding_y * 2)

    draw_rounded_panel(
        image,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        (14, 24, 38),
        alpha=0.76,
        radius=max(10, int(round(14 * scale))),
        border_color=(110, 220, 255),
        border_thickness=max(1, int(round(2 * scale))),
    )

    text_x = rect_x1 + ((rect_x2 - rect_x1) - text_width) // 2 - max(2, int(round(3 * scale)))
    text_y = rect_y1 + ((rect_y2 - rect_y1) + text_height) // 2 - 1
    text_origin = (text_x, text_y)
    cv2.putText(image, text, text_origin, font, font_scale, (242, 248, 252), thickness, cv2.LINE_AA)

    degree_x = text_origin[0] + text_width + max(5, int(round(6 * scale)))
    degree_y = text_origin[1] - text_height + max(5, int(round(7 * scale)))
    cv2.putText(
        image,
        "o",
        (degree_x, degree_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 * scale,
        (242, 248, 252),
        max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )


def draw_center_state_overlay(image, state: str, frame_shape) -> None:
    height, width = frame_shape[:2]
    scale = frame_ui_scale(frame_shape)
    phase_color = PHASE_COLORS.get(state, PHASE_COLORS["Stand"])
    label = PHASE_LABELS.get(state, state.upper())

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.15 * scale
    thickness = max(2, int(round(3 * scale)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    center_x = width // 2
    hud_top = int(height * 0.70)
    center_y = min(int(height * 0.67), hud_top - int(round(34 * scale)))
    pad_x = max(18, int(round(32 * scale)))
    pad_y = max(14, int(round(20 * scale)))
    x1 = center_x - text_width // 2 - pad_x
    x2 = center_x + text_width // 2 + pad_x
    y1 = center_y - text_height - pad_y
    y2 = center_y + baseline + pad_y

    draw_rounded_panel(
        image,
        (x1, y1),
        (x2, y2),
        (16, 22, 34),
        alpha=0.58,
        radius=max(14, int(round(24 * scale))),
        border_color=phase_color,
        border_thickness=max(2, int(round(3 * scale))),
    )
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 3
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (12, 18, 28), thickness + max(2, int(round(4 * scale))), cv2.LINE_AA)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (248, 249, 252), thickness, cv2.LINE_AA)


def draw_filled_arc(image, center, point_a, point_b, *, radius: int, color=(0, 0, 255)) -> None:
    center = np.array(center)
    point_a = np.array(point_a)
    point_b = np.array(point_b)

    angle_a = (math.degrees(math.atan2(point_a[1] - center[1], point_a[0] - center[0])) + 360) % 360
    angle_b = (math.degrees(math.atan2(point_b[1] - center[1], point_b[0] - center[0])) + 360) % 360
    diff = (angle_b - angle_a) % 360

    if diff <= 180:
        start_angle = angle_a
        end_angle = angle_a + diff
    else:
        diff = 360 - diff
        start_angle = angle_b
        end_angle = angle_b + diff

    overlay = image.copy()
    cv2.ellipse(overlay, as_point(center), (radius, radius), 0, start_angle, end_angle, color, -1)
    cv2.addWeighted(overlay, 0.32, image, 0.68, 0, image)
    cv2.ellipse(image, as_point(center), (radius, radius), 0, start_angle, end_angle, color, max(1, radius // 14))
    cv2.ellipse(image, as_point(center), (max(1, radius - 10), max(1, radius - 10)), 0, start_angle, end_angle, (235, 245, 255), 1)


def draw_limb_gradient_line(image, point_a, point_b, color_start, color_end, *, thickness: int, segments: int = 14) -> None:
    point_a = np.array(point_a, dtype=np.float32)
    point_b = np.array(point_b, dtype=np.float32)

    for index in range(segments):
        start_ratio = index / segments
        end_ratio = (index + 1) / segments
        start_point = tuple(np.round(point_a * (1 - start_ratio) + point_b * start_ratio).astype(int))
        end_point = tuple(np.round(point_a * (1 - end_ratio) + point_b * end_ratio).astype(int))
        color = tuple(
            int(color_start[channel] * (1 - start_ratio) + color_end[channel] * start_ratio)
            for channel in range(3)
        )
        cv2.line(image, start_point, end_point, color, thickness, cv2.LINE_AA)


def draw_joint_glow(image, center, core_color, glow_color, *, radius: int) -> None:
    for ring_radius, alpha in ((radius + 12, 0.10), (radius + 7, 0.18), (radius + 2, 0.28)):
        glow = image.copy()
        cv2.circle(glow, center, ring_radius, glow_color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, alpha, image, 1 - alpha, 0, image)

    cv2.circle(image, center, radius, core_color, -1, cv2.LINE_AA)
    cv2.circle(image, center, radius + 1, (245, 245, 245), max(1, radius // 5), cv2.LINE_AA)


def extract_primary_pose(results) -> PoseFrame | None:
    if not results or len(results) == 0:
        return None

    keypoints = getattr(results[0], "keypoints", None)
    if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
        return None

    return PoseFrame.from_keypoints(keypoints.data[0].cpu().numpy())


def render_pose_overlay(frame, results, pullup_state: PullUpState, session_label: str, title_banner=None):
    visual = frame.copy()
    overlay = frame.copy()
    pose = extract_primary_pose(results)

    metrics = pullup_state.metrics()
    if pose is None:
        draw_score_level_bar(overlay, metrics, pullup_state, frame.shape, title_banner=title_banner)
        draw_peak_marker(overlay, metrics, frame.shape)
        draw_hud(overlay, metrics, frame.shape, session_label, pullup_state)
        draw_center_state_overlay(overlay, metrics.state, frame.shape)
        output = cv2.addWeighted(overlay, 0.78, visual, 0.22, 0)
        overlay_title_banner(output, title_banner)
        draw_center_trajectories(output, pullup_state, frame.shape)
        return output, metrics

    metrics = pullup_state.update(pose)
    draw_score_level_bar(overlay, metrics, pullup_state, frame.shape, title_banner=title_banner)

    left_angle = pose.left_angle
    right_angle = pose.right_angle
    person_scale = pose_ui_scale(pose)
    limb_thickness = max(4, int(round(8 * person_scale)))
    joint_radius = max(7, int(round(9 * person_scale)))
    arc_radius = max(22, int(round(pose.body_scale * 0.24)))
    badge_offset = int(round(pose.body_scale * 0.28))

    arm_outer_start = (255, 168, 92)
    arm_outer_end = (255, 111, 157)
    arm_inner_start = (82, 232, 255)
    arm_inner_end = (96, 160, 255)
    torso_start = (255, 146, 178)
    torso_end = (126, 191, 255)

    left_angle_badge_position = None
    right_angle_badge_position = None

    if 2 < left_angle < 180:
        draw_filled_arc(overlay, pose.left_elbow, pose.left_wrist, pose.left_shoulder, radius=arc_radius, color=(94, 220, 255))
        left_angle_badge_position = (pose.left_elbow[0], pose.left_elbow[1] + badge_offset)

    if 2 < right_angle < 180:
        draw_filled_arc(overlay, pose.right_elbow, pose.right_wrist, pose.right_shoulder, radius=arc_radius, color=(94, 220, 255))
        right_angle_badge_position = (pose.right_elbow[0], pose.right_elbow[1] + badge_offset)

    draw_limb_gradient_line(overlay, pose.left_shoulder, pose.left_elbow, arm_outer_start, arm_outer_end, thickness=limb_thickness)

    if pose.left_wrist[0] * pose.left_wrist[1] != 0:
        draw_limb_gradient_line(overlay, pose.left_elbow, pose.left_wrist, arm_inner_start, arm_inner_end, thickness=limb_thickness)

    if pose.right_shoulder[0] * pose.right_elbow[1] != 0:
        draw_limb_gradient_line(overlay, pose.right_shoulder, pose.right_elbow, arm_outer_start, arm_outer_end, thickness=limb_thickness)

    if pose.right_wrist[0] * pose.right_wrist[1] != 0:
        draw_limb_gradient_line(overlay, pose.right_elbow, pose.right_wrist, arm_inner_start, arm_inner_end, thickness=limb_thickness)

    if pose.left_shoulder[0] * pose.right_shoulder[1] != 0:
        draw_limb_gradient_line(overlay, pose.left_shoulder, pose.right_shoulder, torso_start, torso_end, thickness=limb_thickness)

    joint_core_colors = {
        "left_shoulder": (255, 194, 138),
        "right_shoulder": (255, 194, 138),
        "left_elbow": (126, 240, 255),
        "right_elbow": (126, 240, 255),
        "left_wrist": (255, 151, 196),
        "right_wrist": (255, 151, 196),
    }
    joint_glow_colors = {
        "left_shoulder": (90, 144, 255),
        "right_shoulder": (90, 144, 255),
        "left_elbow": (88, 218, 255),
        "right_elbow": (88, 218, 255),
        "left_wrist": (255, 120, 176),
        "right_wrist": (255, 120, 176),
    }

    for joint_name in joint_core_colors:
        point = as_point(getattr(pose, joint_name))
        draw_joint_glow(overlay, point, joint_core_colors[joint_name], joint_glow_colors[joint_name], radius=joint_radius)

    badge_scale = clamp(person_scale, 0.80, 1.40)
    if left_angle_badge_position is not None:
        draw_angle_badge(overlay, left_angle_badge_position, left_angle, badge_scale)

    if right_angle_badge_position is not None:
        draw_angle_badge(overlay, right_angle_badge_position, right_angle, badge_scale)

    draw_peak_marker(overlay, metrics, frame.shape)
    draw_hud(overlay, metrics, frame.shape, session_label, pullup_state)
    draw_center_state_overlay(overlay, metrics.state, frame.shape)

    output = cv2.addWeighted(overlay, 0.78, visual, 0.22, 0)
    overlay_title_banner(output, title_banner)
    draw_center_trajectories(output, pullup_state, frame.shape)
    return output, metrics
