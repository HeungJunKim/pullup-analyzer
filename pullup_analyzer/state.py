from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

STATE_STAND = "Stand"
STATE_READY = "Ready"
STATE_DEADHANG = "Deadhang"
STATE_PULL = "Pull"
STATE_DOWN = "Down"

SCORE_LEVEL_BEGINNER = "Beginner"
SCORE_LEVEL_INTERMEDIATE = "Intermediate"
SCORE_LEVEL_ADVANCED = "Advanced"
SCORE_LEVEL_MASTER = "Master"
SCORE_LEVEL_GOD = "God"

SCORE_GUIDE_VALUES = (800, 1600, 2400, 4000)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def calculate_angle(p1, p2, p3, *, is_right_arm: bool = False) -> float:
    if not is_right_arm:
        angle = math.degrees(
            math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
            math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        )
    else:
        angle = math.degrees(
            math.atan2(p1[1] - p2[1], p1[0] - p2[0]) -
            math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        )
    return angle + 360 if angle < 0 else angle


@dataclass(frozen=True)
class AnalysisThresholds:
    ready_wrist_lift_ratio: float = 0.35
    stand_return_wrist_drop_ratio: float = 0.55
    full_extension_angle: float = 150.0
    pull_entry_angle: float = 118.0
    down_entry_angle: float = 95.0
    motion_ratio: float = 0.012
    min_pull_rise_ratio: float = 0.05
    reset_rise_ratio: float = 0.08
    target_ascent_ratio: float = 0.45
    target_angle: float = 72.0
    max_center_sway_ratio: float = 0.075
    trajectory_history_limit: int = 5
    trajectory_sample_seconds: float = 0.05
    score_sample_seconds: float = 0.30
    reference_smoothing: float = 0.22


@dataclass(frozen=True)
class PoseFrame:
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_elbow: np.ndarray
    right_elbow: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray

    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray) -> "PoseFrame":
        def point(index: int) -> np.ndarray:
            return keypoints[index][:2].astype(float)

        return cls(
            left_shoulder=point(5),
            right_shoulder=point(6),
            left_elbow=point(7),
            right_elbow=point(8),
            left_wrist=point(9),
            right_wrist=point(10),
            left_hip=point(11),
            right_hip=point(12),
        )

    @property
    def shoulder_center(self) -> np.ndarray:
        return (self.left_shoulder + self.right_shoulder) / 2.0

    @property
    def hip_center(self) -> np.ndarray:
        return (self.left_hip + self.right_hip) / 2.0

    @property
    def left_angle(self) -> float:
        return calculate_angle(self.left_wrist, self.left_elbow, self.left_shoulder)

    @property
    def right_angle(self) -> float:
        return calculate_angle(self.right_wrist, self.right_elbow, self.right_shoulder, is_right_arm=True)

    @property
    def shoulder_y(self) -> float:
        return float(self.shoulder_center[1])

    @property
    def wrist_y(self) -> float:
        return float((self.left_wrist[1] + self.right_wrist[1]) / 2.0)

    @property
    def shoulder_width(self) -> float:
        return float(np.linalg.norm(self.right_shoulder - self.left_shoulder))

    @property
    def wrist_width(self) -> float:
        return float(np.linalg.norm(self.right_wrist - self.left_wrist))

    @property
    def torso_length(self) -> float:
        return float(np.linalg.norm(self.hip_center - self.shoulder_center))

    @property
    def average_arm_length(self) -> float:
        left_arm = np.linalg.norm(self.left_shoulder - self.left_elbow) + np.linalg.norm(self.left_elbow - self.left_wrist)
        right_arm = np.linalg.norm(self.right_shoulder - self.right_elbow) + np.linalg.norm(self.right_elbow - self.right_wrist)
        return float((left_arm + right_arm) / 2.0)

    @property
    def body_scale(self) -> float:
        return max(1.0, self.shoulder_width, self.torso_length, self.average_arm_length * 0.7)


@dataclass(frozen=True)
class PullUpMetrics:
    state: str = STATE_STAND
    grip: str = "-"
    count: int = 0
    tempo_spm: float = 0.0
    cycle_time: float = 0.0
    pull_time: float = 0.0
    down_time: float = 0.0
    phase_time: float = 0.0
    tempo_score: float = 0.0
    sway_score: float = 0.0
    angle_score: float = 0.0
    current_angle_score: float = 0.0
    ascent_score: float = 0.0
    total_score: int = 0
    score_level: str = SCORE_LEVEL_BEGINNER
    last_rep_score: int = 0
    average_shoulder_rise_px: float = 0.0
    average_shoulder_rise_ratio: float = 0.0
    best_shoulder_rise_px: float = 0.0
    best_shoulder_rise_ratio: float = 0.0
    average_height_score: int = 0
    best_height_score: int = 0
    current_height_score: int = 0
    current_shoulder_rise_px: float = 0.0
    current_shoulder_rise_ratio: float = 0.0
    baseline_shoulder_y: Optional[float] = None
    peak_shoulder_y: Optional[float] = None
    bar_reference_y: Optional[float] = None
    body_scale: float = 0.0
    peak_left_shoulder_x: Optional[float] = None
    peak_left_wrist_x: Optional[float] = None


@dataclass
class PullUpState:
    fps: float = 30.0
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
    current_state: str = STATE_STAND
    grip_type: str = "-"
    is_standing: bool = True
    is_ready: bool = False
    previous_left_angle: Optional[float] = None
    previous_right_angle: Optional[float] = None
    previous_shoulder_y: Optional[float] = None
    pullup_count: int = 0
    frame_index: int = 0
    phase_start_frame: int = 0
    last_rep_frame: Optional[int] = None
    rep_durations: list[float] = field(default_factory=list)
    pull_phase_durations: list[float] = field(default_factory=list)
    down_phase_durations: list[float] = field(default_factory=list)
    rep_peak_rise_pixels: list[float] = field(default_factory=list)
    rep_peak_rise_ratios: list[float] = field(default_factory=list)
    rep_min_elbow_angles: list[float] = field(default_factory=list)
    reference_bar_y: Optional[float] = None
    reference_shoulder_y: Optional[float] = None
    reference_body_scale: Optional[float] = None
    current_rep_peak_y: Optional[float] = None
    current_rep_min_angle: float = 180.0
    current_rep_started_from_deadhang: bool = False
    best_peak_y: Optional[float] = None
    best_peak_left_shoulder_x: Optional[float] = None
    best_peak_left_wrist_x: Optional[float] = None
    shoulder_center_trace: list[tuple[float, float]] = field(default_factory=list)
    hip_center_trace: list[tuple[float, float]] = field(default_factory=list)
    body_center_trace: list[tuple[float, float]] = field(default_factory=list)
    last_trajectory_sample_frame: Optional[int] = None
    rep_scores: list[int] = field(default_factory=list)
    rep_height_scores: list[int] = field(default_factory=list)
    score_total: int = 0
    score_trace: list[tuple[float, int]] = field(default_factory=list)
    last_score_sample_frame: Optional[int] = None
    last_rep_score_value: Optional[int] = None
    last_rep_score_frame: Optional[int] = None

    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self.fps = fps

    def seconds_from_frames(self, frames: int) -> float:
        return frames / self.fps if self.fps > 0 else 0.0

    def _append_limited(self, items: list[float], value: float, limit: int = 8) -> None:
        items.append(value)
        if len(items) > limit:
            items.pop(0)

    def _append_trace_point(self, trace: list[tuple[float, float]], point: np.ndarray) -> None:
        trace.append((float(point[0]), float(point[1])))
        limit = self.thresholds.trajectory_history_limit
        if len(trace) > limit:
            del trace[:-limit]

    def _record_score_sample(self, score: float, *, force: bool = False) -> None:
        sample_interval = max(1, int(round(self.fps * self.thresholds.score_sample_seconds)))
        if not force and self.last_score_sample_frame is not None:
            if self.frame_index - self.last_score_sample_frame < sample_interval:
                return

        elapsed_seconds = self.seconds_from_frames(self.frame_index)
        self.score_trace.append((elapsed_seconds, int(round(max(0.0, score)))))
        self.last_score_sample_frame = self.frame_index

    def _smooth_value(self, current: Optional[float], new_value: float) -> float:
        if current is None:
            return float(new_value)
        alpha = self.thresholds.reference_smoothing
        return current * (1.0 - alpha) + float(new_value) * alpha

    def _motion_threshold(self, body_scale: float) -> float:
        return max(1.0, body_scale * self.thresholds.motion_ratio)

    def _shoulder_rise(self, shoulder_y: float, body_scale: float) -> tuple[float, float]:
        if self.reference_shoulder_y is None:
            return 0.0, 0.0
        rise_pixels = max(0.0, self.reference_shoulder_y - shoulder_y)
        reference_scale = self.reference_body_scale or body_scale
        rise_ratio = rise_pixels / reference_scale if reference_scale > 0 else 0.0
        return rise_pixels, rise_ratio

    def _height_score(self, shoulder_y: Optional[float]) -> int:
        if shoulder_y is None or self.reference_shoulder_y is None or self.reference_bar_y is None:
            return 0
        target_distance = self.reference_shoulder_y - self.reference_bar_y
        if target_distance <= 1.0:
            return 0
        rise_to_peak = self.reference_shoulder_y - shoulder_y
        return int(round(clamp(rise_to_peak / target_distance, 0.0, 1.0) * 100))

    def _update_hanging_reference(self, pose: PoseFrame) -> None:
        self.reference_bar_y = self._smooth_value(self.reference_bar_y, pose.wrist_y)
        self.reference_shoulder_y = self._smooth_value(self.reference_shoulder_y, pose.shoulder_y)
        self.reference_body_scale = self._smooth_value(self.reference_body_scale, pose.body_scale)

    def _reset_hanging_state(self) -> None:
        self.is_standing = True
        self.is_ready = False
        self.grip_type = "-"
        self.reference_bar_y = None
        self.reference_shoulder_y = None
        self.reference_body_scale = None
        self.current_rep_peak_y = None
        self.current_rep_min_angle = 180.0
        self.current_rep_started_from_deadhang = False
        self.shoulder_center_trace.clear()
        self.hip_center_trace.clear()
        self.body_center_trace.clear()
        self.last_trajectory_sample_frame = None

    def _resolve_grip_type(self, pose: PoseFrame) -> str:
        shoulder_width = pose.shoulder_width
        wrist_width = pose.wrist_width
        ratio = wrist_width / shoulder_width if shoulder_width > 0 else 1.0
        return "Wide" if ratio >= 2.0 else "Narrow"

    def _record_transition(self, new_state: str) -> None:
        if new_state == self.current_state:
            return

        phase_duration = self.seconds_from_frames(self.frame_index - self.phase_start_frame)
        if self.current_state == STATE_PULL:
            self._append_limited(self.pull_phase_durations, phase_duration)
            self._finalize_rep_tracking()
        elif self.current_state == STATE_DOWN:
            self._append_limited(self.down_phase_durations, phase_duration)

        if new_state == STATE_PULL:
            if self.last_rep_frame is not None:
                cycle_duration = self.seconds_from_frames(self.frame_index - self.last_rep_frame)
                self._append_limited(self.rep_durations, cycle_duration)
            self.last_rep_frame = self.frame_index

        self.phase_start_frame = self.frame_index

    def _start_rep_tracking(self, pose: PoseFrame, elbow_min_angle: float, *, started_from_deadhang: bool) -> None:
        shoulder_y = pose.shoulder_y
        self.current_rep_peak_y = shoulder_y
        self.current_rep_min_angle = elbow_min_angle
        self.current_rep_started_from_deadhang = started_from_deadhang
        self._update_best_peak(pose)

    def _update_rep_tracking(self, pose: PoseFrame, elbow_min_angle: float) -> None:
        shoulder_y = pose.shoulder_y
        if self.current_rep_peak_y is None:
            self._start_rep_tracking(pose, elbow_min_angle, started_from_deadhang=self.current_state == STATE_DEADHANG)
            return

        previous_peak_y = self.current_rep_peak_y
        self.current_rep_peak_y = min(self.current_rep_peak_y, shoulder_y)
        self.current_rep_min_angle = min(self.current_rep_min_angle, elbow_min_angle)
        if previous_peak_y is not None and self.current_rep_peak_y >= previous_peak_y:
            return
        self._update_best_peak(pose)

    def _update_best_peak(self, pose: PoseFrame) -> None:
        shoulder_y = pose.shoulder_y
        if self.reference_shoulder_y is None:
            return
        if shoulder_y >= self.reference_shoulder_y:
            return
        if self.best_peak_y is not None and shoulder_y >= self.best_peak_y:
            return
        self.best_peak_y = shoulder_y
        self.best_peak_left_shoulder_x = float(pose.left_shoulder[0])
        self.best_peak_left_wrist_x = float(pose.left_wrist[0])

    def _current_tempo_score(self) -> float:
        if len(self.rep_durations) < 2:
            return 1.0

        average_cycle = float(np.mean(self.rep_durations))
        if average_cycle <= 0:
            return 1.0

        cycle_cv = float(np.std(self.rep_durations) / average_cycle)
        return clamp(1.0 - cycle_cv * 1.8, 0.0, 1.0)

    def _current_sway_score(self) -> float:
        reference_scale = self.reference_body_scale or 1.0
        if len(self.body_center_trace) < 3:
            return 1.0

        center_sway_ratio = self._trace_sway_ratio(self.body_center_trace, reference_scale)
        return clamp(1.0 - center_sway_ratio / self.thresholds.max_center_sway_ratio, 0.0, 1.0)

    def _angle_quality(self, minimum_angle: float) -> float:
        angle_span = max(1.0, self.thresholds.full_extension_angle - self.thresholds.target_angle)
        return clamp(
            (self.thresholds.full_extension_angle - minimum_angle) / angle_span,
            0.0,
            1.0,
        )

    def _build_rep_score(self, peak_shoulder_y: float, rep_min_angle: float) -> tuple[int, int]:
        angle_quality = self._angle_quality(rep_min_angle)
        height_quality = self._height_score(peak_shoulder_y) / 100.0
        sway_quality = self._current_sway_score()
        tempo_quality = self._current_tempo_score()

        sway_penalty = (1.0 - sway_quality) * 5.0
        tempo_penalty = (1.0 - tempo_quality) * 5.0
        height_penalty = (1.0 - height_quality) * 10.0
        angle_penalty = (1.0 - angle_quality) * 10.0
        deadhang_bonus = 10.0 if self.current_rep_started_from_deadhang else 0.0

        rep_score = int(round(clamp(
            90.0 - sway_penalty - tempo_penalty - height_penalty - angle_penalty + deadhang_bonus,
            0.0,
            100.0,
        )))
        return rep_score, int(round(height_quality * 100))

    def _score_level(self, total_score: int) -> str:
        if total_score >= 4000:
            return SCORE_LEVEL_GOD
        if total_score >= 2400:
            return SCORE_LEVEL_MASTER
        if total_score >= 1600:
            return SCORE_LEVEL_ADVANCED
        if total_score >= 800:
            return SCORE_LEVEL_INTERMEDIATE
        return SCORE_LEVEL_BEGINNER

    def _finalize_rep_tracking(self) -> None:
        if self.current_rep_peak_y is None or self.reference_shoulder_y is None:
            self.current_rep_peak_y = None
            self.current_rep_min_angle = 180.0
            self.current_rep_started_from_deadhang = False
            return

        reference_scale = self.reference_body_scale or 1.0
        rise_pixels = max(0.0, self.reference_shoulder_y - self.current_rep_peak_y)
        rise_ratio = rise_pixels / reference_scale if reference_scale > 0 else 0.0

        self._append_limited(self.rep_peak_rise_pixels, rise_pixels)
        self._append_limited(self.rep_peak_rise_ratios, rise_ratio)
        self._append_limited(self.rep_min_elbow_angles, self.current_rep_min_angle)
        rep_score, rep_height_score = self._build_rep_score(self.current_rep_peak_y, self.current_rep_min_angle)
        self.rep_scores.append(rep_score)
        self.rep_height_scores.append(rep_height_score)
        self.score_total += rep_score
        self.last_rep_score_value = rep_score
        self.last_rep_score_frame = self.frame_index
        self._record_score_sample(self.score_total, force=True)

        self.current_rep_peak_y = None
        self.current_rep_min_angle = 180.0
        self.current_rep_started_from_deadhang = False

    def _update_center_traces(self, pose: PoseFrame) -> None:
        sample_interval = max(1, int(round(self.fps * self.thresholds.trajectory_sample_seconds)))
        if self.last_trajectory_sample_frame is not None:
            if self.frame_index - self.last_trajectory_sample_frame < sample_interval:
                return

        self._append_trace_point(self.shoulder_center_trace, pose.shoulder_center)
        self._append_trace_point(self.hip_center_trace, pose.hip_center)
        body_center = (pose.shoulder_center + pose.hip_center) / 2.0
        self._append_trace_point(self.body_center_trace, body_center)
        self.last_trajectory_sample_frame = self.frame_index

    def _trace_sway_ratio(self, trace: list[tuple[float, float]], reference_scale: float) -> float:
        if len(trace) < 3 or reference_scale <= 0:
            return 0.0
        x_positions = np.array([point[0] for point in trace], dtype=float)
        return float(np.std(x_positions) / reference_scale)

    def _can_detect_pull(
        self,
        left_angle: float,
        right_angle: float,
        rise_ratio: float,
        shoulder_velocity: float,
        body_scale: float,
    ) -> bool:
        if self.previous_left_angle is None or self.previous_right_angle is None:
            return False

        elbow_min_angle = min(left_angle, right_angle)
        closing_arms = (
            self.previous_left_angle - left_angle > 1.5 and
            self.previous_right_angle - right_angle > 1.5
        )
        rising_body = shoulder_velocity > self._motion_threshold(body_scale)
        enough_rise = rise_ratio >= self.thresholds.min_pull_rise_ratio

        return (
            elbow_min_angle <= self.thresholds.pull_entry_angle and
            enough_rise and
            (closing_arms or rising_body)
        )

    def _can_detect_down(
        self,
        left_angle: float,
        right_angle: float,
        rise_ratio: float,
        shoulder_velocity: float,
        body_scale: float,
    ) -> bool:
        if self.previous_left_angle is None or self.previous_right_angle is None:
            return False

        elbow_min_angle = min(left_angle, right_angle)
        opening_arms = (
            left_angle - self.previous_left_angle > 1.5 and
            right_angle - self.previous_right_angle > 1.5
        )
        descending_body = shoulder_velocity < -self._motion_threshold(body_scale)

        return (
            self.current_state == STATE_PULL and
            descending_body and
            (opening_arms or elbow_min_angle >= self.thresholds.down_entry_angle or rise_ratio <= self.thresholds.reset_rise_ratio)
        )

    def _should_return_to_stand(self, wrist_y: float, shoulder_y: float, body_scale: float) -> bool:
        if self.reference_bar_y is None:
            return False
        wrist_far_below_bar = wrist_y > self.reference_bar_y + body_scale * self.thresholds.stand_return_wrist_drop_ratio
        shoulder_relaxed = self.reference_shoulder_y is None or shoulder_y >= self.reference_shoulder_y - body_scale * 0.05
        return wrist_far_below_bar and shoulder_relaxed

    def metrics(self) -> PullUpMetrics:
        average_cycle = float(np.mean(self.rep_durations)) if self.rep_durations else 0.0
        average_pull = float(np.mean(self.pull_phase_durations)) if self.pull_phase_durations else 0.0
        average_down = float(np.mean(self.down_phase_durations)) if self.down_phase_durations else 0.0
        tempo_spm = 60.0 / average_cycle if average_cycle > 0 else 0.0

        tempo_score = self._current_tempo_score()
        sway_score = self._current_sway_score()

        current_min_angle = min(
            self.previous_left_angle or 180.0,
            self.previous_right_angle or 180.0,
        )
        average_min_angle = float(np.mean(self.rep_min_elbow_angles)) if self.rep_min_elbow_angles else current_min_angle
        current_angle_score = self._angle_quality(current_min_angle)
        angle_score = self._angle_quality(average_min_angle)

        average_rise_pixels = float(np.mean(self.rep_peak_rise_pixels)) if self.rep_peak_rise_pixels else 0.0
        average_rise_ratio = float(np.mean(self.rep_peak_rise_ratios)) if self.rep_peak_rise_ratios else 0.0
        best_rise_pixels = max(self.rep_peak_rise_pixels) if self.rep_peak_rise_pixels else 0.0
        best_rise_ratio = max(self.rep_peak_rise_ratios) if self.rep_peak_rise_ratios else 0.0

        if self.reference_shoulder_y is not None and self.best_peak_y is not None and best_rise_pixels <= 0:
            best_rise_pixels = max(0.0, self.reference_shoulder_y - self.best_peak_y)
            if self.reference_body_scale:
                best_rise_ratio = best_rise_pixels / self.reference_body_scale

        current_rise_pixels = 0.0
        current_rise_ratio = 0.0
        if self.reference_shoulder_y is not None and self.previous_shoulder_y is not None:
            current_rise_pixels = max(0.0, self.reference_shoulder_y - self.previous_shoulder_y)
            if self.reference_body_scale:
                current_rise_ratio = current_rise_pixels / self.reference_body_scale

        average_height_score = int(round(float(np.mean(self.rep_height_scores)))) if self.rep_height_scores else 0
        best_height_score = self._height_score(self.best_peak_y)
        current_height_score = self._height_score(self.previous_shoulder_y)

        ascent_score = (
            average_height_score / 100.0
            if average_height_score > 0
            else current_height_score / 100.0
        )

        total_score = max(0, int(round(self.score_total)))
        score_level = self._score_level(total_score)

        return PullUpMetrics(
            state=self.current_state,
            grip=self.grip_type,
            count=self.pullup_count,
            tempo_spm=tempo_spm,
            cycle_time=average_cycle,
            pull_time=average_pull,
            down_time=average_down,
            phase_time=self.seconds_from_frames(self.frame_index - self.phase_start_frame),
            tempo_score=tempo_score,
            sway_score=sway_score,
            angle_score=angle_score,
            current_angle_score=current_angle_score,
            ascent_score=ascent_score,
            total_score=total_score,
            score_level=score_level,
            last_rep_score=self.last_rep_score_value or 0,
            average_shoulder_rise_px=average_rise_pixels,
            average_shoulder_rise_ratio=average_rise_ratio,
            best_shoulder_rise_px=best_rise_pixels,
            best_shoulder_rise_ratio=best_rise_ratio,
            average_height_score=average_height_score,
            best_height_score=best_height_score,
            current_height_score=current_height_score,
            current_shoulder_rise_px=current_rise_pixels,
            current_shoulder_rise_ratio=current_rise_ratio,
            baseline_shoulder_y=self.reference_shoulder_y,
            peak_shoulder_y=self.best_peak_y,
            bar_reference_y=self.reference_bar_y,
            body_scale=self.reference_body_scale or 0.0,
            peak_left_shoulder_x=self.best_peak_left_shoulder_x,
            peak_left_wrist_x=self.best_peak_left_wrist_x,
        )

    def update(self, pose: PoseFrame) -> PullUpMetrics:
        self.frame_index += 1

        left_angle = pose.left_angle
        right_angle = pose.right_angle
        elbow_min_angle = min(left_angle, right_angle)
        shoulder_y = pose.shoulder_y
        wrist_y = pose.wrist_y
        body_scale = pose.body_scale

        shoulder_velocity = 0.0 if self.previous_shoulder_y is None else self.previous_shoulder_y - shoulder_y
        rise_pixels, rise_ratio = self._shoulder_rise(shoulder_y, body_scale)

        wrists_above_shoulders = wrist_y < pose.shoulder_y - body_scale * self.thresholds.ready_wrist_lift_ratio
        full_extension = left_angle >= self.thresholds.full_extension_angle and right_angle >= self.thresholds.full_extension_angle

        new_state = self.current_state

        if self.is_standing:
            if wrists_above_shoulders:
                self.is_standing = False
                new_state = STATE_READY
        else:
            if self._should_return_to_stand(wrist_y, shoulder_y, body_scale):
                self._reset_hanging_state()
                new_state = STATE_STAND

        if not self.is_standing and full_extension and wrists_above_shoulders:
            self.is_ready = True
            self.grip_type = self._resolve_grip_type(pose)
            self._update_hanging_reference(pose)
            rise_pixels, rise_ratio = self._shoulder_rise(shoulder_y, body_scale)
            new_state = STATE_DEADHANG

        if self.is_ready:
            self._update_center_traces(pose)
            if new_state != STATE_STAND and self._can_detect_pull(left_angle, right_angle, rise_ratio, shoulder_velocity, body_scale):
                if self.current_state != STATE_PULL:
                    self.pullup_count += 1
                    self._start_rep_tracking(pose, elbow_min_angle, started_from_deadhang=self.current_state == STATE_DEADHANG)
                else:
                    self._update_rep_tracking(pose, elbow_min_angle)
                new_state = STATE_PULL
            elif self.current_state == STATE_PULL:
                self._update_rep_tracking(pose, elbow_min_angle)
                if self._can_detect_down(left_angle, right_angle, rise_ratio, shoulder_velocity, body_scale):
                    new_state = STATE_DOWN
                else:
                    new_state = STATE_PULL
            elif self.current_state == STATE_DOWN and full_extension and rise_ratio <= self.thresholds.reset_rise_ratio and wrists_above_shoulders:
                self._update_hanging_reference(pose)
                new_state = STATE_DEADHANG

        self.previous_left_angle = left_angle
        self.previous_right_angle = right_angle
        self.previous_shoulder_y = shoulder_y

        self._record_transition(new_state)
        self.current_state = new_state
        metrics = self.metrics()
        if self.is_ready:
            self._record_score_sample(metrics.total_score)
        return metrics
