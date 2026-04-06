from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .analyzer import (
    APP_INFO,
    DEFAULT_MODELS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_RESULTS_DIR,
    FFmpegVideoWriter,
    PROJECT_DIR,
    capture_video_metadata,
    ensure_directory,
    load_model,
    resolve_inference_device_with_policy,
    resolve_model_path,
)
from .rendering import draw_joint_glow, draw_limb_gradient_line, extract_primary_pose, format_video_session_label
from .state import PoseFrame, PullUpState, STATE_DEADHANG, STATE_DOWN, STATE_PULL, STATE_STAND


ANALYSIS_COLORS = {
    "video1": (78, 185, 255),
    "video2": (255, 174, 92),
}
BACKGROUND_TOP = np.array((12, 20, 30), dtype=np.float32)
BACKGROUND_BOTTOM = np.array((24, 34, 46), dtype=np.float32)
LABEL_TEXT_COLOR = (242, 247, 255)
LABEL_SUBTEXT_COLOR = (182, 194, 212)
OUTPUT_BORDER_PADDING = 48
SOURCE_MASK_FILL = 255
SNAPSHOT_SAME_POINT_FADE_SECONDS = 1.7
SNAPSHOT_NEXT_POINT_FADE_SECONDS = 1.0
VIDEO2_SCALE_BIAS = 1.05


@dataclass(frozen=True)
class CompareConfig:
    video1: Path
    video2: Path
    models_dir: Path
    results_dir: Path
    output_path: Path
    model_name: str | None
    conf: float
    iou: float
    device: str
    max_reps: int
    target_angles: tuple[float, ...]
    hold_seconds: float
    fade_seconds: float
    output_fps: float


@dataclass
class AngleMatch:
    target_angle: float
    frame_index: int | None = None
    measured_angle: float | None = None
    error: float = math.inf

    def consider(self, frame_index: int, measured_angle: float) -> None:
        error = abs(measured_angle - self.target_angle)
        if error < self.error:
            self.frame_index = frame_index
            self.measured_angle = measured_angle
            self.error = error


@dataclass
class RepSegment:
    rep_number: int
    start_frame: int
    end_frame: int
    pose_by_frame: dict[int, PoseFrame] = field(default_factory=dict)
    angle_matches: dict[float, AngleMatch] = field(default_factory=dict)
    lowest_shoulder_frame_index: int | None = None
    lowest_shoulder_y: float | None = None
    peak_shoulder_frame_index: int | None = None
    peak_shoulder_y: float | None = None
    max_angle_frame_index: int | None = None
    max_angle_value: float | None = None
    min_angle_frame_index: int | None = None
    min_angle_value: float | None = None

    @property
    def show_pose_overlay(self) -> bool:
        return self.rep_number <= 2


@dataclass
class RunningRep:
    rep_number: int
    start_frame: int
    pose_by_frame: dict[int, PoseFrame]
    angle_matches: dict[float, AngleMatch]
    lowest_shoulder_frame_index: int | None = None
    lowest_shoulder_y: float | None = None
    peak_shoulder_frame_index: int | None = None
    peak_shoulder_y: float | None = None
    max_angle_frame_index: int | None = None
    max_angle_value: float | None = None
    min_angle_frame_index: int | None = None
    min_angle_value: float | None = None

    @classmethod
    def create(cls, rep_number: int, start_frame: int, target_angles: tuple[float, ...]) -> "RunningRep":
        return cls(
            rep_number=rep_number,
            start_frame=start_frame,
            pose_by_frame={},
            angle_matches={angle: AngleMatch(target_angle=angle) for angle in target_angles},
        )

    def to_segment(self, end_frame: int) -> RepSegment:
        return RepSegment(
            rep_number=self.rep_number,
            start_frame=self.start_frame,
            end_frame=end_frame,
            pose_by_frame=dict(self.pose_by_frame),
            angle_matches={angle: match for angle, match in self.angle_matches.items()},
            lowest_shoulder_frame_index=self.lowest_shoulder_frame_index,
            lowest_shoulder_y=self.lowest_shoulder_y,
            peak_shoulder_frame_index=self.peak_shoulder_frame_index,
            peak_shoulder_y=self.peak_shoulder_y,
            max_angle_frame_index=self.max_angle_frame_index,
            max_angle_value=self.max_angle_value,
            min_angle_frame_index=self.min_angle_frame_index,
            min_angle_value=self.min_angle_value,
        )

    def consider_shoulder_height(self, frame_index: int, shoulder_y: float) -> None:
        if self.lowest_shoulder_y is None or shoulder_y > self.lowest_shoulder_y:
            self.lowest_shoulder_y = shoulder_y
            self.lowest_shoulder_frame_index = frame_index
        if self.peak_shoulder_y is None or shoulder_y < self.peak_shoulder_y:
            self.peak_shoulder_y = shoulder_y
            self.peak_shoulder_frame_index = frame_index

    def consider_extrema(self, frame_index: int, measured_angle: float) -> None:
        if self.max_angle_value is None or measured_angle > self.max_angle_value:
            self.max_angle_value = measured_angle
            self.max_angle_frame_index = frame_index
        if self.min_angle_value is None or measured_angle < self.min_angle_value:
            self.min_angle_value = measured_angle
            self.min_angle_frame_index = frame_index


@dataclass
class VideoPlacement:
    scale: float
    translate_x: float
    translate_y: float


@dataclass
class VideoAnalysis:
    label: str
    input_path: Path
    metadata: object
    segments: list[RepSegment]
    torso_lengths: list[float]
    hip_widths: list[float]
    shoulder_centers: list[tuple[float, float]]
    hip_centers: list[tuple[float, float]]
    deadhang_shoulder_centers: list[tuple[float, float]]
    deadhang_hip_widths: list[float]
    deadhang_hip_centers: list[tuple[float, float]]
    placement: VideoPlacement | None = None

    @property
    def available_reps(self) -> int:
        return len(self.segments)

    @property
    def session_date_label(self) -> str:
        return format_video_session_label(self.input_path)

    @property
    def median_torso_length(self) -> float:
        if self.torso_lengths:
            return float(np.median(np.asarray(self.torso_lengths, dtype=float)))
        frame_width, frame_height = self.metadata.output_size
        return max(1.0, min(frame_width, frame_height) * 0.24)

    @property
    def median_hip_width(self) -> float:
        samples = self.deadhang_hip_widths or self.hip_widths
        if samples:
            return float(np.median(np.asarray(samples, dtype=float)))
        frame_width, _ = self.metadata.output_size
        return max(1.0, frame_width * 0.18)

    @property
    def baseline_shoulder_center(self) -> np.ndarray:
        samples = self.deadhang_shoulder_centers or self.shoulder_centers
        if samples:
            return np.median(np.asarray(samples, dtype=float), axis=0)
        frame_width, frame_height = self.metadata.output_size
        return np.asarray((frame_width / 2.0, frame_height * 0.42), dtype=float)

    @property
    def baseline_hip_center(self) -> np.ndarray:
        samples = self.deadhang_hip_centers or self.hip_centers
        if samples:
            return np.median(np.asarray(samples, dtype=float), axis=0)
        frame_width, frame_height = self.metadata.output_size
        return np.asarray((frame_width / 2.0, frame_height * 0.66), dtype=float)


class SequentialFrameReader:
    def __init__(self, input_path: Path, metadata) -> None:
        self.input_path = input_path
        self.metadata = metadata
        self.capture = cv2.VideoCapture(str(input_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"영상을 열지 못했습니다: {input_path}")
        self.next_frame_index = 0

    def close(self) -> None:
        self.capture.release()

    def iter_range(self, start_frame: int, end_frame: int):
        if start_frame > end_frame:
            return
        if start_frame != self.next_frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.next_frame_index = start_frame

        while self.next_frame_index <= end_frame:
            has_frame, frame = self.capture.read()
            if not has_frame:
                break
            current_frame_index = self.next_frame_index
            self.next_frame_index += 1
            if self.metadata.rotate_for_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            yield current_frame_index, frame


class _PrintReporter:
    def info(self, message: str) -> None:
        print(f"[Info] {message}")

    def warn(self, message: str) -> None:
        print(f"[Warn] {message}")

    def error(self, message: str) -> None:
        print(f"[Error] {message}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an aligned comparison video for two pull-up clips.")
    parser.add_argument("video1", help="First input video path")
    parser.add_argument("video2", help="Second input video path")
    parser.add_argument("--models-dir", default=str((PROJECT_DIR / "models") if (PROJECT_DIR / "models").exists() else DEFAULT_MODELS_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output")
    parser.add_argument("--model-name", default=os.environ.get("PULLUP_MODEL_NAME") or DEFAULT_MODEL_NAME)
    parser.add_argument("--conf", type=float, default=float(os.environ.get("PULLUP_CONF", "0.55")))
    parser.add_argument("--iou", type=float, default=float(os.environ.get("PULLUP_IOU", "0.50")))
    parser.add_argument("--max-reps", type=int, default=5)
    parser.add_argument("--angles", type=float, nargs="+", default=(100.0, 80.0, 60.0))
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--fade-seconds", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def build_compare_config(args: argparse.Namespace) -> CompareConfig:
    input_video1 = Path(args.video1).resolve()
    input_video2 = Path(args.video2).resolve()
    results_dir = Path(args.results_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else results_dir / f"{input_video1.stem}_vs_{input_video2.stem}_compare.mp4"
    device, device_note = resolve_inference_device_with_policy(allow_gpu=not args.cpu_only)
    print(f"[Info] Inference device: {device} ({device_note})")
    return CompareConfig(
        video1=input_video1,
        video2=input_video2,
        models_dir=Path(args.models_dir).resolve(),
        results_dir=results_dir,
        output_path=output_path,
        model_name=args.model_name,
        conf=args.conf,
        iou=args.iou,
        device=device,
        max_reps=max(1, int(args.max_reps)),
        target_angles=tuple(float(angle) for angle in args.angles),
        hold_seconds=max(0.0, float(args.hold_seconds)),
        fade_seconds=max(0.0, float(args.fade_seconds)),
        output_fps=max(1.0, float(args.output_fps)),
    )


def validate_compare_config(config: CompareConfig) -> None:
    for input_path in (config.video1, config.video2):
        if not input_path.exists():
            raise FileNotFoundError(f"입력 영상을 찾지 못했습니다: {input_path}")
        if not input_path.is_file():
            raise RuntimeError(f"입력 경로가 파일이 아닙니다: {input_path}")


def print_banner() -> None:
    print("=" * 72)
    print(f"{APP_INFO.name} Compare | version {APP_INFO.version}")
    print("=" * 72)


def infer_results(model, frame, *, config: CompareConfig, device: str) -> tuple[object, str]:
    try:
        results = model(
            frame,
            conf=config.conf,
            iou=config.iou,
            classes=[0],
            verbose=False,
            device=device,
        )
        return results, device
    except Exception as exc:
        if device == "0":
            print(f"[Warn] GPU 0 inference failed, falling back to CPU: {exc}")
            results = model(
                frame,
                conf=config.conf,
                iou=config.iou,
                classes=[0],
                verbose=False,
                device="cpu",
            )
            return results, "cpu"
        raise


def analyze_video(model, input_path: Path, *, label: str, config: CompareConfig) -> VideoAnalysis:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열지 못했습니다: {input_path}")

    metadata = capture_video_metadata(capture)
    state = PullUpState(fps=metadata.fps)
    device = config.device
    current_rep: RunningRep | None = None
    pending_rep: RunningRep | None = None
    segments: list[RepSegment] = []
    torso_lengths: list[float] = []
    hip_widths: list[float] = []
    shoulder_centers: list[tuple[float, float]] = []
    hip_centers: list[tuple[float, float]] = []
    deadhang_shoulder_centers: list[tuple[float, float]] = []
    deadhang_hip_widths: list[float] = []
    deadhang_hip_centers: list[tuple[float, float]] = []
    pre_pull_max_frame_index: int | None = None
    pre_pull_max_pose: PoseFrame | None = None
    pre_pull_max_angle: float | None = None
    required_reps = max(config.max_reps, 7)
    last_video_frame_index = -1
    pending_rep_end_frame = -1

    progress = tqdm(
        total=metadata.total_frames if metadata.total_frames > 0 else None,
        desc=f"Analyze {label}",
        unit="frame",
        dynamic_ncols=True,
    )

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            last_video_frame_index += 1
            if metadata.rotate_for_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            previous_state = state.current_state
            previous_count = state.pullup_count

            results, device = infer_results(model, frame, config=config, device=device)
            pose = extract_primary_pose(results)
            if pose is None:
                metrics = state.metrics()
                current_state = state.current_state
                current_count = metrics.count
            else:
                metrics = state.update(pose)
                current_state = state.current_state
                current_count = metrics.count

                if state.is_ready or current_rep is not None or current_count > 0:
                    torso_lengths.append(float(pose.torso_length))
                    hip_width = float(np.linalg.norm(pose.right_hip - pose.left_hip))
                    hip_widths.append(hip_width)
                    shoulder_center = (float(pose.shoulder_center[0]), float(pose.shoulder_center[1]))
                    hip_center = (float(pose.hip_center[0]), float(pose.hip_center[1]))
                    shoulder_centers.append(shoulder_center)
                    hip_centers.append(hip_center)
                    if current_state == STATE_DEADHANG:
                        deadhang_shoulder_centers.append(shoulder_center)
                        deadhang_hip_widths.append(hip_width)
                        deadhang_hip_centers.append(hip_center)

                elbow_angle = min(float(pose.left_angle), float(pose.right_angle))

                if current_state == STATE_DEADHANG:
                    if previous_state != STATE_DEADHANG:
                        pre_pull_max_frame_index = last_video_frame_index
                        pre_pull_max_pose = pose
                        pre_pull_max_angle = elbow_angle
                    elif pre_pull_max_angle is None or elbow_angle > pre_pull_max_angle:
                        pre_pull_max_frame_index = last_video_frame_index
                        pre_pull_max_pose = pose
                        pre_pull_max_angle = elbow_angle

                if previous_state != STATE_PULL and current_state == STATE_PULL and current_rep is None:
                    if pending_rep is not None:
                        segments.append(pending_rep.to_segment(max(pending_rep.start_frame, pending_rep_end_frame)))
                        if len(segments) >= required_reps:
                            progress.update(1)
                            break
                        pending_rep = None
                    current_rep = RunningRep.create(previous_count + 1, last_video_frame_index, config.target_angles)
                    if pre_pull_max_frame_index is not None and pre_pull_max_pose is not None and pre_pull_max_angle is not None:
                        current_rep.pose_by_frame[pre_pull_max_frame_index] = pre_pull_max_pose
                        current_rep.consider_shoulder_height(pre_pull_max_frame_index, float(pre_pull_max_pose.shoulder_y))
                        current_rep.consider_extrema(pre_pull_max_frame_index, pre_pull_max_angle)

                if current_rep is not None:
                    current_rep.pose_by_frame[last_video_frame_index] = pose
                    current_rep.consider_shoulder_height(last_video_frame_index, float(pose.shoulder_y))
                    current_rep.consider_extrema(last_video_frame_index, elbow_angle)
                    if current_state == STATE_PULL:
                        for angle_match in current_rep.angle_matches.values():
                            angle_match.consider(last_video_frame_index, elbow_angle)
                elif pending_rep is not None and current_state != STATE_PULL:
                    pending_rep.pose_by_frame[last_video_frame_index] = pose
                    pending_rep.consider_shoulder_height(last_video_frame_index, float(pose.shoulder_y))
                    pending_rep.consider_extrema(last_video_frame_index, elbow_angle)

            if current_rep is not None and previous_state == STATE_DOWN and current_state in {STATE_DEADHANG, STATE_STAND}:
                if current_count >= current_rep.rep_number:
                    pending_rep = current_rep
                    pending_rep_end_frame = last_video_frame_index
                current_rep = None

            if pending_rep is not None and current_state != STATE_PULL:
                pending_rep_end_frame = last_video_frame_index

            progress.update(1)
    finally:
        capture.release()
        progress.close()

    if current_rep is not None and state.pullup_count >= current_rep.rep_number:
        segments.append(current_rep.to_segment(last_video_frame_index))
    elif pending_rep is not None:
        segments.append(pending_rep.to_segment(max(pending_rep.start_frame, pending_rep_end_frame)))

    if not segments:
        raise RuntimeError(f"{label}에서 비교할 풀업 rep를 찾지 못했습니다.")

    return VideoAnalysis(
        label=label,
        input_path=input_path,
        metadata=metadata,
        segments=segments[: required_reps],
        torso_lengths=torso_lengths,
        hip_widths=hip_widths,
        shoulder_centers=shoulder_centers,
        hip_centers=hip_centers,
        deadhang_shoulder_centers=deadhang_shoulder_centers,
        deadhang_hip_widths=deadhang_hip_widths,
        deadhang_hip_centers=deadhang_hip_centers,
    )


def compute_output_layout(analysis_items: list[VideoAnalysis]) -> tuple[int, int]:
    target_hip_width = float(np.median([item.median_hip_width for item in analysis_items]))
    bounds: list[tuple[float, float, float, float]] = []

    for item in analysis_items:
        scale = target_hip_width / max(1.0, item.median_hip_width)
        if item.label == "video2":
            scale *= VIDEO2_SCALE_BIAS
        baseline_shoulder = item.baseline_shoulder_center
        frame_width, frame_height = item.metadata.output_size
        left = -scale * baseline_shoulder[0]
        top = -scale * baseline_shoulder[1]
        right = left + scale * frame_width
        bottom = top + scale * frame_height
        bounds.append((left, top, right, bottom))
        item.placement = VideoPlacement(scale=scale, translate_x=left, translate_y=top)

    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)

    output_width = max(2, int(math.ceil(max_x - min_x + OUTPUT_BORDER_PADDING * 2)))
    output_height = max(2, int(math.ceil(max_y - min_y + OUTPUT_BORDER_PADDING * 2)))
    if output_width % 2 != 0:
        output_width += 1
    if output_height % 2 != 0:
        output_height += 1

    for item in analysis_items:
        assert item.placement is not None
        item.placement = VideoPlacement(
            scale=item.placement.scale,
            translate_x=item.placement.translate_x - min_x + OUTPUT_BORDER_PADDING,
            translate_y=item.placement.translate_y - min_y + OUTPUT_BORDER_PADDING,
        )

    return output_width, output_height


def build_background(frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    y_ratio = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    background = BACKGROUND_TOP * (1.0 - y_ratio) + BACKGROUND_BOTTOM * y_ratio
    background = np.repeat(background, width, axis=1)

    grid = background.copy()
    for x in range(0, width, 120):
        cv2.line(grid, (x, 0), (x, height - 1), (40, 52, 68), 1, cv2.LINE_AA)
    for y in range(0, height, 120):
        cv2.line(grid, (0, y), (width - 1, y), (40, 52, 68), 1, cv2.LINE_AA)
    return cv2.addWeighted(grid.astype(np.uint8), 0.20, background.astype(np.uint8), 0.80, 0)


def draw_pose_overlay(frame: np.ndarray, pose: PoseFrame) -> None:
    limb_thickness = max(4, int(round(pose.body_scale * 0.035)))
    joint_radius = max(7, int(round(pose.body_scale * 0.040)))

    arm_outer_start = (255, 168, 92)
    arm_outer_end = (255, 111, 157)
    arm_inner_start = (82, 232, 255)
    arm_inner_end = (96, 160, 255)
    torso_start = (255, 146, 178)
    torso_end = (126, 191, 255)

    draw_limb_gradient_line(frame, pose.left_shoulder, pose.left_elbow, arm_outer_start, arm_outer_end, thickness=limb_thickness)
    draw_limb_gradient_line(frame, pose.left_elbow, pose.left_wrist, arm_inner_start, arm_inner_end, thickness=limb_thickness)
    draw_limb_gradient_line(frame, pose.right_shoulder, pose.right_elbow, arm_outer_start, arm_outer_end, thickness=limb_thickness)
    draw_limb_gradient_line(frame, pose.right_elbow, pose.right_wrist, arm_inner_start, arm_inner_end, thickness=limb_thickness)
    draw_limb_gradient_line(frame, pose.left_shoulder, pose.right_shoulder, torso_start, torso_end, thickness=limb_thickness)

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
        point = tuple(np.round(getattr(pose, joint_name)[:2]).astype(int))
        draw_joint_glow(frame, point, joint_core_colors[joint_name], joint_glow_colors[joint_name], radius=joint_radius)


def draw_top_left_date_overlay(
    image: np.ndarray,
    *,
    date_text: str,
    accent_color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(1.0, min(image.shape[1], image.shape[0]) / 900.0)
    title_thickness = max(2, int(round(title_scale * 2.2)))
    title_size, title_baseline = cv2.getTextSize(date_text, font, title_scale, title_thickness)

    x = 74
    y = 92
    box_width = title_size[0] + 56
    box_height = title_size[1] + title_baseline + 34

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (10, 16, 24), -1)
    cv2.rectangle(overlay, (x, y), (x + 8, y + box_height), accent_color, -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)

    text_x = x + 24
    text_y = y + 18 + title_size[1]
    cv2.putText(image, date_text, (text_x + 3, text_y + 3), font, title_scale, (8, 12, 18), title_thickness + 2, cv2.LINE_AA)
    cv2.putText(image, date_text, (text_x, text_y), font, title_scale, LABEL_TEXT_COLOR, title_thickness, cv2.LINE_AA)


def render_aligned_frame(
    frame: np.ndarray,
    *,
    analysis: VideoAnalysis,
    placement: VideoPlacement,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    pose: PoseFrame | None,
    show_pose_overlay: bool,
    x_offset: float = 0.0,
) -> np.ndarray:
    source = frame.copy()
    if show_pose_overlay and pose is not None:
        draw_pose_overlay(source, pose)

    output_width, output_height = canvas_size
    transform = np.asarray(
        [
            [placement.scale, 0.0, placement.translate_x + x_offset],
            [0.0, placement.scale, placement.translate_y],
        ],
        dtype=np.float32,
    )
    transformed_frame = cv2.warpAffine(
        source,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_mask = np.full(source.shape[:2], SOURCE_MASK_FILL, dtype=np.uint8)
    transformed_mask = cv2.warpAffine(
        source_mask,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    result = background.copy()
    result[transformed_mask > 0] = transformed_frame[transformed_mask > 0]
    return result


def decorate_frame(
    frame: np.ndarray,
    *,
    date_text: str,
    accent_color: tuple[int, int, int],
) -> np.ndarray:
    decorated = frame.copy()
    draw_top_left_date_overlay(decorated, date_text=date_text, accent_color=accent_color)
    return decorated


def sample_write_count(state: dict[str, float], *, source_fps: float, output_fps: float) -> int:
    if source_fps <= 0:
        return 1
    state["carry"] += output_fps / source_fps
    write_count = int(state["carry"])
    state["carry"] -= write_count
    return write_count


def open_video_writer(output_path: Path, *, fps: float, frame_size: tuple[int, int]):
    prefer_ffmpeg = os.environ.get("PULLUP_COMPARE_WRITER", "").strip().lower() == "ffmpeg"
    if prefer_ffmpeg:
        try:
            return FFmpegVideoWriter(output_path, fps=fps, frame_size=frame_size)
        except Exception as exc:
            print(f"[Warn] ffmpeg writer unavailable, falling back to OpenCV writer: {exc}")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"결과 영상을 만들지 못했습니다: {output_path}")
    return writer


def read_selected_frames(
    input_path: Path,
    metadata,
    frame_indices: set[int],
    *,
    analysis: VideoAnalysis,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    pose_by_frame: dict[int, PoseFrame],
    show_pose_overlay: bool,
    x_offset: float = 0.0,
) -> dict[int, np.ndarray]:
    if not frame_indices:
        return {}

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열지 못했습니다: {input_path}")

    try:
        results: dict[int, np.ndarray] = {}
        placement = analysis.placement or VideoPlacement(1.0, 0.0, 0.0)
        for frame_index in sorted(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            has_frame, frame = capture.read()
            if not has_frame:
                continue
            if metadata.rotate_for_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aligned = render_aligned_frame(
                frame,
                analysis=analysis,
                placement=placement,
                canvas_size=canvas_size,
                background=background,
                pose=pose_by_frame.get(frame_index),
                show_pose_overlay=show_pose_overlay,
                x_offset=x_offset,
            )
            results[frame_index] = aligned
        return results
    finally:
        capture.release()


def read_decorated_frame(
    *,
    analysis: VideoAnalysis,
    frame_index: int,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    pose_by_frame: dict[int, PoseFrame],
    show_pose_overlay: bool,
    accent_color: tuple[int, int, int],
    x_offset: float = 0.0,
) -> np.ndarray | None:
    aligned_frames = read_selected_frames(
        analysis.input_path,
        analysis.metadata,
        {frame_index},
        analysis=analysis,
        canvas_size=canvas_size,
        background=background,
        pose_by_frame=pose_by_frame,
        show_pose_overlay=show_pose_overlay,
        x_offset=x_offset,
    )
    aligned = aligned_frames.get(frame_index)
    if aligned is None:
        return None
    return decorate_frame(
        aligned,
        date_text=analysis.session_date_label,
        accent_color=accent_color,
    )


def write_frame_interval(
    writer,
    *,
    analysis: VideoAnalysis,
    segment: RepSegment,
    start_frame: int,
    end_frame: int,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    accent_color: tuple[int, int, int],
    output_fps: float,
    x_offset: float = 0.0,
) -> np.ndarray | None:
    if end_frame < start_frame:
        return None

    reader = SequentialFrameReader(analysis.input_path, analysis.metadata)
    sampling_state = {"carry": 0.0}
    wrote_any_frame = False
    last_rendered_frame: np.ndarray | None = None
    placement = analysis.placement or VideoPlacement(1.0, 0.0, 0.0)

    try:
        for frame_index, frame in reader.iter_range(start_frame, end_frame):
            pose = segment.pose_by_frame.get(frame_index)
            aligned = render_aligned_frame(
                frame,
                analysis=analysis,
                placement=placement,
                canvas_size=canvas_size,
                background=background,
                pose=pose,
                show_pose_overlay=segment.show_pose_overlay,
                x_offset=x_offset,
            )
            decorated = decorate_frame(
                aligned,
                date_text=analysis.session_date_label,
                accent_color=accent_color,
            )
            write_count = sample_write_count(sampling_state, source_fps=float(analysis.metadata.fps), output_fps=output_fps)
            if write_count <= 0:
                last_rendered_frame = decorated
                continue
            for _ in range(write_count):
                writer.write(decorated)
                wrote_any_frame = True
            last_rendered_frame = decorated
    finally:
        reader.close()

    if not wrote_any_frame and last_rendered_frame is not None:
        writer.write(last_rendered_frame)

    return last_rendered_frame


def write_hold(writer, frame: np.ndarray, *, fps: float, seconds: float) -> None:
    hold_frame_count = max(1, int(round(max(0.0, seconds) * fps)))
    for _ in range(hold_frame_count):
        writer.write(frame)


def write_crossfade(writer, first_frame: np.ndarray, second_frame: np.ndarray, *, fps: float, seconds: float) -> None:
    if seconds <= 0.0:
        writer.write(second_frame)
        return
    steps = max(2, int(round(seconds * fps)))
    for step in range(1, steps + 1):
        alpha = step / steps
        blended = cv2.addWeighted(second_frame, alpha, first_frame, 1.0 - alpha, 0.0)
        writer.write(blended)


def write_segment(
    writer,
    reader: SequentialFrameReader,
    *,
    analysis: VideoAnalysis,
    segment: RepSegment,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    accent_color: tuple[int, int, int],
    output_fps: float,
    capture_frames: set[int] | None = None,
    x_offset: float = 0.0,
) -> dict[int, np.ndarray]:
    captured_frames: dict[int, np.ndarray] = {}
    sampling_state = {"carry": 0.0}
    wrote_any_frame = False
    last_rendered_frame: np.ndarray | None = None
    placement = analysis.placement or VideoPlacement(1.0, 0.0, 0.0)

    for frame_index, frame in reader.iter_range(segment.start_frame, segment.end_frame):
        pose = segment.pose_by_frame.get(frame_index)
        aligned = render_aligned_frame(
            frame,
            analysis=analysis,
            placement=placement,
            canvas_size=canvas_size,
            background=background,
            pose=pose,
            show_pose_overlay=segment.show_pose_overlay,
            x_offset=x_offset,
        )
        decorated = decorate_frame(
            aligned,
            date_text=analysis.session_date_label,
            accent_color=accent_color,
        )
        if capture_frames and frame_index in capture_frames:
            captured_frames[frame_index] = aligned.copy()

        write_count = sample_write_count(sampling_state, source_fps=float(analysis.metadata.fps), output_fps=output_fps)
        if write_count <= 0:
            last_rendered_frame = decorated
            continue
        for _ in range(write_count):
            writer.write(decorated)
            wrote_any_frame = True
        last_rendered_frame = decorated

    if not wrote_any_frame and last_rendered_frame is not None:
        writer.write(last_rendered_frame)

    return captured_frames


def render_dual_overlay_segment(
    writer,
    *,
    analysis1: VideoAnalysis,
    segment1: RepSegment,
    analysis2: VideoAnalysis,
    segment2: RepSegment,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    output_fps: float,
    start_alpha: float,
    end_alpha: float,
) -> None:
    capture1 = cv2.VideoCapture(str(analysis1.input_path))
    capture2 = cv2.VideoCapture(str(analysis2.input_path))
    if not capture1.isOpened() or not capture2.isOpened():
        if capture1.isOpened():
            capture1.release()
        if capture2.isOpened():
            capture2.release()
        raise RuntimeError("동시 비교용 영상을 열지 못했습니다.")

    placement1 = analysis1.placement or VideoPlacement(1.0, 0.0, 0.0)
    placement2 = analysis2.placement or VideoPlacement(1.0, 0.0, 0.0)
    start_pose1 = first_pose_in_segment(segment1)
    start_pose2 = first_pose_in_segment(segment2)
    x_offset1 = 0.0
    x_offset2 = compute_x_alignment_offset(
        reference_analysis=analysis1,
        reference_pose=start_pose1,
        reference_x_offset=x_offset1,
        target_analysis=analysis2,
        target_pose=start_pose2,
    )
    source_span1 = max(1, segment1.end_frame - segment1.start_frame)
    source_span2 = max(1, segment2.end_frame - segment2.start_frame)
    output_frame_count = max(
        1,
        int(round(max(source_span1 / max(1.0, analysis1.metadata.fps), source_span2 / max(1.0, analysis2.metadata.fps)) * output_fps)),
    )
    date_text = f"{analysis1.session_date_label} | {analysis2.session_date_label}"

    try:
        for output_index in range(output_frame_count):
            progress_ratio = 0.0 if output_frame_count <= 1 else output_index / (output_frame_count - 1)
            alpha = start_alpha + (end_alpha - start_alpha) * progress_ratio
            frame_index1 = int(round(segment1.start_frame + source_span1 * progress_ratio))
            frame_index2 = int(round(segment2.start_frame + source_span2 * progress_ratio))

            capture1.set(cv2.CAP_PROP_POS_FRAMES, frame_index1)
            capture2.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)
            has_frame1, frame1 = capture1.read()
            has_frame2, frame2 = capture2.read()
            if not has_frame1 or not has_frame2:
                continue

            if analysis1.metadata.rotate_for_portrait:
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if analysis2.metadata.rotate_for_portrait:
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            pose1 = segment1.pose_by_frame.get(frame_index1)
            pose2 = segment2.pose_by_frame.get(frame_index2)

            aligned1 = render_aligned_frame(
                frame1,
                analysis=analysis1,
                placement=placement1,
                canvas_size=canvas_size,
                background=background,
                pose=pose1,
                show_pose_overlay=False,
                x_offset=x_offset1,
            )
            aligned2 = render_aligned_frame(
                frame2,
                analysis=analysis2,
                placement=placement2,
                canvas_size=canvas_size,
                background=background,
                pose=pose2,
                show_pose_overlay=False,
                x_offset=x_offset2,
            )
            blended = cv2.addWeighted(aligned1, max(0.0, 1.0 - alpha), aligned2, max(0.0, alpha), 0.0)
            writer.write(
                decorate_frame(
                    blended,
                    date_text=date_text,
                    accent_color=(188, 198, 212),
                )
            )
    finally:
        capture1.release()
        capture2.release()


def build_angle_frame(
    frame: np.ndarray,
    *,
    date_text: str,
    accent_color: tuple[int, int, int],
) -> np.ndarray:
    return decorate_frame(
        frame,
        date_text=date_text,
        accent_color=accent_color,
    )


def transformed_shoulder_y(analysis: VideoAnalysis, pose: PoseFrame) -> float:
    placement = analysis.placement or VideoPlacement(1.0, 0.0, 0.0)
    return float(placement.scale * float(pose.shoulder_center[1]) + placement.translate_y)


def transformed_shoulder_x(analysis: VideoAnalysis, pose: PoseFrame, *, x_offset: float = 0.0) -> float:
    placement = analysis.placement or VideoPlacement(1.0, 0.0, 0.0)
    return float(placement.scale * float(pose.shoulder_center[0]) + placement.translate_x + x_offset)


def segment_shoulder_range(analysis: VideoAnalysis, segment: RepSegment) -> tuple[float, float]:
    transformed_values = [
        transformed_shoulder_y(analysis, pose)
        for pose in segment.pose_by_frame.values()
    ]
    if not transformed_values:
        baseline = float(analysis.baseline_shoulder_center[1])
        return baseline, baseline
    return max(transformed_values), min(transformed_values)


def shift_frame_vertically(frame: np.ndarray, delta_y: float) -> np.ndarray:
    if abs(delta_y) < 0.5:
        return frame
    transform = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, float(delta_y)]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def shift_frame_horizontally(frame: np.ndarray, delta_x: float) -> np.ndarray:
    if abs(delta_x) < 0.5:
        return frame
    transform = np.asarray([[1.0, 0.0, float(delta_x)], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def first_pose_in_segment(segment: RepSegment) -> PoseFrame | None:
    if not segment.pose_by_frame:
        return None
    return segment.pose_by_frame[min(segment.pose_by_frame)]


def compute_x_alignment_offset(
    *,
    reference_analysis: VideoAnalysis,
    reference_pose: PoseFrame | None,
    reference_x_offset: float,
    target_analysis: VideoAnalysis,
    target_pose: PoseFrame | None,
) -> float:
    if reference_pose is None or target_pose is None:
        return 0.0
    reference_x = transformed_shoulder_x(reference_analysis, reference_pose, x_offset=reference_x_offset)
    target_base_x = transformed_shoulder_x(target_analysis, target_pose, x_offset=0.0)
    return reference_x - target_base_x


def height_snapshot_frame_indices(segment: RepSegment) -> list[int]:
    if not segment.pose_by_frame:
        return []

    peak_frame_index = segment.peak_shoulder_frame_index
    if peak_frame_index is None:
        peak_frame_index = max(segment.pose_by_frame)

    ascent_items = [
        (frame_index, pose)
        for frame_index, pose in sorted(segment.pose_by_frame.items())
        if frame_index <= peak_frame_index
    ]
    if not ascent_items:
        ascent_items = sorted(segment.pose_by_frame.items())

    shoulder_values = [float(pose.shoulder_y) for _, pose in ascent_items]
    lowest_y = max(shoulder_values)
    highest_y = min(shoulder_values)
    target_levels = (
        lowest_y,
        lowest_y + (highest_y - lowest_y) / 2.0,
        highest_y,
    )

    selected: list[int] = []
    seen: set[int] = set()
    for target_level in target_levels:
        best_frame_index: int | None = None
        best_error = math.inf
        for frame_index, pose in ascent_items:
            error = abs(float(pose.shoulder_y) - target_level)
            if error < best_error:
                best_error = error
                best_frame_index = frame_index
        if best_frame_index is not None and best_frame_index not in seen:
            selected.append(best_frame_index)
            seen.add(best_frame_index)
    return selected


def write_angle_comparisons(
    writer,
    *,
    segment1: RepSegment,
    segment2: RepSegment,
    stills1: dict[int, np.ndarray],
    stills2: dict[int, np.ndarray],
    analysis1: VideoAnalysis,
    analysis2: VideoAnalysis,
    canvas_size: tuple[int, int],
    background: np.ndarray,
    config: CompareConfig,
) -> None:
    del stills1, stills2

    snapshot_indices1 = height_snapshot_frame_indices(segment1)
    snapshot_indices2 = height_snapshot_frame_indices(segment2)
    point_count = min(len(snapshot_indices1), len(snapshot_indices2))
    if point_count <= 0:
        return

    snapshot_indices1 = snapshot_indices1[:point_count]
    snapshot_indices2 = snapshot_indices2[:point_count]

    if point_count == 1:
        pose1 = segment1.pose_by_frame.get(snapshot_indices1[0])
        pose2 = segment2.pose_by_frame.get(snapshot_indices2[0])
        x_offset1 = 0.0
        x_offset2 = compute_x_alignment_offset(
            reference_analysis=analysis1,
            reference_pose=pose1,
            reference_x_offset=x_offset1,
            target_analysis=analysis2,
            target_pose=pose2,
        )
        frame1 = read_decorated_frame(
            analysis=analysis1,
            frame_index=snapshot_indices1[0],
            canvas_size=canvas_size,
            background=background,
            pose_by_frame=segment1.pose_by_frame,
            show_pose_overlay=segment1.show_pose_overlay,
            accent_color=ANALYSIS_COLORS["video1"],
            x_offset=x_offset1,
        )
        frame2 = read_decorated_frame(
            analysis=analysis2,
            frame_index=snapshot_indices2[0],
            canvas_size=canvas_size,
            background=background,
            pose_by_frame=segment2.pose_by_frame,
            show_pose_overlay=segment2.show_pose_overlay,
            accent_color=ANALYSIS_COLORS["video2"],
            x_offset=x_offset2,
        )
        if frame1 is not None:
            write_hold(writer, frame1, fps=config.output_fps, seconds=config.hold_seconds)
        if frame1 is not None and frame2 is not None:
            write_crossfade(writer, frame1, frame2, fps=config.output_fps, seconds=SNAPSHOT_SAME_POINT_FADE_SECONDS)
            write_hold(writer, frame2, fps=config.output_fps, seconds=config.hold_seconds)
        return

    current_source = 1
    current_analysis = analysis1
    current_segment = segment1
    current_boundary_list = snapshot_indices1
    current_x_offset = 0.0
    current_last_frame = write_frame_interval(
        writer,
        analysis=current_analysis,
        segment=current_segment,
        start_frame=segment1.start_frame,
        end_frame=snapshot_indices1[1],
        canvas_size=canvas_size,
        background=background,
        accent_color=ANALYSIS_COLORS["video1"],
        output_fps=config.output_fps,
        x_offset=current_x_offset,
    )

    for boundary_index in range(1, point_count):
        if current_source == 1:
            next_analysis = analysis2
            next_segment = segment2
            next_frame_index = snapshot_indices2[boundary_index]
            next_accent = ANALYSIS_COLORS["video2"]
            next_source = 2
            next_boundary_list = snapshot_indices2
        else:
            next_analysis = analysis1
            next_segment = segment1
            next_frame_index = snapshot_indices1[boundary_index]
            next_accent = ANALYSIS_COLORS["video1"]
            next_source = 1
            next_boundary_list = snapshot_indices1

        current_pose = current_segment.pose_by_frame.get(current_boundary_list[boundary_index])
        next_pose = next_segment.pose_by_frame.get(next_frame_index)
        next_x_offset = compute_x_alignment_offset(
            reference_analysis=current_analysis,
            reference_pose=current_pose,
            reference_x_offset=current_x_offset,
            target_analysis=next_analysis,
            target_pose=next_pose,
        )
        transition_frame = read_decorated_frame(
            analysis=next_analysis,
            frame_index=next_frame_index,
            canvas_size=canvas_size,
            background=background,
            pose_by_frame=next_segment.pose_by_frame,
            show_pose_overlay=next_segment.show_pose_overlay,
            accent_color=next_accent,
            x_offset=next_x_offset,
        )
        if current_last_frame is not None and transition_frame is not None:
            write_crossfade(
                writer,
                current_last_frame,
                transition_frame,
                fps=config.output_fps,
                seconds=SNAPSHOT_SAME_POINT_FADE_SECONDS,
            )
        elif transition_frame is not None:
            writer.write(transition_frame)

        if boundary_index >= point_count - 1:
            if transition_frame is not None:
                write_hold(writer, transition_frame, fps=config.output_fps, seconds=config.hold_seconds)
            break

        next_start_frame = min(next_frame_index + 1, next_boundary_list[boundary_index + 1])
        current_last_frame = write_frame_interval(
            writer,
            analysis=next_analysis,
            segment=next_segment,
            start_frame=next_start_frame,
            end_frame=next_boundary_list[boundary_index + 1],
            canvas_size=canvas_size,
            background=background,
            accent_color=next_accent,
            output_fps=config.output_fps,
            x_offset=next_x_offset,
        )
        if current_last_frame is None:
            current_last_frame = transition_frame
        current_source = next_source
        current_analysis = next_analysis
        current_segment = next_segment
        current_boundary_list = next_boundary_list
        current_x_offset = next_x_offset


def build_capture_frames(segment: RepSegment) -> set[int]:
    return set(height_snapshot_frame_indices(segment))


def render_comparison_video(
    analysis1: VideoAnalysis,
    analysis2: VideoAnalysis,
    *,
    config: CompareConfig,
) -> Path:
    if analysis1.available_reps < 4 or analysis2.available_reps < 6:
        raise RuntimeError("Need at least 4 reps in video1 and 6 reps in video2.")
    output_width, output_height = compute_output_layout([analysis1, analysis2])
    canvas_size = (output_width, output_height)
    background = build_background(canvas_size)

    ensure_directory(config.results_dir)
    writer = open_video_writer(config.output_path, fps=config.output_fps, frame_size=canvas_size)
    reader1 = SequentialFrameReader(analysis1.input_path, analysis1.metadata)
    reader2 = SequentialFrameReader(analysis2.input_path, analysis2.metadata)
    render_steps = 7
    progress = tqdm(total=render_steps, desc="Render compare", unit="step", dynamic_ncols=True)

    try:
        rep1_segment1 = analysis1.segments[0]
        rep1_capture_frames1 = build_capture_frames(rep1_segment1)
        rep1_stills1 = write_segment(
            writer,
            reader1,
            analysis=analysis1,
            segment=rep1_segment1,
            canvas_size=canvas_size,
            background=background,
            accent_color=ANALYSIS_COLORS["video1"],
            output_fps=config.output_fps,
            capture_frames=rep1_capture_frames1,
        )
        if len(rep1_stills1) < len(rep1_capture_frames1):
            missing1 = rep1_capture_frames1 - set(rep1_stills1)
            rep1_stills1.update(
                read_selected_frames(
                    analysis1.input_path,
                    analysis1.metadata,
                    missing1,
                    analysis=analysis1,
                    canvas_size=canvas_size,
                    background=background,
                    pose_by_frame=rep1_segment1.pose_by_frame,
                    show_pose_overlay=rep1_segment1.show_pose_overlay,
                )
            )
        progress.update(1)

        rep1_segment2 = analysis2.segments[0]
        rep1_capture_frames2 = build_capture_frames(rep1_segment2)
        rep1_stills2 = write_segment(
            writer,
            reader2,
            analysis=analysis2,
            segment=rep1_segment2,
            canvas_size=canvas_size,
            background=background,
            accent_color=ANALYSIS_COLORS["video2"],
            output_fps=config.output_fps,
            capture_frames=rep1_capture_frames2,
        )
        if len(rep1_stills2) < len(rep1_capture_frames2):
            missing2 = rep1_capture_frames2 - set(rep1_stills2)
            rep1_stills2.update(
                read_selected_frames(
                    analysis2.input_path,
                    analysis2.metadata,
                    missing2,
                    analysis=analysis2,
                    canvas_size=canvas_size,
                    background=background,
                    pose_by_frame=rep1_segment2.pose_by_frame,
                    show_pose_overlay=rep1_segment2.show_pose_overlay,
                )
            )
        progress.update(1)

        write_angle_comparisons(
            writer,
            segment1=rep1_segment1,
            segment2=rep1_segment2,
            stills1=rep1_stills1,
            stills2=rep1_stills2,
            analysis1=analysis1,
            analysis2=analysis2,
            canvas_size=canvas_size,
            background=background,
            config=config,
        )
        progress.update(1)

        rep2_segment1 = analysis1.segments[1]
        rep2_segment2 = analysis2.segments[1]
        rep2_stills1 = read_selected_frames(
            analysis1.input_path,
            analysis1.metadata,
            build_capture_frames(rep2_segment1),
            analysis=analysis1,
            canvas_size=canvas_size,
            background=background,
            pose_by_frame=rep2_segment1.pose_by_frame,
            show_pose_overlay=rep2_segment1.show_pose_overlay,
        )
        rep2_stills2 = read_selected_frames(
            analysis2.input_path,
            analysis2.metadata,
            build_capture_frames(rep2_segment2),
            analysis=analysis2,
            canvas_size=canvas_size,
            background=background,
            pose_by_frame=rep2_segment2.pose_by_frame,
            show_pose_overlay=rep2_segment2.show_pose_overlay,
        )
        write_angle_comparisons(
            writer,
            segment1=rep2_segment1,
            segment2=rep2_segment2,
            stills1=rep2_stills1,
            stills2=rep2_stills2,
            analysis1=analysis1,
            analysis2=analysis2,
            canvas_size=canvas_size,
            background=background,
            config=config,
        )
        progress.update(1)

        render_dual_overlay_segment(
            writer,
            analysis1=analysis1,
            segment1=analysis1.segments[2],
            analysis2=analysis2,
            segment2=analysis2.segments[2],
            canvas_size=canvas_size,
            background=background,
            output_fps=config.output_fps,
            start_alpha=0.0,
            end_alpha=0.5,
        )
        render_dual_overlay_segment(
            writer,
            analysis1=analysis1,
            segment1=analysis1.segments[3],
            analysis2=analysis2,
            segment2=analysis2.segments[3],
            canvas_size=canvas_size,
            background=background,
            output_fps=config.output_fps,
            start_alpha=0.5,
            end_alpha=1.0,
        )
        progress.update(1)

        write_segment(
            writer,
            reader2,
            analysis=analysis2,
            segment=analysis2.segments[4],
            canvas_size=canvas_size,
            background=background,
            accent_color=ANALYSIS_COLORS["video2"],
            output_fps=config.output_fps,
        )
        progress.update(1)

        write_segment(
            writer,
            reader2,
            analysis=analysis2,
            segment=analysis2.segments[5],
            canvas_size=canvas_size,
            background=background,
            accent_color=ANALYSIS_COLORS["video2"],
            output_fps=config.output_fps,
        )
        progress.update(1)
    finally:
        progress.close()
        reader1.close()
        reader2.close()
        writer.release()

    return config.output_path


def run_compare(config: CompareConfig) -> Path:
    validate_compare_config(config)
    ensure_directory(config.models_dir)
    ensure_directory(config.results_dir)
    ensure_directory(config.output_path.parent)

    reporter = _PrintReporter()
    model_path = resolve_model_path(config.models_dir, reporter=reporter, requested_model=config.model_name)
    print(f"[Info] Loading pose model: {model_path.name}")
    model = load_model(model_path, reporter=reporter)

    analysis1 = analyze_video(model, config.video1, label="video1", config=config)
    analysis2 = analyze_video(model, config.video2, label="video2", config=config)
    return render_comparison_video(analysis1, analysis2, config=config)


def main(argv: list[str] | None = None) -> int:
    print_banner()
    args = build_argument_parser().parse_args(argv)
    config = build_compare_config(args)

    try:
        output_path = run_compare(config)
    except Exception as exc:
        print(f"[Error] {exc}")
        return 1

    print(f"[Done] Comparison video saved: {output_path}")
    return 0
