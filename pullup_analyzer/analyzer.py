from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable
import urllib.request

try:
    import av
except ImportError:
    av = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
ULTRALYTICS_CONFIG_DIR = PROJECT_DIR / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from .console import AppInfo, ConsoleReporter, VideoMetadata


logging.getLogger("ultralytics").setLevel(logging.ERROR)

SUPPORTED_POSE_MODELS = (
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
)
REMOVED_POSE_MODELS = ("yolo26n-pose.pt",)
MODEL_DOWNLOAD_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
DEFAULT_MODEL_NAME = "yolo26m-pose.pt"
FINAL_FRAME_HOLD_SECONDS = 3.0
OUTPUT_VIDEO_CODEC = "libx264"
OUTPUT_VIDEO_PRESET = "slow"
OUTPUT_VIDEO_CRF = "14"
OUTPUT_VIDEO_PIXEL_FORMAT = "yuv420p"
APP_INFO = AppInfo(
    name="Pull-Up Analyzer",
    version="0.2.0",
    author="Heungjun Kim",
    repository="pullup-analyzer",
)


@dataclass(frozen=True)
class DirectoryLayout:
    project_dir: Path
    models_dir: Path
    videos_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class InferenceSettings:
    conf: float
    iou: float
    device: str = "cpu"
    classes: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class RuntimeConfig:
    directories: DirectoryLayout
    inference: InferenceSettings
    requested_model: str | None = None


@dataclass(frozen=True)
class VideoJob:
    index: int
    total: int
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class AnalysisRunResult:
    success_count: int
    total_count: int
    cancelled: bool = False


class AnalysisCancelled(RuntimeError):
    pass


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_inference_device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0", torch.cuda.get_device_name(0)
        return "cpu", "CUDA unavailable in PyTorch"
    except Exception as exc:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and "GPU 0:" in result.stdout:
                first_line = result.stdout.strip().splitlines()[0]
                return "0", f"nvidia-smi detected {first_line}"
        except Exception:
            pass
        return "cpu", f"GPU check fallback to CPU ({exc})"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze pull-up videos with YOLO pose estimation.")
    parser.add_argument("--models-dir", default=str(PROJECT_DIR / "models"))
    parser.add_argument("--videos-dir", default=str(PROJECT_DIR / "videos"))
    parser.add_argument("--results-dir", default=str(PROJECT_DIR / "results"))
    parser.add_argument("--model-name", default=os.environ.get("PULLUP_MODEL_NAME"))
    parser.add_argument("--conf", type=float, default=float(os.environ.get("PULLUP_CONF", "0.55")))
    parser.add_argument("--iou", type=float, default=float(os.environ.get("PULLUP_IOU", "0.50")))
    return parser


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    device, _ = resolve_inference_device()
    directories = DirectoryLayout(
        project_dir=PROJECT_DIR,
        models_dir=Path(args.models_dir).resolve(),
        videos_dir=Path(args.videos_dir).resolve(),
        results_dir=Path(args.results_dir).resolve(),
    )
    inference = InferenceSettings(
        conf=args.conf,
        iou=args.iou,
        device=device,
    )
    return RuntimeConfig(directories=directories, inference=inference, requested_model=args.model_name)


def download_file(url: str, destination_path: Path, reporter: ConsoleReporter) -> None:
    ensure_directory(destination_path.parent)
    reporter.info(f"모델 파일을 다운로드하고 있습니다: {destination_path.name}")
    urllib.request.urlretrieve(url, destination_path)
    reporter.info(f"모델 다운로드가 끝났습니다: {destination_path}")


def resolve_model_path(models_dir: Path, reporter: ConsoleReporter, requested_model: str | None = None) -> Path:
    ensure_directory(models_dir)

    if requested_model:
        requested_path = Path(requested_model)
        if requested_path.exists():
            resolved = requested_path.resolve()
            if resolved.name in REMOVED_POSE_MODELS:
                removed = ", ".join(REMOVED_POSE_MODELS)
                supported = ", ".join(SUPPORTED_POSE_MODELS)
                raise ValueError(f"unsupported model '{resolved.name}'. Removed: {removed}. Supported models: {supported}")
            reporter.info(f"직접 지정한 모델을 사용합니다: {resolved}")
            return resolved
        candidate_name = requested_path.name
    else:
        candidate_name = DEFAULT_MODEL_NAME

    if candidate_name in REMOVED_POSE_MODELS:
        removed = ", ".join(REMOVED_POSE_MODELS)
        supported = ", ".join(SUPPORTED_POSE_MODELS)
        raise ValueError(f"unsupported model '{candidate_name}'. Removed: {removed}. Supported models: {supported}")

    if candidate_name not in SUPPORTED_POSE_MODELS:
        supported = ", ".join(SUPPORTED_POSE_MODELS)
        raise ValueError(f"unsupported model '{candidate_name}'. Supported models: {supported}")

    candidate_path = models_dir / candidate_name
    if candidate_path.exists():
        reporter.info(f"모델 파일을 찾았습니다: {candidate_path}")
        return candidate_path

    model_url = f"{MODEL_DOWNLOAD_BASE_URL}/{candidate_name}"
    download_file(model_url, candidate_path, reporter)
    return candidate_path


def discover_input_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        return []
    return sorted(path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")


def build_output_path(results_dir: Path, input_path: Path) -> Path:
    return results_dir / f"{input_path.stem}_result.mp4"


def build_video_jobs(input_videos: list[Path], results_dir: Path) -> list[VideoJob]:
    return [
        VideoJob(
            index=index,
            total=len(input_videos),
            input_path=input_path,
            output_path=build_output_path(results_dir, input_path),
        )
        for index, input_path in enumerate(input_videos, start=1)
    ]


def build_temp_output_path(output_path: Path, tag: str) -> Path:
    return output_path.with_name(f"{output_path.stem}.{tag}{output_path.suffix}")


def resolve_system_binary(name: str) -> str | None:
    return shutil.which(name)


class FFmpegVideoWriter:
    def __init__(self, output_path: Path, *, fps: float, frame_size: tuple[int, int]) -> None:
        ffmpeg_path = resolve_system_binary("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg를 찾지 못했습니다.")

        width, height = frame_size
        self.output_path = output_path
        self.frame_size = frame_size
        self.process = subprocess.Popen(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.06f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                OUTPUT_VIDEO_CODEC,
                "-crf",
                OUTPUT_VIDEO_CRF,
                "-preset",
                OUTPUT_VIDEO_PRESET,
                "-pix_fmt",
                OUTPUT_VIDEO_PIXEL_FORMAT,
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self.process.poll() is None and self.process.stdin is not None

    def write(self, frame) -> None:
        expected_width, expected_height = self.frame_size
        if frame.shape[1] != expected_width or frame.shape[0] != expected_height:
            raise ValueError(
                f"unexpected frame size {frame.shape[1]}x{frame.shape[0]} (expected {expected_width}x{expected_height})"
            )
        if not self.isOpened():
            raise RuntimeError("ffmpeg writer가 이미 종료되었습니다.")

        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            stderr_text = ""
            if self.process.stderr is not None:
                stderr_text = self.process.stderr.read().decode("utf-8", "ignore").strip()
            raise RuntimeError(stderr_text or f"ffmpeg writer 쓰기 실패: {exc}") from exc

    def release(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()

        stderr_text = ""
        if self.process.stderr is not None:
            stderr_text = self.process.stderr.read().decode("utf-8", "ignore").strip()
            self.process.stderr.close()

        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(stderr_text or f"ffmpeg writer 종료 실패 (code {return_code})")


def resolve_title_image_path(project_dir: Path) -> Path | None:
    candidates = (
        project_dir / "resource" / "title.png",
        project_dir / "resource" / "tilte.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def capture_video_metadata(capture) -> VideoMetadata:
    import cv2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    rotate_for_portrait = width > height
    output_size = (height, width) if rotate_for_portrait else (width, height)
    duration_seconds = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
        rotate_for_portrait=rotate_for_portrait,
        output_size=output_size,
    )


def merge_original_audio(
    video_only_path: Path,
    source_video_path: Path,
    output_video_path: Path,
    reporter: ConsoleReporter,
) -> bool:
    ffmpeg_path = resolve_system_binary("ffmpeg")
    ffprobe_path = resolve_system_binary("ffprobe")
    temp_output_path = build_temp_output_path(output_video_path, "mux")
    has_audio = False

    if ffmpeg_path is not None and ffprobe_path is not None:
        probe = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(source_video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        has_audio = bool(probe.stdout.strip())
        if not has_audio:
            os.replace(video_only_path, output_video_path)
            return False

        mux = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_only_path),
                "-i",
                str(source_video_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(temp_output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if mux.returncode == 0:
            os.replace(temp_output_path, output_video_path)
            video_only_path.unlink(missing_ok=True)
            return True

        if temp_output_path.exists():
            temp_output_path.unlink()
        reporter.warn(f"원본 오디오를 합치지 못했습니다: {mux.stderr.strip() or mux.stdout.strip()}")
        os.replace(video_only_path, output_video_path)
        return False

    reporter.warn("ffmpeg/ffprobe를 찾지 못해 원본 오디오는 붙이지 못했습니다. 영상은 정상 저장합니다.")
    os.replace(video_only_path, output_video_path)
    return False


def load_model(model_path: Path, reporter: ConsoleReporter):
    reporter.info(f"모델을 불러오고 있습니다: {model_path.name}")
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    reporter.info("모델 준비가 끝났습니다. 영상 분석을 시작합니다.")
    return model


def process_video(
    model,
    job: VideoJob,
    config: RuntimeConfig,
    reporter: ConsoleReporter,
    *,
    frame_callback: Callable | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> bool:
    import cv2

    from .rendering import format_video_session_label, prepare_title_banner, render_pose_overlay
    from .state import PullUpState

    temp_output_path = build_temp_output_path(job.output_path, "video_only")
    capture = cv2.VideoCapture(str(job.input_path))
    writer = None
    progress = None
    release_error = None
    cancelled = False

    if not capture.isOpened():
        reporter.error(f"입력 영상을 열지 못했습니다: {job.input_path}")
        return False

    try:
        metadata = capture_video_metadata(capture)
        reporter.video_started(
            index=job.index,
            total=job.total,
            input_path=job.input_path,
            output_path=job.output_path,
            metadata=metadata,
        )

        try:
            writer = FFmpegVideoWriter(
                temp_output_path,
                fps=metadata.fps,
                frame_size=metadata.output_size,
            )
        except Exception as exc:
            reporter.warn(f"lossless ffmpeg 저장을 시작하지 못해 OpenCV 저장으로 대체합니다: {exc}")
            writer = cv2.VideoWriter(
                str(temp_output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata.fps,
                metadata.output_size,
            )
        if not writer.isOpened():
            reporter.error(f"결과 영상을 만들지 못했습니다: {temp_output_path}")
            return False

        state = PullUpState(fps=metadata.fps)
        session_label = format_video_session_label(job.input_path)
        progress = reporter.open_progress(index=job.index, total=job.total, input_path=job.input_path, total_frames=metadata.total_frames)
        title_banner = None

        title_image_path = resolve_title_image_path(config.directories.project_dir)
        if title_image_path is not None:
            title_image = cv2.imread(str(title_image_path), cv2.IMREAD_UNCHANGED)
            if title_image is None:
                reporter.warn(f"타이틀 이미지를 읽지 못해 건너뜁니다: {title_image_path}")
            else:
                frame_shape = (metadata.output_size[1], metadata.output_size[0], 3)
                title_banner = prepare_title_banner(title_image, frame_shape)

        metrics = state.metrics()
        processed_frames = 0
        inference_device = config.inference.device
        last_output_frame = None

        while True:
            if stop_callback is not None and stop_callback():
                raise AnalysisCancelled("사용자가 분석 중단을 요청했습니다.")

            has_frame, frame = capture.read()
            if not has_frame:
                break

            if metadata.rotate_for_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            try:
                results = model(
                    frame,
                    conf=config.inference.conf,
                    iou=config.inference.iou,
                    classes=list(config.inference.classes),
                    verbose=False,
                    device=inference_device,
                )
            except Exception as exc:
                if inference_device == "0":
                    reporter.warn(f"GPU 0 사용에 실패해 CPU로 전환합니다: {exc}")
                    inference_device = "cpu"
                    results = model(
                        frame,
                        conf=config.inference.conf,
                        iou=config.inference.iou,
                        classes=list(config.inference.classes),
                        verbose=False,
                        device=inference_device,
                    )
                else:
                    raise

            annotated_frame, metrics = render_pose_overlay(frame, results, state, session_label, title_banner)
            if frame_callback is not None:
                frame_callback(
                    annotated_frame,
                    metrics=metrics,
                    frame_index=processed_frames + 1,
                    job=job,
                    metadata=metadata,
                )
            writer.write(annotated_frame)
            last_output_frame = annotated_frame.copy()
            processed_frames += 1
            progress.advance(metrics)

        if last_output_frame is not None:
            if frame_callback is not None:
                frame_callback(
                    last_output_frame,
                    metrics=metrics,
                    frame_index=processed_frames,
                    job=job,
                    metadata=metadata,
                )
            hold_frame_count = max(1, int(round(metadata.fps * FINAL_FRAME_HOLD_SECONDS)))
            for _ in range(hold_frame_count):
                if stop_callback is not None and stop_callback():
                    raise AnalysisCancelled("사용자가 분석 중단을 요청했습니다.")
                writer.write(last_output_frame)

    except AnalysisCancelled as exc:
        reporter.warn(str(exc))
        cancelled = True
    except Exception as exc:
        if temp_output_path.exists():
            temp_output_path.unlink()
        reporter.error(f"{job.input_path.name} 분석 중 문제가 생겼습니다: {exc}")
        return False

    finally:
        capture.release()
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                release_error = exc
        if progress is not None:
            progress.close()

    if release_error is not None:
        reporter.error(f"결과 영상 저장을 마무리하지 못했습니다: {release_error}")
        temp_output_path.unlink(missing_ok=True)
        return False

    if cancelled:
        temp_output_path.unlink(missing_ok=True)
        return False

    audio_merged = merge_original_audio(temp_output_path, job.input_path, job.output_path, reporter)
    reporter.video_finished(
        output_path=job.output_path,
        metrics=metrics,
        processed_frames=processed_frames,
        audio_merged=audio_merged,
    )
    return True


def run_analysis(
    config: RuntimeConfig,
    reporter: ConsoleReporter,
    input_videos: list[Path],
    *,
    frame_callback: Callable | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> AnalysisRunResult:
    _, device_note = resolve_inference_device()

    ensure_directory(config.directories.models_dir)
    ensure_directory(config.directories.videos_dir)
    ensure_directory(config.directories.results_dir)

    model_path = resolve_model_path(config.directories.models_dir, reporter, config.requested_model)
    reporter.session_overview(
        project_dir=config.directories.project_dir,
        model_name=model_path.name,
        device=config.inference.device,
        video_count=len(input_videos),
        results_dir=config.directories.results_dir,
    )
    reporter.info(f"장치 확인 결과: {device_note}")
    model = load_model(model_path, reporter)

    jobs = build_video_jobs(input_videos, config.directories.results_dir)
    success_count = 0
    started_at = time.perf_counter()
    cancelled = False
    for job in jobs:
        if stop_callback is not None and stop_callback():
            cancelled = True
            reporter.warn("분석이 사용자 요청으로 중단됐습니다.")
            break
        if process_video(model, job, config, reporter, frame_callback=frame_callback, stop_callback=stop_callback):
            success_count += 1
        elif stop_callback is not None and stop_callback():
            cancelled = True
            break

    reporter.batch_finished(
        success_count=success_count,
        total_count=len(jobs),
        elapsed_seconds=time.perf_counter() - started_at,
    )
    return AnalysisRunResult(
        success_count=success_count,
        total_count=len(jobs),
        cancelled=cancelled,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = build_runtime_config(args)
    reporter = ConsoleReporter(APP_INFO)

    reporter.banner()

    input_videos = discover_input_videos(config.directories.videos_dir)
    if not input_videos:
        reporter.error(f"분석할 mp4 영상이 없습니다. 다음 폴더를 확인해 주세요: {config.directories.videos_dir}")
        return 1

    try:
        result = run_analysis(config, reporter, input_videos)
    except Exception as exc:
        reporter.error(f"분석 중 문제가 생겼습니다: {exc}")
        return 1

    return 0 if result.success_count == result.total_count and not result.cancelled else 1


if __name__ == "__main__":
    sys.exit(main())
