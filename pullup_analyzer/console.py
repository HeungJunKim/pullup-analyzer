from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

from tqdm.auto import tqdm


ASCII_TITLE = r"""
 ____        _ _ _                              _
|  _ \ _   _| | | |_   _ _ __        __ _ _ __ | |_   _ _______ _ __
| |_) | | | | | | | | | | '_ \_____ / _` | '_ \| | | | |_  / _ \ '__|
|  __/| |_| | | | | |_| | |_) |_____| (_| | | | | | |_| |/ /  __/ |
|_|    \__,_|_|_|_|\__,_| .__/       \__,_|_| |_|_|\__, /___\___|_|
                        |_|                         |___/
""".strip("\n")

DIVIDER_WIDTH = 96
ANSI_RESET = "\033[0m"
TITLE_CODES = ("38;5;81", "38;5;117", "38;5;153")
DIVIDER_CODE = "38;5;75"

PHASE_LABELS = {
    "Stand": "시작 전",
    "Ready": "준비 중",
    "Deadhang": "매달린 자세",
    "Pull": "올라가는 중",
    "Down": "내려오는 중",
}

GRIP_LABELS = {
    "-": "확인 중",
    "Wide": "와이드",
    "Narrow": "내로우",
}

SCORE_LEVEL_LABELS = {
    "Beginner": "초급자",
    "Intermediate": "중급자",
    "Advanced": "고급자",
    "Master": "마스터",
    "God": "신",
}


@dataclass(frozen=True)
class AppInfo:
    name: str
    version: str
    author: str
    repository: str


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    rotate_for_portrait: bool
    output_size: tuple[int, int]


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "--:--"
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def color_enabled() -> bool:
    return sys.stdout.isatty()


def paint(text: str, *codes: str) -> str:
    if not color_enabled() or not codes:
        return text
    return "".join(f"\033[{code}m" for code in codes) + text + ANSI_RESET


def title_text(text: str, line_index: int) -> str:
    return paint(text, "1", TITLE_CODES[line_index % len(TITLE_CODES)])


def humanize_phase(state: str) -> str:
    return PHASE_LABELS.get(state, state)


def humanize_grip(grip: str) -> str:
    return GRIP_LABELS.get(grip, grip)


def humanize_score_level(label: str) -> str:
    return SCORE_LEVEL_LABELS.get(label, label)


def humanize_device(device: str) -> str:
    if device == "cpu":
        return "CPU 전용"
    if device.isdigit():
        return f"GPU {device}"
    return device


def rotation_message(rotate_for_portrait: bool) -> str:
    return "세로 화면에 맞춰 자동 회전" if rotate_for_portrait else "원본 방향 유지"


def analysis_status(metrics: "PullUpMetrics") -> str:
    tempo_label = "계산 중" if metrics.tempo_spm <= 0 else f"분당 {metrics.tempo_spm:.1f}회"
    score_label = f"{metrics.total_score:,d}점"
    height_label = "계산 중" if metrics.best_height_score <= 0 else f"최고 {metrics.best_height_score:>3d}점"
    return (
        f"현재 동작: {humanize_phase(metrics.state)} | "
        f"반복 횟수: {metrics.count:>2d}회 | "
        f"그립: {humanize_grip(metrics.grip)} | "
        f"속도: {tempo_label} | "
        f"높이: {height_label} | "
        f"점수: {score_label} | "
        f"레벨: {humanize_score_level(metrics.score_level)}"
    )


class VideoProgress:
    def __init__(self, label: str, total_frames: int) -> None:
        total = total_frames if total_frames > 0 else None
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            if total is not None
            else "{desc}: {n_fmt} 프레임 처리됨 [{elapsed}]"
        )
        self._bar = tqdm(
            total=total,
            desc=label,
            unit="프레임",
            dynamic_ncols=True,
            mininterval=0.15,
            leave=False,
            colour="cyan" if color_enabled() else None,
            position=0,
            bar_format=bar_format,
        )
        self._status = tqdm(
            total=0,
            desc="현재 동작을 읽는 중입니다...",
            dynamic_ncols=True,
            mininterval=0.15,
            leave=False,
            position=1,
            bar_format="{desc}",
        )
        self._last_status = ""

    def advance(self, metrics: Optional["PullUpMetrics"] = None, step: int = 1) -> None:
        self._bar.update(step)
        if metrics is None:
            return

        status = analysis_status(metrics)
        if status != self._last_status:
            self._status.set_description_str(status)
            self._last_status = status

    def close(self) -> None:
        self._status.close()
        self._bar.close()


class ConsoleReporter:
    def __init__(self, app_info: AppInfo) -> None:
        self.app_info = app_info

    def _write(self, prefix: str, message: str) -> None:
        tqdm.write(f"{paint(prefix, '1', '38;5;117')} {message}")

    def _divider(self) -> str:
        return paint("=" * DIVIDER_WIDTH, "1", DIVIDER_CODE)

    def banner(self) -> None:
        tqdm.write(self._divider())
        for line_index, line in enumerate(ASCII_TITLE.splitlines()):
            tqdm.write(title_text(line, line_index))
        tqdm.write(
            paint(
                f"{self.app_info.name} | version {self.app_info.version} | "
                f"author {self.app_info.author} | repo {self.app_info.repository}",
                "1",
                "38;5;189",
            )
        )
        tqdm.write(self._divider())

    def info(self, message: str) -> None:
        self._write("[안내]", message)

    def warn(self, message: str) -> None:
        self._write("[주의]", message)

    def error(self, message: str) -> None:
        self._write("[오류]", message)

    def session_overview(
        self,
        *,
        project_dir: Path,
        model_name: str,
        device: str,
        video_count: int,
        results_dir: Path,
    ) -> None:
        self.info(f"작업 폴더: {project_dir}")
        self.info(f"사용 모델: {model_name} | 처리 방식: {humanize_device(device)}")
        self.info(f"분석할 영상: {video_count}개 | 결과 저장 위치: {results_dir}")

    def video_started(
        self,
        *,
        index: int,
        total: int,
        input_path: Path,
        output_path: Path,
        metadata: VideoMetadata,
    ) -> None:
        self._write(
            "[진행]",
            (
                f"영상 {index}/{total} 시작: {input_path.name} | "
                f"길이 {format_duration(metadata.duration_seconds)} | "
                f"화면 {metadata.width}x{metadata.height} -> {metadata.output_size[0]}x{metadata.output_size[1]} | "
                f"{rotation_message(metadata.rotate_for_portrait)}"
            ),
        )
        self.info(f"결과 파일: {output_path}")

    def video_finished(
        self,
        *,
        output_path: Path,
        metrics: "PullUpMetrics",
        processed_frames: int,
        audio_merged: bool,
    ) -> None:
        audio_state = "오디오 유지" if audio_merged else "영상만 저장"
        height_text = "높이 점수 계산 중" if metrics.average_height_score <= 0 else f"평균 높이 점수 {metrics.average_height_score:>3d}점"
        self._write(
            "[완료]",
            (
                f"{output_path.name} 저장 완료 | 총 {metrics.count}회 | "
                f"그립 {humanize_grip(metrics.grip)} | "
                f"속도 분당 {metrics.tempo_spm:.1f}회 | "
                f"{height_text} | "
                f"종합 점수 {metrics.total_score:,d}점 | "
                f"레벨 {humanize_score_level(metrics.score_level)} | "
                f"{audio_state}"
            ),
        )

    def batch_finished(self, *, success_count: int, total_count: int, elapsed_seconds: float) -> None:
        self._write(
            "[완료]",
            f"전체 분석 종료: {success_count}/{total_count}개 영상 | 총 소요 시간 {format_duration(elapsed_seconds)}",
        )

    def open_progress(self, *, index: int, total: int, input_path: Path, total_frames: int) -> VideoProgress:
        label = f"영상 {index}/{total} | {input_path.stem}"
        return VideoProgress(label=label, total_frames=total_frames)
