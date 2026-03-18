from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import textwrap
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
    "Beginner": "비기너",
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


def console_width() -> int:
    terminal_width = shutil.get_terminal_size(fallback=(DIVIDER_WIDTH, 24)).columns
    return max(72, min(DIVIDER_WIDTH, terminal_width - 2))


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


def compact_phase_label(state: str) -> str:
    return {
        "Stand": "Stand",
        "Ready": "Ready",
        "Deadhang": "DeadHang",
        "Pull": "Pull",
        "Down": "Down",
    }.get(state, state)


def compact_grip_label(grip: str) -> str:
    return {
        "-": "--",
        "Wide": "Wide",
        "Narrow": "Narrow",
    }.get(grip, grip)


def rep_grade_label(rep_score: int) -> str:
    if rep_score >= 130:
        return "EXCELLENT"
    if rep_score >= 105:
        return "GOOD"
    if rep_score > 90:
        return "NORMAL"
    if rep_score > 0:
        return "BAD"
    return "TRACKING"


def humanize_device(device: str) -> str:
    if device == "cpu":
        return "CPU 전용"
    if device.isdigit():
        return f"GPU {device}"
    return device


def rotation_message(rotate_for_portrait: bool) -> str:
    return "세로 화면에 맞춰 자동 회전" if rotate_for_portrait else "원본 방향 유지"


def analysis_status(metrics: "PullUpMetrics") -> str:
    tempo_label = "--" if metrics.tempo_spm <= 0 else f"{metrics.tempo_spm:.1f}spm"
    rep_grade = rep_grade_label(metrics.last_rep_score)
    rep_score = "--" if metrics.last_rep_score <= 0 else f"+{metrics.last_rep_score:d}"
    total_score = f"{metrics.total_score:,d}"
    columns = (
        f"{compact_phase_label(metrics.state):<8}",
        f"{metrics.count:>2d} reps",
        f"{compact_grip_label(metrics.grip):<6}",
        f"{tempo_label:>8}",
        f"{rep_grade:<10}",
        f"{rep_score:>5}",
        f"{total_score:>7}",
        f"{metrics.score_level:<12}",
    )
    return (
        f"{columns[0]} | {columns[1]} | {columns[2]} | {columns[3]} | "
        f"{columns[4]} | {columns[5]} | {columns[6]} | {columns[7]}"
    )


class VideoProgress:
    def __init__(self, label: str, total_frames: int) -> None:
        total = total_frames if total_frames > 0 else None
        progress_width = console_width()
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            if total is not None
            else "{desc}: {n_fmt} 프레임 처리됨 [{elapsed}]"
        )
        self._bar = tqdm(
            total=total,
            desc=label,
            unit="프레임",
            dynamic_ncols=False,
            ncols=progress_width,
            mininterval=0.15,
            leave=False,
            colour="cyan" if color_enabled() else None,
            position=0,
            bar_format=bar_format,
        )
        self._status = tqdm(
            total=0,
            desc="현재 동작을 읽는 중입니다...",
            dynamic_ncols=False,
            ncols=progress_width,
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

    def _wrap_text(self, text: str, width: int) -> list[str]:
        return textwrap.wrap(
            text,
            width=max(20, width),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]

    def _write_lines(self, prefix: str, lines: list[str]) -> None:
        if not lines:
            return
        prefix_text = paint(prefix, "1", "38;5;117")
        indent = " " * (len(prefix) + 1)
        tqdm.write(f"{prefix_text} {lines[0]}")
        for line in lines[1:]:
            tqdm.write(f"{indent}{line}")

    def _write(self, prefix: str, message: str) -> None:
        content_width = DIVIDER_WIDTH - len(prefix) - 1
        self._write_lines(prefix, self._wrap_text(message, content_width))

    def _write_table(self, prefix: str, title: str, rows: list[tuple[str, str]]) -> None:
        content_width = DIVIDER_WIDTH - len(prefix) - 1
        label_width = max((len(label) for label, _ in rows), default=0)
        lines = [title]
        for label, value in rows:
            key = f"- {label:<{label_width}} : "
            wrapped = self._wrap_text(value, content_width - len(key))
            lines.append(f"{key}{wrapped[0]}")
            continuation = " " * len(key)
            for extra_line in wrapped[1:]:
                lines.append(f"{continuation}{extra_line}")
        self._write_lines(prefix, lines)

    def _divider(self) -> str:
        return paint("=" * DIVIDER_WIDTH, "1", DIVIDER_CODE)

    def banner(self) -> None:
        tqdm.write(self._divider())
        title_lines = ASCII_TITLE.splitlines()
        block_width = max(len(line) for line in title_lines)
        block_indent = max(0, (DIVIDER_WIDTH - block_width) // 2)
        prefix = " " * block_indent
        for line_index, line in enumerate(title_lines):
            tqdm.write(title_text(f"{prefix}{line}", line_index))
        subtitle = (
            f"{self.app_info.name} | version {self.app_info.version} | "
            f"author {self.app_info.author} | repo {self.app_info.repository}"
        )
        tqdm.write(
            paint(
                subtitle.center(DIVIDER_WIDTH),
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
        self._write_table(
            "[안내]",
            "실행 정보",
            [
                ("작업 폴더", str(project_dir)),
                ("사용 모델", model_name),
                ("처리 방식", humanize_device(device)),
                ("분석할 영상", f"{video_count}개"),
                ("결과 위치", str(results_dir)),
            ],
        )

    def video_started(
        self,
        *,
        index: int,
        total: int,
        input_path: Path,
        output_path: Path,
        metadata: VideoMetadata,
    ) -> None:
        self._write_table(
            "[진행]",
            f"영상 {index}/{total} 시작",
            [
                ("파일", input_path.name),
                ("길이", format_duration(metadata.duration_seconds)),
                ("화면", f"{metadata.width}x{metadata.height} -> {metadata.output_size[0]}x{metadata.output_size[1]}"),
                ("방향", rotation_message(metadata.rotate_for_portrait)),
                ("결과 파일", str(output_path)),
            ],
        )

    def video_finished(
        self,
        *,
        output_path: Path,
        metrics: "PullUpMetrics",
        processed_frames: int,
        audio_merged: bool,
    ) -> None:
        audio_state = "오디오 유지" if audio_merged else "영상만 저장"
        rep_grade = rep_grade_label(metrics.last_rep_score)
        rep_score = "판정 없음" if metrics.last_rep_score <= 0 else f"+{metrics.last_rep_score:d}점"
        self._write_table(
            "[완료]",
            f"{output_path.name} 저장 완료",
            [
                ("반복 횟수", f"{metrics.count}회"),
                ("그립", humanize_grip(metrics.grip)),
                ("속도", f"{metrics.tempo_spm:.1f}spm"),
                ("마지막 평가", rep_grade),
                ("마지막 점수", rep_score),
                ("누적 점수", f"{metrics.total_score:,d}점"),
                ("레벨", humanize_score_level(metrics.score_level)),
                ("오디오", audio_state),
            ],
        )

    def batch_finished(self, *, success_count: int, total_count: int, elapsed_seconds: float) -> None:
        self._write(
            "[완료]",
            f"전체 분석 종료: {success_count}/{total_count}개 영상 | 총 소요 시간 {format_duration(elapsed_seconds)}",
        )

    def open_progress(self, *, index: int, total: int, input_path: Path, total_frames: int) -> VideoProgress:
        label = f"영상 {index}/{total} | {input_path.stem}"
        return VideoProgress(label=label, total_frames=total_frames)
