from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
from PIL import Image, ImageTk

from pullup_analyzer.analyzer import (
    APP_INFO,
    PROJECT_DIR,
    DirectoryLayout,
    InferenceSettings,
    RuntimeConfig,
    ensure_directory,
    resolve_inference_device,
    run_analysis,
)
from pullup_analyzer.console import (
    analysis_status,
    format_duration,
    humanize_device,
    humanize_grip,
    humanize_score_level,
    rep_grade_label,
    rotation_message,
)


VIDEO_FILETYPES = (("MP4 videos", "*.mp4"), ("All files", "*.*"))


class GuiProgress:
    def __init__(self, event_queue: queue.Queue, label: str, total_frames: int) -> None:
        self._event_queue = event_queue
        self._label = label
        self._total_frames = total_frames
        self._current_frame = 0

    def advance(self, metrics=None, step: int = 1) -> None:
        self._current_frame += step
        self._event_queue.put(
            {
                "type": "progress",
                "label": self._label,
                "current": self._current_frame,
                "total": self._total_frames,
                "status": analysis_status(metrics) if metrics is not None else "",
            }
        )

    def close(self) -> None:
        return


class GuiReporter:
    def __init__(self, event_queue: queue.Queue) -> None:
        self._event_queue = event_queue

    def _emit(self, event_type: str, **payload) -> None:
        self._event_queue.put({"type": event_type, **payload})

    def _emit_log(self, level: str, message: str) -> None:
        self._emit("log", level=level, text=message)

    def banner(self) -> None:
        self._emit("banner")

    def info(self, message: str) -> None:
        self._emit_log("info", message)

    def warn(self, message: str) -> None:
        self._emit_log("warn", message)

    def error(self, message: str) -> None:
        self._emit_log("error", message)

    def session_overview(
        self,
        *,
        project_dir: Path,
        model_name: str,
        device: str,
        video_count: int,
        results_dir: Path,
    ) -> None:
        self._emit(
            "session",
            project_dir=str(project_dir),
            model_name=model_name,
            device=humanize_device(device),
            video_count=video_count,
            results_dir=str(results_dir),
        )

    def video_started(self, *, index: int, total: int, input_path: Path, output_path: Path, metadata) -> None:
        self._emit(
            "video_started",
            index=index,
            total=total,
            input_path=str(input_path),
            output_path=str(output_path),
            duration=format_duration(metadata.duration_seconds),
            resolution=f"{metadata.width}x{metadata.height} -> {metadata.output_size[0]}x{metadata.output_size[1]}",
            rotation=rotation_message(metadata.rotate_for_portrait),
            total_frames=metadata.total_frames,
        )

    def video_finished(self, *, output_path: Path, metrics, processed_frames: int, audio_merged: bool) -> None:
        rep_score = "판정 없음" if metrics.last_rep_score <= 0 else f"+{metrics.last_rep_score:d}점"
        self._emit(
            "video_finished",
            output_path=str(output_path),
            processed_frames=processed_frames,
            reps=metrics.count,
            grip=humanize_grip(metrics.grip),
            tempo=f"{metrics.tempo_spm:.1f}spm" if metrics.tempo_spm > 0 else "--",
            rep_grade=rep_grade_label(metrics.last_rep_score),
            rep_score=rep_score,
            total_score=f"{metrics.total_score:,d}점",
            score_level=humanize_score_level(metrics.score_level),
            audio_state="오디오 유지" if audio_merged else "영상만 저장",
        )

    def batch_finished(self, *, success_count: int, total_count: int, elapsed_seconds: float) -> None:
        self._emit(
            "batch_finished",
            success_count=success_count,
            total_count=total_count,
            elapsed=format_duration(elapsed_seconds),
        )

    def open_progress(self, *, index: int, total: int, input_path: Path, total_frames: int) -> GuiProgress:
        return GuiProgress(
            self._event_queue,
            label=f"영상 {index}/{total} | {input_path.stem}",
            total_frames=total_frames,
        )


class PullUpAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_INFO.name} GUI")
        self.root.geometry("1460x920")
        self.root.minsize(1200, 760)

        self.selected_videos: list[Path] = []
        self.output_dir = (PROJECT_DIR / "results").resolve()
        self.event_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.worker: threading.Thread | None = None
        self.preview_image = None

        self.output_dir_var = tk.StringVar(value=str(self.output_dir))
        self.selection_var = tk.StringVar(value="선택된 영상이 없습니다.")
        self.current_video_var = tk.StringVar(value="대기 중")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="분석 준비 전")
        self.status_text_var = tk.StringVar(value="진행 상태가 여기에 표시됩니다.")

        self._build_ui()
        self._poll_queues()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        controls = ttk.Frame(self.root, padding=16)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        content = ttk.Frame(self.root, padding=(0, 16, 16, 16))
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=3)
        content.rowconfigure(2, weight=2)

        ttk.Label(
            controls,
            text=APP_INFO.name,
            font=("Segoe UI", 19, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            controls,
            text="동영상 또는 폴더를 고른 뒤 GUI에서 시각화 과정을 실시간으로 볼 수 있습니다.",
            wraplength=320,
            foreground="#5f6b7a",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        input_frame = ttk.LabelFrame(controls, text="입력 영상", padding=12)
        input_frame.grid(row=2, column=0, sticky="nsew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)

        button_row = ttk.Frame(input_frame)
        button_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column_index in range(3):
            button_row.columnconfigure(column_index, weight=1)

        self.select_videos_button = ttk.Button(button_row, text="동영상 선택", command=self._select_videos)
        self.select_videos_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.select_folder_button = ttk.Button(button_row, text="폴더 선택", command=self._select_folder)
        self.select_folder_button.grid(row=0, column=1, sticky="ew", padx=3)
        self.clear_selection_button = ttk.Button(button_row, text="선택 비우기", command=self._clear_selection)
        self.clear_selection_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.video_listbox = tk.Listbox(input_frame, height=14)
        self.video_listbox.grid(row=1, column=0, sticky="nsew")
        ttk.Label(
            input_frame,
            textvariable=self.selection_var,
            foreground="#5f6b7a",
        ).grid(row=2, column=0, sticky="w", pady=(10, 0))

        output_frame = ttk.LabelFrame(controls, text="결과 저장", padding=12)
        output_frame.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        output_frame.columnconfigure(0, weight=1)

        ttk.Entry(output_frame, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.select_output_button = ttk.Button(output_frame, text="폴더 변경", command=self._select_output_dir)
        self.select_output_button.grid(row=0, column=1, sticky="ew")

        start_frame = ttk.Frame(controls)
        start_frame.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        start_frame.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(start_frame, text="분석 시작", command=self._start_analysis)
        self.start_button.grid(row=0, column=0, sticky="ew")

        info_frame = ttk.LabelFrame(controls, text="현재 상태", padding=12)
        info_frame.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(info_frame, text="현재 영상", foreground="#5f6b7a").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.current_video_var, wraplength=320).grid(row=1, column=0, sticky="w", pady=(2, 10))
        ttk.Label(info_frame, text="프로그레스", foreground="#5f6b7a").grid(row=2, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.progress_text_var).grid(row=3, column=0, sticky="w", pady=(2, 10))
        ttk.Label(info_frame, text="분석 상태", foreground="#5f6b7a").grid(row=4, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.status_text_var, wraplength=320).grid(row=5, column=0, sticky="w", pady=(2, 0))

        preview_frame = ttk.LabelFrame(content, text="실시간 시각화", padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            preview_frame,
            text="분석이 시작되면 현재 프레임이 여기에 표시됩니다.",
            anchor="center",
            justify="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        progress_frame = ttk.Frame(content, padding=(0, 12, 0, 12))
        progress_frame.grid(row=1, column=0, sticky="ew")
        progress_frame.columnconfigure(0, weight=1)

        ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        ).grid(row=0, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(content, text="실행 로그", padding=12)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            self.select_videos_button,
            self.select_folder_button,
            self.clear_selection_button,
            self.select_output_button,
            self.start_button,
        ):
            widget.configure(state=state)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_selection_view(self) -> None:
        self.video_listbox.delete(0, "end")
        for path in self.selected_videos:
            self.video_listbox.insert("end", path.name)

        if self.selected_videos:
            self.selection_var.set(f"{len(self.selected_videos)}개 영상 선택됨")
        else:
            self.selection_var.set("선택된 영상이 없습니다.")

    def _add_videos(self, paths: list[Path]) -> None:
        existing = {path.resolve() for path in self.selected_videos}
        for path in paths:
            resolved = path.resolve()
            if resolved.suffix.lower() != ".mp4" or resolved in existing:
                continue
            self.selected_videos.append(resolved)
            existing.add(resolved)

        self.selected_videos.sort()
        self._update_selection_view()

    def _select_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="분석할 동영상 선택",
            filetypes=VIDEO_FILETYPES,
        )
        self._add_videos([Path(path) for path in paths])

    def _select_folder(self) -> None:
        folder = filedialog.askdirectory(title="mp4가 들어 있는 폴더 선택")
        if not folder:
            return
        self._add_videos(sorted(Path(folder).glob("*.mp4")))

    def _clear_selection(self) -> None:
        self.selected_videos.clear()
        self._update_selection_view()

    def _select_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="결과 저장 폴더 선택", initialdir=self.output_dir_var.get())
        if folder:
            self.output_dir_var.set(folder)

    def _start_analysis(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("분석 중", "이미 분석이 진행 중입니다.")
            return

        if not self.selected_videos:
            messagebox.showwarning("입력 필요", "분석할 mp4 동영상이나 폴더를 먼저 선택해 주세요.")
            return

        output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
        ensure_directory(output_dir)
        self.output_dir = output_dir
        self.progress_var.set(0.0)
        self.progress_text_var.set("준비 중")
        self.status_text_var.set("모델과 입력 영상을 준비하고 있습니다.")
        self.current_video_var.set("대기 중")
        self.preview_label.configure(image="", text="분석을 시작했습니다. 첫 프레임을 기다리는 중입니다.")
        self.preview_image = None
        self._set_controls_enabled(False)

        selected_paths = list(self.selected_videos)
        self.worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(selected_paths, output_dir),
            daemon=True,
        )
        self.worker.start()

    def _run_analysis_worker(self, input_videos: list[Path], output_dir: Path) -> None:
        reporter = GuiReporter(self.event_queue)
        try:
            device, device_note = resolve_inference_device()
            config = RuntimeConfig(
                directories=DirectoryLayout(
                    project_dir=PROJECT_DIR,
                    models_dir=(PROJECT_DIR / "models").resolve(),
                    videos_dir=(PROJECT_DIR / "videos").resolve(),
                    results_dir=output_dir,
                ),
                inference=InferenceSettings(
                    conf=0.55,
                    iou=0.50,
                    device=device,
                ),
            )
            reporter.info(f"GUI 분석을 시작합니다. 선택된 영상 {len(input_videos)}개")
            reporter.info(f"장치 확인 결과: {device_note}")
            success_count, total_count = run_analysis(
                config,
                reporter,
                input_videos,
                frame_callback=self._enqueue_preview_frame,
            )
            self.event_queue.put(
                {
                    "type": "done",
                    "success_count": success_count,
                    "total_count": total_count,
                }
            )
        except Exception as exc:
            self.event_queue.put({"type": "error", "text": str(exc)})
            self.event_queue.put({"type": "done", "success_count": 0, "total_count": len(input_videos)})

    def _enqueue_preview_frame(self, frame, **payload) -> None:
        event = {"frame": frame.copy(), **payload}
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        self.frame_queue.put_nowait(event)

    def _poll_queues(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

        latest_frame_event = None
        try:
            while True:
                latest_frame_event = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame_event is not None:
            self._update_preview(latest_frame_event)

        self.root.after(30, self._poll_queues)

    def _handle_event(self, event: dict) -> None:
        event_type = event["type"]
        if event_type == "log":
            level_label = {
                "info": "[안내]",
                "warn": "[주의]",
                "error": "[오류]",
            }.get(event["level"], "[로그]")
            self._append_log(f"{level_label} {event['text']}")
            return

        if event_type == "session":
            self._append_log(
                "\n".join(
                    [
                        "[안내] 실행 정보",
                        f"  작업 폴더 : {event['project_dir']}",
                        f"  사용 모델 : {event['model_name']}",
                        f"  처리 방식 : {event['device']}",
                        f"  분석할 영상 : {event['video_count']}개",
                        f"  결과 위치 : {event['results_dir']}",
                    ]
                )
            )
            return

        if event_type == "video_started":
            self.current_video_var.set(Path(event["input_path"]).name)
            self.progress_var.set(0.0)
            self.progress_text_var.set(f"{event['index']}/{event['total']} | 0 / {event['total_frames']}")
            self.status_text_var.set("첫 프레임을 분석 중입니다.")
            self._append_log(
                "\n".join(
                    [
                        f"[진행] 영상 {event['index']}/{event['total']} 시작",
                        f"  파일 : {Path(event['input_path']).name}",
                        f"  길이 : {event['duration']}",
                        f"  화면 : {event['resolution']}",
                        f"  방향 : {event['rotation']}",
                        f"  결과 파일 : {event['output_path']}",
                    ]
                )
            )
            return

        if event_type == "progress":
            total = max(1, int(event["total"])) if event["total"] else 1
            current = min(int(event["current"]), total)
            self.progress_var.set((current / total) * 100.0)
            self.progress_text_var.set(f"{event['label']} | {current} / {total}")
            if event["status"]:
                self.status_text_var.set(event["status"])
            return

        if event_type == "video_finished":
            self._append_log(
                "\n".join(
                    [
                        f"[완료] {Path(event['output_path']).name} 저장 완료",
                        f"  반복 횟수 : {event['reps']}회",
                        f"  그립 : {event['grip']}",
                        f"  속도 : {event['tempo']}",
                        f"  마지막 평가 : {event['rep_grade']}",
                        f"  마지막 점수 : {event['rep_score']}",
                        f"  누적 점수 : {event['total_score']}",
                        f"  레벨 : {event['score_level']}",
                        f"  오디오 : {event['audio_state']}",
                    ]
                )
            )
            return

        if event_type == "batch_finished":
            self._append_log(
                f"[완료] 전체 분석 종료: {event['success_count']}/{event['total_count']}개 영상 | 총 소요 시간 {event['elapsed']}"
            )
            return

        if event_type == "error":
            self._append_log(f"[오류] {event['text']}")
            self.status_text_var.set(event["text"])
            return

        if event_type == "done":
            self._set_controls_enabled(True)
            self.progress_var.set(100.0 if event["success_count"] and event["success_count"] == event["total_count"] else self.progress_var.get())
            self.progress_text_var.set(
                f"완료 | {event['success_count']} / {event['total_count']}개"
                if event["total_count"] > 0 else "완료"
            )
            if event["total_count"] > 0 and event["success_count"] == event["total_count"]:
                self.status_text_var.set("모든 영상 분석이 완료됐습니다.")
            elif event["total_count"] > 0:
                self.status_text_var.set("일부 영상 분석에 실패했습니다. 로그를 확인해 주세요.")
            return

    def _update_preview(self, event: dict) -> None:
        frame = event["frame"]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        max_width = 960
        max_height = 560
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")


def main() -> None:
    root = tk.Tk()
    app = PullUpAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
