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
    DEFAULT_MODEL_NAME,
    PROJECT_DIR,
    SUPPORTED_POSE_MODELS,
    DirectoryLayout,
    InferenceSettings,
    RuntimeConfig,
    ensure_directory,
    resolve_inference_device,
    resolve_title_image_path,
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
APP_BG = "#0f172a"
PANEL_BG = "#111827"
CARD_BG = "#162033"
TEXT_MAIN = "#e5edf7"
TEXT_SUB = "#8fa2bf"
ACCENT = "#38bdf8"
ACCENT_2 = "#7dd3fc"
SUCCESS = "#34d399"
WARN = "#fbbf24"
ERROR = "#fb7185"


class GuiProgress:
    def __init__(self, event_queue: queue.Queue, *, index: int, total_jobs: int, label: str, total_frames: int) -> None:
        self._event_queue = event_queue
        self._index = index
        self._total_jobs = total_jobs
        self._label = label
        self._total_frames = total_frames
        self._current_frame = 0

    def advance(self, metrics=None, step: int = 1) -> None:
        self._current_frame += step
        self._event_queue.put(
            {
                "type": "progress",
                "job_index": self._index,
                "job_total": self._total_jobs,
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
            index=index,
            total_jobs=total,
            label=f"영상 {index}/{total} | {input_path.stem}",
            total_frames=total_frames,
        )


class PullUpAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_INFO.name} GUI")
        self.root.configure(bg=APP_BG)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1320, max(1080, screen_width - 120))
        window_height = min(860, max(700, screen_height - 120))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(1024, 680)

        self.selected_videos: list[Path] = []
        self.output_dir = (PROJECT_DIR / "results").resolve()
        self.cancel_event = threading.Event()
        self.event_queue: queue.Queue = queue.Queue()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.worker: threading.Thread | None = None
        self.preview_image = None
        self.header_image = None

        self.output_dir_var = tk.StringVar(value=str(self.output_dir))
        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.selection_var = tk.StringVar(value="선택된 영상이 없습니다.")
        self.device_var = tk.StringVar(value="장치 확인 전")
        self.current_video_var = tk.StringVar(value="대기 중")
        self.current_progress_text_var = tk.StringVar(value="현재 영상 진행률이 여기에 표시됩니다.")
        self.overall_progress_text_var = tk.StringVar(value="전체 진행률이 여기에 표시됩니다.")
        self.status_text_var = tk.StringVar(value="분석 상태가 여기에 표시됩니다.")
        self.summary_var = tk.StringVar(value="영상 또는 폴더를 선택한 뒤 분석을 시작해 주세요.")
        self.current_progress_var = tk.DoubleVar(value=0.0)
        self.overall_progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._load_header_title()
        self._refresh_device_label()
        self._poll_queues()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("App.TLabelframe", background=PANEL_BG, foreground=TEXT_MAIN, borderwidth=1)
        style.configure("App.TLabelframe.Label", background=PANEL_BG, foreground=ACCENT_2, font=("Segoe UI", 10, "bold"))
        style.configure("HeaderTitle.TLabel", background=APP_BG, foreground=TEXT_MAIN, font=("Segoe UI", 24, "bold"))
        style.configure("HeaderSub.TLabel", background=APP_BG, foreground=TEXT_SUB, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=PANEL_BG, foreground=TEXT_MAIN, font=("Segoe UI", 10, "bold"))
        style.configure("Body.TLabel", background=PANEL_BG, foreground=TEXT_MAIN, font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=PANEL_BG, foreground=TEXT_SUB, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TEXT_SUB, font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.Horizontal.TProgressbar", background=ACCENT, troughcolor="#223047", bordercolor="#223047")
        style.configure("Success.Horizontal.TProgressbar", background=SUCCESS, troughcolor="#223047", bordercolor="#223047")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(20, 18, 20, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        text_block = ttk.Frame(header, style="App.TFrame")
        text_block.grid(row=0, column=0, sticky="w")
        ttk.Label(text_block, text=APP_INFO.name, style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            text_block,
            text="폴더 또는 동영상을 직접 선택하고, 분석 중 시각화 화면을 실시간으로 확인할 수 있습니다.",
            style="HeaderSub.TLabel",
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.header_image_label = ttk.Label(header, style="App.TFrame")
        self.header_image_label.grid(row=0, column=1, sticky="e", padx=(16, 0))

        body = ttk.Frame(self.root, style="App.TFrame", padding=(20, 0, 20, 20))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0, minsize=330)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, style="Panel.TFrame", padding=14)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(1, weight=1)

        content = ttk.Frame(body, style="Panel.TFrame", padding=14)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=7)
        content.rowconfigure(1, weight=0)
        content.rowconfigure(2, weight=2)

        input_frame = ttk.LabelFrame(controls, text="입력 영상", style="App.TLabelframe", padding=12)
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)

        button_row = ttk.Frame(input_frame, style="Panel.TFrame")
        button_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column_index in range(3):
            button_row.columnconfigure(column_index, weight=1)

        self.select_videos_button = ttk.Button(button_row, text="동영상 선택", command=self._select_videos)
        self.select_videos_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.select_folder_button = ttk.Button(button_row, text="폴더 선택", command=self._select_folder)
        self.select_folder_button.grid(row=0, column=1, sticky="ew", padx=3)
        self.clear_selection_button = ttk.Button(button_row, text="비우기", command=self._clear_selection)
        self.clear_selection_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.video_listbox = tk.Listbox(
            input_frame,
            height=10,
            bg="#0b1220",
            fg=TEXT_MAIN,
            selectbackground="#1d4ed8",
            relief="flat",
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self.video_listbox.grid(row=1, column=0, sticky="nsew")
        ttk.Label(input_frame, textvariable=self.selection_var, style="Hint.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))

        settings_frame = ttk.LabelFrame(controls, text="분석 설정", style="App.TLabelframe", padding=12)
        settings_frame.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        settings_frame.columnconfigure(0, weight=1)

        ttk.Label(settings_frame, text="포즈 모델", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.model_name_var,
            values=SUPPORTED_POSE_MODELS,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.model_combo.grid(row=1, column=0, sticky="ew", pady=(6, 10))

        ttk.Label(settings_frame, text="결과 저장 폴더", style="Title.TLabel").grid(row=2, column=0, sticky="w")
        output_row = ttk.Frame(settings_frame, style="Panel.TFrame")
        output_row.grid(row=3, column=0, sticky="ew", pady=(6, 10))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.select_output_button = ttk.Button(output_row, text="변경", command=self._select_output_dir)
        self.select_output_button.grid(row=0, column=1, sticky="ew")

        ttk.Label(settings_frame, text="추론 장치", style="Title.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(settings_frame, textvariable=self.device_var, style="Body.TLabel").grid(row=5, column=0, sticky="w", pady=(6, 14))

        action_row = ttk.Frame(settings_frame, style="Panel.TFrame")
        action_row.grid(row=6, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(action_row, text="분석 시작", command=self._start_analysis)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(action_row, text="분석 중단", command=self._stop_analysis, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        status_frame = ttk.LabelFrame(controls, text="상태", style="App.TLabelframe", padding=12)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, text="현재 영상", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.current_video_var, style="Body.TLabel", wraplength=280).grid(row=1, column=0, sticky="w", pady=(4, 10))
        ttk.Label(status_frame, text="상태 요약", style="Title.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_text_var, style="Body.TLabel", wraplength=280).grid(row=3, column=0, sticky="w", pady=(4, 10))
        ttk.Label(status_frame, text="세션 메모", style="Title.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.summary_var, style="Hint.TLabel", wraplength=280).grid(row=5, column=0, sticky="w", pady=(4, 0))

        preview_frame = ttk.LabelFrame(content, text="실시간 시각화", style="App.TLabelframe", padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_frame,
            text="분석이 시작되면 현재 프레임이 여기에 크게 표시됩니다.",
            anchor="center",
            justify="center",
            bg="#09111d",
            fg=TEXT_SUB,
            font=("Segoe UI", 12),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        progress_frame = ttk.Frame(content, style="Panel.TFrame")
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=1)

        current_card = ttk.Frame(progress_frame, style="Card.TFrame", padding=12)
        current_card.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        current_card.columnconfigure(0, weight=1)
        ttk.Label(current_card, text="현재 영상 진행률", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(current_card, textvariable=self.current_progress_text_var, style="CardValue.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 8))
        ttk.Progressbar(current_card, variable=self.current_progress_var, maximum=100.0, style="Accent.Horizontal.TProgressbar").grid(row=2, column=0, sticky="ew")

        overall_card = ttk.Frame(progress_frame, style="Card.TFrame", padding=12)
        overall_card.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        overall_card.columnconfigure(0, weight=1)
        ttk.Label(overall_card, text="전체 진행률", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(overall_card, textvariable=self.overall_progress_text_var, style="CardValue.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 8))
        ttk.Progressbar(overall_card, variable=self.overall_progress_var, maximum=100.0, style="Success.Horizontal.TProgressbar").grid(row=2, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(content, text="실행 로그", style="App.TLabelframe", padding=12)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(
            log_frame,
            wrap="word",
            height=9,
            font=("Consolas", 10),
            bg="#0b1220",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief="flat",
            highlightthickness=0,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _load_header_title(self) -> None:
        title_image_path = resolve_title_image_path(PROJECT_DIR)
        if title_image_path is None:
            return

        image = Image.open(title_image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        max_width = 280
        max_height = 88
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.header_image = ImageTk.PhotoImage(image)
        self.header_image_label.configure(image=self.header_image)

    def _refresh_device_label(self) -> None:
        device, note = resolve_inference_device()
        self.device_var.set(f"{humanize_device(device)} | {note}")

    def _set_running_state(self, running: bool) -> None:
        selector_state = "disabled" if running else "normal"
        combo_state = "disabled" if running else "readonly"
        self.select_videos_button.configure(state=selector_state)
        self.select_folder_button.configure(state=selector_state)
        self.clear_selection_button.configure(state=selector_state)
        self.select_output_button.configure(state=selector_state)
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.model_combo.configure(state=combo_state)

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
            self.summary_var.set("선택이 완료됐습니다. 모델과 저장 폴더를 확인한 뒤 분석을 시작해 주세요.")
        else:
            self.selection_var.set("선택된 영상이 없습니다.")
            self.summary_var.set("영상 또는 폴더를 선택하면 이곳에 준비 상태가 표시됩니다.")

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
        paths = filedialog.askopenfilenames(title="분석할 동영상 선택", filetypes=VIDEO_FILETYPES)
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

        self.cancel_event.clear()
        output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
        ensure_directory(output_dir)
        self.output_dir = output_dir
        self.current_progress_var.set(0.0)
        self.overall_progress_var.set(0.0)
        self.current_progress_text_var.set("모델과 입력 영상을 준비 중입니다.")
        self.overall_progress_text_var.set(f"총 {len(self.selected_videos)}개 영상 대기 중")
        self.status_text_var.set("모델을 불러오고 첫 영상을 준비하고 있습니다.")
        self.current_video_var.set("대기 중")
        self.summary_var.set("실시간 프리뷰가 곧 시작됩니다.")
        self.preview_label.configure(image="", text="분석 준비 중입니다. 잠시만 기다려 주세요.")
        self.preview_image = None
        self._set_running_state(True)

        selected_paths = list(self.selected_videos)
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        self.worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(selected_paths, output_dir, model_name),
            daemon=True,
        )
        self.worker.start()

    def _stop_analysis(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        self.cancel_event.set()
        self.status_text_var.set("현재 작업을 정리한 뒤 분석을 중단하고 있습니다.")
        self.summary_var.set("중단 요청을 보냈습니다. 저장 중인 프레임 정리가 끝나면 종료됩니다.")

    def _run_analysis_worker(self, input_videos: list[Path], output_dir: Path, model_name: str) -> None:
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
                requested_model=model_name,
            )
            reporter.info(f"GUI 분석을 시작합니다. 선택된 영상 {len(input_videos)}개")
            reporter.info(f"선택 모델: {model_name}")
            reporter.info(f"장치 확인 결과: {device_note}")
            result = run_analysis(
                config,
                reporter,
                input_videos,
                frame_callback=self._enqueue_preview_frame,
                stop_callback=self.cancel_event.is_set,
            )
            self.event_queue.put(
                {
                    "type": "done",
                    "success_count": result.success_count,
                    "total_count": result.total_count,
                    "cancelled": result.cancelled,
                }
            )
        except Exception as exc:
            self.event_queue.put({"type": "error", "text": str(exc)})
            self.event_queue.put(
                {
                    "type": "done",
                    "success_count": 0,
                    "total_count": len(input_videos),
                    "cancelled": False,
                }
            )

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

        if event_type == "banner":
            self._append_log(f"{APP_INFO.name} {APP_INFO.version}")
            return

        if event_type == "log":
            level_label = {
                "info": "[안내]",
                "warn": "[주의]",
                "error": "[오류]",
            }.get(event["level"], "[로그]")
            self._append_log(f"{level_label} {event['text']}")
            return

        if event_type == "session":
            self.summary_var.set(
                f"모델 {event['model_name']} | {event['video_count']}개 영상 | 결과 폴더 {Path(event['results_dir']).name}"
            )
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
            self.current_progress_var.set(0.0)
            self.current_progress_text_var.set(f"{event['index']}/{event['total']} | 0 / {event['total_frames']} 프레임")
            self.overall_progress_text_var.set(f"전체 {event['index']}/{event['total']} 진행 중")
            self.status_text_var.set("현재 영상을 분석하고 있습니다.")
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
            current_ratio = current / total
            self.current_progress_var.set(current_ratio * 100.0)
            self.current_progress_text_var.set(f"{event['label']} | {current} / {total} 프레임")
            overall_ratio = ((event["job_index"] - 1) + current_ratio) / max(1, event["job_total"])
            self.overall_progress_var.set(overall_ratio * 100.0)
            self.overall_progress_text_var.set(f"전체 {event['job_index']}/{event['job_total']} | {overall_ratio * 100.0:5.1f}%")
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
            self._set_running_state(False)
            if event["cancelled"]:
                self.status_text_var.set("사용자 요청으로 분석이 중단됐습니다.")
                self.summary_var.set("중단된 시점까지의 로그와 프리뷰는 그대로 유지됩니다.")
            elif event["total_count"] > 0 and event["success_count"] == event["total_count"]:
                self.current_progress_var.set(100.0)
                self.overall_progress_var.set(100.0)
                self.status_text_var.set("모든 영상 분석이 완료됐습니다.")
                self.summary_var.set("결과 영상 저장이 끝났습니다.")
            else:
                self.status_text_var.set("일부 영상 분석에 실패했습니다. 로그를 확인해 주세요.")
                self.summary_var.set("오류가 발생한 영상이 있습니다. 로그를 참고해 주세요.")
            return

    def _update_preview(self, event: dict) -> None:
        frame = event["frame"]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        available_width = max(420, self.preview_label.winfo_width() - 24)
        available_height = max(280, self.preview_label.winfo_height() - 24)
        image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)

        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")


def main() -> None:
    root = tk.Tk()
    PullUpAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
