import os
import logging
import math
import sys
import urllib.request

import cv2
import numpy as np

try:
    import av
except ImportError:
    av = None

# Ultralytics가 import 시점에 settings.json을 초기화하므로
# 서버별 홈 디렉터리 이슈를 피하려고 프로젝트 로컬 설정 경로를 먼저 고정한다.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ULTRALYTICS_CONFIG_DIR = os.path.join(PROJECT_DIR, ".ultralytics")
os.makedirs(ULTRALYTICS_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", ULTRALYTICS_CONFIG_DIR)

from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

SUPPORTED_POSE_MODELS = (
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
)
MODEL_DOWNLOAD_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
DEFAULT_MODEL_NAME = "yolo26m-pose.pt"


def log(message):
    print(f"[demo_pullup_visual] {message}", flush=True)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def download_file(url, destination_path):
    log(f"downloading model from {url}")
    urllib.request.urlretrieve(url, destination_path)
    log(f"download_complete path={destination_path}")


def resolve_model_path(models_dir, requested_model_name=None):
    ensure_directory(models_dir)

    candidate_name = requested_model_name or os.environ.get("PULLUP_MODEL_NAME") or DEFAULT_MODEL_NAME
    candidate_name = os.path.basename(candidate_name)

    if candidate_name not in SUPPORTED_POSE_MODELS:
        supported = ", ".join(SUPPORTED_POSE_MODELS)
        raise ValueError(f"unsupported model '{candidate_name}'. Supported models: {supported}")

    candidate_path = os.path.join(models_dir, candidate_name)
    if os.path.exists(candidate_path):
        log(f"using existing model: {candidate_path}")
        return candidate_path

    for model_name in SUPPORTED_POSE_MODELS:
        existing_path = os.path.join(models_dir, model_name)
        if os.path.exists(existing_path):
            log(f"using existing model: {existing_path}")
            return existing_path

    model_url = f"{MODEL_DOWNLOAD_BASE_URL}/{candidate_name}"
    download_file(model_url, candidate_path)
    return candidate_path


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def format_video_date(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return stem


def merge_original_audio(video_only_path, source_video_path, output_video_path):
    if av is None:
        log("PyAV is not installed, keeping video-only output")
        os.replace(video_only_path, output_video_path)
        return False

    temp_output_path = f"{output_video_path}.mux.mp4"

    try:
        with av.open(video_only_path) as video_input, av.open(source_video_path) as source_input:
            if not source_input.streams.audio:
                log("no audio stream found in source video, keeping video-only output")
                os.replace(video_only_path, output_video_path)
                return False

            input_video_stream = video_input.streams.video[0]
            with av.open(temp_output_path, "w") as output:
                output_video_stream = output.add_stream(template=input_video_stream)
                audio_stream_pairs = [
                    (audio_stream, output.add_stream(template=audio_stream))
                    for audio_stream in source_input.streams.audio
                ]

                for packet in video_input.demux(input_video_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = output_video_stream
                    output.mux(packet)

                for input_audio_stream, output_audio_stream in audio_stream_pairs:
                    for packet in source_input.demux(input_audio_stream):
                        if packet.dts is None:
                            continue
                        packet.stream = output_audio_stream
                        output.mux(packet)

    except Exception as exc:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        log(f"PyAV mux failed, keeping video-only output: {exc}")
        os.replace(video_only_path, output_video_path)
        return False

    os.replace(temp_output_path, output_video_path)
    os.remove(video_only_path)
    return True


def draw_rounded_panel(img, top_left, bottom_right, color, alpha=0.75, radius=24, border_color=None, border_thickness=2):
    x1, y1 = top_left
    x2, y2 = bottom_right

    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if border_color is not None:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, border_thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, border_thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, border_thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, border_thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, border_thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, border_thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, border_thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, border_thickness)


def draw_metric_bar(img, origin, size, value, color, label):
    x, y = origin
    width, height = size
    value = clamp(value, 0.0, 1.0)

    cv2.rectangle(img, (x, y), (x + width, y + height), (56, 63, 79), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (105, 112, 130), 1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)
    #cv2.putText(img, label, (x, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 220, 232), 1, cv2.LINE_AA)


def draw_hud(overlay, pull_up_state, frame_shape, session_date):
    height, width = frame_shape[:2]
    metrics = pull_up_state.get_metrics()
    phase_color = {
        "Pull": (98, 211, 179),
        "Down": (247, 196, 88),
        "Deadhang": (96, 174, 255),
        "Ready": (181, 136, 255),
        "Stand": (132, 145, 166),
    }.get(metrics["state"], (132, 145, 166))

    hud_scale = clamp(width / 1080.0, 0.95, 1.05)
    panel_height = 186
    panel_bottom = min(height - 18, int(height * 0.78) + 30)
    panel_top = panel_bottom - panel_height
    panel_left = 34
    panel_right = width - 34
    draw_rounded_panel(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (12, 18, 28),
        alpha=0.68,
        radius=28,
        border_color=(67, 80, 104),
    )

    title_x = panel_left + int(round(24 * hud_scale))
    cv2.putText(overlay, "PULL-UP PERFORMANCE", (title_x, panel_top + int(round(42 * hud_scale))), cv2.FONT_HERSHEY_DUPLEX, 0.92 * hud_scale, (232, 236, 244), 2, cv2.LINE_AA)
    cv2.putText(overlay, session_date, (title_x + 2, panel_top + int(round(84 * hud_scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.70 * hud_scale, (168, 180, 200), 2, cv2.LINE_AA)

    reps_x = title_x
    reps_y = panel_top + int(round(156 * hud_scale))
    cv2.putText(overlay, f"{metrics['count']:2d}", (reps_x, reps_y), cv2.FONT_HERSHEY_DUPLEX, 2.0 * hud_scale, (244, 247, 250), 3, cv2.LINE_AA)
    cv2.putText(overlay, "REPS", (reps_x + int(round(94 * hud_scale)), reps_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.82 * hud_scale, (128, 140, 161), 2, cv2.LINE_AA)

    bar_width = int(round(320 * hud_scale))
    bar_height = int(round(18 * hud_scale))
    bar_x = panel_right - bar_width - int(round(24 * hud_scale))
    bar_y = panel_top + int(round(156 * hud_scale))

    info_x = bar_x
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.46 * hud_scale
    value_scale = 0.60 * hud_scale
    info_value_x = info_x + int(round(104 * hud_scale))
    info_items = [
        ("Grip", metrics["grip"]),
        ("Tempo", f"{metrics['tempo_spm']:.1f} spm"),
        ("Phase", f"{metrics['phase_time']:.2f} s"),
        ("Cycle", f"{metrics['cycle_time']:.2f} s"),
    ]
    for idx, (label, value) in enumerate(info_items):
        base_y = panel_top + int(round(52 * hud_scale)) + idx * int(round(30 * hud_scale))
        label_text = label.upper()
        cv2.putText(overlay, label_text, (info_x, base_y), label_font, label_scale, (126, 137, 156), 2, cv2.LINE_AA)
        cv2.putText(overlay, value, (info_value_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, value_scale, (234, 238, 244), 2, cv2.LINE_AA)

    draw_metric_bar(overlay, (bar_x, bar_y), (bar_width, bar_height), metrics["rom_score"], phase_color, "Range of motion")

def calculate_angle(p1, p2, p3, is_right_arm=False):
    """세 점 사이의 각도를 계산 (p1: 손목, p2: 팔꿈치, p3: 어깨)"""
    if not is_right_arm:
        angle = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0]) -
                             math.atan2(p1[1]-p2[1], p1[0]-p2[0]))
    else:
        angle = math.degrees(math.atan2(p1[1]-p2[1], p1[0]-p2[0]) -
                             math.atan2(p3[1]-p2[1], p3[0]-p2[0]))
    return angle + 360 if angle < 0 else angle

def draw_angle_with_circle(img, pos, angle):
    try:
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.72
        thickness = 2
        text = f"{int(angle):d}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        padding_x, padding_y = 16, 12

        pos_x = int(pos[0].item() if isinstance(pos[0], np.ndarray) else pos[0])
        pos_y = int(pos[1].item() if isinstance(pos[1], np.ndarray) else pos[1])

        rect_x1 = pos_x - (text_width // 2) - padding_x - 10
        rect_y1 = pos_y - (text_height // 2) - padding_y
        rect_x2 = rect_x1 + text_width + padding_x * 2 + 28
        rect_y2 = rect_y1 + text_height + padding_y * 2

        rect_x1 = max(8, rect_x1)
        rect_y1 = max(8, rect_y1)
        rect_x2 = min(img.shape[1] - 8, rect_x2)
        rect_y2 = min(img.shape[0] - 8, rect_y2)

        draw_rounded_panel(
            img,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (14, 24, 38),
            alpha=0.76,
            radius=16,
            border_color=(110, 220, 255),
            border_thickness=2,
        )

        text_x = rect_x1 + ((rect_x2 - rect_x1) - text_width) // 2 - 4
        text_y = rect_y1 + ((rect_y2 - rect_y1) + text_height) // 2 - 1
        text_org = (text_x, text_y)
        cv2.putText(img, text, text_org, font, font_scale, (242, 248, 252), thickness, cv2.LINE_AA)

        degree_center = (text_org[0] + text_width + 9, text_org[1] - text_height + 5)
        cv2.circle(img, degree_center, 3, (242, 248, 252), 1, cv2.LINE_AA)
        cv2.circle(img, degree_center, 5, (110, 220, 255), 1, cv2.LINE_AA)

    except Exception as e:
        print(f"Error in draw_angle_with_circle: {e}")

def draw_text_with_border(img, text, org, color_fill=(255,255,255), color_border=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness_border = 15
    thickness_fill = 3
    cv2.putText(img, text, org, font, font_scale, color_border, thickness_border, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color_fill, thickness_fill, cv2.LINE_AA)

def draw_center_state_overlay(img, state, frame_shape):
    height, width = frame_shape[:2]
    phase_color = {
        "Pull": (98, 211, 179),
        "Down": (247, 196, 88),
        "Deadhang": (96, 174, 255),
        "Ready": (181, 136, 255),
        "Stand": (132, 145, 166),
    }.get(state, (132, 145, 166))

    label = {
        "Pull": "PULL UP",
        "Down": "PULL DOWN",
        "Deadhang": "DEAD HANG",
        "Ready": "READY",
        "Stand": "STAND",
    }.get(state, state.upper())

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.45
    thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    cx = width // 2
    cy = int(height * 0.43) + 200
    pad_x, pad_y = 34, 24
    x1 = cx - text_w // 2 - pad_x
    x2 = cx + text_w // 2 + pad_x
    y1 = cy - text_h - pad_y
    y2 = cy + baseline + pad_y

    draw_rounded_panel(img, (x1, y1), (x2, y2), (16, 22, 34), alpha=0.58, radius=26, border_color=phase_color, border_thickness=3)
    text_x = cx - text_w // 2
    text_y = cy + text_h // 3
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (12, 18, 28), thickness + 5, cv2.LINE_AA)
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (248, 249, 252), thickness, cv2.LINE_AA)

def draw_filled_arc(img, center, pt1, pt2, color=(0, 0, 255)):
    """
    팔꿈치를 중심으로 손목과 어깨 사이의 원호(항상 내부, 즉 작은 각)를 채워진 형태로 그리기.
    center: 팔꿈치, pt1: 손목, pt2: 어깨.
    """
    center = np.array(center)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    
    # 두 선분의 길이 평균을 반지름으로 사용
    # r1 = np.linalg.norm(pt1 - center)
    # r2 = np.linalg.norm(pt2 - center)
    radius = 52
    
    # 각도를 0~360 범위로 정규화
    angle1 = (math.degrees(math.atan2(pt1[1] - center[1], pt1[0] - center[0])) + 360) % 360
    angle2 = (math.degrees(math.atan2(pt2[1] - center[1], pt2[0] - center[0])) + 360) % 360

    # angle1에서 angle2로 가는 각 차이 (모듈로 360)
    diff = (angle2 - angle1) % 360

    # 작은 호(내부 각)를 그리도록 보정:
    if diff <= 180:
        start_angle = angle1
        end_angle = angle1 + diff  # 실제로는 angle2가 될 수도 있음.
    else:
        # diff가 180보다 크면, 반대쪽(360 - diff)이 작은 각임.
        diff = 360 - diff
        start_angle = angle2
        end_angle = angle2 + diff

    overlay = img.copy()
    cv2.ellipse(overlay, tuple(center.astype(int)), (radius, radius), 0, start_angle, end_angle, color, -1)
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)
    cv2.ellipse(img, tuple(center.astype(int)), (radius, radius), 0, start_angle, end_angle, color, 3)
    cv2.ellipse(img, tuple(center.astype(int)), (radius - 10, radius - 10), 0, start_angle, end_angle, (235, 245, 255), 1)

class PullUpState:
    def __init__(self, fps=30):
        self.current_state = "Stand"
        self.last_state = "Stand"
        self.grip_type = "-"
        self.isStand = True
        self.isReady = False
        self.prev_left_angle = None
        self.prev_right_angle = None
        self.pullup_cnt = 0 

        self.bar_hight = 0
        self.fps = fps if fps > 0 else 30
        self.frame_index = 0
        self.phase_start_frame = 0
        self.last_rep_frame = None
        self.last_pull_frame = None
        self.last_down_frame = None
        self.rep_durations = []
        self.pull_phase_durations = []
        self.down_phase_durations = []
        self.left_angle_history = []
        self.right_angle_history = []

    def set_fps(self, fps):
        if fps > 0:
            self.fps = fps

    def _seconds_from_frames(self, frames):
        return frames / self.fps if self.fps > 0 else 0.0

    def _append_limited(self, items, value, limit=8):
        items.append(value)
        if len(items) > limit:
            items.pop(0)

    def _update_transition_metrics(self, new_state):
        if new_state == self.last_state:
            return

        now_frame = self.frame_index
        phase_duration = self._seconds_from_frames(now_frame - self.phase_start_frame)

        if self.last_state == "Pull":
            self._append_limited(self.pull_phase_durations, phase_duration)
        elif self.last_state == "Down":
            self._append_limited(self.down_phase_durations, phase_duration)

        if new_state == "Pull":
            self.last_pull_frame = now_frame
            if self.last_rep_frame is not None:
                self._append_limited(self.rep_durations, self._seconds_from_frames(now_frame - self.last_rep_frame))
            self.last_rep_frame = now_frame
        elif new_state == "Down":
            self.last_down_frame = now_frame

        self.phase_start_frame = now_frame

    def get_metrics(self):
        avg_cycle = float(np.mean(self.rep_durations)) if self.rep_durations else 0.0
        avg_pull = float(np.mean(self.pull_phase_durations)) if self.pull_phase_durations else 0.0
        avg_down = float(np.mean(self.down_phase_durations)) if self.down_phase_durations else 0.0
        tempo_spm = 60.0 / avg_cycle if avg_cycle > 0 else 0.0

        cycle_cv = float(np.std(self.rep_durations) / avg_cycle) if len(self.rep_durations) >= 2 and avg_cycle > 0 else 0.0
        tempo_score = clamp(1.0 - cycle_cv * 2.2, 0.0, 1.0) if self.rep_durations else 0.0

        recent_left = float(np.mean(self.left_angle_history)) if self.left_angle_history else 0.0
        recent_right = float(np.mean(self.right_angle_history)) if self.right_angle_history else 0.0
        balance_gap = abs(recent_left - recent_right)
        symmetry_score = clamp(1.0 - balance_gap / 25.0, 0.0, 1.0) if (recent_left and recent_right) else 0.0

        current_rom = min(self.prev_left_angle or 0.0, self.prev_right_angle or 0.0)
        rom_score = clamp((160.0 - current_rom) / 105.0, 0.0, 1.0) if current_rom else 0.0

        stability_score = int(round((tempo_score * 0.45 + symmetry_score * 0.35 + rom_score * 0.20) * 100))
        if stability_score >= 85:
            label = "Elite control"
        elif stability_score >= 70:
            label = "Very stable"
        elif stability_score >= 55:
            label = "Solid rhythm"
        else:
            label = "Needs tuning"

        return {
            "state": self.current_state,
            "grip": self.grip_type,
            "count": self.pullup_cnt,
            "tempo_spm": tempo_spm,
            "cycle_time": avg_cycle,
            "pull_time": avg_pull,
            "down_time": avg_down,
            "phase_time": self._seconds_from_frames(self.frame_index - self.phase_start_frame),
            "tempo_score": tempo_score,
            "symmetry_score": symmetry_score,
            "range_score": rom_score,
            "rom_score": rom_score,
            "stability_score": stability_score,
            "stability_label": label,
        }
        
    def determine_state(self, kpts):
        self.frame_index += 1
        
        left_shoulder = kpts[5][:2].astype(int)
        right_shoulder = kpts[6][:2].astype(int)
        left_elbow = kpts[7][:2].astype(int)
        right_elbow = kpts[8][:2].astype(int)
        left_wrist = kpts[9][:2].astype(int)
        right_wrist = kpts[10][:2].astype(int)
        left_hip = kpts[11][:2].astype(int)
        right_hip = kpts[12][:2].astype(int)
        
        left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder, is_right_arm=False)
        right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder, is_right_arm=True)
        self._append_limited(self.left_angle_history, left_angle)
        self._append_limited(self.right_angle_history, right_angle)
        
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        base = (right_shoulder[0]-left_shoulder[0])

        new_state = self.last_state

        cur_wrist_y = (left_wrist[1] + right_wrist[1]) / 2

        if self.isStand == True :
            if (cur_wrist_y < shoulder_y - base*0.7) and cur_wrist_y !=0 :
              new_state = "Ready"
              self.isStand = False

        else:
            if (cur_wrist_y > self.bar_hight*1.1) and self.bar_hight != 0:
                self.bar_hight = 0
                self.isStand = True
                self.isReady = False    
                new_state = "Stand"
        
        if self.isStand == False:
            
            # check deadhang
            if left_angle > 155 and right_angle > 155:
                new_state = "Deadhang"

                self.isReady = True
                # check grip type
                shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                wrist_width = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
                ratio = wrist_width / shoulder_width if shoulder_width > 0 else 1
                if ratio >= 2.0:
                    self.grip_type = "Wide"
                else:
                    self.grip_type = "Narrow"
                    
                    
                if self.bar_hight == 0:
                    self.bar_hight = (left_wrist[1]+right_wrist[1]) / 2
                    log("bar height updated")
        
        if self.isReady :
            
            # check pull
            if (left_angle <= 90 and right_angle <= 90) and \
                ((self.prev_left_angle - left_angle > 2 and self.prev_right_angle - right_angle > 2)):
                
                if(self.last_state != "Pull"):
                    log(f"pull detected count={self.pullup_cnt + 1}")
                    self.pullup_cnt = self.pullup_cnt+1

                new_state = "Pull"

            # # ((left_angle - self.prev_left_angle) > 2 and (right_angle - self.prev_right_angle) > 2) or \
            # check down
            if self.last_state == "Pull" and \
                ((left_angle > 50 and right_angle > 50) and (left_angle < 155 and right_angle < 155)) and \
                ((left_angle - self.prev_left_angle > 2 and right_angle - self.prev_right_angle > 2)) :
                
                if(self.last_state != "Down"):
                    log("down detected")
                
                new_state = "Down"
            
        self.prev_left_angle = left_angle
        self.prev_right_angle = right_angle

        self._update_transition_metrics(new_state)
        self.current_state = new_state
        self.last_state = self.current_state
        return self.current_state

def draw_limb_gradient_line(img, pt1, pt2, color_start, color_end, thickness=8, segments=14):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        p1 = tuple(np.round(pt1 * (1 - t1) + pt2 * t1).astype(int))
        p2 = tuple(np.round(pt1 * (1 - t2) + pt2 * t2).astype(int))
        color = tuple(
            int(color_start[c] * (1 - t1) + color_end[c] * t1)
            for c in range(3)
        )
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_joint_glow(img, center, core_color, glow_color, radius=11):
    for r, a in [(radius + 12, 0.10), (radius + 7, 0.18), (radius + 2, 0.28)]:
        glow = img.copy()
        cv2.circle(glow, tuple(center), r, glow_color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, a, img, 1 - a, 0, img)
    cv2.circle(img, tuple(center), radius, core_color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(center), radius + 1, (245, 245, 245), 2, cv2.LINE_AA)


def draw_skeleton_custom_style(frame, results, pull_up_state, session_date):
    visual = frame.copy()
    overlay = frame.copy()

    try:
        if len(results) == 0 or len(results[0].keypoints.data) == 0:
            return visual

        kpts = results[0].keypoints.data[0].cpu().numpy()
        state = pull_up_state.determine_state(kpts)

        left_shoulder = kpts[5][:2].astype(int)
        right_shoulder = kpts[6][:2].astype(int)
        left_elbow = kpts[7][:2].astype(int)
        right_elbow = kpts[8][:2].astype(int)
        left_wrist = kpts[9][:2].astype(int)
        right_wrist = kpts[10][:2].astype(int)
        left_hip = kpts[11][:2].astype(int)
        right_hip = kpts[12][:2].astype(int)

    except Exception as e:
        print(f"Error: {e}")
        return visual

    left_hip[1] = int(left_hip[1] * 0.9)
    right_hip[1] = int(right_hip[1] * 0.9)

    left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder, is_right_arm=False)
    right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder, is_right_arm=True)

    arm_outer_start = (255, 168, 92)
    arm_outer_end = (255, 111, 157)
    arm_inner_start = (82, 232, 255)
    arm_inner_end = (96, 160, 255)
    torso_start = (255, 146, 178)
    torso_end = (126, 191, 255)

    left_angle_badge_pos = None
    right_angle_badge_pos = None
    if 2 < left_angle < 180:
        draw_filled_arc(overlay, left_elbow, left_wrist, left_shoulder, color=(94, 220, 255))
        left_angle_badge_pos = (left_elbow[0] + 2, left_elbow[1] + 58)

    if 2 < right_angle < 180:
        draw_filled_arc(overlay, right_elbow, right_wrist, right_shoulder, color=(94, 220, 255))
        right_angle_badge_pos = (right_elbow[0] + 2, right_elbow[1] + 58)

    draw_limb_gradient_line(overlay, tuple(left_shoulder), tuple(left_elbow), arm_outer_start, arm_outer_end, thickness=9)

    if left_wrist[0] * left_wrist[1] != 0:
        draw_limb_gradient_line(overlay, tuple(left_elbow), tuple(left_wrist), arm_inner_start, arm_inner_end, thickness=9)

    if right_shoulder[0] * right_elbow[1] != 0:
        draw_limb_gradient_line(overlay, tuple(right_shoulder), tuple(right_elbow), arm_outer_start, arm_outer_end, thickness=9)

    if right_wrist[0] * right_wrist[1] != 0:
        draw_limb_gradient_line(overlay, tuple(right_elbow), tuple(right_wrist), arm_inner_start, arm_inner_end, thickness=9)

    if left_shoulder[0] * right_shoulder[1] != 0:
        draw_limb_gradient_line(overlay, tuple(left_shoulder), tuple(right_shoulder), torso_start, torso_end, thickness=9)

    key_points = [5, 6, 7, 8, 9, 10]
    joint_core = {
        5: (255, 194, 138),
        6: (255, 194, 138),
        7: (126, 240, 255),
        8: (126, 240, 255),
        9: (255, 151, 196),
        10: (255, 151, 196),
    }
    joint_glow = {
        5: (90, 144, 255),
        6: (90, 144, 255),
        7: (88, 218, 255),
        8: (88, 218, 255),
        9: (255, 120, 176),
        10: (255, 120, 176),
    }
    for idx in key_points:
        pt = kpts[idx][:2].astype(int)
        draw_joint_glow(overlay, tuple(pt), joint_core[idx], joint_glow[idx], radius=10)

    if left_angle_badge_pos is not None:
        draw_angle_with_circle(overlay, left_angle_badge_pos, left_angle)

    if right_angle_badge_pos is not None:
        draw_angle_with_circle(overlay, right_angle_badge_pos, right_angle)

    draw_hud(overlay, pull_up_state, frame.shape, session_date)
    draw_center_state_overlay(overlay, state, frame.shape)

    visual = cv2.addWeighted(overlay, 0.78, visual, 0.22, 0)
    return visual

    return visual

def process_video(model, input_video_path, output_video_path):
    log(f"input_video_path={os.path.abspath(input_video_path)}")
    log(f"output_video_path={os.path.abspath(output_video_path)}")
    temp_video_path = f"{output_video_path}.video_only.mp4"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        log(f"ERROR: failed to open input video: {input_video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"input_video_info width={width}, height={height}, fps={fps}, frames={total_input_frames}")

    # 만약 영상이 가로로 회전되어 있다면(가로 > 세로), 90도 회전시켜 항상 세로 영상으로 만듦
    if width > height:
        frame_size = (height, width)  # 회전 후 해상도 (세로, 가로)
    else:
        frame_size = (width, height)

    if fps <= 0:
        log("fps 정보를 읽지 못해 기본값 30을 사용합니다.")
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_size[0], frame_size[1]))
    if not out.isOpened():
        log(f"ERROR: failed to create output video: {temp_video_path}")
        cap.release()
        return False

    log(f"writer_opened frame_size={frame_size}, fps={fps}")

    pull_up_state = PullUpState(fps=fps)
    session_date = format_video_date(input_video_path)
    frame_cnt = 0
    total_frame_cnt = total_input_frames if total_input_frames > 0 else 99999
    log(f"start_processing total_frame_cnt={total_frame_cnt}")

    while True:
        ret, frame = cap.read()
        if not ret:
            log(f"frame read finished at frame_cnt={frame_cnt}")
            break
        
        # 만약 영상이 가로로 촬영된 경우, 90도 회전시켜 세로로 만듦
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = model(
            frame,
            conf=0.55,
            iou=0.5,
            classes=[0],
            verbose=False,
            device=0,
        )
        
        annotated_frame = draw_skeleton_custom_style(frame, results, pull_up_state, session_date)
        out.write(annotated_frame)
        frame_cnt += 1
        progress = (frame_cnt / total_frame_cnt) * 100
        print(f"Processing {os.path.basename(input_video_path)} {frame_cnt} / {total_frame_cnt} ({progress:.1f}%)", end="\r")
    
    cap.release()
    out.release()
    print()
    merged = merge_original_audio(temp_video_path, input_video_path, output_video_path)
    if merged:
        log(f"done processed_frames={frame_cnt}, output_with_audio={os.path.abspath(output_video_path)}")
    else:
        log(f"done processed_frames={frame_cnt}, output_video_only={os.path.abspath(output_video_path)}")
    return True


def main():
    models_dir = os.path.join(PROJECT_DIR, "models")
    videos_dir = os.path.join(PROJECT_DIR, "videos")
    results_dir = os.path.join(PROJECT_DIR, "results")

    log(f"script_dir={PROJECT_DIR}")
    log(f"models_dir={models_dir}")
    log(f"videos_dir={videos_dir}")
    log(f"results_dir={results_dir}")
    ensure_directory(results_dir)

    try:
        model_path = resolve_model_path(models_dir)
    except Exception as exc:
        log(f"ERROR: failed to prepare model: {exc}")
        sys.exit(1)

    input_video_paths = sorted(
        os.path.join(videos_dir, name)
        for name in os.listdir(videos_dir)
        if name.lower().endswith(".mp4")
    ) if os.path.isdir(videos_dir) else []

    if not input_video_paths:
        log(f"ERROR: no mp4 files found in {videos_dir}")
        sys.exit(1)

    log(f"model_path={os.path.abspath(model_path)}")
    log("loading YOLO model")
    model = YOLO(model_path)
    log("YOLO model loaded")

    success_count = 0
    for input_video_path in input_video_paths:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(results_dir, f"{base_name}_result.mp4")
        if process_video(model, input_video_path, output_video_path):
            success_count += 1

    log(f"batch_complete processed={success_count}/{len(input_video_paths)}")

if __name__ == "__main__":
    main()
