"""DrowsyAlert - phát hiện buồn ngủ realtime qua webcam.

Phiên bản này ưu tiên giảm false-positive trong tình huống lái xe thực tế:
- CNN mắt được làm mượt theo thời gian + hysteresis.
- Loại frame mắt kém tin cậy khỏi PERCLOS (quay đầu lớn, quá tối/sáng, crop quá ít tương phản).
- Ngáp/gật đầu là tín hiệu phụ; một lần đơn lẻ không đủ kết luận buồn ngủ.
- C1 = CNN + PERCLOS; C2 = C1 + MAR; C3 = C1 + Head Pose; C4 = C1 + Head Pose + MAR.
- Có reset phiên đo để làm thực nghiệm sạch.
- Luồng video độc lập; đồng bộ snapshot AI bằng ID frame và thời điểm nhận frame.
- UI chỉ vẽ snapshot mới; dữ liệu cũ được đánh dấu thay vì giả như vẫn realtime.
- Lưu log theo từng frame AI (không lấy mẫu lặp lại từ UI).
- Bounding box ổn định theo lần nhận diện gần nhất, không ẩn ở các frame xen kẽ.
- Cảnh báo dựa vào nhắm mắt kéo dài, PERCLOS kéo dài hoặc tổ hợp sự kiện lặp lại;
  ngáp/gật đơn lẻ chỉ là dữ kiện quan sát, không tự bật còi.

Chạy: python -m streamlit run app.py
"""

import base64
from pathlib import Path
import sys
APP_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = APP_PATH.parent.parent if APP_PATH.parent.name == "changes" else APP_PATH.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
from scripts.eye_test_diagnostics import DIAGNOSTIC_FIELDS, EyeDiagnosticsMixin, render_eye_diagnostics
from scripts.eye_decision import cnn_eye_decision
from scripts.perclos_experiment import ExperimentMixin, EXPERIMENT_FIELDS, render_experiment
from scripts.alert_presentation import presentation_state, draw_camera_alert
import collections
import math
import os
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ----------------------------------------------------------------------------
# Hằng số
# ----------------------------------------------------------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = dict(top=13, bottom=14, left=78, right=308)
POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)

TEAM_MODEL_PATH = REPOSITORY_ROOT / "models/eye_cnn_team.keras"
EYE_MODEL_PATH = str(TEAM_MODEL_PATH if TEAM_MODEL_PATH.is_file() else
                     REPOSITORY_ROOT / "outputs/model_conversion/train_20261016_045537_835503/eye_cnn.keras")
EYE_INPUT = 64
CLOSED_IDX = 0

# Tối ưu realtime: MediaPipe chỉ cần ảnh nhỏ; camera hiển thị vẫn 640x360.
MP_PROCESS_WIDTH = 384
POSE_EVERY_N_FRAMES = 3
UI_INTERVAL = 0.10       # polling 10Hz nhẹ; chỉ render khi AI có frame mới
DASHBOARD_INTERVAL = 0.125  # mục tiêu 8Hz, giảm tải Streamlit/browser
DEBUG_INTERVAL = 1.00   # debug ở tab thu gọn, không tranh FPS với camera
CHART_INTERVAL = 2.00   # tránh redraw chart thường xuyên
STALE_AFTER_S = 0.85    # nếu AI mất cập nhật, ngừng hiển thị như số liệu hiện tại
OVERLAY_MAX_AGE_S = 1.20  # giữ hộp giữa các lần AI cập nhật; quá tuổi => ẩn để tránh hộp trôi
OVERLAY_MISSING_GRACE_S = 0.50  # FaceMesh rớt 1-2 khung: vẫn giữ hộp tránh nhấp nháy
OVERLAY_DIM_AFTER_S = 0.60  # AI chậm: hộp chuyển xám, không dùng để phát còi
ALARM_EVIDENCE_HOLD_S = 1.35  # tín hiệu kết hợp phải bền, không phát còi tức thì
ALARM_COOLDOWN_S = 7.0  # sau khi hết cảnh báo, tránh báo lại liên tục bởi dữ kiện cũ
PERCLOS_EVIDENCE_HOLD_S = 3.0  # PERCLOS cao phải kéo dài
EYE_RECENT_WINDOW_S = 8.0   # không quyết định chỉ từ PERCLOS quá khứ
EVENT_WINDOW_S = 40.0  # đếm ngáp/gật lặp lại trong 40 giây
MAX_SESSION_LOG_ROWS = 30000  # ~30-50 phút tuỳ AI FPS, giới hạn RAM

DEFAULT_CFG = dict(
    config="C4",
    ear_thr=0.21,
    mar_thr=0.60,
    nod_deg=20.0,
    yaw_limit=35.0,
    window=30,
    perclos_thr=0.30,
    closed_sec=2.0,
    yawn_sec=1.5,
    covered_yawn_sec=1.2,
    nod_sec=1.0,
    min_brightness=35.0,
    max_brightness=220.0,
    min_eye_contrast=10.0,
    detect_covered_yawn=True,
    yawn_audio=True,
)

CONFIG_LABELS = {
    "C1": "C1 · CNN + PERCLOS",
    "C2": "C2 · C1 + MAR",
    "C3": "C3 · C1 + Head Pose",
    "C4": "C4 · C1 + Head Pose + MAR",
}

# OpenCV dùng BGR
C_OK = (165, 209, 63)
C_WARN = (75, 184, 242)
C_BAD = (78, 90, 255)
C_SKIP = (180, 180, 180)


# ----------------------------------------------------------------------------
# Đặc trưng hình học
# ----------------------------------------------------------------------------
def _dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(p):
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


def mouth_aspect_ratio(pts):
    v = _dist(pts[MOUTH["top"]], pts[MOUTH["bottom"]])
    h = _dist(pts[MOUTH["left"]], pts[MOUTH["right"]])
    return v / (h + 1e-6)


def head_pose_angles(pts, w, h):
    """Ước lượng pitch/yaw/roll bằng solvePnP."""
    img_pts = pts[POSE_IDX].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D, img_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = (float(a) for a in cv2.RQDecomp3x3(rmat)[0])

    def norm_angle(a):
        while a > 90:
            a -= 180
        while a < -90:
            a += 180
        return a

    return norm_angle(pitch), norm_angle(yaw), norm_angle(roll)


def crop_eye(gray, eye_pts, size=EYE_INPUT):
    """Crop vuông quanh mắt; output xám 64x64 [0,1], khớp train_eye_cnn.py."""
    x0, y0 = eye_pts.min(axis=0)
    x1, y1 = eye_pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) * 0.75
    x0, x1, y0, y1 = int(cx - r), int(cx + r), int(cy - r), int(cy + r)
    H, W = gray.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (size, size)).astype(np.float32) / 255.0


def load_eye_cnn():
    if not os.path.exists(EYE_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(EYE_MODEL_PATH)
        fn = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec([None, EYE_INPUT, EYE_INPUT, 1], tf.float32)],
        )
        fn(tf.zeros((2, EYE_INPUT, EYE_INPUT, 1)))
        return fn
    except Exception as e:  # noqa: BLE001
        print("Không nạp được CNN:", e)
        return None


# ----------------------------------------------------------------------------
# Bộ xử lý realtime
# ----------------------------------------------------------------------------
class DrowsyProcessor(VideoProcessorBase):
    def __init__(self):
        self.cfg = dict(DEFAULT_CFG)
        self.lock = threading.Lock()
        self.proc_lock = threading.RLock()  # Reset và phân tích không sửa state cùng lúc
        # FaceMesh / CNN thuộc thread phân tích, không chặn luồng video đầu ra.
        self.mesh = None
        self.cnn = None
        self._model_ready = False
        self.history = collections.deque(maxlen=240)
        self.fps = 0.0  # analysis FPS
        self.camera_fps = 0.0  # WebRTC outgoing FPS
        self._last_camera_frame = 0.0
        self.processing_ms = 0.0
        self.cnn_ms = 0.0
        self.frame_idx = 0
        self.last_pose = None
        self._last = time.perf_counter()
        self._last_receive = time.perf_counter()
        self._camera_seq = 0
        self._analysis_seq = 0
        self._new_logs = collections.deque(maxlen=900)
        self._pending = None
        self._ready = threading.Event()
        self.overlay = None
        self._worker = None
        self.hands = None
        self.reset_session(reset_pose=True)
        self._start_worker()

    def reset_session(self, reset_pose=False):
        """Xóa trạng thái tích lũy để bắt đầu một phiên đo sạch."""
        with self.proc_lock, self.lock:
            self.eye_hist = collections.deque()       # (time, closed)
            self.eye_prob_hist = collections.deque(maxlen=2)
            self.blink_times = collections.deque()
            self.slow_closure_times = collections.deque()
            self.last_eye_sample_at = None
            self.perclos_high_since = None
            self.last_alarm_ended_at = -1e9
            self.prev_drowsy = False
            self.nod_recovered_since = None
            self.last_nod_event_at = -1e9
            self.yawn_recovered_since = None
            self.last_yawn_event_at = -1e9
            self.open_ear_samples = collections.deque(maxlen=90)
            self.open_ear_ref = None
            self.eye_open_since = None
            self.eye_closed_count = 0
            self.eye_state_closed = False
            self.eye_confident = False
            self.session_started_at = time.time()
            self.face_missing_since = None
            self.closed_since = None
            self.yawn_since = None
            self.yawn_counted = False
            self.yawn_times = collections.deque()
            self.last_yawn_type = ""
            self.open_yawns_count = 0
            self.covered_yawns_count = 0
            self.covered_yawn_since = None
            self.covered_yawn_counted = False
            self.hand_near_mouth = False
            self.nod_since = None
            self.nod_counted = False
            self.nod_times = collections.deque()
            self.drowsy_since = None
            self.drowsy_until = 0.0
            self.reason = ""
            self.score = 0.0
            self.risk_level = "Tỉnh táo"
            self.left_closed_prob = None
            self.right_closed_prob = None
            self.left_eye_crop = None
            self.right_eye_crop = None
            self.history.clear()
            self._new_logs.clear()
            # Đảm bảo UI không tiếp tục đọc kết quả AI của phiên cũ.
            self._analysis_seq = 0
            self.overlay = None
            self.last_face_overlay_mono = 0.0
            if reset_pose or not hasattr(self, "baseline"):
                self.pitch_buf, self.yaw_buf = [], []
                self.baseline, self.yaw_baseline = None, None
            self.stats = self._empty_stats()

    def _empty_stats(self):
        return dict(
            face=False, drowsy=False, reason="", risk_level="Chưa đủ dữ liệu", score=0.0,
            perclos=0.0, perclos_ready=False, ear=0.0, mar=0.0,
            pitch_dev=0.0, yaw_dev=0.0, yawns=0, nods=0,
            closed=False, eye_valid=False, eye_confident=False, mouth_valid=False,
            yawn_type="", open_yawns=0, covered_yawns=0, hand_near_mouth=False,
            blinks=0, slow_closures=0, recent_eye_ratio=0.0, eye_closed_for=0.0,
            alarm_evidence="",
            brightness=0.0, eye_contrast=0.0, observation_quality=0.0,
            eye_score=0.0, yawn_score=0.0, nod_score=0.0,
            left_closed_prob=None, right_closed_prob=None,
            left_eye_crop=None, right_eye_crop=None,
            fps=self.fps, camera_fps=self.camera_fps,
            processing_ms=self.processing_ms, cnn_ms=self.cnn_ms,
            frame_seq=0, sampled_at=0.0, inference_completed_at=0.0,
            sample_age_ms=0.0, inference_delay_ms=0.0, queue_ms=0.0,
            stale=False, live=False,
            source="CNN chính · EAR theo dõi" if self.cnn else ("Đang nạp" if not self._model_ready else "EAR"),
        )

    def reset_baseline(self):
        with self.proc_lock, self.lock:
            self.pitch_buf, self.yaw_buf = [], []
            self.baseline, self.yaw_baseline = None, None

    def get_stats(self):
        """Snapshot nhất quán: thống kê/hình debug/log cùng kết quả phân tích.

        sample_age_ms đo từ thời điểm server nhận frame tới lúc UI hỏi;
        không bao gồm độ trễ webcam/browser/network trước khi vào Python.
        """
        with self.lock:
            data = dict(self.stats)
            history = list(self.history)
            logs = list(self._new_logs)
            self._new_logs.clear()
            data["camera_fps"] = self.camera_fps
            # Đồng hồ monotonic đảm bảo không ảnh hưởng chỉnh giờ hệ thống.
            capture = data.pop("_capture_mono", None)
            age_s = max(0.0, time.perf_counter() - capture) if capture is not None else 0.0
            data["sample_age_ms"] = age_s * 1000.0
            data["stale"] = capture is not None and age_s > STALE_AFTER_S
            data["live"] = capture is not None and not data["stale"]
            if data["stale"]:
                # Không dùng nhãn đã quá hạn để phát còi hoặc biểu diễn như khung hiện tại.
                data.update(face=False, drowsy=False, reason="",
                            risk_level="AI tạm chậm / chưa có kết quả mới",
                            eye_valid=False, eye_confident=False, closed=False,
                            left_closed_prob=None, right_closed_prob=None)
        return data, history, logs

    def _start_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._analysis_loop, name="DrowsyAlert-analysis", daemon=True)
            self._worker.start()

    def _analysis_loop(self):
        try:
            if self.mesh is None:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                )
            if self.hands is None:
                try:
                    self.hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.45,
                        min_tracking_confidence=0.45,
                    )
                except Exception as he:
                    print("Không khởi tạo được MediaPipe Hands:", repr(he))
                    self.hands = None
            if not self._model_ready:
                self.cnn = load_eye_cnn()
                self._model_ready = True
            while True:
                # Nhận frame mới nhất: mọi frame cũ đến khi inference đang bận sẽ được bỏ qua.
                # Không tạo queue dài và không làm tăng thời gian chờ video.
                self._ready.wait(timeout=0.5)
                with self.lock:
                    item = self._pending
                    self._pending = None
                    self._ready.clear()
                    idle = time.perf_counter() - self._last_receive
                if item is None:
                    if idle > 8.0:
                        break
                    continue
                img, sampled_at, captured_mono, frame_seq = item
                try:
                    with self.proc_lock:
                        self._analyze_frame(img, sampled_at, captured_mono, frame_seq)
                except Exception as e:
                    print("Lỗi phân tích frame DrowsyAlert:", repr(e))
        except Exception as e:
            print("Không khởi tạo được worker DrowsyAlert:", repr(e))
        finally:
            if self.mesh is not None:
                try:
                    self.mesh.close()
                except Exception:
                    pass
                self.mesh = None
            if self.hands is not None:
                try:
                    self.hands.close()
                except Exception:
                    pass
                self.hands = None

    def recv(self, frame):
        """Camera không chờ AI; giữ hộp xuyên suốt giữa các snapshot AI.

        FaceMesh cập nhật lại tọa độ hộp theo tốc độ phân tích; khi AI đứng quá
        lâu, hộp đổi xám rồi ẩn thay vì hiển thị vị trí sai vô hạn.
        """
        t = time.perf_counter()
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        now = time.time()
        with self.lock:
            if self._last_camera_frame:
                dt = t - self._last_camera_frame
                if dt > 0:
                    fps = min(90.0, 1.0 / dt)
                    self.camera_fps = (0.88 * self.camera_fps + 0.12 * fps) if self.camera_fps else fps
            self._last_camera_frame = t
            self._camera_seq += 1
            frame_seq = self._camera_seq
            # Mailbox 1 chỗ: thay frame cũ, không có hàng đợi tồn đọng.
            self._pending = (img.copy(), now, t, frame_seq)
            self._last_receive = t
            overlay = self.overlay
        self._ready.set()
        if self._worker is None or not self._worker.is_alive():
            self._start_worker()

        # KHÔNG cắt hộp theo số frame/180 ms như bản trước: đó là nguyên nhân nháy.
        # Chỉ bỏ hộp khi thực sự mất theo dõi hoặc worker không cập nhật lâu.
        if overlay is not None and 0 <= t - overlay["finished_mono"] <= OVERLAY_MAX_AGE_S:
            boxes = overlay.get("boxes", ())
            if boxes:
                age = t - overlay["finished_mono"]
                for coords, color in boxes:
                    draw_color = C_SKIP if age > OVERLAY_DIM_AFTER_S else color
                    cv2.rectangle(img, tuple(coords[:2]), tuple(coords[2:]), draw_color, 2, cv2.LINE_AA)
                if overlay.get("alarm") and age <= OVERLAY_DIM_AFTER_S:
                    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), C_BAD, 3)
                    draw_camera_alert(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _eye_closed(self, right_crop, left_crop, ear, eye_valid):
        """Phân loại mắt từ crop đã tính sẵn để tránh crop/resize hai lần mỗi frame."""
        self.right_closed_prob = None
        self.left_closed_prob = None
        self.right_eye_crop = None
        self.left_eye_crop = None

        if right_crop is not None:
            self.right_eye_crop = (np.clip(right_crop, 0, 1) * 255).astype(np.uint8)
        if left_crop is not None:
            self.left_eye_crop = (np.clip(left_crop, 0, 1) * 255).astype(np.uint8)

        if not eye_valid:
            self.eye_confident = False
            self.cnn_ms = 0.0
            return self.eye_state_closed

        if self.cnn is None:
            ear_threshold = (0.74 * self.open_ear_ref) if self.open_ear_ref is not None else self.cfg["ear_thr"]
            self.eye_state_closed = ear < ear_threshold
            self.eye_confident = True
            self.cnn_ms = 0.0
            return self.eye_state_closed

        valid = [("right", right_crop), ("left", left_crop)]
        valid = [(name, crop) for name, crop in valid if crop is not None]
        if not valid:
            self.eye_confident = False
            self.cnn_ms = 0.0
            return self.eye_state_closed

        x = np.stack([crop for _, crop in valid])[..., None]
        t_cnn = time.perf_counter()
        prob = np.asarray(self.cnn(x))
        self.cnn_ms = (time.perf_counter() - t_cnn) * 1000.0
        closed_prob = (
            prob[:, CLOSED_IDX]
            if prob.ndim == 2 and prob.shape[1] > 1
            else 1 - prob.ravel()
        )

        for (name, _), p in zip(valid, closed_prob):
            if name == "right":
                self.right_closed_prob = float(p)
            else:
                self.left_closed_prob = float(p)

        # Chỉ giữ 2 frame và đặt trọng số lớn cho frame mới để trạng thái bám chuyển động.
        mean_prob = float(np.mean(closed_prob))
        self.eye_prob_hist.append(mean_prob)
        if len(self.eye_prob_hist) == 1:
            smooth_prob = mean_prob
        else:
            smooth_prob = 0.75 * self.eye_prob_hist[-1] + 0.25 * self.eye_prob_hist[-2]

        # Bằng chứng nhắm: (1) cả HAI mắt CNN đóng, (2) EAR giảm so với mốc
        # mở mắt cá nhân. Chặn lỗi Kaggle->webcam gây P(Closed) 90% khi mắt mở.
        ear_ref = self.open_ear_ref
        bilateral_close = (len(closed_prob) == 2 and
                           float(np.min(closed_prob)) >= 0.60 and smooth_prob >= 0.72)
        if getattr(self, "EYE_DECISION_POLICY", "legacy") == "cnn_primary_v1":
            self.eye_state_closed, self.eye_confident, self._cnn_rule = cnn_eye_decision(
                list(closed_prob), smooth_prob, self.eye_state_closed)
        elif ear_ref is None:
            self.eye_state_closed = False
            self.eye_confident = False   # chưa chuẩn hóa mắt: không phán đoán
        elif ear >= 0.82 * ear_ref:
            self.eye_state_closed = False
            self.eye_confident = True
        elif ear <= 0.74 * ear_ref and bilateral_close:
            self.eye_state_closed = True
            self.eye_confident = True
        elif smooth_prob <= 0.35 and ear > 0.74 * ear_ref:
            self.eye_state_closed = False
            self.eye_confident = True
        else:
            self.eye_state_closed = False
            self.eye_confident = False

        return self.eye_state_closed

    def _compute_scores(self, cfg, perclos, eye_history_support, closed_long, slow_count):
        """Điểm số theo thời gian. Ngáp/gật đơn lẻ không đủ tạo cảnh báo."""
        # Mắt là tín hiệu chính.
        eye_score = 0.0
        if eye_history_support:
            ratio = perclos / max(cfg["perclos_thr"], 1e-6)
            if ratio >= 1.35:
                eye_score += 2.5
            elif ratio >= 1.0:
                eye_score += 2.0
            elif ratio >= 0.75:
                eye_score += 1.0
        if slow_count >= 2:
            eye_score = max(eye_score, 1.0)
        if closed_long:
            eye_score = max(eye_score, 3.0)  # microsleep / nhắm kéo dài

        # Ngáp chỉ là tín hiệu phụ. Một lần gần đây chỉ +0.5.
        yawn_count = len(self.yawn_times)
        yawn_score = 0.0
        if yawn_count == 1:
            yawn_score = 0.5
        elif yawn_count >= 2:
            yawn_score = 1.0

        # Gật đầu cũng là tín hiệu phụ; lặp lại mới có trọng lượng rõ.
        nod_count = len(self.nod_times)
        nod_score = 0.0
        if nod_count == 1:
            nod_score = 0.75
        elif nod_count >= 2:
            nod_score = 1.5

        if cfg["config"] == "C1":
            total = eye_score
        elif cfg["config"] == "C2":
            total = eye_score + yawn_score
        elif cfg["config"] == "C3":
            total = eye_score + nod_score
        else:
            total = eye_score + yawn_score + nod_score

        return eye_score, yawn_score, nod_score, total

    def _analyze_frame(self, img, now, captured_mono, frame_seq):
        t_frame = time.perf_counter()
        queue_ms = max(0.0, (t_frame - captured_mono) * 1000.0)
        h, w = img.shape[:2]
        cfg = dict(self.cfg)
        self.frame_idx += 1

        # FaceMesh là khâu nặng nhất: chạy trên frame 384px, landmarks vẫn là tọa độ chuẩn hóa
        # nên có thể quy đổi trực tiếp về frame hiển thị 640x360.
        if w > MP_PROCESS_WIDTH:
            scale = MP_PROCESS_WIDTH / float(w)
            small = cv2.resize(img, (MP_PROCESS_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = img
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_small.flags.writeable = False
        res = self.mesh.process(rgb_small)

        perf_now = time.perf_counter()
        dt = perf_now - self._last
        self._last = perf_now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = 0.82 * self.fps + 0.18 * inst_fps if self.fps else inst_fps

        if not res.multi_face_landmarks:
            self.closed_since = self.yawn_since = self.nod_since = None
            self.drowsy_until = 0.0
            self.drowsy_since = None
            self.eye_open_since = None
            self.last_eye_sample_at = None
            self.eye_prob_hist.clear()
            self.cnn_ms = 0.0
            if self.face_missing_since is None:
                self.face_missing_since = now
            elif now - self.face_missing_since >= 1.0:
                # Nếu camera mất khuôn mặt lâu, không kế thừa số liệu cũ.
                self.eye_hist.clear()
                self.eye_closed_count = 0
                self.blink_times.clear()
                self.slow_closure_times.clear()
                self.perclos_high_since = None
                self.yawn_times.clear()
                self.nod_times.clear()
                self.last_pose = None
                self.baseline, self.yaw_baseline = None, None
                self.pitch_buf.clear()
                self.yaw_buf.clear()
                if now - self.face_missing_since >= 2.0:
                    self.open_ear_samples.clear()
                    self.open_ear_ref = None
            self.processing_ms = (time.perf_counter() - t_frame) * 1000.0
            self._push(dict(face=False, risk_level="Không thấy khuôn mặt"),
                       now, cfg, captured_mono, frame_seq, queue_ms)
            with self.lock:
                # Ngắt một frame FaceMesh không làm hộp biến mất rồi nháy lên.
                # Sau 0.5 s không thấy mặt mới thật sự xóa hộp.
                if (self.overlay is None or
                    time.perf_counter() - self.last_face_overlay_mono > OVERLAY_MISSING_GRACE_S):
                    self.overlay = {"finished_mono": time.perf_counter(),
                                    "capture_mono": captured_mono, "frame_seq": frame_seq,
                                    "boxes": (), "alarm": False}
                elif self.overlay is not None:
                    self.overlay = {**self.overlay, "alarm": False}
            return
        self.face_missing_since = None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ear = (eye_aspect_ratio(pts[RIGHT_EYE]) + eye_aspect_ratio(pts[LEFT_EYE])) / 2
        mar = mouth_aspect_ratio(pts)
        brightness = float(np.mean(gray))

        # Head pose thay đổi chậm hơn mắt: chạy mỗi vài frame để ưu tiên FPS cho camera/CNN.
        if self.frame_idx % POSE_EVERY_N_FRAMES == 0 or self.last_pose is None:
            self.last_pose = head_pose_angles(pts, w, h)
        pose = self.last_pose
        pitch = yaw = None
        if pose is not None:
            pitch, yaw, _ = pose

        if pitch is not None and yaw is not None and self.baseline is None:
            self.pitch_buf.append(pitch)
            self.yaw_buf.append(yaw)
            if len(self.pitch_buf) >= 45:
                self.baseline = float(np.median(self.pitch_buf))
                self.yaw_baseline = float(np.median(self.yaw_buf))

        pitch_dev = (pitch - self.baseline) if (pitch is not None and self.baseline is not None) else 0.0
        yaw_dev = (yaw - self.yaw_baseline) if (yaw is not None and self.yaw_baseline is not None) else 0.0

        # Nhận diện bàn tay trên frame thu nhỏ để phát hiện che miệng (Hand Occlusion)
        hand_res = None
        if getattr(self, "hands", None) is not None and cfg.get("detect_covered_yawn", True):
            try:
                hand_res = self.hands.process(rgb_small)
            except Exception:
                hand_res = None

        hand_near_mouth = False
        hand_box = None
        if hand_res and hand_res.multi_hand_landmarks:
            mouth_w = float(np.linalg.norm(pts[78] - pts[308]))
            mouth_cx = float((pts[78, 0] + pts[308, 0]) / 2.0)
            mouth_cy = float((pts[13, 1] + pts[14, 1]) / 2.0)
            
            mouth_x_min = float(min(pts[78, 0], pts[61, 0]) - 0.35 * mouth_w)
            mouth_x_max = float(max(pts[308, 0], pts[291, 0]) + 0.35 * mouth_w)
            mouth_y_min = float(pts[2, 1] - 0.15 * mouth_w)
            mouth_y_max = float(pts[152, 1] + 0.20 * mouth_w)
            
            for h_lm in hand_res.multi_hand_landmarks:
                h_pts = [(p.x * w, p.y * h) for p in h_lm.landmark]
                key_indices = [0, 4, 8, 9, 12, 16, 20]
                pts_in = sum(1 for idx in key_indices if mouth_x_min <= h_pts[idx][0] <= mouth_x_max and mouth_y_min <= h_pts[idx][1] <= mouth_y_max)
                d_center = min(np.hypot(h_pts[idx][0] - mouth_cx, h_pts[idx][1] - mouth_cy) for idx in [4, 8, 9, 12])
                
                if pts_in >= 2 or d_center <= 0.85 * mouth_w:
                    hand_near_mouth = True
                    hx_all = [h_pts[idx][0] for idx in key_indices if mouth_x_min <= h_pts[idx][0] <= mouth_x_max and mouth_y_min <= h_pts[idx][1] <= mouth_y_max]
                    hy_all = [h_pts[idx][1] for idx in key_indices if mouth_x_min <= h_pts[idx][0] <= mouth_x_max and mouth_y_min <= h_pts[idx][1] <= mouth_y_max]
                    if hx_all and hy_all:
                        hand_box = (max(0, int(min(hx_all)) - 10), max(0, int(min(hy_all)) - 10),
                                    min(w - 1, int(max(hx_all)) + 10), min(h - 1, int(max(hy_all)) + 10))
                    break
        self.hand_near_mouth = hand_near_mouth

        # Crop đúng một lần/frame; dùng chung cho kiểm tra chất lượng và CNN.
        right_crop_q = crop_eye(gray, pts[RIGHT_EYE])
        left_crop_q = crop_eye(gray, pts[LEFT_EYE])
        contrasts = [float(np.std(c * 255.0)) for c in (right_crop_q, left_crop_q) if c is not None]
        eye_contrast = float(np.mean(contrasts)) if contrasts else 0.0

        yaw_ok = self.yaw_baseline is None or abs(yaw_dev) <= cfg["yaw_limit"]
        light_ok = cfg["min_brightness"] <= brightness <= cfg["max_brightness"]
        contrast_ok = eye_contrast >= cfg["min_eye_contrast"]
        eye_valid = bool(yaw_ok and light_ok and contrast_ok)

        # Miệng chỉ đáng tin khi mặt không quay quá lớn và ánh sáng chấp nhận được.
        # Lưu ý: đây KHÔNG phải bộ phát hiện khẩu trang. Nếu đeo khẩu trang, nhánh ngáp
        # nên được xem là không đáng tin trong thực nghiệm và không được dùng để kết luận đơn lẻ.
        mouth_valid = bool(light_ok and (self.yaw_baseline is None or abs(yaw_dev) <= 30.0))

        quality_parts = [float(yaw_ok), float(light_ok), float(contrast_ok)]
        observation_quality = float(np.mean(quality_parts))

        if (eye_valid and 0.12 <= ear <= 0.40 and
            (self.open_ear_ref is None or ear >= 0.78 * self.open_ear_ref) and
            (now - self.session_started_at <= 3.5 or not self.eye_state_closed)):
            self.open_ear_samples.append(ear)
            if len(self.open_ear_samples) >= 8:
                est = float(np.percentile(self.open_ear_samples, 70))
                # Không cập nhật baseline xuống trong lúc mắt có thể đang nhắm.
                self.open_ear_ref = max(self.open_ear_ref or 0, est)

        closed = bool(self._eye_closed(right_crop_q, left_crop_q, ear, eye_valid=eye_valid))

        # PERCLOS chỉ dùng frame: đủ sáng/góc/tương phản + CNN đủ tự tin.
        # Khởi tạo baseline trước khi đưa frame vào PERCLOS để tránh nhiễu lúc camera/model vừa khởi động.
        eye_measure_valid = bool(
            eye_valid and self.eye_confident and (self.open_ear_ref is not None or getattr(self, "EYE_DECISION_POLICY", "legacy") == "cnn_primary_v1")
            and (now - self.session_started_at >= 1.5)
        )
        if eye_measure_valid:
            self.eye_hist.append((now, closed))
            self.eye_closed_count += int(closed)
        while self.eye_hist and now - self.eye_hist[0][0] > cfg["window"]:
            _, was_closed = self.eye_hist.popleft()
            self.eye_closed_count -= int(was_closed)

        if self.eye_hist:
            span = now - self.eye_hist[0][0]
            perclos = self.eye_closed_count / len(self.eye_hist)
        else:
            span, perclos = 0.0, 0.0
        perclos_ready = span >= min(10.0, cfg["window"] * 0.5) and len(self.eye_hist) >= 30

        # Ghi nhận nhắm mắt/chớp mắt dựa vào THỜI GIAN THỰC, không dựa số frame.
        # Một chớp mắt thông thường không bao giờ đủ điều kiện tự phát cảnh báo.
        if eye_measure_valid:
            if self.last_eye_sample_at is not None and now - self.last_eye_sample_at > 0.48:
                # Mất frame giữa chừng => không giả định nhắm liên tục.
                self.closed_since = None
            self.last_eye_sample_at = now
            if closed:
                self.closed_since = self.closed_since or now
                self.eye_open_since = None
            else:
                if self.closed_since is not None:
                    duration = now - self.closed_since
                    if 0.10 <= duration < 0.65:
                        self.blink_times.append(now)
                    elif 0.65 <= duration <= 8.0:
                        self.slow_closure_times.append(now)
                self.closed_since = None
                self.eye_open_since = self.eye_open_since or now
        else:
            self.closed_since = None
            self.eye_open_since = None
            self.last_eye_sample_at = None

        for events in (self.blink_times, self.slow_closure_times):
            while events and now - events[0] > 60.0:
                events.popleft()
        eye_closed_for = (max(0.0, now - self.closed_since)
                          if closed and self.closed_since is not None else 0.0)
        closed_long = bool(eye_measure_valid and closed and
                           eye_closed_for >= cfg["closed_sec"])

        # 1. Ngáp mở miệng truyền thống (Open Yawn)
        open_yawn_active = bool(mouth_valid and mar >= cfg["mar_thr"] and not hand_near_mouth)
        
        # 2. Ngáp che miệng (Covered-Mouth Yawn)
        # Kết hợp: Bàn tay che vùng miệng + Phản xạ nheo mắt/nhắm mắt khi ngáp (Eye squinting) + Đầu nhìn phía trước
        eye_squint = bool(
            closed or 
            ear <= 0.18 or 
            (self.open_ear_ref is not None and ear <= 0.76 * self.open_ear_ref) or
            (len(self.eye_prob_hist) > 0 and self.eye_prob_hist[-1] >= 0.48)
        )
        head_aligned = bool(self.yaw_baseline is None or abs(yaw_dev) <= 28.0)
        
        covered_yawn_active = bool(
            cfg.get("detect_covered_yawn", True) and
            hand_near_mouth and 
            eye_squint and 
            head_aligned
        )

        if open_yawn_active:
            self.yawn_recovered_since = None
            self.yawn_since = self.yawn_since or now
            if (now - self.yawn_since >= cfg["yawn_sec"] and
                not self.yawn_counted and now - self.last_yawn_event_at >= 3.0):
                self.yawn_times.append(now)
                self.last_yawn_event_at = now
                self.last_yawn_type = "open"
                self.open_yawns_count += 1
                self.yawn_counted = True
        elif not mouth_valid or hand_near_mouth:
            self.yawn_since = None
        elif mar <= 0.75 * cfg["mar_thr"]:
            self.yawn_recovered_since = self.yawn_recovered_since or now
            if now - self.yawn_recovered_since >= 0.30:
                self.yawn_since, self.yawn_counted = None, False
        else:
            self.yawn_since = None

        covered_sec = float(cfg.get("covered_yawn_sec", 1.2))
        if covered_yawn_active:
            self.covered_yawn_since = self.covered_yawn_since or now
            if (now - self.covered_yawn_since >= covered_sec and
                not self.covered_yawn_counted and now - self.last_yawn_event_at >= 3.0):
                self.yawn_times.append(now)
                self.last_yawn_event_at = now
                self.last_yawn_type = "covered"
                self.covered_yawns_count += 1
                self.covered_yawn_counted = True
        else:
            self.covered_yawn_since = None
            if not hand_near_mouth:
                self.covered_yawn_counted = False

        while self.yawn_times and now - self.yawn_times[0] > 60.0:
            self.yawn_times.popleft()

        yawn_active = open_yawn_active or covered_yawn_active

        # Gật: phải giữ nghiêng đủ lâu và quay về mốc tư thế trước lần kế tiếp.
        # Không đếm các dao động qua lại gần ngưỡng như nhiều lần gật.
        nod_active = bool(self.baseline is not None and yaw_ok and light_ok and
                          abs(pitch_dev) >= cfg["nod_deg"])
        if nod_active:
            self.nod_recovered_since = None
            self.nod_since = self.nod_since or now
            if (now - self.nod_since >= cfg["nod_sec"] and
                not self.nod_counted and now - self.last_nod_event_at >= 2.0):
                self.nod_times.append(now)
                self.last_nod_event_at = now
                self.nod_counted = True
        elif self.baseline is None or not yaw_ok:
            self.nod_since = None
        elif abs(pitch_dev) <= 0.55 * cfg["nod_deg"]:
            self.nod_recovered_since = self.nod_recovered_since or now
            if now - self.nod_recovered_since >= 0.30:
                self.nod_since, self.nod_counted = None, False
        else:
            self.nod_since = None
        while self.nod_times and now - self.nod_times[0] > 60.0:
            self.nod_times.popleft()

        recent_eye = [(t, c) for t, c in self.eye_hist
                      if 0 <= now - t <= EYE_RECENT_WINDOW_S]
        recent_eye_ready = bool(len(recent_eye) >= 12 and
                                recent_eye[-1][0] - recent_eye[0][0] >= 3.0)
        recent_eye_ratio = (sum(int(c) for _, c in recent_eye) / len(recent_eye)
                            if recent_eye_ready else 0.0)
        slow_count = sum(now - t <= EVENT_WINDOW_S for t in self.slow_closure_times)
        yawns_recent = sum(now - t <= EVENT_WINDOW_S for t in self.yawn_times)
        nods_recent = sum(now - t <= EVENT_WINDOW_S for t in self.nod_times)

        # Chỉ đánh giá PERCLOS nếu lượng dữ liệu đủ và cả 8 giây gần đây cũng có
        # dấu hiệu đóng mắt. Đòi ít nhất 2 lần đóng chậm hoặc tỷ lệ gần đây rất cao.
        eye_history_support = bool(
            perclos_ready and perclos >= cfg["perclos_thr"] and
            recent_eye_ready and
            recent_eye_ratio >= max(0.22, cfg["perclos_thr"] * 0.75) and
            (slow_count >= 2 or recent_eye_ratio >= 0.42)
        )
        if eye_history_support:
            self.perclos_high_since = self.perclos_high_since or now
        else:
            self.perclos_high_since = None
        perclos_confirmed = bool(
            self.perclos_high_since is not None and
            now - self.perclos_high_since >= PERCLOS_EVIDENCE_HOLD_S
        )
        recent_eye_support = bool(recent_eye_ready and recent_eye_ratio >= 0.18)
        repeated_eye = bool(slow_count >= 2 or recent_eye_support)
        recent_aux_event = bool(
            (self.yawn_times and now - self.yawn_times[-1] <= 5.0) or
            (self.nod_times and now - self.nod_times[-1] <= 5.0)
        )
        yawn_combination = bool(yawns_recent >= 2 and repeated_eye and recent_aux_event)
        nod_combination = bool(nods_recent >= 2 and repeated_eye and recent_aux_event)
        multi_aux_combination = bool(
            yawns_recent >= 2 and nods_recent >= 2 and recent_aux_event
        )

        eye_score, yawn_score, nod_score, total_score = self._compute_scores(
            cfg, perclos, eye_history_support, closed_long, slow_count
        )
        self.score = total_score

        # C1..C4 vẫn so sánh công bằng: tín hiệu phụ chỉ được dùng khi cấu hình có nó.
        if cfg["config"] == "C1":
            combo_evidence = False
        elif cfg["config"] == "C2":
            combo_evidence = yawn_combination
        elif cfg["config"] == "C3":
            combo_evidence = nod_combination
        else:
            combo_evidence = yawn_combination or nod_combination or multi_aux_combination

        open_confirmed = bool(
            self.eye_open_since is not None and now - self.eye_open_since >= 0.75
        )
        # Khi đã mở mắt liên tục, PERCLOS cũ không còn được tự tạo cảnh báo.
        # C4 vẫn có thể cảnh báo nếu *ngay gần đây* có hai loại sự kiện lặp lại.
        if open_confirmed:
            perclos_confirmed = False
            if not (cfg["config"] == "C4" and multi_aux_combination):
                combo_evidence = False

        evidence_alarm = bool((perclos_confirmed or combo_evidence) and
                              observation_quality >= (2.0 / 3.0))
        if evidence_alarm:
            self.drowsy_since = self.drowsy_since or now
        else:
            self.drowsy_since = None

        # Một lần nhắm mắt thật sự kéo dài >= closed_sec là trường hợp nguy cấp.
        # Các tổ hợp khác phải tồn tại ổn định và qua khoảng nghỉ sau cảnh báo.
        persistent_combo = bool(self.drowsy_since is not None and
                                now - self.drowsy_since >= ALARM_EVIDENCE_HOLD_S)
        cooldown_ok = now - self.last_alarm_ended_at >= ALARM_COOLDOWN_S
        new_alarm = bool(closed_long or (cooldown_ok and persistent_combo))
        if new_alarm:
            reasons = []
            if closed_long:
                reasons.append("Nhắm mắt liên tục đủ lâu")
            elif perclos_confirmed:
                reasons.append("PERCLOS cao kéo dài")
            if cfg["config"] in ("C2", "C4") and yawn_combination:
                reasons.append("Ngáp lặp lại kèm dấu hiệu mắt")
            if cfg["config"] in ("C3", "C4") and nod_combination:
                reasons.append("Gật đầu lặp lại kèm dấu hiệu mắt")
            if cfg["config"] == "C4" and multi_aux_combination:
                reasons.append("Ngáp và gật đầu đều lặp lại")
            self.reason = " · ".join(reasons) or "Nhiều dấu hiệu kéo dài"
            self.drowsy_until = now + 0.70

        if open_confirmed and not new_alarm and not (cfg["config"] == "C4" and multi_aux_combination):
            # Tắt trạng thái nhanh khi mở mắt; dữ kiện lịch sử vẫn giữ để phân tích.
            self.drowsy_until = 0.0
        drowsy = bool(now < self.drowsy_until)
        if self.prev_drowsy and not drowsy:
            self.last_alarm_ended_at = now
        self.prev_drowsy = drowsy

        if drowsy:
            risk_level = "Cảnh báo mạnh"
        elif closed or eye_history_support or repeated_eye or yawns_recent >= 2 or nods_recent >= 2:
            risk_level = "Nguy cơ buồn ngủ" if eye_history_support or repeated_eye else "Có dấu hiệu mệt"
        else:
            risk_level = "Tỉnh táo"
        if not drowsy and observation_quality < (2.0 / 3.0):
            risk_level = "Quan sát chưa đủ tin cậy"
        if not drowsy and open_confirmed and not combo_evidence and not eye_history_support:
            risk_level = "Tỉnh táo" if slow_count == 0 else "Có dấu hiệu mệt"

        self.risk_level = risk_level
        self.processing_ms = (time.perf_counter() - t_frame) * 1000.0

        self._push(dict(
            face=True, drowsy=drowsy, reason=self.reason if drowsy else "",
            risk_level=risk_level, score=total_score,
            perclos=perclos, perclos_ready=perclos_ready,
            ear=ear, mar=mar, pitch_dev=pitch_dev, yaw_dev=yaw_dev,
            yawns=len(self.yawn_times), nods=len(self.nod_times), closed=closed,
            eye_valid=eye_measure_valid, eye_confident=self.eye_confident, mouth_valid=mouth_valid,
            brightness=brightness, eye_contrast=eye_contrast,
            observation_quality=observation_quality,
            blinks=len(self.blink_times), slow_closures=len(self.slow_closure_times),
            recent_eye_ratio=recent_eye_ratio, eye_closed_for=eye_closed_for,
            alarm_evidence=self.reason if drowsy else ("Đang xác nhận dấu hiệu" if evidence_alarm else ""),
            eye_score=eye_score, yawn_score=yawn_score, nod_score=nod_score,
            yawn_type=self.last_yawn_type,
            open_yawns=self.open_yawns_count,
            covered_yawns=self.covered_yawns_count,
            hand_near_mouth=hand_near_mouth,
        ), now, cfg, captured_mono, frame_seq, queue_ms)

        drowsy = bool(self.stats.get("exp_audio_on"))  # Camera alarm follows notification state.

        # Chỉ gửi tọa độ nhỏ sang callback video; không gửi frame đã xử lý.
        def box_xyxy(points, padding=5):
            return (
                max(0, int(np.min(points[:, 0])) - padding),
                max(0, int(np.min(points[:, 1])) - padding),
                min(w - 1, int(np.max(points[:, 0])) + padding),
                min(h - 1, int(np.max(points[:, 1])) + padding),
            )

        # Hộp luôn có mặt khi face tracking đang tốt. Chỉ báo đỏ khi THẬT SỰ
        # có cảnh báo đã xác nhận; nhắm/ngáp đơn lẻ chỉ báo vàng (quan sát).
        eye_color = C_SKIP if not eye_valid else (C_BAD if drowsy else (C_WARN if closed else C_OK))
        mouth_pts = pts[[MOUTH["left"], MOUTH["top"], MOUTH["right"], MOUTH["bottom"]]]
        mouth_color = C_SKIP if not mouth_valid else (C_BAD if (drowsy and yawn_active) else
                                                     (C_WARN if yawn_active else C_OK))
        boxes_list = [
            (box_xyxy(pts[RIGHT_EYE]), eye_color),
            (box_xyxy(pts[LEFT_EYE]), eye_color),
            (box_xyxy(mouth_pts, padding=6), mouth_color),
        ]
        if hand_box is not None:
            boxes_list.append((hand_box, C_WARN if not drowsy else C_BAD))

        with self.lock:
            self.last_face_overlay_mono = time.perf_counter()
            self.overlay = {
                "finished_mono": time.perf_counter(),
                "capture_mono": captured_mono,
                "frame_seq": frame_seq,
                "alarm": bool(drowsy),
                "boxes": tuple(boxes_list),
            }

    def _push(self, data, now, cfg, captured_mono, frame_seq, queue_ms):
        # Gắn mốc thời gian cùng một frame, không lấy thời gian lúc UI render.
        finished_mono = time.perf_counter()
        finished_wall = time.time()
        with self.lock:
            base = self._empty_stats()
            base.update(data)
            base.update(
                left_closed_prob=self.left_closed_prob if data.get("face") else None,
                right_closed_prob=self.right_closed_prob if data.get("face") else None,
                left_eye_crop=self.left_eye_crop if data.get("face") else None,
                right_eye_crop=self.right_eye_crop if data.get("face") else None,
                fps=self.fps, camera_fps=self.camera_fps,
                processing_ms=self.processing_ms, cnn_ms=self.cnn_ms,
                frame_seq=frame_seq, sampled_at=now,
                inference_completed_at=finished_wall,
                inference_delay_ms=(finished_mono - captured_mono) * 1000.0,
                queue_ms=queue_ms,
                _capture_mono=captured_mono,
                source="CNN chính · EAR theo dõi" if self.cnn else ("Đang nạp" if not self._model_ready else "EAR"),
            )
            self.stats = base
            self._analysis_seq += 1
            self.history.append((now, base["perclos"], cfg["perclos_thr"], base["score"]))
            self._new_logs.append(dict(
                **{name: base.get(name) for name in DIAGNOSTIC_FIELDS},
                **{name: base.get(name) for name in EXPERIMENT_FIELDS},
                frame_seq=frame_seq,
                sample_time=round(now, 3), result_time=round(finished_wall, 3),
                config=cfg["config"],
                face=base["face"], drowsy=base["drowsy"],
                risk_level=base["risk_level"], score=base["score"],
                perclos=base["perclos"], perclos_ready=base["perclos_ready"],
                closed=base["closed"], ear=base["ear"], mar=base["mar"],
                pitch_dev=base["pitch_dev"], yaw_dev=base["yaw_dev"],
                yawns=base["yawns"], nods=base["nods"],
                eye_valid=base["eye_valid"], eye_confident=base["eye_confident"],
                mouth_valid=base["mouth_valid"], brightness=base["brightness"],
                eye_contrast=base["eye_contrast"],
                observation_quality=base["observation_quality"],
                eye_score=base["eye_score"], yawn_score=base["yawn_score"],
                nod_score=base["nod_score"],
                blinks=base["blinks"], slow_closures=base["slow_closures"],
                recent_eye_ratio=base["recent_eye_ratio"],
                eye_closed_for=base["eye_closed_for"],
                alarm_evidence=base["alarm_evidence"],
                left_p_closed=base["left_closed_prob"],
                right_p_closed=base["right_closed_prob"],
                camera_fps=base["camera_fps"], ai_fps=base["fps"],
                processing_ms=base["processing_ms"], cnn_ms=base["cnn_ms"],
                inference_delay_ms=base["inference_delay_ms"], queue_ms=queue_ms,
            ))


# ----------------------------------------------------------------------------
# Giao diện Streamlit
# ----------------------------------------------------------------------------
class DiagnosticProcessor(EyeDiagnosticsMixin, DrowsyProcessor):
    DIAGNOSTIC_MODEL_PATH = EYE_MODEL_PATH
    DIAGNOSTIC_BASE_PATH = APP_PATH
    DIAGNOSTIC_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/webcam_test"
    EYE_DECISION_POLICY = "cnn_primary_v1"


class ExperimentProcessor(ExperimentMixin, DiagnosticProcessor):
    DIAGNOSTIC_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/c1_c4_experiments/webcam"


DrowsyProcessor = ExperimentProcessor
st.set_page_config(page_title="DrowsyAlert", page_icon="🌙", layout="wide")

@st.cache_resource
def _load_test_model():
    loaded = load_eye_cnn()
    if loaded is None:
        raise RuntimeError(f"Không nạp được CNN. Đặt model đã bàn giao tại {TEAM_MODEL_PATH}. "
                           f"Đường dẫn đang dùng: {EYE_MODEL_PATH}")
    return loaded

_TEST_CNN = _load_test_model()

def load_eye_cnn():
    return _TEST_CNN

st.info("Thí nghiệm C1–C4: cùng CNN và PERCLOS theo thời gian. Nhánh nhắm liên tục 2 giây bật cảnh báo riêng, không chờ PERCLOS. "
        "C2/C4 thông báo từ 2 lần ngáp trong 40 giây, không cần dấu hiệu mắt; mỗi ngáp mới thông báo một lần. "
        "Nhánh PERCLOS cần ít nhất 10 giây mắt hợp lệ. Còi nhắm kéo dài dừng sau mở mắt ổn định 1,5 giây.")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
:root{--ink:#0E1A22;--panel:#15262F;--panel2:#10212A;--line:#25404D;--sand:#E8E4DA;--mute:#8FA3AD;
--ok:#3FD1A5;--warn:#F2B84B;--bad:#FF5A4E;--accent:#5FB3D9;}
html,body,[class*="css"],.stApp{font-family:'Be Vietnam Pro',system-ui,sans-serif;}
.stApp{background:radial-gradient(1200px 650px at 70% -10%,#173341 0%,var(--ink) 62%);color:var(--sand);}
header[data-testid="stHeader"]{background:rgba(11,20,26,.96);height:2.5rem;}
.block-container{padding-top:3.0rem!important;max-width:1560px;padding-left:.65rem;padding-right:.65rem;padding-bottom:.2rem;}
section[data-testid="stSidebar"]{background:#0B141A;border-right:1px solid var(--line);}
/* Sidebar: sát đỉnh, nhịp dọc gọn và đồng đều */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{
    height:1.65rem!important;
    min-height:1.65rem!important;
    padding:.05rem .55rem 0!important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
    padding-top:0!important;
    margin-top:-.20rem!important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{
    padding-top:0!important;
    padding-bottom:.25rem!important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:.50rem!important;}
section[data-testid="stSidebar"] .stMarkdown{margin:0!important;}
section[data-testid="stSidebar"] h3{
    font-size:.86rem!important;
    line-height:1.15!important;
    font-weight:700!important;
    margin:1rem 0 .35rem!important;
    color:var(--sand)!important;
}
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stCheckbox label p,
section[data-testid="stSidebar"] .stToggle label p{
    font-size:.72rem!important;
    line-height:1.15!important;
    color:#9cb0ba!important;
    margin-bottom:.02rem!important;
}
/* Slider: số đỏ 16px; toàn bộ mốc min/max 8px; không dùng margin âm để tránh đè label. */
section[data-testid="stSidebar"] [data-testid="stSlider"]{
    padding-top:.04rem!important;
    padding-bottom:0!important;
    margin-bottom:.02rem!important;
    overflow:visible!important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] > div{
    margin-top:0!important;
    margin-bottom:0!important;
    overflow:visible!important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"]{
    min-height:1.58rem!important;
    overflow:visible!important;
}
/* Giá trị hiện tại trên thumb. */
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] span,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] p,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div{
    font-size:16px!important;
    line-height:1!important;
}
/* Ghi đè riêng toàn bộ mốc min/max sau cùng để luôn là 8px. */
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"],
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] *,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child *{
    font-size:5px!important;
    line-height:5px!important;
    opacity:.38!important;
    overflow:visible!important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"],
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child{
    min-height:5px!important;
    margin-top:-1px!important;
    margin-bottom:0!important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]{
    width:.70rem!important;
    height:.70rem!important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"]{padding-bottom:.25rem!important;margin-bottom:.20rem!important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div{gap:.28rem!important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{align-items:flex-start!important;padding:.12rem 0!important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p{
    font-size:.86rem!important;line-height:1.40!important;color:var(--sand)!important;
    white-space:normal!important;overflow-wrap:anywhere!important;
}
section[data-testid="stSidebar"] button{min-height:2.25rem!important;font-size:.80rem!important;padding:.40rem .70rem!important;}
section[data-testid="stSidebar"] [data-testid="stToggle"] label p{font-size:.80rem!important;line-height:1.45!important;}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{padding:.75rem .85rem!important;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{font-size:.72rem!important;line-height:1.5!important;margin-top:.55rem!important;}

.hero{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:2px 10px;margin-bottom:8px;
background:linear-gradient(135deg,rgba(21,38,47,.96),rgba(14,31,40,.92));border:1px solid var(--line);border-radius:11px;
min-height:72px;padding:8px 14px;box-sizing:border-box;overflow:hidden;}
.hero-left{display:flex;align-items:center;gap:10px;min-width:0;height:100%;flex:1 1 auto;overflow:hidden;}
.hero-copy{display:flex;flex-direction:column;justify-content:center;gap:6px;min-width:0;overflow:hidden;flex:1 1 auto;}
.hero-mark{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.95rem;background:#0B141A;border:1px solid #2c5060;flex:0 0 auto;}
.hero h1{font-size:1.10rem;line-height:1;margin:0;padding:0;font-weight:700;letter-spacing:-.02em;white-space:nowrap;}
.hero-subtitle{margin:0;color:var(--mute);font-size:12px;max-width:850px;line-height:1.35;white-space:normal;overflow-wrap:anywhere;}
.hero-badges{display:flex;gap:4px;flex-wrap:nowrap;justify-content:flex-end;align-items:center;flex:0 0 auto;align-self:center;padding-top:0;}
.badge{padding:3px 7px;border-radius:999px;background:#0d2029;border:1px solid #294b5a;color:#b9d3df;font-size:.58rem;white-space:nowrap;}

.panel-stack{display:flex;flex-direction:column;gap:8px;width:100%;min-width:0;}
.metric-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:3px 12px;min-width:0;min-height:52px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;overflow:hidden;}
.metric-card .metric-head{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;column-gap:8px;min-width:0;}
.metric-card .metric-label{font-size:.76rem;color:var(--mute);font-weight:600;line-height:1.18;min-width:0;overflow-wrap:anywhere;}
.metric-card .metric-value{font-size:1.12rem;color:var(--sand);font-weight:700;line-height:1.08;text-align:right;white-space:nowrap;}
.metric-card .metric-meta{font-size:.70rem;color:var(--mute);margin-top:2px;line-height:1.30;overflow-wrap:anywhere;}
.signal-card{min-height:90px;}
.signal-card .metric-meta{font-size:.71rem;line-height:1.42;}
.component-card{min-height:64px;}
.component-value{text-align:left!important;margin-top:2px;}
.banner{border-radius:10px;padding:3px 13px;border:1px solid var(--line);background:var(--panel);min-height:68px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;overflow:hidden;}
.banner .big{font-size:1.22rem;font-weight:700;line-height:1.10;overflow-wrap:anywhere;}
.banner .why{color:var(--mute);margin-top:2px;font-size:.72rem;line-height:1.28;overflow-wrap:anywhere;}
.banner.ok{border-color:#1f6b57;background:linear-gradient(135deg,#123229,#15262F);}
.banner.ok .big{color:var(--ok);}
.banner.idle .big,.banner.warn .big{color:var(--warn);}
.banner.bad{border-color:var(--bad);background:linear-gradient(135deg,#4a1b1b,#2a1518);}
.banner.bad .big{color:var(--bad);}

.camera-card{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:4px;box-shadow:0 8px 24px rgba(0,0,0,.16);}
.camera-card div[data-testid="stVideo"]{border-radius:10px!important;overflow:hidden;}
.camera-card video{border-radius:10px!important;width:100%!important;height:44vh!important;max-height:365px!important;object-fit:contain!important;background:#091219;}
.section-title{font-size:.84rem;font-weight:600;margin:.15rem 0 .4rem;}
.compact-expander [data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:rgba(16,33,42,.55);margin-top:1px;transform:translateY(-3px);}
/* Ở viewport laptop, không chồng lên các nút Deploy/menu của Streamlit. */
@media (max-width:1500px){.hero-badges{display:none}}
@media (max-width:1100px){.hero-subtitle{white-space:normal}.camera-card video{height:40vh!important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-left"><div class="hero-mark">🌙</div><div class="hero-copy">'
    '<h1>DrowsyAlert</h1>'
    '<div class="hero-subtitle">Giám sát buồn ngủ theo thời gian thực bằng CNN mắt, PERCLOS, ngáp và tư thế đầu; ưu tiên giảm false positive trong điều kiện lái xe thực tế.</div>'
    '</div></div>'
    '<div class="hero-badges"><span class="badge">Realtime</span><span class="badge">CNN + MediaPipe</span><span class="badge">C1–C4</span></div>'
    '</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def beep_wav():
    import io
    import wave
    sr, dur = 22050, 1.2
    t = np.arange(int(sr * dur)) / sr
    env = np.where((t * 4).astype(int) % 2 == 0, 1.0, 0.25)
    data = (np.sin(2 * math.pi * 880 * t) * env * 0.6 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())
    return buf.getvalue()


with st.sidebar:
    st.markdown("### Cấu hình so sánh")
    config = st.radio(
        "Đặc trưng dùng để quyết định", list(CONFIG_LABELS),
        format_func=CONFIG_LABELS.get, index=3, label_visibility="collapsed",
    )
    st.markdown("### Điều khiển phiên")
    sound_on = st.toggle("Bật âm thanh cảnh báo", value=True)
    safety_audio = st.toggle("Cảnh báo nhắm kéo dài riêng", value=True)
    yawn_audio = st.toggle("Phát chuông cảnh báo khi ngáp", value=True)
    detect_covered_yawn = st.toggle("Phát hiện ngáp khi che miệng", value=True)
    recal = st.button("Đặt lại mốc tư thế đầu")
    reset_measurement = st.button("Reset phiên đo")

    st.markdown("### Cài đặt nâng cao")
    with st.expander("Ngưỡng chính", expanded=False):
        ear_thr = st.slider("EAR fallback", 0.10, 0.35, 0.21, 0.01)
        mar_thr = st.slider("MAR mở miệng", 0.30, 1.00, 0.60, 0.01)
        covered_yawn_sec = st.slider("Thời gian che miệng tính ngáp (giây)", 0.8, 2.5, 1.2, 0.1)
        nod_deg = st.slider("Góc gật/cúi đầu (độ)", 8, 40, 20)
        yaw_limit = st.slider("Góc quay đầu tối đa còn tin cậy (độ)", 20, 50, 35)

    with st.expander("PERCLOS và thời gian", expanded=False):
        window = st.slider("Cửa sổ PERCLOS (giây)", 10, 60, 30)
        perclos_thr = st.slider("Ngưỡng PERCLOS (%)", 10, 60, 30) / 100
        closed_sec = st.slider("Ngưỡng cảnh báo nhắm kéo dài riêng (giây)", 1.0, 4.0, 2.0, 0.5)
        yawn_sec = st.slider("Há miệng liên tục để tính ngáp (giây)", 0.8, 3.0, 1.5, 0.1)
        nod_sec = st.slider("Gật/cúi liên tục (giây)", 0.5, 3.0, 1.0, 0.5)

    with st.expander("Chất lượng quan sát", expanded=False):
        min_brightness = st.slider("Độ sáng tối thiểu", 10, 100, 35)
        max_brightness = st.slider("Độ sáng tối đa", 150, 250, 220)
        min_eye_contrast = st.slider("Tương phản mắt tối thiểu", 3, 30, 10)

    st.caption("C2/C4 thông báo từ 2 lần ngáp trong 40 giây, không cần dấu hiệu mắt. Nhánh gật đầu vẫn cần hỗ trợ mắt. Chớp mắt tự nhiên không tự báo động.")

cfg = dict(
    config=config, ear_thr=ear_thr, mar_thr=mar_thr, nod_deg=float(nod_deg),
    yaw_limit=float(yaw_limit), window=window, perclos_thr=perclos_thr,
    closed_sec=closed_sec, yawn_sec=yawn_sec, covered_yawn_sec=covered_yawn_sec, nod_sec=nod_sec,
    min_brightness=float(min_brightness), max_brightness=float(max_brightness),
    min_eye_contrast=float(min_eye_contrast), safety_audio=safety_audio,
    yawn_audio=yawn_audio, detect_covered_yawn=detect_covered_yawn,
)

if "log" not in st.session_state:
    st.session_state.log = []

# Một màn hình chính: thông tin cốt lõi hai bên, camera lớn ở trung tâm.
left_col, center_col, right_col = st.columns([1.02, 3.35, 1.18], gap="small")

with left_col:
    left_panel_ph = st.empty()

with center_col:
    st.markdown('<div class="camera-card">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="drowsyalert",
        video_processor_factory=DrowsyProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": 640, "height": 360, "frameRate": 30}, "audio": False},
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    right_panel_ph = st.empty()

experiment_ph = st.empty()
diagnostic_ph = st.empty()
sound_ph = st.empty()

# Phân tích sâu để dưới màn hình chính, mặc định thu gọn.
st.markdown('<div class="compact-expander">', unsafe_allow_html=True)
with st.expander("Phân tích chi tiết: PERCLOS · CNN input · chất lượng frame", expanded=False):
    analysis_left, analysis_right = st.columns([1.2, 1], gap="large")
    with analysis_left:
        st.markdown('<div class="section-title">PERCLOS theo thời gian</div>', unsafe_allow_html=True)
        chart_ph = st.empty()
    with analysis_right:
        st.markdown('<div class="section-title">CNN INPUT — ảnh 64×64 thực sự đưa vào model</div>', unsafe_allow_html=True)
        eye_cols = st.columns(2)
        with eye_cols[0]:
            left_eye_ph = st.empty()
        with eye_cols[1]:
            right_eye_ph = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# Hiển thị Sơ đồ kiến trúc & Lưu đồ thuật toán (Hình 3.1, Hình 3.2, Hình 3.3)
st.markdown('<div class="compact-expander" style="margin-top:6px;">', unsafe_allow_html=True)
with st.expander("📐 Sơ đồ kiến trúc & Lưu đồ thuật toán (Hình 3.1, Hình 3.2, Hình 3.3)", expanded=False):
    fig_dir = REPOSITORY_ROOT / "figures"
    f31_path = fig_dir / "hinh_3_1_kien_truc_tong_the.png"
    f32_path = fig_dir / "hinh_3_2_quy_trinh_small_cnn.png"
    f33_path = fig_dir / "hinh_3_3_quy_trinh_canh_bao.png"

    # Tự động sinh ảnh nếu chưa có trên đĩa
    if not (f31_path.is_file() and f32_path.is_file() and f33_path.is_file()):
        try:
            import generate_figures
            generate_figures.draw_figure_3_1()
            generate_figures.draw_figure_3_2()
            generate_figures.draw_figure_3_3()
        except Exception as _fig_err:
            st.error(f"Lỗi tạo sơ đồ: {_fig_err}")

    tab_31, tab_32, tab_33 = st.tabs([
        "🏛️ Hình 3.1: Kiến trúc tổng thể",
        "🧠 Hình 3.2: Quy trình Small CNN",
        "🚨 Hình 3.3: Lưu đồ quy trình cảnh báo",
    ])

    with tab_31:
        st.markdown("##### Hình 3.1. Sơ đồ kiến trúc tổng thể hệ thống DrowsyAlert")
        if f31_path.is_file():
            st.image(str(f31_path), use_container_width=True)
            with open(f31_path, "rb") as fp:
                st.download_button("📥 Tải ảnh Hình 3.1 (PNG chất lượng cao)", fp.read(),
                                   file_name="Hinh_3_1_Kien_truc_tong_the.png", mime="image/png", key="dl_f31")
        else:
            st.info("Chưa tìm thấy file ảnh Hình 3.1.")

    with tab_32:
        st.markdown("##### Hình 3.2. Quy trình phân loại trạng thái mắt bằng mạng Small CNN")
        if f32_path.is_file():
            st.image(str(f32_path), use_container_width=True)
            with open(f32_path, "rb") as fp:
                st.download_button("📥 Tải ảnh Hình 3.2 (PNG chất lượng cao)", fp.read(),
                                   file_name="Hinh_3_2_Quy_trinh_Small_CNN.png", mime="image/png", key="dl_f32")
        else:
            st.info("Chưa tìm thấy file ảnh Hình 3.2.")

    with tab_33:
        st.markdown("##### Hình 3.3. Lưu đồ thuật toán quy trình ra quyết định cảnh báo buồn ngủ")
        if f33_path.is_file():
            st.image(str(f33_path), use_container_width=True)
            with open(f33_path, "rb") as fp:
                st.download_button("📥 Tải ảnh Hình 3.3 (PNG chất lượng cao)", fp.read(),
                                   file_name="Hinh_3_3_Quy_trinh_canh_bao.png", mime="image/png", key="dl_f33")
        else:
            st.info("Chưa tìm thấy file ảnh Hình 3.3.")
st.markdown('</div>', unsafe_allow_html=True)

def _banner_html(s, playing):
    cls, big, why = presentation_state(s, playing)
    return f'<div class="banner {cls}"><div class="big">{big}</div><div class="why">{why}</div></div>'


def render_dashboard(s, playing):
    if not playing:
        def empty_card(label):
            return ('<div class="metric-card"><div class="metric-head">'
                    f'<div class="metric-label">{label}</div><div class="metric-value">—</div>'
                    '</div><div class="metric-meta">Chưa có dữ liệu</div></div>')
        left_labels = ("PERCLOS / ngưỡng", "PERCLOS", "Độ tin cậy quan sát",
                       "Tốc độ xử lý", "Frame mắt", "Tín hiệu nhanh")
        right_labels = ("CNN P(Closed) mắt trái", "CNN P(Closed) mắt phải", "Độ sáng",
                        "Tương phản mắt", "Tỷ lệ PERCLOS / số ngáp / số gật", "Nguồn trạng thái mắt")
        left_panel_ph.markdown('<div class="panel-stack left-panel">' +
                               ''.join(empty_card(label) for label in left_labels) + '</div>',
                               unsafe_allow_html=True)
        right_panel_ph.markdown('<div class="panel-stack right-panel">' + _banner_html(s, False) +
                                ''.join(empty_card(label) for label in right_labels) + '</div>',
                                unsafe_allow_html=True)
        diagnostic_ph.markdown('**Chẩn đoán mắt — Chưa có dữ liệu**')
        render_experiment(experiment_ph, dict(s, _ui_no_data=True))
        return
    q = int(round(s.get("quality_ratio", s.get("observation_quality", 0.0)) * 100))
    frame_state = "OK" if s.get("eye_valid", False) else "BỎ QUA"
    ready_text = ("Đang chờ khung AI mới" if s.get("stale") else
                  "Đủ dữ liệu" if s.get("perclos_ready", False) else "Đang khởi tạo")
    age = s.get("sample_age_ms", 0.0)
    fresh_text = (f"AI mới {age:.0f} ms" if s.get("live") else
                  "Đang chờ AI" if not s.get("frame_seq", 0) else f"AI trễ {age:.0f} ms")
    lp = s.get("left_closed_prob")
    rp = s.get("right_closed_prob")
    lp_text = "--" if lp is None else f"{lp * 100:.1f}%"
    rp_text = "--" if rp is None else f"{rp * 100:.1f}%"

    yawn_breakdown = f' (Mở: {s.get("open_yawns", 0)} · Che: {s.get("covered_yawns", 0)})' if s.get("yawns", 0) > 0 else ""
    hand_tag = " · <span style=\'color:var(--warn);font-weight:600\'>[Tay che miệng]</span>" if s.get("hand_near_mouth") else ""
    left_html = (
        '<div class="panel-stack left-panel">'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">PERCLOS / ngưỡng</div><div class="metric-value">{s["score"]:.2f}</div></div><div class="metric-meta">{s["risk_level"]}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">PERCLOS</div><div class="metric-value">{s["perclos"]*100:.1f}%</div></div><div class="metric-meta">{ready_text}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Độ tin cậy quan sát</div><div class="metric-value">{q}%</div></div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Tốc độ xử lý</div><div class="metric-value">{s.get("camera_fps",0):.0f} FPS</div></div><div class="metric-meta">Camera · AI {s["fps"]:.0f} FPS · {s.get("processing_ms",0):.0f} ms/frame<br>{fresh_text} · CNN {s.get("cnn_ms",0):.0f} ms</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Frame mắt</div><div class="metric-value">{frame_state}</div></div></div>'
        '<div class="metric-card signal-card"><div class="metric-label">Tín hiệu nhanh</div>'
        f'<div class="metric-meta">EAR <b style="color:var(--sand)">{s["ear"]:.2f}</b> · MAR <b style="color:var(--sand)">{s["mar"]:.2f}</b>{hand_tag}<br>'
        f'Pitch <b style="color:var(--sand)">{s["pitch_dev"]:+.0f}°</b> · Yaw <b style="color:var(--sand)">{s["yaw_dev"]:+.0f}°</b><br>'
        f'Ngáp/Gật do bộ dò ghi nhận <b style="color:var(--sand)">{s["yawns"]} / {s["nods"]}</b>{yawn_breakdown}<br>'
        f'Chớp mắt/nhắm chậm <b style="color:var(--sand)">{s.get("blinks",0)} / {s.get("slow_closures",0)}</b></div></div>'
        '</div>'
    )
    left_panel_ph.markdown(left_html, unsafe_allow_html=True)

    right_html = (
        '<div class="panel-stack right-panel">'
        + _banner_html(s, playing)
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">CNN P(Closed) mắt trái</div><div class="metric-value">{lp_text}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">CNN P(Closed) mắt phải</div><div class="metric-value">{rp_text}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Độ sáng</div><div class="metric-value">{s["brightness"]:.0f}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Tương phản mắt</div><div class="metric-value">{s["eye_contrast"]:.1f}</div></div></div>'
        + f'<div class="metric-card component-card"><div class="metric-label">Tỷ lệ PERCLOS / số ngáp / số gật</div><div class="metric-value component-value">{s["eye_score"]:.1f} / {s["yawn_score"]:.0f} / {s["nod_score"]:.0f}</div><div class="metric-meta">Các giá trị riêng, không cộng thành điểm</div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Nguồn trạng thái mắt</div><div class="metric-value">{s["source"]}</div></div></div>'
        + '</div>'
    )
    right_panel_ph.markdown(right_html, unsafe_allow_html=True)
    render_eye_diagnostics(diagnostic_ph, s)
    render_experiment(experiment_ph, s)


def render_eye_debug(s):
    left_crop = s.get("left_eye_crop")
    right_crop = s.get("right_eye_crop")
    lp = s.get("left_closed_prob")
    rp = s.get("right_closed_prob")

    if left_crop is None:
        left_eye_ph.caption("Mắt trái: chưa có ảnh")
    else:
        ptxt = "--" if lp is None else f"{lp * 100:.1f}%"
        left_eye_ph.image(left_crop, caption=f"Mắt trái · P(Closed)={ptxt}", clamp=True, width=220)

    if right_crop is None:
        right_eye_ph.caption("Mắt phải: chưa có ảnh")
    else:
        ptxt = "--" if rp is None else f"{rp * 100:.1f}%"
        right_eye_ph.image(right_crop, caption=f"Mắt phải · P(Closed)={ptxt}", clamp=True, width=220)


def render_chart(hist):
    if len(hist) < 2:
        return
    t0 = hist[0][0]
    df = pd.DataFrame(
        {
            "PERCLOS": [h[1] * 100 for h in hist],
            "Ngưỡng": [h[2] * 100 for h in hist],
        },
        index=[round(h[0] - t0, 1) for h in hist],
    )
    chart_ph.line_chart(df, height=210, color=["#3FD1A5", "#F2B84B"])


render_dashboard(dict(
    face=False, drowsy=False, reason="", risk_level="Chưa bật camera", score=0.0,
    perclos=0.0, perclos_ready=False, ear=0.0, mar=0.0, pitch_dev=0.0, yaw_dev=0.0,
    yawns=0, nods=0, blinks=0, slow_closures=0,
    closed=False, eye_valid=False, eye_confident=False, mouth_valid=False, brightness=0.0,
    eye_contrast=0.0, observation_quality=0.0, quality_ratio=0.0, eye_score=0.0, yawn_score=0.0, nod_score=0.0,
    left_closed_prob=None, right_closed_prob=None, fps=0.0, camera_fps=0.0, processing_ms=0.0,
    cnn_ms=0.0, frame_seq=0, live=False, stale=False, sample_age_ms=0.0, source="Đang nạp",
), False)

# ----------------------------------------------------------------------------
# Vòng cập nhật UI
# ----------------------------------------------------------------------------
last_beep = 0.0
reset_done = False
recal_done = False
last_debug = 0.0
last_chart = 0.0
last_dashboard = 0.0
last_seq = -1
last_stale = False
last_alarm_on = False

while ctx.state.playing:
    vp = ctx.video_processor
    if vp is None:
        time.sleep(UI_INTERVAL)
        continue

    vp.cfg = cfg

    if reset_measurement and not reset_done:
        vp.reset_session(reset_pose=False)
        st.session_state.log = []
        last_seq = -1
        reset_done = True

    if recal and not recal_done:
        vp.reset_baseline()
        recal_done = True

    s, hist, new_logs = vp.get_stats()
    if new_logs:
        st.session_state.log.extend(new_logs)
        if len(st.session_state.log) > MAX_SESSION_LOG_ROWS:
            del st.session_state.log[:len(st.session_state.log) - MAX_SESSION_LOG_ROWS]

    ui_now = time.perf_counter()
    seq = s.get("frame_seq", 0)
    changed = seq != last_seq
    stale_changed = s.get("stale", False) != last_stale
    dashboard_updated = False
    if (changed or stale_changed) and (ui_now - last_dashboard >= DASHBOARD_INTERVAL or stale_changed):
        render_dashboard(s, True)
        last_dashboard = ui_now
        last_seq = seq
        last_stale = s.get("stale", False)
        dashboard_updated = True

    # Debug/chart chỉ render với đúng snapshot vừa hiện trên dashboard.
    if dashboard_updated and changed and s.get("live") and ui_now - last_debug >= DEBUG_INTERVAL:
        render_eye_debug(s)
        last_debug = ui_now
    if dashboard_updated and changed and s.get("live") and ui_now - last_chart >= CHART_INTERVAL:
        render_chart(hist)
        last_chart = ui_now

    now = time.time()
    alert_on = bool(s.get("exp_audio_on") and s.get("live"))
    sound_active = bool(alert_on and sound_on)
    if sound_active and not last_alarm_on:
        wav_b64 = base64.b64encode(beep_wav()).decode("ascii")
        sound_ph.markdown(
            f'<audio autoplay loop style="display:none" data-nonce="{time.time_ns()}"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
    elif not sound_active and last_alarm_on:
        sound_ph.empty()
    last_alarm_on = sound_active
    time.sleep(UI_INTERVAL)

sound_ph.empty()  # Stop audio when camera stops.

# ----------------------------------------------------------------------------
# Xuất log
# ----------------------------------------------------------------------------
if st.session_state.log:
    df_log = pd.DataFrame(st.session_state.log)
    with right_col:
        st.download_button(
            "Tải log phiên chạy (CSV)",
            df_log.to_csv(index=False).encode("utf-8"),
            file_name="drowsyalert_log.csv",
            mime="text/csv",
        )
