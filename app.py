"""DrowsyAlert - phát hiện buồn ngủ realtime qua webcam.

Phiên bản này ưu tiên giảm false-positive trong tình huống lái xe thực tế:
- CNN mắt được làm mượt theo thời gian + hysteresis.
- Loại frame mắt kém tin cậy khỏi PERCLOS (quay đầu lớn, quá tối/sáng, crop quá ít tương phản).
- Ngáp/gật đầu là tín hiệu phụ; một lần đơn lẻ không đủ kết luận buồn ngủ.
- C1..C4 dùng cùng các đặc trưng nhưng quyết định theo điểm số theo thời gian.
- Có reset phiên đo để làm thực nghiệm sạch.
- Luồng video độc lập; đồng bộ snapshot AI bằng ID frame và thời điểm nhận frame.
- UI chỉ vẽ snapshot mới; dữ liệu cũ được đánh dấu thay vì giả như vẫn realtime.
- Lưu log theo từng frame AI (không lấy mẫu lặp lại từ UI).
- Bounding box ổn định theo lần nhận diện gần nhất, không ẩn ở các frame xen kẽ.
- Cảnh báo dựa vào nhắm mắt kéo dài, PERCLOS kéo dài hoặc tổ hợp sự kiện lặp lại;
  ngáp/gật đơn lẻ chỉ là dữ kiện quan sát, không tự bật còi.

Chạy: python -m streamlit run app.py
"""

import os
import sys

# Đảm bảo bảng mã UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Tự động đồng bộ và chuyển sang Streamlit khi chạy bằng `python app.py` hoặc bấm nút Run ▶ trong VS Code
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_exists
        if not _st_exists():
            import subprocess
            os.environ["PYTHONUTF8"] = "1"
            _dir = os.path.dirname(os.path.abspath(__file__))
            _py_bin = sys.executable
            _venv_py = os.path.join(_dir, "venv", "Scripts", "python.exe")
            if os.path.exists(_venv_py) and os.path.abspath(_py_bin) != os.path.abspath(_venv_py):
                _py_bin = _venv_py
            cmd = [_py_bin, "-m", "streamlit", "run", os.path.abspath(__file__)]
            sys.exit(subprocess.call(cmd))
    except Exception:
        pass

# Tự động khắc phục lỗi đường dẫn tiếng Việt / non-ASCII của MediaPipe C++ trên Windows
try:
    if os.name == "nt":
        import ctypes
        import mediapipe.python.solution_base as _mp_sb

        def _get_short_path(path):
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024):
                return buf.value
            return str(path)

        _short_file = _get_short_path(_mp_sb.__file__)
        if _short_file and os.path.exists(_short_file):
            _orig_sb_init = _mp_sb.SolutionBase.__init__

            def _patched_sb_init(self, *args, **kwargs):
                orig_file = _mp_sb.__file__
                _mp_sb.__file__ = _short_file
                try:
                    return _orig_sb_init(self, *args, **kwargs)
                finally:
                    _mp_sb.__file__ = orig_file

            _mp_sb.SolutionBase.__init__ = _patched_sb_init
except Exception:
    pass

import base64
import collections
import math
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

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EYE_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "eye_cnn.keras")
EYE_INPUT = 64
CLOSED_IDX = 0

# Tối ưu realtime: MediaPipe chỉ cần ảnh nhỏ; camera hiển thị vẫn 640x360.
MP_PROCESS_WIDTH = 384
POSE_EVERY_N_FRAMES = 3
UI_INTERVAL = 0.10       # polling 10Hz nhẹ; chỉ render khi AI có frame mới
DASHBOARD_INTERVAL = 0.125  # mục tiêu 8Hz, giảm tải Streamlit/browser
DEBUG_INTERVAL = 1.00   # debug ở tab thu gọn, không tranh FPS với camera
CHART_INTERVAL = 2.00   # tránh redraw chart thường xuyên
STALE_AFTER_S = 1.80    # dung sai mạng/CPU (1.8s) để không ngắt còi và reset dữ liệu oan
OVERLAY_MAX_AGE_S = 1.50  # giữ hộp giữa các lần AI cập nhật
OVERLAY_MISSING_GRACE_S = 0.60  # FaceMesh rớt 1-2 khung: vẫn giữ hộp tránh nhấp nháy
OVERLAY_DIM_AFTER_S = 1.00  # AI chậm: hộp chuyển xám
ALARM_EVIDENCE_HOLD_S = 0.80  # tín hiệu kết hợp phải bền, không phát còi tức thì
ALARM_COOLDOWN_S = 5.0  # sau khi hết cảnh báo, tránh báo lại liên tục bởi dữ kiện cũ
PERCLOS_EVIDENCE_HOLD_S = 2.2  # PERCLOS cao kéo dài 2.2s là đủ tin cậy
EYE_RECENT_WINDOW_S = 8.0   # không quyết định chỉ từ PERCLOS quá khứ
EVENT_WINDOW_S = 40.0  # đếm ngáp/gật lặp lại trong 40 giây
MAX_SESSION_LOG_ROWS = 30000  # ~30-50 phút tuỳ AI FPS, giới hạn RAM
POSE_INIT_SAMPLES = 15  # chỉ cần 15 mẫu (~2s) để xác lập mốc góc đầu ban đầu

DEFAULT_CFG = dict(
    config="C4",
    ear_thr=0.21,
    mar_thr=0.60,
    nod_deg=20.0,
    yaw_limit=35.0,
    window=30,
    perclos_thr=0.30,
    closed_sec=1.5,
    yawn_sec=1.5,
    nod_sec=1.0,
    min_brightness=35.0,
    max_brightness=220.0,
    min_eye_contrast=3.0,
    distraction_enabled=True,
    distract_yaw_deg=35.0,
    distract_sec=3.0,
    glasses_mode="auto",
)

CONFIG_LABELS = {
    "C1": "C1 · PERCLOS / mắt",
    "C2": "C2 · PERCLOS + ngáp",
    "C3": "C3 · PERCLOS + tư thế đầu",
    "C4": "C4 · PERCLOS + ngáp + tư thế đầu",
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


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def crop_eye(gray, eye_pts, size=EYE_INPUT, enhance=False):
    """Crop vuông quanh mắt; output xám 64x64 [0,1], khớp train_eye_cnn.py.
    enhance=True: áp dụng CLAHE để làm dịu phản quang tròng kính và bóng gọng kính.
    """
    x0, y0 = eye_pts.min(axis=0)
    x1, y1 = eye_pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) * 0.75
    x0, x1, y0, y1 = int(cx - r), int(cx + r), int(cy - r), int(cy + r)
    H, W = gray.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = gray[y0:y1, x0:x1]
    if enhance:
        patch = _CLAHE.apply(patch)
    return cv2.resize(patch, (size, size)).astype(np.float32) / 255.0


def check_glasses_features(gray, pts):
    """Phát hiện gọng kính mắt dựa vào cầu nối sống mũi (nose bridge).
    - Không kính: Sống mũi là gờ dọc, cạnh dọc (Sobel X) chiếm ưu thế, cạnh ngang (Sobel Y) rất yếu,
      độ sáng tăng/giảm đơn điệu dọc theo sống mũi.
    - Có kính: Thanh cầu kính vắt ngang tạo cạnh ngang (Sobel Y) nổi bật và/hoặc vết trũng tối
      (dark valley) ở giữa sống mũi kẹp giữa 2 vùng da sáng hơn.
    """
    if pts is None or len(pts) < 400:
        return False, 0.0
    p_r = pts[133]  # khóe trong mắt phải
    p_l = pts[362]  # khóe trong mắt trái
    p_mid = pts[6]   # điểm chính giữa 2 mắt trên sống mũi
    dx = float(p_l[0] - p_r[0])
    if dx < 12.0:
        return False, 0.0

    w_box = int(dx * 0.48)
    h_box = int(max(8, dx * 0.40))
    cx, cy = int(p_mid[0]), int(p_mid[1])
    H, W = gray.shape[:2]
    x0, x1 = max(0, cx - w_box // 2), min(W, cx + w_box // 2)
    y0, y1 = max(0, cy - h_box // 2), min(H, cy + h_box // 2)
    if (x1 - x0) < 6 or (y1 - y0) < 6:
        return False, 0.0

    bridge_patch = gray[y0:y1, x0:x1]
    sobel_y = cv2.Sobel(bridge_patch, cv2.CV_32F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(bridge_patch, cv2.CV_32F, 1, 0, ksize=3)
    mean_sy = float(np.mean(np.abs(sobel_y)))
    mean_sx = float(np.mean(np.abs(sobel_x)))

    # Kiểm tra vết trũng tối (dark valley) theo chiều dọc (đặc trưng thanh ngang gọng kính)
    row_means = np.mean(bridge_patch, axis=1)
    min_idx = int(np.argmin(row_means))
    has_valley = False
    valley_depth = 0.0
    if 1 < min_idx < len(row_means) - 2:
        top_diff = row_means[0] - row_means[min_idx]
        bot_diff = row_means[-1] - row_means[min_idx]
        valley_depth = float(min(top_diff, bot_diff))
        if valley_depth >= 8.0:
            has_valley = True

    sy_sx_ratio = mean_sy / (mean_sx + 1e-6)
    # Cạnh ngang của gọng kính phải rõ nét và không bị lấn át bởi gờ sống mũi dọc
    has_bridge = bool((has_valley and mean_sy >= 26.0) or (mean_sy >= 48.0 and sy_sx_ratio >= 0.50))
    return has_bridge, mean_sy


_GLOBAL_CNN = None
_GLOBAL_CNN_LOADING = False
_GLOBAL_CNN_LOCK = threading.Lock()


def load_eye_cnn():
    if not os.path.exists(EYE_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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


def get_cached_eye_cnn_nonblocking():
    """Lấy CNN đã nạp hoặc kích hoạt nạp nền bất đồng bộ để camera không phải chờ."""
    global _GLOBAL_CNN, _GLOBAL_CNN_LOADING
    if _GLOBAL_CNN is not None:
        return _GLOBAL_CNN
    with _GLOBAL_CNN_LOCK:
        if _GLOBAL_CNN is not None:
            return _GLOBAL_CNN
        if not _GLOBAL_CNN_LOADING:
            _GLOBAL_CNN_LOADING = True

            def _loader():
                global _GLOBAL_CNN, _GLOBAL_CNN_LOADING
                try:
                    m = load_eye_cnn()
                    with _GLOBAL_CNN_LOCK:
                        _GLOBAL_CNN = m
                finally:
                    _GLOBAL_CNN_LOADING = False

            threading.Thread(target=_loader, name="Preload-CNN-Worker", daemon=True).start()
    return _GLOBAL_CNN


# Tự động kích hoạt nạp trước CNN ngay khi vừa mở ứng dụng
get_cached_eye_cnn_nonblocking()


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
            self.open_ear_ref = float(self.cfg.get("ear_thr", 0.21)) / 0.78
            self.eye_open_since = None
            self.eye_open_candidate_since = None
            self.eye_closed_count = 0
            self.eye_state_closed = False
            self.eye_confident = True
            self.session_started_at = time.time()
            self.face_missing_since = None
            self.closed_since = None
            self.yawn_since = None
            self.yawn_counted = False
            self.yawn_times = collections.deque()
            self.nod_since = None
            self.nod_counted = False
            self.nod_times = collections.deque()
            self.drowsy_since = None
            self.drowsy_until = 0.0
            self.distracted_since = None
            self.distracted_until = 0.0
            self.distracted = False
            self.distracted_counted = False
            self.distraction_times = collections.deque()
            self.mask_detected = False
            self.glasses_detected = False
            self.glasses_active = False
            self.glasses_hist = collections.deque(maxlen=25)
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
            distracted=False, distracted_for=0.0, distractions=0, mask_detected=False,
            glasses_detected=False, glasses_active=False, glasses_cfg=self.cfg.get("glasses_mode", "auto"),
            perclos=0.0, perclos_ready=False, ear=0.0, mar=0.0,
            pitch_dev=0.0, yaw_dev=0.0, yawns=0, nods=0,
            closed=False, eye_valid=False, eye_confident=False, mouth_valid=False,
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
            source="CNN + EAR" if self.cnn else ("Đang nạp" if not self._model_ready else "EAR"),
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
            self.cnn = get_cached_eye_cnn_nonblocking()
            self._model_ready = self.cnn is not None
            while True:
                if self.cnn is None:
                    self.cnn = get_cached_eye_cnn_nonblocking()
                    if self.cnn is not None:
                        self._model_ready = True
                # Nhận frame mới nhất: mọi frame cũ đến khi inference đang bận sẽ được bỏ qua.
                # Không tạo queue dài và không làm tăng thời gian chờ video.
                self._ready.wait(timeout=0.5)
                with self.lock:
                    item = self._pending
                    self._pending = None
                    self._ready.clear()
                    idle = time.perf_counter() - self._last_receive
                if item is None:
                    if idle > 90.0:
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
            self._pending = (img, now, t, frame_seq)
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
                    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), C_BAD, 4)
                    cv2.putText(img, "CANH BAO BUON NGU!", (12, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, .78, C_BAD, 2, cv2.LINE_AA)
                elif overlay.get("distracted") and age <= OVERLAY_DIM_AFTER_S:
                    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (40, 140, 255), 4)
                    cv2.putText(img, "CANH BAO MAT TAP TRUNG!", (12, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, .78, (40, 140, 255), 2, cv2.LINE_AA)
                elif overlay.get("closed_sec", 0.0) >= 0.35 and age <= OVERLAY_DIM_AFTER_S:
                    c_sec = overlay["closed_sec"]
                    cv2.putText(img, f"Nham mat: {c_sec:.1f}s", (12, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, .70, C_WARN, 2, cv2.LINE_AA)
                elif overlay.get("distracted_sec", 0.0) >= 0.8 and age <= OVERLAY_DIM_AFTER_S:
                    d_sec = overlay["distracted_sec"]
                    cv2.putText(img, f"Lech huong: {d_sec:.1f}s", (12, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, .70, (40, 140, 255), 2, cv2.LINE_AA)
                if overlay.get("glasses_active") and age <= OVERLAY_DIM_AFTER_S:
                    cv2.putText(img, "KINH MAT", (img.shape[1] - 110, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, (242, 184, 75), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _eye_closed(self, right_crop, left_crop, ear, eye_valid, glasses_active=False):
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
            fallback_thr = float(self.cfg.get("ear_thr", 0.21))
            base_ref = self.open_ear_ref if self.open_ear_ref is not None else (fallback_thr / 0.78)
            ear_threshold = (0.72 * base_ref) if glasses_active else (0.78 * base_ref)
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

        # Xác định mốc tham chiếu EAR (cá nhân hoặc fallback)
        fallback_thr = float(self.cfg.get("ear_thr", 0.21))
        # Người đeo kính có mốc mở mắt tự nhiên thấp hơn (thường ~0.18-0.22) do gọng kính che bớt
        default_ref = 0.22 if glasses_active else (fallback_thr / 0.78)
        base_ref = self.open_ear_ref if self.open_ear_ref is not None else default_ref

        min_p = float(np.min(closed_prob)) if len(closed_prob) else smooth_prob
        max_p = float(np.max(closed_prob)) if len(closed_prob) else smooth_prob
        mean_p = smooth_prob

        # Tỷ lệ EAR hiện tại so với lúc mắt mở bình thường
        ear_ratio = ear / max(base_ref, 0.01)

        if glasses_active:
            # === CHẾ ĐỘ ĐEO KÍNH MẮT ===
            # Gọng kính tạo viền tối khiến CNN có xu hướng dự đoán nhắm cao (false positive).
            # Do đó, chỉ dựa vào sự sụt giảm thực sự của độ mở mắt (Delta EAR):
            # 1. Khi mắt mở bình thường: dù CNN báo nhắm do gọng kính, nhưng EAR không giảm -> MẮT MỞ!
            if ear_ratio >= 0.76 or ear >= (0.82 * fallback_thr):
                self.eye_state_closed = False
                self.eye_confident = True
            # 2. Khi nhắm mắt thật (ngủ gật): mí mắt khép lại, EAR sụt giảm rõ rệt (> 32%)
            elif ear_ratio <= 0.62 or ear <= 0.12:
                self.eye_state_closed = True
                self.eye_confident = True
            # 3. Vùng trung gian (mắt lim dim / nửa nhắm nửa mở):
            # Cần cả CNN đồng thuận (mean_p >= 0.55) VÀ EAR có giảm (ear_ratio <= 0.72)
            else:
                self.eye_state_closed = bool(mean_p >= 0.55 and ear_ratio <= 0.72)
                self.eye_confident = True
        else:
            # === CHẾ ĐỘ TIÊU CHUẨN (KHÔNG ĐEO KÍNH) ===
            ear_close_thr = 0.78 * base_ref
            ear_open_thr = 0.86 * base_ref

            ear_closed_signal = (ear <= ear_close_thr) and (mean_p >= 0.35)
            cnn_closed_signal = (mean_p >= 0.58 and ear <= ear_open_thr) or (max_p >= 0.75 and ear <= ear_open_thr)
            deep_closed_signal = ear <= (0.68 * base_ref)

            clearly_open = (ear >= ear_open_thr) and (mean_p <= 0.55)

            if clearly_open:
                self.eye_state_closed = False
                self.eye_confident = True
            elif deep_closed_signal or ear_closed_signal or cnn_closed_signal:
                self.eye_state_closed = True
                self.eye_confident = True
            elif mean_p <= 0.30 and ear > ear_close_thr:
                self.eye_state_closed = False
                self.eye_confident = True
            else:
                # Vùng chuyển tiếp / nửa nhắm nửa mở
                self.eye_state_closed = bool(ear <= ear_close_thr or mean_p >= 0.50)
                self.eye_confident = True

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
            small = cv2.resize(img, (MP_PROCESS_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
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
            if self.face_missing_since is None:
                self.face_missing_since = now
            missing_dur = now - self.face_missing_since

            # Nếu chỉ rớt 1-2 khung hình ngắn (< 0.55s), giữ nguyên trạng thái nhắm mắt
            # không xóa timers đột ngột để chống đứt đoạn khi buồn ngủ / gật đầu
            if missing_dur >= 0.55:
                self.closed_since = self.yawn_since = self.nod_since = None
                self.drowsy_since = None
                self.eye_open_since = None
                self.last_eye_sample_at = None
                self.eye_prob_hist.clear()
                self.cnn_ms = 0.0

            if missing_dur >= 1.5:
                # Nếu camera mất khuôn mặt lâu (> 1.5s), mới dọn dẹp lịch sử tích lũy
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
                if missing_dur >= 2.5:
                    self.open_ear_samples.clear()
                    self.open_ear_ref = float(self.cfg.get("ear_thr", 0.21)) / 0.78

            self.processing_ms = (time.perf_counter() - t_frame) * 1000.0
            self._push(dict(face=False, risk_level="Không thấy khuôn mặt"),
                       now, cfg, captured_mono, frame_seq, queue_ms)
            with self.lock:
                if (self.overlay is None or
                    time.perf_counter() - self.last_face_overlay_mono > OVERLAY_MISSING_GRACE_S):
                    self.overlay = {"finished_mono": time.perf_counter(),
                                    "capture_mono": captured_mono, "frame_seq": frame_seq,
                                    "boxes": (), "alarm": False, "closed_sec": 0.0}
                elif self.overlay is not None:
                    keep_alarm = bool(now < self.drowsy_until)
                    self.overlay = {**self.overlay, "alarm": keep_alarm}
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

        if pitch is not None and yaw is not None:
            self.pitch_buf.append(pitch)
            self.yaw_buf.append(yaw)
            if self.baseline is None and len(self.pitch_buf) >= POSE_INIT_SAMPLES:
                self.baseline = float(np.median(self.pitch_buf))
                self.yaw_baseline = float(np.median(self.yaw_buf))
            elif self.baseline is not None and len(self.pitch_buf) > 60:
                self.pitch_buf.pop(0)
                self.yaw_buf.pop(0)
                self.baseline = float(np.median(self.pitch_buf))
                self.yaw_baseline = float(np.median(self.yaw_buf))

        pitch_dev = (pitch - self.baseline) if (pitch is not None and self.baseline is not None) else 0.0
        yaw_dev = (yaw - self.yaw_baseline) if (yaw is not None and self.yaw_baseline is not None) else 0.0

        # Nhận diện kính mắt (tự động hoặc theo cấu hình)
        glasses_cfg = cfg.get("glasses_mode", "auto")
        has_glasses_bridge, bridge_edge_score = check_glasses_features(gray, pts)
        self.glasses_hist.append(int(has_glasses_bridge))
        glasses_ratio = sum(self.glasses_hist) / len(self.glasses_hist) if self.glasses_hist else 0.0
        if glasses_ratio >= 0.65:
            self.glasses_detected = True
        elif glasses_ratio <= 0.25:
            self.glasses_detected = False

        if glasses_cfg == "on":
            glasses_active = True
        elif glasses_cfg == "off":
            glasses_active = False
        else:
            glasses_active = bool(self.glasses_detected)
        self.glasses_active = glasses_active

        # Crop đúng một lần/frame; dùng chung cho kiểm tra chất lượng và CNN (nếu đeo kính bật enhance CLAHE)
        right_crop_q = crop_eye(gray, pts[RIGHT_EYE], enhance=glasses_active)
        left_crop_q = crop_eye(gray, pts[LEFT_EYE], enhance=glasses_active)
        contrasts = [float(np.std(c * 255.0)) for c in (right_crop_q, left_crop_q) if c is not None]
        eye_contrast = float(np.mean(contrasts)) if contrasts else 0.0

        yaw_ok = self.yaw_baseline is None or abs(yaw_dev) <= (cfg["yaw_limit"] + 5.0)
        light_ok = (cfg["min_brightness"] - 10.0) <= brightness <= (cfg["max_brightness"] + 15.0)
        # Độ tương phản mắt: nếu mắt đang nhắm thì tương phản tự nhiên giảm.
        # Chỉ loại bỏ nếu crop hoàn toàn đen/hỏng (std < 1.0)
        contrast_ok = eye_contrast >= min(cfg["min_eye_contrast"], 1.5)
        eye_valid = bool(yaw_ok and light_ok and contrast_ok)

        # Kiểm tra khẩu trang / che miệng
        mouth_pts_box = pts[[MOUTH["left"], MOUTH["top"], MOUTH["right"], MOUTH["bottom"]]]
        mouth_crop_q = crop_eye(gray, mouth_pts_box, size=48)
        mouth_contrast = float(np.std(mouth_crop_q * 255.0)) if mouth_crop_q is not None else 0.0
        # Vùng miệng nếu bị che phẳng hoặc đeo khẩu trang, độ tương phản rất thấp (< 3.2)
        mask_detected = bool(mouth_crop_q is not None and mouth_contrast < 3.2 and light_ok)
        self.mask_detected = mask_detected

        # Miệng chỉ đáng tin khi mặt không quay quá lớn, ánh sáng chấp nhận được và không bị che / khẩu trang.
        mouth_valid = bool(light_ok and not mask_detected and (self.yaw_baseline is None or abs(yaw_dev) <= 30.0))

        quality_parts = [float(yaw_ok), float(light_ok), float(contrast_ok)]
        observation_quality = float(np.mean(quality_parts))

        # Tự động cập nhật baseline EAR mở mắt khi mắt mở rõ (CNN dự đoán nhắm thấp hoặc ở chế độ kính)
        # Hỗ trợ tự nhiên cho mọi người dùng (mắt to, mắt nhỏ, mắt một mí, đeo kính cận)
        mean_p_raw = float(np.mean(self.eye_prob_hist)) if self.eye_prob_hist else 0.5
        allow_ear_sample = (
            (glasses_active and (now - self.session_started_at <= 6.0 or not self.eye_state_closed))
            or (mean_p_raw <= 0.35 or not self.eye_state_closed)
            or (now - self.session_started_at <= 4.0)
        )
        if (eye_valid and 0.11 <= ear <= 0.45 and allow_ear_sample):
            self.open_ear_samples.append(ear)
            if len(self.open_ear_samples) >= 8:
                est = float(np.percentile(self.open_ear_samples, 75))
                # Cập nhật mượt mà (smooth exponential) để thích ứng cả khi người dùng đổi tư thế
                if self.open_ear_ref is None:
                    self.open_ear_ref = est
                else:
                    self.open_ear_ref = 0.85 * self.open_ear_ref + 0.15 * est

        closed = bool(self._eye_closed(right_crop_q, left_crop_q, ear, eye_valid=eye_valid, glasses_active=glasses_active))

        # PERCLOS chỉ dùng frame: đủ sáng/góc/tương phản + CNN đủ tự tin.
        # Khởi tạo baseline trước khi đưa frame vào PERCLOS để tránh nhiễu lúc camera/model vừa khởi động.
        # PERCLOS và thời gian nhắm: đo ngay khi frame hợp lệ và có tín hiệu
        eye_measure_valid = bool(
            eye_valid and self.eye_confident
            and (now - self.session_started_at >= 0.5)
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

        # Ghi nhận nhắm mắt dựa vào thời gian thực, có dung sai 0.30s chống rớt frame
        if eye_measure_valid:
            if self.last_eye_sample_at is not None and now - self.last_eye_sample_at > 0.85:
                self.closed_since = None
                self.eye_open_candidate_since = None
            self.last_eye_sample_at = now
            if closed:
                self.closed_since = self.closed_since or now
                self.eye_open_candidate_since = None
                self.eye_open_since = None
            else:
                if self.eye_open_candidate_since is None:
                    self.eye_open_candidate_since = now
                if now - self.eye_open_candidate_since >= 0.30:
                    if self.closed_since is not None:
                        duration = now - self.closed_since
                        if 0.10 <= duration < 0.65:
                            self.blink_times.append(now)
                        elif 0.65 <= duration <= 8.0:
                            self.slow_closure_times.append(now)
                    self.closed_since = None
                    self.eye_open_since = self.eye_open_since or now
        else:
            if self.last_eye_sample_at is not None and now - self.last_eye_sample_at > 0.85:
                self.closed_since = None
                self.eye_open_since = None
                self.eye_open_candidate_since = None

        for events in (self.blink_times, self.slow_closure_times):
            while events and now - events[0] > 60.0:
                events.popleft()
        eye_closed_for = (max(0.0, now - self.closed_since)
                          if closed and self.closed_since is not None else 0.0)
        closed_long = bool(eye_measure_valid and closed and
                           eye_closed_for >= cfg["closed_sec"])

        # Ngáp: MAR phải duy trì đủ lâu, miệng phải đóng rõ mới được đếm lần sau;
        # thêm cooldown để nói chuyện/há miệng không làm tăng sự kiện quá nhanh.
        yawn_active = bool(mouth_valid and mar >= cfg["mar_thr"])
        if yawn_active:
            self.yawn_recovered_since = None
            self.yawn_since = self.yawn_since or now
            if (now - self.yawn_since >= cfg["yawn_sec"] and
                not self.yawn_counted and now - self.last_yawn_event_at >= 3.0):
                self.yawn_times.append(now)
                self.last_yawn_event_at = now
                self.yawn_counted = True
        elif not mouth_valid:
            self.yawn_since = None
        elif mar <= 0.75 * cfg["mar_thr"]:
            self.yawn_recovered_since = self.yawn_recovered_since or now
            if now - self.yawn_recovered_since >= 0.30:
                self.yawn_since, self.yawn_counted = None, False
        else:
            self.yawn_since = None
        while self.yawn_times and now - self.yawn_times[0] > 60.0:
            self.yawn_times.popleft()

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

        # Kiểm tra mất tập trung (nhìn lệch hướng sang hai bên hoặc cúi nhìn điện thoại quá lâu)
        distract_enabled = cfg.get("distraction_enabled", True)
        distract_yaw_deg = float(cfg.get("distract_yaw_deg", 35.0))
        distract_sec = float(cfg.get("distract_sec", 3.0))

        head_away = bool(
            distract_enabled and (
                (self.yaw_baseline is not None and abs(yaw_dev) >= distract_yaw_deg) or
                (self.baseline is not None and pitch_dev <= -24.0)
            )
        )
        if head_away:
            self.distracted_since = self.distracted_since or now
            distracted_for = max(0.0, now - self.distracted_since)
            if distracted_for >= distract_sec:
                self.distracted_until = max(self.distracted_until, now + 1.80)
                if not self.distracted_counted:
                    self.distraction_times.append(now)
                    self.distracted_counted = True
        else:
            self.distracted_since = None
            self.distracted_counted = False

        while self.distraction_times and now - self.distraction_times[0] > 60.0:
            self.distraction_times.popleft()

        distracted = bool(now < self.distracted_until)
        distracted_for = (max(0.0, now - self.distracted_since)
                          if self.distracted_since is not None else 0.0)

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
        # dấu hiệu đóng mắt.
        eye_history_support = bool(
            perclos_ready and perclos >= cfg["perclos_thr"] and
            recent_eye_ready and
            recent_eye_ratio >= max(0.18, cfg["perclos_thr"] * 0.70) and
            (slow_count >= 1 or recent_eye_ratio >= max(0.26, cfg["perclos_thr"] * 0.85))
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

        # C1..C4 so sánh chuẩn hóa: tín hiệu phụ chỉ được dùng khi cấu hình cho phép.
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
        # C4 vẫn có thể cảnh báo nếu ngay gần đây có hai loại sự kiện lặp lại.
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

        # Nhắm mắt kéo dài hoặc gục đầu cúi sâu là trường hợp khẩn cấp
        nod_down_long = bool(
            self.baseline is not None and nod_active and self.nod_since is not None and
            (now - self.nod_since >= max(1.8, cfg["nod_sec"] * 1.5))
        )
        persistent_combo = bool(self.drowsy_since is not None and
                                now - self.drowsy_since >= ALARM_EVIDENCE_HOLD_S)
        cooldown_ok = now - self.last_alarm_ended_at >= ALARM_COOLDOWN_S

        # C1 chỉ mắt; C2 mắt + ngáp; C3 mắt + tư thế đầu; C4 toàn bộ
        nod_emergency = bool(nod_down_long and (cfg["config"] in ("C3", "C4")))
        new_alarm = bool(closed_long or nod_emergency or (cooldown_ok and persistent_combo))
        if new_alarm:
            reasons = []
            if closed_long:
                reasons.append("Nhắm mắt liên tục đủ lâu")
            if nod_emergency:
                reasons.append("Gục đầu cúi sâu kéo dài")
            elif perclos_confirmed:
                reasons.append("PERCLOS cao kéo dài")
            if cfg["config"] in ("C2", "C4") and yawn_combination:
                reasons.append("Ngáp lặp lại kèm dấu hiệu mắt")
            if cfg["config"] in ("C3", "C4") and nod_combination:
                reasons.append("Gật đầu lặp lại kèm dấu hiệu mắt")
            if cfg["config"] == "C4" and multi_aux_combination:
                reasons.append("Ngáp và gật đầu đều lặp lại")
            self.reason = " · ".join(reasons) or "Nhiều dấu hiệu kéo dài"
            self.drowsy_until = max(self.drowsy_until, now + 1.80)

        if open_confirmed and not new_alarm and not (cfg["config"] == "C4" and multi_aux_combination) and not closed:
            # Tắt trạng thái nhanh khi mở mắt; dữ kiện lịch sử vẫn giữ để phân tích.
            self.drowsy_until = 0.0
        drowsy = bool(now < self.drowsy_until)
        if self.prev_drowsy and not drowsy:
            self.last_alarm_ended_at = now
        self.prev_drowsy = drowsy

        if drowsy:
            risk_level = "Cảnh báo buồn ngủ"
        elif distracted:
            risk_level = "Cảnh báo mất tập trung"
        elif closed or eye_history_support or repeated_eye or yawns_recent >= 2 or nods_recent >= 2:
            risk_level = "Nguy cơ buồn ngủ" if eye_history_support or repeated_eye or closed else "Có dấu hiệu mệt"
        elif distracted_for >= 1.0:
            risk_level = "Đang nhìn lệch đường"
        else:
            risk_level = "Tỉnh táo"
        if not drowsy and not distracted and observation_quality < (2.0 / 3.0):
            risk_level = "Quan sát chưa đủ tin cậy"
        if not drowsy and not distracted and open_confirmed and not combo_evidence and not eye_history_support and not closed:
            risk_level = "Tỉnh táo" if slow_count == 0 else "Có dấu hiệu mệt"

        self.risk_level = risk_level
        self.processing_ms = (time.perf_counter() - t_frame) * 1000.0

        push_reason = self.reason if drowsy else ("Nhìn lệch hướng lái xe" if distracted else "")
        self._push(dict(
            face=True, drowsy=drowsy, reason=push_reason,
            distracted=distracted, distracted_for=distracted_for,
            distractions=len(self.distraction_times), mask_detected=self.mask_detected,
            glasses_detected=self.glasses_detected, glasses_active=self.glasses_active,
            glasses_cfg=glasses_cfg,
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
        ), now, cfg, captured_mono, frame_seq, queue_ms)

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
        mouth_color = C_SKIP if not mouth_valid else (C_BAD if drowsy and yawn_active else
                                                     (C_WARN if yawn_active else C_OK))
        with self.lock:
            self.last_face_overlay_mono = time.perf_counter()
            self.overlay = {
                "finished_mono": time.perf_counter(),
                "capture_mono": captured_mono,
                "frame_seq": frame_seq,
                "alarm": bool(drowsy),
                "distracted": bool(distracted),
                "closed_sec": eye_closed_for,
                "distracted_sec": distracted_for,
                "glasses_active": bool(self.glasses_active),
                "boxes": (
                    (box_xyxy(pts[RIGHT_EYE]), eye_color),
                    (box_xyxy(pts[LEFT_EYE]), eye_color),
                    (box_xyxy(mouth_pts, padding=6), mouth_color),
                ),
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
                source="CNN + EAR" if self.cnn else ("Đang nạp" if not self._model_ready else "EAR"),
            )
            self.stats = base
            self._analysis_seq += 1
            self.history.append((now, base["perclos"], cfg["perclos_thr"], base["score"]))
            self._new_logs.append(dict(
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
                distracted=base["distracted"],
                distracted_for=base["distracted_for"],
                mask_detected=base["mask_detected"],
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
st.set_page_config(page_title="DrowsyAlert", page_icon="🌙", layout="wide")

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
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:.06rem!important;}
section[data-testid="stSidebar"] .stMarkdown{margin:0!important;}
section[data-testid="stSidebar"] h3{
    font-size:.86rem!important;
    line-height:1.15!important;
    font-weight:700!important;
    margin:.30rem 0 .06rem!important;
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
section[data-testid="stSidebar"] [data-testid="stRadio"]{padding-bottom:.02rem!important;margin-bottom:.08rem!important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div{gap:.06rem!important;}
section[data-testid="stSidebar"] button{min-height:1.9rem!important;font-size:.72rem!important;padding:.16rem .52rem!important;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{font-size:.66rem!important;line-height:1.3!important;margin-top:.10rem!important;}

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
.banner.distract{border-color:#F28C28;background:linear-gradient(135deg,#3d220a,#1e1a17);}
.banner.distract .big{color:#F28C28;}

.summary-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-top:14px;box-shadow:0 6px 20px rgba(0,0,0,.2);}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:12px;}
.summary-item{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:9px 13px;}
.summary-item .label{font-size:.72rem;color:var(--mute);font-weight:600;}
.summary-item .val{font-size:1.15rem;font-weight:700;color:var(--sand);margin-top:2px;}

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


@st.cache_resource
def distract_wav():
    import io
    import wave
    sr, dur = 22050, 0.9
    t = np.arange(int(sr * dur)) / sr
    # Âm thanh cảnh báo mất tập trung: bíp kép tần số 660 Hz
    env = np.where((t * 6).astype(int) % 2 == 0, 1.0, 0.1)
    data = (np.sin(2 * math.pi * 660 * t) * env * 0.6 * 32767).astype(np.int16)
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
    st.markdown("### Ngưỡng chính")
    ear_thr = st.slider("EAR fallback", 0.10, 0.35, 0.21, 0.01)
    mar_thr = st.slider("MAR mở miệng", 0.30, 1.00, 0.60, 0.01)
    nod_deg = st.slider("Góc gật/cúi đầu (độ)", 8, 40, 20)
    yaw_limit = st.slider("Góc quay đầu tối đa còn tin cậy (độ)", 20, 50, 35)

    st.markdown("### PERCLOS và thời gian")
    window = st.slider("Cửa sổ PERCLOS (giây)", 10, 60, 30)
    perclos_thr = st.slider("Ngưỡng PERCLOS (%)", 10, 60, 30) / 100
    closed_sec = st.slider("Nhắm mắt liên tục (giây)", 0.5, 4.0, 1.5, 0.25)
    yawn_sec = st.slider("Há miệng liên tục để tính ngáp (giây)", 0.8, 3.0, 1.5, 0.1)
    nod_sec = st.slider("Gật/cúi liên tục (giây)", 0.5, 3.0, 1.0, 0.5)

    st.markdown("### Cảnh báo mất tập trung")
    distraction_on = st.toggle("Bật cảnh báo mất tập trung (nhìn lệch)", value=True)
    distract_sec = st.slider("Nhìn lệch đường liên tục (giây)", 1.5, 5.0, 3.0, 0.5)

    st.markdown("### Chất lượng quan sát & Kính mắt")
    glasses_option = st.selectbox(
        "👓 Chế độ kính mắt",
        ["Tự động (Khuyên dùng)", "Luôn bật (Đang đeo kính)", "Tắt (Không đeo kính)"],
        index=0,
        help="Tự động phát hiện hoặc ép bật khi đeo kính cận/loạn/kính râm để chống báo động nhầm do viền gọng kính."
    )
    glasses_mode_map = {
        "Tự động (Khuyên dùng)": "auto",
        "Luôn bật (Đang đeo kính)": "on",
        "Tắt (Không đeo kính)": "off",
    }
    glasses_mode = glasses_mode_map[glasses_option]
    min_brightness = st.slider("Độ sáng tối thiểu", 10, 100, 35)
    max_brightness = st.slider("Độ sáng tối đa", 150, 250, 220)
    min_eye_contrast = st.slider("Tương phản mắt tối thiểu", 1, 30, 3, 1)

    st.markdown("### Tiện ích điều khiển")
    sound_on = st.toggle("Bật âm thanh cảnh báo", value=True)
    test_sound = st.button("🔊 Thử loa / còi cảnh báo")
    recal = st.button("🎯 Đặt lại mốc tư thế đầu")
    reset_measurement = st.button("🔄 Reset phiên đo")
    st.caption("Cảnh báo khi nhắm mắt đủ lâu hoặc PERCLOS cao kéo dài; ngáp/gật phải lặp lại và kết hợp dấu hiệu khác. Chớp mắt tự nhiên không tự báo động.")

cfg = dict(
    config=config, ear_thr=ear_thr, mar_thr=mar_thr, nod_deg=float(nod_deg),
    yaw_limit=float(yaw_limit), window=window, perclos_thr=perclos_thr,
    closed_sec=closed_sec, yawn_sec=yawn_sec, nod_sec=nod_sec,
    min_brightness=float(min_brightness), max_brightness=float(max_brightness),
    min_eye_contrast=float(min_eye_contrast),
    distraction_enabled=distraction_on,
    distract_yaw_deg=float(yaw_limit),
    distract_sec=float(distract_sec),
    glasses_mode=glasses_mode,
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
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 360}, "frameRate": {"ideal": 30}}, "audio": False},
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    right_panel_ph = st.empty()

sound_ph = st.empty()
if test_sound:
    wav_b64 = base64.b64encode(beep_wav()).decode("ascii")
    sound_ph.markdown(
        f'<audio autoplay data-nonce="{time.time_ns()}" style="display:none"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )

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

def _banner_html(s, playing):
    if not playing:
        cls, big, why = "idle", "Chưa bật camera", "Bấm START để bắt đầu."
    elif s.get("stale"):
        cls, big, why = "warn", "AI đang chậm cập nhật", "Video vẫn tiếp tục; thông số cũ tạm thời không được dùng cảnh báo."
    elif not s.get("frame_seq", 0):
        cls, big, why = "idle", "Đang khởi động AI", "Camera đang hoạt động, hãy chờ mô hình khởi tạo."
    elif not s["face"]:
        cls, big, why = "idle", "Đang tìm khuôn mặt", "Ngồi trong vùng camera và đủ sáng."
    elif s.get("drowsy"):
        cls, big, why = "bad", "Cảnh báo buồn ngủ", s.get("reason", "Phát hiện buồn ngủ kéo dài")
    elif s.get("distracted"):
        cls, big, why = "distract", "Cảnh báo mất tập trung", "Tài xế đang nhìn lệch đường hoặc cúi đầu quá lâu!"
    elif s["risk_level"] == "Quan sát chưa đủ tin cậy":
        cls, big, why = "warn", s["risk_level"], "Frame kém chất lượng đang được bỏ qua khỏi quyết định."
    elif s["risk_level"] == "Có dấu hiệu mệt":
        cls, big, why = "warn", s["risk_level"], "Chưa đủ bằng chứng để cảnh báo buồn ngủ."
    elif s["risk_level"] == "Nguy cơ buồn ngủ":
        cls, big, why = "warn", s["risk_level"], "Đang chờ tín hiệu tồn tại đủ lâu trước khi cảnh báo."
    elif s["risk_level"] == "Đang nhìn lệch đường":
        cls, big, why = "warn", s["risk_level"], "Chú ý hướng nhìn thẳng về phía trước khi lái xe."
    else:
        cls, big, why = "ok", s["risk_level"], "Chưa ghi nhận tổ hợp dấu hiệu bất thường."
    return f'<div class="banner {cls}"><div class="big">{big}</div><div class="why">{why}</div></div>'


def render_dashboard(s, playing):
    q = int(round(s.get("quality_ratio", s.get("observation_quality", 0.0)) * 100))
    if not s.get("eye_valid", False):
        frame_state = '<span style="color:var(--mute)">BỎ QUA</span>'
    elif s.get("closed", False):
        c_for = s.get("eye_closed_for", 0.0)
        c_color = "var(--bad)" if c_for >= 1.0 else "var(--warn)"
        frame_state = f'<span style="color:{c_color}">NHẮM ({c_for:.1f}s)</span>'
    else:
        frame_state = '<span style="color:var(--ok)">MỞ</span>'
    ready_text = ("Đang chờ khung AI mới" if s.get("stale") else
                  "Đủ dữ liệu" if s.get("perclos_ready", False) else "Đang khởi tạo")
    age = s.get("sample_age_ms", 0.0)
    fresh_text = (f"AI mới {age:.0f} ms" if s.get("live") else
                  "Đang chờ AI" if not s.get("frame_seq", 0) else f"AI trễ {age:.0f} ms")
    lp = s.get("left_closed_prob")
    rp = s.get("right_closed_prob")
    lp_text = "--" if lp is None else f"{lp * 100:.1f}%"
    rp_text = "--" if rp is None else f"{rp * 100:.1f}%"

    distract_label = (
        f'<span style="color:#F28C28">LỆCH ({s.get("distracted_for",0.0):.1f}s)</span>'
        if s.get("distracted") or s.get("distracted_for", 0.0) >= 1.0
        else '<span style="color:var(--ok)">Bình thường</span>'
    )

    left_html = (
        '<div class="panel-stack left-panel">'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Drowsiness score</div><div class="metric-value">{s["score"]:.2f}</div></div><div class="metric-meta">{s["risk_level"]}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">PERCLOS</div><div class="metric-value">{s["perclos"]*100:.1f}%</div></div><div class="metric-meta">{ready_text}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Độ tin cậy quan sát</div><div class="metric-value">{q}%</div></div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Tốc độ xử lý</div><div class="metric-value">{s.get("camera_fps",0):.0f} FPS</div></div><div class="metric-meta">Camera · AI {s["fps"]:.0f} FPS · {s.get("processing_ms",0):.0f} ms/frame<br>{fresh_text} · CNN {s.get("cnn_ms",0):.0f} ms</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Frame mắt</div><div class="metric-value">{frame_state}</div></div></div>'
        '<div class="metric-card signal-card"><div class="metric-label">Tín hiệu nhanh</div>'
        f'<div class="metric-meta">EAR <b style="color:var(--sand)">{s["ear"]:.2f}</b> · MAR <b style="color:var(--sand)">{s["mar"]:.2f}</b><br>'
        f'Pitch <b style="color:var(--sand)">{s["pitch_dev"]:+.0f}°</b> · Yaw <b style="color:var(--sand)">{s["yaw_dev"]:+.0f}°</b><br>'
        f'Ngáp/Gật trong 60 giây <b style="color:var(--sand)">{s["yawns"]} / {s["nods"]}</b><br>'
        f'Chớp mắt/nhắm chậm <b style="color:var(--sand)">{s.get("blinks",0)} / {s.get("slow_closures",0)}</b><br>'
        f'Mất tập trung: {distract_label} · <b>{s.get("distractions",0)} lần</b></div></div>'
        '</div>'
    )
    left_panel_ph.markdown(left_html, unsafe_allow_html=True)

    mask_text = (
        '<span style="color:var(--warn)">Khẩu trang / Che</span>'
        if s.get("mask_detected")
        else '<span style="color:var(--ok)">Bình thường</span>'
    )

    glasses_mode_cfg = s.get("glasses_cfg", "auto")
    if glasses_mode_cfg == "on":
        glasses_text = '<span style="color:#4BB8F2">👓 Bật thủ công (Đang đeo kính)</span>'
    elif glasses_mode_cfg == "off":
        glasses_text = '<span style="color:var(--mute)">Tắt thủ công (Không kính)</span>'
    elif s.get("glasses_active"):
        glasses_text = '<span style="color:#4BB8F2">👓 Tự phát hiện: Có kính</span>'
    else:
        glasses_text = '<span style="color:var(--mute)">Không phát hiện kính</span>'

    right_html = (
        '<div class="panel-stack right-panel">'
        + _banner_html(s, playing)
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">CNN P(Closed) mắt trái</div><div class="metric-value">{lp_text}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">CNN P(Closed) mắt phải</div><div class="metric-value">{rp_text}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Độ sáng</div><div class="metric-value">{s["brightness"]:.0f}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Tương phản mắt</div><div class="metric-value">{s["eye_contrast"]:.1f}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Trạng thái kính mắt</div><div class="metric-value">{glasses_text}</div></div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Vùng miệng</div><div class="metric-value">{mask_text}</div></div></div>'
        + f'<div class="metric-card component-card"><div class="metric-label">Điểm thành phần</div><div class="metric-value component-value">{s["eye_score"]:.1f} + {s["yawn_score"]:.1f} + {s["nod_score"]:.1f}</div><div class="metric-meta">Mắt + ngáp + gật đầu</div></div>'
        + f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Nguồn trạng thái mắt</div><div class="metric-value">{s["source"]}</div></div></div>'
        + '</div>'
    )
    right_panel_ph.markdown(right_html, unsafe_allow_html=True)


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
    distracted=False, distracted_for=0.0, distractions=0, mask_detected=False,
    glasses_detected=False, glasses_active=False, glasses_cfg=glasses_mode,
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
last_distract_on = False

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
    alert_on = bool(s.get("drowsy") and s.get("live"))
    distract_on = bool(s.get("distracted") and s.get("live") and not alert_on)

    if alert_on and sound_on and (not last_alarm_on or now - last_beep >= 2.0):
        wav_b64 = base64.b64encode(beep_wav()).decode("ascii")
        sound_ph.markdown(
            f'<audio autoplay data-nonce="{time.time_ns()}" style="display:none"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        last_beep = now
    elif distract_on and sound_on and (not last_distract_on or now - last_beep >= 2.2):
        wav_b64 = base64.b64encode(distract_wav()).decode("ascii")
        sound_ph.markdown(
            f'<audio autoplay data-nonce="{time.time_ns()}" style="display:none"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        last_beep = now
    elif not alert_on and not distract_on and (last_alarm_on or last_distract_on) and (now - last_beep >= 1.25):
        sound_ph.empty()

    last_alarm_on = alert_on
    last_distract_on = distract_on
    time.sleep(UI_INTERVAL)

# ----------------------------------------------------------------------------
# Bảng tổng kết phiên & Xuất log
# ----------------------------------------------------------------------------
if st.session_state.log:
    df_log = pd.DataFrame(st.session_state.log)
    total_samples = len(df_log)
    drowsy_alarms = int(df_log["drowsy"].diff().gt(0).sum()) if "drowsy" in df_log else 0
    distract_alarms = int(df_log["distracted"].diff().gt(0).sum()) if "distracted" in df_log else 0
    avg_perclos = float(df_log["perclos"].mean() * 100) if "perclos" in df_log else 0.0
    max_score = float(df_log["score"].max()) if "score" in df_log else 0.0
    total_yawns = int(df_log["yawns"].max()) if "yawns" in df_log else 0
    total_nods = int(df_log["nods"].max()) if "nods" in df_log else 0

    t_start = df_log["sample_time"].min() if "sample_time" in df_log else 0
    t_end = df_log["sample_time"].max() if "sample_time" in df_log else 0
    duration_s = max(0, int(t_end - t_start))
    mins, secs = divmod(duration_s, 60)
    duration_str = f"{mins}p {secs:02d}s" if mins > 0 else f"{secs}s"

    if drowsy_alarms == 0 and distract_alarms == 0:
        verdict = '<span style="color:var(--ok);font-weight:700">🟢 An toàn xuất sắc</span>'
        verdict_desc = "Tài xế duy trì trạng thái tỉnh táo và tập trung quan sát đường."
    elif drowsy_alarms <= 2 and distract_alarms <= 2:
        verdict = '<span style="color:var(--warn);font-weight:700">🟡 Mệt mỏi nhẹ / Mất tập trung</span>'
        verdict_desc = "Có dấu hiệu buồn ngủ hoặc nhìn lệch đường. Nên uống nước hoặc nghỉ ngơi sớm."
    else:
        verdict = '<span style="color:var(--bad);font-weight:700">🔴 Nguy cơ cao — Cần nghỉ ngơi ngay</span>'
        verdict_desc = "Xuất hiện nhiều lần cảnh báo buồn ngủ hoặc mất tập trung kéo dài. Rất nguy hiểm!"

    badge_drowsy_color = "var(--bad)" if drowsy_alarms > 0 else "var(--sand)"
    badge_distract_color = "#F28C28" if distract_alarms > 0 else "var(--sand)"
    glasses_active_count = int(df_log["glasses_active"].sum()) if "glasses_active" in df_log else 0
    glasses_used_str = '<span style="color:#4BB8F2">Đã tối ưu</span>' if glasses_active_count > 0 else "Không dùng"

    st.markdown(
        f'''
        <div class="summary-card">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                    <h3 style="margin:0;font-size:1.05rem;color:var(--sand);">📊 Báo cáo tổng kết phiên giám sát</h3>
                    <div style="font-size:0.78rem;color:var(--mute);margin-top:2px;">Thời lượng: <b>{duration_str}</b> · {total_samples} khung hình AI · Đánh giá: {verdict}</div>
                </div>
                <div style="font-size:0.74rem;color:var(--sand);max-width:420px;text-align:right;">{verdict_desc}</div>
            </div>
            <div class="summary-grid">
                <div class="summary-item"><div class="label">🚨 Cảnh báo buồn ngủ</div><div class="val" style="color:{badge_drowsy_color}">{drowsy_alarms} lần</div></div>
                <div class="summary-item"><div class="label">👀 Cảnh báo mất tập trung</div><div class="val" style="color:{badge_distract_color}">{distract_alarms} lần</div></div>
                <div class="summary-item"><div class="label">📈 PERCLOS trung bình</div><div class="val">{avg_perclos:.1f}%</div></div>
                <div class="summary-item"><div class="label">👓 Kính mắt</div><div class="val">{glasses_used_str}</div></div>
                <div class="summary-item"><div class="label">⚡ Score tối đa</div><div class="val">{max_score:.2f}</div></div>
                <div class="summary-item"><div class="label">🥱 Số lần ngáp</div><div class="val">{total_yawns}</div></div>
                <div class="summary-item"><div class="label">🙇 Số lần gật đầu</div><div class="val">{total_nods}</div></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
    st.download_button(
        "📥 Tải toàn bộ log chi tiết của phiên (CSV)",
        df_log.to_csv(index=False).encode("utf-8"),
        file_name=f"drowsyalert_session_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)
