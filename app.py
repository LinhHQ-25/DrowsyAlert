"""DrowsyAlert - phát hiện buồn ngủ realtime qua webcam.

Phiên bản này ưu tiên giảm false-positive trong tình huống lái xe thực tế:
- CNN mắt được làm mượt theo thời gian + hysteresis.
- Loại frame mắt kém tin cậy khỏi PERCLOS (quay đầu lớn, quá tối/sáng, crop quá ít tương phản).
- Ngáp/gật đầu là tín hiệu phụ; một lần đơn lẻ không đủ kết luận buồn ngủ.
- C1..C4 dùng cùng các đặc trưng nhưng quyết định theo điểm số theo thời gian.
- Có reset phiên đo để làm thực nghiệm sạch.

Chạy: python -m streamlit run app.py
"""

import base64
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

EYE_MODEL_PATH = "models/eye_cnn.keras"
EYE_INPUT = 64
CLOSED_IDX = 0

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
    nod_sec=1.0,
    min_brightness=35.0,
    max_brightness=220.0,
    min_eye_contrast=10.0,
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
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cnn = load_eye_cnn()
        self.history = collections.deque(maxlen=240)
        self.fps = 0.0
        self._last = time.time()
        self.reset_session(reset_pose=True)

    def reset_session(self, reset_pose=False):
        """Xóa trạng thái tích lũy để bắt đầu một phiên đo sạch."""
        with self.lock:
            self.eye_hist = collections.deque()       # (time, closed)
            self.eye_prob_hist = collections.deque(maxlen=5)
            self.eye_state_closed = False
            self.eye_confident = False
            self.session_started_at = time.time()
            self.closed_since = None
            self.yawn_since = None
            self.yawn_counted = False
            self.yawn_times = collections.deque()
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
            brightness=0.0, eye_contrast=0.0, observation_quality=0.0,
            eye_score=0.0, yawn_score=0.0, nod_score=0.0,
            left_closed_prob=None, right_closed_prob=None,
            left_eye_crop=None, right_eye_crop=None,
            fps=self.fps, source="CNN" if self.cnn else "EAR",
        )

    def reset_baseline(self):
        with self.lock:
            self.pitch_buf, self.yaw_buf = [], []
            self.baseline, self.yaw_baseline = None, None

    def get_stats(self):
        with self.lock:
            return dict(self.stats), list(self.history)

    def _eye_closed(self, gray, pts, ear, eye_valid):
        self.right_closed_prob = None
        self.left_closed_prob = None
        self.right_eye_crop = None
        self.left_eye_crop = None

        right_crop = crop_eye(gray, pts[RIGHT_EYE])
        left_crop = crop_eye(gray, pts[LEFT_EYE])
        if right_crop is not None:
            self.right_eye_crop = (np.clip(right_crop, 0, 1) * 255).astype(np.uint8)
        if left_crop is not None:
            self.left_eye_crop = (np.clip(left_crop, 0, 1) * 255).astype(np.uint8)

        if not eye_valid:
            # Frame không đạt chất lượng: giữ trạng thái hiển thị nhưng tuyệt đối không dùng cho PERCLOS.
            self.eye_confident = False
            return self.eye_state_closed

        if self.cnn is None:
            self.eye_state_closed = ear < self.cfg["ear_thr"]
            self.eye_confident = True
            return self.eye_state_closed

        valid = [("right", right_crop), ("left", left_crop)]
        valid = [(name, crop) for name, crop in valid if crop is not None]
        if not valid:
            return self.eye_state_closed

        x = np.stack([crop for _, crop in valid])[..., None]
        prob = np.asarray(self.cnn(x))
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

        mean_prob = float(np.mean(closed_prob))
        self.eye_prob_hist.append(mean_prob)
        smooth_prob = float(np.median(self.eye_prob_hist))

        # Vùng mơ hồ 35–65% KHÔNG được đưa vào PERCLOS.
        # Hysteresis chỉ dùng để giữ trạng thái hiển thị; quyết định đo chỉ nhận frame tự tin.
        if smooth_prob >= 0.65:
            self.eye_state_closed = True
            self.eye_confident = True
        elif smooth_prob <= 0.35:
            self.eye_state_closed = False
            self.eye_confident = True
        else:
            self.eye_confident = False

        return self.eye_state_closed

    def _compute_scores(self, cfg, perclos, perclos_ready, closed_long):
        """Điểm số theo thời gian. Ngáp/gật đơn lẻ không đủ tạo cảnh báo."""
        # Mắt là tín hiệu chính.
        eye_score = 0.0
        if perclos_ready:
            ratio = perclos / max(cfg["perclos_thr"], 1e-6)
            if ratio >= 1.35:
                eye_score += 2.5
            elif ratio >= 1.0:
                eye_score += 2.0
            elif ratio >= 0.75:
                eye_score += 1.0
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

    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        h, w = img.shape[:2]
        now = time.time()
        cfg = dict(self.cfg)
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        dt = now - self._last
        self._last = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps else inst_fps

        if not res.multi_face_landmarks:
            self.closed_since = self.yawn_since = self.nod_since = None
            self._push(dict(face=False, risk_level="Không thấy khuôn mặt"), now, cfg)
            cv2.putText(img, "Khong thay khuon mat", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, C_WARN, 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ear = (eye_aspect_ratio(pts[RIGHT_EYE]) + eye_aspect_ratio(pts[LEFT_EYE])) / 2
        mar = mouth_aspect_ratio(pts)
        brightness = float(np.mean(gray))

        pose = head_pose_angles(pts, w, h)
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

        # Đánh giá chất lượng frame.
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

        closed = bool(self._eye_closed(gray, pts, ear, eye_valid=eye_valid))

        # PERCLOS chỉ dùng frame: đủ sáng/góc/tương phản + CNN đủ tự tin.
        # Ba giây đầu là warm-up để tránh nhiễu lúc camera/model vừa khởi động.
        eye_measure_valid = bool(
            eye_valid and self.eye_confident and (now - self.session_started_at >= 3.0)
        )
        if eye_measure_valid:
            self.eye_hist.append((now, closed))
        while self.eye_hist and now - self.eye_hist[0][0] > cfg["window"]:
            self.eye_hist.popleft()

        if self.eye_hist:
            span = now - self.eye_hist[0][0]
            perclos = sum(c for _, c in self.eye_hist) / len(self.eye_hist)
        else:
            span, perclos = 0.0, 0.0
        perclos_ready = span >= min(10.0, cfg["window"] * 0.5) and len(self.eye_hist) >= 30

        # Nhắm kéo dài / microsleep cũng chỉ tính khi CNN đang tự tin.
        if eye_measure_valid:
            self.closed_since = (self.closed_since or now) if closed else None
        else:
            self.closed_since = None
        closed_long = (
            eye_measure_valid and closed and self.closed_since is not None
            and now - self.closed_since >= cfg["closed_sec"]
        )

        # Ngáp: phải há miệng liên tục đủ lâu. Một lần vẫn chỉ là bằng chứng yếu.
        if mouth_valid and mar > cfg["mar_thr"]:
            self.yawn_since = self.yawn_since or now
            if now - self.yawn_since >= cfg["yawn_sec"] and not self.yawn_counted:
                self.yawn_times.append(now)
                self.yawn_counted = True
        else:
            self.yawn_since, self.yawn_counted = None, False
        while self.yawn_times and now - self.yawn_times[0] > 60:
            self.yawn_times.popleft()

        # Gật đầu: phải vượt ngưỡng liên tục. Một lần vẫn chỉ là tín hiệu phụ.
        if self.baseline is not None and abs(pitch_dev) > cfg["nod_deg"]:
            self.nod_since = self.nod_since or now
            if now - self.nod_since >= cfg["nod_sec"] and not self.nod_counted:
                self.nod_times.append(now)
                self.nod_counted = True
        else:
            self.nod_since = None
            self.nod_counted = False
        while self.nod_times and now - self.nod_times[0] > 60:
            self.nod_times.popleft()

        eye_score, yawn_score, nod_score, total_score = self._compute_scores(
            cfg, perclos, perclos_ready, closed_long
        )
        self.score = total_score

        if total_score < 1.0:
            risk_level = "Tỉnh táo"
        elif total_score < 2.0:
            risk_level = "Có dấu hiệu mệt"
        elif total_score < 3.5:
            risk_level = "Nguy cơ buồn ngủ"
        else:
            risk_level = "Cảnh báo mạnh"

        # Điều kiện cảnh báo phụ thuộc cấu hình. Với C2–C4, PERCLOS cao một mình
        # không còn đủ để phát cảnh báo trừ khi rất nghiêm trọng. Điều này giảm false positive
        # do vài frame mắt bị nhận sai, quay đầu, mắt hí hoặc ánh sáng thay đổi.
        aux_score = yawn_score + nod_score
        severe_eye = bool(
            closed_long or
            (perclos_ready and perclos >= max(0.55, cfg["perclos_thr"] * 1.7))
        )
        if cfg["config"] == "C1":
            evidence_alarm = eye_score >= 2.0
        elif cfg["config"] == "C2":
            evidence_alarm = severe_eye or (eye_score >= 1.0 and yawn_score >= 1.0)
        elif cfg["config"] == "C3":
            evidence_alarm = severe_eye or (eye_score >= 1.0 and nod_score >= 1.5)
        else:  # C4
            evidence_alarm = severe_eye or (eye_score >= 1.0 and aux_score >= 1.5)

        score_alarm = bool(evidence_alarm and observation_quality >= (2.0 / 3.0))
        if score_alarm:
            self.drowsy_since = self.drowsy_since or now
        else:
            self.drowsy_since = None

        # Tổ hợp tín hiệu phải ổn định 2 giây; microsleep/nhắm kéo dài được cảnh báo ngay.
        persistent_alarm = (
            closed_long or
            (self.drowsy_since is not None and now - self.drowsy_since >= 2.0)
        )

        if persistent_alarm:
            reasons = []
            if closed_long:
                reasons.append("Nhắm mắt kéo dài")
            elif perclos_ready and perclos >= cfg["perclos_thr"]:
                reasons.append("PERCLOS cao")
            if cfg["config"] in ("C2", "C4") and len(self.yawn_times) >= 2:
                reasons.append("Ngáp lặp lại")
            if cfg["config"] in ("C3", "C4") and len(self.nod_times) >= 2:
                reasons.append("Gật đầu lặp lại")
            if not reasons:
                reasons.append("Nhiều dấu hiệu đồng thời")
            self.reason = " · ".join(reasons)
            self.drowsy_until = now + 3.0

        drowsy = now < self.drowsy_until
        if not drowsy and observation_quality < (2.0 / 3.0):
            risk_level = "Quan sát chưa đủ tin cậy"

        self.risk_level = risk_level

        self._push(dict(
            face=True, drowsy=drowsy, reason=self.reason if drowsy else "",
            risk_level=risk_level, score=total_score,
            perclos=perclos, perclos_ready=perclos_ready,
            ear=ear, mar=mar, pitch_dev=pitch_dev, yaw_dev=yaw_dev,
            yawns=len(self.yawn_times), nods=len(self.nod_times), closed=closed,
            eye_valid=eye_measure_valid, eye_confident=self.eye_confident, mouth_valid=mouth_valid,
            brightness=brightness, eye_contrast=eye_contrast,
            observation_quality=observation_quality,
            eye_score=eye_score, yawn_score=yawn_score, nod_score=nod_score,
        ), now, cfg)

        # Vẽ bounding box mỏng, không che chi tiết.
        if not eye_valid:
            eye_color = C_SKIP
        else:
            eye_color = C_BAD if drowsy else (C_WARN if closed else C_OK)

        def draw_box(points, box_color, padding=5, thickness=1):
            x_min = max(0, int(np.min(points[:, 0])) - padding)
            y_min = max(0, int(np.min(points[:, 1])) - padding)
            x_max = min(w - 1, int(np.max(points[:, 0])) + padding)
            y_max = min(h - 1, int(np.max(points[:, 1])) + padding)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), box_color, thickness, cv2.LINE_AA)

        draw_box(pts[RIGHT_EYE], eye_color, padding=5)
        draw_box(pts[LEFT_EYE], eye_color, padding=5)
        mouth_pts = pts[[MOUTH["left"], MOUTH["top"], MOUTH["right"], MOUTH["bottom"]]]
        mouth_color = C_SKIP if not mouth_valid else (C_WARN if mar > cfg["mar_thr"] else C_OK)
        draw_box(mouth_pts, mouth_color, padding=6)

        cv2.putText(img, f"PERCLOS {perclos * 100:4.1f}%", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, f"SCORE {total_score:.2f}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (240, 240, 240), 2, cv2.LINE_AA)
        if not eye_valid:
            cv2.putText(img, "EYE FRAME SKIPPED", (16, 94), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, C_WARN, 2, cv2.LINE_AA)
        if drowsy:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), C_BAD, 8)
            cv2.putText(img, "CANH BAO BUON NGU!", (16, 126), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, C_BAD, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _push(self, data, now, cfg):
        with self.lock:
            base = self._empty_stats()
            base.update(data)
            base.update(
                left_closed_prob=self.left_closed_prob,
                right_closed_prob=self.right_closed_prob,
                left_eye_crop=self.left_eye_crop,
                right_eye_crop=self.right_eye_crop,
                fps=self.fps,
                source="CNN" if self.cnn else "EAR",
            )
            self.stats = base
            self.history.append((now, base["perclos"], cfg["perclos_thr"], base["score"]))


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
header[data-testid="stHeader"]{background:transparent;height:.65rem;}
.block-container{padding-top:.32rem;max-width:1560px;padding-left:.65rem;padding-right:.65rem;padding-bottom:.2rem;}
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
min-height:54px;box-sizing:border-box;overflow:hidden;}
.hero-left{display:flex;align-items:center;gap:10px;min-width:0;height:100%;flex:1 1 auto;overflow:hidden;}
.hero-copy{display:flex;flex-direction:column;justify-content:center;gap:4px;min-width:0;overflow:hidden;flex:1 1 auto;}
.hero-mark{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.95rem;background:#0B141A;border:1px solid #2c5060;flex:0 0 auto;}
.hero h1{font-size:1.10rem;line-height:1;margin:0;padding:0;font-weight:700;letter-spacing:-.02em;white-space:nowrap;}
.hero-subtitle{margin:0;color:var(--mute);font-size:14px;max-width:900px;line-height:1.02;white-space:normal;overflow-wrap:anywhere;}
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
@media (max-width:1100px){.hero-badges{display:none}.hero-subtitle{white-space:normal}.camera-card video{height:40vh!important;}}
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
    st.markdown("### Ngưỡng chính")
    ear_thr = st.slider("EAR fallback", 0.10, 0.35, 0.21, 0.01)
    mar_thr = st.slider("MAR mở miệng", 0.30, 1.00, 0.60, 0.01)
    nod_deg = st.slider("Góc gật/cúi đầu (độ)", 8, 40, 20)
    yaw_limit = st.slider("Góc quay đầu tối đa còn tin cậy (độ)", 20, 50, 35)

    st.markdown("### PERCLOS và thời gian")
    window = st.slider("Cửa sổ PERCLOS (giây)", 10, 60, 30)
    perclos_thr = st.slider("Ngưỡng PERCLOS (%)", 10, 60, 30) / 100
    closed_sec = st.slider("Nhắm mắt liên tục (giây)", 1.0, 4.0, 2.0, 0.5)
    yawn_sec = st.slider("Há miệng liên tục để tính ngáp (giây)", 0.8, 3.0, 1.5, 0.1)
    nod_sec = st.slider("Gật/cúi liên tục (giây)", 0.5, 3.0, 1.0, 0.5)

    st.markdown("### Chất lượng quan sát")
    min_brightness = st.slider("Độ sáng tối thiểu", 10, 100, 35)
    max_brightness = st.slider("Độ sáng tối đa", 150, 250, 220)
    min_eye_contrast = st.slider("Tương phản mắt tối thiểu", 3, 30, 10)

    sound_on = st.toggle("Bật âm thanh cảnh báo", value=True)
    recal = st.button("Đặt lại mốc tư thế đầu")
    reset_measurement = st.button("Reset phiên đo")
    st.caption("Ngáp/gật là tín hiệu phụ; frame chất lượng thấp bị loại khỏi PERCLOS để giảm false positive.")

cfg = dict(
    config=config, ear_thr=ear_thr, mar_thr=mar_thr, nod_deg=float(nod_deg),
    yaw_limit=float(yaw_limit), window=window, perclos_thr=perclos_thr,
    closed_sec=closed_sec, yawn_sec=yawn_sec, nod_sec=nod_sec,
    min_brightness=float(min_brightness), max_brightness=float(max_brightness),
    min_eye_contrast=float(min_eye_contrast),
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
        media_stream_constraints={"video": {"width": 960, "height": 540}, "audio": False},
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    right_panel_ph = st.empty()

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

def _banner_html(s, playing):
    if not playing:
        cls, big, why = "idle", "Chưa bật camera", "Bấm START để bắt đầu."
    elif not s["face"]:
        cls, big, why = "idle", "Đang tìm khuôn mặt", "Ngồi trong vùng camera và đủ sáng."
    elif s["drowsy"]:
        cls, big, why = "bad", "Cảnh báo buồn ngủ", s["reason"]
    elif s["risk_level"] == "Quan sát chưa đủ tin cậy":
        cls, big, why = "warn", s["risk_level"], "Frame kém chất lượng đang được bỏ qua khỏi quyết định."
    elif s["risk_level"] == "Có dấu hiệu mệt":
        cls, big, why = "warn", s["risk_level"], "Chưa đủ bằng chứng để cảnh báo buồn ngủ."
    elif s["risk_level"] == "Nguy cơ buồn ngủ":
        cls, big, why = "warn", s["risk_level"], "Đang chờ tín hiệu tồn tại đủ lâu trước khi cảnh báo."
    else:
        cls, big, why = "ok", s["risk_level"], "Chưa ghi nhận tổ hợp dấu hiệu buồn ngủ."
    return f'<div class="banner {cls}"><div class="big">{big}</div><div class="why">{why}</div></div>'


def render_dashboard(s, playing):
    q = int(round(s.get("quality_ratio", s.get("observation_quality", 0.0)) * 100))
    frame_state = "OK" if s.get("eye_valid", False) else "BỎ QUA"
    ready_text = "Đủ dữ liệu" if s.get("perclos_ready", False) else "Đang khởi tạo"
    lp = s.get("left_closed_prob")
    rp = s.get("right_closed_prob")
    lp_text = "--" if lp is None else f"{lp * 100:.1f}%"
    rp_text = "--" if rp is None else f"{rp * 100:.1f}%"

    left_html = (
        '<div class="panel-stack left-panel">'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Drowsiness score</div><div class="metric-value">{s["score"]:.2f}</div></div><div class="metric-meta">{s["risk_level"]}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">PERCLOS</div><div class="metric-value">{s["perclos"]*100:.1f}%</div></div><div class="metric-meta">{ready_text}</div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Độ tin cậy quan sát</div><div class="metric-value">{q}%</div></div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Tốc độ xử lý</div><div class="metric-value">{s["fps"]:.0f} FPS</div></div></div>'
        f'<div class="metric-card"><div class="metric-head"><div class="metric-label">Frame mắt</div><div class="metric-value">{frame_state}</div></div></div>'
        '<div class="metric-card signal-card"><div class="metric-label">Tín hiệu nhanh</div>'
        f'<div class="metric-meta">EAR <b style="color:var(--sand)">{s["ear"]:.2f}</b> · MAR <b style="color:var(--sand)">{s["mar"]:.2f}</b><br>'
        f'Pitch <b style="color:var(--sand)">{s["pitch_dev"]:+.0f}°</b> · Yaw <b style="color:var(--sand)">{s["yaw_dev"]:+.0f}°</b><br>'
        f'Ngáp/Gật trong 60 giây <b style="color:var(--sand)">{s["yawns"]} / {s["nods"]}</b></div></div>'
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
    perclos=0.0, perclos_ready=False, ear=0.0, mar=0.0, pitch_dev=0.0, yaw_dev=0.0,
    yawns=0, nods=0, closed=False, eye_valid=False, eye_confident=False, mouth_valid=False, brightness=0.0,
    eye_contrast=0.0, observation_quality=0.0, quality_ratio=0.0, eye_score=0.0, yawn_score=0.0, nod_score=0.0,
    left_closed_prob=None, right_closed_prob=None, fps=0.0, source="EAR",
), False)

# ----------------------------------------------------------------------------
# Vòng cập nhật UI
# ----------------------------------------------------------------------------
last_beep = 0.0
reset_done = False
recal_done = False

while ctx.state.playing:
    vp = ctx.video_processor
    if vp is None:
        time.sleep(0.1)
        continue

    vp.cfg = cfg

    if reset_measurement and not reset_done:
        vp.reset_session(reset_pose=False)
        st.session_state.log = []
        reset_done = True

    if recal and not recal_done:
        vp.reset_baseline()
        recal_done = True

    s, hist = vp.get_stats()
    render_dashboard(s, True)
    render_eye_debug(s)
    render_chart(hist)

    st.session_state.log.append(dict(
        time=round(time.time(), 3),
        config=cfg["config"],
        face=s["face"],
        drowsy=s["drowsy"],
        risk_level=s["risk_level"],
        score=s["score"],
        perclos=s["perclos"],
        perclos_ready=s["perclos_ready"],
        closed=s["closed"],
        ear=s["ear"],
        mar=s["mar"],
        pitch_dev=s["pitch_dev"],
        yaw_dev=s["yaw_dev"],
        yawns=s["yawns"],
        nods=s["nods"],
        eye_valid=s["eye_valid"],
        eye_confident=s.get("eye_confident", False),
        mouth_valid=s["mouth_valid"],
        brightness=s["brightness"],
        eye_contrast=s["eye_contrast"],
        observation_quality=s["observation_quality"],
        eye_score=s["eye_score"],
        yawn_score=s["yawn_score"],
        nod_score=s["nod_score"],
        left_p_closed=s.get("left_closed_prob"),
        right_p_closed=s.get("right_closed_prob"),
    ))

    now = time.time()
    if s["drowsy"] and sound_on and now - last_beep > 4:
        wav_b64 = base64.b64encode(beep_wav()).decode("ascii")
        sound_ph.markdown(
            f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        last_beep = now
    elif not s["drowsy"] and now - last_beep > 2:
        sound_ph.empty()

    time.sleep(0.25)

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
