"""Read-only decision instrumentation for the webcam test app."""
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo


DIAGNOSTIC_FIELDS = (
    "diag_eye_state", "diag_raw_cnn_state", "diag_mean_prob", "diag_smooth_prob",
    "diag_bilateral_closed", "diag_rule", "diag_rule_text", "diag_conflict",
    "diag_ear_ref", "diag_ear_ratio", "diag_ear_open_threshold", "diag_ear_closed_threshold",
    "diag_gate", "diag_reset_event", "diag_reset_count", "diag_last_reset_reason",
    "diag_sample_gap_s", "diag_closed_threshold_s", "diag_measurement_id",
    "diag_policy", "diag_legacy_ear_veto",
    "diag_alarm_count", "diag_last_alarm_at", "diag_last_alarm_reason",
)


class EyeDiagnosticsMixin:
    def __init__(self):
        stamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S_%f")
        self._diag_run = Path(self.DIAGNOSTIC_OUTPUT_ROOT) / f"run_{stamp}"
        self._diag_run.mkdir(parents=True, exist_ok=False)
        self._diag_log_error = ""
        super().__init__()
        config = dict(model_path=str(self.DIAGNOSTIC_MODEL_PATH),
                      model_sha256=hashlib.sha256(Path(self.DIAGNOSTIC_MODEL_PATH).read_bytes()).hexdigest(),
                      base_app=str(self.DIAGNOSTIC_BASE_PATH),
                      base_sha256=hashlib.sha256(Path(self.DIAGNOSTIC_BASE_PATH).read_bytes()).hexdigest(),
                      initial_config=self.cfg,
                      decision_source_sha256=hashlib.sha256(
                          Path(__file__).with_name("eye_decision.py").read_bytes()).hexdigest(),
                      decision_policy=getattr(self, "EYE_DECISION_POLICY", "legacy"),
                      decision_logic_changed=getattr(self, "EYE_DECISION_POLICY", "legacy") != "legacy")
        (self._diag_run / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset_session(self, reset_pose=False):
        with self.proc_lock:
            self._diag_eye = {}
            self._diag_previous_timer = None
            self._diag_previous_sample = None
            self._diag_reset_count = 0
            self._diag_last_reset_reason = "Chưa có bộ đếm bị reset"
            self._diag_measurement_id = getattr(self, "_diag_measurement_id", 0) + 1
            self._diag_alarm_active = False
            self._diag_alarm_count = 0
            self._diag_last_alarm_at = None
            self._diag_last_alarm_reason = ""
            super().reset_session(reset_pose=reset_pose)

    def _empty_stats(self):
        data = super()._empty_stats()
        data.update({name: None for name in DIAGNOSTIC_FIELDS})
        data.update(diag_eye_state="unknown", diag_raw_cnn_state="unknown",
                    diag_rule_text="Chưa có phép đo", diag_gate="Chưa có frame",
                    diag_reset_count=0, diag_last_reset_reason="--", diag_conflict=False)
        return data

    def _eye_closed(self, right_crop, left_crop, ear, eye_valid):
        closed = super()._eye_closed(right_crop, left_crop, ear, eye_valid)
        probabilities = [p for p in (self.left_closed_prob, self.right_closed_prob) if p is not None]
        mean = sum(probabilities) / len(probabilities) if probabilities else None
        history = list(self.eye_prob_hist)
        smooth = (0.75 * history[-1] + 0.25 * history[-2] if len(history) >= 2 else
                  history[-1] if history else None)
        bilateral = bool(eye_valid and len(probabilities) == 2 and min(probabilities) >= 0.60
                         and smooth is not None and smooth >= 0.72)
        ref = self.open_ear_ref
        if not eye_valid:
            rule, text = "quality_rejected", "Frame bị loại bởi chất lượng quan sát"
        elif self.cnn is None:
            rule, text = "ear_fallback", "Không có CNN, dùng EAR fallback"
        elif not probabilities:
            rule, text = "no_crop", "Không có crop mắt để chạy CNN"
        elif getattr(self, "EYE_DECISION_POLICY", "legacy") == "cnn_primary_v1":
            rule = self._cnn_rule
            text = {
                "cnn_closed": "CNN hai mắt xác nhận nhắm; EAR không phủ quyết",
                "cnn_hold_closed": "Duy trì nhắm bằng ngưỡng hysteresis CNN",
                "cnn_open": "CNN hai mắt xác nhận mở",
                "cnn_uncertain": "CNN chưa đủ chắc chắn: không tính thời gian nhắm",
                "cnn_missing_eye": "Thiếu một crop mắt: không tính thời gian nhắm",
            }[rule]
        elif ref is None:
            rule, text = "waiting_baseline", "Chưa đủ mẫu lấy mốc EAR"
        elif ear >= 0.82 * ref:
            rule, text = "ear_open_veto", "EAR >= 0.82 × mốc: luật cũ chọn mắt mở"
        elif ear <= 0.74 * ref and bilateral:
            rule, text = "cnn_and_ear_closed", "CNN hai mắt và EAR cùng xác nhận nhắm"
        elif smooth is not None and smooth <= 0.35 and ear > 0.74 * ref:
            rule, text = "cnn_open", "CNN và EAR đáp ứng điều kiện mắt mở"
        else:
            rule, text = "uncertain", "Chưa đủ điều kiện xác nhận mở hoặc nhắm"
        self._diag_eye = dict(
            diag_eye_state=("closed" if closed else "open") if self.eye_confident else "unknown",
            diag_raw_cnn_state="unknown" if mean is None else "closed" if mean >= 0.5 else "open",
            diag_mean_prob=mean, diag_smooth_prob=smooth if eye_valid else None,
            diag_bilateral_closed=bilateral, diag_rule=rule, diag_rule_text=text,
            diag_conflict=bool(bilateral and rule == "ear_open_veto"),
            diag_policy=getattr(self, "EYE_DECISION_POLICY", "legacy"),
            diag_legacy_ear_veto=bool(bilateral and ref and ear >= 0.82 * ref),
            diag_ear_ref=ref, diag_ear_ratio=ear / ref if ref else None,
            diag_ear_open_threshold=0.82 * ref if ref else None,
            diag_ear_closed_threshold=0.74 * ref if ref else None)
        return closed

    def _analyze_frame(self, *args, **kwargs):
        self._diag_previous_timer = self.closed_since
        self._diag_previous_sample = self.last_eye_sample_at
        self._diag_eye = {}
        return super()._analyze_frame(*args, **kwargs)

    def _push(self, data, now, cfg, captured_mono, frame_seq, queue_ms):
        previous = self._diag_previous_timer
        gap = now - self._diag_previous_sample if self._diag_previous_sample is not None else None
        if not data.get("face"):
            gate = "Không thấy khuôn mặt"
        elif not data.get("eye_valid"):
            reasons = []
            if not cfg["min_brightness"] <= data.get("brightness", 0) <= cfg["max_brightness"]:
                reasons.append("Ánh sáng ngoài ngưỡng")
            if data.get("eye_contrast", 0) < cfg["min_eye_contrast"]:
                reasons.append("Tương phản mắt thấp")
            if self.yaw_baseline is not None and abs(data.get("yaw_dev", 0)) > cfg["yaw_limit"]:
                reasons.append("Quay đầu quá ngưỡng")
            if self.open_ear_ref is None and getattr(self, "EYE_DECISION_POLICY", "legacy") == "legacy":
                reasons.append("Chưa có mốc EAR")
            if now - self.session_started_at < 1.5:
                reasons.append("Đang warm-up 1.5 giây")
            if not self.eye_confident:
                reasons.append("Quyết định mắt chưa đủ tin cậy")
            gate = " · ".join(reasons) or "Frame không hợp lệ cho phép đo mắt"
        elif not data.get("closed"):
            gate = "Luật quyết định trả về mắt mở"
        elif gap is not None and gap > 0.48:
            gate = "Khoảng cách hai mẫu > 0.48 giây: bắt đầu đếm lại"
        else:
            gate = "Đang đếm nhắm liên tục"
        reset = previous is not None and self.closed_since != previous
        if reset:
            self._diag_reset_count += 1
            self._diag_last_reset_reason = gate
        diagnostic = dict(self._diag_eye)
        audible_alarm = bool(data.get("exp_audio_on", data.get("drowsy")))
        if audible_alarm and not self._diag_alarm_active:
            self._diag_alarm_count += 1
            self._diag_last_alarm_at = now
            self._diag_last_alarm_reason = data.get("exp_audio_reason") or data.get("reason", "")
        self._diag_alarm_active = audible_alarm
        if not data.get("face"):
            diagnostic.update(diag_eye_state="unknown", diag_raw_cnn_state="unknown",
                              diag_rule="no_face", diag_rule_text=gate, diag_conflict=False)
        diagnostic.update(diag_gate=gate, diag_reset_event=reset,
                          diag_reset_count=self._diag_reset_count,
                          diag_last_reset_reason=self._diag_last_reset_reason,
                          diag_sample_gap_s=gap, diag_closed_threshold_s=cfg["closed_sec"],
                          diag_measurement_id=self._diag_measurement_id)
        diagnostic.update(diag_alarm_count=self._diag_alarm_count,
                          diag_last_alarm_at=self._diag_last_alarm_at,
                          diag_last_alarm_reason=self._diag_last_alarm_reason)
        return super()._push(dict(data, **diagnostic), now, cfg, captured_mono, frame_seq, queue_ms)

    def get_stats(self):
        data, history, logs = super().get_stats()
        if logs:
            path = self._diag_run / "frames.csv"
            try:
                exists = path.exists()
                with path.open("a", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(logs[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(logs)
                self._diag_log_error = ""
            except OSError as exc:
                self._diag_log_error = str(exc)
        if data.get("stale"):
            data.update(diag_eye_state="unknown", diag_gate="AI chưa có frame mới")
        data["diag_log_path"] = str(self._diag_run / "frames.csv")
        data["diag_log_error"] = self._diag_log_error
        return data, history, logs


def render_eye_diagnostics(placeholder, data):
    def fmt(value, digits=3):
        return "--" if value is None else f"{float(value):.{digits}f}"
    def text(value):
        return html.escape(str(value if value is not None else "--"))
    labels = {"open": "Mở", "closed": "Nhắm", "unknown": "Chưa xác định"}
    eye = labels.get(data.get("diag_eye_state"), "Chưa xác định")
    raw = labels.get(data.get("diag_raw_cnn_state"), "Chưa xác định")
    warning = ("CNN đủ điều kiện nhắm nhưng EAR đang phủ quyết."
               if data.get("diag_conflict") else "")
    cards = [
        ("CNN thô / quyết định cuối", f"{raw} / {eye}",
         f"P nhắm TB {fmt(data.get('diag_mean_prob'))} · làm mượt {fmt(data.get('diag_smooth_prob'))}"),
        ("EAR / mốc cá nhân (tham khảo)", f"{fmt(data.get('ear'))} / {fmt(data.get('diag_ear_ref'))}",
         f"Ngưỡng luật cũ: nhắm {fmt(data.get('diag_ear_closed_threshold'))} · mở {fmt(data.get('diag_ear_open_threshold'))}"),
        ("Nhắm liên tục / ngưỡng ghi riêng" if data.get("exp_policy") else "Nhắm liên tục / ngưỡng cảnh báo",
         f"{fmt(data.get('eye_closed_for', 0), 2)} / {fmt(data.get('diag_closed_threshold_s'), 2)} giây",
         f"Khoảng cách mẫu {fmt(data.get('diag_sample_gap_s'), 3)} giây"),
        ("Bộ đếm bị reset", str(data.get("diag_reset_count") or 0), data.get("diag_last_reset_reason", "--")),
    ]
    content = '<section style="padding:12px;border:1px solid #5FB3D9;border-radius:10px;margin:12px 0">'
    content += '<b>Chẩn đoán mắt — ' + text(data.get('diag_policy') or 'Chưa có phép đo') + '</b>'
    content += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:8px">'
    for title, value, detail in cards:
        content += f'<div><small>{text(title)}</small><div><b>{text(value)}</b></div><small>{text(detail)}</small></div>'
    content += '</div><p style="margin:8px 0">Luật: ' + text(data.get("diag_rule_text"))
    content += '<br>Phép đo: ' + text(data.get("diag_gate")) + '</p>'
    if data.get("diag_last_alarm_at"):
        stamp = datetime.fromtimestamp(float(data["diag_last_alarm_at"]), ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S")
        content += '<p style="color:#F2B84B"><b>Đã ghi nhận ' + text(data.get("diag_alarm_count"))
        content += ' đợt cảnh báo. Gần nhất ' + text(stamp) + ': '
        content += text(data.get("diag_last_alarm_reason")) + '</b></p>'
    if warning:
        content += '<p style="color:#F2B84B">' + text(warning) + '</p>'
    if data.get("diag_log_error"):
        content += '<p style="color:#FF5A4E">Không ghi được log: ' + text(data["diag_log_error"]) + '</p>'
    if data.get("diag_log_path"):
        content += '<small>Log tự lưu: ' + text(data["diag_log_path"]) + '</small>'
    content += '</section>'
    placeholder.markdown(content, unsafe_allow_html=True)
