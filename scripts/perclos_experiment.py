"""Four controlled PERCLOS variants; prolonged closure is an independent output."""
import collections
from dataclasses import asdict, dataclass
import html
import math
from scripts.experiment_notification import ExperimentNotification, NotificationPolicy, NOTIFICATION_FIELDS
from scripts.alert_presentation import presentation_state

LABELS = {"C1": "CNN + PERCLOS", "C2": "C1 + MAR", "C3": "C1 + Head Pose",
          "C4": "C1 + Head Pose + MAR"}
POLICY_VERSION = "perclos_repeated_yawn_v2"


@dataclass(frozen=True)
class ExperimentPolicy:
    window_s: float = 30.0
    perclos_threshold: float = .30
    support_fraction: float = .60
    min_valid_s: float = 10.0
    min_coverage: float = .60
    max_sample_gap_s: float = .48
    event_window_s: float = 40.0
    event_fresh_s: float = 5.0
    min_aux_events: int = 2
    confirmation_s: float = 1.35
    release_s: float = .70

    def __post_init__(self):
        if not (0 < self.perclos_threshold <= 1 and 0 < self.support_fraction <= 1
                and 0 < self.min_coverage <= 1 and 0 < self.min_valid_s <= self.window_s
                and self.max_sample_gap_s > 0 and self.confirmation_s >= 0 and self.release_s >= 0
                and self.event_window_s >= self.event_fresh_s > 0 and self.min_aux_events >= 1):
            raise ValueError("Invalid experiment policy")


class PerclosExperiment:
    def __init__(self, policy=None):
        self.policy = policy or ExperimentPolicy()
        self.intervals = collections.deque()
        self.previous = None
        self.started = None
        self.pending = {key: None for key in LABELS}
        self.alarm_until = {key: float("-inf") for key in LABELS}
        self.last_reason = {key: "" for key in LABELS}

    def update(self, now, *, face, eye_valid, closed, yawn_times=(), nod_times=(),
               mouth_valid=False, pose_valid=False):
        p = self.policy
        if not math.isfinite(now) or (self.previous and now <= self.previous[0]):
            raise ValueError("Experiment timestamps must be finite and strictly increasing")
        if self.started is None:
            self.started = now
        valid = bool(face and eye_valid)
        gap = self.previous is not None and now-self.previous[0] > p.max_sample_gap_s
        if self.previous:
            before, before_valid, before_closed = self.previous
            # Do not bridge missing/uncertain frames. Use the previous observed state
            # over each accepted interval (left endpoint, zero-order hold).
            if valid and before_valid and not gap:
                self.intervals.append((before, now, before_closed))
        self.previous = (now, valid, bool(closed))
        cutoff = now-p.window_s
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        valid_s = closed_s = 0.0
        for start, end, was_closed in self.intervals:
            span = end-max(start, cutoff)
            valid_s += span
            closed_s += span*int(was_closed)
        observed_s = min(p.window_s, now-self.started)
        coverage = valid_s/observed_s if observed_s > 0 else 0.0
        perclos = closed_s/valid_s if valid_s > 0 else 0.0
        ready = valid_s+1e-9 >= p.min_valid_s and coverage+1e-9 >= p.min_coverage
        yawns = sorted(set(float(t) for t in yawn_times if 0 <= now-t <= p.event_window_s))
        nods = sorted(set(float(t) for t in nod_times if 0 <= now-t <= p.event_window_s))
        outputs = {}
        for config in LABELS:
            enabled = []
            if config in ("C2", "C4") and mouth_valid:
                enabled += yawns
            if config in ("C3", "C4") and pose_valid:
                enabled += nods
            high = ready and perclos+1e-12 >= p.perclos_threshold
            assisted = (ready and perclos+1e-12 >= p.perclos_threshold*p.support_fraction
                        and len(enabled) >= p.min_aux_events
                        and any(now-t <= p.event_fresh_s for t in enabled))
            repeated_yawn = bool(config in ('C2', 'C4') and face and mouth_valid
                                 and len(yawns) >= p.min_aux_events
                                 and now-yawns[-1] <= p.event_fresh_s)
            available = valid or repeated_yawn
            evidence = bool((valid and (high or assisted)) or repeated_yawn)
            if gap or not available:
                self.pending[config] = None
                self.alarm_until[config] = float("-inf")
            if evidence:
                if self.pending[config] is None:
                    self.pending[config] = now
                # Yawns have already passed the MAR duration/recovery detector.
                if repeated_yawn or now-self.pending[config]+1e-9 >= p.confirmation_s:
                    self.alarm_until[config] = now+p.release_s
                    self.last_reason[config] = ("Ngáp lặp lại trong 40 giây" if repeated_yawn else
                        "PERCLOS vượt ngưỡng" if high else
                        "PERCLOS có hỗ trợ + sự kiện " +
                        ("ngáp/gật" if config == "C4" else "ngáp" if config == "C2" else "gật"))
            else:
                self.pending[config] = None
            alarm = bool(available and now < self.alarm_until[config])
            outputs[config] = dict(drowsy=alarm, evidence=evidence,
                auxiliary_qualified=bool(assisted), auxiliary_latest=max(enabled) if enabled else None,
                repeated_yawn=repeated_yawn,
                repeated_yawn_event_s=(yawns[-1] if repeated_yawn and abs(now-yawns[-1]) <= 1e-6 else None),
                reason=self.last_reason[config] if alarm else "",
                enabled_aux_events=len(enabled),
                confirming_s=now-self.pending[config] if self.pending[config] is not None else 0.0)
        return dict(policy=POLICY_VERSION, perclos=perclos, valid_s=valid_s, closed_s=closed_s,
                    coverage=coverage, ready=ready, observable=valid,
                    yawn_count=len(yawns), nod_count=len(nods), variants=outputs)


EXPERIMENT_FIELDS = ("exp_policy", "exp_selected", "exp_valid_s", "exp_closed_s", "exp_coverage",
    "exp_perclos", "exp_ready", "exp_safety_alarm", "exp_safety_threshold_s", "exp_safety_audio_enabled",
    "exp_yawns", "exp_nods") + NOTIFICATION_FIELDS + tuple(
        f"exp_{config}_{field}" for config in LABELS
        for field in ("alarm", "evidence", "reason", "aux_events", "confirming_s", "aux_qualified", "aux_latest_s", "repeated_yawn")) + (
            "exp_recent_yawn", "exp_recent_nod", "exp_display_level", "exp_display_title", "exp_display_reason")


class ExperimentMixin:
    def reset_session(self, reset_pose=False):
        with self.proc_lock:
            self._experiment = None
            self._experiment_signature = None
            self._notification = None
            super().reset_session(reset_pose=reset_pose)

    def _push(self, data, now, cfg, captured_mono, frame_seq, queue_ms):
        # Selecting a display variant does not reset any of the four trackers.
        # Changing actual feature thresholds/window starts a new experiment segment.
        signature = tuple(sorted((k,v) for k,v in cfg.items() if k not in {"config","safety_audio"}))
        if self._experiment is None or signature != self._experiment_signature:
            self._experiment = PerclosExperiment(ExperimentPolicy(
                window_s=float(cfg["window"]), perclos_threshold=cfg["perclos_thr"],
                min_valid_s=min(10.0, float(cfg["window"])*.5)))
            self._experiment_signature = signature
            self._notification = ExperimentNotification(NotificationPolicy(rearm_closed_s=float(cfg["closed_sec"])))
            if hasattr(self, "_diag_run"):
                # Append a record when parameters actually change, not for each frame.
                import json
                import hashlib
                from pathlib import Path
                with (self._diag_run/"experiment_policy.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(sample_time=now, policy=asdict(self._experiment.policy),
                        version=POLICY_VERSION, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        notification_policy=asdict(self._notification.policy),
                        safety_audio_enabled=cfg.get('safety_audio',True),
                        notification_sha256=hashlib.sha256(Path(__file__).with_name('experiment_notification.py').read_bytes()).hexdigest(),
                        presentation_sha256=hashlib.sha256(Path(__file__).with_name('alert_presentation.py').read_bytes()).hexdigest(),
                        feature_config=cfg, prolonged_closure_included=False),ensure_ascii=False)+"\n")
        result = self._experiment.update(now, face=data.get("face",False),
            eye_valid=data.get("eye_valid",False), closed=data.get("closed",False),
            yawn_times=tuple(self.yawn_times), nod_times=tuple(self.nod_times),
            mouth_valid=data.get("mouth_valid",False),
            pose_valid=self.baseline is not None and self.yaw_baseline is not None)
        selected = result["variants"][cfg["config"]]
        safety = bool(data.get("eye_valid") and data.get("closed")
                      and data.get("eye_closed_for",0) >= cfg["closed_sec"])
        ratio = result["perclos"]/cfg["perclos_thr"]
        data = dict(data, perclos=result["perclos"], perclos_ready=result["ready"],
            drowsy=selected["drowsy"], reason=selected["reason"], score=ratio,
            risk_level=("Cảnh báo " + cfg["config"] if selected["drowsy"] else
                        "Chưa đủ quan sát" if not result["ready"] else
                        "Quan sát chưa đủ tin cậy" if not result["observable"] else "Chưa vượt luật cảnh báo"),
            alarm_evidence=selected["reason"] or ("Đang xác nhận" if selected["evidence"] else ""),
            eye_score=ratio, yawn_score=result["yawn_count"], nod_score=result["nod_count"],
            exp_policy=POLICY_VERSION, exp_selected=cfg["config"], exp_valid_s=result["valid_s"],
            exp_closed_s=result["closed_s"], exp_coverage=result["coverage"], exp_perclos=result["perclos"],
            exp_ready=result["ready"], exp_safety_alarm=safety, exp_safety_threshold_s=cfg["closed_sec"],
            exp_safety_audio_enabled=cfg.get('safety_audio',True),
            exp_yawns=result["yawn_count"], exp_nods=result["nod_count"])
        data['exp_recent_yawn'] = bool(self.yawn_times and 0 <= now - self.yawn_times[-1] <= 6)
        data['exp_recent_nod'] = bool(self.nod_times and 0 <= now - self.nod_times[-1] <= 6)
        for config, state in result["variants"].items():
            for name, value in dict(alarm=state["drowsy"], evidence=state["evidence"], reason=state["reason"],
                                    aux_events=state["enabled_aux_events"], confirming_s=state["confirming_s"],
                                    aux_qualified=state['auxiliary_qualified'], aux_latest_s=state['auxiliary_latest'],
                                    repeated_yawn=state['repeated_yawn']).items():
                data[f"exp_{config}_{name}"] = value
        latest_yawn_time = (self.yawn_times[-1] if self.yawn_times and (now - self.yawn_times[-1] <= 1.0 and self.yawn_times[-1] > self._notification.last_yawn_notified)
                            else selected['repeated_yawn_event_s'])
        data.update(self._notification.update(now, raw_alarm=selected["drowsy"],
            face=data.get('face', False), eye_valid=data.get('eye_valid', False), closed=data.get('closed', False),
            safety_alarm=safety and cfg.get('safety_audio', True),
            mouth_valid=data.get('mouth_valid', False) or data.get('hand_near_mouth', False),
            repeated_yawn_enabled=cfg.get('yawn_audio', True),
            repeated_yawn_time=latest_yawn_time,
            auxiliary_qualified=selected['auxiliary_qualified'] and not selected['repeated_yawn'],
            auxiliary_event_time=selected['auxiliary_latest'],
            auxiliary_kind=('ngáp lặp lại' if cfg['config'] == 'C2' else 'gật đầu lặp lại' if cfg['config'] == 'C3' else 'ngáp/gật lặp lại')))
        level,title,why=presentation_state(dict(data,frame_seq=frame_seq))
        data.update(exp_display_level=level,exp_display_title=title,exp_display_reason=why,risk_level=title)
        return super()._push(data,now,cfg,captured_mono,frame_seq,queue_ms)


def render_experiment(placeholder, data):
    if data.get('_ui_no_data'):
        rows = ''.join(f'<tr><td>{config}</td><td>{html.escape(label)}</td>'
                       '<td>Chưa có dữ liệu</td><td>—</td></tr>' for config, label in LABELS.items())
        placeholder.markdown(
            '<section style="padding:14px;border:1px solid #55aabb;border-radius:8px;margin:12px 0">'
            '<b>So sánh đồng thời C1–C4 · PERCLOS theo thời gian</b>'
            '<p>Chưa có dữ liệu. Bấm START để bắt đầu quan sát.</p>'
            '<table style="width:100%;text-align:left"><tr><th>Mã</th><th>Tín hiệu</th>'
            '<th>Kết quả mô hình (cửa sổ)</th><th>Sự kiện phụ hợp lệ</th></tr>' + rows +
            '</table><p>Nhắm kéo dài: — · Âm thanh: chưa có dữ liệu · Đợt phát: —</p></section>',
            unsafe_allow_html=True)
        return
    rows = []
    for config,label in LABELS.items():
        state = ("Vượt luật — ngáp lặp lại" if data.get(f'exp_{config}_alarm') and data.get(f'exp_{config}_repeated_yawn') else
                 "Chờ đủ quan sát" if not data.get("exp_ready") else
                 "Không đủ tin cậy" if not data.get("eye_valid") or data.get("stale") else
                 "Vượt luật" if data.get(f"exp_{config}_alarm") else
                 "Đang xác nhận" if data.get(f"exp_{config}_evidence") else "Chưa vượt luật")
        rows.append(f"<tr><td>{config}</td><td>{html.escape(label)}</td><td>{state}</td>"
                    f"<td>{data.get(f'exp_{config}_aux_events',0)}</td></tr>")
    content = '<section style="padding:14px;border:1px solid #55aabb;border-radius:8px;margin:12px 0">'
    content += '<b>So sánh đồng thời C1–C4 · PERCLOS theo thời gian</b><p>'
    content += 'C1 = CNN + PERCLOS · MAR dùng phát hiện ngáp · Head Pose dùng phát hiện gật/cúi đầu.<br>'
    content += f"Quan sát hợp lệ: {data.get('exp_valid_s',0):.1f}s · Độ phủ {data.get('exp_coverage',0)*100:.0f}%"
    content += '<br>Các cấu hình dùng cùng frame và đặc trưng; lựa chọn bên trái chỉ đổi cấu hình hiển thị/phát âm thanh.</p>'
    content += '<table style="width:100%;text-align:left"><tr><th>Mã</th><th>Tín hiệu</th><th>Kết quả mô hình (cửa sổ)</th><th>Sự kiện phụ hợp lệ</th></tr>'
    content += ''.join(rows)+'</table><p>Nhắm kéo dài (theo dõi riêng, không nhập vào C1–C4): <b>'
    content += ('ĐẠT NGƯỠNG' if data.get('exp_safety_alarm') and not data.get('stale') else 'Chưa đạt / chưa đủ quan sát')
    content += '</b></p><small>C2/C4: từ 2 lần ngáp trong 40 giây, thông báo mỗi ngáp mới, không cần dấu hiệu mắt. Nhắm kéo dài là nhánh riêng.</small></section>'
    content = content.replace('</section>', '<p><b>Âm thanh: ' + html.escape(data.get('exp_audio_reason','Chưa có phép đo')) +
        f"</b> · Đợt phát: {data.get('exp_audio_count',0)}</p>" +
        '<small>Kết quả C1–C4 giữ nguyên theo cửa sổ. Ngáp/gật mới đủ điều kiện có thông báo ngắn riêng; lịch sử cũ không tự phát lại.</small></section>')
    placeholder.markdown(content,unsafe_allow_html=True)
