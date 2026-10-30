"""Presentation-only sound gate. Never changes C1-C4 experimental decisions."""
from dataclasses import dataclass
import math

VERSION = "open_eye_ack_repeated_yawn_v5"


@dataclass(frozen=True)
class NotificationPolicy:
    open_confirm_s: float = 1.5
    rearm_closed_s: float = 2.0
    max_sample_gap_s: float = .48
    auxiliary_confirmation_s: float = 1.35
    auxiliary_sound_s: float = 1.2
    auxiliary_notice_s: float = 3.0


class ExperimentNotification:
    def __init__(self, policy=None):
        self.policy = policy or NotificationPolicy()
        self.previous = None
        self.open_since = None
        self.closed_since = None
        self.acknowledged_at = None
        self.active = False
        self.count = 0
        self.trigger = ""
        self.started = None
        self.auxiliary_pending = None
        self.auxiliary_since = None
        self.last_auxiliary_notified = float('-inf')
        self.auxiliary_sound_until = float('-inf')
        self.auxiliary_notice_until = float('-inf')
        self.auxiliary_kind = ''
        self.last_yawn_notified = float('-inf')
        self.yawn_sound_until = float('-inf')
        self.yawn_notice_until = float('-inf')

    def update(self, now, *, raw_alarm, face, eye_valid, closed, safety_alarm=False,
               auxiliary_qualified=False, auxiliary_event_time=None, auxiliary_kind='ngáp/gật',
               mouth_valid=False, repeated_yawn_time=None, repeated_yawn_enabled=False):
        if not math.isfinite(now) or (self.previous is not None and now <= self.previous):
            raise ValueError("Notification timestamps must increase")
        valid = bool(face and eye_valid)
        if self.started is None:
            self.started = now
        gap = self.previous is not None and now-self.previous > self.policy.max_sample_gap_s
        self.previous = now
        was_active = self.active
        if self.trigger == 'repeated_yawn':
            self.active = False
        if gap or not valid:
            self.open_since = self.closed_since = None
            self.active = False
            self.auxiliary_pending = self.auxiliary_since = None
            self.auxiliary_sound_until = float('-inf')
        open_s = closed_s = 0.0
        if valid:
            if closed:
                self.open_since = None
                if self.closed_since is None:
                    self.closed_since = now
                closed_s = now-self.closed_since
            else:
                self.closed_since = None
                if self.open_since is None:
                    self.open_since = now
                open_s = now-self.open_since
            if open_s+1e-9 >= self.policy.open_confirm_s:
                self.acknowledged_at = now if self.acknowledged_at is None else self.acknowledged_at
            if (self.acknowledged_at is not None and closed_s+1e-9 >= self.policy.rearm_closed_s
                    and self.closed_since > self.acknowledged_at):
                self.acknowledged_at = None
            safety_alarm = bool(safety_alarm and closed)
            # The source closure timer already proved continuous valid closure.
            # Never add the PERCLOS confirmation hold to this independent trigger.
            if safety_alarm:
                self.acknowledged_at = None
            safety_grace = (self.active and self.trigger == 'prolonged_closure'
                            and not closed and open_s < self.policy.open_confirm_s)
            requested = bool(raw_alarm or safety_alarm or safety_grace)
            if not requested:
                self.active = False
            elif self.acknowledged_at is not None:
                self.active = False
            elif self.active:
                # An isolated open prediction does not interrupt the sound until
                # opening is confirmed; invalid observation always stops it above.
                self.active = open_s < self.policy.open_confirm_s
            else:
                # Historical PERCLOS alone cannot restart sound while eyes are open.
                self.active = bool(closed)
            if self.active:
                if safety_alarm:
                    self.trigger = 'prolonged_closure'
                elif not safety_grace:
                    self.trigger = 'selected_config'
            # Only a new, qualified auxiliary event can bypass an open-eye
            # acknowledgment. Old PERCLOS and the same event cannot repeat sound.
            event_new = (auxiliary_event_time is not None and math.isfinite(auxiliary_event_time)
                         and self.started <= auxiliary_event_time <= now
                         and auxiliary_event_time > self.last_auxiliary_notified
                         and (self.acknowledged_at is None or auxiliary_event_time > self.acknowledged_at))
            if auxiliary_qualified and event_new:
                if self.auxiliary_pending != auxiliary_event_time:
                    self.auxiliary_pending = auxiliary_event_time
                    self.auxiliary_since = now
                confirmed = now-self.auxiliary_since+1e-9 >= self.policy.auxiliary_confirmation_s
                if confirmed and raw_alarm:
                    self.last_auxiliary_notified = auxiliary_event_time
                    self.auxiliary_kind = auxiliary_kind
                    if not self.active:
                        self.auxiliary_sound_until = now+self.policy.auxiliary_sound_s
                        self.auxiliary_notice_until = now+self.policy.auxiliary_notice_s
                        # A short event notice also acknowledges the old model
                        # state; it must not turn into an endless raw-alarm loop.
                        self.acknowledged_at = now
            else:
                self.auxiliary_pending = self.auxiliary_since = None
            if now < self.auxiliary_sound_until and not (self.active and self.trigger == 'prolonged_closure'):
                self.active = True
                self.trigger = 'new_auxiliary'
            elif self.active and not closed and self.trigger == 'new_auxiliary':
                self.active = False
        # This reminder uses visible mouth measurements, independently of eye/PERCLOS readiness.
        yawn_valid = bool(repeated_yawn_enabled and face and mouth_valid)
        if gap or not yawn_valid:
            self.yawn_sound_until = self.yawn_notice_until = float('-inf')
        new_yawn = (yawn_valid and repeated_yawn_time is not None
                    and math.isfinite(repeated_yawn_time)
                    and repeated_yawn_time > self.last_yawn_notified
                    and (abs(now - repeated_yawn_time) <= 1e-6 or now - repeated_yawn_time <= 1.0))
        if new_yawn:
            self.last_yawn_notified = repeated_yawn_time
            self.yawn_sound_until = now + self.policy.auxiliary_sound_s
            self.yawn_notice_until = now + self.policy.auxiliary_notice_s
            self.acknowledged_at = now
        yawn_notice = bool(yawn_valid and now < self.yawn_notice_until)
        if yawn_valid and now < self.yawn_sound_until and not (self.active and self.trigger == 'prolonged_closure'):
            self.active = True
            self.trigger = 'yawn'
        if self.active and not was_active:
            self.count += 1
        silenced = bool(raw_alarm and valid and not self.active)
        reason = ("Nhắm mắt liên tục đủ ngưỡng — cảnh báo riêng" if self.active and self.trigger == 'prolonged_closure' else
                  "Phát hiện ngáp — cảnh báo buồn ngủ" if self.active and self.trigger in ('yawn', 'repeated_yawn') else
                  "Không đủ quan sát" if not valid else
                  "Sự kiện " + self.auxiliary_kind + " mới đủ điều kiện kèm dấu hiệu mắt" if self.active and self.trigger == 'new_auxiliary' else
                  "Đang phát theo cấu hình được chọn" if self.active else
                  "Đã xác nhận mở mắt; không phát lại theo lịch sử PERCLOS" if silenced and self.acknowledged_at is not None else
                  "Mắt đang mở; PERCLOS lịch sử chỉ được theo dõi" if silenced else
                  "Cấu hình chưa cảnh báo")
        return dict(exp_audio_on=self.active, exp_audio_silenced=silenced,
                    exp_audio_acknowledged=self.acknowledged_at is not None,
                    exp_audio_count=self.count, exp_open_stable_s=open_s,
                    exp_audio_reason=reason, exp_audio_trigger=self.trigger if self.active else "",
                    exp_aux_notice_active=bool(valid and now < self.auxiliary_notice_until),
                    exp_aux_notice_kind=self.auxiliary_kind,
                    exp_yawn_notice_active=yawn_notice,
                    exp_yawn_notice_event_s=self.last_yawn_notified if math.isfinite(self.last_yawn_notified) else None,
                    exp_notification_policy=VERSION)


NOTIFICATION_FIELDS = ("exp_audio_on", "exp_audio_silenced", "exp_audio_acknowledged",
                       "exp_audio_count", "exp_open_stable_s", "exp_audio_reason", "exp_audio_trigger", "exp_notification_policy",
                       "exp_aux_notice_active", "exp_aux_notice_kind", "exp_yawn_notice_active", "exp_yawn_notice_event_s")
