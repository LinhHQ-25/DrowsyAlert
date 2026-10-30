"""Readable live states; independent of the raw four-variant decision labels."""
def presentation_state(data, playing=True):
    if not playing:
        return 'idle', 'Chưa bật camera', 'Bấm START để bắt đầu.'
    if data.get('stale'):
        return 'warn', 'Chưa có dữ liệu AI mới', 'Không dùng kết quả cũ để phát âm thanh.'
    if not data.get('frame_seq'):
        return 'idle', 'Đang khởi động AI', 'Mở mắt, nhìn thẳng và chờ mô hình khởi tạo.'
    if not data.get('face'):
        return 'idle', 'Chưa thấy khuôn mặt', 'Điều chỉnh vị trí và ánh sáng.'
    trigger = data.get('exp_audio_trigger')
    if data.get('exp_audio_on') and trigger == 'prolonged_closure':
        return 'bad', 'Cảnh báo buồn ngủ — nhắm mắt kéo dài', 'Đã đủ ngưỡng nhắm liên tục; không chờ PERCLOS.'
    if data.get('exp_yawn_notice_active') or data.get('exp_recent_yawn'):
        yawn_type = data.get('yawn_type', '')
        type_str = " (che miệng)" if yawn_type == 'covered' else ""
        yawns_count = data.get('exp_yawns', data.get('yawns', 1))
        if yawns_count >= 2:
            return 'bad', f'Cảnh báo nguy cơ cao — Ngáp liên tục{type_str}!', 'Đã ghi nhận nhiều lần ngáp trong 40 giây. Nguy cơ ngủ gật rất cao, hãy dừng xe nghỉ ngơi an toàn!'
        return 'bad', f'Cảnh báo buồn ngủ — Phát hiện ngáp{type_str}!', 'Tài xế đang có dấu hiệu mệt mỏi! Hãy tập trung quan sát đường.'
    if data.get('exp_aux_notice_active'):
        kind = data.get('exp_aux_notice_kind') or 'ngáp/gật'
        return 'bad', 'Cảnh báo buồn ngủ — ' + kind, 'Sự kiện mới lặp lại kèm dấu hiệu mắt; thông báo ngắn một lần.'
    if data.get('exp_audio_on'):
        return 'bad', 'Cảnh báo buồn ngủ', data.get('exp_audio_reason', 'Đã đủ điều kiện cảnh báo.')
    recent = []
    if data.get('exp_recent_nod'):
        recent.append('gật đầu')
    if recent:
        return 'warn', 'Có dấu hiệu mệt — đã ghi nhận ' + ' và '.join(recent), 'Một sự kiện chưa tự kết luận buồn ngủ. Theo dõi dấu hiệu mắt và sự kiện lặp lại.'
    if not data.get('eye_valid'):
        return 'warn', 'Quan sát mắt chưa đủ tin cậy', 'Điều chỉnh ánh sáng hoặc hướng nhìn; không coi dữ liệu thiếu là mắt mở.'
    if not data.get('exp_ready'):
        return 'idle', 'Đang tích lũy quan sát', 'Nhánh PERCLOS cần ít nhất 10 giây mắt hợp lệ. Ngáp lặp lại ở C2/C4 và nhắm kéo dài có luật riêng.'
    if data.get('drowsy'):
        if data.get('exp_'+data.get('exp_selected','')+'_repeated_yawn'):
            return 'warn', 'Đã thông báo ngáp lặp lại', 'Chờ sự kiện mới; không phát lại từ lịch sử ngáp cũ.'
        return 'warn', 'PERCLOS lịch sử còn cao', 'Mắt đã mở; không phát lại chỉ vì lịch sử nhắm mắt cũ.'
    return 'ok', 'Chưa ghi nhận dấu hiệu buồn ngủ', 'Tiếp tục theo dõi mắt, ngáp và tư thế đầu.'
from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=8)
def _camera_alert_badge(width):
    """Cache the Unicode badge so camera frames do not reload fonts or rasterize text."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    text = 'CẢNH BÁO BUỒN NGỦ!'
    size = min(24, max(12, width // 22))
    candidates = [str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/arialbd.ttf'),
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVuSans.ttf']
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size=size)
    bbox = font.getbbox(text)
    badge = Image.new('RGB', (bbox[2]-bbox[0]+20, bbox[3]-bbox[1]+16), (165, 35, 45))
    ImageDraw.Draw(badge).text((10-bbox[0], 8-bbox[1]), text, font=font, fill='white')
    return np.asarray(badge)[:, :, ::-1].copy()


def draw_camera_alert(image):
    """Draw Vietnamese text on an OpenCV BGR frame; Hershey fonts lack Vietnamese glyphs."""
    badge = _camera_alert_badge(image.shape[1])
    height = min(badge.shape[0], max(0, image.shape[0]-12))
    width = min(badge.shape[1], max(0, image.shape[1]-12))
    image[8:8+height, 8:8+width] = badge[:height, :width]
    return image

