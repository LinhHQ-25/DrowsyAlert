"""
Script tự động sinh các hình vẽ sơ đồ cho báo cáo / khóa luận:
- Hình 3.1: Sơ đồ kiến trúc tổng thể hệ thống DrowsyAlert
- Hình 3.2: Quy trình phân loại trạng thái mắt bằng mạng Small CNN
- Hình 3.3: Lưu đồ thuật toán quy trình ra quyết định cảnh báo buồn ngủ
"""
import os
import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Đảm bảo hiển thị chuẩn tiếng Việt trên Windows
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def draw_figure_3_1():
    """Hình 3.1: Kiến trúc tổng thể hệ thống DrowsyAlert"""
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Tiêu đề chính
    ax.text(7.5, 9.1, "HÌNH 3.1: KIẾN TRÚC TỔNG THỂ HỆ THỐNG DROWSYALERT",
            fontsize=15, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(7.5, 8.75, "Hệ thống giám sát và phát hiện buồn ngủ thời gian thực kết hợp thị giác máy tính và học sâu",
            fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#475569')

    # Định nghĩa các tầng (Layers)
    layers = [
        {"x": 0.5, "w": 2.4, "title": "1. THU NHẬN DỮ LIỆU\n(Video Input)", "color": "#0284C7", "bg": "#E0F2FE",
         "items": ["Webcam / Camera Stream", "Độ phân giải: 640×360", "Tốc độ: 30 FPS", "WebRTC Streamer (Async)", "Mailbox buffer 1-slot"]},
        
        {"x": 3.3, "w": 2.7, "title": "2. TRÍCH XUẤT ĐIỂM MỐC\n(Landmark Extraction)", "color": "#0D9488", "bg": "#CCFBF1",
         "items": ["MediaPipe FaceMesh\n(468 điểm mốc khuôn mặt)", "MediaPipe Hands\n(21 điểm mốc bàn tay)", "SolvePnP Head Pose\n(Ước lượng Pitch, Yaw, Roll)", "Chuyển ảnh xám & Resize 384px\n(Tối ưu tải CPU)"]},
        
        {"x": 6.4, "w": 2.7, "title": "3. ĐẶC TRƯNG ĐA TÍN HIỆU\n(Feature Engineering)", "color": "#7C3AED", "bg": "#EDE9FE",
         "items": ["Crop ROI Mắt (64×64)\n→ Small CNN Inference", "Tỷ lệ co giãn mắt (EAR)\n& Mốc cá nhân (open_ear_ref)", "Tỷ lệ mở miệng (MAR)", "Phát hiện tay che miệng\n(Hand-to-Mouth Occlusion)"]},
        
        {"x": 9.5, "w": 2.6, "title": "4. TÍCH LŨY THỜI GIAN\n(Temporal Aggregation)", "color": "#D97706", "bg": "#FEF3C7",
         "items": ["Cửa sổ trượt PERCLOS (30s)", "Nhắm mắt liên tục (≥ 2.0s)", "Bộ phát hiện ngáp:\n• Ngáp mở miệng (MAR ≥ 0.6)\n• Ngáp che miệng (Tay+Nheo mắt)", "Bộ phát hiện gật đầu (Pitch ≥ 20°)", "Thực nghiệm C1 - C4"]},
        
        {"x": 12.5, "w": 2.1, "title": "5. CẢNH BÁO & UI\n(Output & Alerts)", "color": "#DC2626", "bg": "#FEE2E2",
         "items": ["Còi hú / Chuông bíp\n(Web Audio API)", "Banner cảnh báo đỏ/vàng\n(Đa mức độ)", "Video Bounding Boxes\n(Mắt, Miệng, Bàn tay)", "Báo cáo phiên & CSV Log"]}
    ]

    for col in layers:
        # Box header
        box_card = patches.FancyBboxPatch((col["x"], 0.8), col["w"], 7.6,
                                          boxstyle="round,pad=0.15,rounding_size=0.25",
                                          linewidth=1.8, edgecolor=col["color"], facecolor=col["bg"])
        ax.add_patch(box_card)
        
        # Header banner
        header_card = patches.FancyBboxPatch((col["x"], 7.3), col["w"], 1.1,
                                            boxstyle="round,pad=0.1,rounding_size=0.18",
                                            linewidth=1.5, edgecolor=col["color"], facecolor=col["color"])
        ax.add_patch(header_card)
        ax.text(col["x"] + col["w"]/2, 7.85, col["title"],
                fontsize=9.5, fontweight='bold', ha='center', va='center', color='white', multialignment='center')

        # Items
        y_pos = 6.6
        for itm in col["items"]:
            # Sub-card
            h_sub = 1.05 if "\n" in itm else 0.75
            sub_patch = patches.FancyBboxPatch((col["x"] + 0.12, y_pos - h_sub + 0.4), col["w"] - 0.24, h_sub,
                                               boxstyle="round,pad=0.08,rounding_size=0.12",
                                               linewidth=1.0, edgecolor="#CBD5E1", facecolor="white")
            ax.add_patch(sub_patch)
            ax.text(col["x"] + col["w"]/2, y_pos - h_sub/2 + 0.4, itm,
                    fontsize=8.5, ha='center', va='center', color='#1E293B', multialignment='center')
            y_pos -= (h_sub + 0.22)

    # Vẽ các mũi tên luồng dữ liệu giữa các tầng
    arrow_props = dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.2, color="#475569")
    ax.annotate("", xy=(3.3, 4.8), xytext=(2.9, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(6.4, 4.8), xytext=(6.0, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(9.5, 4.8), xytext=(9.1, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(12.5, 4.8), xytext=(12.1, 4.8), arrowprops=arrow_props)

    # Ghi chú chân trang
    ax.text(7.5, 0.35, "Luồng xử lý bất đồng bộ (Multithreading): Luồng Video 30 FPS không bị chặn bởi luồng AI phân tích",
            fontsize=9.0, ha='center', va='center', color='#64748B', fontweight='semibold')

    out_path = FIGURES_DIR / "hinh_3_1_kien_truc_tong_the.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Đã tạo:", out_path)
    return out_path


def draw_figure_3_2():
    """Hình 3.2: Quy trình phân loại trạng thái mắt bằng mạng Small CNN"""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    ax.text(7.5, 8.1, "HÌNH 3.2: QUY TRÌNH PHÂN LOẠI TRẠNG THÁI MẮT BẰNG SMALL CNN",
            fontsize=15, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(7.5, 7.7, "Kiến trúc mạng nơ-ron tích chập tùy biến siêu nhẹ (288K tham số) cho phân loại mắt nhắm/mở thời gian thực",
            fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#475569')

    stages = [
        {"x": 0.5, "w": 2.2, "title": "1. ẢNH ĐẦU VÀO", "color": "#2563EB", "bg": "#EFF6FF",
         "items": ["Crop ROI Mắt từ FaceMesh", "Ảnh xám (Grayscale)", "Kích thước: 64 × 64 × 1", "Chuẩn hóa giá trị: [0, 1]"]},
        
        {"x": 3.0, "w": 2.6, "title": "2. KHỐI TÍCH CHẬP 1", "color": "#0284C7", "bg": "#E0F2FE",
         "items": ["Conv2D (32 bộ lọc, 3×3)", "BatchNormalization(0.9)", "Hàm kích hoạt ReLU", "Conv2D (32 bộ lọc, 3×3)", "BatchNormalization + ReLU", "MaxPooling2D (2×2)", "→ Output: 32 × 32 × 32"]},
        
        {"x": 5.9, "w": 2.6, "title": "3. KHỐI TÍCH CHẬP 2", "color": "#0D9488", "bg": "#CCFBF1",
         "items": ["Conv2D (64 bộ lọc, 3×3)", "BatchNormalization(0.9)", "Hàm kích hoạt ReLU", "Conv2D (64 bộ lọc, 3×3)", "BatchNormalization + ReLU", "MaxPooling2D (2×2)", "→ Output: 16 × 16 × 64"]},
        
        {"x": 8.8, "w": 2.6, "title": "4. KHỐI TÍCH CHẬP 3", "color": "#7C3AED", "bg": "#EDE9FE",
         "items": ["Conv2D (128 bộ lọc, 3×3)", "BatchNormalization(0.9)", "Hàm kích hoạt ReLU", "Conv2D (128 bộ lọc, 3×3)", "BatchNormalization + ReLU", "MaxPooling2D (2×2)", "→ Output: 8 × 8 × 128"]},
        
        {"x": 11.7, "w": 2.8, "title": "5. PHÂN LOẠI & HẬU XỬ LÝ", "color": "#C026D3", "bg": "#FAE8FF",
         "items": ["GlobalAveragePooling2D", "Dropout (0.4)", "Dense (2 nút, Softmax)", "→ P(Closed), P(Open)", "EMA Smoothing (2 frames)", "Quyết định: MẮT NHẮM / MỞ"]}
    ]

    for col in stages:
        box_card = patches.FancyBboxPatch((col["x"], 1.0), col["w"], 6.2,
                                          boxstyle="round,pad=0.15,rounding_size=0.2",
                                          linewidth=1.8, edgecolor=col["color"], facecolor=col["bg"])
        ax.add_patch(box_card)
        
        header_card = patches.FancyBboxPatch((col["x"], 6.4), col["w"], 0.8,
                                            boxstyle="round,pad=0.08,rounding_size=0.15",
                                            linewidth=1.5, edgecolor=col["color"], facecolor=col["color"])
        ax.add_patch(header_card)
        ax.text(col["x"] + col["w"]/2, 6.8, col["title"],
                fontsize=9.2, fontweight='bold', ha='center', va='center', color='white')

        y_pos = 5.9
        for itm in col["items"]:
            is_hl = "→" in itm or "MẮT NHẮM" in itm
            bg_item = "#FEF3C7" if is_hl else "white"
            border_c = "#F59E0B" if is_hl else "#CBD5E1"
            fw = 'bold' if is_hl else 'normal'
            fc = '#92400E' if is_hl else '#1E293B'
            
            sub_patch = patches.FancyBboxPatch((col["x"] + 0.12, y_pos - 0.45), col["w"] - 0.24, 0.52,
                                               boxstyle="round,pad=0.06,rounding_size=0.1",
                                               linewidth=1.0, edgecolor=border_c, facecolor=bg_item)
            ax.add_patch(sub_patch)
            ax.text(col["x"] + col["w"]/2, y_pos - 0.19, itm,
                    fontsize=8.2, ha='center', va='center', color=fc, fontweight=fw)
            y_pos -= 0.65

    arrow_props = dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.2, color="#475569")
    ax.annotate("", xy=(3.0, 4.0), xytext=(2.7, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(5.9, 4.0), xytext=(5.6, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(8.8, 4.0), xytext=(8.5, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(11.7, 4.0), xytext=(11.4, 4.0), arrowprops=arrow_props)

    ax.text(7.5, 0.45, "Độ trễ suy luận (Inference Latency): ~5 - 8 ms/frame trên CPU Intel/AMD thông dụng · Bộ nhớ ~1.1 MB",
            fontsize=9.2, ha='center', va='center', color='#64748B', fontweight='semibold')

    out_path = FIGURES_DIR / "hinh_3_2_quy_trinh_small_cnn.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Đã tạo:", out_path)
    return out_path


def draw_figure_3_3():
    """Hình 3.3: Lưu đồ thuật toán quy trình cảnh báo buồn ngủ"""
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    ax.text(7.5, 9.1, "HÌNH 3.3: LƯU ĐỒ THUẬT TOÁN QUY TRÌNH RA QUYẾT ĐỊNH CẢNH BÁO",
            fontsize=15, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(7.5, 8.7, "Cơ chế phân tầng cảnh báo đa tiêu chí: Nhánh khẩn cấp, Nhánh ngáp và Nhánh tích lũy PERCLOS",
            fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#475569')

    # Start block
    start_box = patches.FancyBboxPatch((0.5, 4.3), 2.2, 1.0, boxstyle="round,pad=0.1,rounding_size=0.4",
                                       fc="#3B82F6", ec="#1D4ED8", lw=1.8)
    ax.add_patch(start_box)
    ax.text(1.6, 4.8, "Khung hình Camera\n(RGB 640×360)", fontsize=9.0, ha='center', va='center', color='white', fontweight='bold')

    # Pre-check
    check_box = patches.FancyBboxPatch((3.2, 4.1), 2.4, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       fc="#FEF08A", ec="#CA8A04", lw=1.5)
    ax.add_patch(check_box)
    ax.text(4.4, 4.8, "Kiểm tra chất lượng:\n• Đủ sáng (35 - 220)\n• Góc Yaw ≤ 35°\n• Tương phản mắt ≥ 10",
            fontsize=8.2, ha='center', va='center', color="#854D0E", fontweight='semibold')

    # 3 Branches
    # Nhánh 1: Cấp bách (Top)
    b1_box = patches.FancyBboxPatch((6.3, 6.7), 4.2, 1.4, boxstyle="round,pad=0.1,rounding_size=0.18",
                                    fc="#FEE2E2", ec="#DC2626", lw=1.8)
    ax.add_patch(b1_box)
    ax.text(8.4, 7.4, "NHÁNH 1: NHẮM MẮT LIÊN TỤC KÉO DÀI\n• Mắt nhắm liên tục t_closed ≥ 2.0s\n(Không phụ thuộc cửa sổ PERCLOS)",
            fontsize=8.5, ha='center', va='center', color="#991B1B", fontweight='bold')

    # Nhánh 2: Ngáp (Middle)
    b2_box = patches.FancyBboxPatch((6.3, 4.1), 4.2, 1.4, boxstyle="round,pad=0.1,rounding_size=0.18",
                                    fc="#FFEDD5", ec="#EA580C", lw=1.8)
    ax.add_patch(b2_box)
    ax.text(8.4, 4.8, "NHÁNH 2: PHÁT HIỆN HÀNH VI NGÁP\n• Ngáp mở miệng: MAR ≥ 0.60 trong 1.5s\n• Ngáp che miệng: Tay che miệng + Nheo mắt 1.2s",
            fontsize=8.5, ha='center', va='center', color="#9A3412", fontweight='bold')

    # Nhánh 3: PERCLOS (Bottom)
    b3_box = patches.FancyBboxPatch((6.3, 1.5), 4.2, 1.4, boxstyle="round,pad=0.1,rounding_size=0.18",
                                    fc="#E0E7FF", ec="#4F46E5", lw=1.8)
    ax.add_patch(b3_box)
    ax.text(8.4, 2.2, "NHÁNH 3: TÍCH LŨY PERCLOS (CỬA SỔ 30 GIÂY)\n• C1: PERCLOS ≥ 30% duy trì ≥ 2.2s\n• C2/C3/C4: Kết hợp Ngáp lặp lại & Gật đầu",
            fontsize=8.5, ha='center', va='center', color="#3730A3", fontweight='bold')

    # Action / Alert blocks
    al1_box = patches.FancyBboxPatch((11.4, 6.7), 3.1, 1.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                                     fc="#EF4444", ec="#991B1B", lw=2.0)
    ax.add_patch(al1_box)
    ax.text(12.95, 7.4, "[BÁO ĐỘNG ĐỎ]\n(Khẩn cấp - Ngủ gật)\nPhát còi báo liên tục!",
            fontsize=8.8, ha='center', va='center', color="white", fontweight='bold')

    al2_box = patches.FancyBboxPatch((11.4, 4.1), 3.1, 1.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                                     fc="#F97316", ec="#C2410C", lw=2.0)
    ax.add_patch(al2_box)
    ax.text(12.95, 4.8, "[CẢNH BÁO ĐỎ / BÍP]\n(Phát hiện ngáp mệt mỏi)\nNgáp ≥ 2 lần → Báo động cao",
            fontsize=8.5, ha='center', va='center', color="white", fontweight='bold')

    al3_box = patches.FancyBboxPatch((11.4, 1.5), 3.1, 1.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                                     fc="#6366F1", ec="#3730A3", lw=2.0)
    ax.add_patch(al3_box)
    ax.text(12.95, 2.2, "[CẢNH BÁO CẤU HÌNH]\n(C1, C2, C3, C4)\nCòi báo xác nhận theo mô hình",
            fontsize=8.5, ha='center', va='center', color="white", fontweight='bold')

    # Mũi tên liên kết
    arr = dict(arrowstyle="->,head_width=0.4,head_length=0.5", lw=1.8, color="#475569")
    ax.annotate("", xy=(3.2, 4.8), xytext=(2.7, 4.8), arrowprops=arr)
    
    # Rẽ 3 nhánh
    ax.annotate("", xy=(6.3, 7.4), xytext=(5.6, 5.2), arrowprops=arr)
    ax.annotate("", xy=(6.3, 4.8), xytext=(5.6, 4.8), arrowprops=arr)
    ax.annotate("", xy=(6.3, 2.2), xytext=(5.6, 4.4), arrowprops=arr)

    # Nối sang cảnh báo
    ax.annotate("", xy=(11.4, 7.4), xytext=(10.5, 7.4), arrowprops=arr)
    ax.annotate("", xy=(11.4, 4.8), xytext=(10.5, 4.8), arrowprops=arr)
    ax.annotate("", xy=(11.4, 2.2), xytext=(10.5, 2.2), arrowprops=arr)

    # Ghi chú giải phóng
    ack_box = patches.FancyBboxPatch((4.5, 0.4), 6.0, 0.65, boxstyle="round,pad=0.08,rounding_size=0.12",
                                     fc="#F1F5F9", ec="#94A3B8", lw=1.0)
    ax.add_patch(ack_box)
    ax.text(7.5, 0.72, "Cơ chế tắt còi: Mở mắt ổn định ≥ 1.5 giây → Dừng chuông và chuyển sang theo dõi",
            fontsize=8.5, ha='center', va='center', color="#334155", fontweight='semibold')

    out_path = FIGURES_DIR / "hinh_3_3_quy_trinh_canh_bao.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Đã tạo:", out_path)
    return out_path


if __name__ == "__main__":
    draw_figure_3_1()
    draw_figure_3_2()
    draw_figure_3_3()
    print("Hoàn tất tạo bộ 3 hình ảnh tại thư mục figures/")
