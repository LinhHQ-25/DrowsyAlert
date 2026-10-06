"""Huấn luyện CNN phân loại trạng thái mắt (nhắm / mở) cho DrowsyAlert.

Ví dụ:
    python train_eye_cnn.py --data "D:/Datasets/eyes" --model both
    python train_eye_cnn.py --data "D:/Datasets/eyes" --model small --limit 500 --epochs 2   # chạy thử nhanh

Quy ước (khớp với app.py):
    - ảnh xám 64x64, giá trị 0..1, shape (64, 64, 1)
    - đầu ra softmax 2 lớp: chỉ số 0 = nhắm (closed), 1 = mở (open)
    - model tốt nhất (theo val accuracy) được chép sang models/eye_cnn.keras
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras import callbacks, layers, models

IMG = 64
CLASSES = ["closed", "open"]          # chỉ số 0 = closed, 1 = open
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
AUTOTUNE = tf.data.AUTOTUNE


class Tee:
    """In ra màn hình đồng thời ghi vào file console.txt cùng thư mục kết quả."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def open_files(paths):
    """Tự mở ảnh kết quả bằng trình xem ảnh mặc định (Windows)."""
    for p in paths:
        try:
            os.startfile(str(p))
        except Exception:  # noqa: BLE001 - không phải Windows hoặc không có ảnh
            pass


# ----------------------------------------------------------------------------
# 1. Quét và chia dữ liệu
# ----------------------------------------------------------------------------
def scan_dataset(root):
    """Duyệt đệ quy. Thư mục có tên chứa 'close' -> nhắm, chứa 'open' -> mở.
    Thư mục khác (yawn, no_yawn...) bị bỏ qua. Nhận diện train/val/test nếu có."""
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
        label = None
        for part in reversed(parts):
            if "close" in part:
                label = 0
                break
            if "open" in part:
                label = 1
                break
        if label is None:
            continue
        split = None
        for part in parts:
            if part in ("train", "training"):
                split = "train"
            elif part in ("val", "valid", "validation"):
                split = "val"
            elif part in ("test", "testing"):
                split = "test"
        rows.append(dict(path=str(p), label=label, split=split, group=p.stem.split("_")[0]))
    return pd.DataFrame(rows)


def make_splits(df, seed, group_split):
    if df["split"].notna().all() and {"train", "test"} <= set(df["split"]):
        tr, te = df[df.split == "train"], df[df.split == "test"]
        va = df[df.split == "val"]
        if va.empty:
            tr, va = train_test_split(tr, test_size=0.15, stratify=tr.label, random_state=seed)
        return tr, va, te
    if group_split and df.group.nunique() > 10:
        # chia theo "người" (tiền tố tên file trước dấu _) để không rò rỉ dữ liệu
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr_i, rest_i = next(g1.split(df, groups=df.group))
        tr, rest = df.iloc[tr_i], df.iloc[rest_i]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        va_i, te_i = next(g2.split(rest, groups=rest.group))
        return tr, rest.iloc[va_i], rest.iloc[te_i]
    tr, rest = train_test_split(df, test_size=0.3, stratify=df.label, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=seed)
    return tr, va, te


def load_img(path, label):
    x = tf.io.read_file(path)
    x = tf.io.decode_image(x, channels=1, expand_animations=False)
    x = tf.image.resize(x, (IMG, IMG))
    x = tf.cast(x, tf.float32) / 255.0
    x.set_shape((IMG, IMG, 1))
    return x, label


def make_ds(df, batch, train, seed):
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("int32")))
    if train:
        ds = ds.shuffle(min(len(df), 10000), seed=seed)
    return ds.map(load_img, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# 2. Kiến trúc
# ----------------------------------------------------------------------------
def augment_block():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2, value_range=(0, 1)),
        layers.RandomContrast(0.2),
    ], name="augment")


def build_small():
    inp = layers.Input((IMG, IMG, 1))
    x = augment_block()(inp)
    for f in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)   # 0.9 để thống kê BN cập nhật kịp
            x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(2, activation="softmax")(x)
    return models.Model(inp, out, name="small_cnn"), None


def build_mobilenet():
    inp = layers.Input((IMG, IMG, 1))
    x = augment_block()(inp)
    x = layers.Concatenate()([x, x, x])                 # xám -> 3 kênh
    x = layers.Rescaling(2.0, offset=-1.0)(x)           # [0,1] -> [-1,1] cho MobileNetV2
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    except Exception as e:  # noqa: BLE001 - không tải được trọng số (mạng chậm)
        print("Không tải được trọng số ImageNet, huấn luyện từ đầu:", e)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG, IMG, 3), include_top=False, weights=None)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation="softmax")(x)
    return models.Model(inp, out, name="mobilenetv2"), base


# ----------------------------------------------------------------------------
# 3. Huấn luyện
# ----------------------------------------------------------------------------
def compile_model(model, lr):
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def merge_history(*hs):
    out = {}
    for h in hs:
        for k, v in h.history.items():
            out.setdefault(k, []).extend([float(i) for i in v])
    return out


def plot_history(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title(f"{title} - Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train")
    ax[1].plot(hist["val_accuracy"], label="val")
    ax[1].set_title(f"{title} - Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_history(hist, name):
    lr_key = "lr" if "lr" in hist else "learning_rate"
    best = int(np.argmin(hist["val_loss"]))
    print(f"\n--- Lịch sử huấn luyện: {name} (* = epoch tốt nhất theo val_loss) ---")
    print(f"{'Epoch':>6} {'loss':>8} {'acc':>8} {'val_loss':>9} {'val_acc':>8} {'lr':>10}")
    for i in range(len(hist["loss"])):
        lr = hist[lr_key][i] if lr_key in hist else float("nan")
        mark = "*" if i == best else " "
        print(f"{i + 1:>5}{mark} {hist['loss'][i]:>8.4f} {hist['accuracy'][i]:>8.4f} "
              f"{hist['val_loss'][i]:>9.4f} {hist['val_accuracy'][i]:>8.4f} {lr:>10.2e}")


def train_model(name, build_fn, args, ds_tr, ds_va, cw, run_dir):
    model, base = build_fn()
    compile_model(model, 1e-3)
    ckpt = run_dir / "best.keras"
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        callbacks.CSVLogger(str(run_dir / "history_log.csv")),
    ]
    t0 = time.time()
    h1 = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, class_weight=cw,
                   callbacks=cbs, verbose=2)
    hs = [h1]
    if base is not None and args.epochs_ft > 0:          # tinh chỉnh (fine-tune) 30 lớp cuối
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        hs.append(model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs_ft,
                            class_weight=cw, callbacks=cbs, verbose=2))
    train_time = time.time() - t0
    hist = merge_history(*hs)
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, run_dir / "curves.png", name)
    print_history(hist, name)
    return model, train_time, len(hist["loss"])


# ----------------------------------------------------------------------------
# 4. Đánh giá
# ----------------------------------------------------------------------------
def evaluate(model, ds_te, te_df, run_dir, name):
    y_true = np.concatenate([y.numpy() for _, y in ds_te])
    probs = model.predict(ds_te, verbose=0)
    y_pred = probs.argmax(1)
    acc = float((y_pred == y_true).mean())
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    auc = float(roc_auc_score(y_true == 0, probs[:, 0]))

    print(f"\n--- Kết quả trên tập TEST: {name} ({len(y_true)} ảnh) ---")
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print(f"{'':>10}{'closed':>10}{'open':>10}")
    for i, c in enumerate(CLASSES):
        print(f"{c:>10}{cm[i, 0]:>10}{cm[i, 1]:>10}")
    wrong_all = np.where(y_pred != y_true)[0]
    print(f"Số ảnh dự đoán sai: {len(wrong_all)}/{len(y_true)}")
    for i in wrong_all[:10]:
        print(f"  {Path(te_df.path.iloc[i]).name}: thực tế={CLASSES[y_true[i]]}, "
              f"dự đoán={CLASSES[y_pred[i]]} (P_closed={probs[i, 0]:.2f})")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"{name} - Confusion matrix")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true == 0, probs[:, 0])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} - ROC (lớp nhắm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "roc.png", dpi=150)
    plt.close(fig)

    wrong = np.where(y_pred != y_true)[0][:16]              # ảnh dự đoán sai để phân tích lỗi
    if len(wrong):
        fig, axs = plt.subplots(2, 8, figsize=(14, 4))
        for a in axs.ravel():
            a.axis("off")
        for a, i in zip(axs.ravel(), wrong):
            img, _ = load_img(te_df.path.iloc[i], 0)
            a.imshow(img.numpy()[..., 0], cmap="gray", vmin=0, vmax=1)
            a.set_title(f"T:{CLASSES[y_true[i]][0]} D:{CLASSES[y_pred[i]][0]}", fontsize=8)
        fig.suptitle(f"{name} - Ảnh dự đoán sai (T: thực tế, D: dự đoán)")
        fig.tight_layout()
        fig.savefig(run_dir / "misclassified.png", dpi=150)
        plt.close(fig)

    x = np.zeros((2, IMG, IMG, 1), np.float32)              # giống app: 2 mắt / khung hình
    model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(100):
        model(x, training=False)
    latency_ms = (time.perf_counter() - t0) / 100 * 1000

    metrics = dict(
        model=name, test_accuracy=acc, roc_auc=auc,
        precision_closed=rep["closed"]["precision"], recall_closed=rep["closed"]["recall"],
        f1_closed=rep["closed"]["f1-score"], precision_open=rep["open"]["precision"],
        recall_open=rep["open"]["recall"], f1_open=rep["open"]["f1-score"],
        macro_f1=rep["macro avg"]["f1-score"], params=int(model.count_params()),
        latency_ms_per_frame=latency_ms, confusion_matrix=cm.tolist(),
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print(f"[{name}] Test accuracy = {acc:.4f} | AUC = {auc:.4f} | {latency_ms:.1f} ms/khung hình")
    return metrics


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Thư mục gốc chứa ảnh (có thư mục closed/open)")
    ap.add_argument("--model", choices=["small", "mobilenet", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--epochs-ft", type=int, default=10, help="Số epoch fine-tune MobileNetV2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-split", action="store_true",
                    help="Chia theo người (tiền tố tên file trước dấu _), dùng cho bộ có ID người")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh mỗi lớp (chạy thử nhanh)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--show", action="store_true", help="Tự mở ảnh kết quả sau mỗi model (Windows)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = scan_dataset(args.data)
    if df.empty:
        raise SystemExit("Không tìm thấy ảnh nào trong thư mục có tên chứa 'close'/'open'. "
                         "Kiểm tra lại đường dẫn --data.")
    if args.limit:
        df = pd.concat([g.sample(min(len(g), args.limit), random_state=args.seed)
                        for _, g in df.groupby("label")]).reset_index(drop=True)
    tr, va, te = make_splits(df, args.seed, args.group_split)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.out) / stamp
    root_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, open(root_dir / "console.txt", "w", encoding="utf-8"))
    print(f"Thư mục kết quả của lần chạy này: {root_dir}")
    summary = {n: {CLASSES[k]: int((d.label == k).sum()) for k in (0, 1)}
               for n, d in (("train", tr), ("val", va), ("test", te))}
    (root_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print("Số ảnh mỗi tập:", json.dumps(summary, indent=2))
    if args.group_split is False and df["split"].isna().all():
        print("Lưu ý: chia ngẫu nhiên theo ảnh, có thể rò rỉ nếu cùng một người xuất hiện ở nhiều tập.")

    ds_tr = make_ds(tr, args.batch, True, args.seed)
    ds_va = make_ds(va, args.batch, False, args.seed)
    ds_te = make_ds(te, args.batch, False, args.seed)
    counts = np.bincount(tr.label.values, minlength=2)
    cw = {k: float(len(tr) / (2 * counts[k])) for k in (0, 1)}      # cân bằng lớp

    builders = {"small": build_small, "mobilenet": build_mobilenet}
    names = ["small", "mobilenet"] if args.model == "both" else [args.model]
    results = []
    for name in names:
        run_dir = root_dir / name
        run_dir.mkdir(exist_ok=True)
        print(f"\n===== Huấn luyện: {name} =====")
        model, t_train, n_epochs = train_model(name, builders[name], args, ds_tr, ds_va, cw, run_dir)
        val_loss, val_acc = model.evaluate(ds_va, verbose=0)
        m = evaluate(model, ds_te, te, run_dir, name)
        m.update(val_accuracy=float(val_acc), train_seconds=t_train, epochs_run=n_epochs)
        (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
        model.save(run_dir / "final.keras")
        results.append(m)
        if args.show:
            open_files([run_dir / "curves.png", run_dir / "confusion_matrix.png"])

    table = pd.DataFrame(results).drop(columns=["confusion_matrix"])
    table.to_csv(root_dir / "compare.csv", index=False)
    print("\n", table[["model", "val_accuracy", "test_accuracy", "roc_auc", "macro_f1",
                        "params", "latency_ms_per_frame"]].to_string(index=False))

    print("\n================ TỔNG KẾT ================")
    for r in results:
        verdict = "ĐẠT" if r["test_accuracy"] >= 0.90 else "CHƯA ĐẠT"
        print(f"{r['model']:>10}: test acc = {r['test_accuracy']:.4f} (mục tiêu >= 0,90: {verdict}) | "
              f"F1 macro = {r['macro_f1']:.4f} | AUC = {r['roc_auc']:.4f} | "
              f"{r['latency_ms_per_frame']:.1f} ms/khung hình | {r['epochs_run']} epoch, "
              f"{r['train_seconds'] / 60:.1f} phút")
    best = max(results, key=lambda r: r["val_accuracy"])           # chọn theo val, không theo test
    Path("models").mkdir(exist_ok=True)
    shutil.copy(root_dir / best["model"] / "final.keras", "models/eye_cnn.keras")
    print(f"\nĐã chép model '{best['model']}' sang models/eye_cnn.keras (app sẽ tự dùng CNN).")
    print(f"Toàn bộ kết quả (biểu đồ, số liệu cho báo cáo) nằm ở: {root_dir}")


if __name__ == "__main__":
    main()
