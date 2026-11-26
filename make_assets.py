"""Build README / portfolio images from results/benchmark.csv and the test split.

Writes assets/tradeoff_chart.png, assets/before_after.png, assets/cover_1600x1200.png
and copies 3 test images to assets/examples/ (dataset is AGPL-3.0, redistribution allowed).
Headline numbers are computed from results/benchmark.csv, never typed in.
"""
import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
A = ROOT / "assets"
BG, FG, MUTED, ACC1, ACC2 = "#0e1117", "#e6edf3", "#8b949e", "#f78166", "#3fb950"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": "#30363d", "font.size": 12})


def pick_examples(n=3, seed=0):
    lbl = ROOT / "datasets/brain-tumor/labels/test"
    names = sorted(p.stem for p in lbl.glob("*.txt") if p.read_text().strip())
    random.Random(seed).shuffle(names)
    return [ROOT / "datasets/brain-tumor/images/test" / f"{s}.jpg" for s in names[:n]]


def draw_gt(img, stem):
    out = img.copy()
    h, w = out.shape[:2]
    for ln in (ROOT / "datasets/brain-tumor/labels/test" / f"{stem}.txt").read_text().split("\n"):
        if ln.strip():
            _, cx, cy, bw, bh = map(float, ln.split())
            cv2.rectangle(out, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (80, 200, 80), 2)
    return out


def predict(model, path, imgsz, conf):
    r = model.predict(str(path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    return r.plot(line_width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="weights/baseline_yolov8s.pt")
    ap.add_argument("--light", default="weights/light_yolov8n_int8.onnx")
    ap.add_argument("--base-row", default="YOLOv8s baseline (PyTorch FP32)")
    ap.add_argument("--light-row", default="YOLOv8n (ONNX INT8)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "results/benchmark.csv").set_index("model")
    b, l = df.loc[a.base_row], df.loc[a.light_row]
    size_x = b.size_MB / l.size_MB
    speed_x = b.cpu_latency_ms_median / l.cpu_latency_ms_median
    dmap = (l.mAP50 - b.mAP50) * 100
    headline = (f"{size_x:.1f}x smaller · {speed_x:.1f}x faster on CPU · "
                f"mAP@50 {dmap:+.1f} pts")

    # tradeoff chart: numbered markers keyed to a list; zoom inset for the crowded 10-13 ms region
    models = list(df.index)
    short = lambda m: m.replace("PyTorch ", "PT ")
    fig = plt.figure(figsize=(12.5, 6.4), dpi=150)
    ax = fig.add_axes([0.075, 0.12, 0.59, 0.78])
    colr = lambda m: ACC1 if m == a.base_row else (ACC2 if m == a.light_row else "#58a6ff")
    for k, m in enumerate(models, 1):
        r = df.loc[m]
        ax.scatter(r.cpu_latency_ms_median, r.mAP50, s=r.size_MB * 30 + 40, color=colr(m), alpha=.75,
                   edgecolor=FG, linewidth=.6, zorder=3)
    crowd = lambda r: 9.5 <= r.cpu_latency_ms_median <= 13.8 and r.mAP50 > 0.979
    tag = {"YOLOv8s (ONNX FP32)": (0, 22), "YOLOv8n (PyTorch FP32)": (0, 20),
           "YOLO11n (PyTorch FP32)": (0, -22), "YOLOv8n (ONNX INT8)": (0, -20)}
    for k, m in enumerate(models, 1):
        r = df.loc[m]
        if crowd(r):
            continue
        if m in (a.base_row, a.light_row):
            ax.annotate(f"{k}  {short(m)}\n{r.size_MB:.2f} MB", (r.cpu_latency_ms_median, r.mAP50),
                        xytext=((0, -52) if m == a.base_row else (12, 10)), textcoords="offset points",
                        ha="center" if m == a.base_row else "left", va="top" if m == a.base_row else "bottom", fontsize=10, weight="bold", color=colr(m))
        else:
            ax.annotate(str(k), (r.cpu_latency_ms_median, r.mAP50), xytext=tag.get(m, (0, 18)),
                        textcoords="offset points", ha="center", fontsize=10, color=FG)
    ax.add_patch(plt.Rectangle((9.5, 0.9788), 4.3, 0.0030, fill=False, ec=MUTED, ls="--", lw=.8, zorder=2))
    ax.set_xlabel("CPU latency, ms per image (batch 1, median, lower is better)")
    ax.set_ylabel("mAP@50 on held-out test split")
    ax.set_title("Accuracy vs CPU latency (bubble area = model size)", color=FG, loc="left")
    ax.grid(color="#21262d")
    ax.set_xlim(5, 54)
    ax.set_ylim(0.9700, 0.9905)
    ins = ax.inset_axes([0.55, 0.08, 0.43, 0.38])
    ins.set_xlim(9.4, 13.9); ins.set_ylim(0.9790, 0.9818)
    ins.set_facecolor("#161b22"); ins.grid(color="#21262d")
    ins.tick_params(labelsize=7)
    ins_off = {"YOLO11n (ONNX FP32)": (0, -16), "YOLO11n (ONNX FP16)": (-12, -14), "YOLO11n (ONNX INT8)": (0, 14),
               "YOLOv8n (ONNX FP32)": (0, 16), "YOLOv8n (ONNX FP16)": (0, -16)}
    for k, m in enumerate(models, 1):
        r = df.loc[m]
        if crowd(r):
            ins.scatter(r.cpu_latency_ms_median, r.mAP50, s=r.size_MB * 6 + 25, color=colr(m), alpha=.8,
                        edgecolor=FG, linewidth=.5, zorder=3)
            ins.annotate(str(k), (r.cpu_latency_ms_median, r.mAP50), xytext=ins_off.get(m, (0, 14)),
                         textcoords="offset points", ha="center", fontsize=9, color=FG)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED)
    # key
    fig.text(0.70, 0.88, "Key (size MB, CPU ms, mAP@50)", color=FG, fontsize=10, weight="bold")
    for k, m in enumerate(models, 1):
        r = df.loc[m]
        hi = m in (a.base_row, a.light_row)
        fig.text(0.70, 0.84 - (k - 1) * 0.062,
                 f"{k}  {short(m)}\n     {r.size_MB:.2f} MB, {r.cpu_latency_ms_median:.2f} ms, {r.mAP50:.3f}",
                 color=colr(m) if hi else FG, fontsize=8.5, weight="bold" if hi else "normal", va="top")
    fig.text(0.01, 0.01, "Demo on public data. Not for clinical use. Latency: CPU only.",
             color=MUTED, fontsize=8)
    fig.savefig(A / "tradeoff_chart.png")
    plt.close(fig)

    # before / after
    ex = pick_examples()
    (A / "examples").mkdir(exist_ok=True)
    base_m, light_m = YOLO(a.base), YOLO(a.light, task="detect")
    rows = []
    for p in ex:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem).strip("_") + p.suffix  # no spaces/parentheses
        shutil.copy(p, A / "examples" / safe)
        img = cv2.imread(str(p))
        rows.append([draw_gt(img, p.stem), predict(base_m, p, a.imgsz, a.conf),
                     predict(light_m, p, a.imgsz, a.conf)])
    fig, axs = plt.subplots(3, 3, figsize=(10.5, 10.5), dpi=130)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            axs[i, j].imshow(im[:, :, ::-1])
            axs[i, j].axis("off")
            if i == 0:
                axs[i, j].set_title(["Ground truth", f"Baseline ({b.size_MB} MB)",
                                     f"Light ({l.size_MB} MB)"][j], color=FG)
    fig.tight_layout()
    fig.savefig(A / "before_after.png")
    plt.close(fig)

    # cover 1600x1200
    fig = plt.figure(figsize=(16, 12), dpi=100)
    fig.text(0.05, 0.92, "Shrinking a YOLO tumour detector", fontsize=40, weight="bold", color=FG)
    fig.text(0.05, 0.855, headline, fontsize=28, color=ACC2)
    fig.text(0.05, 0.81, f"{a.base_row}  ->  {a.light_row}", fontsize=15, color=MUTED)
    for k, (title, im) in enumerate([("Baseline", rows[0][1]), ("Light", rows[0][2])]):
        ax = fig.add_axes([0.05 + k * 0.46, 0.30, 0.44, 0.46])
        ax.imshow(im[:, :, ::-1]); ax.axis("off")
        ax.set_title(f"{title}: {[b, l][k].size_MB} MB, {[b, l][k].cpu_latency_ms_median} ms CPU, "
                     f"mAP@50 {[b, l][k].mAP50:.3f}", color=FG, fontsize=15)
    fig.text(0.05, 0.20, f"Size {b.size_MB} -> {l.size_MB} MB    Latency {b.cpu_latency_ms_median} -> "
             f"{l.cpu_latency_ms_median} ms    mAP@50 {b.mAP50:.3f} -> {l.mAP50:.3f}",
             fontsize=20, color=FG)
    fig.text(0.05, 0.12, "PyTorch · Ultralytics YOLO · ONNX Runtime · INT8 quantization · Gradio",
             fontsize=17, color=MUTED)
    fig.text(0.05, 0.06, "Demo on public data. Not a medical device and not for clinical use.",
             fontsize=14, color=MUTED)
    fig.savefig(A / "cover_1600x1200.png", dpi=100)
    plt.close(fig)
    print(headline)


if __name__ == "__main__":
    main()
