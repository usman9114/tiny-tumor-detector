"""Train the lightweight candidates (YOLOv8n and YOLO11n) with the same
data, image size, epochs and seed as the baseline.

A torchvision mobile-backbone detector (SSDLite320-MobileNetV3) was considered
and dropped: it needs a separate training loop and evaluation path, which does
not fit the CPU-only time budget and would make the comparison less like-for-like.
"""
import argparse

from train_baseline import train

CANDIDATES = {"yolov8n": "yolov8n.pt", "yolo11n": "yolo11n.pt"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    for m in a.models:
        train(CANDIDATES[m], f"light_{m}", a.imgsz, a.epochs, a.batch)
