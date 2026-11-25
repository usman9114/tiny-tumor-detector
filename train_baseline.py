"""Train the baseline detector (YOLOv8s by default) on the prepared split.

Saves weights to weights/<name>.pt and metrics to results/train_<name>.json.
Resumable: re-running the same command continues from runs/<name>/weights/last.pt.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "brain-tumor" / "data.yaml"


def train(weights: str, name: str, imgsz: int, epochs: int, batch: int, workers: int = 4):
    last = ROOT / "runs" / name / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(weights)
        model.train(
            data=str(DATA), imgsz=imgsz, epochs=epochs, batch=batch, workers=workers,
            device="cpu", project=str(ROOT / "runs"), name=name, exist_ok=True,
            seed=0, deterministic=True, patience=0, plots=False, amp=False,
            pretrained=True, cache=True,
        )
    wall = time.time() - t0
    best = ROOT / "runs" / name / "weights" / "best.pt"
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / f"{name}.pt")
    m = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=imgsz, device="cpu",
                            batch=8, plots=False, verbose=False)
    out = {
        "name": name, "base_weights": weights, "imgsz": imgsz, "epochs": epochs,
        "batch": batch, "train_wall_seconds_this_invocation": round(wall, 1),
        "val_map50": float(m.box.map50), "val_map50_95": float(m.box.map),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"train_{name}.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--name", default="baseline_yolov8s")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    train(a.weights, a.name, a.imgsz, a.epochs, a.batch)
