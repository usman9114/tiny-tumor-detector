"""Gradio demo: run the baseline and the light model side by side on one MRI slice.

Demo on public data. Not a medical device and not for clinical use.

Env vars (defaults in brackets): BASE_MODEL [weights/baseline_yolov8s.pt],
LIGHT_MODEL [weights/light_yolov8n_int8.onnx], IMGSZ [416], PORT [7860].
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("BASE_MODEL", str(ROOT / "weights" / "baseline_yolov8s.pt"))
LIGHT = os.environ.get("LIGHT_MODEL", str(ROOT / "weights" / "light_yolov8n_int8.onnx"))
IMGSZ = int(os.environ.get("IMGSZ", 416))
DISCLAIMER = "Demo on public data. Not a medical device and not for clinical use."

_models = {}


def missing_weights():
    return [p for p in (BASE, LIGHT) if not Path(p).exists()]


def get(path):
    if path not in _models:
        _models[path] = YOLO(path, task="detect")
    return _models[path]


def run(path, img, conf):
    r = get(path).predict(img, imgsz=IMGSZ, conf=conf, device="cpu", verbose=False)[0]
    drawn = r.plot()[:, :, ::-1]
    ms = r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]
    return drawn, f"{ms:.1f} ms end-to-end, {len(r.boxes)} box(es)"


def compare(img, conf):
    gone = missing_weights()
    if gone:
        raise gr.Error("Model weights not found: " + ", ".join(Path(p).name for p in gone)
                       + ". Put them in the weights/ folder (Docker: mount it with "
                       "-v \"$PWD/weights:/app/weights\"). See the README, section Run it.")
    if img is None:
        return None, "", None, ""
    img = np.asarray(img)[:, :, ::-1]  # RGB -> BGR for ultralytics
    run(BASE, img, conf), run(LIGHT, img, conf)  # warm both once so latency is fair
    a, ta = run(BASE, img, conf)
    b, tb = run(LIGHT, img, conf)
    return a, ta, b, tb


examples = sorted((ROOT / "assets" / "examples").glob("*.jpg")) if (ROOT / "assets" / "examples").exists() else []

with gr.Blocks(title="Tiny tumor detector") as demo:
    gr.Markdown(f"# Tiny tumor detector\nBaseline vs light model on one MRI slice.\n\n**{DISCLAIMER}**")
    with gr.Row():
        inp = gr.Image(type="numpy", label="MRI slice", sources=["upload"])
        conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
    btn = gr.Button("Detect", variant="primary", elem_id="go")
    with gr.Row():
        with gr.Column():
            o1 = gr.Image(label=f"Baseline: {Path(BASE).name}")
            t1 = gr.Textbox(label="Baseline latency (CPU)", elem_id="t1")
        with gr.Column():
            o2 = gr.Image(label=f"Light: {Path(LIGHT).name}")
            t2 = gr.Textbox(label="Light latency (CPU)", elem_id="t2")
    if examples:
        gr.Examples([[str(p), 0.25] for p in examples], [inp, conf])
    btn.click(compare, [inp, conf], [o1, t1, o2, t2])

if __name__ == "__main__":
    if missing_weights():
        print("WARNING: weights not found: " + ", ".join(missing_weights())
              + "\nThe page will load, but Detect will show an error until the weights are in place "
              "(Docker: -v \"$PWD/weights:/app/weights\").", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
