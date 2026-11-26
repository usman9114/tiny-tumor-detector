"""Evaluate every variant on the same held-out test split and write
results/benchmark.csv and results/benchmark.md.

Accuracy : Ultralytics val() on split=test (same code path for .pt and .onnx),
           conf=0.001, iou=0.6 (library defaults for mAP).
Recall   : recall of the tumour class at the confidence that maximises F1.
Latency  : forward pass only (no pre/post-processing), batch 1, fixed thread
           count, median over N timed runs after warm-up. Variants are timed
           interleaved over several rounds so background load hits all alike.
GPU      : a GPU latency column is added only when a CUDA GPU is available; otherwise
           the column is omitted and the table footnote says all latency is CPU only.
--from-csv re-renders benchmark.md/csv from the existing CSV without re-timing.
Peak RAM : max RSS of a fresh process that loads the model and runs 20 inferences.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "brain-tumor" / "data.yaml"
W = ROOT / "weights"
THREADS = 4


def variants():
    """(label, file, source_pt_for_params)"""
    v = []
    for label, pt in [("YOLOv8s baseline (PyTorch FP32)", "baseline_yolov8s"),
                      ("YOLOv8n (PyTorch FP32)", "light_yolov8n"),
                      ("YOLO11n (PyTorch FP32)", "light_yolo11n")]:
        if (W / f"{pt}.pt").exists():
            v.append((label, W / f"{pt}.pt", pt))
    for f in sorted(W.glob("*.onnx")):
        stem = f.stem
        base = stem.rsplit("_", 1)[0] if stem.endswith(("_fp16", "_int8")) else stem
        if base == "baseline_yolov8s" and stem != base:
            continue  # baseline is compared in PyTorch FP32 and ONNX FP32 only
        prec = {"fp16": "FP16", "int8": "INT8"}.get(stem.rsplit("_", 1)[-1], "FP32")
        pretty = {"baseline_yolov8s": "YOLOv8s", "light_yolov8n": "YOLOv8n",
                  "light_yolo11n": "YOLO11n"}.get(base, base)
        v.append((f"{pretty} (ONNX {prec})", f, base))
    return v


def load_runner(path: Path, imgsz: int):
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    if path.suffix == ".pt":
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(THREADS)
        net = YOLO(str(path)).model.float().fuse().eval()
        t = torch.from_numpy(x)

        def run():
            with torch.inference_mode():
                net(t)
        return run
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return lambda: s.run(None, {name: x})


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def gpu_latency(path: Path, imgsz, runs=100):
    """Median forward-pass ms on CUDA (batch 1). Only called when a GPU exists."""
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    if path.suffix == ".pt":
        import torch
        from ultralytics import YOLO
        net = YOLO(str(path)).model.float().fuse().eval().cuda()
        t = torch.from_numpy(x).cuda()

        def run():
            with torch.inference_mode():
                net(t)
            torch.cuda.synchronize()
    else:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
        s = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
        name = s.get_inputs()[0].name

        def run():
            s.run(None, {name: x})
    for _ in range(10):
        run()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def worker_ram(path, imgsz):
    run = load_runner(Path(path), imgsz)
    for _ in range(20):
        run()
    # VmHWM = peak resident set of this process (ru_maxrss is inherited across exec)
    hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(json.dumps({"ram_mb": hwm / 1024}))


def peak_ram(path, imgsz):
    out = subprocess.run([sys.executable, __file__, "--ram-worker", str(path), "--imgsz", str(imgsz)],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)["ram_mb"]


def model_stats(pt: Path, imgsz):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = YOLO(str(pt))
    net = m.model.float().fuse()
    params = sum(p.numel() for p in net.parameters())
    gflops = get_flops(net, imgsz)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(net.state_dict(), f.name)
        size = os.path.getsize(f.name) / 1e6
    return params / 1e6, gflops, size


def accuracy(path: Path, imgsz):
    from ultralytics import YOLO
    r = YOLO(str(path), task="detect").val(data=str(DATA), split="test", imgsz=imgsz, batch=1,
                                           device="cpu", plots=False, verbose=False)
    return float(r.box.map50), float(r.box.map), float(r.box.mr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ram-worker")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-render benchmark.md/csv from the existing CSV (no timing)")
    a = ap.parse_args()
    if a.from_csv:
        return from_csv()
    if a.ram_worker:
        return worker_ram(a.ram_worker, a.imgsz)

    import torch
    torch.set_num_threads(THREADS)
    load = os.getloadavg()
    print("load average at start:", load)
    cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))

    use_gpu = cuda_available()
    vs = variants()
    rows, runners, times = [], {}, {}
    for label, path, src in vs:
        print("loading", label)
        runners[label] = load_runner(path, a.imgsz)
        for _ in range(10):  # warm-up
            runners[label]()
        times[label] = []
    for r in range(a.rounds):  # interleaved timing
        for label, _, _ in vs:
            for _ in range(a.runs):
                t0 = time.perf_counter()
                runners[label]()
                times[label].append((time.perf_counter() - t0) * 1000)
    del runners

    stats_cache = {}
    for label, path, src in vs:
        print("evaluating", label)
        if src not in stats_cache:
            stats_cache[src] = model_stats(W / f"{src}.pt", a.imgsz)
        params, gflops, size_fp32 = stats_cache[src]
        m50, m5095, rec = accuracy(path, a.imgsz)
        g = gpu_latency(path, a.imgsz) if use_gpu else None
        t = np.array(times[label])
        rows.append({
            "model": label, "params_M": round(params, 2),
            "size_MB": round(path.stat().st_size / 1e6, 2) if path.suffix == ".onnx" else round(size_fp32, 2),
            "GFLOPs": round(gflops, 2), "mAP50": round(m50, 4), "mAP50_95": round(m5095, 4),
            "tumour_recall": round(rec, 4),
            "cpu_latency_ms_median": round(float(np.median(t)), 2),
            "cpu_latency_ms_p95": round(float(np.percentile(t, 95)), 2),
            "peak_ram_MB": round(peak_ram(path, a.imgsz), 0),
        })
        if use_gpu:
            rows[-1]["gpu_latency_ms"] = round(g, 2) if g is not None else ""
    note = (f"\nTest split: {len(list((ROOT / 'datasets/brain-tumor/images/test').glob('*.jpg')))} images, "
            f"imgsz {a.imgsz}. CPU: {cpu}, {THREADS} threads, batch 1, forward pass only, "
            f"median of {a.runs * a.rounds} runs (interleaved, 10 warm-up). "
            f"Load average at start: {load[0]:.2f}.")
    write_outputs(rows, note, gpu=use_gpu)


NO_GPU_FOOTNOTE = "GPU not used: all latency numbers are CPU only (no GPU was available)."


def fmt_row(r, gpu):
    """Same decimals per column everywhere: params/GFLOPs/size/ms 2, mAP/recall 3, RAM integer."""
    cells = [r["model"], f"{float(r['params_M']):.2f}", f"{float(r['size_MB']):.2f}",
             f"{float(r['GFLOPs']):.2f}", f"{float(r['mAP50']):.3f}", f"{float(r['mAP50_95']):.3f}",
             f"{float(r['tumour_recall']):.3f}", f"{float(r['cpu_latency_ms_median']):.2f}"]
    if gpu:
        g = r.get("gpu_latency_ms")
        cells.append("n/a" if g in (None, "") else f"{float(g):.2f}")
    cells.append(f"{int(round(float(r['peak_ram_MB'])))}")
    return cells


def write_outputs(rows, note, gpu=False):
    res = ROOT / "results"
    cols = ["model", "params_M", "size_MB", "GFLOPs", "mAP50", "mAP50_95", "tumour_recall",
            "cpu_latency_ms_median", "cpu_latency_ms_p95"] + (["gpu_latency_ms"] if gpu else []) + ["peak_ram_MB"]
    two = {"params_M", "size_MB", "GFLOPs", "cpu_latency_ms_median", "cpu_latency_ms_p95", "gpu_latency_ms"}
    with open(res / "benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            out = []
            for c in cols:
                v = r.get(c)
                if c in two and v not in (None, ""):
                    v = f"{float(v):.2f}"
                elif c in ("mAP50", "mAP50_95", "tumour_recall"):
                    v = f"{float(v):.4f}"
                elif c == "peak_ram_MB":
                    v = int(round(float(v)))
                out.append(v)
            w.writerow(out)
    hdr = ["Model", "Params (M)", "Size (MB)", "GFLOPs", "mAP@50", "mAP@50-95", "Tumour recall",
           "CPU ms (median)"] + (["GPU ms"] if gpu else []) + ["Peak RAM (MB)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines += ["| " + " | ".join(fmt_row(r, gpu)) + " |" for r in rows]
    foot = "" if gpu else "\n" + NO_GPU_FOOTNOTE + "\n\n"
    text = "\n".join(lines) + "\n" + foot + note.rstrip() + "\n"
    (res / "benchmark.md").write_text(text)
    print(text)


def from_csv():
    """Re-render results/benchmark.{csv,md} from the existing CSV, without re-timing."""
    res = ROOT / "results"
    with open(res / "benchmark.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    gpu = "gpu_latency_ms" in rows[0] and any(
        re.match(r"^[0-9.]+$", r["gpu_latency_ms"] or "") for r in rows)
    md = (res / "benchmark.md").read_text().split("\n")
    note = next((l for l in md if l.startswith("Test split:")), "")
    note = re.sub(r"\s*GPU latency: [^.]*\.[^ ]*$", "", note).strip()
    if not gpu:
        for r in rows:
            r.pop("gpu_latency_ms", None)
    write_outputs(rows, note, gpu=gpu)


if __name__ == "__main__":
    main()
