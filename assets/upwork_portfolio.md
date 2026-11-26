# Upwork portfolio item

**Title (max 40 chars):** Shrinking a YOLO tumour detector 12x

**Role:** ML engineer (solo demo project)

**Description (about 300 chars):**
Demo on public data. Heavy YOLO detectors are hard to run on CPU-only hardware. I trained YOLOv8s and YOLOv8n/YOLO11n on a public brain-MRI set, then exported YOLOv8n to ONNX INT8. Measured: 12.5x smaller (44.6 to 3.6 MB), 4.5x faster on CPU (46 to 10 ms), mAP@50 -1.4 points. Not for clinical use.

**Skills tags:** Computer Vision, PyTorch, YOLO, Model Optimization, ONNX, Deep Learning

**Media order:** cover_1600x1200.png, tradeoff_chart.png, before_after.png, demo.gif

**Repo:** https://github.com/usman9114/tiny-tumor-detector

Numbers come from `results/benchmark.csv` (CPU latency: 4 threads, batch 1, forward pass only; no GPU used, all latency is CPU only).
