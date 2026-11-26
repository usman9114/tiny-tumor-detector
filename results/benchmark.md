| Model | Params (M) | Size (MB) | GFLOPs | mAP@50 | mAP@50-95 | Tumour recall | CPU ms (median) | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|
| YOLOv8s baseline (PyTorch FP32) | 11.13 | 44.55 | 12.01 | 0.988 | 0.730 | 0.922 | 46.29 | 484 |
| YOLOv8n (PyTorch FP32) | 3.13 | 12.58 | 3.65 | 0.982 | 0.705 | 0.924 | 19.63 | 434 |
| YOLO11n (PyTorch FP32) | 2.60 | 10.46 | 2.70 | 0.974 | 0.700 | 0.907 | 23.62 | 447 |
| YOLOv8s (ONNX FP32) | 11.13 | 44.65 | 12.01 | 0.987 | 0.728 | 0.902 | 29.15 | 196 |
| YOLO11n (ONNX FP32) | 2.60 | 10.57 | 2.70 | 0.980 | 0.695 | 0.924 | 11.97 | 130 |
| YOLO11n (ONNX FP16) | 2.60 | 5.34 | 2.70 | 0.980 | 0.692 | 0.924 | 10.22 | 140 |
| YOLO11n (ONNX INT8) | 2.60 | 3.33 | 2.70 | 0.981 | 0.698 | 0.932 | 10.70 | 103 |
| YOLOv8n (ONNX FP32) | 3.13 | 12.68 | 3.65 | 0.980 | 0.704 | 0.922 | 13.01 | 121 |
| YOLOv8n (ONNX FP16) | 3.13 | 6.38 | 3.65 | 0.980 | 0.706 | 0.922 | 11.20 | 134 |
| YOLOv8n (ONNX INT8) | 3.13 | 3.57 | 3.65 | 0.973 | 0.690 | 0.915 | 10.35 | 99 |

GPU not used: all latency numbers are CPU only (no GPU was available).

Test split: 110 images, imgsz 416. CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 threads, batch 1, forward pass only, median of 150 runs (interleaved, 10 warm-up). Load average at start: 3.03.
