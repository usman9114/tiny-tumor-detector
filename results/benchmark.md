| Model | Params (M) | Size (MB) | GFLOPs | mAP@50 | mAP@50-95 | Tumour recall | CPU ms (median) | GPU ms | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8s baseline (PyTorch FP32) | 11.13 | 44.55 | 12.01 | 0.988 | 0.730 | 0.922 | 47.37 | not measured | 852 |
| YOLOv8n (PyTorch FP32) | 3.13 | 12.58 | 3.65 | 0.982 | 0.705 | 0.924 | 20.03 | not measured | 852 |
| YOLO11n (PyTorch FP32) | 2.6 | 10.46 | 2.7 | 0.974 | 0.700 | 0.907 | 24.39 | not measured | 852 |
| YOLOv8s (ONNX FP32) | 11.13 | 44.65 | 12.01 | 0.987 | 0.728 | 0.902 | 29.4 | not measured | 852 |
| YOLO11n (ONNX FP32) | 2.6 | 10.57 | 2.7 | 0.980 | 0.695 | 0.924 | 12.06 | not measured | 852 |
| YOLO11n (ONNX FP16) | 2.6 | 5.34 | 2.7 | 0.980 | 0.692 | 0.924 | 10.15 | not measured | 852 |
| YOLO11n (ONNX INT8) | 2.6 | 3.33 | 2.7 | 0.981 | 0.698 | 0.932 | 10.66 | not measured | 852 |
| YOLOv8n (ONNX FP32) | 3.13 | 12.68 | 3.65 | 0.980 | 0.704 | 0.922 | 13.22 | not measured | 852 |
| YOLOv8n (ONNX FP16) | 3.13 | 6.38 | 3.65 | 0.980 | 0.706 | 0.922 | 11.26 | not measured | 852 |
| YOLOv8n (ONNX INT8) | 3.13 | 3.57 | 3.65 | 0.973 | 0.690 | 0.915 | 10.15 | not measured | 852 |

Test split: 110 images, imgsz 416. CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 threads, batch 1, forward pass only, median of 150 runs (interleaved, 10 warm-up). Load average at start: 1.93. GPU latency: not measured (no GPU available).
