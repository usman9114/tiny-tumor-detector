"""Export a trained model to ONNX (FP32, FP16) and INT8 (ONNX Runtime static
post-training quantization, calibrated on training images).

Usage: python compress.py --weights weights/light_yolov8n.pt --name light_yolov8n --imgsz 416
Outputs: weights/<name>.onnx, <name>_fp16.onnx, <name>_int8.onnx
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def preprocess(path, size):
    img = cv2.imread(str(path))
    x = letterbox(img, size)[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


class Calib(CalibrationDataReader):
    def __init__(self, files, size, input_name):
        self.it = iter([{input_name: preprocess(f, size)} for f in files])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--calib", type=int, default=128)
    a = ap.parse_args()

    wdir = ROOT / "weights"
    fp32 = wdir / f"{a.name}.onnx"
    exported = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True,
                                      opset=17, dynamic=False, half=False)
    Path(exported).replace(fp32)
    print("fp32 ->", fp32, round(fp32.stat().st_size / 1e6, 2), "MB")

    # FP16: convert weights, keep float32 inputs/outputs
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m16 = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
    fp16 = wdir / f"{a.name}_fp16.onnx"
    onnx.save(m16, str(fp16))
    print("fp16 ->", fp16, round(fp16.stat().st_size / 1e6, 2), "MB")

    # INT8 static PTQ (QDQ format, per-channel weights)
    pre = wdir / f"{a.name}_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    files = sorted((ROOT / "datasets" / "brain-tumor" / "images" / "train").glob("*.jpg"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), a.calib, replace=False)]
    name_in = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    int8 = wdir / f"{a.name}_int8.onnx"
    quantize_static(str(pre), str(int8), Calib(files, a.imgsz, name_in),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()
    print("int8 ->", int8, round(int8.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
