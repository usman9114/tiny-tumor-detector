#!/usr/bin/env bash
# Train all models sequentially with identical settings (CPU budget).
set -e
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-25}
python train_light.py --epochs $EPOCHS --imgsz 416
python train_baseline.py --epochs $EPOCHS --imgsz 416
