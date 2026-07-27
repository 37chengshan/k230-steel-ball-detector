#!/usr/bin/env python3
"""Train the same NMS-free YOLO26 family used by the reference project."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "training" / "data" / "reference_v1" / "steel_ball_reference.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo26n.pt"
DEFAULT_PROJECT = ROOT / "training" / "runs" / "detect"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO26n steel-ball detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="steel_ball_reference_yolo26n_1024")
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        raise SystemExit("dataset or YOLO26 pretrained weights missing")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print("data=%s weights=%s device=%s" % (args.data, args.weights, device))
    YOLO(str(args.weights)).train(
        data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=device, workers=0, project=str(DEFAULT_PROJECT), name=args.name,
        exist_ok=True, pretrained=True, optimizer="AdamW", lr0=0.001, lrf=0.01,
        patience=15, seed=20260727, deterministic=True,
        degrees=5.0, translate=0.06, scale=0.25, shear=0.5, perspective=0.0,
        hsv_h=0.006, hsv_s=0.18, hsv_v=0.20, fliplr=0.5, flipud=0.0,
        mosaic=0.30, close_mosaic=10, mixup=0.0, copy_paste=0.0,
        erasing=0.02, cache="ram", amp=True, plots=True,
    )


if __name__ == "__main__":
    main()
