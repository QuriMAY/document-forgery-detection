#!/usr/bin/env python3
"""
Train YOLOv8 for stamp and signature detection.

Prerequisites:
  1. Label document images on Roboflow (classes: signature, stamp)
  2. Export as "YOLOv8" format — you'll get a data.yaml file

Usage:
    python train_yolo.py --data path/to/data.yaml
    python train_yolo.py --data path/to/data.yaml --device 0 --config config.yaml
"""

import argparse
import logging
import shutil
from pathlib import Path

from src.utils import ensure_dirs, load_config, log_config, setup_logging

logger = logging.getLogger(__name__)


def train(config: dict, args):
    from ultralytics import YOLO

    cfg = config["yolo"]
    ensure_dirs("models", "runs/yolo")
    logger.info(f"Data YAML     : {args.data}")
    logger.info(f"Device        : {args.device}")
    log_config(config, logger)

    model = YOLO(cfg.get("base_model", "yolov8s.pt"))

    results = model.train(
        data=args.data,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        device=args.device,
        project="runs/yolo",
        name="stamp_sig_detector",
        save=True,
        plots=True,
        val=True,
        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    best_pt = Path("runs/yolo/stamp_sig_detector/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, "models/best.pt")
        logger.info("Best weights copied → models/best.pt")

    metrics = model.val()
    logger.info("=" * 70)
    logger.info("YOLO training finished.")
    logger.info(f"  mAP50      : {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95   : {metrics.box.map:.4f}")
    logger.info(f"  Model saved: models/best.pt")
    logger.info("=" * 70)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 stamp/signature detector")
    p.add_argument("--data", required=True, help="Path to data.yaml (Roboflow export)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cpu", help="GPU id (0, 0,1), 'cpu', or 'auto'")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("train_yolo")
    args = parse_args()
    cfg  = load_config(args.config)
    train(cfg, args)
