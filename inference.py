#!/usr/bin/env python3
"""
Run forgery detection on a single image or a whole directory.

Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/folder/ --visualize --output results.json
"""

import argparse
import json
import logging
from pathlib import Path

from src.analysis import analyze
from src.classifier import ForgeryClassifier
from src.detector import DocumentDetector
from src.ela import ELAAnalyzer
from src.utils import load_config, log_config, setup_logging

logger = logging.getLogger(__name__)

_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Image file or directory")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--visualize", action="store_true", help="Save ELA visualizations")
    p.add_argument("--output", default=None, help="Write JSON results to this file")
    args = p.parse_args()

    setup_logging("inference")
    cfg = load_config(args.config)
    log_config(cfg, logging.getLogger("inference"))

    ela        = ELAAnalyzer(cfg.get("ela", {}))
    detector   = DocumentDetector()
    classifier = ForgeryClassifier() if Path("models/classifier.pth").exists() else None

    logger.info(f"Classifier : {'loaded' if classifier else 'NOT FOUND — ELA score only'}")
    logger.info(f"YOLO       : {'loaded' if detector.is_available else 'NOT FOUND — no bbox detection'}")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return

    images = [input_path] if input_path.is_file() else [
        p for p in input_path.iterdir() if p.suffix.lower() in _EXTS
    ]

    if not images:
        logger.error("No images found.")
        return

    logger.info(f"Analyzing {len(images)} image(s) from: {input_path}")
    logger.info("─" * 70)
    results = []
    for img in sorted(images):
        result = analyze(str(img), ela, detector, classifier, cfg)
        results.append(result)
        verdict = "FORGED ⚠" if result["is_forged"] else "REAL   ✓"
        sigs = len(result["detections"].get("signatures", []))
        stamps = len(result["detections"].get("stamps", []))
        logger.info(
            f"{img.name:40s}  {verdict}  "
            f"prob={result['forgery_probability']:.3f}  "
            f"ela={result['scores']['ela']:.3f}  "
            f"sigs={sigs}  stamps={stamps}"
        )

        if args.visualize:
            ela.visualize(str(img), save_path=f"ela_{img.stem}.png")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
