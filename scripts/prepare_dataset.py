"""
Split a labelled folder into stratified train / val / test splits.

Expected input structure:
    source_dir/
        real/    ← authentic images
        forged/  ← tampered images

Writes:
    output_dir/train/real/
    output_dir/train/forged/
    output_dir/val/real/
    output_dir/val/forged/
    output_dir/test/real/
    output_dir/test/forged/

Usage:
    python scripts/prepare_dataset.py --source-dir data/generated --output-dir data
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

from src.utils import setup_logging

logger = logging.getLogger(__name__)
_EXTS = {".jpg", ".jpeg", ".png"}


def split_class(
    cls_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
) -> dict:
    images = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in _EXTS])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    for split, imgs in splits.items():
        dest = output_dir / split / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest / img.name)

    return {k: len(v) for k, v in splits.items()}


def main():
    p = argparse.ArgumentParser(description="Train/val/test splitter")
    p.add_argument("--source-dir",   required=True, help="Directory with class sub-folders")
    p.add_argument("--output-dir",   default="data")
    p.add_argument("--train-ratio",  type=float, default=0.70)
    p.add_argument("--val-ratio",    type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    setup_logging("prepare_dataset")
    random.seed(args.seed)
    logger.info(f"Source dir   : {args.source_dir}")
    logger.info(f"Output dir   : {args.output_dir}")
    logger.info(f"Split ratio  : train={args.train_ratio}  val={args.val_ratio}  test={round(1-args.train_ratio-args.val_ratio, 2)}")
    logger.info(f"Seed         : {args.seed}")

    source = Path(args.source_dir)
    output = Path(args.output_dir)

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No class directories found in {source}")
        sys.exit(1)

    logger.info(f"Classes found: {[d.name for d in class_dirs]}")

    total = {"train": 0, "val": 0, "test": 0}
    for cls_dir in class_dirs:
        counts = split_class(cls_dir, output, args.train_ratio, args.val_ratio)
        for split, n in counts.items():
            total[split] += n
        logger.info(f"  {cls_dir.name:10s}  train={counts['train']}  val={counts['val']}  test={counts['test']}")

    logger.info("=" * 70)
    logger.info("Split complete.")
    logger.info(f"  train: {total['train']} images")
    logger.info(f"  val  : {total['val']} images")
    logger.info(f"  test : {total['test']} images")
    logger.info(f"  Output → {output}/train  /val  /test")
    logger.info("Next step: python train_classifier.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
