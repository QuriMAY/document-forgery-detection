#!/usr/bin/env python3
"""
Generate synthetic forged documents from a folder of real images.

Usage:
    python scripts/generate_synthetic_data.py \
        --real-dir data/real \
        --output-dir data/generated \
        --num-forged 1000

    # With donor images for splice attacks:
    python scripts/generate_synthetic_data.py \
        --real-dir data/real \
        --donor-dir data/patches \
        --output-dir data/generated \
        --num-forged 1000
"""

import argparse
import logging
from pathlib import Path

from src.data_generator import SyntheticForgeryGenerator
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Synthetic forgery dataset generator")
    p.add_argument("--real-dir",    required=True, help="Directory of authentic document images")
    p.add_argument("--output-dir",  default="data/generated")
    p.add_argument("--donor-dir",   default=None,  help="Optional images used as splice patches")
    p.add_argument("--num-forged",  type=int, default=None, help="Forgeries to generate (default = #real)")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    setup_logging("generate_synthetic_data")
    logger.info("Starting synthetic data generation")
    logger.info(f"  real_dir   : {args.real_dir}")
    logger.info(f"  output_dir : {args.output_dir}")
    logger.info(f"  donor_dir  : {args.donor_dir or 'none (copy-move only)'}")
    logger.info(f"  num_forged : {args.num_forged or 'same count as real images'}")
    logger.info(f"  seed       : {args.seed}")

    gen = SyntheticForgeryGenerator(seed=args.seed)
    stats = gen.generate_dataset(
        real_dir=args.real_dir,
        output_dir=args.output_dir,
        num_forged=args.num_forged,
        donor_dir=args.donor_dir,
    )

    logger.info("=" * 70)
    logger.info("Generation complete.")
    logger.info(f"  Real images  : {stats['real_count']}")
    logger.info(f"  Forged images: {stats['forged_count']}")
    logger.info(f"  Errors       : {stats['errors']}")
    logger.info(f"  Output       : {args.output_dir}/real  and  {args.output_dir}/forged")
    logger.info("Next step: python scripts/prepare_dataset.py --source-dir data/generated --output-dir data")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
