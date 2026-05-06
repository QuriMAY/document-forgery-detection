#!/usr/bin/env python3
"""
Download and organize Kaggle datasets for the forgery detection pipeline.

Prerequisites:
  1. pip install kaggle
  2. Get your API key from https://www.kaggle.com/settings → API → Create New Token
  3. Place kaggle.json at ~/.kaggle/kaggle.json   (Linux/Mac)
                        C:\\Users\\<you>\\.kaggle\\kaggle.json  (Windows)
     OR set env vars:  KAGGLE_USERNAME and KAGGLE_KEY

Available dataset groups:
  --documents   Real document images  → data/real/
  --signatures  Signature images      → data/patches/  (used for splice attacks)
  --all         Download everything

Usage:
  python kaggle_download.py --all
  python kaggle_download.py --documents
  python kaggle_download.py --signatures
  python kaggle_download.py --documents --limit 2000
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: (kaggle_slug, description, dest_subdir, file_patterns)
# dest_subdir: "real" → data/real/   "patches" → data/patches/
# file_patterns: glob patterns to collect images from extracted folder

DOCUMENT_DATASETS = [
    {
        "slug":        "shaz13/real-world-documents-collections",
        "description": "Real-World Documents (invoices, letters, forms, etc.)",
        "dest":        "real",
        "patterns":    ["**/*.jpg", "**/*.jpeg", "**/*.png"],
        "max_images":  None,
    },
    {
        "slug":        "patrickaudriaz/tobacco3482dataset",
        "description": "Tobacco3482 — scanned office documents",
        "dest":        "real",
        "patterns":    ["**/*.jpg", "**/*.jpeg", "**/*.png"],
        "max_images":  None,
    },
]

SIGNATURE_DATASETS = [
    {
        "slug":        "robinreni/signature-verification-dataset",
        "description": "Signature Verification Dataset (genuine + forged signatures)",
        "dest":        "patches",
        "patterns":    ["**/*.png", "**/*.jpg", "**/*.jpeg"],
        "max_images":  None,
    },
    {
        "slug":        "ishanikathuria/handwritten-signatures",
        "description": "Handwritten Signatures dataset",
        "dest":        "patches",
        "patterns":    ["**/*.png", "**/*.jpg", "**/*.jpeg"],
        "max_images":  None,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_credentials() -> bool:
    """Return True if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()
    has_env  = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

    if not has_file and not has_env:
        logger.error("Kaggle credentials not found.")
        logger.error("  Option 1: place kaggle.json at ~/.kaggle/kaggle.json")
        logger.error("            Download it from https://www.kaggle.com/settings → API")
        logger.error("  Option 2: set env vars KAGGLE_USERNAME and KAGGLE_KEY")
        return False

    if has_file:
        kaggle_json.chmod(0o600)   # Kaggle API requires restricted permissions
        logger.info(f"Credentials found: {kaggle_json}")
    else:
        logger.info("Credentials found: environment variables")

    return True


def get_api():
    """Return authenticated KaggleApi instance."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(api, slug: str, download_dir: Path) -> Path:
    """Download and unzip a Kaggle dataset. Returns extraction path."""
    download_dir.mkdir(parents=True, exist_ok=True)

    owner, dataset = slug.split("/")
    zip_path = download_dir / f"{dataset}.zip"

    logger.info(f"Downloading  {slug}  →  {download_dir}/")
    api.dataset_download_files(
        dataset=slug,
        path=str(download_dir),
        unzip=False,
        quiet=False,
    )

    # Find the downloaded zip (name may differ from dataset slug)
    zips = list(download_dir.glob("*.zip"))
    if not zips:
        logger.warning(f"No zip found in {download_dir} — may already be extracted")
        return download_dir

    zip_file = zips[0]
    extract_dir = download_dir / "extracted"
    logger.info(f"Extracting {zip_file.name} ...")

    with zipfile.ZipFile(zip_file, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_dir)

    zip_file.unlink()
    logger.info(f"Extracted → {extract_dir}")
    return extract_dir


def collect_images(
    src_dir: Path,
    dest_dir: Path,
    patterns: list[str],
    prefix: str,
    limit: int = None,
) -> int:
    """Copy matched images from src_dir into dest_dir with a unique prefix."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[Path] = []
    for pattern in patterns:
        all_files.extend(src_dir.glob(pattern))

    # Deduplicate and sort
    seen, unique = set(), []
    for f in sorted(all_files):
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)

    if limit:
        unique = unique[:limit]

    copied = 0
    for img_path in tqdm(unique, desc=f"Copying → {dest_dir.name}/", unit="img"):
        dest = dest_dir / f"{prefix}_{img_path.stem}{img_path.suffix.lower()}"
        try:
            shutil.copy2(img_path, dest)
            copied += 1
        except Exception as exc:
            logger.debug(f"Skip {img_path.name}: {exc}")

    return copied


def process_dataset(api, entry: dict, data_root: Path, limit: int = None) -> int:
    """Download one dataset entry and move images to the right place."""
    slug = entry["slug"]
    dest_dir = data_root / entry["dest"]

    # Download to a temp staging area
    staging = data_root / "_staging" / slug.replace("/", "_")

    try:
        extract_dir = download_dataset(api, slug, staging)
        prefix = slug.split("/")[1].replace("-", "_")
        n = collect_images(
            src_dir=extract_dir,
            dest_dir=dest_dir,
            patterns=entry["patterns"],
            prefix=prefix,
            limit=limit or entry.get("max_images"),
        )
        logger.info(f"  ✓  {n} images copied → data/{entry['dest']}/")
        return n

    except Exception as exc:
        logger.error(f"Failed to process {slug}: {exc}")
        return 0

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Download Kaggle datasets for forgery detection")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",        action="store_true", help="Download all datasets")
    group.add_argument("--documents",  action="store_true", help="Download document image datasets → data/real/")
    group.add_argument("--signatures", action="store_true", help="Download signature datasets    → data/patches/")

    p.add_argument("--limit", type=int, default=None,
                   help="Max images to copy per dataset (useful for quick tests, e.g. --limit 500)")
    p.add_argument("--data-dir", default="data", help="Root data directory (default: data/)")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Dataset slugs to skip, e.g. --skip shaz13/real-world-documents-collections")
    args = p.parse_args()

    setup_logging("kaggle_download")

    if not check_credentials():
        sys.exit(1)

    try:
        api = get_api()
    except Exception as exc:
        logger.error(f"Kaggle authentication failed: {exc}")
        sys.exit(1)

    data_root = Path(args.data_dir)
    datasets_to_run: list[dict] = []

    if args.all or args.documents:
        datasets_to_run.extend(DOCUMENT_DATASETS)
    if args.all or args.signatures:
        datasets_to_run.extend(SIGNATURE_DATASETS)

    if args.skip:
        before = len(datasets_to_run)
        datasets_to_run = [d for d in datasets_to_run if d["slug"] not in args.skip]
        logger.info(f"Skipped {before - len(datasets_to_run)} dataset(s)")

    if not datasets_to_run:
        logger.warning("Nothing to download.")
        return

    logger.info(f"Will download {len(datasets_to_run)} dataset(s)")
    logger.info("─" * 60)

    total_images = 0
    for i, entry in enumerate(datasets_to_run, 1):
        logger.info(f"[{i}/{len(datasets_to_run)}]  {entry['description']}")
        n = process_dataset(api, entry, data_root, limit=args.limit)
        total_images += n
        logger.info("─" * 60)

    # Summary
    real_count    = sum(1 for _ in (data_root / "real").glob("*")    if _.is_file()) if (data_root / "real").exists()    else 0
    patches_count = sum(1 for _ in (data_root / "patches").glob("*") if _.is_file()) if (data_root / "patches").exists() else 0

    logger.info("Download complete.")
    logger.info(f"  data/real/    : {real_count} images")
    logger.info(f"  data/patches/ : {patches_count} images")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Generate synthetic forgeries:")
    logger.info("       python scripts/generate_synthetic_data.py \\")
    logger.info("           --real-dir data/real --donor-dir data/patches --num-forged 1000")
    logger.info("  2. Split into train/val/test:")
    logger.info("       python scripts/prepare_dataset.py --source-dir data/generated --output-dir data")
    logger.info("  3. Train:")
    logger.info("       python train_classifier.py")


if __name__ == "__main__":
    main()
