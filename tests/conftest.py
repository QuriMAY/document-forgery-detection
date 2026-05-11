"""Shared fixtures: tiny synthetic JPEGs and stub models so tests stay fast
and never touch the network or real checkpoints."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make `src.` imports resolve when pytest is invoked from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Redirect logs to a tmp dir so tests never touch the repo's logs/ tree
# (which may not be writable when previous Docker runs created it as root).
_LOG_TMP = Path(tempfile.mkdtemp(prefix="docforg-test-logs-"))
os.environ.setdefault("LOG_DIR", str(_LOG_TMP))


def _make_image_bytes(size: tuple[int, int] = (64, 64), fmt: str = "JPEG") -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(*size[::-1], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, quality=90)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return _make_image_bytes(fmt="JPEG")


@pytest.fixture
def png_bytes() -> bytes:
    return _make_image_bytes(fmt="PNG")


@pytest.fixture
def jpeg_path(tmp_path, jpeg_bytes) -> Path:
    p = tmp_path / "sample.jpg"
    p.write_bytes(jpeg_bytes)
    return p


class StubDetector:
    """Mimics src.detector.DocumentDetector without loading YOLO."""
    is_available = False

    def detect(self, _path: str) -> dict:
        return {"signatures": [], "stamps": []}


class StubClassifier:
    """Returns a fixed forgery probability."""
    def __init__(self, prob: float = 0.8):
        self.prob = prob

    def predict(self, _path: str) -> dict:
        return {
            "is_forged": self.prob > 0.5,
            "forgery_probability": self.prob,
            "real_probability": 1.0 - self.prob,
        }


@pytest.fixture
def stub_detector() -> StubDetector:
    return StubDetector()


@pytest.fixture
def stub_classifier() -> StubClassifier:
    return StubClassifier()
