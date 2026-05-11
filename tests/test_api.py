"""FastAPI smoke tests. The lifespan handler loads models from disk; for these
tests we override `app.state` after startup so we don't need real weights."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.ela import ELAAnalyzer
from tests.conftest import StubClassifier, StubDetector


@pytest.fixture
def client():
    with TestClient(app) as c:
        # Replace whatever lifespan loaded with deterministic stubs.
        c.app.state.ela = ELAAnalyzer()
        c.app.state.detector = StubDetector()
        c.app.state.classifier = StubClassifier(prob=0.9)
        c.app.state.config = {"inference": {"ela_weight": 0.35,
                                            "classifier_weight": 0.65,
                                            "forgery_threshold": 0.5}}
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_analyze_happy_path(client, jpeg_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("doc.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["confidence"] in {"low", "medium", "high"}
    assert 0.0 <= body["forgery_probability"] <= 1.0
    assert "ela" in body["scores"]


def test_analyze_rejects_bad_mime(client, jpeg_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("doc.jpg", jpeg_bytes, "application/pdf")},
    )
    assert r.status_code == 415


def test_analyze_rejects_spoofed_magic_bytes(client):
    """Content-Type says JPEG but the body is plain text — magic-byte check must reject."""
    r = client.post(
        "/analyze",
        files={"file": ("evil.jpg", b"not an image at all", "image/jpeg")},
    )
    assert r.status_code == 415


def test_analyze_rejects_oversized_file(client, monkeypatch):
    # Shrink the cap so we don't actually have to allocate 10MB.
    import api.main as api_main
    import src.constants as constants
    monkeypatch.setattr(api_main, "MAX_FILE_BYTES", 1024)
    monkeypatch.setattr(constants, "MAX_FILE_BYTES", 1024)

    # 2KB JPEG > 1KB cap.
    big = b"\xff\xd8\xff" + b"\x00" * 2048
    r = client.post(
        "/analyze",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413


def test_analyze_accepts_png(client, png_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("doc.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
