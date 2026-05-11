"""
Document Forgery Detection — FastAPI backend.

Endpoints:
    POST /analyze  — upload a document image, returns forgery verdict + scores
    GET  /health   — liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.analysis import analyze
from src.classifier import ForgeryClassifier
from src.constants import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_IMAGE_FORMATS,
    MAX_FILE_BYTES,
)
from src.detector import DocumentDetector
from src.ela import ELAAnalyzer
from src.utils import load_config, setup_logging

setup_logging("api")
logger = logging.getLogger(__name__)

TEMP_DIR = Path("temp")


def _allowed_origins() -> list[str]:
    """Comma-separated CORS origins from CORS_ALLOW_ORIGINS env var.
    Defaults to localhost dev origins; set to a real domain list in prod."""
    raw = os.environ.get(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:7860,http://localhost:3000,http://localhost:3001",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    TEMP_DIR.mkdir(exist_ok=True)

    cfg = load_config()
    app.state.config = cfg
    app.state.ela = ELAAnalyzer(cfg.get("ela", {}))
    app.state.detector = DocumentDetector()

    try:
        app.state.classifier = (
            ForgeryClassifier() if Path("models/classifier.pth").exists() else None
        )
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        logger.warning("Classifier failed to load: %s", exc)
        app.state.classifier = None

    cls_status = "loaded" if app.state.classifier else "not found"
    yolo_status = "loaded" if app.state.detector.is_available else "not found"
    logger.info("API ready | classifier=%s | yolo=%s", cls_status, yolo_status)

    yield

    logger.info("Shutting down API")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Forgery Detection API",
    description="Detects tampered signatures, stamps, and copy-move forgeries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _validate_image_bytes(content: bytes) -> str:
    """Verify magic bytes match an allowed image format. Returns the detected
    format (lowercase). Raises HTTPException(415) on mismatch."""
    try:
        with Image.open(io.BytesIO(content)) as probe:
            fmt = (probe.format or "").lower()
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=415,
            detail=f"File is not a valid image: {exc}",
        ) from exc

    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format: {fmt}. Allowed: {sorted(ALLOWED_IMAGE_FORMATS)}",
        )
    return fmt


@app.post("/analyze", summary="Analyze a document image for forgery")
async def analyze_document(file: UploadFile = File(...)):
    # Cheap MIME check first — fail fast on obvious mismatches.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    # Streaming size cap: stop reading after MAX_FILE_BYTES + 1.
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    # Magic-byte validation — defeats spoofed Content-Type headers.
    _validate_image_bytes(content)

    temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"

    try:
        temp_path.write_bytes(content)
        t0 = time.perf_counter()

        result = analyze(
            str(temp_path),
            app.state.ela,
            app.state.detector,
            app.state.classifier,
            app.state.config,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return JSONResponse({
            "status": "success",
            "is_forged": result["is_forged"],
            "forgery_probability": result["forgery_probability"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "detections": result["detections"],
            "suspicious_regions": result["suspicious_regions"],
            "processing_time_ms": elapsed_ms,
        })

    except HTTPException:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/health", summary="Service health check")
def health():
    return {
        "status": "running",
        "models": {
            "classifier": app.state.classifier is not None,
            "yolo": app.state.detector.is_available,
        },
    }
