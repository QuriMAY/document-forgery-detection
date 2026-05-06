"""
Document Forgery Detection — FastAPI backend.

Endpoints:
    POST /analyze  — upload a document image, returns forgery verdict + scores
    GET  /health   — liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.ela import ELAAnalyzer
from src.detector import DocumentDetector
from src.classifier import ForgeryClassifier
from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TEMP_DIR = Path("temp")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


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
    except Exception as exc:
        logger.warning(f"Classifier failed to load: {exc}")
        app.state.classifier = None

    cls_status  = "loaded" if app.state.classifier else "not found"
    yolo_status = "loaded" if app.state.detector.is_available else "not found"
    logger.info(f"API ready | classifier={cls_status} | yolo={yolo_status}")

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
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", summary="Analyze a document image for forgery")
async def analyze_document(file: UploadFile = File(...)):
    # ---- validation ----
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG or PNG.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"

    try:
        temp_path.write_bytes(content)
        t0 = time.perf_counter()

        # ---- ELA ----
        ela_score = app.state.ela.get_forgery_score(str(temp_path))
        suspicious_regions = app.state.ela.get_suspicious_regions(str(temp_path))

        # ---- YOLO detection ----
        detections = app.state.detector.detect(str(temp_path))

        # ---- CNN classifier ----
        cls_score = None
        if app.state.classifier:
            cls_score = app.state.classifier.predict(str(temp_path))["forgery_probability"]

        # ---- combine scores ----
        inf_cfg = app.state.config.get("inference", {})
        ela_w   = inf_cfg.get("ela_weight", 0.35)
        cls_w   = inf_cfg.get("classifier_weight", 0.65)
        threshold = inf_cfg.get("forgery_threshold", 0.5)

        combined = (ela_w * ela_score + cls_w * cls_score) if cls_score is not None else ela_score
        combined = round(combined, 4)

        gap = abs(combined - 0.5)
        confidence = "high" if gap > 0.3 else "medium" if gap > 0.15 else "low"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return JSONResponse({
            "status": "success",
            "is_forged": combined > threshold,
            "forgery_probability": combined,
            "confidence": confidence,
            "scores": {
                "ela":        ela_score,
                "classifier": cls_score,
                "combined":   combined,
            },
            "detections": detections,
            "suspicious_regions": suspicious_regions[:5],
            "processing_time_ms": elapsed_ms,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/health", summary="Service health check")
def health():
    return {
        "status": "running",
        "models": {
            "classifier": app.state.classifier is not None,
            "yolo":       app.state.detector.is_available,
        },
    }
