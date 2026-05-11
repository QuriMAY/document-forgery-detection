"""Single forgery-analysis pipeline used by the FastAPI service, the Gradio
demo, and the inference CLI. Returns a canonical result dict so callers
don't drift apart."""

from __future__ import annotations

from src.constants import (
    CONFIDENCE_HIGH_GAP,
    CONFIDENCE_MEDIUM_GAP,
    DEFAULT_CLASSIFIER_WEIGHT,
    DEFAULT_ELA_WEIGHT,
    DEFAULT_FORGERY_THRESHOLD,
)


def _confidence_label(combined: float, threshold: float) -> str:
    gap = abs(combined - threshold)
    if gap > CONFIDENCE_HIGH_GAP:
        return "high"
    if gap > CONFIDENCE_MEDIUM_GAP:
        return "medium"
    return "low"


def analyze(
    image_path: str,
    ela,
    detector,
    classifier,
    cfg: dict | None = None,
) -> dict:
    """Run the full forgery pipeline on a single image.

    `cfg` is the loaded config.yaml dict; pass None to use defaults.
    Classifier may be None — pipeline degrades to ELA-only scoring.
    """
    inf = (cfg or {}).get("inference", {})
    ela_w = inf.get("ela_weight", DEFAULT_ELA_WEIGHT)
    cls_w = inf.get("classifier_weight", DEFAULT_CLASSIFIER_WEIGHT)
    threshold = inf.get("forgery_threshold", DEFAULT_FORGERY_THRESHOLD)

    ela_score = ela.get_forgery_score(image_path)
    regions = ela.get_suspicious_regions(image_path)
    detections = detector.detect(image_path)

    cls_score: float | None = None
    if classifier is not None:
        cls_score = classifier.predict(image_path)["forgery_probability"]

    if cls_score is not None:
        combined = ela_w * ela_score + cls_w * cls_score
    else:
        combined = ela_score
    combined = round(combined, 4)

    return {
        "path": image_path,
        "is_forged": combined > threshold,
        "forgery_probability": combined,
        "confidence": _confidence_label(combined, threshold),
        "scores": {
            "ela": ela_score,
            "classifier": cls_score,
            "combined": combined,
        },
        "suspicious_regions": regions[:5],
        "detections": detections,
    }
