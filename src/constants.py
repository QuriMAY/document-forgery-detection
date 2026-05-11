"""Shared defaults — single source of truth used when config.yaml is missing
or partial. Update here, not in callers."""

DEFAULT_ELA_WEIGHT = 0.35
DEFAULT_CLASSIFIER_WEIGHT = 0.65
DEFAULT_FORGERY_THRESHOLD = 0.5

CONFIDENCE_HIGH_GAP = 0.30
CONFIDENCE_MEDIUM_GAP = 0.15

ALLOWED_IMAGE_FORMATS = {"jpeg", "png", "webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
