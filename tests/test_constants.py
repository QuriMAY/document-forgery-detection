import pytest

from src.constants import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_IMAGE_FORMATS,
    CONFIDENCE_HIGH_GAP,
    CONFIDENCE_MEDIUM_GAP,
    DEFAULT_CLASSIFIER_WEIGHT,
    DEFAULT_ELA_WEIGHT,
    DEFAULT_FORGERY_THRESHOLD,
    MAX_FILE_BYTES,
)


def test_weights_sum_to_one():
    assert DEFAULT_ELA_WEIGHT + DEFAULT_CLASSIFIER_WEIGHT == pytest.approx(1.0)


def test_threshold_in_range():
    assert 0.0 < DEFAULT_FORGERY_THRESHOLD < 1.0


def test_confidence_bands_ordered():
    assert CONFIDENCE_HIGH_GAP > CONFIDENCE_MEDIUM_GAP > 0


def test_allowed_formats_consistent():
    # Each allowed image format should map to at least one allowed content type.
    for fmt in ALLOWED_IMAGE_FORMATS:
        assert any(fmt in ct for ct in ALLOWED_CONTENT_TYPES), fmt


def test_max_file_bytes_positive():
    assert MAX_FILE_BYTES > 0
