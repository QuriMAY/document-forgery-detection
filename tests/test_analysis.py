import pytest

from src.analysis import _confidence_label, analyze
from src.constants import DEFAULT_FORGERY_THRESHOLD
from src.ela import ELAAnalyzer


def test_confidence_bands():
    t = DEFAULT_FORGERY_THRESHOLD
    assert _confidence_label(t, t) == "low"
    assert _confidence_label(t + 0.20, t) == "medium"
    assert _confidence_label(t + 0.40, t) == "high"


def test_analyze_ela_only(jpeg_path, stub_detector):
    """Without a classifier, combined == ela_score."""
    ela = ELAAnalyzer()
    r = analyze(str(jpeg_path), ela, stub_detector, classifier=None)
    assert r["scores"]["classifier"] is None
    assert r["scores"]["combined"] == pytest.approx(r["scores"]["ela"], abs=1e-4)
    assert isinstance(r["is_forged"], bool)
    assert r["confidence"] in {"low", "medium", "high"}


def test_analyze_with_classifier(jpeg_path, stub_detector, stub_classifier):
    """Classifier weight should dominate at 0.65."""
    ela = ELAAnalyzer()
    cfg = {"inference": {"ela_weight": 0.35, "classifier_weight": 0.65,
                         "forgery_threshold": 0.5}}
    r = analyze(str(jpeg_path), ela, stub_detector, stub_classifier, cfg)
    ela_score = r["scores"]["ela"]
    expected = round(0.35 * ela_score + 0.65 * stub_classifier.prob, 4)
    assert r["scores"]["combined"] == pytest.approx(expected, abs=1e-4)


def test_analyze_uses_defaults_when_cfg_missing(jpeg_path, stub_detector):
    ela = ELAAnalyzer()
    r1 = analyze(str(jpeg_path), ela, stub_detector, classifier=None, cfg=None)
    r2 = analyze(str(jpeg_path), ela, stub_detector, classifier=None, cfg={})
    assert r1["scores"]["combined"] == r2["scores"]["combined"]
