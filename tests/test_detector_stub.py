from src.detector import DocumentDetector


def test_detector_graceful_when_weights_missing(tmp_path):
    """Constructor should not raise when the YOLO weights file is absent."""
    det = DocumentDetector(model_path=str(tmp_path / "does_not_exist.pt"))
    assert det.is_available is False
    out = det.detect("anything.jpg")
    assert out == {
        "signatures": [],
        "stamps": [],
        "warning": "YOLO model not found — run train_yolo.py first.",
    }
