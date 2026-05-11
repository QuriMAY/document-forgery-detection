import numpy as np

from src.ela import ELAAnalyzer


def test_analyze_returns_uint8_rgb(jpeg_path):
    ela = ELAAnalyzer({"quality": 90, "amplification": 15})
    out = ela.analyze(str(jpeg_path))
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3


def test_forgery_score_in_unit_interval(jpeg_path):
    ela = ELAAnalyzer()
    score = ela.get_forgery_score(str(jpeg_path))
    assert 0.0 <= score <= 1.0


def test_suspicious_regions_shape(jpeg_path):
    ela = ELAAnalyzer()
    regions = ela.get_suspicious_regions(str(jpeg_path))
    assert isinstance(regions, list)
    for r in regions:
        assert set(r.keys()) >= {"bbox", "area"}
        assert len(r["bbox"]) == 4


def test_visualize_closes_figure_on_save(jpeg_path, tmp_path):
    """Regression test: previous code leaked matplotlib figures on save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    ela = ELAAnalyzer()
    save_path = tmp_path / "viz.png"
    ela.visualize(str(jpeg_path), save_path=str(save_path))
    assert save_path.exists()
    # Figure should have been closed by visualize() after saving.
    assert plt.get_fignums() == []
