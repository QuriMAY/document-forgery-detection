"""
Gradio demo — Document Forgery Detection
Run locally:  python app.py
In Docker:    started automatically via docker-compose (port 7860)
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")

from src.analysis import analyze
from src.detector import DocumentDetector
from src.ela import ELAAnalyzer
from src.utils import load_config, setup_logging

setup_logging("demo_app")
logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

_ela        = None
_detector   = None
_classifier = None
_cfg        = None


def get_models():
    """Lazy-load models. Re-checks for classifier on every call so a model
    trained after the demo started is picked up automatically."""
    global _ela, _detector, _classifier, _cfg

    if _ela is None:
        _cfg      = load_config()
        _ela      = ELAAnalyzer(_cfg.get("ela", {}))
        _detector = DocumentDetector()

    if _classifier is None and Path("models/classifier.pth").exists():
        from src.classifier import ForgeryClassifier
        try:
            _classifier = ForgeryClassifier()
            logger.info("CNN classifier loaded")
        except (RuntimeError, OSError, KeyError, ValueError) as e:
            logger.warning("Classifier not loaded: %s", e)

    return _ela, _detector, _classifier, _cfg


# ── Image helpers ─────────────────────────────────────────────────────────────

def draw_regions(image: np.ndarray, regions: list) -> np.ndarray:
    out = image.copy()
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 30, 30), 2)
        cv2.putText(out, f"#{i+1}", (x1 + 3, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 30, 30), 1)
    return out


# ── Plotly gauge ──────────────────────────────────────────────────────────────

def make_gauge(value: float) -> go.Figure:
    if value < 0.4:
        color, label = "#27ae60", "AUTHENTIC"
    elif value < 0.6:
        color, label = "#f39c12", "UNCERTAIN"
    else:
        color, label = "#e74c3c", "FORGED"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value * 100, 1),
        delta={"reference": 50, "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#27ae60"}},
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": f"<b>{label}</b>", "font": {"size": 20, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.2},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#d5f5e3"},
                {"range": [40, 60], "color": "#fef9e7"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 4},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="white",
    )
    return fig


# ── Main analysis pipeline called by Gradio ───────────────────────────────────

def run_analysis(image):
    if image is None:
        return [None, None, None], make_gauge(0), "", ""

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        Image.fromarray(image).save(f.name, quality=92)
        tmp = f.name

    try:
        ela, detector, classifier, cfg = get_models()
        result = analyze(tmp, ela, detector, classifier, cfg)

        original = image
        ela_map  = ela.analyze(tmp)
        overlay  = draw_regions(image, result["suspicious_regions"])

        combined = result["forgery_probability"]
        confidence = result["confidence"].upper()
        ela_score = result["scores"]["ela"]
        cls_score = result["scores"]["classifier"]

        # Verdict HTML
        if result["is_forged"]:
            verdict_html = f"""
            <div style="background:#fdecea;border-left:6px solid #e74c3c;
                        padding:18px 22px;border-radius:8px">
              <h2 style="color:#c0392b;margin:0 0 6px 0">⚠ FORGED</h2>
              <p style="color:#c0392b;margin:0;font-size:15px">
                Probability: <b>{combined:.1%}</b> &nbsp;|&nbsp;
                Confidence: <b>{confidence}</b> &nbsp;|&nbsp;
                Suspicious regions: <b>{len(result['suspicious_regions'])}</b>
              </p>
            </div>"""
        else:
            verdict_html = f"""
            <div style="background:#eafaf1;border-left:6px solid #27ae60;
                        padding:18px 22px;border-radius:8px">
              <h2 style="color:#1e8449;margin:0 0 6px 0">✓ AUTHENTIC</h2>
              <p style="color:#1e8449;margin:0;font-size:15px">
                Probability: <b>{combined:.1%}</b> &nbsp;|&nbsp;
                Confidence: <b>{confidence}</b> &nbsp;|&nbsp;
                Suspicious regions: <b>{len(result['suspicious_regions'])}</b>
              </p>
            </div>"""

        def bar(label, val, color):
            if val is None:
                return f'<div style="margin:8px 0;color:#95a5a6">{label}: <i>model not trained</i></div>'
            pct = val * 100
            return f"""
            <div style="margin:8px 0">
              <div style="display:flex;justify-content:space-between;
                          font-size:13px;color:#2c3e50;margin-bottom:4px">
                <span>{label}</span><span><b>{val:.4f}</b></span>
              </div>
              <div style="background:#ecf0f1;border-radius:4px;height:12px">
                <div style="width:{pct:.1f}%;background:{color};
                            border-radius:4px;height:12px"></div>
              </div>
            </div>"""

        def score_color(v):
            return "#27ae60" if v < 0.4 else "#f39c12" if v < 0.6 else "#e74c3c"

        sigs   = len(result["detections"].get("signatures", []))
        stamps = len(result["detections"].get("stamps", []))

        scores_html = f"""
        <div style="padding:4px 0">
          {bar("ELA Score",      ela_score, score_color(ela_score))}
          {bar("CNN Score",      cls_score, score_color(cls_score) if cls_score else '#95a5a6')}
          {bar("Combined Score", combined,  score_color(combined))}
          <div style="margin-top:16px;display:flex;gap:16px;font-size:13px;color:#2c3e50">
            <span>🔏 Signatures: <b>{sigs}</b></span>
            <span>🔵 Stamps: <b>{stamps}</b></span>
          </div>
        </div>"""

        return (
            [original, ela_map, overlay],
            make_gauge(combined),
            verdict_html,
            scores_html,
        )

    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Model status for sidebar ──────────────────────────────────────────────────

def model_status_html() -> str:
    _, detector, classifier, _ = get_models()
    cnn  = "✅ Loaded" if classifier       else "⚠️ Run <code>make train</code>"
    yolo = "✅ Loaded" if detector.is_available else "⚠️ Run <code>python train_yolo.py</code>"
    return f"""
    <div style="font-size:13px;line-height:1.8">
      <b>CNN Classifier</b><br>{cnn}<br><br>
      <b>YOLO Detector</b><br>{yolo}
    </div>"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Document Forgery Detection",
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="gray",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        footer { display: none !important; }
        """,
    ) as app:

        gr.Markdown("""
        # 🔍 Document Forgery Detection
        Upload a document image to detect tampering — forged signatures, replaced stamps, or copy-move manipulations.
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                image_input = gr.Image(
                    label="Upload Document",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=300,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Document",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### Model Status")
                gr.HTML(value=model_status_html)

                gr.Markdown("""
                ### Links
                🔗 [API docs](http://localhost:8000/docs) &nbsp;
                📊 [MLflow](http://localhost:5000) &nbsp;
                📋 [Grafana](http://localhost:3000)
                """)

            with gr.Column(scale=2):
                verdict_box = gr.HTML(
                    value='<div style="color:#95a5a6;padding:18px;'
                          'border:1px dashed #ddd;border-radius:8px;text-align:center">'
                          'Upload an image and click Analyze</div>'
                )

                with gr.Row():
                    gallery = gr.Gallery(
                        label="Analysis: Original | ELA Map | Suspicious Regions",
                        columns=3,
                        height=280,
                        object_fit="contain",
                        show_label=True,
                    )

                with gr.Row():
                    with gr.Column():
                        gauge_plot = gr.Plot(label="Forgery Probability", show_label=False)
                    with gr.Column():
                        scores_box = gr.HTML(label="Score Breakdown")

        analyze_btn.click(
            fn=run_analysis,
            inputs=[image_input],
            outputs=[gallery, gauge_plot, verdict_box, scores_box],
        )

        image_input.upload(
            fn=run_analysis,
            inputs=[image_input],
            outputs=[gallery, gauge_plot, verdict_box, scores_box],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
