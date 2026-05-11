#!/usr/bin/env python3
"""
Demo — Document Forgery Detection System

Generates synthetic documents, runs the full analysis pipeline, and produces
publication-quality visual reports.

Usage:
    python demo.py                          # auto-generate + analyze both real & forged
    python demo.py --image doc.jpg          # analyze your own image
    python demo.py --compare                # side-by-side comparison figure
    python demo.py --save-dir results/      # custom output folder (default: demo_output/)
"""

import argparse
import logging
import tempfile
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Wedge
from PIL import Image

from src.ela import ELAAnalyzer
from src.detector import DocumentDetector
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
G = "#27ae60"   # green  — authentic / safe
Y = "#f39c12"   # yellow — uncertain
R = "#e74c3c"   # red    — forged / danger
B = "#2980b9"   # blue   — neutral accent
D = "#2c3e50"   # dark   — text / borders
LG = "#ecf0f1"  # light grey — backgrounds


# ─────────────────────────────────────────────────────────────────────────────
#  1. Sample document generation
# ─────────────────────────────────────────────────────────────────────────────

def _draw_signature(img: np.ndarray, ox: int, oy: int, color=(20, 20, 120)) -> None:
    """Draw a hand-written-style signature scribble."""
    pts = np.array([
        [ox,      oy + 20], [ox + 25, oy],      [ox + 55, oy + 25],
        [ox + 85, oy + 5],  [ox + 115, oy + 22], [ox + 140, oy + 8],
        [ox + 165, oy + 28],
    ], np.int32)
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    # under-line flourish
    cv2.line(img, (ox, oy + 35), (ox + 165, oy + 35), (*color, 255), 1)


def _draw_stamp(img: np.ndarray, cx: int, cy: int, color=(0, 80, 160)) -> None:
    """Draw a circular official stamp."""
    cv2.circle(img, (cx, cy), 52, color, 2)
    cv2.circle(img, (cx, cy), 44, color, 1)
    for text, dy in [("OFFICIAL", -12), ("APPROVED", 8), ("2026", 26)]:
        tw = len(text) * 9
        cv2.putText(img, text, (cx - tw // 2, cy + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def create_document(width: int = 620, height: int = 860) -> np.ndarray:
    """Create a realistic-looking synthetic contract document."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 250  # off-white paper

    # Outer border
    cv2.rectangle(img, (28, 28), (width - 28, height - 28), (180, 180, 180), 1)
    cv2.rectangle(img, (33, 33), (width - 33, height - 33), (210, 210, 210), 1)

    # Title
    title = "SERVICE AGREEMENT"
    tw = len(title) * 12
    cv2.putText(img, title, (width // 2 - tw // 2, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(img, (55, 108), (width - 55, 108), (160, 160, 160), 1)

    # Meta
    cv2.putText(img, "Date: 2026-05-06", (55, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.putText(img, "Ref No: SA-2026-0042", (width - 240, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)

    # Body paragraphs
    body = [
        "This Agreement is entered into as of the date written above between",
        "the undersigned parties, and shall be binding upon both parties.",
        "",
        "1. SCOPE OF WORK",
        "   The service provider shall deliver agreed services within the",
        "   timeframe and budget specified in Schedule A hereto.",
        "",
        "2. PAYMENT TERMS",
        "   Invoices are payable within 30 days of receipt. Overdue amounts",
        "   accrue interest at 1.5% per month.",
        "",
        "3. INTELLECTUAL PROPERTY",
        "   All work product created under this agreement is the exclusive",
        "   property of the client upon full payment.",
        "",
        "4. CONFIDENTIALITY",
        "   Both parties agree to protect all confidential information and",
        "   not to disclose it to third parties without prior written consent.",
        "",
        "5. TERMINATION",
        "   Either party may terminate with 30 days written notice.",
        "",
        "6. GOVERNING LAW",
        "   This agreement is governed by applicable jurisdiction laws.",
    ]
    y = 175
    for line in body:
        if line:
            cv2.putText(img, line, (55, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (55, 55, 55), 1, cv2.LINE_AA)
        y += 26

    # Signature section
    sy = height - 215
    cv2.line(img, (55, sy), (width - 55, sy), (180, 180, 180), 1)
    label = "AUTHORIZED SIGNATURES"
    cv2.putText(img, label, (width // 2 - len(label) * 6, sy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # Party A
    cv2.putText(img, "Party A:", (55, sy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    _draw_signature(img, 60, sy + 75)
    cv2.putText(img, "John A. Smith", (65, sy + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

    # Party B
    bx = width // 2 + 20
    cv2.putText(img, "Party B:", (bx, sy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    _draw_signature(img, bx + 5, sy + 75, color=(120, 20, 20))
    cv2.putText(img, "Jane B. Doe", (bx + 10, sy + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

    # Stamp
    _draw_stamp(img, width - 105, sy + 88)

    return img


def create_forged(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Forge a document by copy-moving a region over the signature area."""
    forged = image.copy()
    h, w = image.shape[:2]

    # Copy a patch from the body text area ...
    sy, sx = 250, 55
    ph, pw = 80, 180
    patch = image[sy : sy + ph, sx : sx + pw].copy()

    # ... and paste it over Party A's signature
    sig_y = h - 215 + 65
    sig_x = 55

    # Slight brightness shift to simulate different scanner conditions
    patch = np.clip(patch.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
    forged[sig_y : sig_y + ph, sig_x : sig_x + pw] = patch

    meta = {
        "type": "copy_move",
        "source": [sx, sy, sx + pw, sy + ph],
        "dest":   [sig_x, sig_y, sig_x + pw, sig_y + ph],
    }
    return forged, meta


# ─────────────────────────────────────────────────────────────────────────────
#  2. Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(image_path: str, ela: ELAAnalyzer, detector: DocumentDetector,
            classifier, cfg: dict) -> dict:
    # Delegate to the canonical pipeline; remap keys to the legacy demo schema
    # consumed by plot_analysis / plot_comparison below.
    from src.analysis import analyze as _analyze
    r = _analyze(image_path, ela, detector, classifier, cfg)
    return {
        "ela_score":   round(r["scores"]["ela"], 4),
        "cls_score":   round(r["scores"]["classifier"], 4) if r["scores"]["classifier"] is not None else None,
        "combined":    r["forgery_probability"],
        "is_forged":   r["is_forged"],
        "confidence":  r["confidence"],
        "regions":     r["suspicious_regions"],
        "detections":  r["detections"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  3. Visual components
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(v: float) -> str:
    if v < 0.4: return G
    if v < 0.6: return Y
    return R


def draw_gauge(ax, value: float, label: str = "Forgery Probability") -> None:
    """Draw a semicircular gauge needle chart."""
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colored arc zones
    for lo, hi, color in [(0.0, 0.4, G), (0.4, 0.6, Y), (0.6, 1.0, R)]:
        a1 = 180 - lo * 180
        a2 = 180 - hi * 180
        w = Wedge((0, 0), 1.0, a2, a1, width=0.28, color=color, alpha=0.85, zorder=2)
        ax.add_patch(w)

    # Grey track underneath
    bg = Wedge((0, 0), 1.0, 0, 180, width=0.28, color="#dfe6e9", alpha=0.4, zorder=1)
    ax.add_patch(bg)

    # Needle
    angle = np.pi * (1.0 - value)
    nx, ny = 0.72 * np.cos(angle), 0.72 * np.sin(angle)
    ax.annotate("", xy=(nx, ny), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=D,
                                lw=2.0, mutation_scale=14))

    # Hub circle
    hub = plt.Circle((0, 0), 0.07, color=D, zorder=5)
    ax.add_patch(hub)

    # Tick labels
    for v_tick, txt in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
        a = np.pi * (1.0 - v_tick)
        tx, ty = 1.16 * np.cos(a), 1.16 * np.sin(a)
        ax.text(tx, ty, txt, ha="center", va="center", fontsize=7.5, color=D)

    # Zone labels on arc
    for mid, txt in [(0.2, "REAL"), (0.5, "?"), (0.8, "FORGED")]:
        a = np.pi * (1.0 - mid)
        lx, ly = 0.86 * np.cos(a), 0.86 * np.sin(a)
        ax.text(lx, ly, txt, ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold")

    # Value display
    color = _score_color(value)
    ax.text(0, -0.18, f"{value:.1%}", ha="center", va="top",
            fontsize=17, fontweight="bold", color=color)
    ax.text(0, -0.32, label, ha="center", va="top",
            fontsize=8, color=D, style="italic")


def draw_score_bars(ax, ela: float, cls, combined: float) -> None:
    """Horizontal bar chart of all three scores."""
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.set_facecolor(LG)

    scores = [("ELA Score", ela), ("Combined", combined)]
    if cls is not None:
        scores.insert(1, ("CNN Score", cls))

    bar_h = 0.18
    total = len(scores)
    ys = np.linspace(0.75, 0.15, total)

    ax.text(0.5, 0.95, "Score Breakdown", ha="center", va="top",
            fontsize=10, fontweight="bold", color=D,
            transform=ax.transAxes)

    for (name, val), y in zip(scores, ys):
        color = _score_color(val)
        # Background track
        bg = FancyBboxPatch((0.02, y - bar_h / 2), 0.96, bar_h,
                            boxstyle="round,pad=0.01",
                            facecolor="#dfe6e9", edgecolor="none",
                            transform=ax.transAxes)
        ax.add_patch(bg)
        # Filled bar
        bar = FancyBboxPatch((0.02, y - bar_h / 2), max(0.015, 0.96 * val), bar_h,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor="none", alpha=0.88,
                             transform=ax.transAxes)
        ax.add_patch(bar)
        # Labels
        ax.text(0.04, y, name, va="center", fontsize=8.5,
                color="white" if val > 0.15 else D, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.97, y, f"{val:.3f}", va="center", ha="right",
                fontsize=8.5, color=D, fontweight="bold",
                transform=ax.transAxes)


def draw_summary(ax, result: dict) -> None:
    """Clean text summary panel."""
    ax.axis("off")
    ax.set_facecolor(LG)

    verdict = "⚠ FORGED" if result["is_forged"] else "✓ AUTHENTIC"
    vcolor  = R if result["is_forged"] else G

    ax.text(0.5, 0.93, "Analysis Summary", ha="center", va="top",
            fontsize=10, fontweight="bold", color=D,
            transform=ax.transAxes)

    # Verdict badge
    ax.text(0.5, 0.76, verdict, ha="center", va="center",
            fontsize=15, fontweight="bold", color=vcolor,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=vcolor,
                      alpha=0.12, edgecolor=vcolor, linewidth=1.5))

    dets = result["detections"]
    sigs   = len(dets.get("signatures", []))
    stamps = len(dets.get("stamps", []))
    lines = [
        ("Confidence",   result["confidence"].upper()),
        ("Probability",  f"{result['combined']:.1%}"),
        ("Regions flagged", str(len(result["regions"]))),
        ("Signatures found", str(sigs)),
        ("Stamps found",    str(stamps)),
    ]

    for i, (k, v) in enumerate(lines):
        y = 0.54 - i * 0.10
        ax.text(0.08, y, k, va="center", fontsize=8, color=D,
                transform=ax.transAxes)
        ax.text(0.92, y, v, va="center", ha="right", fontsize=8,
                fontweight="bold", color=D, transform=ax.transAxes)
        if i < len(lines) - 1:
            ax.plot([0.05, 0.95], [y - 0.035, y - 0.035],
                    color="#bdc3c7", linewidth=0.5,
                    transform=ax.transAxes)

    if not result["is_forged"]:
        note = "No tampering detected."
    elif result["confidence"] == "high":
        note = "High-confidence forgery detected."
    else:
        note = "Possible tampering — inspect manually."

    ax.text(0.5, 0.04, note, ha="center", va="bottom",
            fontsize=7.5, color=vcolor, style="italic",
            transform=ax.transAxes)


def overlay_regions(image: np.ndarray, regions: list) -> np.ndarray:
    """Draw red bounding boxes over suspicious regions."""
    out = image.copy()
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 30, 30), 2)
        cv2.putText(out, f"#{i+1}", (x1 + 3, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 30, 30), 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  4. Dashboard figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_analysis(image_path: str, result: dict,
                  title: str = "Analysis", save_path: str = None) -> plt.Figure:
    """
    Full analysis dashboard for a single image.
    Layout (2 rows × 3 cols):
      [Original] [ELA Heatmap] [Suspicious Regions]
      [Gauge   ] [Score Bars ] [Summary           ]
    """
    image    = np.array(Image.open(image_path).convert("RGB"))
    ela_map  = ELAAnalyzer().analyze(image_path)
    overlay  = overlay_regions(image, result["regions"])

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    fig.suptitle(title, fontsize=15, fontweight="bold", color=D, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25,
                          left=0.04, right=0.97, top=0.93, bottom=0.04)

    # ── Row 1: images ──
    for col, (img_data, subtitle) in enumerate([
        (image,   "Original Document"),
        (ela_map, "ELA Map  (bright = inconsistent compression)"),
        (overlay, f"Suspicious Regions  ({len(result['regions'])} found)"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_data)
        ax.set_title(subtitle, fontsize=9, color=D, pad=5)
        ax.axis("off")
        # Thin border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#bdc3c7")
            spine.set_linewidth(0.8)

    # ── Row 2: metrics ──
    ax_gauge = fig.add_subplot(gs[1, 0])
    draw_gauge(ax_gauge, result["combined"])

    ax_bars = fig.add_subplot(gs[1, 1])
    draw_score_bars(ax_bars, result["ela_score"], result["cls_score"], result["combined"])

    ax_summ = fig.add_subplot(gs[1, 2])
    draw_summary(ax_summ, result)

    # Footer
    fig.text(0.5, 0.005,
             "Document Forgery Detection  |  ELA + ResNet + YOLOv8  |  github.com/your-username/document-forgery-detection",
             ha="center", fontsize=7, color="#7f8c8d")

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        logger.info(f"Saved → {save_path}")

    return fig


def plot_comparison(real_path: str, forged_path: str,
                    real_res: dict, forged_res: dict,
                    save_path: str = None) -> plt.Figure:
    """
    Side-by-side comparison: authentic on left, forged on right.
    Layout (3 rows × 4 cols):
      [Real orig] [Real ELA] | [Forged orig] [Forged ELA]
      [Real regs] [Real gge] | [Forged regs] [Forged gge]
      [Real bars] [Real sum] | [Forged bars] [Forged sum]
    """
    ela_inst = ELAAnalyzer()

    real_img   = np.array(Image.open(real_path).convert("RGB"))
    forged_img = np.array(Image.open(forged_path).convert("RGB"))
    real_ela   = ela_inst.analyze(real_path)
    forged_ela = ela_inst.analyze(forged_path)
    real_ov    = overlay_regions(real_img,   real_res["regions"])
    forged_ov  = overlay_regions(forged_img, forged_res["regions"])

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle("Forgery Detection — Side-by-Side Comparison",
                 fontsize=16, fontweight="bold", color=D, y=0.99)

    # Dividing line between left and right halves
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.02, 0.97],
                              transform=fig.transFigure,
                              color="#bdc3c7", linewidth=1.2))

    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.28,
                          left=0.03, right=0.98, top=0.95, bottom=0.03)

    sides = [
        (0, real_img,   real_ela,   real_ov,   real_res,   "AUTHENTIC DOCUMENT", G),
        (2, forged_img, forged_ela, forged_ov, forged_res, "FORGED DOCUMENT",    R),
    ]

    for col_off, img, ela_map, ov, res, heading, hcolor in sides:
        # Column heading
        fig.text(0.255 + col_off * 0.245, 0.965, heading,
                 ha="center", fontsize=12, fontweight="bold", color=hcolor)

        # Row 0: original + ELA
        for c, (data, sub) in enumerate([(img, "Original"), (ela_map, "ELA Map")]):
            ax = fig.add_subplot(gs[0, col_off + c])
            ax.imshow(data)
            ax.set_title(sub, fontsize=8.5, color=D, pad=4)
            ax.axis("off")

        # Row 1: regions + gauge
        ax_ov = fig.add_subplot(gs[1, col_off])
        ax_ov.imshow(ov)
        ax_ov.set_title(f"Regions ({len(res['regions'])} flagged)", fontsize=8.5, color=D, pad=4)
        ax_ov.axis("off")

        ax_g = fig.add_subplot(gs[1, col_off + 1])
        draw_gauge(ax_g, res["combined"])

        # Row 2: bars + summary
        ax_b = fig.add_subplot(gs[2, col_off])
        draw_score_bars(ax_b, res["ela_score"], res["cls_score"], res["combined"])

        ax_s = fig.add_subplot(gs[2, col_off + 1])
        draw_summary(ax_s, res)

    fig.text(0.5, 0.005,
             "Document Forgery Detection  |  ELA + ResNet + YOLOv8",
             ha="center", fontsize=7, color="#7f8c8d")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _load_models(cfg: dict):
    ela = ELAAnalyzer(cfg.get("ela", {}))
    det = DocumentDetector()

    classifier = None
    if Path("models/classifier.pth").exists():
        from src.classifier import ForgeryClassifier
        try:
            classifier = ForgeryClassifier()
            logger.info("CNN classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")
    else:
        logger.warning("No classifier model found — running ELA-only mode")

    return ela, det, classifier


def main():
    p = argparse.ArgumentParser(description="Forgery detection visual demo")
    p.add_argument("--image",    default=None, help="Analyze your own image instead of synthetic")
    p.add_argument("--compare",  action="store_true", help="Show side-by-side real vs forged")
    p.add_argument("--save-dir", default="demo_output", help="Output folder for saved figures")
    p.add_argument("--no-show",  action="store_true", help="Do not display figures (just save)")
    args = p.parse_args()

    setup_logging("demo")
    cfg = load_config()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ela, det, classifier = _load_models(cfg)
    mode = "ELA-only" if classifier is None else "ELA + CNN"
    logger.info(f"Running in {mode} mode")

    # ── Case 1: user provided their own image ──────────────────────────────
    if args.image:
        img_path = args.image
        logger.info(f"Analyzing: {img_path}")
        result = analyze(img_path, ela, det, classifier, cfg)

        verdict = "FORGED" if result["is_forged"] else "AUTHENTIC"
        print(f"\n{'─'*50}")
        print(f"  File       : {img_path}")
        print(f"  Verdict    : {verdict}  (confidence: {result['confidence']})")
        print(f"  Probability: {result['combined']:.1%}")
        print(f"  ELA score  : {result['ela_score']:.4f}")
        if result["cls_score"] is not None:
            print(f"  CNN score  : {result['cls_score']:.4f}")
        print(f"  Regions    : {len(result['regions'])} suspicious areas found")
        print(f"{'─'*50}\n")

        fig = plot_analysis(img_path, result,
                            title=f"Forgery Analysis — {Path(img_path).name}",
                            save_path=str(save_dir / "analysis.png"))
        if not args.no_show:
            plt.show()
        return

    # ── Case 2: generate synthetic documents and demo ─────────────────────
    logger.info("Generating synthetic document images...")
    real_doc   = create_document()
    forged_doc, forge_meta = create_forged(real_doc)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        real_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        forged_path = f.name

    Image.fromarray(real_doc).save(real_path,   quality=92)
    Image.fromarray(forged_doc).save(forged_path, quality=88)

    # Also save them for user inspection
    Image.fromarray(real_doc).save(str(save_dir / "sample_authentic.jpg"), quality=92)
    Image.fromarray(forged_doc).save(str(save_dir / "sample_forged.jpg"),  quality=88)
    logger.info(f"Sample images saved to {save_dir}/")

    logger.info("Running analysis on both documents...")
    real_result   = analyze(real_path,   ela, det, classifier, cfg)
    forged_result = analyze(forged_path, ela, det, classifier, cfg)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  DOCUMENT FORGERY DETECTION — DEMO RESULTS")
    print(f"{'═'*56}")
    for label, res in [("AUTHENTIC", real_result), ("FORGED", forged_result)]:
        verdict = "FORGED ⚠" if res["is_forged"] else "REAL   ✓"
        print(f"\n  [{label}]")
        print(f"    Verdict    : {verdict}  ({res['confidence']} confidence)")
        print(f"    Probability: {res['combined']:.1%}")
        print(f"    ELA score  : {res['ela_score']:.4f}")
        if res["cls_score"] is not None:
            print(f"    CNN score  : {res['cls_score']:.4f}")
        print(f"    Regions    : {len(res['regions'])} flagged")
    print(f"\n{'═'*56}\n")

    # ── Individual dashboards ─────────────────────────────────────────────
    plot_analysis(real_path, real_result,
                  title="Forgery Analysis — Authentic Document",
                  save_path=str(save_dir / "dashboard_authentic.png"))

    plot_analysis(forged_path, forged_result,
                  title="Forgery Analysis — Forged Document",
                  save_path=str(save_dir / "dashboard_forged.png"))

    # ── Comparison figure ─────────────────────────────────────────────────
    if args.compare:
        plot_comparison(real_path, forged_path, real_result, forged_result,
                        save_path=str(save_dir / "comparison.png"))

    logger.info(f"\nAll figures saved to  {save_dir}/")
    logger.info("  dashboard_authentic.png")
    logger.info("  dashboard_forged.png")
    if args.compare:
        logger.info("  comparison.png")
    logger.info("  sample_authentic.jpg")
    logger.info("  sample_forged.jpg")

    if not args.no_show:
        plt.show()

    # Clean up temp files
    Path(real_path).unlink(missing_ok=True)
    Path(forged_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
