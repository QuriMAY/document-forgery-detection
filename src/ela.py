import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class ELAAnalyzer:
    """Error Level Analysis — detects JPEG compression inconsistencies caused by editing."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.quality = cfg.get("quality", 90)
        self.amplification = cfg.get("amplification", 15)
        self.threshold = cfg.get("threshold", 0.1)

    def analyze(self, image_path: str) -> np.ndarray:
        """Return amplified ELA map as uint8 RGB array."""
        original = Image.open(image_path).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")

        orig = np.array(original, dtype=np.float32)
        comp = np.array(compressed, dtype=np.float32)

        ela = np.abs(orig - comp) * self.amplification
        ela = np.clip(ela, 0, 255).astype(np.uint8)
        return ela

    def get_forgery_score(self, image_path: str) -> float:
        """Score in [0, 1]. Higher = more likely tampered."""
        ela = self.analyze(image_path)
        flat = ela.flatten().astype(np.float32)
        flat.sort()
        # Mean of top-10% brightest values is a robust tamper indicator
        top10 = flat[int(len(flat) * 0.90):]
        return round(float(np.mean(top10)) / 255.0, 4)

    def get_suspicious_regions(
        self, image_path: str, min_area: int = 500
    ) -> list[dict]:
        """Return bounding boxes of high-ELA regions sorted by area (largest first)."""
        ela = self.analyze(image_path)
        gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({"bbox": [x, y, x + w, y + h], "area": int(area)})

        return sorted(regions, key=lambda r: r["area"], reverse=True)

    def visualize(
        self,
        image_path: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Side-by-side: original | ELA map | suspicious regions overlay."""
        original = np.array(Image.open(image_path).convert("RGB"))
        ela = self.analyze(image_path)
        regions = self.get_suspicious_regions(image_path)

        overlay = original.copy()
        for r in regions[:5]:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(ela);      axes[1].set_title(f"ELA Map (q={self.quality})"); axes[1].axis("off")
        axes[2].imshow(overlay);  axes[2].set_title("Suspicious Regions"); axes[2].axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
