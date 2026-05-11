import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SyntheticForgeryGenerator:
    """Generates forged documents via copy-move and splicing attacks."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feather_blend(
        self,
        base: np.ndarray,
        patch: np.ndarray,
        x: int,
        y: int,
        feather: bool = True,
    ) -> np.ndarray:
        h, w = patch.shape[:2]
        result = base.copy()

        if not feather:
            result[y : y + h, x : x + w] = patch
            return result

        # Build a per-pixel alpha mask with feathered border
        border = min(8, h // 5, w // 5)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(border):
            alpha = i / border
            mask[i, :] *= alpha
            mask[h - 1 - i, :] *= alpha
            mask[:, i] *= alpha
            mask[:, w - 1 - i] *= alpha

        mask = mask[:, :, np.newaxis]
        region = base[y : y + h, x : x + w].astype(np.float32)
        blended = patch.astype(np.float32) * mask + region * (1.0 - mask)
        result[y : y + h, x : x + w] = blended.clip(0, 255).astype(np.uint8)
        return result

    def _safe_crop(self, image: np.ndarray, pw: int, ph: int) -> tuple:
        """Return (patch, sx, sy) ensuring patch fits inside image."""
        h, w = image.shape[:2]
        pw = min(pw, w)
        ph = min(ph, h)
        sx = random.randint(0, w - pw)
        sy = random.randint(0, h - ph)
        return image[sy : sy + ph, sx : sx + pw].copy(), sx, sy

    # ------------------------------------------------------------------
    # Forgery primitives
    # ------------------------------------------------------------------

    def copy_move(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Copy a region from the image and paste it at a different location."""
        h, w = image.shape[:2]
        pw = random.randint(w // 8, w // 4)
        ph = random.randint(h // 8, h // 4)

        patch, sx, sy = self._safe_crop(image, pw, ph)

        # Optional: rotate or scale the patch slightly
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (pw, ph))

        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_pw = max(10, int(pw * scale))
            new_ph = max(10, int(ph * scale))
            patch = cv2.resize(patch, (new_pw, new_ph))
            ph, pw = patch.shape[:2]
            # Clamp so it still fits
            pw, ph = min(pw, w), min(ph, h)
            patch = patch[:ph, :pw]

        dx = random.randint(0, w - pw)
        dy = random.randint(0, h - ph)
        feather = random.random() > 0.4

        forged = self._feather_blend(image, patch, dx, dy, feather)
        meta = {
            "type": "copy_move",
            "source_bbox": [sx, sy, sx + pw, sy + ph],
            "dest_bbox": [dx, dy, dx + pw, dy + ph],
        }
        return forged, meta

    def splice(
        self, base: np.ndarray, donor: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Paste a region from a donor image onto the base document."""
        bh, bw = base.shape[:2]
        pw = random.randint(bw // 8, bw // 3)
        ph = random.randint(bh // 8, bh // 3)

        patch, _, _ = self._safe_crop(donor, pw, ph)
        patch = cv2.resize(patch, (pw, ph))

        # Subtle color jitter to simulate different scanner/camera conditions
        if random.random() > 0.4:
            factor = random.uniform(0.85, 1.15)
            patch = np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        dx = random.randint(0, bw - pw)
        dy = random.randint(0, bh - ph)
        feather = random.random() > 0.3

        forged = self._feather_blend(base, patch, dx, dy, feather)
        meta = {
            "type": "splice",
            "dest_bbox": [dx, dy, dx + pw, dy + ph],
        }
        return forged, meta

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        real_dir: str,
        output_dir: str,
        num_forged: int | None = None,
        donor_dir: str | None = None,
    ) -> dict:
        real_dir = Path(real_dir)
        output_dir = Path(output_dir)

        real_out = output_dir / "real"
        forged_out = output_dir / "forged"
        real_out.mkdir(parents=True, exist_ok=True)
        forged_out.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png"}
        real_images = [p for p in real_dir.iterdir() if p.suffix.lower() in exts]
        if not real_images:
            raise ValueError(f"No images found in {real_dir}")

        for img_path in real_images:
            shutil.copy(img_path, real_out / img_path.name)

        donor_images = None
        if donor_dir:
            donor_images = [p for p in Path(donor_dir).iterdir() if p.suffix.lower() in exts]

        num_forged = num_forged or len(real_images)
        generated, errors = 0, 0

        for i in range(num_forged):
            try:
                src = random.choice(real_images)
                raw = cv2.imread(str(src))
                if raw is None:
                    raise OSError(f"cv2 could not read {src}")
                image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

                if donor_images and random.random() > 0.5:
                    donor_path = random.choice(donor_images)
                    donor_raw = cv2.imread(str(donor_path))
                    if donor_raw is None:
                        raise OSError(f"cv2 could not read {donor_path}")
                    donor = cv2.cvtColor(donor_raw, cv2.COLOR_BGR2RGB)
                    forged, _ = self.splice(image, donor)
                else:
                    forged, _ = self.copy_move(image)

                out = forged_out / f"forged_{i:05d}.jpg"
                Image.fromarray(forged).save(str(out), quality=85)
                generated += 1

            except (OSError, ValueError, cv2.error) as exc:
                logger.warning("Sample %d failed: %s", i, exc)
                errors += 1

        logger.info(f"Generated {generated} forged images ({errors} errors)")
        return {"real_count": len(real_images), "forged_count": generated, "errors": errors}
