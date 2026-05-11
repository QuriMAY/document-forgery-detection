import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


_BACKBONES: dict[str, tuple] = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1,  2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
}


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    if backbone not in _BACKBONES:
        raise ValueError(f"backbone must be one of {list(_BACKBONES)}")

    model_fn, weights_cls, in_features = _BACKBONES[backbone]
    model = model_fn(weights=weights_cls if pretrained else None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


class ForgeryClassifier:
    """Inference wrapper around trained ResNet forgery classifier."""

    def __init__(
        self,
        model_path: str = "models/classifier.pth",
        backbone: str = "resnet50",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_model(self, path: str) -> nn.Module:
        if not Path(path).exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            backbone = checkpoint.get("backbone", self.backbone)
            state = checkpoint["model_state_dict"]
        else:
            # Legacy: bare state-dict on disk. Accept but warn so the next
            # save reaches the canonical format.
            logger.warning(
                "Loading legacy checkpoint (no 'model_state_dict' key) from %s — "
                "re-train or re-save to upgrade.",
                path,
            )
            backbone = self.backbone
            state = checkpoint

        model = build_model(backbone=backbone, num_classes=2, pretrained=False)
        model.load_state_dict(state)
        model.eval()
        return model.to(self.device)

    def predict(self, image: str | np.ndarray) -> dict:
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("RGB")
        else:
            pil = Image.open(image).convert("RGB")

        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.softmax(logits, dim=1)
            # ImageFolder trains classes alphabetically: forged=0, real=1.
            forgery_prob = float(prob[0, 0])

        return {
            "is_forged": forgery_prob > 0.5,
            "forgery_probability": round(forgery_prob, 4),
            "real_probability": round(1.0 - forgery_prob, 4),
        }
