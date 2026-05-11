import pytest
import torch
import torch.nn as nn

from src.classifier import build_model


@pytest.mark.parametrize("backbone", ["resnet18", "resnet50", "resnet101"])
def test_build_model_returns_module(backbone):
    model = build_model(backbone=backbone, num_classes=2, pretrained=False)
    assert isinstance(model, nn.Module)


def test_build_model_output_shape():
    model = build_model(backbone="resnet18", num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        build_model(backbone="vgg99", num_classes=2, pretrained=False)
