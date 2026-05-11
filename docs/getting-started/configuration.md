# Configuration

All settings live in a single `config.yaml` at the project root.

---

## Full reference

```yaml
ela:
  quality: 90         # Re-compression quality (70–95). Lower = more sensitive.
  amplification: 15   # Brightness multiplier for the ELA map.
  threshold: 0.1      # ELA score above this is flagged as suspicious.

classifier:
  backbone: resnet18  # Model architecture (see table below).
  num_classes: 2      # Always 2: real / forged.
  image_size: 224     # Input resolution in pixels.
  batch_size: 16      # Images per training step.
  epochs: 15          # Total training epochs.
  lr: 0.001           # Peak learning rate (OneCycleLR).
  weight_decay: 0.0001
  patience: 10        # Early stopping patience (epochs without improvement).

yolo:
  base_model: yolov8s.pt   # Pretrained weights to fine-tune from.
  epochs: 100
  imgsz: 640
  batch: 16
  patience: 20

data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test

inference:
  ela_weight: 0.35          # Weight of ELA in the combined score.
  classifier_weight: 0.65   # Weight of CNN in the combined score.
  forgery_threshold: 0.5    # Score above this → FORGED verdict.
```

---

## CPU vs GPU settings

| Setting | CPU (default) | GPU |
|---|---|---|
| `backbone` | `resnet18` | `resnet50` |
| `batch_size` | `16` | `32` |
| `epochs` | `15` | `30` |
| Mixed precision | Off (auto) | On (auto) |
| Est. training time | 30–60 min | 5–10 min |

---

## Backbone options

| Backbone | Parameters | Speed | Accuracy |
|---|---|---|---|
| `resnet18` | 11M | ⚡⚡⚡ Fast | Good |
| `resnet50` | 25M | ⚡⚡ Medium | Better |
| `resnet101` | 44M | ⚡ Slow | Best |

!!! tip "Start with resnet18"
    Always start with `resnet18` to verify your pipeline works. Switch to `resnet50` once you have a GPU.

---

## Adjusting the forgery threshold

The `forgery_threshold` controls the sensitivity of the system.

| Value | Effect |
|---|---|
| `0.3` | Very sensitive — catches more forgeries, more false alarms |
| `0.5` | Balanced (default) |
| `0.7` | Conservative — only flags high-confidence forgeries |

!!! warning
    Lower thresholds increase recall (catch more forgeries) but also increase false positives (flagging real docs as forged).
