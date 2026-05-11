# Configuration Reference

All settings are in `config.yaml` at the project root.

---

## `ela`

Controls the Error Level Analysis module.

```yaml
ela:
  quality: 90
  amplification: 15
  threshold: 0.1
```

| Key | Type | Default | Description |
|---|---|---|---|
| `quality` | int | `90` | JPEG re-compression quality (1–95). Lower → more sensitive. |
| `amplification` | int | `15` | Multiplier applied to the difference map for visibility. |
| `threshold` | float | `0.1` | ELA score above this is considered suspicious. |

---

## `classifier`

Controls the ResNet CNN training.

```yaml
classifier:
  backbone: resnet18
  num_classes: 2
  image_size: 224
  batch_size: 16
  epochs: 15
  lr: 0.001
  weight_decay: 0.0001
  patience: 10
```

| Key | Type | Default | Description |
|---|---|---|---|
| `backbone` | str | `resnet18` | `resnet18` · `resnet50` · `resnet101` |
| `num_classes` | int | `2` | Always 2 (real / forged) |
| `image_size` | int | `224` | Input resolution |
| `batch_size` | int | `16` | Images per training step |
| `epochs` | int | `15` | Max training epochs |
| `lr` | float | `0.001` | Peak learning rate |
| `weight_decay` | float | `0.0001` | L2 regularisation |
| `patience` | int | `10` | Early stopping patience |

---

## `yolo`

Controls YOLOv8 detector training.

```yaml
yolo:
  base_model: yolov8s.pt
  epochs: 100
  imgsz: 640
  batch: 16
  patience: 20
```

| Key | Type | Default | Description |
|---|---|---|---|
| `base_model` | str | `yolov8s.pt` | Pretrained weights: `n` · `s` · `m` · `l` · `x` |
| `epochs` | int | `100` | Max training epochs |
| `imgsz` | int | `640` | Input image size |
| `batch` | int | `16` | Batch size |
| `patience` | int | `20` | Early stopping patience |

---

## `data`

Paths to dataset splits.

```yaml
data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
```

---

## `inference`

Controls how scores are combined at inference time.

```yaml
inference:
  ela_weight: 0.35
  classifier_weight: 0.65
  forgery_threshold: 0.5
```

| Key | Type | Default | Description |
|---|---|---|---|
| `ela_weight` | float | `0.35` | Weight of ELA score in final score |
| `classifier_weight` | float | `0.65` | Weight of CNN score in final score |
| `forgery_threshold` | float | `0.5` | Score above this → FORGED verdict |

!!! note
    `ela_weight + classifier_weight` must equal `1.0`.
