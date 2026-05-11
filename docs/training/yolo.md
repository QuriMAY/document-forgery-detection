# YOLO Detector

Fine-tunes YOLOv8 to draw bounding boxes around signatures and stamps in documents.

!!! note "Optional component"
    YOLO is optional. The API and demo app work without it — they just won't include bounding box detections.

---

## Labelling data with Roboflow

YOLO requires labelled bounding boxes. Use [Roboflow](https://roboflow.com) (free tier):

1. **Create account** at roboflow.com
2. **New project** → Object Detection
3. **Upload** your document images
4. **Label** each image:
   - Draw boxes around signatures → class `signature`
   - Draw boxes around stamps → class `stamp`
5. **Export** → YOLOv8 format → Download ZIP
6. Extract the ZIP — note the path to `data.yaml`

!!! tip "Minimum images"
    Label at least 200–300 images for usable results. 500+ gives noticeably better accuracy.

---

## Training command

```bash
python train_yolo.py --data path/to/data.yaml
python train_yolo.py --data path/to/data.yaml --device cpu
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Path to Roboflow `data.yaml` |
| `--device` | `cpu` | `cpu`, `0` (GPU), `0,1` (multi-GPU) |
| `--config` | `config.yaml` | Project config file |

---

## After training

Best weights are automatically copied to `models/best.pt`.

The API and demo app detect `models/best.pt` on startup and load it — no extra configuration needed.

Check that it loaded:

```bash
curl http://localhost:8000/health
# "models": {"classifier": true, "yolo": true}
```

---

## YOLO config reference

In `config.yaml`:

```yaml
yolo:
  base_model: yolov8s.pt   # yolov8n (tiny) | yolov8s (small) | yolov8m (medium)
  epochs: 100
  imgsz: 640
  batch: 16
  patience: 20
```

| Base model | Speed | Accuracy | Best for |
|---|---|---|---|
| `yolov8n.pt` | Fastest | Lower | CPU / quick test |
| `yolov8s.pt` | Fast | Good | Default |
| `yolov8m.pt` | Slower | Better | GPU recommended |
