<div align="center">

# Document Forgery Detection

**AI-powered system for detecting tampered documents — forged signatures, replaced stamps, and copy-move manipulations.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Loki-F46800?logo=grafana&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

Takes a document image as input and determines whether it has been tampered with using three independent methods combined into one confidence score.

| Method | What it detects | Requires training |
|---|---|---|
| **ELA** (Error Level Analysis) | JPEG compression inconsistencies from editing | No |
| **CNN** (ResNet classifier) | Whole-document real vs. forged classification | Yes |
| **YOLO** (Object detector) | Bounding boxes around signatures and stamps | Yes + labelling |

`forgery_probability = 0.35 × ela_score + 0.65 × cnn_score`  
Verdict: probability > 0.5 → **FORGED**

---

## Features

- **Three-layer detection** — ELA + CNN + YOLO, independently scored and combined
- **Synthetic data generation** — copy-move and splice attack simulation
- **MLflow experiment tracking** — every run auto-logged; compare in a web UI
- **Grafana + Loki log stack** — search and filter all script logs in a browser
- **FastAPI backend** — REST endpoint with file validation, CORS, health check
- **One-command Docker stack** — `make up` starts API + MLflow + Grafana + Loki
- **Visual demo** — `make demo` generates analysis dashboards with gauge charts
- **CPU-friendly** — fully functional without a GPU; auto-detects CUDA
- **Structured logging** — timestamped per-script log files, shipped to Loki

---

## Architecture

```
Input image
     │
     ├──► ELA Analyzer ──────────────────► ela_score  (0–1)
     │        JPEG compression diff map
     │        suspicious region bboxes
     │
     ├──► ResNet Classifier ─────────────► cls_score  (0–1)
     │        pretrained backbone
     │        Dropout → Linear → ReLU → Linear head
     │        trained on real + synthetic forged docs
     │
     └──► YOLOv8 Detector ──────────────► detections (bbox + conf)
              trained on labelled signatures / stamps

Combined = 0.35 × ela + 0.65 × cnn  →  verdict
```

```
Training scripts (host)       Docker services
───────────────────────       ───────────────────────────────
train_classifier.py  ──writes──► mlflow.db  ◄── MLflow UI  :5000
scripts/evaluate.py  ──writes──► mlartifacts
                                              
All scripts          ──writes──► logs/  ◄────── Promtail → Loki → Grafana :3000
                                              
models/classifier.pth ◄─────────────────────── API  :8000
```

---

## Project Structure

```
document-forgery-detection/
├── Makefile                      ← make up / down / train / demo / ...
├── config.yaml                   ← all hyperparameters in one place
├── requirements.txt
├── Dockerfile
├── docker-compose.yml            ← all 5 services in one file
├── .env                          ← secrets (never committed)
├── STEPS.txt                     ← detailed step-by-step guide
│
├── src/
│   ├── ela.py                    ← ELA analysis + region localization
│   ├── classifier.py             ← ResNet builder + inference wrapper
│   ├── detector.py               ← YOLOv8 wrapper (graceful fallback)
│   ├── data_generator.py         ← copy-move + splice forgery generator
│   └── utils.py                  ← logging, config loader, .env loader
│
├── kaggle_download.py            ← downloads datasets from Kaggle
├── train_classifier.py           ← CNN training (MLflow + TensorBoard)
├── train_yolo.py                 ← YOLO detector training
├── inference.py                  ← CLI: analyze any image or folder
├── demo.py                       ← visual demo with gauge charts + dashboards
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── prepare_dataset.py        ← stratified train / val / test split
│   └── evaluate.py               ← full metrics + MLflow artifact logging
│
├── api/
│   └── main.py                   ← FastAPI server
│
├── monitoring/
│   ├── loki-config.yaml          ← Loki log storage config
│   ├── promtail-config.yaml      ← log shipper (tails logs/ → Loki)
│   └── grafana/provisioning/     ← auto-provisions Loki datasource
│
└── notebooks/
    ├── 01_ELA_analysis.ipynb
    ├── 02_dataset_preparation.ipynb
    └── 03_model_evaluation.ipynb
```

---

## Quick Start

### 1. Clone and create environment

```bash
git clone https://github.com/your-username/document-forgery-detection.git
cd document-forgery-detection

make venv
source forgery_env/bin/activate
make install
```

### 2. Configure credentials

Edit `.env`:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
PYTHONPATH=.
```

Get your key at: https://www.kaggle.com/settings → API → Create New Token

### 3. Download data, generate forgeries, split

```bash
make download        # fetches 4 Kaggle datasets into data/real/ and data/patches/
make generate        # creates 1000 synthetic forged documents
make split           # stratified 70/15/15 train/val/test split
```

### 4. Train

```bash
make train           # ResNet18 on CPU (~30–60 min for 1000 images × 15 epochs)
make evaluate        # full test-set metrics + MLflow artifact upload
```

### 5. Run the demo

```bash
make demo            # generates dashboards in demo_output/
```

### 6. Start all services

```bash
make up              # builds + starts all 5 Docker services
```

| Service | URL | Purpose |
|---|---|---|
| Forgery API | http://localhost:8000 | REST endpoint |
| API docs | http://localhost:8000/docs | Swagger UI |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| Grafana | http://localhost:3000 | Log search & visualization |

---

## Makefile Reference

```bash
# Docker
make up              # start all services (builds image first)
make down            # stop all services
make restart         # rebuild + restart
make status          # show container health
make logs            # tail all container stdout
make build           # force rebuild the API image

# Training (runs on host, not in Docker)
make download        # python kaggle_download.py --all
make generate        # generate 1000 synthetic forgeries
make split           # prepare train/val/test split
make train           # python train_classifier.py
make evaluate        # python scripts/evaluate.py

# Tools
make demo            # python demo.py  →  demo_output/
make venv            # create forgery_env/
make install         # pip install -r requirements.txt
```

---

## Configuration

All settings in `config.yaml`:

```yaml
classifier:
  backbone: resnet18      # resnet18 (CPU) | resnet50 (GPU)
  batch_size: 16          # lower for CPU, 32+ for GPU
  epochs: 15
  lr: 0.001
  patience: 10            # early stopping

inference:
  ela_weight: 0.35
  classifier_weight: 0.65
  forgery_threshold: 0.5  # above this → FORGED
```

| Setting | CPU | GPU |
|---|---|---|
| `backbone` | `resnet18` | `resnet50` |
| `batch_size` | `16` | `32` |
| `epochs` | `15` | `30` |
| Mixed precision | off (auto) | on (auto) |

---

## Experiment Tracking — MLflow

Every `make train` logs automatically. Start the UI:

```bash
# Via Docker (recommended — starts with make up)
open http://localhost:5000

# Or directly on host
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Logged per run:** backbone, lr, batch_size, device, per-epoch val_auc/val_loss/f1, best checkpoint, evaluation plots, `early_stopped` tag.

**Useful queries in the UI:**
```
metrics.val_auc > 0.85
params.backbone = "resnet50"
```

---

## Log Visualization — Grafana

All script logs are shipped to Loki automatically when the Docker stack is running. Open **http://localhost:3000** (no login required).

**LogQL queries in Explore:**
```logql
{script="train_classifier"}                    # training logs only
{app="forgery-detection", level="ERROR"}       # all errors
{script="train_classifier"} |= "VlAUC"        # lines containing AUC metric
{script="train_classifier"} | live            # live tail during training
```

Filter by `script` label: `train_classifier`, `evaluate`, `kaggle_download`, `demo`, `inference`  
Filter by `level` label: `INFO`, `WARNING`, `ERROR`

---

## API Reference

### `POST /analyze`

```bash
curl -X POST http://localhost:8000/analyze -F "file=@document.jpg"
```

```json
{
  "status": "success",
  "is_forged": true,
  "forgery_probability": 0.823,
  "confidence": "high",
  "scores": { "ela": 0.312, "classifier": 0.921, "combined": 0.823 },
  "detections": {
    "signatures": [{"bbox": [120, 450, 380, 510], "confidence": 0.91}],
    "stamps": []
  },
  "suspicious_regions": [{"bbox": [100, 200, 300, 350], "area": 30000}],
  "processing_time_ms": 312.4
}
```

| `confidence` | Meaning |
|---|---|
| `high` | gap from threshold > 0.30 |
| `medium` | gap 0.15–0.30 |
| `low` | gap < 0.15 — inspect manually |

### `GET /health`

```json
{"status": "running", "models": {"classifier": true, "yolo": false}}
```

---

## CLI Inference

```bash
python inference.py document.jpg                        # single image
python inference.py folder/                             # entire folder
python inference.py folder/ --output results.json       # save JSON
python inference.py document.jpg --visualize            # save ELA images
```

```
document.jpg    FORGED ⚠  prob=0.823  ela=0.312  sigs=1  stamps=0
contract.jpg    REAL   ✓  prob=0.134  ela=0.051  sigs=0  stamps=1
```

---

## Visual Demo

```bash
make demo
# or: python demo.py --image path/to/doc.jpg
```

Saves to `demo_output/`:

| File | Content |
|---|---|
| `dashboard_authentic.png` | Original / ELA map / regions / gauge / score bars / summary |
| `dashboard_forged.png` | Same layout for the forged version |
| `comparison.png` | Side-by-side 3×4 comparison grid |
| `sample_authentic.jpg` | Generated authentic document |
| `sample_forged.jpg` | Generated forged document |

---

## YOLO — Stamp & Signature Detection (Optional)

1. Label images at [roboflow.com](https://roboflow.com) — classes: `signature`, `stamp`
2. Export as **YOLOv8 format**
3. Train:

```bash
python train_yolo.py --data path/to/data.yaml
```

Best weights auto-copied to `models/best.pt` and loaded by the API on restart.

---

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| Real-World Documents | [Kaggle](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) | Authentic training images |
| Tobacco3482 | [Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482dataset) | Authentic scanned documents |
| Signature Verification | [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) | Splice attack donor images |
| Handwritten Signatures | [Kaggle](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signatures) | Splice attack donor images |

Forgeries are generated synthetically — real forgery datasets are rare and often restricted.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Docker (for `make up`)
- 8 GB RAM minimum for CPU training

---

## License

MIT License — see [LICENSE](LICENSE) for details.
