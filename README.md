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
- **One-command Docker stack** — `make up` starts all 7 services
- **Gradio demo app** — upload any document and see live analysis with gauge charts
- **Mintlify documentation** — modern docs site at localhost:3001
- **CPU-friendly** — fully functional without a GPU; auto-detects CUDA
- **Hot-reload model** — train while the demo is running; it picks up the new model automatically

---

## Architecture

```
Training scripts (host)       Docker services
───────────────────────       ─────────────────────────────────────
train_classifier.py  ──saves──► models/classifier.pth ◄── API  :8000
                                                        ◄── Demo :7860 (hot-reload)

train_classifier.py  ──writes──► mlflow.db  ◄── MLflow UI  :5000
scripts/evaluate.py  ──writes──► mlartifacts

All scripts          ──writes──► logs/  ◄──── Promtail → Loki → Grafana :3000

mintlify-docs/       ──────────────────────────────────── Docs  :3001
```

---

## Project Structure

```
document-forgery-detection/
├── Makefile                      ← make up / down / train / demo / ...
├── config.yaml                   ← all hyperparameters in one place
├── requirements.txt
├── Dockerfile                    ← CPU-only PyTorch (faster builds)
├── docker-compose.yml            ← all 7 services in one file
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
├── train_classifier.py           ← CNN training (MLflow + TensorBoard + tqdm)
├── train_yolo.py                 ← YOLO detector training
├── inference.py                  ← CLI: analyze any image or folder
├── demo.py                       ← static visual demo (saves PNG dashboards)
├── app.py                        ← Gradio web app (hot-reloads trained model)
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
│   ├── loki-config.yaml
│   ├── promtail-config.yaml      ← tails logs/ → Loki
│   └── grafana/provisioning/     ← auto-provisions Loki datasource
│
├── mintlify-docs/                ← Mintlify documentation source
│   ├── mint.json
│   └── **/*.mdx
│
└── notebooks/
    ├── 01_ELA_analysis.ipynb
    ├── 02_dataset_preparation.ipynb
    └── 03_model_evaluation.ipynb
```

---

## Quick Start

### What ships in this repo

The git repo contains **only source code**. Trained weights (`models/classifier.pth`, `models/best.pt`) and the training datasets (`data/real/`, `data/forged/`, …) are gitignored because they're too large. There are two ways to bring the model online:

| Path | Setup time | Needs |
|---|---|---|
| **A — Use pre-trained weights** | ~1 minute | A published `classifier.pth` URL |
| **B — Train from scratch** | hours (CPU) / ~20 min (GPU) | Kaggle API key + ~5 GB disk |

### Path A — Pre-trained weights (fastest)

```bash
git clone https://github.com/<you>/document-forgery-detection.git
cd document-forgery-detection
make venv && source forgery_env/bin/activate && make install
make fetch-weights MODEL_URL=https://github.com/<you>/<repo>/releases/download/v1.0/classifier.pth
make up
```

> Maintainer note: upload `models/classifier.pth` once as a GitHub Release asset; everyone else can then `make fetch-weights` without Kaggle credentials. YOLO weights (`best.pt`) can be distributed the same way if you've trained the detector.

### Path B — Train from scratch

#### 1. Clone and create environment

```bash
git clone https://github.com/<you>/document-forgery-detection.git
cd document-forgery-detection

make venv
source forgery_env/bin/activate
make install
```

#### 2. Configure credentials

Create `.env` (see [.env.example] template below — never commit this file):

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
PYTHONPATH=.
```

Get your key at: https://www.kaggle.com/settings → API → Create New Token.

#### 3. Data → Train → Launch

```bash
make download        # fetch 4 Kaggle datasets (~3 GB)
make generate        # create 1000 synthetic forgeries
make split           # 70/15/15 train/val/test split
make train           # ResNet18 — live tqdm progress bars (~20 min GPU, hours CPU)
make evaluate        # test metrics + MLflow upload
make up              # start all 7 Docker services
```

Optional — train the signature/stamp detector after labelling at [roboflow.com](https://roboflow.com):

```bash
make train-yolo      # writes models/best.pt
```

#### 4. Try the included sample images

Three deterministic synthetic samples ship in [data/sample_images/](data/sample_images/) so you can sanity-check the pipeline without any data download:

```bash
python inference.py data/sample_images/                  # all three
python inference.py data/sample_images/authentic.jpg     # one
```

> The samples are programmatically generated by [demo.py](demo.py) and are useful for end-to-end smoke testing. A model trained only on real Kaggle documents may flag them all as "forged" because they're out-of-distribution — real-world inputs are the better accuracy benchmark.

#### 5. Open the demo

Go to **[localhost:7860](http://localhost:7860)** — the Gradio demo detects your trained model automatically via the volume mount `./models:/app/models`.

> **Train while demo is running?** No restart needed. The demo checks for `models/classifier.pth` on every upload — it picks up newly trained models automatically.

---

## Services

| Service | URL | Purpose |
|---|---|---|
| Forgery API | http://localhost:8000 | REST endpoint |
| API docs | http://localhost:8000/docs | Swagger interactive UI |
| Demo app | http://localhost:7860 | Gradio visual interface |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| **Docs site** | **http://localhost:3001** | **Mintlify documentation** |
| Grafana | http://localhost:3000 | Log search & visualization |

---

## Makefile Reference

```bash
# Docker
make up              # start all 7 services (builds images first)
make down            # stop all services
make restart         # rebuild + restart
make status          # show container health
make logs            # tail all container stdout
make build           # force rebuild API + demo images

# Training (runs on host, not in Docker)
make download        # python kaggle_download.py --all
make generate        # generate 1000 synthetic forgeries
make split           # prepare train/val/test split
make train           # python train_classifier.py
make train-yolo      # python train_yolo.py
make evaluate        # python scripts/evaluate.py
make fetch-weights   # download a published classifier.pth (set MODEL_URL=...)
make test            # run the pytest suite

# Tools
make demo            # python demo.py  →  demo_output/
make docs            # MkDocs → localhost:8080
make mintlify        # Mintlify → localhost:3001 (requires Node 20 LTS)
make venv            # create forgery_env/
make install         # pip install -r requirements.txt
```

---

## Configuration

```yaml
classifier:
  backbone: resnet18      # resnet18 (CPU) | resnet50 (GPU)
  batch_size: 16          # lower for CPU, 32+ for GPU
  epochs: 15
  lr: 0.001
  patience: 10

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

Every `make train` logs automatically. Open **http://localhost:5000** after `make up`.

**Logged per run:** backbone, lr, batch_size, device, per-epoch val_auc/val_loss/f1, best checkpoint, evaluation plots, `early_stopped` tag.

---

## Log Monitoring — Grafana

Open **http://localhost:3000** after `make up`. No login required.

```logql
{script="train_classifier"}                 # all training logs
{script="train_classifier"} |= "VlAUC"     # AUC metric lines only
{app="forgery-detection", level="ERROR"}    # all errors
{script="train_classifier"} | live         # live tail during training
```

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

---

## Demo App (Gradio)

After `make up`, open **http://localhost:7860**.

- Upload via drag-and-drop or clipboard paste
- Original · ELA map · suspicious regions shown side by side
- Interactive Plotly gauge with green/yellow/red zones
- Score breakdown: ELA / CNN / combined
- Model status shown in sidebar (loaded / not trained yet)
- **Hot-reload:** train a new model and upload the next image — demo uses it immediately

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

## YOLO — Stamp & Signature Detection (Optional)

1. Label images at [roboflow.com](https://roboflow.com) — classes: `signature`, `stamp`
2. Export as **YOLOv8 format**
3. Train: `python train_yolo.py --data path/to/data.yaml`

Best weights auto-copied to `models/best.pt` and loaded by the API and demo on next upload.

---

## Documentation

Full documentation site included:

```bash
make up        # docs served at http://localhost:3001
# or
make mintlify  # requires Node 20 LTS via fnm
```

---

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| Real-World Documents | [Kaggle](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) | Authentic training images |
| Tobacco3482 | [Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482dataset) | Authentic scanned documents |
| Signature Verification | [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) | Splice attack donors |
| Handwritten Signatures | [Kaggle](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signatures) | Splice attack donors |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Docker (for `make up`)
- 8 GB RAM minimum for CPU training

---

## License

MIT License — see [LICENSE](LICENSE) for details.
