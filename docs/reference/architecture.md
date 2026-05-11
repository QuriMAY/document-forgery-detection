# Architecture

## Detection pipeline

```
Input image (JPEG / PNG / WebP)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                       │
│                                                            │
│  ┌─────────────────┐   ┌──────────────────────────────┐  │
│  │  ELA Analyzer   │   │    ResNet Classifier          │  │
│  │                 │   │                               │  │
│  │  re-compress    │   │  224×224 resize               │  │
│  │  at quality=90  │   │  ImageNet normalise           │  │
│  │  abs diff × 15  │   │  backbone + custom head       │  │
│  │  top-10% score  │   │  softmax → P(forged)          │  │
│  │  region contours│   │                               │  │
│  └────────┬────────┘   └──────────────┬───────────────┘  │
│           │                           │                    │
│           │   ela_score (0–1)         │  cls_score (0–1)  │
│           └─────────────┬─────────────┘                   │
│                         │                                  │
│           combined = 0.35×ela + 0.65×cnn                  │
│           verdict  = combined > 0.5 → FORGED               │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  YOLOv8 Detector  (optional)                         │ │
│  │  → signature bboxes  → stamp bboxes                  │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## Infrastructure

```
Host machine                          Docker network
──────────────────────                ─────────────────────────────────
train_classifier.py
scripts/evaluate.py   ─writes──► mlflow.db  ◄──── forgery_mlflow :5000
                      ─writes──► mlartifacts

kaggle_download.py
prepare_dataset.py    ─writes──► logs/ ◄──── Promtail → Loki → Grafana :3000
train_classifier.py

models/classifier.pth ◄──────────────────────────── forgery_api  :8000
models/best.pt                                       forgery_demo :7860
```

---

## Component responsibilities

| Component | File | Role |
|---|---|---|
| ELA Analyzer | `src/ela.py` | Detects compression inconsistencies |
| Classifier | `src/classifier.py` | ResNet inference wrapper |
| Detector | `src/detector.py` | YOLOv8 inference wrapper |
| Data Generator | `src/data_generator.py` | Copy-move + splice forgery synthesis |
| Utils | `src/utils.py` | Logging, config, .env loading |
| FastAPI | `api/main.py` | REST endpoint |
| Gradio app | `app.py` | Web demo interface |
| Training | `train_classifier.py` | Full training loop |
| Evaluation | `scripts/evaluate.py` | Test-set metrics + MLflow upload |

---

## Model weights

| File | Created by | Used by |
|---|---|---|
| `models/classifier.pth` | `train_classifier.py` | API, demo app, `inference.py` |
| `models/best.pt` | `train_yolo.py` | API, demo app, `inference.py` |

Both are in `.gitignore` — copy them manually or use DVC for versioning.
