---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Document Forgery Detection

**AI-powered system for detecting tampered documents —  
forged signatures, replaced stamps, and copy-move manipulations.**

<div class="hero-badges">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white">
  <img src="https://img.shields.io/badge/Gradio-4.x-FF7C00">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white">
</div>

</div>

<div class="cards" markdown>

<div class="card" markdown>
<div class="card-icon">🔍</div>
**Three-Layer Detection**

ELA compression analysis, ResNet CNN classifier, and YOLOv8 stamp/signature detector — all combined into one confidence score.

[Learn more →](reference/architecture.md)
</div>

<div class="card" markdown>
<div class="card-icon">⚡</div>
**One-Command Stack**

`make up` starts all six services: API, demo app, MLflow, Grafana, Loki, and Promtail.

[Get started →](getting-started/quickstart.md)
</div>

<div class="card" markdown>
<div class="card-icon">📊</div>
**Experiment Tracking**

Every training run auto-logged to MLflow — compare models, view metric curves, download checkpoints.

[MLflow guide →](training/mlflow.md)
</div>

<div class="card" markdown>
<div class="card-icon">🖥️</div>
**Interactive Demo**

Gradio web app — drag and drop any document to get an instant analysis with gauge charts and ELA visualisation.

[Demo app →](services/demo.md)
</div>

<div class="card" markdown>
<div class="card-icon">📋</div>
**Log Monitoring**

All script logs shipped to Grafana + Loki. Search, filter, and live-tail training runs from the browser.

[Grafana guide →](services/monitoring.md)
</div>

<div class="card" markdown>
<div class="card-icon">🧪</div>
**Synthetic Data**

No forgery dataset? Generate thousands of copy-move and splice forgeries from any real document images.

[Data pipeline →](data-pipeline/generate.md)
</div>

</div>

---

## How it works

```
Input image
     │
     ├──► ELA Analyzer ──────────────► ela_score  (0 – 1)
     │        JPEG compression diff
     │        suspicious region boxes
     │
     ├──► ResNet Classifier ─────────► cls_score  (0 – 1)
     │        pretrained backbone
     │        fine-tuned on real + synthetic docs
     │
     └──► YOLOv8 Detector ──────────► detections  (bbox + conf)
              signatures and stamps

final_score = 0.35 × ela + 0.65 × cnn
verdict     = "FORGED" if final_score > 0.5
```

---

## Services at a glance

After running `make up`:

| Service | URL | Purpose |
|---|---|---|
| REST API | [localhost:8000](http://localhost:8000) | Forgery detection endpoint |
| API docs | [localhost:8000/docs](http://localhost:8000/docs) | Swagger interactive UI |
| Demo app | [localhost:7860](http://localhost:7860) | Gradio visual interface |
| MLflow | [localhost:5000](http://localhost:5000) | Experiment tracking |
| Grafana | [localhost:3000](http://localhost:3000) | Log search & visualization |

---

## Quick install

```bash
git clone https://github.com/your-username/document-forgery-detection.git
cd document-forgery-detection

make venv && source forgery_env/bin/activate
make install
make up
```

[Full installation guide →](getting-started/installation.md)
