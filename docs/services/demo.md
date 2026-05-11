# Demo App

The Gradio web interface lets you analyze documents interactively without writing any code.

**URL:** [localhost:7860](http://localhost:7860)

---

## Features

- **Drag-and-drop upload** — JPEG, PNG, WebP, or paste from clipboard
- **Auto-analysis** — runs immediately on upload, no button needed
- **Three-image gallery** — Original | ELA Map | Suspicious Regions
- **Interactive gauge** — Plotly gauge chart with green/yellow/red zones and delta indicator
- **Score breakdown** — animated bars for ELA, CNN, and combined scores
- **Verdict banner** — green for authentic, red for forged
- **Detection count** — signatures and stamps detected by YOLO
- **Model status sidebar** — shows which models are loaded and links to other services

---

## ELA Map explained

The ELA (Error Level Analysis) map shows JPEG compression inconsistencies:

- **Dark areas** — compressed uniformly, likely original content
- **Bright areas** — different compression history, possible editing
- **Red boxes** — top suspicious regions found by contour analysis

---

## Before vs after training

| State | What you see |
|---|---|
| Before `make train` | ELA score only. CNN gauge uses ELA as fallback. Sidebar: "⚠️ Run make train" |
| After `make train` | Full ELA + CNN combined score. Gauge reflects true model confidence. |
| After `train_yolo.py` | Signature/stamp count appears in the score breakdown |

---

## Run locally (without Docker)

```bash
source forgery_env/bin/activate
python app.py
# Open http://localhost:7860
```

---

## Static demo (save PNG reports)

For reports, presentations, or GitHub screenshots:

```bash
make demo
# Saves to demo_output/: dashboard_authentic.png, dashboard_forged.png, comparison.png
```
