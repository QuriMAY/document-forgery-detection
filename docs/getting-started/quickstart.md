# Quick Start

Get from zero to a running system in one session.

---

## Complete workflow

```bash
# 1. Environment (one time)
make venv && source forgery_env/bin/activate && make install

# 2. Data pipeline
make download                  # fetch datasets from Kaggle
make generate                  # create 1000 synthetic forgeries
make split                     # 70/15/15 train/val/test split

# 3. Train
make train                     # ResNet18 classifier (~30–60 min on CPU)
make evaluate                  # test-set metrics + MLflow upload

# 4. Launch everything
make up                        # all six Docker services
```

Open **[localhost:7860](http://localhost:7860)** → upload any document → instant analysis.

---

## Step by step

### Download datasets

```bash
make download
```

Downloads 4 Kaggle datasets into `data/real/` and `data/patches/`.

!!! tip "Want a quick test first?"
    ```bash
    python kaggle_download.py --all --limit 300
    ```
    Downloads only 300 images per dataset — much faster, good for verifying the pipeline.

---

### Generate synthetic forgeries

```bash
make generate
```

Creates 1000 forged documents in `data/generated/forged/` using copy-move and splice attacks.

---

### Split the data

```bash
make split
```

Produces stratified splits:

```
data/train/real/     data/val/real/     data/test/real/
data/train/forged/   data/val/forged/   data/test/forged/
```

---

### Train the classifier

```bash
make train
```

You'll see live progress bars during training:

```
Epochs:  47%|██████████████                | 7/15 [08:23<09:32, best=0.8471]
  train: 100%|████████████████████████████| 350/350 [01:02<00:00, loss=0.3241]
    val: 100%|████████████████████████████|  75/75  [00:13<00:00, loss=0.4102]
Epoch   7/15 | TrLoss 0.3241 TrAcc 0.8712 | VlLoss 0.4102 VlAcc 0.8450 VlF1 0.8312 VlAUC 0.9021 | 75.4s
  ✓ Best model saved  (AUC=0.9021)
```

Best model is saved to `models/classifier.pth` automatically.

---

### Evaluate

```bash
make evaluate
```

Produces `evaluation/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, and `report.txt`.  
Results are also uploaded to MLflow.

---

### Launch all services

```bash
make up
```

All six services start in Docker:

| Service | URL |
|---|---|
| REST API | [localhost:8000](http://localhost:8000) |
| API docs | [localhost:8000/docs](http://localhost:8000/docs) |
| Demo app | [localhost:7860](http://localhost:7860) |
| MLflow  | [localhost:5000](http://localhost:5000) |
| Grafana | [localhost:3000](http://localhost:3000) |

Check health:
```bash
make status
```
