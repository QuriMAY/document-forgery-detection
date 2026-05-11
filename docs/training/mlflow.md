# Experiment Tracking with MLflow

Every training run is logged automatically — no extra commands needed.

---

## Open the UI

MLflow starts with `make up` at [localhost:5000](http://localhost:5000).

Or run standalone on the host:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## What is logged automatically

### Parameters (once per run)

`backbone` · `batch_size` · `lr` · `weight_decay` · `epochs` · `image_size` · `device` · `train_images` · `val_images` · `amp`

### Metrics (every epoch)

`train_loss` · `train_acc` · `train_f1` · `val_loss` · `val_acc` · `val_f1` · `val_auc` · `lr` · `epoch_time_s`

### Artifacts

- `checkpoints/classifier.pth` — saved whenever a new best AUC is achieved
- `evaluation/confusion_matrix.png` — after running `make evaluate`
- `evaluation/roc_curve.png`
- `evaluation/pr_curve.png`
- `evaluation/report.txt`

### Tags

`early_stopped` · `pytorch_version`

---

## Using the UI

### Find the best run

1. Open the **forgery-classifier** experiment
2. Click the **`val_auc`** column header to sort descending
3. Top row = best model

### Compare two runs

1. Tick checkboxes for two runs
2. Click **Compare**
3. Get overlaid metric charts — see which converges faster

### Filter runs

Use the search bar:

```
metrics.val_auc > 0.85
params.backbone = "resnet50"
params.lr < 0.001
```

### Download a checkpoint

1. Click a run name
2. Go to **Artifacts** tab
3. `checkpoints/classifier.pth` → click **Download**

---

## Run naming

Runs are named automatically:

```
resnet18_bs16_lr0.001
resnet50_bs32_lr0.0005
```

Format: `{backbone}_bs{batch_size}_lr{lr}`

---

## Storage

| File | Contents |
|---|---|
| `mlflow.db` | SQLite database — all run metadata and metrics |
| `mlartifacts/` | Binary artifacts — model weights and plots |

Both are listed in `.gitignore` and should not be committed.
