# CNN Classifier

Trains a ResNet image classifier to distinguish authentic documents from forged ones.

---

## Architecture

```
Input (224×224 RGB)
       │
  ResNet backbone (pretrained ImageNet)
       │
  Global Average Pooling
       │
  Dropout(0.3) → Linear(→256) → ReLU → Dropout(0.2) → Linear(→2)
       │
  Softmax → [P(real), P(forged)]
```

The backbone is frozen during warmup (first 10% of steps via OneCycleLR) then fine-tuned end-to-end.

---

## Training command

```bash
make train
# or
python train_classifier.py
python train_classifier.py --device cpu
python train_classifier.py --resume models/classifier.pth   # continue from checkpoint
```

---

## What you see

```
Epochs:  47%|██████████████                | 7/15 [08:23<09:32, best=0.8471]
  train: 100%|████████████████████████████| 350/350 [01:02<00:00, loss=0.3241]
    val: 100%|████████████████████████████|  75/75  [00:13<00:00, loss=0.4102]
Epoch   7/15 | TrLoss 0.3241 TrAcc 0.8712 | VlLoss 0.4102 VlAcc 0.8450 VlF1 0.8312 VlAUC 0.9021 | 75.4s
  ✓ Best model saved  (AUC=0.9021)
```

### Reading the metrics

| Metric | Meaning | Target |
|---|---|---|
| `TrLoss` / `VlLoss` | Cross-entropy loss | Decreasing |
| `TrAcc` / `VlAcc` | Accuracy | > 0.80 |
| `VlF1` | Val F1 score | > 0.80 |
| `VlAUC` | Val ROC-AUC | > 0.85 = good, > 0.90 = excellent |

AUC is the primary metric — best AUC triggers a model save.

---

## Training features

| Feature | Details |
|---|---|
| Optimiser | AdamW |
| LR schedule | OneCycleLR (warmup + cosine decay) |
| Loss | CrossEntropyLoss with class weights |
| Regularisation | Dropout, weight decay, random erasing |
| Augmentation | Flip, crop, colour jitter, rotation, grayscale |
| Mixed precision | Auto-enabled on CUDA |
| Early stopping | Stops if val loss does not improve for `patience` epochs |
| Checkpointing | Best model (by AUC) saved to `models/classifier.pth` |

---

## Expected training time

| Hardware | Images | Epochs | Time |
|---|---|---|---|
| CPU (any) | 1 000 | 15 | 30–60 min |
| CPU (any) | 3 000 | 30 | 3–6 hours |
| NVIDIA GPU | 3 000 | 30 | 10–20 min |

!!! tip "First run on CPU"
    Set `epochs: 5` in `config.yaml` to verify the full pipeline works before committing to a long run.

---

## After training

```bash
make evaluate          # test-set metrics + MLflow artifact upload
```

This produces `evaluation/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, and `report.txt`.

---

## TensorBoard

Monitor live training curves in a second terminal:

```bash
tensorboard --logdir runs/classifier
# Open http://localhost:6006
```
