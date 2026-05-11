# Prepare Dataset Splits

Split the generated data into stratified train / val / test sets before training.

---

## Why three splits?

| Split | Size | Purpose |
|---|---|---|
| **train** | 70% | Model learns from this |
| **val** | 15% | Checked after every epoch to detect overfitting |
| **test** | 15% | Used **once** at the end for the final honest score |

!!! danger "Never train on test data"
    The test split must never influence any training or hyperparameter decision. It is your unbiased final metric.

---

## Command

```bash
make split
```

Which runs:

```bash
python scripts/prepare_dataset.py \
    --source-dir data/generated \
    --output-dir data
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--source-dir` | — | Directory with `real/` and `forged/` subdirs |
| `--output-dir` | `data` | Where to write train/val/test |
| `--train-ratio` | `0.70` | Fraction for training |
| `--val-ratio` | `0.15` | Fraction for validation |
| `--seed` | `42` | Shuffle seed |

---

## Output structure

```
data/
├── train/
│   ├── real/    (3 500 images)
│   └── forged/  (2 100 images)
├── val/
│   ├── real/    (  750 images)
│   └── forged/  (  450 images)
└── test/
    ├── real/    (  750 images)
    └── forged/  (  450 images)
```

!!! note "Stratification"
    The split is performed per class, so the real:forged ratio is preserved across all three splits.
