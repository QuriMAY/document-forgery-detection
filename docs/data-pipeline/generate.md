# Generate Synthetic Forgeries

Since real forgery datasets are rare, forgeries are generated programmatically using two attack types.

---

## Attack types

### Copy-Move

A region is copied from the same document and pasted at a different location — possibly rotated or scaled. Simulates someone duplicating a stamp or covering text.

```
Original:  [text][signature][stamp]
Forged:    [text][stamp    ][stamp]
                  ↑ pasted over signature
```

### Splice

A region from a completely different document is pasted onto the target. Simulates replacing a signature with someone else's, or inserting a fake stamp.

```
Donor doc:   [...][signature B][...]
Target doc:  [text][signature A][stamp]
Forged:      [text][signature B][stamp]
                    ↑ replaced
```

Both attack types include optional feathered edge blending to make the forgery less obvious.

---

## Command

```bash
make generate
```

Which runs:

```bash
python scripts/generate_synthetic_data.py \
    --real-dir   data/real    \
    --donor-dir  data/patches \
    --num-forged 1000
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--real-dir` | — | Directory of authentic images |
| `--donor-dir` | — | Signature images for splice attacks |
| `--num-forged` | Same as real count | Number of forgeries to generate |
| `--output-dir` | `data/generated` | Output directory |
| `--seed` | `42` | Random seed for reproducibility |

---

## Recommended sizes

| Size | Real | Forged | Use case |
|---|---|---|---|
| Minimum viable | 300 | 300 | Pipeline test |
| Good | 1 000 | 1 000 | First model |
| Best | 3 000+ | 3 000+ | Production |

---

## Output structure

```
data/generated/
├── real/      ← copies of your authentic images
└── forged/    ← newly created forgeries
```

!!! warning "Balance matters"
    Keep real and forged counts roughly equal. The training script handles class imbalance with weighted loss, but extreme imbalance (>5:1) will still hurt performance.
