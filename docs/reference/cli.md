# CLI Reference

## `make` commands

```bash
# Docker stack
make up              # build + start all 6 services
make down            # stop all services
make restart         # down + up
make status          # show container health and ports
make logs            # live tail of all container stdout
make build           # force-rebuild API + demo images

# Training (runs on host, not in Docker)
make download        # fetch Kaggle datasets
make generate        # create synthetic forgeries
make split           # train/val/test split
make train           # train CNN classifier
make evaluate        # evaluate on test set

# Tools
make demo            # generate static PNG dashboards → demo_output/
make venv            # create forgery_env/
make install         # pip install -r requirements.txt
```

---

## `inference.py`

Analyze any image or folder from the command line.

```bash
# Single image
python inference.py path/to/document.jpg

# Entire folder
python inference.py path/to/folder/

# Save results to JSON
python inference.py path/to/folder/ --output results.json

# Save ELA visualization for each image
python inference.py document.jpg --visualize
```

**Output:**
```
document.jpg    FORGED ⚠  prob=0.823  ela=0.312  sigs=1  stamps=0
contract.jpg    REAL   ✓  prob=0.134  ela=0.051  sigs=0  stamps=1
```

---

## `demo.py`

Generate static analysis dashboards.

```bash
python demo.py                              # auto-generate test images
python demo.py --image path/to/doc.jpg      # analyze your own image
python demo.py --no-show                    # save only (no GUI window)
python demo.py --save-dir path/             # custom output folder
```

**Output files in `demo_output/`:**

| File | Content |
|---|---|
| `dashboard_authentic.png` | Original · ELA map · regions · gauge · bars · summary |
| `dashboard_forged.png` | Same layout for the forged version |
| `comparison.png` | Side-by-side 3×4 comparison grid |
| `sample_authentic.jpg` | Generated authentic document |
| `sample_forged.jpg` | Generated forged document |

---

## `kaggle_download.py`

```bash
python kaggle_download.py --all
python kaggle_download.py --documents
python kaggle_download.py --signatures
python kaggle_download.py --all --limit 300
python kaggle_download.py --all --skip slug/name
```

---

## `train_classifier.py`

```bash
python train_classifier.py
python train_classifier.py --device cpu
python train_classifier.py --resume models/classifier.pth
python train_classifier.py --config path/to/config.yaml
```

---

## `train_yolo.py`

```bash
python train_yolo.py --data path/to/data.yaml
python train_yolo.py --data path/to/data.yaml --device cpu
```

---

## `scripts/evaluate.py`

```bash
python scripts/evaluate.py --test-dir data/test
python scripts/evaluate.py --test-dir data/test --model models/classifier.pth
```

---

## `app.py`

```bash
python app.py           # starts Gradio on http://localhost:7860
```
