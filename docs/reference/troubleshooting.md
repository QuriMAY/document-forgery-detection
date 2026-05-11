# Troubleshooting

---

## Installation

??? failure "Docker build fails with `exit code: 100`"
    Package name changed in Debian Bookworm. The Dockerfile already uses the correct names:
    ```
    libgl1         (not libgl1-mesa-glx)
    libxrender1    (not libxrender-dev)
    ```
    If you see this, pull the latest code and retry `make up`.

??? failure "`ModuleNotFoundError: No module named 'src'`"
    The `PYTHONPATH` is not set. Check your `.env` file contains:
    ```
    PYTHONPATH=.
    ```
    Or set it manually: `export PYTHONPATH=.`

??? failure "`pip install` fails or is very slow"
    Upgrade pip first:
    ```bash
    forgery_env/bin/pip install --upgrade pip
    ```

---

## Data

??? failure "`No images found in data/real/`"
    Run `make download` first, or manually copy images into `data/real/`.

??? failure "Kaggle credentials not found"
    Check `.env` has `KAGGLE_USERNAME` and `KAGGLE_KEY` set correctly.  
    Run scripts from the project root directory (where `.env` lives).

??? failure "`KaggleApi` import error"
    Your Kaggle package is v2+. The project uses `KaggleApi` (not `KaggleApiExtended`).  
    Pull the latest code — the import has already been fixed.

---

## Training

??? failure "Training appears frozen (no output)"
    Training is running — there was no per-batch progress bar.  
    Pull the latest code — `tqdm` progress bars have been added:
    ```
    Epochs:  47%|████████                  | 7/15 [best=0.847]
      train: 100%|████████████████████████| 350/350 [loss=0.324]
    ```

??? failure "Training loss not decreasing"
    - Dataset is probably too small. Use `--limit 1000` or more in `kaggle_download.py`.
    - Check that `data/train/real/` and `data/train/forged/` both contain images.
    - Try reducing the learning rate: `lr: 0.0003`

??? failure "Out of memory during training"
    Lower `batch_size` in `config.yaml`:
    ```yaml
    batch_size: 8   # or 4
    ```

??? failure "`FutureWarning: torch.cuda.amp.GradScaler is deprecated`"
    Pull the latest code — this has been fixed (uses `torch.amp.GradScaler` now).

??? failure "`pin_memory` UserWarning on CPU"
    Pull the latest code — `pin_memory` is now only enabled when CUDA is available.

---

## Services

??? failure "`make up` fails with Docker build error"
    Try a clean rebuild:
    ```bash
    make down
    docker system prune -f    # clears cached layers
    make up
    ```

??? failure "Container unhealthy or not starting"
    Check which container failed:
    ```bash
    make status
    make logs
    ```

??? failure "MLflow UI shows no runs"
    - Confirm `make up` ran successfully and `mlflow.db` exists in the project root.
    - Training must complete at least one epoch to create a run.
    - Make sure you're running training from the project root.

??? failure "Grafana shows no logs"
    - `make up` must be running **before** you start training scripts.
    - Promtail only ships lines written after it started.
    - Existing `.log` files are readable in `logs/` on disk regardless.

??? failure "Port already in use"
    Another process is using the port. Find and kill it:
    ```bash
    lsof -i :8000    # or :7860, :5000, :3000
    kill -9 <PID>
    ```
    Or change the port in `docker-compose.yml`.

---

## Inference

??? failure "Classifier model not found"
    `models/classifier.pth` does not exist yet. Run `make train` first.

??? failure "YOLO model not found (warning only)"
    `models/best.pt` is optional. The API works without it.  
    Run `python train_yolo.py --data path/to/data.yaml` to train it.

---

## Getting help

If the issue persists:

1. Check the log file in `logs/<script_name>/` — it contains DEBUG-level details.
2. Open an issue on [GitHub](https://github.com/your-username/document-forgery-detection/issues).
