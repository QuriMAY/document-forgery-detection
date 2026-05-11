# Installation

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 20 GB |
| GPU | Not required | NVIDIA (CUDA) |
| Docker | 24+ | Latest |

---

## 1. Clone the repository

```bash
git clone https://github.com/your-username/document-forgery-detection.git
cd document-forgery-detection
```

---

## 2. Create the virtual environment

```bash
make venv
```

Activate it — you must do this every time you open a new terminal:

=== "Linux / Mac"
    ```bash
    source forgery_env/bin/activate
    ```

=== "Windows"
    ```bash
    forgery_env\Scripts\activate
    ```

!!! tip
    You'll see `(forgery_env)` at the start of your prompt when it's active.

---

## 3. Install Python dependencies

```bash
make install
```

This installs: PyTorch, OpenCV, FastAPI, YOLOv8, MLflow, Gradio, Kaggle API, scikit-learn, TensorBoard, and all other dependencies from `requirements.txt`.

!!! note "Takes 3–10 minutes"
    Download time depends on your internet connection. PyTorch is the largest package (~2 GB).

---

## 4. Configure credentials

Open `.env` in the project root and set your Kaggle API key:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
PYTHONPATH=.
```

**Get your Kaggle key:**

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Account → API → **Create New Token**
3. Opens `kaggle.json` — copy the values into `.env`

!!! warning "Never commit .env"
    The `.env` file is listed in `.gitignore` and will never be committed. Your keys are safe.

---

## 5. Install documentation dependencies (optional)

Only needed if you want to build this documentation site locally:

```bash
pip install -r requirements-docs.txt
```

---

## Verify the installation

```bash
python -c "import torch, cv2, ultralytics, fastapi, gradio, mlflow; print('All OK')"
```

Expected output: `All OK`
