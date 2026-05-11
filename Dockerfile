FROM python:3.10-slim

# libgl1-mesa-glx was renamed to libgl1 in Debian Bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch CPU-only ─────────────────────────────────────────────────────────
# Containers never use the GPU, so we install the CPU wheel (~180 MB vs 530 MB).
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    "torch>=2.0.0,<3.0.0" "torchvision>=0.15.0,<1.0.0" \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining dependencies ───────────────────────────────────────────────────
# Strip torch/torchvision so pip doesn't try to reinstall them.
COPY requirements.txt .
RUN grep -v -E "^(torch|torchvision)" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir --timeout 300 --retries 5 -r /tmp/req.txt

COPY . .

# Create writable dirs and a non-root user that owns them.
RUN mkdir -p temp models logs && \
    useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
