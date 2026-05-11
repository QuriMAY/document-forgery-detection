# ─────────────────────────────────────────────────────────────────────────────
#  Document Forgery Detection — Makefile
#  Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   = forgery_env/bin/python
COMPOSE  = docker compose
VENV     = forgery_env

# Weights download — override with `make fetch-weights MODEL_URL=https://...`
MODEL_URL ?=
MODEL_PATH ?= models/classifier.pth

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Document Forgery Detection — available commands"
	@echo ""
	@echo "  DOCKER"
	@echo "    make up          Start all services (API + MLflow + Grafana + Loki)"
	@echo "    make down        Stop all services"
	@echo "    make restart     Restart all services"
	@echo "    make status      Show running containers and their ports"
	@echo "    make logs        Tail live logs from all containers"
	@echo "    make build       Rebuild the API Docker image"
	@echo ""
	@echo "  TRAINING  (run on host, outside Docker)"
	@echo "    make download       Download datasets from Kaggle"
	@echo "    make generate       Generate synthetic forged documents"
	@echo "    make split          Split data into train / val / test"
	@echo "    make train          Train the CNN classifier"
	@echo "    make train-yolo     Train the YOLO signature/stamp detector"
	@echo "    make evaluate       Evaluate trained model on test set"
	@echo "    make fetch-weights  Download a published classifier.pth (set MODEL_URL=...)"
	@echo "    make test           Run the pytest suite"
	@echo ""
	@echo "  TOOLS"
	@echo "    make demo           Run visual demo and save figures to demo_output/"
	@echo "    make docs           Serve documentation locally (http://localhost:8080)"
	@echo "    make docs-build     Build static documentation site → site/"
	@echo "    make venv           Create the virtual environment"
	@echo "    make install        Install Python dependencies into venv"
	@echo ""
	@echo "  URLs (after make up)"
	@echo "    API         →  http://localhost:8000"
	@echo "    API docs    →  http://localhost:8000/docs"
	@echo "    MLflow      →  http://localhost:5000"
	@echo "    Grafana     →  http://localhost:3000"
	@echo ""

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: up
up:
	@echo "→ Preparing data directories..."
	@mkdir -p mlartifacts logs models
	@echo "→ Starting all services..."
	@$(COMPOSE) up -d --build
	@echo ""
	@echo "  All services are starting. URLs:"
	@echo "    API         →  http://localhost:8000"
	@echo "    API docs    →  http://localhost:8000/docs"
	@echo "    Demo app    →  http://localhost:7860"
	@echo "    MLflow UI   →  http://localhost:5000"
	@echo "    Docs site   →  http://localhost:3001   (takes ~60s on first start)"
	@echo "    Grafana     →  http://localhost:3000"
	@echo ""
	@echo "  Run 'make status' to check container health."

.PHONY: down
down:
	@$(COMPOSE) down
	@echo "All services stopped."

.PHONY: restart
restart: down up

.PHONY: status
status:
	@$(COMPOSE) ps

.PHONY: logs
logs:
	@$(COMPOSE) logs -f --tail=50

.PHONY: build
build:
	@$(COMPOSE) build --no-cache api mlflow

# ── Training (runs on host, uses virtual environment) ─────────────────────────
.PHONY: download
download:
	@$(PYTHON) kaggle_download.py --all

.PHONY: generate
generate:
	@$(PYTHON) scripts/generate_synthetic_data.py \
		--real-dir data/real \
		--donor-dir data/patches \
		--num-forged 1000

.PHONY: split
split:
	@$(PYTHON) scripts/prepare_dataset.py \
		--source-dir data/generated \
		--output-dir data

.PHONY: train
train:
	@$(PYTHON) train_classifier.py

.PHONY: train-yolo
train-yolo:
	@$(PYTHON) train_yolo.py

.PHONY: evaluate
evaluate:
	@$(PYTHON) scripts/evaluate.py --test-dir data/test

# Download a published classifier checkpoint instead of training from scratch.
# Usage: make fetch-weights MODEL_URL=https://github.com/<you>/<repo>/releases/download/v1.0/classifier.pth
.PHONY: fetch-weights
fetch-weights:
	@if [ -z "$(MODEL_URL)" ]; then \
		echo ""; \
		echo "  MODEL_URL is empty. Either:"; \
		echo "    1. Upload classifier.pth as a GitHub release asset, then run:"; \
		echo "       make fetch-weights MODEL_URL=https://github.com/<you>/<repo>/releases/download/<tag>/classifier.pth"; \
		echo "    2. Or train from scratch:  make download && make generate && make split && make train"; \
		echo ""; \
		exit 1; \
	fi
	@mkdir -p models
	@echo "→ Fetching $(MODEL_URL) → $(MODEL_PATH)"
	@curl -fsSL -o $(MODEL_PATH) "$(MODEL_URL)"
	@echo "→ Downloaded $$(du -h $(MODEL_PATH) | cut -f1) to $(MODEL_PATH)"

.PHONY: test
test:
	@MPLBACKEND=Agg PYTHONPATH=. $(VENV)/bin/pytest -q

# ── Tools ─────────────────────────────────────────────────────────────────────
.PHONY: demo
demo:
	@$(PYTHON) demo.py

.PHONY: venv
venv:
	python -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

.PHONY: install
install:
	@$(VENV)/bin/pip install --upgrade pip -q
	@$(VENV)/bin/pip install -r requirements.txt
	@echo "Dependencies installed."

# ── Docs ──────────────────────────────────────────────────────────────────────
.PHONY: docs
docs:
	@$(VENV)/bin/pip install -r requirements-docs.txt -q
	@$(VENV)/bin/mkdocs serve
	@echo "MkDocs → http://localhost:8080"

.PHONY: docs-build
docs-build:
	@$(VENV)/bin/pip install -r requirements-docs.txt -q
	@$(VENV)/bin/mkdocs build
	@echo "Built → site/"

.PHONY: mintlify
mintlify:
	@echo "Starting Mintlify docs..."
	@if command -v fnm > /dev/null 2>&1; then \
		cd mintlify-docs && fnm exec --using=20 npx mintlify dev; \
	elif command -v nvm > /dev/null 2>&1; then \
		cd mintlify-docs && nvm exec 20 npx mintlify dev; \
	else \
		echo ""; \
		echo "  Node 20 LTS required. Install fnm first:"; \
		echo "    curl -fsSL https://fnm.vercel.app/install | bash"; \
		echo "    source ~/.zshrc  (or ~/.bashrc)"; \
		echo "    fnm install 20"; \
		echo ""; \
		echo "  Then retry: make mintlify"; \
		echo ""; \
	fi
