# ─────────────────────────────────────────────────────────────────────────────
#  Document Forgery Detection — Makefile
#  Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   = forgery_env/bin/python
COMPOSE  = docker compose
VENV     = forgery_env

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
	@echo "    make download    Download datasets from Kaggle"
	@echo "    make generate    Generate synthetic forged documents"
	@echo "    make split       Split data into train / val / test"
	@echo "    make train       Train the CNN classifier"
	@echo "    make evaluate    Evaluate trained model on test set"
	@echo ""
	@echo "  TOOLS"
	@echo "    make demo        Run visual demo and save figures to demo_output/"
	@echo "    make venv        Create the virtual environment"
	@echo "    make install     Install Python dependencies into venv"
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
	@touch mlflow.db
	@mkdir -p mlartifacts logs models
	@echo "→ Starting all services..."
	@$(COMPOSE) up -d --build
	@echo ""
	@echo "  All services are starting. URLs:"
	@echo "    API         →  http://localhost:8000"
	@echo "    API docs    →  http://localhost:8000/docs"
	@echo "    MLflow UI   →  http://localhost:5000"
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

.PHONY: evaluate
evaluate:
	@$(PYTHON) scripts/evaluate.py --test-dir data/test

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
