# Log Monitoring

Grafana + Loki + Promtail give you a searchable, filterable view of all script logs from the browser.

**URL:** [localhost:3000](http://localhost:3000) — no login required.

---

## How it works

```
Your training scripts    Promtail              Loki          Grafana
────────────────────     ────────              ────          ───────
python train_classifier  tails logs/           stores        web UI
python make evaluate  →  ships lines  →  →  →  indexed  →   query
python kaggle_download   parses format         by label      explore
```

Promtail watches `./logs/**/*.log` on your host machine and ships every new line to Loki with parsed labels.

---

## Labels

Every log line gets two labels:

| Label | Values |
|---|---|
| `script` | `train_classifier` · `evaluate` · `kaggle_download` · `demo` · `inference` · `prepare_dataset` · `generate_synthetic_data` |
| `level` | `INFO` · `WARNING` · `ERROR` |

---

## Useful LogQL queries

Open **Explore** in Grafana, select **Loki** as datasource, and type:

```logql
# All training logs
{script="train_classifier"}

# Live tail while training
{script="train_classifier"} | live

# AUC metric lines only
{script="train_classifier"} |= "VlAUC"

# All errors across every script
{app="forgery-detection", level="ERROR"}

# Kaggle download progress
{script="kaggle_download"}

# Everything from evaluation run
{script="evaluate"}
```

---

## Important note

!!! warning "Start make up before training"
    Promtail only ships log lines written **after** it starts. If you run training before `make up`, those logs exist in `logs/` files but won't appear in Grafana.
    
    Always start the stack first:
    ```bash
    make up
    make train
    ```

---

## Log files on disk

All logs are also saved permanently to `logs/<script_name>/` regardless of whether the Docker stack is running.

```
logs/
├── train_classifier/
│   └── train_classifier_2026-05-06_10-31-00.log
├── evaluate/
│   └── evaluate_2026-05-06_14-30-00.log
└── kaggle_download/
    └── kaggle_download_2026-05-06_09-00-00.log
```

Console → INFO level  
Log file → DEBUG level (full tracebacks included)
