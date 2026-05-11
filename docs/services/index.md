# Services

All services start with one command:

```bash
make up
```

<div class="cards" markdown>

<div class="card" markdown>
<div class="card-icon">🔌</div>
**[REST API](api.md)**  
`localhost:8000` — FastAPI endpoint for programmatic document analysis.
</div>

<div class="card" markdown>
<div class="card-icon">🖥️</div>
**[Demo App](demo.md)**  
`localhost:7860` — Gradio interface for drag-and-drop document analysis.
</div>

<div class="card" markdown>
<div class="card-icon">📋</div>
**[Log Monitoring](monitoring.md)**  
`localhost:3000` — Grafana + Loki for searching and filtering all script logs.
</div>

</div>

## All services

| Container | Port | Accessible? |
|---|---|---|
| `forgery_api` | 8000 | Yes |
| `forgery_demo` | 7860 | Yes |
| `forgery_mlflow` | 5000 | Yes |
| `forgery_grafana` | 3000 | Yes |
| `forgery_loki` | 3100 | Internal only |
| `forgery_promtail` | — | Internal only |

## Useful commands

```bash
make status    # health of all containers
make logs      # tail stdout of all containers
make down      # stop everything
make restart   # rebuild + restart
```
