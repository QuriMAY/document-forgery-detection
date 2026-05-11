# REST API

The FastAPI backend exposes a single analysis endpoint and a health check.

**Base URL:** `http://localhost:8000`  
**Interactive docs:** [localhost:8000/docs](http://localhost:8000/docs)

---

## `POST /analyze`

Upload a document image for forgery analysis.

### Request

```bash
curl -X POST http://localhost:8000/analyze \
     -F "file=@document.jpg"
```

| Parameter | Type | Description |
|---|---|---|
| `file` | form-data | JPEG, PNG, or WebP image. Max 10 MB. |

### Response

```json
{
  "status": "success",
  "is_forged": true,
  "forgery_probability": 0.823,
  "confidence": "high",
  "scores": {
    "ela": 0.312,
    "classifier": 0.921,
    "combined": 0.823
  },
  "detections": {
    "signatures": [
      {"bbox": [120, 450, 380, 510], "confidence": 0.91}
    ],
    "stamps": []
  },
  "suspicious_regions": [
    {"bbox": [100, 200, 300, 350], "area": 30000}
  ],
  "processing_time_ms": 312.4
}
```

### Confidence levels

| Value | Gap from threshold | Meaning |
|---|---|---|
| `high` | > 0.30 | Very reliable verdict |
| `medium` | 0.15 – 0.30 | Fairly confident |
| `low` | < 0.15 | Borderline — inspect manually |

### Error responses

| Code | Reason |
|---|---|
| `415` | Unsupported file type |
| `413` | File too large (> 10 MB) |
| `500` | Internal analysis error |

---

## `GET /health`

Liveness check — also shows which models are loaded.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "running",
  "models": {
    "classifier": true,
    "yolo": false
  }
}
```

---

## Python client example

```python
import requests

with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
    )

result = response.json()
print(f"Forged: {result['is_forged']}")
print(f"Probability: {result['forgery_probability']:.1%}")
```
