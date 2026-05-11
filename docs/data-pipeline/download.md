# Download Datasets

The downloader fetches four Kaggle datasets and organises them automatically.

---

## Datasets

### Real documents → `data/real/`

| Dataset | Content | Size |
|---|---|---|
| [shaz13/real-world-documents-collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) | Invoices, letters, forms | ~400 MB |
| [patrickaudriaz/tobacco3482dataset](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482dataset) | Scanned office documents | ~1.2 GB |

### Signature images → `data/patches/`

| Dataset | Content | Used for |
|---|---|---|
| [robinreni/signature-verification-dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) | Genuine + forged signatures | Splice attack donors |
| [ishanikathuria/handwritten-signatures](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signatures) | Handwritten signatures | Splice attack donors |

---

## Commands

```bash
# Download everything
make download

# Or use the script directly for more control
python kaggle_download.py --all
python kaggle_download.py --documents          # only real documents
python kaggle_download.py --signatures         # only signature patches
python kaggle_download.py --all --limit 300    # 300 images per dataset (fast test)

# Skip a specific dataset
python kaggle_download.py --all --skip patrickaudriaz/tobacco3482dataset
```

---

## Expected download time

| Mode | Time |
|---|---|
| `--limit 300` | 2–5 minutes |
| `--documents` only | 10–20 minutes |
| Full `--all` | 20–60 minutes |

---

## Logs

All download activity is logged to:

```
logs/kaggle_download/kaggle_download_YYYY-MM-DD_HH-MM-SS.log
```

!!! note "Grafana"
    If `make up` is running, logs appear in real time at [localhost:3000](http://localhost:3000).  
    Use the query: `{script="kaggle_download"}`
