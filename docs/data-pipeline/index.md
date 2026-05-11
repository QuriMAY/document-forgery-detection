# Data Pipeline

Real forgery datasets are rare and often restricted. This project uses a three-stage pipeline to build a training dataset from scratch.

```
Kaggle datasets          Synthetic forgeries         Training splits
─────────────            ───────────────────         ───────────────
data/real/    ──────────► data/generated/real/   ──► data/train/
data/patches/ ──forge──► data/generated/forged/  ──► data/val/
                                                  ──► data/test/
```

<div class="cards" markdown>

<div class="card" markdown>
<div class="card-icon">📥</div>
**[Download Datasets](download.md)**  
Fetch authentic documents and signature images from Kaggle automatically.
</div>

<div class="card" markdown>
<div class="card-icon">🔧</div>
**[Generate Forgeries](generate.md)**  
Create copy-move and splice forgeries from your real images.
</div>

<div class="card" markdown>
<div class="card-icon">✂️</div>
**[Prepare Splits](prepare.md)**  
Stratified 70/15/15 train/val/test split.
</div>

</div>
