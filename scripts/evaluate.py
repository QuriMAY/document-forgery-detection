"""
Full evaluation of the trained classifier on the test split.

Outputs:
    evaluation/confusion_matrix.png
    evaluation/roc_curve.png
    evaluation/pr_curve.png
    evaluation/report.txt

Usage:
    python scripts/evaluate.py --test-dir data/test
    python scripts/evaluate.py --test-dir data/test --model models/classifier.pth
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.classifier import build_model
from src.utils import load_config, log_config, setup_logging

logger = logging.getLogger(__name__)


def evaluate(config: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config["classifier"]

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=min(4, os.cpu_count() or 1))

    class_names = [k for k, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])]
    logger.info(f"Test dir      : {args.test_dir}")
    logger.info(f"Model         : {args.model}")
    logger.info(f"Test images   : {len(dataset)}")
    logger.info(f"Classes       : {dataset.class_to_idx}")

    # ---- load model ----
    ckpt = torch.load(args.model, map_location=device)
    backbone = ckpt.get("backbone", cfg.get("backbone", "resnet50")) if isinstance(ckpt, dict) else cfg.get("backbone", "resnet50")
    model = build_model(backbone=backbone, num_classes=2, pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    # ---- inference ----
    all_preds, all_labels, all_probs = [], [], []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ---- metrics ----
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    auc    = roc_auc_score(all_labels, all_probs)
    ap     = average_precision_score(all_labels, all_probs)
    acc    = float(np.mean(all_preds == all_labels))

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Avg Prec : {ap:.4f}")
    print("=" * 60)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "report.txt"
    report_path.write_text(f"ROC-AUC: {auc:.4f}\nAvg Prec: {ap:.4f}\nAccuracy: {acc:.4f}\n\n{report}")

    # ---- confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve")
    ax.legend(); plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    # ---- Precision-Recall curve ----
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve")
    ax.legend(); plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=150); plt.close()

    # ---- MLflow — log test metrics + plots into the latest training run ----
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("forgery-classifier")

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_auc":      round(auc, 6),
            "test_avg_prec": round(ap,  6),
            "test_accuracy": round(acc, 6),
        })
        mlflow.log_artifact(str(out / "confusion_matrix.png"), artifact_path="evaluation")
        mlflow.log_artifact(str(out / "roc_curve.png"),        artifact_path="evaluation")
        mlflow.log_artifact(str(out / "pr_curve.png"),         artifact_path="evaluation")
        mlflow.log_artifact(str(report_path),                  artifact_path="evaluation")

    logger.info("=" * 70)
    logger.info("Evaluation complete.")
    logger.info(f"  ROC-AUC    : {auc:.4f}")
    logger.info(f"  Avg Prec   : {ap:.4f}")
    logger.info(f"  Accuracy   : {acc:.4f}")
    logger.info(f"  Output dir : {out}/")
    logger.info(f"  Files      : confusion_matrix.png  roc_curve.png  pr_curve.png  report.txt")
    logger.info(f"  MLflow UI  : mlflow ui  →  http://localhost:5000")
    logger.info("=" * 70)


def main():
    p = argparse.ArgumentParser(description="Evaluate trained forgery classifier")
    p.add_argument("--test-dir",    required=True, help="data/test folder with class sub-dirs")
    p.add_argument("--model",       default="models/classifier.pth")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--output-dir",  default="evaluation")
    args = p.parse_args()

    setup_logging("evaluate")
    cfg = load_config(args.config)
    log_config(cfg, logging.getLogger("evaluate"))
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
