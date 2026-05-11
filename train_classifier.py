#!/usr/bin/env python3
"""
Train the forgery detection CNN.

Expected data layout:
    data/train/real/     ← genuine document images
    data/train/forged/   ← tampered document images
    data/val/real/
    data/val/forged/

Usage:
    python train_classifier.py
    python train_classifier.py --config config.yaml --device cuda --resume models/classifier.pth
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from src.classifier import build_model
from src.utils import ensure_dirs, load_config, log_config, setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs so a run is reproducible. cuDNN benchmark is off in favor
    of determinism — small throughput cost but stable comparisons across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transform(image_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def __call__(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, use_amp):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  train", unit="batch", leave=False,
                bar_format="{l_bar}{bar:25}{r_bar}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def val_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc="    val", unit="batch", leave=False,
                bar_format="{l_bar}{bar:25}{r_bar}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, auc


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, args):
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    logger.info(f"Device        : {device}")
    logger.info(f"Mixed precision: {use_amp}")
    logger.info(f"PyTorch       : {torch.__version__}")
    logger.info(f"Seed          : {args.seed}")
    log_config(config, logger)

    cfg      = config["classifier"]
    data_cfg = config["data"]
    ensure_dirs("models", "runs/classifier")

    # ---- datasets ----
    train_ds = datasets.ImageFolder(data_cfg["train_dir"], transform=get_transform(cfg["image_size"], augment=True))
    val_ds   = datasets.ImageFolder(data_cfg["val_dir"],   transform=get_transform(cfg["image_size"], augment=False))

    logger.info(f"Train images  : {len(train_ds)}")
    logger.info(f"Val images    : {len(val_ds)}")
    logger.info(f"Classes       : {train_ds.class_to_idx}")

    counts  = np.bincount([s[1] for s in train_ds.samples])
    weights = torch.FloatTensor(len(train_ds) / (len(counts) * counts)).to(device)

    num_workers  = min(4, os.cpu_count() or 1)
    pin = device.type == "cuda"

    def _worker_init(worker_id: int) -> None:
        s = args.seed + worker_id
        np.random.seed(s)
        random.seed(s)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0,
                              worker_init_fn=_worker_init, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0,
                              worker_init_fn=_worker_init)

    # ---- model ----
    model = build_model(backbone=cfg.get("backbone", "resnet50"), num_classes=2, pretrained=True).to(device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ---- optimisation ----
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=cfg["epochs"], steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    scaler  = GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(patience=cfg.get("patience", 10))
    writer  = SummaryWriter("runs/classifier")

    # ---- MLflow ----
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("forgery-classifier")

    with mlflow.start_run(run_name=f"{cfg.get('backbone','resnet50')}_bs{cfg['batch_size']}_lr{cfg['lr']}"):

        # Log every hyperparameter from config
        mlflow.log_params({
            "backbone":     cfg.get("backbone", "resnet50"),
            "image_size":   cfg["image_size"],
            "batch_size":   cfg["batch_size"],
            "epochs":       cfg["epochs"],
            "lr":           cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "patience":     cfg.get("patience", 10),
            "device":       str(device),
            "train_images": len(train_ds),
            "val_images":   len(val_ds),
            "amp":          use_amp,
        })

        best_auc, best_epoch = 0.0, 0
        epoch_bar = tqdm(range(start_epoch, cfg["epochs"]), desc="Epochs",
                         unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")

        for epoch in epoch_bar:
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, use_amp)
            vl_loss, vl_acc, vl_f1, vl_auc = val_epoch(model, val_loader, criterion, device, use_amp)
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            # TensorBoard
            writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": tr_acc,  "val": vl_acc},  epoch)
            writer.add_scalars("F1",       {"train": tr_f1,   "val": vl_f1},   epoch)
            writer.add_scalar("Val/AUC", vl_auc, epoch)
            writer.add_scalar("LR", lr_now, epoch)

            # MLflow — one call logs all metrics for this step
            mlflow.log_metrics({
                "train_loss": round(tr_loss, 6),
                "train_acc":  round(tr_acc,  6),
                "train_f1":   round(tr_f1,   6),
                "val_loss":   round(vl_loss, 6),
                "val_acc":    round(vl_acc,  6),
                "val_f1":     round(vl_f1,   6),
                "val_auc":    round(vl_auc,  6),
                "lr":         round(lr_now,  8),
                "epoch_time_s": round(elapsed, 2),
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
                f"TrLoss {tr_loss:.4f} TrAcc {tr_acc:.4f} | "
                f"VlLoss {vl_loss:.4f} VlAcc {vl_acc:.4f} VlF1 {vl_f1:.4f} VlAUC {vl_auc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if vl_auc > best_auc:
                best_auc, best_epoch = vl_auc, epoch + 1
                ckpt_data = {
                    "epoch":                epoch + 1,
                    "backbone":             cfg.get("backbone", "resnet50"),
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc":              vl_auc,
                    "val_f1":              vl_f1,
                    "config":               cfg,
                }
                torch.save(ckpt_data, "models/classifier.pth")
                mlflow.log_artifact("models/classifier.pth", artifact_path="checkpoints")
                logger.info(f"  ✓ Best model saved  (AUC={vl_auc:.4f})")

            epoch_bar.set_postfix(vl_auc=f"{vl_auc:.4f}", vl_loss=f"{vl_loss:.4f}",
                                  best=f"{best_auc:.4f}")

            if stopper(vl_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Final summary metrics
        mlflow.log_metrics({"best_val_auc": best_auc, "best_epoch": best_epoch})
        mlflow.set_tags({
            "early_stopped": stopper.counter >= stopper.patience,
            "pytorch_version": torch.__version__,
        })

    writer.close()
    logger.info("=" * 70)
    logger.info("Training finished.")
    logger.info(f"  Best AUC   : {best_auc:.4f}")
    logger.info(f"  Best epoch : {best_epoch}")
    logger.info("  Model saved: models/classifier.pth")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train forgery classifier")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("train_classifier")
    args = parse_args()
    cfg  = load_config(args.config)
    train(cfg, args)
