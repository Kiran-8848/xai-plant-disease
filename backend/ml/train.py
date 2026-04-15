"""
Training pipeline for PlantDisease ResNet18.
Runs Experiment 1: Train → Evaluate accuracy.
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import yaml

from backend.ml.models.resnet_model import PlantDiseaseResNet, save_checkpoint, get_device
from backend.ml.utils.data_utils import build_dataloaders


# ─────────────────────────────────────────────
#  Core training / validation loops
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:  # AMP
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  Full training run
# ─────────────────────────────────────────────

def train(config: dict) -> PlantDiseaseResNet:
    device    = get_device()
    cfg_data  = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]

    # ── Data ──────────────────────────────────
    train_loader, val_loader, test_loader, dataset = build_dataloaders(
        data_root   = cfg_data["root"],
        image_size  = cfg_data["image_size"],
        batch_size  = cfg_data["batch_size"],
        num_workers = cfg_data["num_workers"],
        train_split = cfg_data["train_split"],
        val_split   = cfg_data["val_split"],
    )

    dataset.save_class_mapping(
        str(Path(cfg_train["save_dir"]) / "class_mapping.json")
    )

    # ── Model ─────────────────────────────────
    model = PlantDiseaseResNet(
        num_classes   = dataset.num_classes,
        pretrained    = cfg_model["pretrained"],
        dropout       = cfg_model["dropout"],
        freeze_layers = cfg_model["freeze_layers"],
    ).to(device)

    # ── Optimiser / Scheduler ─────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    if cfg_train["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg_train["epochs"]
        )
    elif cfg_train["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # AMP for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── TensorBoard ───────────────────────────
    writer = SummaryWriter(log_dir=cfg_train["log_dir"])

    best_val_acc   = 0.0
    patience_count = 0
    best_ckpt      = Path(cfg_train["save_dir"]) / "best_model.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"Training {cfg_model['architecture']} | Classes: {dataset.num_classes}")
    print(f"{'─'*60}\n")

    for epoch in range(1, cfg_train["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr      = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg_train['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", lr, epoch)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Best model + early stopping
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": val_acc, "val_loss": val_loss},
                str(best_ckpt),
                class_mapping=dataset.class_to_idx,
            )
        else:
            patience_count += 1
            if patience_count >= cfg_train["early_stopping_patience"]:
                print(f"\n[Early Stop] No improvement for {patience_count} epochs.")
                break

    writer.close()
    print(f"\n[Done] Best Val Accuracy: {best_val_acc:.4f}")

    # ── Final test evaluation ──────────────────
    test_results = full_evaluation(model, test_loader, device, dataset.idx_to_class)
    print("\n[Test Set Results]")
    print(test_results["report"])

    return model


# ─────────────────────────────────────────────
#  Detailed evaluation (test set)
# ─────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str],
) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_labels = sorted(set(all_labels.tolist()))
    class_names   = [idx_to_class[i] for i in unique_labels]
    report        = classification_report(all_labels, all_preds, labels=unique_labels, target_names=class_names)
    cm          = confusion_matrix(all_labels, all_preds)
    accuracy    = (all_preds == all_labels).mean()

    return {
        "accuracy":    accuracy,
        "report":      report,
        "confusion_matrix": cm,
        "all_preds":   all_preds,
        "all_labels":  all_labels,
    }


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
