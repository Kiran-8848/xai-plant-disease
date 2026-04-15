import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from backend.ml.models.resnet_model import load_checkpoint, get_device
from backend.ml.utils.data_utils import build_dataloaders

def run_evaluation():
    # ── Load model ────────────────────────────────────────
    device = get_device()

    model, meta = load_checkpoint(
        'backend/ml/checkpoints/best_model.pth',
        device
    )

    with open('backend/ml/checkpoints/class_mapping.json') as f:
        mapping      = json.load(f)
        idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

    print(f"Model loaded. Classes: {len(idx_to_class)}")

    # ── Load test data ─────────────────────────────────────
    _, _, test_loader, _ = build_dataloaders(
        data_root   = './data/PlantVillage',
        batch_size  = 32,
        num_workers = 0,       # ← fix for Windows
    )

    # ── Run predictions ────────────────────────────────────
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            print(f"  Batch {i+1}/{len(test_loader)}", end="\r")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Print results ──────────────────────────────────────
    unique_labels = sorted(set(all_labels.tolist()))
    class_names   = [idx_to_class[i] for i in unique_labels]
    accuracy      = (all_preds == all_labels).mean()

    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"{'='*60}\n")
    print(classification_report(
        all_labels, all_preds,
        labels      = unique_labels,
        target_names= class_names
    ))

    # ── Confusion matrix ───────────────────────────────────
    Path('outputs/evaluation').mkdir(parents=True, exist_ok=True)

    cm  = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = 'd',
        cmap       = 'Greens',
        xticklabels= class_names,
        yticklabels= class_names,
        ax         = ax,
        linewidths = 0.5
    )
    ax.set_title(f'Confusion Matrix — Test Accuracy: {accuracy*100:.2f}%', fontsize=13, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/evaluation/confusion_matrix.png', dpi=150)
    plt.show()
    print("\nConfusion matrix saved → outputs/evaluation/confusion_matrix.png")


if __name__ == '__main__':       # ← this is the key fix for Windows
    run_evaluation()