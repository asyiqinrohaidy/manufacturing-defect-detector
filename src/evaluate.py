"""
Full test-set evaluation for the manufacturing defect classifier.

Outputs:
    outputs/eval/confusion_matrix.png
    outputs/eval/classification_report.txt
    outputs/eval/predictions.csv

Usage:
    py -3.13 src\\evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from model import build_model
from dataset import get_dataloaders

ROOT = Path(__file__).resolve().parent.parent


def run():
    device = torch.device("cpu")

    checkpoint = torch.load(ROOT / "models" / "best_model.pt", map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes), device=device, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_dir = ROOT / "data" / "processed"
    _, _, test_loader, dataset_classes = get_dataloaders(data_dir, batch_size=32, num_workers=0)

    if dataset_classes != classes:
        print("WARNING: class order mismatch between checkpoint and dataset.")
        print(f"  Checkpoint: {classes}")
        print(f"  Dataset:    {dataset_classes}")

    image_paths = [s[0] for s in test_loader.dataset.samples]

    all_preds, all_labels, all_confidences = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Test macro F1: {f1_macro:.4f}\n")
    print(report)

    output_dir = ROOT / "outputs" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro F1: {f1_macro:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [classes[i] for i in all_labels],
        "pred_label": [classes[i] for i in all_preds],
        "confidence": all_confidences,
        "correct": all_labels == all_preds,
    })
    df.to_csv(output_dir / "predictions.csv", index=False)

    misclassified = df[~df["correct"]]
    print(f"\nMisclassified: {len(misclassified)} / {len(df)}")
    if len(misclassified) > 0:
        print(misclassified[["image_path", "true_label", "pred_label", "confidence"]]
              .to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run()
