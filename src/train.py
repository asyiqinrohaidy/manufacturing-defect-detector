"""
Training script for the manufacturing surface defect classifier.

Usage:
    python src/train.py --epochs 15 --batch-size 32 --lr 1e-4
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import get_dataloaders
from model import build_model

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Train from random init (use if pretrained weight download is blocked)")
    parser.add_argument("--subset", type=int, default=None,
                         help="Limit dataset to first N samples per loader (for fast smoke-testing on CPU)")
    parser.add_argument("--out-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    if args.subset:
        from torch.utils.data import Subset, DataLoader as DL
        def limit(loader):
            n = min(args.subset, len(loader.dataset))
            idx = list(range(n))
            return DL(Subset(loader.dataset, idx), batch_size=args.batch_size,
                      shuffle=(loader is train_loader))
        train_loader = limit(train_loader)
        val_loader = limit(val_loader)
        test_loader = limit(test_loader)

    print(f"Classes: {classes}")
    print(f"Train/Val/Test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    model = build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone,
                         device=device, pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
            }, out_dir / "best_model.pt")
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    # Final test evaluation using best checkpoint
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    print(f"\nTest results -> loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "classes": classes,
        "history": history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
