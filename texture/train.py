#!/usr/bin/env python3
"""
Train the texture classifier on labeled region crops.

Usage:
    python texture/train.py

Expects:
  - data/region_crops/   (images exported by export_crops.py)
  - data/labels/texture_labels.csv  (crop_filename,texture_class)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import config
from texture.dataset import RegionCropDataset, load_labels
from texture.model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def confusion_matrix_report(model, loader, device):
    n = len(config.TEXTURE_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            for true, pred in zip(labels.numpy(), preds):
                cm[true][pred] += 1

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in config.TEXTURE_CLASSES)
    print(header)
    for i, row_name in enumerate(config.TEXTURE_CLASSES):
        row = f"{row_name:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(n))
        print(row)

    print("\nPer-class metrics:")
    print(f"{'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    for i, cls in enumerate(config.TEXTURE_CLASSES):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec  = tp / max(cm[i, :].sum(), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"{cls:<12} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")


def main():
    device = torch.device("cpu")

    if not os.path.exists(config.LABELS_CSV):
        sys.exit(f"Labels file not found: {config.LABELS_CSV}\n"
                 "Run export_crops.py first, then manually label crops.")

    full_dataset = RegionCropDataset(
        config.REGION_CROPS_DIR, config.LABELS_CSV, augment=True
    )
    if len(full_dataset) == 0:
        sys.exit("No labeled samples found.")

    indices = list(range(len(full_dataset)))
    labels_all = [full_dataset.class_to_idx[lbl]
                  for _, lbl in load_labels(config.LABELS_CSV)]

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=labels_all if len(set(labels_all)) > 1 else None,
    )

    train_set = Subset(full_dataset, train_idx)
    val_dataset = RegionCropDataset(
        config.REGION_CROPS_DIR, config.LABELS_CSV, augment=False
    )
    val_set = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=16)

    model = build_model().to(device)
    weights = full_dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("texture/checkpoints", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, 101):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        print(f"Epoch {epoch:3d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), config.TEXTURE_CHECKPOINT)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    model.load_state_dict(torch.load(config.TEXTURE_CHECKPOINT))
    confusion_matrix_report(model, val_loader, device)


if __name__ == "__main__":
    main()
