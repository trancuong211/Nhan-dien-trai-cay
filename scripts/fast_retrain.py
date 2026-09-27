#!/usr/bin/env python3
"""
Fast retrain - MobileNetV3 Small, fewer epochs for CPU
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import create_dataloaders

def create_small_model(num_classes, pretrained=True, freeze_base=True):
    from torchvision import models
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b1(weights=weights)
    if freeze_base:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_features, num_classes)
    )
    return model

def unfreeze_small(model, n=8):
    feats = list(model.features.children())
    for i, block in enumerate(feats):
        if i >= len(feats) - n:
            for p in block.parameters():
                p.requires_grad = True
    return model

def train_one_epoch(model, dl, criterion, optimizer, device, clip_grad=None):
    model.train()
    running_loss = correct = total = 0
    for images, labels in dl:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, pred = model(images).max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, dl, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return running_loss / total, correct / total

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloaders, class_names, num_classes = create_dataloaders("data", batch_size=32)

    # Use EfficientNet-B1 - stronger model
    model = create_small_model(num_classes=num_classes, pretrained=True, freeze_base=True).to(device)
    print(f"Model: EfficientNet-B1 | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    best_val_acc = 0.0

    # PHASE 1: 15 epochs
    print("\n=== PHASE 1: Classifier (15 epochs, lr=1e-3) ===")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    es = EarlyStopping(patience=7)

    for epoch in range(1, 16):
        start = time.time()
        tl, ta = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, clip_grad=1.0)
        vl, va = evaluate(model, dataloaders["val"], criterion, device)
        print(f"  Epoch {epoch}/15 | Train: {tl:.4f}/{ta:.4f} | Val: {vl:.4f}/{va:.4f} | {time.time()-start:.1f}s")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"    >> Saved best: {va:.4f}")
        scheduler.step()
        es(vl)
        if es.should_stop: break

    # PHASE 2: 20 epochs fine-tune
    print("\n=== PHASE 2: Fine-tune (20 epochs, lr=1e-5) ===")
    model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
    model = unfreeze_small(model, 8)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    es = EarlyStopping(patience=7)

    for epoch in range(1, 21):
        start = time.time()
        tl, ta = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, clip_grad=1.0)
        vl, va = evaluate(model, dataloaders["val"], criterion, device)
        print(f"  Epoch {epoch}/20 | Train: {tl:.4f}/{ta:.4f} | Val: {vl:.4f}/{va:.4f} | {time.time()-start:.1f}s")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"    >> Saved best: {va:.4f}")
        scheduler.step()
        es(vl)
        if es.should_stop: break

    # Test
    model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, dataloaders["test"], criterion, device)
    print(f"\n=== TEST: Loss={test_loss:.4f} Acc={test_acc:.4f} ===")
    print(f"Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()