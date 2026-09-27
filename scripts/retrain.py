#!/usr/bin/env python3
"""
Quick retrain with better config - Phase 1: 20 epochs, Phase 2: 15 epochs
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
from src.model import create_model, unfreeze_base

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  Early stopping triggered")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloaders, class_names, num_classes = create_dataloaders("data", batch_size=32)

    model = create_model(num_classes=num_classes, pretrained=True, freeze_base=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    best_val_acc = 0.0

    # PHASE 1
    print("\n" + "="*60)
    print("PHASE 1: Train Classifier (20 epochs, lr=3e-3)")
    print("="*60)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7)

    for epoch in range(1, 21):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
        elapsed = time.time() - start

        print(f"  Epoch {epoch:2d}/20 | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"    >> Saved best (val_acc={val_acc:.4f})")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print("  Early stopping!")
            break

    # PHASE 2
    print("\n" + "="*60)
    print("PHASE 2: Fine-tune last 6 blocks (15 epochs, lr=5e-6)")
    print("="*60)

    model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
    model = unfreeze_base(model, num_layers_to_unfreeze=6)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=7)

    for epoch in range(1, 16):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)
        elapsed = time.time() - start

        print(f"  Epoch {epoch:2d}/15 | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"    >> Saved best (val_acc={val_acc:.4f})")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print("  Early stopping!")
            break

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, dataloaders["test"], criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print("Done!")

if __name__ == "__main__":
    main()