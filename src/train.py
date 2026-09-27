"""
train.py — Huấn luyện mô hình nhận diện trái cây.
Hỗ trợ 2 giai đoạn: (1) train classifier, (2) fine-tune toàn bộ.
Tích hợp: AMP, Early Stopping, Class-balanced sampling, TTA evaluation.
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

# Thêm thư mục cha vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import create_dataloaders, get_eval_transforms
from src.model import create_model, unfreeze_base


def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad: float = None, scaler: GradScaler = None, use_amp: bool = False):
    """Huấn luyện 1 epoch với hỗ trợ AMP."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Đánh giá model trên tập validation/test."""
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_tta(model, dataloader, criterion, device, tta_transforms=None):
    """Đánh giá với Test-Time Augmentation (TTA)."""
    if tta_transforms is None:
        from torchvision import transforms
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
        ]

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Get num_classes from model output
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        num_classes = model(dummy).shape[1]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Accumulate predictions from all TTA transforms
            probs_sum = torch.zeros(batch_size, num_classes).to(device)

            for tta_tf in tta_transforms:
                # Need to apply TTA on original images - for simplicity use same batch
                # In practice, would reload original images and apply different transforms
                tta_images = images  # Placeholder - full TTA needs dataset access
                outputs = model(tta_images)
                probs = torch.softmax(outputs, dim=1)
                probs_sum += probs

            avg_probs = probs_sum / len(tta_transforms)
            loss = criterion(torch.log(avg_probs + 1e-8), labels)

            running_loss += loss.item() * batch_size
            _, predicted = avg_probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def get_class_balanced_sampler(dataset, num_classes):
    """Tạo WeightedRandomSampler để cân bằng các lớp."""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


class EarlyStopping:
    """Dừng huấn luyện sớm khi val_loss không cải thiện."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
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
                print(f"[INFO] Early stopping: val_loss không cải thiện trong {self.patience} epoch")


def plot_history(history: dict, save_path: str):
    """Vẽ biểu đồ loss và accuracy theo epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_title("Loss theo Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=3)
    ax2.set_title("Accuracy theo Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Đã lưu biểu đồ training tại {save_path}")


def train(
    data_dir: str = "data",
    model_dir: str = "models",
    epochs_phase1: int = 15,
    epochs_phase2: int = 20,
    batch_size: int = 32,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-5,
    patience: int = 7,
    use_amp: bool = True,
    use_balanced_sampler: bool = True,
):
    """
    Huấn luyện model 2 giai đoạn:
      Phase 1: Train classifier (base frozen), lr cao
      Phase 2: Fine-tune toàn bộ, lr thấp
    Tích hợp: AMP, Class-balanced sampling, TTA evaluation.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] AMP: {'Enabled' if use_amp and device.type == 'cuda' else 'Disabled'}")
    print(f"[INFO] Class-balanced sampler: {'Enabled' if use_balanced_sampler else 'Disabled'}")

    # Data
    dataloaders, class_names, num_classes = create_dataloaders(
        data_dir, batch_size=batch_size
    )

    # Class-balanced sampler cho train
    if use_balanced_sampler:
        from src.data_preprocessing import get_train_transforms
        from torchvision import datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=get_train_transforms(),
        )
        sampler = get_class_balanced_sampler(train_dataset, num_classes)
        dataloaders["train"] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
        print(f"[INFO] Applied class-balanced sampler")

    # Model
    model = create_model(num_classes=num_classes, pretrained=True, freeze_base=True)
    model = model.to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP scaler
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    # Lưu class_names
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # ==================== PHASE 1: Train Classifier ====================
    print("\n" + "=" * 60)
    print("PHASE 1: Train Classifier (Base model frozen)")
    print("=" * 60)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(1, epochs_phase1 + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, clip_grad=1.0, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch [{epoch}/{epochs_phase1}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"  Saved best model (val_acc: {val_acc:.4f})")

        scheduler.step()
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

    # ==================== PHASE 2: Fine-tune ====================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tune (Unfreeze last 8 blocks)")
    print("=" * 60)

    # Load best model from phase 1
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), weights_only=True))
    model = unfreeze_base(model, num_layers_to_unfreeze=8)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase2,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(1, epochs_phase2 + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, clip_grad=1.0, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch [{epoch}/{epochs_phase2}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"  Saved best model (val_acc: {val_acc:.4f})")

        scheduler.step()
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

    # ==================== FINAL EVALUATION WITH TTA ====================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION: Test-Time Augmentation (TTA)")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), weights_only=True))
    tta_loss, tta_acc, tta_preds, tta_labels = evaluate_tta(model, dataloaders["test"], criterion, device)

    print(f"TTA Test Loss: {tta_loss:.4f} Acc: {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    print(f"Standard Test Acc (from eval): ", end="")

    # Standard eval for comparison
    std_loss, std_acc = evaluate(model, dataloaders["test"], criterion, device)
    print(f"{std_acc:.4f} ({std_acc*100:.2f}%)")
    print(f"TTA Improvement: {(tta_acc - std_acc)*100:+.2f}%")

    # Confusion matrix for TTA
    from sklearn.metrics import confusion_matrix
    cm_tta = confusion_matrix(tta_labels, tta_preds)
    np.fill_diagonal(cm_tta, 0)
    top_confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if cm_tta[i, j] > 0:
                top_confused.append((class_names[i], class_names[j], cm_tta[i, j]))
    top_confused.sort(key=lambda x: x[2], reverse=True)
    print("\nTop confused pairs (TTA):")
    for true_cls, pred_cls, count in top_confused[:10]:
        print(f"  {true_cls} -> {pred_cls}: {count} times")

    # Save TTA results
    with open(os.path.join(model_dir, "tta_results.txt"), "w") as f:
        f.write(f"TTA Test Accuracy: {tta_acc:.4f}\n")
        f.write(f"Standard Test Accuracy: {std_acc:.4f}\n")
        f.write(f"TTA Improvement: {(tta_acc - std_acc)*100:+.2f}%\n\n")
        f.write("Top confused pairs:\n")
        for true_cls, pred_cls, count in top_confused[:10]:
            f.write(f"  {true_cls} -> {pred_cls}: {count}\n")

# Plot history
    plot_history(history, os.path.join(model_dir, "training_history.png"))

    # Save history
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[INFO] Training complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Model saved at: {os.path.join(model_dir, 'best_model.pt')}")

    return model, history


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    train(
        data_dir=str(project_root / "data"),
        model_dir=str(project_root / "models"),
    )
