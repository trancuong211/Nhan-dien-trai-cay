"""
evaluate.py — Đánh giá mô hình trên tập test.
Tạo classification report, confusion matrix, và biểu đồ.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import create_dataloaders
from src.model import create_model


def evaluate_model(
    data_dir: str = "data",
    model_dir: str = "models",
    model_path: str = None,
):
    """
    Đánh giá model trên tập test.
    In classification report và vẽ confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load class names
    class_names_path = Path(model_dir) / "class_names.json"
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Load data
    dataloaders, _, _ = create_dataloaders(data_dir)

    # Load model
    model = create_model(num_classes=num_classes, pretrained=False, freeze_base=False)
    if model_path is None:
        model_path = str(Path(model_dir) / "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Dự đoán trên tập test
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Tính accuracy tổng
    accuracy = (all_preds == all_labels).mean()
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS ON TEST SET")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Test samples: {len(all_labels)}")
    print(f"Classes: {num_classes}")

    # Classification report
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION REPORT")
    print(f"{'=' * 60}")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # Lưu report
    report_path = Path(model_dir) / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Saved report to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.8), max(10, num_classes * 0.7)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2%})", fontsize=14)
    plt.tight_layout()

    cm_path = Path(model_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")

    # Find most confused class pairs
    print(f"\n{'=' * 60}")
    print("MOST CONFUSED CLASS PAIRS")
    print(f"{'=' * 60}")
    np.fill_diagonal(cm, 0)
    top_confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                top_confused.append((class_names[i], class_names[j], cm[i, j]))
    top_confused.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count in top_confused[:10]:
        print(f"  {true_cls} -> {pred_cls}: {count} times")

    return accuracy


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    evaluate_model(
        data_dir=str(project_root / "data"),
        model_dir=str(project_root / "models"),
    )
