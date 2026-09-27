"""
model.py — Xây dựng mô hình nhận diện trái cây.
Sử dụng Transfer Learning với EfficientNet-B1 (pretrained trên ImageNet).
"""
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, pretrained: bool = True, freeze_base: bool = True):
    """
    Tạo model EfficientNet-B1 cho bài toán phân loại trái cây.

    Args:
        num_classes: Số lớp trái cây cần phân loại
        pretrained: Sử dụng weights pretrained trên ImageNet
        freeze_base: Đóng băng base model ban đầu

    Returns:
        model: PyTorch model
    """
    # Tải EfficientNet-B1 pretrained
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b1(weights=weights)

    # Đóng băng base model (feature extractor)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        print("[INFO] Frozen feature extractor (base model)")

    # Thay thế lớp classifier cuối cùng
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )

    # Đếm số params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: EfficientNet-B1")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Output classes: {num_classes}")

    return model


def unfreeze_base(model, num_layers_to_unfreeze: int = 8):
    """
    Mở khóa một số layer cuối của base model để fine-tune.

    Args:
        model: Model đã tạo bởi create_model()
        num_layers_to_unfreeze: Số block cuối cùng cần mở khóa
    """
    # EfficientNet features là Sequential chứa các block
    features = list(model.features.children())
    total_blocks = len(features)

    # Mở khóa các block cuối
    for i, block in enumerate(features):
        if i >= total_blocks - num_layers_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Unfroze {num_layers_to_unfreeze} last blocks for fine-tune")
    print(f"  Trainable params: {trainable_params:,}")

    return model
