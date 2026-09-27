"""
data_preprocessing.py — Pipeline tiền xử lý và augmentation dữ liệu.
Sử dụng torchvision transforms cho chuẩn hóa và tăng cường dữ liệu.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Chuẩn ImageNet (dùng cho MobileNetV3 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
BATCH_SIZE = 32


def get_train_transforms():
    """Transform cho tập train với data augmentation mạnh."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.7, 1.3),
            shear=10,
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_eval_transforms():
    """Transform cho tập val/test (không augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Tạo DataLoaders cho train, val, test từ cấu trúc thư mục.

    data_dir/
        train/class1/*.jpg, train/class2/*.jpg, ...
        val/class1/*.jpg, ...
        test/class1/*.jpg, ...

    Returns:
        dict với keys 'train', 'val', 'test' chứa DataLoader
        class_names: list tên các lớp
        num_classes: số lượng lớp
    """
    data_path = Path(data_dir)

    # Tạo datasets
    train_dataset = datasets.ImageFolder(
        root=str(data_path / "train"),
        transform=get_train_transforms(),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_path / "val"),
        transform=get_eval_transforms(),
    )
    test_dataset = datasets.ImageFolder(
        root=str(data_path / "test"),
        transform=get_eval_transforms(),
    )

    # Tạo DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"[INFO] Data loaded:")
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes: {class_names}")

    return dataloaders, class_names, num_classes


def denormalize(tensor):
    """Chuyển tensor đã normalize về dạng hiển thị được."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
