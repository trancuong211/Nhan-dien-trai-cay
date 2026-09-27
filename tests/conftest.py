import os
import tempfile
import shutil

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    """Tạo ảnh RGB sample 100x100."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_tensor():
    """Tạo tensor sample có shape (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def fake_data_dir(tmp_path):
    """Tạo thư mục data giả lập với cấu trúc ImageFolder."""
    num_classes = 3
    class_names = ["apple", "banana", "cherry"]

    for split in ["train", "val", "test"]:
        for cls in class_names:
            cls_dir = tmp_path / "data" / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(3):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(cls_dir / f"img_{i}.jpg")

    return str(tmp_path / "data"), class_names, num_classes
